"""Benchmark an OpenAI-compatible inference server.

Measures what matters for agent-style workloads against any engine that speaks the OpenAI chat
completions API (vLLM, SGLang, llama.cpp server, ...):

- decode throughput for one stream and for N concurrent streams,
- prefill throughput and time to first token (TTFT) for long synthetic prompts,
- tool calling (several tools, parallel calls requested),
- JSON-schema structured output (``response_format`` with ``strict: true``).

All prompts are synthetic and generated from a fixed seed, so runs are reproducible.

Subcommands::

    serve_bench.py wait  --base-url URL [--started-at EPOCH]   # wait for readiness, print load time
    serve_bench.py evict PATH [PATH ...]                         # drop files from the page cache
    serve_bench.py run   --base-url URL --model NAME --label L   # run the benchmark suite

See ``docs/benchmarking.md`` for the full workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_SEED = 1234
ALL_TESTS = ("decode", "concurrency", "prefill", "tools", "json")

# --------------------------------------------------------------------------------------------
# Synthetic prompts
# --------------------------------------------------------------------------------------------

_SERVICES = ["api-gw", "auth", "orders", "payments", "inventory", "search", "notify", "billing"]
_LEVELS = ["INFO", "INFO", "INFO", "DEBUG", "WARN", "ERROR"]
_OPS = ["GET /v1/items", "POST /v1/orders", "PUT /v1/cart", "rpc.Reserve", "rpc.Charge", "db.query"]
_WORDS = (
    "request completed retry scheduled connection pool exhausted cache miss upstream timeout "
    "lease renewed queue depth rising checkpoint written replica lagging handshake failed "
    "token refreshed shard rebalanced backoff applied circuit opened circuit closed"
).split()


def synthetic_log_lines(n: int, seed: int) -> list[str]:
    """Return ``n`` deterministic, synthetic log-like lines (no real data)."""
    rng = random.Random(seed)
    lines = []
    t = 1_700_000_000.0
    for _ in range(n):
        t += rng.random() * 0.5
        svc = rng.choice(_SERVICES)
        msg = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(3, 8)))
        lines.append(
            f"{t:.3f} {rng.choice(_LEVELS):5s} svc={svc} trace={rng.getrandbits(64):016x} "
            f"op={rng.choice(_OPS)!r} latency_ms={rng.randint(1, 4000)} msg={msg!r}"
        )
    return lines


def build_prefill_prompt(n_lines: int, seed: int) -> str:
    """A long prompt: a unique header (defeats prefix caching across seeds) plus log lines."""
    header = f"Run {seed}: summarize the anomalies in the following synthetic log excerpt.\n"
    return header + "\n".join(synthetic_log_lines(n_lines, seed)) + "\nAnswer in one sentence."


def lines_for_tokens(target_tokens: int, tokens_per_line: float, overhead_tokens: int = 64) -> int:
    """Number of log lines needed to reach roughly ``target_tokens`` prompt tokens."""
    if tokens_per_line <= 0:
        raise ValueError("tokens_per_line must be positive")
    return max(1, int((target_tokens - overhead_tokens) / tokens_per_line))


# --------------------------------------------------------------------------------------------
# Tool-calling and structured-output fixtures
# --------------------------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": "Search log lines for a service in a time window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "query": {"type": "string"},
                    "window_minutes": {"type": "integer"},
                },
                "required": ["service", "query", "window_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trace",
            "description": "Fetch all spans of one distributed trace.",
            "parameters": {
                "type": "object",
                "properties": {"trace_id": {"type": "string"}},
                "required": ["trace_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric",
            "description": "Fetch a metric time series for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "metric": {"type": "string", "enum": ["p99_latency_ms", "error_rate", "cpu"]},
                    "window_minutes": {"type": "integer"},
                },
                "required": ["service", "metric", "window_minutes"],
            },
        },
    },
]

TOOL_PROMPT = (
    "Checkout requests started failing 20 minutes ago. Gather evidence now, calling all three "
    "tools in parallel in this single turn: search the 'payments' service logs for 'timeout' over "
    "the last 30 minutes, fetch trace 'a1b2c3d4e5f60718', and get the 'p99_latency_ms' metric for "
    "'payments' over the last 30 minutes. Do not answer in text; only call the tools."
)

_PY_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^ev-[0-9]+$")
    summary: str


class Hypothesis(BaseModel):
    """Pydantic mirror of ``HYPOTHESIS_SCHEMA``, used to validate model output."""

    model_config = ConfigDict(extra="forbid", strict=True)
    component: str
    category: Literal["network", "storage", "configuration", "resource_exhaustion", "dependency"]
    confidence: float = Field(ge=0, le=1)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=5)


HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "component": {"type": "string"},
        "category": {
            "type": "string",
            "enum": ["network", "storage", "configuration", "resource_exhaustion", "dependency"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": "^ev-[0-9]+$"},
                    "summary": {"type": "string"},
                },
                "required": ["id", "summary"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["component", "category", "confidence", "evidence"],
    "additionalProperties": False,
}


def json_prompt(seed: int) -> str:
    lines = "\n".join(
        f"ev-{i}: {line}" for i, line in enumerate(synthetic_log_lines(12, seed), start=1)
    )
    return (
        "You are given numbered evidence lines from a synthetic incident. Return your single most "
        "likely root-cause hypothesis as JSON matching the provided schema. Cite evidence by id.\n"
        + lines
    )


# --------------------------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------------------------


def check_tool_args(name: str, raw_args: str) -> list[str]:
    """Return problems with one tool call's arguments (empty list means valid)."""
    spec = next((t["function"] for t in TOOLS if t["function"]["name"] == name), None)
    if spec is None:
        return [f"unknown tool {name!r}"]
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as exc:
        return [f"{name}: arguments are not JSON ({exc.msg})"]
    if not isinstance(args, dict):
        return [f"{name}: arguments are not an object"]
    problems = []
    props = spec["parameters"]["properties"]
    for key in spec["parameters"]["required"]:
        if key not in args:
            problems.append(f"{name}: missing {key!r}")
    for key, value in args.items():
        if key not in props:
            problems.append(f"{name}: unexpected {key!r}")
            continue
        expected = _PY_TYPES[props[key]["type"]]
        if isinstance(value, bool) and props[key]["type"] != "boolean":
            problems.append(f"{name}: {key!r} has type bool")
        elif not isinstance(value, expected):
            problems.append(f"{name}: {key!r} has type {type(value).__name__}")
        elif "enum" in props[key] and value not in props[key]["enum"]:
            problems.append(f"{name}: {key!r}={value!r} not in enum")
    return problems


def grade_tool_calls(calls: list[tuple[str, str]]) -> dict[str, Any]:
    """Grade (name, arguments) pairs: pass needs all three tools, once each, with valid args."""
    names = [name for name, _ in calls]
    problems = [p for name, args in calls for p in check_tool_args(name, args)]
    expected = {t["function"]["name"] for t in TOOLS}
    missing = sorted(expected - set(names))
    if missing:
        problems.append(f"missing calls: {missing}")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        problems.append(f"duplicate calls: {duplicates}")
    return {"passed": not problems, "n_calls": len(calls), "problems": problems}


def grade_json_output(content: str | None) -> dict[str, Any]:
    if not content:
        return {"passed": False, "problems": ["empty content"]}
    try:
        Hypothesis.model_validate_json(content)
    except ValidationError as exc:
        return {"passed": False, "problems": [e["msg"] for e in exc.errors()][:5]}
    return {"passed": True, "problems": []}


# --------------------------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------------------------


@dataclass
class StreamResult:
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_s: float

    @property
    def decode_tps(self) -> float:
        """Tokens per second after the first token (excludes prefill)."""
        decode_time = self.total_s - self.ttft_s
        if self.completion_tokens <= 1 or decode_time <= 0:
            return 0.0
        return (self.completion_tokens - 1) / decode_time

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / self.ttft_s if self.ttft_s > 0 else 0.0


@dataclass
class BenchConfig:
    base_url: str
    model: str
    label: str
    seed: int = DEFAULT_SEED
    thinking: bool = False
    # tools/json requests budget this many tokens when thinking is on, since the model spends
    # part of it on the <think> block before any tool call or JSON content. 4096 was not always
    # enough (observed: finish_reason=length with truncated JSON); see docs/benchmarking.md.
    thinking_max_tokens: int = 8192
    decode_tokens: int = 256
    repeats: int = 3
    concurrency: int = 2
    prefill_lengths: list[int] = field(default_factory=lambda: [8192, 32768])
    json_trials: int = 3
    timeout_s: float = 1800.0


def _extra_body(cfg: BenchConfig, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": cfg.thinking}}
    body.update(extra)
    return body


def stream_once(
    client: OpenAI, cfg: BenchConfig, prompt: str, max_tokens: int, ignore_eos: bool
) -> StreamResult:
    """Send one streaming request and time the first token and the end of the stream."""
    start = time.perf_counter()
    first: float | None = None
    usage = None
    stream = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        seed=cfg.seed,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=_extra_body(cfg, ignore_eos=ignore_eos),
    )
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if first is None and chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content or getattr(delta, "reasoning_content", None):
                first = time.perf_counter()
    end = time.perf_counter()
    if usage is None:
        raise RuntimeError("server did not return usage; stream_options.include_usage unsupported")
    return StreamResult(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        ttft_s=(first or end) - start,
        total_s=end - start,
    )


def measure_decode(client: OpenAI, cfg: BenchConfig) -> dict[str, Any]:
    prompt = "Write a long, detailed technical essay about log analysis in distributed systems."
    stream_once(client, cfg, prompt, 32, ignore_eos=True)  # warm-up
    runs = [
        stream_once(client, cfg, prompt, cfg.decode_tokens, ignore_eos=True)
        for _ in range(cfg.repeats)
    ]
    return {
        "decode_tps_median": round(statistics.median(r.decode_tps for r in runs), 2),
        "ttft_s_median": round(statistics.median(r.ttft_s for r in runs), 3),
        "completion_tokens": [r.completion_tokens for r in runs],
    }


def measure_concurrency(client: OpenAI, cfg: BenchConfig) -> dict[str, Any]:
    prompts = [
        f"Stream {i}: write a long technical essay about capacity planning."
        for i in range(cfg.concurrency)
    ]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        runs = list(
            pool.map(
                lambda p: stream_once(client, cfg, p, cfg.decode_tokens, ignore_eos=True), prompts
            )
        )
    wall = time.perf_counter() - start
    return {
        "concurrency": cfg.concurrency,
        "per_stream_decode_tps_mean": round(statistics.mean(r.decode_tps for r in runs), 2),
        "aggregate_tps": round(sum(r.completion_tokens for r in runs) / wall, 2),
    }


def calibrate_tokens_per_line(client: OpenAI, cfg: BenchConfig, n_lines: int = 200) -> float:
    base = stream_once(client, cfg, build_prefill_prompt(1, cfg.seed), 1, ignore_eos=False)
    full = stream_once(client, cfg, build_prefill_prompt(n_lines, cfg.seed), 1, ignore_eos=False)
    return (full.prompt_tokens - base.prompt_tokens) / (n_lines - 1)


def measure_prefill(client: OpenAI, cfg: BenchConfig) -> list[dict[str, Any]]:
    per_line = calibrate_tokens_per_line(client, cfg)
    rows = []
    for i, target in enumerate(cfg.prefill_lengths):
        n_lines = lines_for_tokens(target, per_line)
        # A different seed per request keeps the prefix cache from serving the prompt.
        prompt = build_prefill_prompt(n_lines, cfg.seed + 1000 + i)
        try:
            r = stream_once(client, cfg, prompt, 16, ignore_eos=True)
        except Exception as exc:  # noqa: BLE001 - record any server-side failure
            rows.append({"target_tokens": target, "error": str(exc)[:300]})
            continue
        rows.append(
            {
                "target_tokens": target,
                "prompt_tokens": r.prompt_tokens,
                "ttft_s": round(r.ttft_s, 2),
                "prefill_tps": round(r.prefill_tps, 1),
            }
        )
    return rows


def measure_tools(client: OpenAI, cfg: BenchConfig) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=cast(
                list[ChatCompletionMessageParam], [{"role": "user", "content": TOOL_PROMPT}]
            ),
            tools=cast(list[ChatCompletionToolParam], TOOLS),
            tool_choice="auto",
            parallel_tool_calls=True,
            temperature=0,
            seed=cfg.seed,
            max_tokens=cfg.thinking_max_tokens if cfg.thinking else 1024,
            extra_body=_extra_body(cfg),
        )
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "n_calls": 0, "problems": [f"request failed: {exc}"[:300]]}
    message = resp.choices[0].message
    calls = [
        (c.function.name, c.function.arguments)
        for c in (message.tool_calls or [])
        if isinstance(c, ChatCompletionMessageFunctionToolCall)
    ]
    result = grade_tool_calls(calls)
    if not calls and message.content:
        result["problems"].append(f"text instead of calls: {message.content[:160]!r}")
    if not result["passed"]:
        result["problems"].append(f"finish_reason={resp.choices[0].finish_reason}")
    result["latency_s"] = round(time.perf_counter() - start, 2)
    return result


def measure_json(client: OpenAI, cfg: BenchConfig) -> dict[str, Any]:
    trials = []
    for i in range(cfg.json_trials):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": json_prompt(cfg.seed + i)}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "hypothesis",
                        "strict": True,
                        "schema": HYPOTHESIS_SCHEMA,
                    },
                },
                temperature=0,
                seed=cfg.seed,
                max_tokens=cfg.thinking_max_tokens if cfg.thinking else 1024,
                extra_body=_extra_body(cfg),
            )
            graded = grade_json_output(resp.choices[0].message.content)
            if not graded["passed"]:
                graded["problems"].append(f"finish_reason={resp.choices[0].finish_reason}")
            trials.append(graded)
        except Exception as exc:  # noqa: BLE001
            trials.append({"passed": False, "problems": [f"request failed: {exc}"[:300]]})
    passed = sum(t["passed"] for t in trials)
    return {
        "passed": passed == len(trials),
        "pass_rate": f"{passed}/{len(trials)}",
        "problems": [p for t in trials for p in t["problems"]][:5],
    }


def read_psi_memory(path: Path = Path("/proc/pressure/memory")) -> dict[str, float] | None:
    """Memory pressure (``path``): share of time stalled on memory, last 10s/60s/300s.

    Linux-only and absent in some containers/kernels; callers must tolerate ``None``.
    """
    try:
        text = path.read_text()
    except OSError:
        return None
    psi: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]
        for field_str in parts[1:]:
            key, _, value = field_str.partition("=")
            if key in ("avg10", "avg60", "avg300"):
                psi[f"{kind}_{key}"] = float(value)
    return psi or None


def memory_snapshot(container: str | None) -> dict[str, Any]:
    """Host memory (``free -m``), memory pressure (PSI), and optionally a container's usage."""
    snap: dict[str, Any] = {}
    try:
        mem = subprocess.run(["free", "-m"], capture_output=True, text=True, check=True).stdout
        fields = mem.splitlines()[1].split()
        snap["host_used_mib"] = int(fields[2])
        snap["host_available_mib"] = int(fields[6])
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        pass
    psi = read_psi_memory()
    if psi:
        snap["psi_memory"] = psi
    if container:
        try:
            out = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            snap["container_mem"] = out
        except (OSError, subprocess.CalledProcessError):
            pass
    return snap


def server_max_model_len(base_url: str, model: str) -> int | None:
    """The served context length, if the server reports it in ``/models``."""
    try:
        data = httpx.get(f"{base_url}/models", timeout=10).json().get("data", [])
    except (httpx.HTTPError, ValueError):
        return None
    for entry in data:
        if entry.get("id") == model:
            value = entry.get("max_model_len") or entry.get("meta", {}).get("n_ctx")
            return int(value) if value else None
    return None


def run_suite(cfg: BenchConfig, tests: list[str], container: str | None) -> dict[str, Any]:
    client = OpenAI(base_url=cfg.base_url, api_key=os.getenv("BENCH_API_KEY", "not-needed"))
    client = client.with_options(timeout=cfg.timeout_s, max_retries=0)
    result: dict[str, Any] = {
        "label": cfg.label,
        "config": asdict(cfg),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "max_model_len": server_max_model_len(cfg.base_url, cfg.model),
        "memory_before": memory_snapshot(container),
    }
    steps = {
        "decode": measure_decode,
        "concurrency": measure_concurrency,
        "prefill": measure_prefill,
        "tools": measure_tools,
        "json": measure_json,
    }
    for name in tests:
        print(f"[{cfg.label}] {name} ...", file=sys.stderr, flush=True)
        try:
            result[name] = steps[name](client, cfg)
        except Exception as exc:  # noqa: BLE001 - keep going and record the failure
            result[name] = {"error": str(exc)[:500]}
        print(f"[{cfg.label}] {name}: {json.dumps(result[name])}", file=sys.stderr, flush=True)
    result["memory_after"] = memory_snapshot(container)
    return result


CSV_FIELDS = [
    "label",
    "started_at",
    "max_model_len",
    "decode_tps",
    "concurrent_per_stream_tps",
    "concurrent_aggregate_tps",
    "prefill",
    "tools",
    "json",
    "host_used_mib",
    "host_available_mib",
    "psi_some_avg10",
    "container_mem",
]


def csv_row(result: dict[str, Any]) -> dict[str, Any]:
    def get(section: str, key: str) -> Any:
        value = result.get(section)
        return value.get(key, "") if isinstance(value, dict) else ""

    prefill = result.get("prefill")
    prefill_text = (
        "; ".join(
            f"{p.get('prompt_tokens', p['target_tokens'])}tok: "
            + (f"{p['ttft_s']}s ({p['prefill_tps']} tok/s)" if "ttft_s" in p else "error")
            for p in prefill
        )
        if isinstance(prefill, list)
        else ""
    )
    return {
        "label": result["label"],
        "started_at": result["started_at"],
        "max_model_len": result.get("max_model_len") or "",
        "decode_tps": get("decode", "decode_tps_median"),
        "concurrent_per_stream_tps": get("concurrency", "per_stream_decode_tps_mean"),
        "concurrent_aggregate_tps": get("concurrency", "aggregate_tps"),
        "prefill": prefill_text,
        "tools": {True: "pass", False: "fail"}.get(get("tools", "passed"), ""),
        "json": get("json", "pass_rate"),
        "host_used_mib": result["memory_before"].get("host_used_mib", ""),
        "host_available_mib": result["memory_before"].get("host_available_mib", ""),
        "psi_some_avg10": result["memory_before"].get("psi_memory", {}).get("some_avg10", ""),
        "container_mem": result["memory_before"].get("container_mem", ""),
    }


def write_results(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['label']}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    csv_path = out_dir / "results.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(csv_row(result))
    return path


# --------------------------------------------------------------------------------------------
# Readiness and page-cache helpers
# --------------------------------------------------------------------------------------------


def wait_ready(
    base_url: str, started_at: float, timeout_s: float, interval_s: float = 2.0
) -> float:
    """Poll ``/models`` until it answers 200; return seconds since ``started_at``."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/models", timeout=5).status_code == 200:
                return time.time() - started_at
        except httpx.HTTPError:
            pass
        time.sleep(interval_s)
    raise TimeoutError(f"{base_url} not ready after {timeout_s:.0f}s")


def evict_page_cache(paths: list[Path]) -> int:
    """Ask the kernel to drop cached pages of these files (no root needed). Returns bytes."""
    total = 0
    seen: set[Path] = set()
    for root in paths:
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        # Resolve symlinks (Hugging Face snapshots link to blobs) so each file counts once.
        for f in {p.resolve() for p in files} - seen:
            seen.add(f)
            try:
                fd = os.open(f, os.O_RDONLY)
            except PermissionError:
                continue  # e.g. metadata written by a container running as root
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                total += os.fstat(fd).st_size
            finally:
                os.close(fd)
    return total


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    wait = sub.add_parser("wait", help="wait until the server is ready and print the load time")
    wait.add_argument("--base-url", required=True)
    wait.add_argument("--started-at", type=float, default=None, help="epoch seconds")
    wait.add_argument("--timeout", type=float, default=3600)

    evict = sub.add_parser("evict", help="drop files from the page cache before a cold load")
    evict.add_argument("paths", nargs="+", type=Path)

    run = sub.add_parser("run", help="run the benchmark suite")
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True, help="served model name")
    run.add_argument("--label", required=True, help="result name, e.g. vllm-fp8")
    run.add_argument("--tests", default=",".join(ALL_TESTS))
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--thinking", action="store_true", help="enable the model's thinking mode")
    run.add_argument(
        "--thinking-max-tokens",
        type=int,
        default=BenchConfig.thinking_max_tokens,
        help="max_tokens for tools/json requests when --thinking is set",
    )
    run.add_argument("--decode-tokens", type=int, default=256)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument("--prefill-lengths", type=_parse_ints, default=[8192, 32768])
    run.add_argument("--json-trials", type=int, default=3)
    run.add_argument("--container", default=None, help="container name for docker stats")
    run.add_argument("--out", type=Path, default=Path("bench/results"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "wait":
        started = args.started_at if args.started_at is not None else time.time()
        print(f"ready after {wait_ready(args.base_url, started, args.timeout):.1f}s")
        return 0
    if args.command == "evict":
        print(f"evicted {evict_page_cache(args.paths) / 1e9:.1f} GB from the page cache")
        return 0
    tests = [t for t in args.tests.split(",") if t]
    unknown = sorted(set(tests) - set(ALL_TESTS))
    if unknown:
        raise SystemExit(f"unknown tests: {unknown}")
    cfg = BenchConfig(
        base_url=args.base_url,
        model=args.model,
        label=args.label,
        seed=args.seed,
        thinking=args.thinking,
        thinking_max_tokens=args.thinking_max_tokens,
        decode_tokens=args.decode_tokens,
        repeats=args.repeats,
        concurrency=args.concurrency,
        prefill_lengths=args.prefill_lengths,
        json_trials=args.json_trials,
    )
    result = run_suite(cfg, tests, args.container)
    print(f"wrote {write_results(result, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
