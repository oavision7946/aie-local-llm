import json
import os

import pytest

from bench import serve_bench as sb


def test_synthetic_lines_are_deterministic_per_seed():
    assert sb.synthetic_log_lines(5, seed=7) == sb.synthetic_log_lines(5, seed=7)
    assert sb.synthetic_log_lines(5, seed=7) != sb.synthetic_log_lines(5, seed=8)


def test_prefill_prompt_header_differs_per_seed():
    # A unique first line keeps a server's prefix cache from reusing an earlier prompt.
    a = sb.build_prefill_prompt(3, seed=1).splitlines()[0]
    b = sb.build_prefill_prompt(3, seed=2).splitlines()[0]
    assert a != b


def test_lines_for_tokens():
    assert sb.lines_for_tokens(1064, tokens_per_line=10.0) == 100
    assert sb.lines_for_tokens(10, tokens_per_line=10.0) == 1
    with pytest.raises(ValueError):
        sb.lines_for_tokens(100, tokens_per_line=0)


def test_stream_result_rates_exclude_first_token():
    r = sb.StreamResult(prompt_tokens=1000, completion_tokens=101, ttft_s=0.5, total_s=10.5)
    assert r.decode_tps == pytest.approx(10.0)
    assert r.prefill_tps == pytest.approx(2000.0)


def test_stream_result_degenerate_cases_are_zero():
    assert sb.StreamResult(10, 1, 0.1, 0.2).decode_tps == 0.0
    assert sb.StreamResult(10, 5, 0.0, 0.0).prefill_tps == 0.0


def _call(name, **args):
    return (name, json.dumps(args))


GOOD_CALLS = [
    _call("search_logs", service="payments", query="timeout", window_minutes=30),
    _call("get_trace", trace_id="a1b2c3d4e5f60718"),
    _call("get_metric", service="payments", metric="p99_latency_ms", window_minutes=30),
]


def test_grade_tool_calls_pass():
    result = sb.grade_tool_calls(GOOD_CALLS)
    assert result == {"passed": True, "n_calls": 3, "problems": []}


def test_grade_tool_calls_missing_and_duplicate():
    result = sb.grade_tool_calls([GOOD_CALLS[1], GOOD_CALLS[1]])
    assert not result["passed"]
    assert any("missing calls" in p for p in result["problems"])
    assert any("duplicate calls" in p for p in result["problems"])


@pytest.mark.parametrize(
    "name,raw,fragment",
    [
        ("get_trace", "{not json", "not JSON"),
        ("get_trace", "[]", "not an object"),
        ("get_trace", "{}", "missing 'trace_id'"),
        ("get_trace", '{"trace_id": "x", "extra": 1}', "unexpected 'extra'"),
        ("get_metric", '{"service": "a", "metric": "mem", "window_minutes": 5}', "not in enum"),
        ("search_logs", '{"service": "a", "query": "q", "window_minutes": "5"}', "type str"),
        ("search_logs", '{"service": "a", "query": "q", "window_minutes": true}', "type bool"),
        ("delete_all", "{}", "unknown tool"),
    ],
)
def test_check_tool_args_problems(name, raw, fragment):
    problems = sb.check_tool_args(name, raw)
    assert any(fragment in p for p in problems), problems


VALID_HYPOTHESIS = {
    "component": "payments",
    "category": "dependency",
    "confidence": 0.7,
    "evidence": [{"id": "ev-3", "summary": "upstream timeout"}],
}


def test_grade_json_output_pass():
    assert sb.grade_json_output(json.dumps(VALID_HYPOTHESIS)) == {"passed": True, "problems": []}


@pytest.mark.parametrize(
    "mutation",
    [
        {"category": "cosmic_rays"},
        {"confidence": 1.5},
        {"evidence": []},
        {"evidence": [{"id": "3", "summary": "bad id"}]},
        {"extra": True},
    ],
)
def test_grade_json_output_rejects_schema_violations(mutation):
    assert not sb.grade_json_output(json.dumps({**VALID_HYPOTHESIS, **mutation}))["passed"]


@pytest.mark.parametrize("content", [None, "", "not json"])
def test_grade_json_output_rejects_non_json(content):
    assert not sb.grade_json_output(content)["passed"]


def test_schema_and_pydantic_model_agree_on_fields():
    assert set(sb.HYPOTHESIS_SCHEMA["required"]) == set(sb.Hypothesis.model_fields)


def test_csv_row_and_write_results(tmp_path):
    result = {
        "label": "engine-x",
        "started_at": "2026-01-01T00:00:00",
        "max_model_len": 32768,
        "memory_before": {"host_used_mib": 1000},
        "decode": {"decode_tps_median": 12.5},
        "concurrency": {"per_stream_decode_tps_mean": 10.0, "aggregate_tps": 19.0},
        "prefill": [
            {"target_tokens": 8192, "prompt_tokens": 8100, "ttft_s": 4.0, "prefill_tps": 2025.0},
            {"target_tokens": 32768, "error": "too long"},
        ],
        "tools": {"passed": True},
        "json": {"pass_rate": "3/3"},
    }
    row = sb.csv_row(result)
    assert row["decode_tps"] == 12.5
    assert row["tools"] == "pass"
    assert row["prefill"] == "8100tok: 4.0s (2025.0 tok/s); 32768tok: error"

    sb.write_results(result, tmp_path)
    sb.write_results({**result, "label": "engine-y"}, tmp_path)
    assert json.loads((tmp_path / "engine-x.json").read_text())["label"] == "engine-x"
    lines = (tmp_path / "results.csv").read_text().splitlines()
    assert lines[0].startswith("label,")
    assert len(lines) == 3


def test_evict_page_cache_counts_bytes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 50)
    if not hasattr(os, "posix_fadvise"):
        pytest.skip("posix_fadvise not available on this platform")
    assert sb.evict_page_cache([tmp_path]) == 150


def test_evict_page_cache_counts_symlinked_files_once(tmp_path):
    if not hasattr(os, "posix_fadvise"):
        pytest.skip("posix_fadvise not available on this platform")
    (tmp_path / "blob").write_bytes(b"z" * 40)
    (tmp_path / "link").symlink_to(tmp_path / "blob")
    assert sb.evict_page_cache([tmp_path]) == 40


def test_cli_rejects_unknown_tests():
    with pytest.raises(SystemExit):
        sb.main(["run", "--base-url", "http://x", "--model", "m", "--label", "l", "--tests", "x"])


def test_evict_page_cache_skips_unreadable_files(tmp_path):
    if not hasattr(os, "posix_fadvise") or os.geteuid() == 0:
        pytest.skip("needs posix_fadvise and a non-root user")
    (tmp_path / "ok").write_bytes(b"a" * 10)
    locked = tmp_path / "locked"
    locked.write_bytes(b"b" * 10)
    locked.chmod(0)
    assert sb.evict_page_cache([tmp_path]) == 10
