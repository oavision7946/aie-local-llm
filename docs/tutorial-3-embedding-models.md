# Tutorial 3: Configure a retrieval-tuned embedding model

Embeddings always run **in-process** in the gateway, via `sentence-transformers`
(`app/services/rag/embeddings/sentence_transformers_embedder.py`) — never in a Docker container, and
never on the engine's GPU memory budget unless you explicitly set `device: cuda`. Switching models is a
`config/rag.yaml` change, not a code change.

This works two ways:

- **Isolated** — no LLM running at all. Start the gateway and call `/documents` directly; embeddings
  don't depend on `config/engines.yaml`'s active engine being up.
- **Alongside a large LLM** — same as [Tutorial 2](tutorial-2-llm-and-embeddings.md), budgeting GPU
  memory across both if you put the embedder on `cuda`.

## 0. The `embeddings` config fields

```yaml
# config/rag.yaml
embeddings:
  model: BAAI/bge-small-en-v1.5   # any sentence-transformers-compatible repo id
  device: cpu                     # or "cuda", headroom permitting
  trust_remote_code: false        # required by some models with custom modeling code
  query_prompt: null              # literal string prepended to queries only
  document_prompt: null           # literal string prepended to documents only
  truncate_dim: null              # shrink the output vector (Matryoshka-trained models)
  dtype: null                     # forwarded as model_kwargs={"dtype": ...}; e.g. "bfloat16"
```

`query_prompt`/`document_prompt` exist because several newer embedding models expect a literal
instruction prefix — and expect a **different** one (or none) on the query side than the document side.
`RagService` calls `Embedder.embed_query()` for the search query and `Embedder.embed()` for ingested
chunks, and `SentenceTransformersEmbedder` applies the matching prefix to each. Get this wrong (e.g. same
prefix on both, or no prefix where the model needs one) and retrieval quality degrades or the model's own
documented usage is violated — this project treats it as plain config rather than per-model code, so you
copy the prefix out of the model card verbatim into YAML.

## 1. Memory budget

| Model | Repo | Checkpoint | Dim (max) | Max input |
|---|---|--:|--:|--:|
| Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B` | ~1.2 GB | 1024 (Matryoshka) | 32K tokens |
| Qwen3-Embedding-4B | `Qwen/Qwen3-Embedding-4B` | ~8.0 GB | 2560 (Matryoshka) | 32K tokens |
| Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B` | ~15.1 GB | 4096 (Matryoshka) | 32K tokens |
| BGE-M3 | `BAAI/bge-m3` | ~2.3 GB | 1024 | 8K tokens |
| EmbeddingGemma-300m | `google/embeddinggemma-300m` | ~1.2 GB | 768 (Matryoshka to 128) | 2K tokens |
| Harrier-OSS-v1-27B | `microsoft/harrier-oss-v1-27b` | ~54 GB (BF16) | 5376 | 32K tokens |

All of these fit on CPU on this box's 128GB unified memory; Harrier-OSS-v1-27B is the one case where you
should think about it explicitly — see step 4.

## 2. Verified configs

Each of these was actually loaded and used to embed a query and a document on this box (CPU, this
project's current `sentence-transformers==6.1.0` / `transformers==5.18.0`) before being written here —
not just copied from the model card.

**Qwen3-Embedding-0.6B** (swap in `-4B` or `-8B` for better quality at more memory; same loading path,
not independently re-verified at those sizes):

```yaml
embeddings:
  model: Qwen/Qwen3-Embedding-0.6B
  device: cpu
  query_prompt: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
  # document_prompt stays unset — Qwen3-Embedding wants no prefix on documents.
  # truncate_dim: 512  # optional: shrink the 1024-dim output to save vector-store memory
```

**BGE-M3** — no prompt needed; the card explicitly says it doesn't require one:

```yaml
embeddings:
  model: BAAI/bge-m3
  device: cpu
```

> BGE-M3 also supports sparse (lexical) and multi-vector (ColBERT-style) retrieval, but **not** through
> plain `sentence-transformers`, which only exposes the dense vector used above. Those modes need the
> separate `FlagEmbedding` package and aren't implemented in this project — `bge-m3` here is a dense
> embedder like the others, nothing more.

**EmbeddingGemma-300m:**

```yaml
embeddings:
  model: google/embeddinggemma-300m
  device: cpu
  query_prompt: "task: search result | query: "
  document_prompt: "title: none | text: "
  # truncate_dim: 256  # optional Matryoshka truncation (also 512, 128)
```

> Hugging Face may show this model as requiring you to accept Google's Gemma license on the model page
> before your token can download it. It loaded without any extra step on the account this was tested
> with; if yours gets a 403, accept the license at the model's HF page first.

## 3. Documented but not run here: Harrier-OSS-v1-27B

```yaml
embeddings:
  model: microsoft/harrier-oss-v1-27b
  device: cpu            # or cuda, sized per Tutorial 2, if you have the headroom
  dtype: bfloat16         # the card loads it as model_kwargs={"dtype": "auto"}; pin bfloat16
                           # explicitly so the checkpoint doesn't default to float32 and double
                           # its ~54GB footprint
  query_prompt: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
```

This one was **not loaded on this box** — 54GB (BF16) was more than this pass justified downloading just
to confirm a config. It's a decoder-only model (unusual for an embedder) using last-token pooling, which
its card's own `sentence-transformers` example handles internally — no `trust_remote_code` needed. Smaller
siblings exist in the same family (`microsoft/harrier-oss-v1-270m`, `microsoft/harrier-oss-v1-0.6b`) if
you want to sanity-check the config path before committing to the 27B download.

## 4. Known not to work here: Nomic Embed Text v2 MoE

`nomic-ai/nomic-embed-text-v2-moe` needs `trust_remote_code: true` (it ships custom modeling code), and
that code requires the `einops` package, which isn't a dependency of this project. Installing `einops`
gets past that, but the model then fails to load on this project's `transformers==5.18.0`:

```
AttributeError: 'NomicBertModel' object has no attribute 'get_extended_attention_mask'
```

This is Nomic's custom code calling a `transformers` internal (`ModuleUtilsMixin.get_extended_attention_mask`)
that no longer exists in `transformers` 5.x. Fixing it would mean pinning an older `transformers` in this
project's shared environment, which risks breaking the other embedding models above — not worth it for
one model. Treat this one as unsupported until either Nomic updates the remote code or this project moves
to a different `sentence-transformers`/`transformers` pin and someone re-checks it.

## 5. Running isolated (no LLM)

```bash
uv sync
uv run uvicorn app.main:app --port 9100
```

```bash
curl -X POST http://127.0.0.1:9100/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "The aie-local-llm gateway loads embedding models in-process via sentence-transformers.", "corpus": "docs"}'
```

This triggers the model's first load (slow: download + load) — everything after reuses it. No chat
engine needs to be running for this to work; `config/engines.yaml`'s active engine is only touched by
`/v1/chat/completions`.

## 6. Running alongside a large LLM

Follow [Tutorial 2](tutorial-2-llm-and-embeddings.md) for the memory-budgeting pattern (size the LLM's
`--gpu-memory-utilization` to leave headroom, then set `embeddings.device: cuda`). The same applies here
for any of this tutorial's models — swap `nvidia/Nemotron-3-Embed-8B-BF16` in that tutorial's step 3 for
any config block above.
