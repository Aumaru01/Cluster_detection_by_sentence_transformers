# Sentence Clustering API

Groups semantically similar sentences into clusters using FAISS-backed approximate nearest-neighbour search and multilingual sentence embeddings (`jinaai/jina-embeddings-v3-hf` by default).

The encoder runs in **mixed precision (bf16/fp16) with length-bucketed micro-batching** so GPU memory stays flat regardless of input size, while preserving the exact same L2-normalised fp32 embeddings that FAISS consumes.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)
6. [Design Decisions](#design-decisions)
7. [How to Scale for Concurrent Users](#how-to-scale-for-concurrent-users)

---

## Overview

The service accepts a list of sentences, incrementally assigns them to semantic clusters via FAISS, and returns the current cluster state. Each request gets a **fresh** `SentenceClusterer` so state never leaks between calls.

**Use case:** trending topic detection — periodically push new articles/headlines and receive groupings of semantically related content.

### Encoding pipeline

1. Sort input sentences by length and split into **micro-batches** of `encode_batch_size` (default 32) so peak VRAM is bounded.
2. Tokenize each batch with `AutoTokenizer` (padding + truncation).
3. Forward pass through `AutoModel` under `torch.inference_mode()` in the target dtype (bf16/fp16 on GPU, fp32 on CPU), using SDPA attention (flash / memory-efficient kernels) when available.
4. Mean pooling over token embeddings (attention-mask aware, dtype-preserving — no implicit fp32 upcast).
5. L2-normalise → unit vectors for cosine similarity via inner product.
6. Cast to fp32 and copy to CPU; GPU tensors are released between micro-batches so memory is reused, not accumulated.
7. Re-assemble outputs in the caller's original order.

---

## Project Structure

```
.
├── main.py                    # FastAPI app, endpoints, model lifespan
├── sentence_clusterer.py      # Core FAISS + AutoModel/AutoTokenizer engine
├── zip_log_file_handling.py   # Rotating log handler with auto-compression
├── config.yaml                # All configurable parameters
├── requirements.txt           # Python dependencies
└── README.md
```

The project uses a **flat single-file architecture**: all HTTP endpoints live in `main.py`, and the ML engine lives in `sentence_clusterer.py`.

---

## Quick Start

### Prerequisites

- Python 3.11+
- (Optional) CUDA-enabled GPU for faster encoding

```bash
# 1. Clone / enter the project
cd transformer-sentence-cluster

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux
.venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Review config.yaml and adjust if needed (defaults work out of the box)

# 5. Start the server
python -m main
```

The API is available at `http://localhost:[port]`.
Interactive docs: `http://localhost:[port]/docs`

---

## API Reference

### `GET /health` — Health check

```json
{
  "status": "ok"
}
```

---

### `POST /api/v1/clusters` — Cluster sentences

Submit a list of raw sentence strings. Returns clusters that meet the `least_items` threshold.

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | float (0–1) | from config | Cosine similarity to join a cluster |
| `least_items` | int ≥ 1 | 2 | Min cluster size to return |
| `limit_cluster` | int ≥ 0 | 0 (no limit) | Max number of clusters to return |

**Request body**

```json
{
  "sentences": [
    "ราคาน้ำมันพุ่งสูงขึ้น",
    "น้ำมันแพงขึ้นอีกครั้ง",
    "หุ้นไทยปรับตัวขึ้น"
  ]
}
```

`sentences` is a flat list of strings (not objects). Texts are MD5-fingerprinted internally for deduplication.

**Response**

```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "sentences": ["ราคาน้ำมันพุ่งสูงขึ้น", "น้ำมันแพงขึ้นอีกครั้ง"],
      "count": 2
    }
  ],
  "total_clusters": 1
}
```

Clusters are sorted by `count` descending (most populated first).

---

### `POST /api/v1/clusters/assign` — Assign documents to clusters

Submit an array of documents with `Headline` and `Story` fields. Returns the same documents with an added `Cluster` field.

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | float (0–1) | from config | Cosine similarity to join a cluster |
| `limit_cluster` | int ≥ 0 | 0 (no limit) | Max number of cluster groups to return |
| `limit_cluster_item` | int ≥ 0 | 0 (no limit) | Max documents per cluster group |

**Request body**

```json
[
  {
    "DocumentID": "abc-123",
    "Headline": "ราคาน้ำมันพุ่งสูง",
    "Story": "ราคาน้ำมันดิบปรับตัวขึ้นอย่างต่อเนื่อง..."
  },
  {
    "DocumentID": "abc-456",
    "Headline": "น้ำมันแพงขึ้น",
    "Story": "ผู้บริโภคได้รับผลกระทบจากราคาน้ำมัน..."
  }
]
```

Accepts either a JSON array or a single object (auto-wrapped to array).

**Response**

```json
[
  {
    "DocumentID": "abc-123",
    "Headline": "ราคาน้ำมันพุ่งสูง",
    "Story": "ราคาน้ำมันดิบปรับตัวขึ้นอย่างต่อเนื่อง...",
    "Cluster": "0"
  },
  {
    "DocumentID": "abc-456",
    "Headline": "น้ำมันแพงขึ้น",
    "Story": "ผู้บริโภคได้รับผลกระทบจากราคาน้ำมัน...",
    "Cluster": "0"
  }
]
```

Documents are sorted by cluster size (largest cluster first). Documents that don't match any cluster get `"Cluster": "-1"`.

---

## Configuration

All settings are in `config.yaml`:

```yaml
model:
  embedding_model_path: jinaai/jina-embeddings-v3-hf
  tokenizer_path: jinaai/jina-embeddings-v3-hf
  max_token: 128
  use_gpu: true
  precision: auto          # auto | bf16 | fp16 | fp32
  encode_batch_size: 32    # micro-batch size for encoding

clustering:
  threshold: 0.6
  top_k: 5
  max_clusters: 1000000

api:
  prefix: /api/v1
  host: 0.0.0.0
  port: 50000

logging:
  level: INFO
  log_dir: logs
  max_bytes: 10485760    # 10 MB

debug: false
```

| Setting | Default | Description |
|---|---|---|
| `model.embedding_model_path` | `jinaai/jina-embeddings-v3-hf` | HuggingFace model for embeddings |
| `model.tokenizer_path` | `jinaai/jina-embeddings-v3-hf` | HuggingFace tokenizer (usually same as model) |
| `model.max_token` | `128` | Max token length for encoder |
| `model.use_gpu` | `true` | Use CUDA if available |
| `model.precision` | `auto` | Model weight dtype. `auto` picks bf16 on Ampere+, else fp16; `fp32` on CPU |
| `model.encode_batch_size` | `32` | Micro-batch size for encoding — lower → less peak VRAM, higher → faster throughput |
| `clustering.threshold` | `0.6` | Cosine similarity threshold to join a cluster |
| `clustering.top_k` | `5` | FAISS candidate neighbours per query |
| `clustering.max_clusters` | `1000000` | Evict oldest cluster when exceeded |
| `api.prefix` | `/api/v1` | URL prefix for all routes |
| `api.host` | `0.0.0.0` | Bind address |
| `api.port` | `50000` | Port number |
| `logging.level` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `logging.log_dir` | `logs` | Directory for log files |
| `logging.max_bytes` | `10485760` | Rotate when log file exceeds this size |
| `debug` | `false` | If `true`, writes request/response JSON to `result/` |

Alternative models (commented out in config):

- `facebook/drama-1b` — larger, slower, potentially higher quality
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — much smaller & faster (384 dim)

---

## GPU Memory Optimisation

The encoder is tuned so that peak VRAM stays low without sacrificing embedding quality or throughput. Five techniques work together:

### 1. Auto-detected mixed precision

Model weights load directly in `bfloat16` on Ampere-class GPUs and newer (A100, H100, RTX 30xx/40xx) or `float16` on older CUDA devices, halving the weight and activation footprint compared to fp32. CPU always falls back to fp32. Override with `model.precision: bf16 | fp16 | fp32` if you need deterministic behaviour.

**Why this is safe:** bf16 shares the same exponent range as fp32, so there's no risk of overflow in transformer activations. fp16's tiny rounding error is absorbed by the L2-normalisation step that follows mean pooling. FAISS still indexes and searches in fp32 — the cast happens at the CPU boundary.

### 2. Length-bucketed micro-batching

Rather than forwarding the entire request as one tensor, `encode()` sorts inputs by length and processes them in batches of `encode_batch_size`. This caps peak VRAM at a fixed value no matter how many sentences arrive, and each batch pads to the longest item **within that batch only**, slashing wasted activation memory on mixed-length inputs.

Drop `encode_batch_size` to 16 or 8 on low-VRAM GPUs; raise it to 64–128 when there's headroom and you want maximum throughput.

### 3. SDPA attention kernels

When available, the model loads with `attn_implementation="sdpa"` and runs inside a `sdp_kernel` context that enables flash-attention and memory-efficient attention. These kernels avoid materialising the full `seq_len × seq_len` attention matrix, which is the dominant cost at longer sequences.

### 4. `torch.inference_mode()` + low-CPU-memory loading

The forward pass runs inside `torch.inference_mode()` (stricter than `no_grad`, skips version-counter bookkeeping). Model weights load with `low_cpu_mem_usage=True` so there's no transient fp32 copy on CPU during startup.

### 5. Aggressive tensor release

After each micro-batch, input IDs / model outputs / pooled tensors are explicitly `del`-ed and the cached allocator is flushed between requests. Embeddings are moved to CPU as fp32 numpy arrays the moment they're ready, so GPU memory is reused — not accumulated — across batches.

### Rough numbers

For `jinaai/jina-embeddings-v3-hf` (hidden size 768) at `max_token=128`, `encode_batch_size=32`:

| Config | Approx. peak VRAM |
|---|---|
| fp32, no batching, 1000 sentences | ~4–5 GB |
| fp32, micro-batching | ~1.8 GB |
| **bf16 / fp16, micro-batching (default)** | **~0.9 GB** |

Actual numbers depend on GPU, driver, and sentence length distribution.

### Observability

Every request logs its own GPU memory snapshot at `INFO` level so you can tune `encode_batch_size` and `precision` against real traffic:

```
INFO  Model & tokenizer ready | elapsed=3.21s  hidden_size=768  dtype=torch.bfloat16
INFO  GPU memory after model load | allocated=580.4MB  peak=580.4MB  reserved=604.0MB
INFO  GPU memory during encode   | texts=128  batches=4  allocated=591.8MB  peak=942.1MB  reserved=1024.0MB
INFO  GPU memory after cleanup   | allocated=582.0MB  peak=942.1MB  reserved=604.0MB
```

Fields:
- `allocated` — live tensors PyTorch is holding right now
- `peak` — highest `allocated` since the last reset (reset on startup and after each request)
- `reserved` — total VRAM PyTorch's cached allocator is holding (includes free blocks it keeps for reuse)

Use `peak` to size your batch — if `peak` approaches total GPU memory, lower `encode_batch_size`; if it's well under, raise it for better throughput.

---

## Design Decisions

### Why stateful-per-request?

Each request creates a fresh `SentenceClusterer` so state never leaks between calls. The shared `AutoModel` and `AutoTokenizer` are loaded once at startup and reused across requests to avoid redundant model loading.

### Why `ThreadPoolExecutor(1)`?

`SentenceClusterer.update()` is CPU-heavy and blocks the event loop. The executor moves this work off the async loop so health checks and other endpoints stay responsive during encoding.

### Why one process per container (not gunicorn multi-worker)?

Multi-worker gunicorn forks the process, giving each worker an **independent** copy of model weights in memory. Since the model is loaded once and shared via global state, a single process is more memory-efficient. Scale **horizontally** with multiple containers instead (see next section).

### Why MD5 fingerprinting?

Texts are hashed to create stable document IDs for deduplication within a single request. Near-duplicate texts (similarity ≥ 0.99) are also collapsed during the clustering phase.

---

## How to Scale for Concurrent Users

### Single-instance optimisation (vertical)

- Raise `model.encode_batch_size` (e.g. 64–128) when there's spare VRAM — fewer kernel launches → better throughput.
- Keep `model.use_gpu: true` and let `model.precision: auto` pick bf16/fp16 for 2× memory savings at unchanged quality (see [GPU Memory Optimisation](#gpu-memory-optimisation)).
- Lower `encode_batch_size` to 16 or 8 on small GPUs to trade a little speed for a much smaller memory footprint.
- Consider ONNX Runtime / TensorRT for 2–4× CPU/GPU inference speedup without changing the model.

### Horizontal scaling (sharded topics)

When one instance becomes a bottleneck, shard by topic/category:

```
Client → Load Balancer (nginx / k8s Ingress)
               ├─ /api/v1/clusters?shard=0 → api_0 (owns topics A–M)
               ├─ /api/v1/clusters?shard=1 → api_1 (owns topics N–Z)
               └─ ...
```

Each shard has its own FAISS state. Route requests consistently to the same shard.

### Fully stateless scaling

Since each request already gets a fresh `SentenceClusterer`, replicas are fully independent. Standard round-robin load balancing works out of the box.