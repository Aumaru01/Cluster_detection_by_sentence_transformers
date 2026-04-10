# Sentence Clustering API

Groups semantically similar sentences into clusters using FAISS-backed approximate nearest-neighbour search and multilingual sentence embeddings (`facebook/drama-large`).

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

1. Tokenize with `AutoTokenizer` (padding + truncation)
2. Forward pass through `AutoModel`
3. Mean pooling over token embeddings (attention-mask aware)
4. L2-normalise → unit vectors for cosine similarity via inner product

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

The API is available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

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
  embedding_model_path: facebook/drama-large
  tokenizer_path: facebook/drama-large
  max_token: 128
  use_gpu: false

clustering:
  threshold: 0.6
  top_k: 5
  max_clusters: 1000000
  batch_size: 64

persistence:
  save_path: null

api:
  prefix: /api/v1
  host: 0.0.0.0
  port: 50000

logging:
  level: INFO
  log_dir: logs
  max_bytes: 10485760    # 10 MB
```

| Setting | Default | Description |
|---|---|---|
| `model.embedding_model_path` | `facebook/drama-large` | HuggingFace model for embeddings |
| `model.tokenizer_path` | `facebook/drama-large` | HuggingFace tokenizer (usually same as model) |
| `model.max_token` | `128` | Max token length for encoder |
| `model.use_gpu` | `false` | Use CUDA if available |
| `clustering.threshold` | `0.6` | Cosine similarity threshold to join a cluster |
| `clustering.top_k` | `5` | FAISS candidate neighbours per query |
| `clustering.max_clusters` | `1000000` | Evict oldest cluster when exceeded |
| `clustering.batch_size` | `64` | Texts per forward pass |
| `persistence.save_path` | `null` | Set to a path to persist state on shutdown |
| `api.prefix` | `/api/v1` | URL prefix for all routes |
| `api.host` | `0.0.0.0` | Bind address |
| `api.port` | `50000` | Port number |
| `logging.level` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `logging.log_dir` | `logs` | Directory for log files |
| `logging.max_bytes` | `10485760` | Rotate when log file exceeds this size |

Alternative models (commented out in config):

- `facebook/drama-1b` — larger, slower, potentially higher quality
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — much smaller & faster (384 dim)

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

- Set `batch_size` higher (e.g. 128) to improve GPU throughput.
- Enable `use_gpu: true` for 10–100× faster encoding.
- Consider ONNX Runtime for 2–4× CPU inference speedup without changing the model.

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
                   