# Sentence Clustering API

Groups semantically similar sentences into clusters using FAISS-backed approximate nearest-neighbour search and multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start (Local)](#quick-start-local)
5. [Quick Start (Docker)](#quick-start-docker)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Design Decisions](#design-decisions)
9. [How to Scale for Concurrent Users](#how-to-scale-for-concurrent-users)
10. [How to Add a Feature](#how-to-add-a-feature)

---

## Overview

The service accepts a list of sentences (with IDs and optional timestamps), incrementally assigns them to semantic clusters, and returns the current cluster state. Sentences can be removed from clusters on subsequent calls via `remove_ids`.

**Use case:** trending topic detection — periodically push new articles/headlines and receive groupings of semantically related content.

---

## Architecture

The project follows **Hexagonal Architecture** (Ports & Adapters), also known as the *Clean Architecture* model.

```
┌──────────────────────────────────────────────────────────┐
│                        Adapters                          │
│                                                          │
│  Inbound (HTTP)          Outbound (ML / Storage)         │
│  ┌───────────────┐       ┌──────────────────────────┐    │
│  │  FastAPI      │       │  FaissClustererAdapter   │    │
│  │  Router       │       │  (wraps SentenceClusterer│    │
│  │  Schemas      │       │   + asyncio.Lock)        │    │
│  └──────┬────────┘       └────────────┬─────────────┘    │
│         │                             │                  │
│         ▼                             ▼                  │
│  ┌──────────────────────────────────────────────────┐    │
│  │              Application Layer                   │    │
│  │           ClusteringService (use case)           │    │
│  └──────────────────────┬───────────────────────────┘    │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │                  Domain Layer                    │    │
│  │   Sentence  ·  Cluster  ·  ClusteringPort (ABC) │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  Infrastructure: Settings · DI wiring · lifespan        │
└──────────────────────────────────────────────────────────┘
        │
        ▼
   ml/sentence_clusterer.py   ← pure ML engine (no framework deps)
```

### Layer responsibilities

| Layer | Responsibility | May depend on |
|---|---|---|
| **Domain** | Business models (`Sentence`, `Cluster`) and the `ClusteringPort` interface | Nothing |
| **Application** | Use-case orchestration (`ClusteringService`) | Domain only |
| **Adapters/Inbound** | HTTP concerns: Pydantic schemas, routing | Application + Domain |
| **Adapters/Outbound** | Concrete ML implementation | Domain port + `ml/` |
| **Infrastructure** | Framework wiring, config, DI | All layers |
| **ml/** | Standalone ML engine | NumPy, FAISS, SentenceTransformers |

---

## Project Structure

```
.
├── app/
│   ├── main.py                          # FastAPI app + lifespan
│   ├── domain/
│   │   ├── models/
│   │   │   ├── sentence.py              # Sentence dataclass
│   │   │   └── cluster.py              # Cluster dataclass
│   │   └── ports/
│   │       └── clustering_port.py      # Abstract ClusteringPort
│   ├── application/
│   │   └── clustering_service.py       # Use-case: cluster_sentences / get_stats
│   ├── adapters/
│   │   ├── inbound/http/
│   │   │   ├── router.py               # FastAPI endpoints
│   │   │   └── schemas.py              # Pydantic request/response
│   │   └── outbound/
│   │       └── faiss_clusterer_adapter.py  # Implements ClusteringPort
│   └── infrastructure/
│       ├── config.py                   # Pydantic Settings
│       └── dependencies.py             # FastAPI DI wiring
├── ml/
│   └── sentence_clusterer.py           # Core FAISS+SentenceTransformer engine
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- (Optional) CUDA-enabled GPU for faster encoding

```bash
# 1. Clone / enter the project
cd api_for_pjack

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env if needed (defaults work out of the box)

# 5. Start the server
uvicorn app.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## Quick Start (Docker)

```bash
# Build and start
docker compose up --build

# Stop
docker compose down
```

To persist the cluster state across restarts, set `CLUSTER_MODEL_SAVE_PATH=./data/model` in `.env`. The compose file already mounts `./data/model` into the container.

---

## API Reference

### `POST /api/v1/clusters` — Cluster sentences

Submit sentences to the clustering engine. Returns all clusters that currently meet the `least_items` threshold.

**Request body**

```json
{
  "sentences": [
    { "id": "doc-1", "text": "ราคาน้ำมันพุ่งสูงขึ้น" },
    { "id": "doc-2", "text": "น้ำมันแพงขึ้นอีกครั้ง" },
    { "id": "doc-3", "text": "หุ้นไทยปรับตัวขึ้น", "timestamp": "2024-01-15T08:00:00" }
  ],
  "remove_ids": [],
  "least_items": 2
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `sentences` | array | ✓ | Sentences to add |
| `sentences[].id` | string | ✓ | Unique doc ID |
| `sentences[].text` | string | ✓ | Sentence text |
| `sentences[].timestamp` | ISO datetime | | Defaults to request time |
| `remove_ids` | string[] | | IDs to evict from all clusters |
| `least_items` | int ≥ 1 | | Min cluster size to return (default 2) |

**Response**

```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "sentence_ids": ["doc-1", "doc-2"],
      "count": 2,
      "created_at": "2024-01-15T08:00:00"
    }
  ],
  "total_clusters": 1
}
```

Clusters are sorted by `count` descending (most populated first), then `created_at` descending.

---

### `GET /api/v1/clusters/stats` — Engine stats

```json
{
  "total_clusters": 42,
  "active_faiss_entries": 42,
  "running_cluster_index": 57
}
```

---

### `GET /health` — Health check

```json
{
  "status": "ok",
  "total_clusters": 42,
  "active_faiss_entries": 42,
  "running_cluster_index": 57
}
```

---

## Configuration

All settings use the `CLUSTER_` prefix and can be set as environment variables or in `.env`.

| Variable | Default | Description |
|---|---|---|
| `CLUSTER_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace model ID |
| `CLUSTER_MAX_TOKEN` | `128` | Max token length for encoder |
| `CLUSTER_USE_GPU` | `false` | Use CUDA if available |
| `CLUSTER_THRESHOLD` | `0.9` | Cosine similarity to join a cluster |
| `CLUSTER_TOP_K` | `5` | FAISS candidate neighbours |
| `CLUSTER_MAX_CLUSTERS` | `1000000` | Evict oldest when exceeded |
| `CLUSTER_BATCH_SIZE` | `64` | Encoder batch size |
| `CLUSTER_MODEL_SAVE_PATH` | _(empty)_ | Persist state here; leave blank for ephemeral |
| `CLUSTER_API_PREFIX` | `/api/v1` | URL prefix for all routes |

---

## Design Decisions

### Why hexagonal architecture?

Each layer has a single reason to change:
- Switch from FAISS to a different vector DB → only change `FaissClustererAdapter`
- Add gRPC transport → only add a new inbound adapter
- Change business rules (e.g. scoring) → only change `ClusteringService`

### Why stateful clustering?

FAISS maintains the embedding index in memory for sub-millisecond retrieval. Rebuilding the index on every request would be O(n²) per batch. Incremental updates keep latency constant regardless of total corpus size.

### Why `asyncio.Lock` + `ThreadPoolExecutor(1)`?

`SentenceClusterer.update()` mutates shared FAISS state and is **not thread-safe**. The lock ensures serial access. The executor moves CPU-heavy work off the event loop so other async tasks (health checks, etc.) stay responsive during encoding.

### Why one process per container (not gunicorn multi-worker)?

Multi-worker gunicorn forks the process, giving each worker an **independent** FAISS index. This is fine for read-only or sharded workloads, but breaks the single shared state model. Instead, scale **horizontally** (see next section).

---

## How to Scale for Concurrent Users

### Single-instance optimisation (vertical)

- Set `CLUSTER_BATCH_SIZE` higher (e.g. 128) to improve GPU throughput.
- Enable `CLUSTER_USE_GPU=true` and use `faiss-gpu` for 10–100× faster search.
- Requests are already queued behind an `asyncio.Lock` — no changes needed.

### Horizontal scaling (sharded topics)

When one instance becomes a bottleneck, shard by topic/category:

```
Client → Load Balancer (nginx / k8s Ingress)
               ├─ /api/v1/clusters?shard=0 → api_0 (owns topics A–M)
               ├─ /api/v1/clusters?shard=1 → api_1 (owns topics N–Z)
               └─ ...
```

Each shard has its own FAISS state. Route requests consistently to the same shard (e.g. hash `where` or topic key).

**docker compose scale example**

```bash
# Start 3 replicas — add a matching nginx upstream for each
docker compose up --scale api=3
```

### Fully stateless scaling (no shared state needed)

If you don't need state across calls (one-shot clustering per request), disable persistence and add a new endpoint that creates a temporary `SentenceClusterer`, clusters, and discards it. Each replica is then fully independent and stateless — standard round-robin load balancing works.

### Message-queue approach (async / high-throughput)

For very high ingest rates:

```
Producer → Kafka/RabbitMQ → Worker pool → Results store (Redis/DB)
                                                 ↑
                                           GET /results/{job_id}
```

Add a `POST /api/v1/clusters/async` endpoint that enqueues the request and returns a `job_id`, then a `GET /api/v1/clusters/results/{job_id}` to poll. Workers consume from the queue sequentially, maintaining cluster state safely.

---

## How to Add a Feature

### Example: add a new endpoint `GET /api/v1/clusters/{cluster_id}`

1. **schemas.py** — add `ClusterDetailResponse` if needed
2. **router.py** — add the new route, call the service
3. **clustering_service.py** — add `get_cluster(cluster_id)` use-case method
4. **clustering_port.py** — add `get_cluster` to the abstract interface
5. **faiss_clusterer_adapter.py** — implement `get_cluster`
6. **ml/sentence_clusterer.py** — add the data accessor if needed

This flow ensures: HTTP concerns stay in the adapter, business logic stays in the service, ML details stay in `ml/`.

### Example: swap FAISS for a hosted vector DB (e.g. Qdrant)

1. Create `app/adapters/outbound/qdrant_clusterer_adapter.py` implementing `ClusteringPort`
2. In `main.py` lifespan, instantiate `QdrantClustererAdapter` instead of `FaissClustererAdapter`
3. Zero changes to domain, application, or inbound adapter layers
