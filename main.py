"""
FastAPI application — single-file sentence clustering API.

Pipeline flow:
  1. Request arrives (POST /clusters or /clusters/assign)
  2. Texts are MD5-fingerprinted for deduplication
  3. SentenceClusterer (FAISS + AutoModel/AutoTokenizer) processes them
  4. Results are mapped back to human-readable text and returned
"""
import time
import asyncio
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

import yaml
from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModel, AutoTokenizer

from sentence_clusterer import SentenceClusterer
from zip_log_file_handling import setup_logging


# ══════════════════════════════════════════════════════════════════════════════
# Configuration (loaded from config.yaml)
# ══════════════════════════════════════════════════════════════════════════════

cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}

setup_logging(cfg["logging"])
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Shared state & clustering helper
# ══════════════════════════════════════════════════════════════════════════════

_model: Optional[AutoModel] = None
_tokenizer: Optional[AutoTokenizer] = None


def _new_clusterer() -> SentenceClusterer:
    if _model is None or _tokenizer is None:
        raise RuntimeError("Shared model/tokenizer not initialised — check lifespan.")
    return SentenceClusterer(
        model=_model,
        tokenizer=_tokenizer,
        max_token=cfg["model"]["max_token"],
        threshold=cfg["clustering"]["threshold"],
        top_k=cfg["clustering"]["top_k"],
        max_clusters=cfg["clustering"]["max_clusters"],
        use_gpu=cfg["model"]["use_gpu"],
    )


def _text_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


async def _run_clustering(
    texts: list[str],
    least_items: int = 2,
    threshold: float | None = None,
) -> list[dict]:
    """
    Run the full clustering pipeline: fingerprint → FAISS update → map back to text.
    Each call gets a fresh SentenceClusterer so state never leaks between requests.
    """
    if threshold is None:
        threshold = cfg["clustering"]["threshold"]

    now = datetime.now()
    doc_ids = [_text_id(t) for t in texts]
    timestamps = [now] * len(texts)

    text_registry: dict[str, str] = {}
    for doc_id, text in zip(doc_ids, texts):
        text_registry.setdefault(doc_id, text)

    clusterer = _new_clusterer()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="faiss")

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(
        executor,
        lambda: clusterer.update(
            texts=texts,
            doc_ids=doc_ids,
            timestamps=timestamps,
            batch_size=cfg["clustering"]["batch_size"],
            least_items=least_items,
            threshold=threshold,
        ),
    )
    executor.shutdown(wait=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info("FAISS update complete | raw_clusters=%d  elapsed=%.1fms", len(raw), elapsed_ms)

    return [
        {
            "cluster_id": r["cluster_id"],
            "sentences": [text_registry.get(did, did) for did in r["doc_id_list"]],
            "count": r["count"],
        }
        for r in raw
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Application bootstrap & endpoints
# ══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _model, _tokenizer

    m = cfg["model"]
    device = "cuda" if m["use_gpu"] else "cpu"

    embedding_model_path = m["embedding_model_path"]
    tokenizer_path = m["tokenizer_path"]

    logger.info(
        "Startup — loading model & tokenizer | embedding_model=%s  tokenizer=%s  device=%s  max_token=%d",
        embedding_model_path, tokenizer_path, device, m["max_token"],
    )

    t0 = time.perf_counter()
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _model = AutoModel.from_pretrained(embedding_model_path).to(device)
    _model.eval()
    elapsed = time.perf_counter() - t0

    logger.info("Model & tokenizer ready | elapsed=%.2fs  hidden_size=%d", elapsed, _model.config.hidden_size)
    logger.info(
        "Clustering config | threshold=%.2f  top_k=%d  max_clusters=%s  batch_size=%d",
        cfg["clustering"]["threshold"], cfg["clustering"]["top_k"],
        cfg["clustering"]["max_clusters"], cfg["clustering"]["batch_size"],
    )

    a = cfg["api"]
    logger.info("API listening | host=%s  port=%d  prefix=%s", a["host"], a["port"], a["prefix"])
    print(
        f"\n"
        f"  Sentence Clustering API is running\n"
        f"  Swagger UI : http://{a['host']}:{a['port']}/docs\n"
        f"  ReDoc      : http://{a['host']}:{a['port']}/redoc\n"
        f"  Health     : http://{a['host']}:{a['port']}/health\n"
    )

    yield

    logger.info("Shutdown signal received — cleanup complete.")


def create_app() -> FastAPI:
    prefix = cfg["api"]["prefix"]

    app = FastAPI(
        title="Sentence Clustering API",
        description=(
            "Groups semantically similar sentences into clusters using "
            "FAISS-backed approximate nearest-neighbour search and "
            "multilingual sentence embeddings."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def access_log(request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("HTTP %s %s | status=%d  elapsed=%.1fms", request.method, request.url.path, response.status_code, elapsed_ms)
        return response

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok"}

    # ── POST /clusters ────────────────────────────────────────────────────────
    @app.post(f"{prefix}/clusters", tags=["clusters"])
    async def cluster_sentences(
        body: dict,
        threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
        least_items: int = Query(default=2, ge=1),
        limit_cluster: int = Query(default=0, ge=0),
        debug: bool = Query(default=False)
    ):
        if threshold is None:
            threshold = cfg["clustering"]["threshold"]

        sentences = body.get("sentences", [])
        texts = [t or "" for t in sentences]
        logger.info("POST /clusters | sentences=%d  least_items=%d", len(texts), least_items)

        clusters = await _run_clustering(texts, least_items=least_items, threshold=threshold)
        if limit_cluster > 0:
            clusters = clusters[:limit_cluster]

        output = {"clusters": clusters, "total_clusters": len(clusters)}

        if debug:
            os.makedirs("result", exist_ok=True)
            with open(f"result/clusters_{int(time.time())}.json", "w", encoding="utf-8") as f:
                yaml.dump(output, f, allow_unicode=True)
                
        return output

    # ── POST /clusters/assign ─────────────────────────────────────────────────
    @app.post(f"{prefix}/clusters/assign", tags=["clusters"])
    async def assign_clusters(  body: list[dict],
                                threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
                                limit_cluster: int = Query(default=0, ge=0),
                                limit_cluster_item: int = Query(default=0, ge=0),
                                debug: bool = Query(default=False)
                            ):
        logger.info("POST /clusters/assign | documents=%d", len(body))
        try:
            if threshold is None:
                threshold = cfg["clustering"]["threshold"]

            texts = [f"{doc.get('Headline') or ''} {doc.get('Story') or ''}".strip() for doc in body]
            doc_id_list = [doc.get("DocumentID", "") for doc in body]

            clusters = await _run_clustering(texts, least_items=1, threshold=threshold)

            # Map text → cluster_id
            text_to_cluster: dict[str, int] = {}
            for c in clusters:
                for sentence in c["sentences"]:
                    text_to_cluster[sentence] = c["cluster_id"]

            assignments = [(did, text_to_cluster.get(txt, -1)) for txt, did in zip(texts, doc_id_list)]

            # Group by cluster, sort by size descending
            cluster_groups: dict[int, list[tuple[str, int]]] = {}
            for doc_id, cluster_id in assignments:
                cluster_groups.setdefault(cluster_id, []).append((doc_id, cluster_id))

            sorted_clusters = sorted(cluster_groups.values(), key=len, reverse=True)
            if limit_cluster > 0:
                sorted_clusters = sorted_clusters[:limit_cluster]
            if limit_cluster_item > 0:
                sorted_clusters = [g[:limit_cluster_item] for g in sorted_clusters]

            result_items = [item for group in sorted_clusters for item in group]

            headline_map = {doc.get("DocumentID", ""): doc.get("Headline") for doc in body}
            story_map = {doc.get("DocumentID", ""): doc.get("Story") for doc in body}
            
            output = [
                {
                    "DocumentID": doc_id,
                    "Headline": headline_map.get(doc_id),
                    "Story": story_map.get(doc_id),
                    "Cluster": str(cluster_id),
                }
                for doc_id, cluster_id in result_items
            ]
            
            if debug:
                os.makedirs("result", exist_ok=True)
                logger.info("Debug mode enabled — saving cluster assignments to result/assign_{timestamp}.json")
                logger.info(f"Threshold: {threshold}\n")
                with open(f"result/assign_{int(time.time())}.json", "w", encoding="utf-8") as f:
                    yaml.dump(output, f, allow_unicode=True)
            return output
        except Exception as e:
            logger.exception("Error in /clusters/assign: %s", str(e))
            return {"error": str(e)}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=False,
    )
