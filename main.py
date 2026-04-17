"""
FastAPI application — single-file sentence clustering API.

Pipeline flow:
  1. Request arrives (POST /clusters or /clusters/assign)
  2. Texts are MD5-fingerprinted for deduplication
  3. SentenceClusterer (FAISS + AutoModel/AutoTokenizer) processes them
  4. Results are mapped back to human-readable text and returned
"""
import json
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
from typing import AsyncIterator, Optional, Union

import yaml
from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
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
    now = datetime.now()
    doc_ids = [_text_id(t) for t in texts]
    timestamps = [now] * len(texts)

    text_registry: dict[str, str] = {}
    for doc_id, text in zip(doc_ids, texts):
        text_registry.setdefault(doc_id, text)

    clusterer = SentenceClusterer()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="faiss")

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(
        executor,
        lambda: clusterer.update(
            texts=texts,
            doc_ids=doc_ids,
            timestamps=timestamps,
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
    # Eagerly load model & tokenizer on startup (cached inside SentenceClusterer)
    SentenceClusterer()

    c = cfg["clustering"]
    logger.info(
        "Clustering config | threshold=%.2f  top_k=%d  max_clusters=%s",
        c["threshold"], c["top_k"], c["max_clusters"],
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
        version="2.1.1",
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
        ):
        
        debug = cfg["debug"]

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
                json.dump(output, f, ensure_ascii=False, indent=4)

        return output

    # ── POST /clusters/assign ─────────────────────────────────────────────────
    @app.post(f"{prefix}/clusters/assign", tags=["clusters"])
    async def assign_clusters(  body: Union[list[dict], dict],
                                threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
                                limit_cluster: int = Query(default=0, ge=0),
                                limit_cluster_item: int = Query(default=0, ge=0),
                            ):
        
        debug = cfg["debug"]
        #save body in debug
        if debug:
                os.makedirs("result", exist_ok=True)
                logger.info("Debug mode enabled — saving input documents to result/assign_input_{timestamp}.json")
                with open(f"result/body_{int(time.time())}.json", "w", encoding="utf-8") as f:
                    json.dump(body, f, ensure_ascii=False, indent=4)
        
        if isinstance(body, dict):
            body = [body]
            
        logger.info("POST /clusters/assign | documents=%d", len(body))
        try:
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
                with open(f"result/clustered_{int(time.time())}.json", "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=4)
                    
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