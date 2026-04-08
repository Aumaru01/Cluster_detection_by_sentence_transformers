"""
Core sentence clustering engine using FAISS for approximate nearest-neighbour search
and SentenceTransformer for multilingual embeddings.

Removed from original: NER extraction/matching, trending score weights,
blacklist logic, dead/commented-out methods (ask_tmp).
"""

import logging
import os
import pickle
import time
from datetime import datetime
from typing import Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceClusterer:
    def __init__(
        self,
        load_path: Optional[str] = None,
        model: Optional[SentenceTransformer] = None,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_token: int = 128,
        threshold: float = 0.9,
        top_k: int = 5,
        max_clusters: Optional[int] = None,
        use_gpu: bool = False,
    ) -> None:
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"

        if load_path:
            logger.info("Loading SentenceClusterer from disk | path=%s", load_path)
            self._load(load_path)
            logger.info(
                "Loaded | clusters=%d  embedding_dim=%d  assigned_ids=%d",
                len(self.trend),
                self.embedding_dim,
                len(self._assigned_ids),
            )
            return

        if model:
            self.model = model
            logger.debug("Using pre-loaded shared model | id=%d", id(model))
        else:
            logger.info("Loading SentenceTransformer | name=%s  device=%s", model_name, self.device)
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.max_seq_length = max_token
            logger.info("SentenceTransformer loaded")

        self.threshold = threshold
        self.top_k = top_k
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))

        # cluster_id → set of doc_ids
        self.trend: dict[int, set] = {}
        # cluster_id → {"sum": np.ndarray, "count": int, "created_at": datetime}
        self.cluster_stats: dict[int, dict] = {}
        self.cluster_now_idx: int = 0
        self.oldest_cluster_idx: int = 0
        self.max_clusters = max_clusters
        # flat set of all doc_ids currently assigned to any cluster (maintained incrementally)
        self._assigned_ids: set[str] = set()

        logger.debug(
            "SentenceClusterer initialised | embedding_dim=%d  threshold=%.2f  top_k=%d  max_clusters=%s",
            self.embedding_dim,
            self.threshold,
            self.top_k,
            self.max_clusters,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: str) -> None:
        logger.info("Saving state | dir=%s  clusters=%d", save_dir, len(self.trend))
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "faiss_index"))
        self.model.save(os.path.join(save_dir, "model"))
        with open(os.path.join(save_dir, "attributes.pkl"), "wb") as f:
            pickle.dump(
                {
                    "threshold": self.threshold,
                    "top_k": self.top_k,
                    "trend": self.trend,
                    "cluster_stats": self.cluster_stats,
                    "cluster_now_idx": self.cluster_now_idx,
                    "oldest_cluster_idx": self.oldest_cluster_idx,
                    "max_clusters": self.max_clusters,
                },
                f,
            )
        logger.info("State saved successfully | dir=%s", save_dir)

    def _load(self, load_dir: str) -> None:
        self.model = SentenceTransformer(
            os.path.join(load_dir, "model"), device=self.device
        )
        self.index = faiss.read_index(os.path.join(load_dir, "faiss_index"))
        with open(os.path.join(load_dir, "attributes.pkl"), "rb") as f:
            attrs = pickle.load(f)
        self.threshold = attrs["threshold"]
        self.top_k = attrs["top_k"]
        self.trend = attrs["trend"]
        self.cluster_stats = attrs["cluster_stats"]
        self.cluster_now_idx = attrs["cluster_now_idx"]
        self.oldest_cluster_idx = attrs["oldest_cluster_idx"]
        self.max_clusters = attrs["max_clusters"]
        self.embedding_dim = self.index.d
        self._assigned_ids = set().union(*self.trend.values()) if self.trend else set()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        logger.debug("Encoding %d texts | device=%s", len(texts), self.device)
        t0 = time.perf_counter()
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=128,
            normalize_embeddings=True,
            device=self.device,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result = embeddings.astype(np.float32)
        logger.debug(
            "Encoding done | texts=%d  shape=%s  elapsed=%.1fms",
            len(texts),
            result.shape,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Internal cluster operations
    # ------------------------------------------------------------------

    def _match_cluster(self, embeddings: np.ndarray, threshold: float | None = None) -> list[int]:
        """Return the best matching cluster id (≥0) or -1 for each embedding."""
        effective_threshold = threshold if threshold is not None else self.threshold
        n = len(embeddings)
        if self.index.ntotal > 0:
            D, I = self.index.search(embeddings, self.top_k)
            logger.debug(
                "FAISS search | queries=%d  index_size=%d  top_k=%d",
                n,
                self.index.ntotal,
                self.top_k,
            )
        else:
            D = np.zeros((n, self.top_k), dtype=np.float32)
            I = -np.ones((n, self.top_k), dtype=np.int64)
            logger.debug("FAISS index empty — all queries will create new clusters")

        matched: list[int] = []
        n_matched = 0
        for idx in range(n):
            found = -1
            for dist, cluster_id in zip(D[idx], I[idx]):
                if dist >= effective_threshold and cluster_id != -1:
                    found = int(cluster_id)
                    n_matched += 1
                    break
            matched.append(found)

        logger.debug(
            "FAISS match results | queries=%d  matched=%d  new=%d  threshold=%.2f",
            n,
            n_matched,
            n - n_matched,
            effective_threshold,
        )
        return matched

    def _add_to_cluster(
        self,
        cluster_id: int,
        embedding: np.ndarray,
        doc_ids: list[str],
        created_at: datetime,
    ) -> None:
        is_new = cluster_id not in self.cluster_stats
        self.trend.setdefault(cluster_id, set()).update(doc_ids)
        self._assigned_ids.update(doc_ids)
        if is_new:
            self.cluster_stats[cluster_id] = {
                "sum": embedding.copy(),
                "count": 1,
                "created_at": created_at,
            }
            logger.debug(
                "New cluster created | cluster_id=%d  doc_ids=%s",
                cluster_id,
                doc_ids,
            )
        else:
            self.cluster_stats[cluster_id]["sum"] += embedding
            self.cluster_stats[cluster_id]["count"] += 1
            logger.debug(
                "Added to existing cluster | cluster_id=%d  cluster_size=%d  doc_ids=%s",
                cluster_id,
                len(self.trend[cluster_id]),
                doc_ids,
            )

    def _evict_oldest_cluster(self) -> None:
        if self.oldest_cluster_idx not in self.trend:
            self.oldest_cluster_idx = min(self.trend.keys())
        evicted_ids = self.trend[self.oldest_cluster_idx]
        logger.info(
            "Evicting oldest cluster (max_clusters limit reached) | cluster_id=%d  size=%d  total_clusters=%d",
            self.oldest_cluster_idx,
            len(evicted_ids),
            len(self.trend),
        )
        self._assigned_ids -= evicted_ids
        del self.trend[self.oldest_cluster_idx]
        del self.cluster_stats[self.oldest_cluster_idx]
        self.index.remove_ids(
            np.array([self.oldest_cluster_idx], dtype=np.int64)
        )
        self.oldest_cluster_idx += 1

    def approximate_group_vectors(
        self, embeddings: np.ndarray, threshold: float = 0.99
    ) -> list[list[int]]:
        """Greedily group near-duplicate vectors by cosine similarity."""
        groups: list[list[int]] = []
        ungrouped = set(range(len(embeddings)))
        while ungrouped:
            i = ungrouped.pop()
            group = [i]
            to_remove = [
                j for j in ungrouped
                if np.dot(embeddings[i], embeddings[j]) >= threshold
            ]
            for j in to_remove:
                ungrouped.remove(j)
            group.extend(to_remove)
            groups.append(group)
        logger.debug(
            "Within-batch grouping | inputs=%d  groups=%d  threshold=%.2f",
            len(embeddings),
            len(groups),
            threshold,
        )
        return groups

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update(
        self,
        texts: list[str],
        doc_ids: list[str],
        timestamps: list[datetime],
        batch_size: int = 64,
        least_items: int = 2,
        threshold: float | None = None,
    ) -> list[dict]:
        """
        Incrementally update cluster state with new sentences and return the
        current cluster list filtered to clusters with at least `least_items` sentences.
        """
        new_ids = set(doc_ids) - self._assigned_ids
        skipped = len(doc_ids) - len(new_ids)

        logger.info(
            "Update called | total=%d  new=%d  already_assigned=%d",
            len(doc_ids),
            len(new_ids),
            skipped,
        )
        if skipped:
            logger.debug("Skipping %d already-assigned doc_ids", skipped)

        filtered = [
            (t, d, ts)
            for t, d, ts in zip(texts, doc_ids, timestamps)
            if d in new_ids
        ]

        if filtered:
            f_texts, f_ids, f_times = zip(*filtered)
            effective_threshold = threshold if threshold is not None else self.threshold
            logger.debug(
                "Processing %d new documents | threshold=%.2f",
                len(f_texts), effective_threshold,
            )
            self._process_batch(list(f_texts), list(f_ids), list(f_times), batch_size, effective_threshold)
        else:
            logger.debug("No new documents to process — all inputs were already assigned")

        self._cleanup_memory()
        result = self._build_result(least_items)

        logger.info(
            "Update complete | total_clusters=%d  qualifying_clusters=%d  "
            "assigned_ids=%d  index_size=%d",
            len(self.trend),
            len(result),
            len(self._assigned_ids),
            self.index.ntotal,
        )
        return result

    def _process_batch(
        self,
        texts: list[str],
        doc_ids: list[str],
        timestamps: list[datetime],
        batch_size: int,
        threshold: float,
    ) -> None:
        sorted_triples = sorted(zip(texts, doc_ids, timestamps), key=lambda x: x[0])
        texts, doc_ids, timestamps = map(list, zip(*sorted_triples))

        t0 = time.perf_counter()
        # Encoding is still batched for performance
        embeddings = self.encode(texts)
        total = len(texts)

        # Phase 1: group near-identical sentences across ALL data at once (0.99 threshold)
        # Each group is treated as one representative centroid going into FAISS
        groups = self.approximate_group_vectors(embeddings)
        centroids, group_ids, group_times = [], [], []
        for g_indices in groups:
            centroid = np.mean(embeddings[g_indices], axis=0).astype(np.float32)
            centroids.append(centroid)
            group_ids.append([doc_ids[j] for j in g_indices])
            group_times.append(min(timestamps[j] for j in g_indices))

        logger.debug(
            "Phase 1 complete | texts=%d  groups=%d  duplicates_collapsed=%d",
            total, len(groups), total - len(groups),
        )

        # Phase 2: assign each centroid to a FAISS cluster sequentially
        # Sequential so each newly created cluster is immediately visible to the next centroid
        n_new = n_existing = 0
        for g_doc_ids, centroid, ts in zip(group_ids, centroids, group_times):
            matched = self._match_cluster(centroid.reshape(1, -1), threshold=threshold)
            cluster_id = matched[0]

            if cluster_id >= 0:
                self._add_to_cluster(cluster_id, centroid, g_doc_ids, ts)
                n_existing += 1
            else:
                if (
                    self.max_clusters is not None
                    and len(self.cluster_stats) >= self.max_clusters
                ):
                    self._evict_oldest_cluster()

                new_id = self.cluster_now_idx
                self.index.add_with_ids(
                    np.array([centroid]),
                    np.array([new_id], dtype=np.int64),
                )
                self._add_to_cluster(new_id, centroid, g_doc_ids, ts)
                self.cluster_now_idx += 1
                n_new += 1

        logger.debug(
            "Phase 2 complete | centroids=%d  join_existing=%d  create_new=%d",
            len(groups), n_existing, n_new,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "_process_batch complete | texts=%d  groups=%d  elapsed=%.1fms",
            total, len(groups), elapsed_ms,
        )

    def _cleanup_memory(self) -> None:
        if self.use_gpu:
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")

    def _build_result(self, least_items: int) -> list[dict]:
        result = sorted(
            [
                {
                    "cluster_id": int(cid),
                    "doc_id_list": sorted(doc_ids),
                    "count": len(doc_ids),
                }
                for cid, doc_ids in self.trend.items()
                if len(doc_ids) >= least_items
            ],
            key=lambda x: (x["count"], self.cluster_stats[x["cluster_id"]]["created_at"]),
            reverse=True,
        )
        logger.debug(
            "Built result | qualifying=%d  total_clusters=%d  least_items=%d",
            len(result),
            len(self.trend),
            least_items,
        )
        return result
