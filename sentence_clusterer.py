"""
Core sentence clustering engine using FAISS for approximate nearest-neighbour search
and HuggingFace Transformers (AutoModel + AutoTokenizer) for multilingual embeddings.

Encoding pipeline:
  1. Tokenize with AutoTokenizer (padding + truncation)
  2. Forward pass through AutoModel
  3. Mean pooling over token embeddings (attention-mask aware)
  4. L2-normalise → unit vectors for cosine similarity via inner product
"""

import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# ── Load config once at module level ─────────────────────────────────────────
_cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings weighted by the attention mask."""
    token_embeddings = model_output[0]  # (batch, seq_len, hidden)
    # keep the mask in the same dtype as token_embeddings to avoid an implicit fp32 upcast
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(
        mask_expanded.sum(dim=1), min=1e-9
    )


def _gpu_mem_report(reset_peak: bool = False) -> str:
    """
    Human-readable GPU memory snapshot for the current CUDA device.

    Format: ``alloc=1.23GB  peak=2.45GB  reserved=2.67GB  free=5.12/24.00GB``

    * ``alloc``     — tensors currently held by PyTorch.
    * ``peak``      — high-water mark since the last reset (reset on startup by default).
    * ``reserved``  — memory the caching allocator has reserved from CUDA.
    * ``free/total``— device-wide free and total VRAM (other processes included).

    Returns ``"gpu_unavailable"`` when CUDA isn't usable. Pass ``reset_peak=True``
    to start a fresh per-operation peak measurement.
    """
    if not torch.cuda.is_available():
        return "gpu_unavailable"

    GB = 1024 ** 3
    try:
        alloc = torch.cuda.memory_allocated() / GB
        peak = torch.cuda.max_memory_allocated() / GB
        reserved = torch.cuda.memory_reserved() / GB
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb, total_gb = free_bytes / GB, total_bytes / GB

        if reset_peak:
            torch.cuda.reset_peak_memory_stats()

        return (
            f"alloc={alloc:.2f}GB  peak={peak:.2f}GB  "
            f"reserved={reserved:.2f}GB  free={free_gb:.2f}/{total_gb:.2f}GB"
        )
    except Exception as e:  # pragma: no cover — diagnostics only, never crash
        return f"gpu_mem_report_failed: {e}"


def _resolve_dtype(use_gpu: bool, requested: str) -> torch.dtype:
    """
    Pick the best torch dtype for the target device.

    Rules:
      * CPU → always fp32 (fp16/bf16 on CPU is slower and not worth it).
      * GPU + 'auto' → bf16 if hardware supports it (Ampere+), else fp16.
      * GPU + explicit 'bf16' / 'fp16' / 'fp32' → honour the request, with a safe
        fallback to fp16 if bf16 isn't supported on this GPU.
    """
    req = (requested or "auto").strip().lower()

    if not use_gpu or not torch.cuda.is_available():
        return torch.float32

    if req == "fp32":
        return torch.float32
    if req == "fp16":
        return torch.float16
    if req == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        logger.warning("bf16 requested but GPU does not support it — falling back to fp16")
        return torch.float16

    # auto
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class SentenceClusterer:
    # Class-level cache — model & tokenizer are loaded once and shared across all instances
    _model: Optional[AutoModel] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _dtype: torch.dtype = torch.float32

    def __init__(self, load_path: Optional[str] = None) -> None:
        _m = _cfg["model"]
        _c = _cfg["clustering"]

        self.use_gpu: bool = _m["use_gpu"]
        self.device: str = "cuda" if self.use_gpu else "cpu"
        self.max_token: int = _m["max_token"]
        # micro-batch size for encoding — caps peak VRAM regardless of input length
        self.encode_batch_size: int = int(_m.get("encode_batch_size", 32))

        # ── Model & Tokenizer (load once, cache at class level, Run only when starting "class SentenceClusterer") ──
        if SentenceClusterer._model is None:
            precision_cfg = _m.get("precision", "auto")
            dtype = _resolve_dtype(self.use_gpu, precision_cfg)
            SentenceClusterer._dtype = dtype

            logger.info(
                "Loading AutoModel & AutoTokenizer | model=%s  tokenizer=%s  device=%s  dtype=%s  precision=%s",
                _m["embedding_model_path"], _m["tokenizer_path"], self.device, dtype, precision_cfg,
            )
            t0 = time.perf_counter()
            SentenceClusterer._tokenizer = AutoTokenizer.from_pretrained(_m["tokenizer_path"])

            # Load weights directly into the target dtype — avoids the fp32 copy on CPU
            # that `low_cpu_mem_usage=True` further reduces. SDPA attention cuts the
            # attention-matrix memory footprint when the model's config supports it.
            load_kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
            }
            try:
                SentenceClusterer._model = AutoModel.from_pretrained(
                    _m["embedding_model_path"],
                    attn_implementation="sdpa",
                    **load_kwargs,
                ).to(self.device)
            except (TypeError, ValueError) as e:
                logger.info("SDPA attention not supported for this model — using default attention (%s)", e)
                SentenceClusterer._model = AutoModel.from_pretrained(
                    _m["embedding_model_path"],
                    **load_kwargs,
                ).to(self.device)

            SentenceClusterer._model.eval()
            elapsed = time.perf_counter() - t0

            # Reset peak stats so the first encode() reports per-operation peak only,
            # not the transient spike from weight upload.
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            logger.info(
                "Model & tokenizer ready | elapsed=%.2fs  hidden_size=%d  dtype=%s  gpu_mem=[%s]",
                elapsed, SentenceClusterer._model.config.hidden_size, SentenceClusterer._dtype,
                _gpu_mem_report(),
            )

        self.model = SentenceClusterer._model
        self.tokenizer = SentenceClusterer._tokenizer
        self._dtype = SentenceClusterer._dtype

        if load_path:
            logger.info("Loading SentenceClusterer state from disk | path=%s", load_path)
            self._load(load_path)
            logger.info(
                "Loaded | clusters=%d  embedding_dim=%d  assigned_ids=%d",
                len(self.trend),
                self.embedding_dim,
                len(self._assigned_ids),
            )
            return

        self.threshold: float = _c["threshold"]
        self.top_k: int = _c["top_k"]
        self.embedding_dim: int = self.model.config.hidden_size
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))

        # cluster_id → set of doc_ids
        self.trend: dict[int, set] = {}
        # cluster_id → {"sum": np.ndarray, "count": int, "created_at": datetime}
        self.cluster_stats: dict[int, dict] = {}
        self.cluster_now_idx: int = 0
        self.oldest_cluster_idx: int = 0
        self.max_clusters: Optional[int] = _c["max_clusters"]
        # flat set of all doc_ids currently assigned to any cluster (maintained incrementally)
        self._assigned_ids: set[str] = set()

        logger.debug(f"SentenceClusterer initialised | embedding_dim={self.embedding_dim}  threshold={self.threshold:.2f}  top_k={self.top_k}  max_clusters={self.max_clusters}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    # save index + attributes needed to reconstruct state, but not the model/tokenizer which are cached at class level
    def save(self, save_dir: str) -> None:
        logger.info(f"Saving state | dir={save_dir}  clusters={len(self.trend)}")
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "faiss_index"))
        self.model.save_pretrained(os.path.join(save_dir, "model"))
        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
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
                    "max_token": self.max_token,
                },
                f,
            )
        logger.info(f"State saved successfully | dir={save_dir}")

    # load index + attributes, but not the model/tokenizer which are cached at class level
    def _load(self, load_dir: str) -> None:
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
        self.max_token = attrs.get("max_token", 128)
        self.embedding_dim = self.index.d
        self._assigned_ids = set().union(*self.trend.values()) if self.trend else set()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Tokenize → forward → mean pooling → L2 normalise.

        Memory-minimising strategy:
          * Split inputs into micro-batches of ``self.encode_batch_size`` so peak VRAM
            stays constant regardless of how many texts the caller passes in.
          * Within each call, process batches in ascending-length order. Shorter batches
            pad to shorter sequences, wasting less activation memory than a single
            uniformly long batch would.
          * Run the forward pass under ``torch.inference_mode()`` — stricter than
            ``no_grad`` and skips version-counter bookkeeping, saving a small amount
            of memory and overhead.
          * Release GPU tensors the moment embeddings are copied to CPU.

        Returns (N, hidden_size) float32 numpy array of unit-length embeddings
        in the caller's original order.
        """
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        logger.debug("Encoding %d texts | device=%s  dtype=%s  batch_size=%d", n, self.device, self._dtype, self.encode_batch_size)
        t0 = time.perf_counter()

        # Reset peak so the peak reported below reflects this encode() call only.
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Sort indices by raw character length — a cheap, tokenizer-free proxy for
        # token count. Groups similar-length items together so padding waste stays low.
        order = sorted(range(n), key=lambda i: len(texts[i]))

        out = np.empty((n, self.embedding_dim), dtype=np.float32)
        batch_size = max(1, self.encode_batch_size)

        # Prefer the memory-efficient SDPA kernel when we're on GPU with fp16/bf16.
        use_sdpa_ctx = self.use_gpu and self._dtype in (torch.float16, torch.bfloat16)

        for start in range(0, n, batch_size):
            batch_idx = order[start:start + batch_size]
            batch_texts = [texts[i] for i in batch_idx]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_token,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                if use_sdpa_ctx:
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_mem_efficient=True, enable_math=True
                    ):
                        model_output = self.model(**encoded)
                else:
                    model_output = self.model(**encoded)
                pooled = mean_pooling(model_output, encoded["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)

            # Copy to CPU as fp32 — FAISS IndexFlatIP requires float32. Doing the
            # cast here (instead of keeping GPU tensors alive) lets autograd/tensor
            # memory be reclaimed before the next batch.
            emb_cpu = pooled.detach().to(dtype=torch.float32, device="cpu").numpy()
            del encoded, model_output, pooled

            for row_idx, original_i in enumerate(batch_idx):
                out[original_i] = emb_cpu[row_idx]

        # Capture peak BEFORE flushing the cache — empty_cache() doesn't affect peak
        # but this keeps the ordering explicit.
        gpu_mem = _gpu_mem_report() if self.use_gpu else ""
        if self.use_gpu:
            torch.cuda.empty_cache()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        n_batches = (n + batch_size - 1) // batch_size
        if self.use_gpu:
            logger.info(
                "Encoding done | texts=%d  shape=%s  elapsed=%.1fms  batches=%d  gpu_mem=[%s]",
                n, out.shape, elapsed_ms, n_batches, gpu_mem,
            )
        else:
            logger.debug(
                "Encoding done | texts=%d  shape=%s  elapsed=%.1fms  batches=%d",
                n, out.shape, elapsed_ms, n_batches,
            )
        return out

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
            self._process_batch(list(f_texts), list(f_ids), list(f_times), effective_threshold)
        else:
            logger.debug("No new documents to process — all inputs were already assigned")

        self._cleanup_memory()
        result = self._build_result(least_items)

        extra = f"  gpu_mem=[{_gpu_mem_report()}]" if self.use_gpu else ""
        logger.info(
            "Update complete | total_clusters=%d  qualifying_clusters=%d  "
            "assigned_ids=%d  index_size=%d%s",
            len(self.trend),
            len(result),
            len(self._assigned_ids),
            self.index.ntotal,
            extra,
        )
        return result

    def _process_batch(
        self,
        texts: list[str],
        doc_ids: list[str],
        timestamps: list[datetime],
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

        logger.debug(f"Phase 2 complete | centroids={len(groups)}  join_existing={n_existing} create_new={n_new}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"_process_batch complete | texts={total}  groups={len(groups)}  elapsed={elapsed_ms:.1f}ms")

    def _cleanup_memory(self) -> None:
        if self.use_gpu:
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared | gpu_mem=[%s]", _gpu_mem_report())

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
