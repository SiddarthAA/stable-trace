"""Dense embedding index using FAISS HNSW + BAAI/bge-large-en-v1.5.

Architecture notes:
- Embeddings are L2-normalised so inner-product == cosine similarity.
- FAISS HNSW is used as the vector store (M=64, efSearch=256).
- During reranking we look up precomputed vectors for BM25 candidates
  and compute dot-product similarity directly — no ANN search needed here.
"""

from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Config
from .loader import Requirement


class DenseIndex:
    def __init__(
        self,
        requirements: List[Requirement],
        config: Config,
        model: Optional[SentenceTransformer] = None,
    ) -> None:
        self.requirements = requirements
        self.config = config
        self._id_to_idx: dict[str, int] = {r.id: i for i, r in enumerate(requirements)}

        # Accept a pre-loaded model to avoid re-downloading between level pairs
        self.model = model or SentenceTransformer(config.dense_model, device=config.device)
        self._build_index()

    def _build_index(self) -> None:
        texts = [r.cleaned_text for r in self.requirements]
        self.embeddings: np.ndarray = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=self.config.batch_size,
        ).astype(np.float32)  # (N, dim)

        dim = self.embeddings.shape[1]
        self._index = faiss.IndexHNSWFlat(dim, self.config.faiss_m)
        self._index.hnsw.efSearch = self.config.faiss_ef_search
        self._index.add(self.embeddings)

    def encode_query(self, text: str) -> np.ndarray:
        """Return normalised query embedding of shape (1, dim)."""
        return self.model.encode(
            [text],
            normalize_embeddings=True,
            batch_size=1,
        ).astype(np.float32)

    def rerank_candidates(
        self,
        query: str,
        candidates: List[Tuple[Requirement, float]],
    ) -> List[Tuple[Requirement, float, float]]:
        """
        Compute cosine similarity between query and each BM25 candidate.

        Returns list of (requirement, bm25_score, dense_score) filtered to:
          - cosine >= dense_threshold, OR
          - top dense_top_k if nothing clears the threshold.
        """
        if not candidates:
            return []

        query_emb = self.encode_query(query)  # (1, dim)

        # Look up precomputed embeddings for the candidate subset
        cand_embs = np.stack(
            [self.embeddings[self._id_to_idx[r.id]] for r, _ in candidates]
        )  # (n, dim)

        # Cosine similarity: inner product of normalised vectors
        sims = (query_emb @ cand_embs.T).flatten()  # (n,)

        results: List[Tuple[Requirement, float, float]] = [
            (req, bm25, float(sim))
            for (req, bm25), sim in zip(candidates, sims)
        ]
        results.sort(key=lambda x: x[2], reverse=True)

        above = [r for r in results if r[2] >= self.config.dense_threshold]
        pool = above if above else results
        return pool[: self.config.dense_top_k]
