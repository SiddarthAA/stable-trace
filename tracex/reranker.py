"""Cross-encoder precision scoring.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, good quality)
Higher accuracy: cross-encoder/ms-marco-electra-base
Maximum accuracy: fine-tuned DeBERTa-v3-large on your domain data

Raw logits are converted to probabilities via sigmoid so all downstream
code sees scores in (0, 1).
"""

from typing import List, Tuple

import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder

from .config import Config
from .loader import Requirement


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


class CrossEncoderReranker:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = CrossEncoder(config.cross_encoder_model, device=config.device)

    def score(
        self,
        query: str,
        candidates: List[Tuple[Requirement, float, float]],
    ) -> List[Tuple[Requirement, float, float, float]]:
        """
        Score each (query, candidate) pair.

        Input:  (requirement, bm25_score, dense_score)
        Output: (requirement, bm25_score, dense_score, cross_score)

        cross_score is in (0, 1); higher = more likely to be a true link.
        """
        if not candidates:
            return []

        pairs = [(query, r.cleaned_text) for r, _, _ in candidates]
        raw: np.ndarray = np.asarray(self.model.predict(pairs), dtype=np.float32)
        cross_scores = _sigmoid(raw)

        return [
            (req, bm25, dense, float(cross))
            for (req, bm25, dense), cross in zip(candidates, cross_scores)
        ]
