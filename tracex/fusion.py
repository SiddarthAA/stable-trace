"""Score fusion and thresholding.

Each raw score is min-max normalised across the current candidate set so
all three signals contribute on the same [0, 1] scale regardless of their
natural magnitudes.

Final = α·BM25_norm + β·Dense_norm + γ·Cross_norm
"""

from typing import List, Tuple

import numpy as np

from .config import Config
from .loader import Requirement


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


def fuse_and_filter(
    candidates: List[Tuple[Requirement, float, float, float]],
    config: Config,
) -> List[dict]:
    """
    Normalise, fuse, filter by threshold, and return top-N results.

    Each result dict contains:
        "requirement": Requirement
        "scores":
            bm25        – raw BM25 score
            bm25_norm   – normalised BM25
            dense       – cosine similarity
            cross       – cross-encoder probability
            final       – fused score
    """
    if not candidates:
        return []

    bm25_arr = np.array([c[1] for c in candidates], dtype=np.float32)
    dense_arr = np.array([c[2] for c in candidates], dtype=np.float32)
    cross_arr = np.array([c[3] for c in candidates], dtype=np.float32)

    bm25_norm = _minmax(bm25_arr)
    dense_norm = _minmax(dense_arr)
    cross_norm = _minmax(cross_arr)

    final_scores = (
        config.alpha * bm25_norm
        + config.beta * dense_norm
        + config.gamma * cross_norm
    )

    results = []
    for i, (req, bm25, dense, cross) in enumerate(candidates):
        final = float(final_scores[i])
        if final >= config.final_threshold:
            results.append(
                {
                    "requirement": req,
                    "scores": {
                        "bm25": round(float(bm25), 4),
                        "bm25_norm": round(float(bm25_norm[i]), 4),
                        "dense": round(float(dense), 4),
                        "cross": round(float(cross), 4),
                        "final": round(final, 4),
                    },
                }
            )

    results.sort(key=lambda x: x["scores"]["final"], reverse=True)
    return results[: config.top_n]
