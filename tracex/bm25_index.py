"""BM25 lexical recall index.

Uses rank-bm25 (BM25Okapi) with k1/b tuned to match the Elasticsearch
BM25 configuration described in the architecture. Swap this module for
an Elasticsearch client if you need synonym filters or lemmatization at scale.
"""

from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from .loader import Requirement

# Download tokenizer data silently on first run
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

_STOPWORDS = set(stopwords.words("english"))


def _tokenize(text: str) -> List[str]:
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in _STOPWORDS]


class BM25Index:
    def __init__(
        self,
        requirements: List[Requirement],
        k1: float = 1.7,
        b: float = 0.7,
    ) -> None:
        self.requirements = requirements
        self._id_to_idx = {r.id: i for i, r in enumerate(requirements)}
        tokenized = [_tokenize(r.cleaned_text) for r in requirements]
        self.bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Requirement, float]]:
        """Return top-k candidates sorted by BM25 score (descending)."""
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        # Filter zero-score hits to avoid noisy candidates
        return [
            (self.requirements[i], float(s))
            for i, s in ranked[:top_k]
            if s > 0.0
        ]
