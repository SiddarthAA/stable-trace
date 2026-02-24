from dataclasses import dataclass


@dataclass
class Config:
    # BM25 (rank-bm25, mirrors Elasticsearch BM25 tuning)
    bm25_top_k: int = 50
    bm25_k1: float = 1.7
    bm25_b: float = 0.7

    # Dense embedding (BAAI/bge-large-en-v1.5 via sentence-transformers)
    dense_model: str = "BAAI/bge-large-en-v1.5"
    dense_top_k: int = 15
    dense_threshold: float = 0.65  # cosine similarity cutoff
    faiss_m: int = 64              # HNSW M parameter
    faiss_ef_search: int = 256     # HNSW efSearch parameter

    # Cross-encoder reranker
    # Upgrade to: cross-encoder/ms-marco-electra-base for higher accuracy
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Score fusion weights (must sum to 1.0)
    alpha: float = 0.2   # BM25 weight
    beta: float = 0.3    # dense cosine weight
    gamma: float = 0.5   # cross-encoder weight

    # Final filtering
    final_threshold: float = 0.4   # minimum fused score to include a link
    top_n: int = 10                # max links returned per source requirement

    # Hardware
    device: str = "cpu"  # "cuda" for GPU
    batch_size: int = 32
