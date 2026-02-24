from dataclasses import dataclass, field


def resolve_device(requested: str) -> str:
    """
    Resolve the requested device string to a concrete "cpu" or "cuda" value.

    Accepted values:
      "cpu"   — always use CPU
      "cuda"  — use GPU; raises a clear error if CUDA is not available
      "auto"  — use CUDA if available, otherwise fall back to CPU silently
    """
    requested = requested.lower().strip()

    if requested == "cpu":
        return "cpu"

    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if requested == "auto":
        return "cuda" if cuda_available else "cpu"

    if requested == "cuda":
        if not cuda_available:
            raise ValueError(
                "Device 'cuda' was requested but CUDA is not available on this machine. "
                "Use --device cpu or --device auto to run on CPU."
            )
        return "cuda"

    raise ValueError(
        f"Unknown device '{requested}'. Valid options: cpu, cuda, auto."
    )


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

    # Hardware — resolved to "cpu" or "cuda" by resolve_device()
    device: str = "cpu"
    batch_size: int = 32
