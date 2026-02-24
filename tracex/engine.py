"""Traceability engine — orchestrates the full BM25→Dense→CrossEncoder pipeline."""

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import track

from .bm25_index import BM25Index
from .config import Config
from .dense_index import DenseIndex
from .fusion import fuse_and_filter
from .loader import Requirement
from .normalizer import normalize_requirements
from .reranker import CrossEncoderReranker

console = Console()


class TraceabilityEngine:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._reranker: Optional[CrossEncoderReranker] = None
        self._dense_model = None  # shared across level pairs to avoid double-loading

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            console.print(
                f"  Loading cross-encoder: [yellow]{self.config.cross_encoder_model}[/yellow]"
            )
            self._reranker = CrossEncoderReranker(self.config)
        return self._reranker

    def _get_dense_model(self):
        if self._dense_model is None:
            from sentence_transformers import SentenceTransformer
            console.print(
                f"  Loading dense model: [yellow]{self.config.dense_model}[/yellow]"
            )
            self._dense_model = SentenceTransformer(
                self.config.dense_model, device=self.config.device
            )
        return self._dense_model

    def _match_pair(
        self,
        source_reqs: List[Requirement],
        target_reqs: List[Requirement],
        label: str,
    ) -> Dict[str, List[dict]]:
        """Run the full 4-step pipeline for one source→target level pair."""
        console.print(f"\n[bold cyan]Building indexes for {label}...[/bold cyan]")
        bm25_idx = BM25Index(target_reqs, k1=self.config.bm25_k1, b=self.config.bm25_b)
        dense_idx = DenseIndex(target_reqs, self.config, model=self._get_dense_model())
        reranker = self._get_reranker()

        results: Dict[str, List[dict]] = {}

        console.print(f"[bold cyan]Matching {len(source_reqs)} {label}...[/bold cyan]")
        for src in track(source_reqs, description=f"  {label}"):
            # Step 1 — Lexical recall (BM25)
            bm25_cands = bm25_idx.search(src.cleaned_text, top_k=self.config.bm25_top_k)
            if not bm25_cands:
                results[src.id] = []
                continue

            # Step 2 — Dense semantic reranking
            dense_cands = dense_idx.rerank_candidates(src.cleaned_text, bm25_cands)
            if not dense_cands:
                results[src.id] = []
                continue

            # Step 3 — Cross-encoder precision scoring
            cross_cands = reranker.score(src.cleaned_text, dense_cands)

            # Step 4 — Score fusion + threshold
            results[src.id] = fuse_and_filter(cross_cands, self.config)

        return results

    def run(
        self,
        system_reqs: Optional[List[Requirement]],
        high_reqs: List[Requirement],
        low_reqs: Optional[List[Requirement]],
    ) -> Dict[str, Any]:
        """
        Run traceability for all available level pairs and return a nested dict.

        Supported configurations:
          - system + high             → SLR→HLR
          - high + low                → HLR→LLR
          - system + high + low       → SLR→HLR→LLR (full bidirectional)
        """
        # Normalise all requirements in one pass
        all_reqs: List[Requirement] = []
        if system_reqs:
            all_reqs.extend(system_reqs)
        all_reqs.extend(high_reqs)
        if low_reqs:
            all_reqs.extend(low_reqs)
        normalize_requirements(all_reqs)

        slr_to_hlr: Optional[Dict[str, List[dict]]] = None
        hlr_to_llr: Optional[Dict[str, List[dict]]] = None

        if system_reqs:
            slr_to_hlr = self._match_pair(system_reqs, high_reqs, "SLR→HLR")
        if low_reqs:
            hlr_to_llr = self._match_pair(high_reqs, low_reqs, "HLR→LLR")

        return self._build_output(
            system_reqs, high_reqs, low_reqs, slr_to_hlr, hlr_to_llr
        )

    # ------------------------------------------------------------------
    # Output construction
    # ------------------------------------------------------------------

    def _build_output(
        self,
        system_reqs: Optional[List[Requirement]],
        high_reqs: List[Requirement],
        low_reqs: Optional[List[Requirement]],
        slr_to_hlr: Optional[Dict[str, List[dict]]],
        hlr_to_llr: Optional[Dict[str, List[dict]]],
    ) -> Dict[str, Any]:
        forward: Dict[str, Any] = {}

        if system_reqs and slr_to_hlr is not None:
            for slr in system_reqs:
                hlr_links = slr_to_hlr.get(slr.id, [])
                linked_hlr = []
                for link in hlr_links:
                    hlr_req: Requirement = link["requirement"]
                    entry: Dict[str, Any] = {
                        "hlr_id": hlr_req.id,
                        "description": hlr_req.description,
                        "confidence": link["scores"]["final"],
                        "scores": link["scores"],
                    }
                    if hlr_to_llr is not None:
                        entry["linked_llr"] = [
                            {
                                "llr_id": l["requirement"].id,
                                "description": l["requirement"].description,
                                "confidence": l["scores"]["final"],
                                "scores": l["scores"],
                            }
                            for l in hlr_to_llr.get(hlr_req.id, [])
                        ]
                    linked_hlr.append(entry)

                forward[slr.id] = {
                    "description": slr.description,
                    "linked_hlr": linked_hlr,
                }

        elif hlr_to_llr is not None:
            # 2-level: HLR → LLR only
            for hlr in high_reqs:
                forward[hlr.id] = {
                    "description": hlr.description,
                    "linked_llr": [
                        {
                            "llr_id": l["requirement"].id,
                            "description": l["requirement"].description,
                            "confidence": l["scores"]["final"],
                            "scores": l["scores"],
                        }
                        for l in hlr_to_llr.get(hlr.id, [])
                    ],
                }

        levels = []
        if system_reqs:
            levels.append("system")
        levels.append("high")
        if low_reqs:
            levels.append("low")

        return {
            "forward_links": forward,
            "reverse_links": self._build_reverse(forward, levels),
            "_meta": {
                "levels": levels,
                "counts": {
                    "system": len(system_reqs) if system_reqs else 0,
                    "high": len(high_reqs),
                    "low": len(low_reqs) if low_reqs else 0,
                },
                "config": {
                    "dense_model": self.config.dense_model,
                    "cross_encoder_model": self.config.cross_encoder_model,
                    "bm25_top_k": self.config.bm25_top_k,
                    "dense_top_k": self.config.dense_top_k,
                    "dense_threshold": self.config.dense_threshold,
                    "final_threshold": self.config.final_threshold,
                    "fusion_weights": {
                        "alpha_bm25": self.config.alpha,
                        "beta_dense": self.config.beta,
                        "gamma_cross": self.config.gamma,
                    },
                },
            },
        }

    def _build_reverse(
        self,
        forward: Dict[str, Any],
        levels: List[str],
    ) -> Dict[str, List[dict]]:
        """
        Invert the forward link graph for bidirectional traceability.

        Result:
          hlr_id → [{"source_id": slr_id, "confidence": float}]
          llr_id → [{"source_id": hlr_id, "confidence": float}]
        """
        reverse: Dict[str, List[dict]] = {}

        for src_id, src_data in forward.items():
            # SLR→HLR links
            for hlr_link in src_data.get("linked_hlr", []):
                tgt = hlr_link["hlr_id"]
                reverse.setdefault(tgt, []).append(
                    {"source_id": src_id, "confidence": hlr_link["confidence"]}
                )
                # HLR→LLR links (nested inside the SLR→HLR entry)
                for llr_link in hlr_link.get("linked_llr", []):
                    tgt2 = llr_link["llr_id"]
                    reverse.setdefault(tgt2, []).append(
                        {"source_id": tgt, "confidence": llr_link["confidence"]}
                    )

            # HLR→LLR links (2-level HLR/LLR mode)
            for llr_link in src_data.get("linked_llr", []):
                tgt = llr_link["llr_id"]
                reverse.setdefault(tgt, []).append(
                    {"source_id": src_id, "confidence": llr_link["confidence"]}
                )

        # Sort each target's back-links by confidence descending
        for tgt in reverse:
            reverse[tgt].sort(key=lambda x: x["confidence"], reverse=True)

        return reverse
