"""TraceX CLI — bidirectional requirements traceability.

Usage examples
--------------
# 3-level (SLR → HLR → LLR):
  python main.py -s data/system-requirements.csv \\
                 -h data/high-requirements.csv \\
                 -l data/low-requirements.csv \\
                 -o traceability.json

# 2-level SLR→HLR:
  python main.py -s data/system-requirements.csv \\
                 -h data/high-requirements.csv

# 2-level HLR→LLR:
  python main.py -h data/high-requirements.csv \\
                 -l data/low-requirements.csv

# GPU inference, higher quality cross-encoder:
  python main.py -s data/system-requirements.csv \\
                 -h data/high-requirements.csv \\
                 -l data/low-requirements.csv \\
                 --device cuda \\
                 --cross-encoder cross-encoder/ms-marco-electra-base
"""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from tracex.config import Config
from tracex.engine import TraceabilityEngine
from tracex.loader import load_requirements

console = Console()


@click.command(context_settings={"show_default": True})
@click.option(
    "--system-reqs", "-s",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="System-level requirements CSV (id, description)",
)
@click.option(
    "--high-reqs", "-h",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="High-level requirements CSV (id, description)",
)
@click.option(
    "--low-reqs", "-l",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Low-level requirements CSV (id, description)",
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False),
    default="traceability.json",
    help="Output JSON file path",
)
# ── BM25 ──────────────────────────────────────────────────────────────────
@click.option("--bm25-top-k", default=50, help="BM25 recall candidates per query")
@click.option("--bm25-k1", default=1.7, help="BM25 k1 (term saturation)")
@click.option("--bm25-b", default=0.7, help="BM25 b (length normalisation)")
# ── Dense ─────────────────────────────────────────────────────────────────
@click.option(
    "--dense-model",
    default="BAAI/bge-large-en-v1.5",
    help="Sentence-transformer model for dense embeddings",
)
@click.option("--dense-top-k", default=15, help="Max dense candidates after reranking")
@click.option("--dense-threshold", default=0.65, help="Min cosine similarity to keep")
@click.option("--faiss-m", default=64, help="FAISS HNSW M parameter")
@click.option("--faiss-ef-search", default=256, help="FAISS HNSW efSearch")
# ── Cross-encoder ─────────────────────────────────────────────────────────
@click.option(
    "--cross-encoder",
    default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    help="Cross-encoder model for precision scoring",
)
# ── Fusion & threshold ────────────────────────────────────────────────────
@click.option("--alpha", default=0.2, help="BM25 fusion weight")
@click.option("--beta", default=0.3, help="Dense fusion weight")
@click.option("--gamma", default=0.5, help="Cross-encoder fusion weight")
@click.option("--final-threshold", default=0.4, help="Min fused score to emit a link")
@click.option("--top-n", default=10, help="Max links per source requirement")
# ── Hardware ──────────────────────────────────────────────────────────────
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    help="Inference device",
)
@click.option("--batch-size", default=32, help="Embedding batch size")
def main(
    system_reqs, high_reqs, low_reqs, output,
    bm25_top_k, bm25_k1, bm25_b,
    dense_model, dense_top_k, dense_threshold, faiss_m, faiss_ef_search,
    cross_encoder,
    alpha, beta, gamma, final_threshold, top_n,
    device, batch_size,
):
    """TraceX — requirements traceability via BM25 + Dense + Cross-Encoder."""
    if system_reqs is None and low_reqs is None:
        raise click.UsageError(
            "Provide at least two requirement levels: "
            "(-s and -h) or (-h and -l) or all three."
        )

    console.print(
        Panel.fit(
            "[bold green]TraceX — Requirements Traceability[/bold green]\n"
            f"Dense model   : [yellow]{dense_model}[/yellow]\n"
            f"Cross-encoder : [yellow]{cross_encoder}[/yellow]\n"
            f"Device        : [yellow]{device}[/yellow]",
            border_style="green",
        )
    )

    config = Config(
        bm25_top_k=bm25_top_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        dense_model=dense_model,
        dense_top_k=dense_top_k,
        dense_threshold=dense_threshold,
        faiss_m=faiss_m,
        faiss_ef_search=faiss_ef_search,
        cross_encoder_model=cross_encoder,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        final_threshold=final_threshold,
        top_n=top_n,
        device=device,
        batch_size=batch_size,
    )

    # Load CSVs
    console.print("\n[bold]Loading requirements...[/bold]")
    slr = load_requirements(system_reqs) if system_reqs else None
    hlr = load_requirements(high_reqs)
    llr = load_requirements(low_reqs) if low_reqs else None

    counts = []
    if slr:
        counts.append(f"[cyan]{len(slr)}[/cyan] SLR")
    counts.append(f"[cyan]{len(hlr)}[/cyan] HLR")
    if llr:
        counts.append(f"[cyan]{len(llr)}[/cyan] LLR")
    console.print("  " + "  •  ".join(counts))

    # Run pipeline
    engine = TraceabilityEngine(config)
    result = engine.run(slr, hlr, llr)

    # Write output
    out_path = Path(output)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    n_links = sum(
        len(v.get("linked_hlr", v.get("linked_llr", [])))
        for v in result["forward_links"].values()
    )
    console.print(
        f"\n[bold green]Done![/bold green]  "
        f"{n_links} forward links written to [underline]{out_path}[/underline]"
    )


if __name__ == "__main__":
    main()
