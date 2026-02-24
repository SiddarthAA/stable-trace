# TraceX

Bidirectional requirements traceability via hybrid NLP — BM25 lexical recall, dense semantic embeddings, and cross-encoder reranking fused into a single confidence score.

Supports two or three levels of requirements (System → High → Low) and outputs a nested JSON with forward links and auto-generated reverse links.

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Installation](#installation)
3. [Input format](#input-format)
4. [Running TraceX](#running-tracex)
5. [All CLI options](#all-cli-options)
6. [Output format](#output-format)
7. [Tuning guide](#tuning-guide)
8. [Model selection](#model-selection)
9. [Using the Python API directly](#using-the-python-api-directly)

---

## How it works

For each source requirement (SLR or HLR), TraceX runs a four-step pipeline against the target level (HLR or LLR):

```
Source requirement
       │
       ▼
① BM25 lexical recall         top-50 candidates  (fast keyword match)
       │
       ▼
② Dense cosine reranking      top-15 candidates  (semantic similarity via BGE)
       │
       ▼
③ Cross-encoder scoring       probability per pair  (deep attention over both texts)
       │
       ▼
④ Score fusion + threshold    Final = α·BM25 + β·Dense + γ·Cross
       │
       ▼
    Ranked links
```

Each score is min-max normalised across the candidate set before fusion, so all three signals contribute on the same [0, 1] scale.

---

## Installation

Requires Python 3.11+. Uses `uv` (recommended) or any standard pip workflow.

```bash
# Clone / enter the project
cd tracex

# Install all dependencies
uv pip install -e .

# Verify
uv run python main.py --help
```

> **First run note:** The default dense model (`BAAI/bge-large-en-v1.5`, ~1.3 GB) and
> cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90 MB) are downloaded from
> HuggingFace on first use and cached locally. Subsequent runs are instant.

---

## Input format

Each requirements level is a CSV file with exactly two columns:

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique identifier, e.g. `SLR-001`, `HLR-3.2` |
| `description` | string | Full requirement text |

Column names are case-insensitive and leading/trailing whitespace is stripped.

**Example — `data/system-requirements.csv`:**

```csv
id,description
SLR-001,The system shall continuously monitor vehicle speed during all operating modes
SLR-002,The system shall apply emergency braking automatically when a forward collision is imminent
SLR-003,The system shall issue a lane departure warning when the vehicle unintentionally crosses lane markings
```

**Example — `data/high-requirements.csv`:**

```csv
id,description
HLR-001,The speed sensing subsystem shall measure vehicle velocity with an accuracy of ±0.5 km/h
HLR-002,The brake control unit shall apply brake force within 150ms of receiving an emergency braking signal
HLR-003,The camera subsystem shall process road lane markings at a frame rate of at least 30 fps
```

> IDs embedded in the description text are automatically stripped during normalisation,
> so writing `"SLR-001 The system shall..."` and `"The system shall..."` are equivalent.

---

## Running TraceX

### Three-level traceability (SLR → HLR → LLR)

The most common use case. Traces from system level all the way down to low level.

```bash
uv run python main.py \
  -s data/system-requirements.csv \
  -h data/high-requirements.csv \
  -l data/low-requirements.csv \
  -o traceability.json
```

### Two-level: System → High only

```bash
uv run python main.py \
  -s data/system-requirements.csv \
  -h data/high-requirements.csv \
  -o slr_to_hlr.json
```

### Two-level: High → Low only

```bash
uv run python main.py \
  -h data/high-requirements.csv \
  -l data/low-requirements.csv \
  -o hlr_to_llr.json
```

### Quick test with a small model (no large download)

Useful for validating your CSVs and pipeline logic before the full production run.

```bash
uv run python main.py \
  -s data/system-requirements.csv \
  -h data/high-requirements.csv \
  -l data/low-requirements.csv \
  --dense-model all-MiniLM-L6-v2 \
  --dense-threshold 0.0 \
  --final-threshold 0.3
```

### GPU inference (faster for large requirement sets)

```bash
uv run python main.py \
  -s data/system-requirements.csv \
  -h data/high-requirements.csv \
  -l data/low-requirements.csv \
  --device cuda \
  --cross-encoder cross-encoder/ms-marco-electra-base \
  --batch-size 64
```

---

## All CLI options

```
-s, --system-reqs FILE      System-level requirements CSV           [optional]
-h, --high-reqs FILE        High-level requirements CSV             [required]
-l, --low-reqs FILE         Low-level requirements CSV              [optional]
-o, --output FILE           Output JSON path        [default: traceability.json]

BM25
  --bm25-top-k INTEGER      Candidates retrieved by BM25            [default: 50]
  --bm25-k1 FLOAT           Term saturation parameter               [default: 1.7]
  --bm25-b  FLOAT           Length normalisation parameter          [default: 0.7]

Dense embedding
  --dense-model TEXT        HuggingFace model name                  [default: BAAI/bge-large-en-v1.5]
  --dense-top-k INTEGER     Max candidates after cosine rerank      [default: 15]
  --dense-threshold FLOAT   Min cosine similarity to keep           [default: 0.65]
  --faiss-m INTEGER         HNSW M (graph degree)                   [default: 64]
  --faiss-ef-search INTEGER HNSW efSearch (query-time beam width)   [default: 256]

Cross-encoder
  --cross-encoder TEXT      HuggingFace model name                  [default: cross-encoder/ms-marco-MiniLM-L-6-v2]

Score fusion
  --alpha FLOAT             BM25 weight in fused score              [default: 0.2]
  --beta  FLOAT             Dense weight in fused score             [default: 0.3]
  --gamma FLOAT             Cross-encoder weight in fused score     [default: 0.5]
  --final-threshold FLOAT   Min fused score to emit a link          [default: 0.4]
  --top-n INTEGER           Max links returned per requirement      [default: 10]

Hardware
  --device [cpu|cuda]       Inference device                        [default: cpu]
  --batch-size INTEGER      Embedding batch size                    [default: 32]
```

---

## Output format

The output is a single JSON file with three top-level keys.

### `forward_links`

The primary traceability graph. For a 3-level run:

```json
{
  "forward_links": {
    "SLR-001": {
      "description": "The system shall continuously monitor vehicle speed...",
      "linked_hlr": [
        {
          "hlr_id": "HLR-001",
          "description": "The speed sensing subsystem shall measure vehicle velocity...",
          "confidence": 1.0,
          "scores": {
            "bm25":      4.006,
            "bm25_norm": 1.0,
            "dense":     0.812,
            "cross":     0.934,
            "final":     1.0
          },
          "linked_llr": [
            {
              "llr_id": "LLR-001",
              "description": "The wheel speed encoder shall output 64 pulses per revolution...",
              "confidence": 0.926,
              "scores": { "bm25": 3.1, "bm25_norm": 0.9, "dense": 0.74, "cross": 0.89, "final": 0.926 }
            },
            {
              "llr_id": "LLR-015",
              "description": "The velocity calculator shall integrate wheel encoder pulses...",
              "confidence": 0.386,
              "scores": { ... }
            }
          ]
        }
      ]
    },
    "SLR-002": { ... }
  }
}
```

**Score fields explained:**

| Field | Range | Meaning |
|---|---|---|
| `bm25` | 0 → ∞ | Raw BM25 score (keyword overlap) |
| `bm25_norm` | 0 – 1 | BM25 min-max normalised across candidates |
| `dense` | −1 – 1 | Cosine similarity from BGE embeddings |
| `cross` | 0 – 1 | Cross-encoder relevance probability (sigmoid of logit) |
| `final` | 0 – 1 | Fused score: α·bm25_norm + β·dense + γ·cross |
| `confidence` | 0 – 1 | Same as `final`, top-level shorthand |

### `reverse_links`

Auto-generated inverse index for tracing upward through the hierarchy.

```json
{
  "reverse_links": {
    "HLR-001": [
      { "source_id": "SLR-001", "confidence": 1.0 }
    ],
    "LLR-001": [
      { "source_id": "HLR-001", "confidence": 0.926 }
    ]
  }
}
```

Use this to answer: *"Which SLR does HLR-001 implement?"* or *"Which HLR does LLR-007 derive from?"*

### `_meta`

Run configuration baked into the output for reproducibility.

```json
{
  "_meta": {
    "levels": ["system", "high", "low"],
    "counts": { "system": 7, "high": 12, "low": 18 },
    "config": {
      "dense_model": "BAAI/bge-large-en-v1.5",
      "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "bm25_top_k": 50,
      "dense_top_k": 15,
      "dense_threshold": 0.65,
      "final_threshold": 0.4,
      "fusion_weights": { "alpha_bm25": 0.2, "beta_dense": 0.3, "gamma_cross": 0.5 }
    }
  }
}
```

---

## Tuning guide

### Too many false positives (noise links)

Raise the thresholds to be stricter:

```bash
--dense-threshold 0.75   # was 0.65 — drop semantically weak candidates earlier
--final-threshold 0.55   # was 0.40 — only keep high-confidence links
--top-n 5                # cap at 5 links per requirement
```

### Missing known links (too strict / low recall)

Lower the thresholds and increase candidate pool:

```bash
--dense-threshold 0.50   # allow looser semantic matches through
--final-threshold 0.30   # keep more marginal links
--bm25-top-k 100         # retrieve more BM25 candidates to start with
--dense-top-k 25         # pass more to the cross-encoder
```

### Boosting the cross-encoder signal

If you trust the cross-encoder more than BM25 (typical for domain-specific text):

```bash
--alpha 0.1 --beta 0.2 --gamma 0.7
```

### Boosting lexical matching

Useful when requirements use tightly controlled terminology:

```bash
--alpha 0.4 --beta 0.3 --gamma 0.3
```

> **Rule:** alpha + beta + gamma should sum to 1.0 for the final score to stay in [0, 1].

---

## Model selection

| Component | Default | Faster / Lighter | Higher Accuracy |
|---|---|---|---|
| Dense embeddings | `BAAI/bge-large-en-v1.5` | `all-MiniLM-L6-v2` | same (already top-tier) |
| Cross-encoder | `ms-marco-MiniLM-L-6-v2` | `ms-marco-TinyBERT-L-2-v2` | `ms-marco-electra-base` |

**Switching models via CLI:**

```bash
# Lighter pipeline (fast iteration / CI)
uv run python main.py -h high.csv -l low.csv \
  --dense-model all-MiniLM-L6-v2 \
  --cross-encoder cross-encoder/ms-marco-TinyBERT-L-2-v2

# Maximum accuracy pipeline
uv run python main.py -s slr.csv -h hlr.csv -l llr.csv \
  --dense-model BAAI/bge-large-en-v1.5 \
  --cross-encoder cross-encoder/ms-marco-electra-base \
  --device cuda
```

> For domain fine-tuning: the cross-encoder model path accepts any local directory
> (`--cross-encoder ./my_finetuned_deberta`) alongside HuggingFace model IDs.

---

## Using the Python API directly

You can drive the engine from Python without the CLI, which is useful for notebooks,
batch jobs, or integrating into a larger pipeline.

```python
import json
from tracex.config import Config
from tracex.engine import TraceabilityEngine
from tracex.loader import load_requirements

# Configure
config = Config(
    dense_model="BAAI/bge-large-en-v1.5",
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    final_threshold=0.4,
    top_n=10,
    device="cpu",
)

# Load your CSVs
slr = load_requirements("path/to/system-requirements.csv")
hlr = load_requirements("path/to/high-requirements.csv")
llr = load_requirements("path/to/low-requirements.csv")  # optional

# Run
engine = TraceabilityEngine(config)
result = engine.run(
    system_reqs=slr,   # pass None to skip this level
    high_reqs=hlr,
    low_reqs=llr,      # pass None to skip this level
)

# result["forward_links"]  — nested SLR→HLR→LLR graph
# result["reverse_links"]  — inverted index (HLR→SLR, LLR→HLR)
# result["_meta"]          — run config snapshot

# Inspect a single SLR's links
for slr_id, data in result["forward_links"].items():
    print(f"\n{slr_id}: {data['description']}")
    for link in data["linked_hlr"]:
        print(f"  → {link['hlr_id']} (confidence={link['confidence']:.3f})")
        for llr_link in link.get("linked_llr", []):
            print(f"      → {llr_link['llr_id']} (confidence={llr_link['confidence']:.3f})")

# Save
with open("traceability.json", "w") as f:
    json.dump(result, f, indent=2)
```

### Accessing the Requirement object

`load_requirements` returns `List[Requirement]`. Each has:

```python
req.id            # str  — e.g. "SLR-001"
req.description   # str  — original text
req.cleaned_text  # str  — normalised text (set after engine.run() is called)
```

---

## Project layout

```
tracex/
├── main.py                  # CLI entry point
├── pyproject.toml           # project metadata + dependencies
├── data/                    # sample requirement CSVs
│   ├── system-requirements.csv
│   ├── high-requirements.csv
│   └── low-requirements.csv
└── tracex/                  # core package
    ├── config.py            # Config dataclass — all tunable parameters
    ├── loader.py            # CSV → List[Requirement]
    ├── normalizer.py        # text cleaning, synonym mapping
    ├── bm25_index.py        # BM25 lexical index (rank-bm25 + NLTK stopwords)
    ├── dense_index.py       # FAISS HNSW vector store + BGE encoder
    ├── reranker.py          # cross-encoder (sentence-transformers CrossEncoder)
    ├── fusion.py            # min-max normalisation + weighted score fusion
    └── engine.py            # pipeline orchestrator → JSON output builder
```
