from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class Requirement:
    id: str
    description: str
    cleaned_text: str = field(default="", repr=False)


def load_requirements(path: str | Path) -> List[Requirement]:
    """Load requirements from a CSV file with 'id' and 'description' columns."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = {"id", "description"} - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV at '{path}' is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    reqs = []
    for _, row in df.iterrows():
        req_id = str(row["id"]).strip()
        desc = str(row["description"]).strip()
        if req_id and desc and desc.lower() != "nan":
            reqs.append(Requirement(id=req_id, description=desc))

    if not reqs:
        raise ValueError(f"No valid requirements found in '{path}'")

    return reqs
