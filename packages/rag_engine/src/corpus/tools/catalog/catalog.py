import json
import logging
import time
from pathlib import Path

import pandas as pd

from tools.common.paths import MANIFESTS_PATH

log = logging.getLogger("rag.catalog")

# Absolute paths for catalog files (using parquet for manifest)
MAN_PATH = MANIFESTS_PATH


def update_catalog(branch: str, base: Path):
    """Update catalog via manifest (parquet-based)."""
    # Catalog is now managed through manifest parquet file
    pass


def update_manifest(branch: str, base: Path, stats: dict):
    """Update manifest file (uses pandas/parquet, no external DB needed)."""
    MAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "branch": branch,
        "snapshot": base.name,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    } | (stats or {})
    if MAN_PATH.exists():
        old = pd.read_parquet(str(MAN_PATH))
        pd.concat([old, pd.DataFrame([rec])], ignore_index=True).to_parquet(
            str(MAN_PATH), index=False
        )
    else:
        pd.DataFrame([rec]).to_parquet(str(MAN_PATH), index=False)
