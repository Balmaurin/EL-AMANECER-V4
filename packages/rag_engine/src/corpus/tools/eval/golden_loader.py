import json
from pathlib import Path
from typing import Dict, Iterable


def load_golden(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Expect {"question": str, "gold": [str]}
            if isinstance(obj.get("gold"), list) and obj.get("question"):
                yield obj
