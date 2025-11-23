import logging
from pathlib import Path

from tools.common.paths import LOGS_DIR


def setup_logging(
    level: str = "INFO",
    file: str = None,
    fmt: str = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
):
    """Setup logging with absolute paths."""
    if file is None:
        file = LOGS_DIR / "rag.log"
    else:
        file = Path(file)

    # Ensure directory exists
    file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[logging.StreamHandler(), logging.FileHandler(file, encoding="utf-8")],
    )
    logging.getLogger("whoosh").setLevel(logging.WARNING)
    return logging.getLogger("rag")
