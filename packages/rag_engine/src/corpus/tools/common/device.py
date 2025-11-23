import logging
import os

logger = logging.getLogger(__name__)


def pick_device(preferred: str = "auto") -> str:
    # Return best-guess device for SentenceTransformers
    # - "cuda" if torch+CUDA available
    # - "privateuseone:0" if torch-directml exists (Windows AMD/Intel)
    # - "mps" on Apple
    # - "cpu" otherwise
    if preferred and preferred.lower() != "auto":
        return preferred
    try:
        import torch  # type: ignore

        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            return "cuda"
        try:
            import torch_directml  # type: ignore # noqa: F401

            return "privateuseone:0"
        except Exception as e:
            logger.debug(f"torch-directml not available: {e}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception as e:
        logger.debug(f"torch not available: {e}")
    return "cpu"
