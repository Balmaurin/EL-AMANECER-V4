"""
Multi-GPU distributed search support using PyTorch native and Ray.
Handles embedding and search across multiple GPUs transparently.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.nn.parallel import DataParallel, DistributedDataParallel

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class MultiGPUSearcher:
    """Multi-GPU distributed searcher using PyTorch DataParallel."""

    def __init__(self, model, device_ids: Optional[List[int]] = None):
        """
        Initialize multi-GPU searcher.

        Args:
            model: SentenceTransformer or embedding model
            device_ids: List of GPU device IDs to use. If None, uses all available.
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to CPU")
            self.use_gpu = False
            self.model = model
            return

        # Detect available GPUs
        if not torch.cuda.is_available():
            logger.warning("No GPUs detected, using CPU")
            self.use_gpu = False
            self.model = model
            return

        num_gpus = torch.cuda.device_count()
        logger.info(f"🎮 Detected {num_gpus} GPU(s)")

        if device_ids is None:
            device_ids = list(range(num_gpus))

        device_ids = [d for d in device_ids if d < num_gpus]

        if len(device_ids) > 1:
            logger.info(f"📦 Using DataParallel with devices: {device_ids}")
            # Move model to first device
            model = model.to(f"cuda:{device_ids[0]}")
            # Wrap with DataParallel
            self.model = DataParallel(model, device_ids=device_ids)
            self.use_gpu = True
            self.primary_device = device_ids[0]
        elif len(device_ids) == 1:
            logger.info(f"🎮 Using single GPU: {device_ids[0]}")
            self.model = model.to(f"cuda:{device_ids[0]}")
            self.use_gpu = True
            self.primary_device = device_ids[0]
        else:
            logger.warning("No valid GPUs found, using CPU")
            self.use_gpu = False
            self.model = model

        self.device_ids = device_ids

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> Any:
        """
        Encode texts efficiently across GPUs.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            Embeddings array
        """
        if hasattr(self.model, "module"):  # DataParallel wrapped
            embeddings = self.model.module.encode(
                texts, batch_size=batch_size, convert_to_numpy=True
            )
        else:
            embeddings = self.model.encode(
                texts, batch_size=batch_size, convert_to_numpy=True
            )

        logger.debug(f"[+] Encoded {len(texts)} texts on {len(self.device_ids)} GPU(s)")
        return embeddings

    def encode_single(self, text: str) -> Any:
        """Encode single text."""
        return self.encode_batch([text])[0]

    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        info = {
            "use_gpu": self.use_gpu,
            "num_gpus": len(self.device_ids),
            "device_ids": self.device_ids,
        }

        if self.use_gpu and TORCH_AVAILABLE:
            for i, device_id in enumerate(self.device_ids):
                try:
                    props = torch.cuda.get_device_properties(device_id)
                    info[f"gpu_{i}_name"] = props.name
                    info[f"gpu_{i}_memory_total"] = f"{props.total_memory / 1e9:.1f}GB"
                except Exception as e:
                    logger.warning(f"Failed to get GPU {device_id} info: {e}")

        return info


class RayDistributedSearcher:
    """Distributed searcher using Ray for large-scale deployments."""

    def __init__(self, model, num_workers: Optional[int] = None):
        """
        Initialize Ray-based distributed searcher.

        Args:
            model: SentenceTransformer model
            num_workers: Number of Ray workers. If None, uses num_gpus.
        """
        if not RAY_AVAILABLE:
            logger.warning("Ray not installed, use: pip install ray")
            self.use_ray = False
            self.model = model
            return

        if not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count())
                logger.info("[+] Ray initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Ray: {e}")
                self.use_ray = False
                self.model = model
                return

        self.use_ray = True

        # Determine num_workers
        if num_workers is None:
            num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
            num_workers = max(1, num_workers)

        logger.info(f"[>>] Creating {num_workers} Ray workers for distributed search")

        # Create remote model actors
        self.workers = [
            SearchWorker.remote(
                model,
                device_id=(
                    i % torch.cuda.device_count() if torch.cuda.is_available() else -1
                ),
            )
            for i in range(num_workers)
        ]
        self.worker_index = 0

    def encode_batch_distributed(self, texts: List[str]) -> Any:
        """
        Distribute encoding across Ray workers.

        Args:
            texts: List of texts to encode

        Returns:
            Embeddings array
        """
        if not self.use_ray:
            return self.model.encode(texts, convert_to_numpy=True)

        import numpy as np

        # Split texts among workers
        chunk_size = (len(texts) + len(self.workers) - 1) // len(self.workers)
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

        # Distribute work
        futures = [
            worker.encode.remote(chunk) for worker, chunk in zip(self.workers, chunks)
        ]

        # Gather results
        results = ray.get(futures)
        embeddings = np.vstack(results)

        logger.debug(f"[+] Distributed encoding across {len(self.workers)} workers")
        return embeddings

    def shutdown(self):
        """Shutdown Ray."""
        if self.use_ray:
            try:
                ray.shutdown()
                logger.info("[+] Ray shutdown")
            except Exception as e:
                logger.warning(f"Failed to shutdown Ray: {e}")


@ray.remote
class SearchWorker:
    """Remote worker for distributed search."""

    def __init__(self, model, device_id: int = -1):
        """Initialize worker with model on specific device."""
        self.device_id = (
            device_id if device_id >= 0 and torch.cuda.is_available() else -1
        )

        if self.device_id >= 0:
            self.device = f"cuda:{self.device_id}"
            self.model = model.to(self.device)
            logger.info(f"[+] Worker initialized on GPU {self.device_id}")
        else:
            self.device = "cpu"
            self.model = model.to("cpu")
            logger.info("[+] Worker initialized on CPU")

    def encode(self, texts: List[str]) -> Any:
        """Encode texts on this worker's device."""
        return self.model.encode(texts, convert_to_numpy=True)


def get_optimal_searcher(model, distributed: bool = False) -> Any:
    """
    Get optimal searcher based on hardware availability.

    Args:
        model: Embedding model
        distributed: Whether to use Ray distributed (requires Ray)

    Returns:
        Searcher instance
    """
    num_gpus = torch.cuda.device_count() if TORCH_AVAILABLE else 0

    if distributed and RAY_AVAILABLE and num_gpus > 0:
        logger.info("[>>] Using Ray distributed searcher")
        return RayDistributedSearcher(model, num_workers=num_gpus)
    elif num_gpus > 1:
        logger.info("📦 Using PyTorch DataParallel searcher")
        return MultiGPUSearcher(model, device_ids=list(range(num_gpus)))
    elif num_gpus == 1:
        logger.info("🎮 Using single GPU")
        return MultiGPUSearcher(model, device_ids=[0])
    else:
        logger.info("💻 Using CPU")
        return MultiGPUSearcher(model, device_ids=[])
