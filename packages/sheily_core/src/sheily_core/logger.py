"""
Simple logger wrapper for sheily_core
"""
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class LogContext:
    """Context for structured logging"""
    component: str
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextLogger(logging.Logger):
    """Logger with context support"""
    @contextmanager
    def context(self, **kwargs):
        """Context manager for logging"""
        # Just yield, we don't need complex context tracking for now
        yield

def get_logger(name: str) -> ContextLogger:
    """Get a logger with the given name"""
    # Get the logger and change its class
    logger = logging.getLogger(name)
    logger.__class__ = ContextLogger
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
