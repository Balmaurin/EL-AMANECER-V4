#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sheily_pdf_extractor.py
=======================
Simple PDF/Text extractor for Sheily memory system
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add backend to path for security module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from apps.backend.src.security import SecurityError, sanitize_path

logger = logging.getLogger(__name__)


def extract_chunks_with_meta(
    file_path: Path, base_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Extract text chunks from PDF with metadata

    Args:
        file_path: Path to file (can be user input)
        base_dir: Base directory for security validation (optional)
    """
    try:
        # SECURITY: Validate file path if base_dir provided
        if base_dir:
            try:
                safe_path = sanitize_path(
                    str(file_path),
                    base_dir,
                    allowed_extensions=[".txt", ".pdf"],
                    max_depth=15,
                )
            except SecurityError as se:
                logger.error(f"Invalid file path: {se}")
                return []
        else:
            safe_path = Path(file_path).resolve()

        # Simple fallback - just read as text if it's actually a text file
        if safe_path.suffix.lower() == ".txt":
            text = safe_path.read_text(encoding="utf-8")
            return [
                {
                    "text": text,
                    "meta": {"source": str(safe_path), "page": 1, "chunk_id": 0},
                }
            ]

        # For PDF files, return empty for now
        logger.warning(f"PDF extraction not implemented for {safe_path}")
        return []

    except Exception as e:
        logger.error(f"Error extracting from {file_path}: {e}")
        return []
