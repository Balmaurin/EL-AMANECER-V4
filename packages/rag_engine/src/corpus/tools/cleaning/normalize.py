"""
Text normalization and cleaning module for RAG document processing.

This module handles text preprocessing, including:
- PII (Personally Identifiable Information) removal
- HTML cleaning
- Text normalization
- Language detection
- Quality assessment
- Deduplication
"""

import hashlib
import json
import logging
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ftfy
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH
from langdetect import DetectorFactory, detect

from .quality import quality_score, shingle_words

# Ensure deterministic language detection
DetectorFactory.seed = 0

# Configure logging
log = logging.getLogger("rag.normalize")

# Regular expressions for PII detection and replacement
PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Email addresses
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    # Phone numbers (international format support)
    (
        re.compile(
            r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,3}\)?[-.\s]?)?\d{3}[-.\s]?\d{3,4}\b"
        ),
        "[PHONE]",
    ),
    # IBAN numbers
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(\d{0,16})?\b"), "[IBAN]"),
    # URLs
    (re.compile(r"https?://\S+"), "[URL]"),
]


def _clean_html(text: str) -> str:
    """Clean HTML from text, extracting meaningful content.

    Args:
        text: Input text that may contain HTML

    Returns:
        Cleaned text with HTML removed

    Note:
        Uses BeautifulSoup for robust HTML parsing when HTML tags are detected.
    """
    if not isinstance(text, str):
        return ""

    if "<html" in text.lower() or "<body" in text.lower():
        try:
            soup = BeautifulSoup(text, "lxml")
            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()
            return " ".join(soup.get_text(separator=" ").split())
        except Exception as e:
            log.warning(f"HTML cleaning failed: {e}")

    return text


def _hash(text: str) -> str:
    """Generate SHA-256 hash of input text.

    Args:
        text: Input text to hash

    Returns:
        Hexadecimal string of SHA-256 hash
    """
    if not isinstance(text, str):
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _process_file(
    file_path: Path,
    cleaned_dir: Path,
    min_len: int,
    min_quality: float,
    lsh: Optional[MinHashLSH] = None,
) -> bool:
    """Process a single file through the cleaning pipeline.

    Args:
        file_path: Path to input file
        cleaned_dir: Output directory for cleaned files
        min_len: Minimum text length to process
        min_quality: Minimum quality score to accept
        lsh: Optional LSH index for deduplication

    Returns:
        bool: True if file was processed successfully, False otherwise
    """
    try:
        # Read and clean raw text
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        raw = _clean_html(raw)

        # Fix text encoding and normalize
        text = ftfy.fix_text(raw)
        text = unicodedata.normalize("NFKC", text).strip()

        # Check minimum length
        if len(text) < min_len:
            log.debug(f"Text too short: {file_path}")
            return False

        # Remove PII
        for pattern, replacement in PII_PATTERNS:
            text = pattern.sub(replacement, text)

        # Detect language
        try:
            lang = detect(text)
        except Exception as e:
            log.warning(f"Language detection failed for {file_path}: {e}")
            lang = "unk"

        # Calculate quality score
        quality = quality_score(text)

        # Check quality threshold
        if quality < min_quality:
            log.debug(f"Quality score too low: {file_path} ({quality:.2f})")
            return False

        # Check for duplicates if LSH index provided
        if lsh is not None:
            # Generate MinHash
            shingles = shingle_words(text, k=5)
            minhash = MinHash(num_perm=128)
            for s in shingles:
                minhash.update(s.encode("utf-8"))

            # Check for duplicates
            if list(lsh.query(minhash)):
                log.debug(f"Duplicate content detected: {file_path}")
                return False

            # Add to LSH index
            lsh.insert(_hash(text), minhash)

        # Generate document ID and metadata
        doc_id = _hash(str(file_path))
        cleaned_doc = {
            "doc_id": doc_id,
            "text": text,
            "lang": lang,
            "quality": quality,
            "title": file_path.stem,
            "tags": [],
        }

        # Write cleaned document
        output_path = cleaned_dir / f"{file_path.stem}.jsonl"
        output_path.write_text(
            json.dumps(cleaned_doc, ensure_ascii=False), encoding="utf-8"
        )

        return True

    except Exception as e:
        log.error(f"Error processing file {file_path}: {e}")
        return False
    if q < min_quality:
        return False
    if lsh is not None:
        shingles = shingle_words(txt, k=5)
        if shingles:
            mh = MinHash(num_perm=128)
            for s in shingles:
                mh.update(s.encode("utf-8"))
            if lsh.query(mh):
                return False
            lsh.insert(_hash(txt), mh)
    doc_id = _hash(str(fp))
    out = {
        "doc_id": doc_id,
        "text": txt,
        "lang": lang,
        "quality": q,
        "title": fp.stem,
        "tags": [],
    }
    (cleaned_dir / f"{fp.stem}.jsonl").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8"
    )
    return True


def normalize_corpus(base: Path, min_len: int = 200):
    import yaml

    cfg = yaml.safe_load(open("config/universal.yaml", "r", encoding="utf-8"))
    ent = cfg.get("enterprise", {})
    min_len = int(ent.get("normalize", {}).get("min_chars", min_len))
    min_quality = float(ent.get("normalize", {}).get("min_quality", 0.5))
    dedup = ent.get("normalize", {}).get("dedup", {"enabled": False})
    jacc = float(dedup.get("jaccard_threshold", 0.9))
    n_perm = int(dedup.get("n_perm", 128))

    raw = base / "raw"
    cleaned = base / "cleaned"
    cleaned.mkdir(parents=True, exist_ok=True)

    lsh = (
        MinHashLSH(threshold=jacc, num_perm=n_perm)
        if dedup.get("enabled", False)
        else None
    )

    n_in, n_out = 0, 0
    for f in raw.glob("*"):
        if not f.is_file():
            continue
        n_in += 1
        try:
            if _process_file(f, cleaned, min_len, min_quality, lsh=lsh):
                n_out += 1
        except Exception as e:
            logging.debug(f"Failed processing file {f}: {e}")
            continue
    return {"docs_in": n_in, "docs_out": n_out, "min_quality": min_quality}
