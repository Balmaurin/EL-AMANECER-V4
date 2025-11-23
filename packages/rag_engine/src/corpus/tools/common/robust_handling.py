"""
Robust edge case handling for file operations, indexing, and searches.
Handles corrupted files, missing directories, permission errors, etc.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Raised when file validation fails."""

    pass


class RobustFileHandler:
    """Handles files robustly with error recovery."""

    @staticmethod
    def ensure_directory(path: Path, create: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Ensure directory exists and is writable.

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        try:
            if not path.exists():
                if not create:
                    return False, f"Directory does not exist: {path}"
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"[+] Created directory: {path}")

            # Test writability
            test_file = path / ".test_write"
            try:
                test_file.write_text("test")
                test_file.unlink()
                logger.debug(f"[+] Directory is writable: {path}")
                return True, None
            except (PermissionError, OSError) as e:
                return False, f"Directory is not writable: {path} ({e})"

        except Exception as e:
            return False, f"Failed to ensure directory {path}: {e}"

    @staticmethod
    def validate_file(
        file_path: Path, min_size: int = 0, max_size: int = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate file exists and meets size requirements.

        Returns:
            Tuple of (valid: bool, error_message: Optional[str])
        """
        try:
            if not file_path.exists():
                return False, f"File does not exist: {file_path}"

            file_size = file_path.stat().st_size

            if file_size < min_size:
                return (
                    False,
                    f"File too small: {file_path} ({file_size} < {min_size} bytes)",
                )

            if max_size and file_size > max_size:
                return (
                    False,
                    f"File too large: {file_path} ({file_size} > {max_size} bytes)",
                )

            # Test readability
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.read(100)
            except (PermissionError, OSError) as e:
                return False, f"File is not readable: {file_path} ({e})"

            logger.debug(f"[+] File valid: {file_path} ({file_size} bytes)")
            return True, None

        except Exception as e:
            return False, f"Failed to validate file {file_path}: {e}"

    @staticmethod
    def read_file_safe(
        file_path: Path, encoding: str = "utf-8"
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Safely read file with fallback encodings.

        Returns:
            Tuple of (content: Optional[str], error: Optional[str])
        """
        encodings = [encoding, "utf-8", "latin-1", "cp1252", "ascii"]

        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc, errors="replace") as f:
                    content = f.read()
                    if content:
                        logger.debug(f"[+] Read file with {enc}: {file_path}")
                        return content, None
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.debug(f"Failed to read {file_path} with {enc}: {e}")
                continue

        return None, f"Could not read file with any encoding: {file_path}"

    @staticmethod
    def write_file_safe(file_path: Path, content: str) -> Tuple[bool, Optional[str]]:
        """
        Safely write file with atomic operations.

        Returns:
            Tuple of (success: bool, error: Optional[str])
        """
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first
            temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Atomic rename
            temp_path.replace(file_path)
            logger.debug(f"[+] Wrote file: {file_path}")
            return True, None

        except (PermissionError, OSError) as e:
            return False, f"Permission denied writing to {file_path}: {e}"
        except Exception as e:
            return False, f"Failed to write file {file_path}: {e}"


class RobustIndexHandler:
    """Handles index files robustly."""

    @staticmethod
    def validate_index_files(index_dir: Path) -> Tuple[bool, List[str]]:
        """
        Validate all index files exist and are healthy.

        Returns:
            Tuple of (all_valid: bool, error_messages: List[str])
        """
        errors = []

        # Check directory exists
        if not index_dir.exists():
            errors.append(f"Index directory does not exist: {index_dir}")
            return False, errors

        # Check for at least one index file
        index_files = [
            "hnsw.idx",
            "corpus_faiss.index",
            "faiss.index",
            "index.faiss",
        ]

        index_found = False
        for filename in index_files:
            index_path = index_dir / filename
            if index_path.exists():
                valid, error = RobustFileHandler.validate_file(
                    index_path, min_size=1000
                )
                if valid:
                    logger.info(f"[+] Valid index file: {filename}")
                    index_found = True
                else:
                    errors.append(f"Invalid index file {filename}: {error}")

        if not index_found:
            errors.append(f"No valid index files found in {index_dir}")
            return False, errors

        # Check for mapping file
        mapping_path = index_dir / "corpus_mapping.parquet"
        if not mapping_path.exists():
            errors.append(f"Missing mapping file: corpus_mapping.parquet")
            return False, errors

        valid, error = RobustFileHandler.validate_file(mapping_path, min_size=100)
        if not valid:
            errors.append(f"Invalid mapping file: {error}")
            return False, errors

        logger.info("[+] All index files valid")
        return True, errors

    @staticmethod
    def rebuild_if_corrupted(
        index_dir: Path, rebuild_func
    ) -> Tuple[bool, Optional[str]]:
        """
        Check indexes and rebuild if corrupted.

        Returns:
            Tuple of (success: bool, message: Optional[str])
        """
        valid, errors = RobustIndexHandler.validate_index_files(index_dir)

        if valid:
            return True, "Indexes already valid"

        logger.warning(f"[X] Index validation failed: {errors}")
        logger.info("[~] Attempting to rebuild indexes...")

        try:
            rebuild_func()

            # Validate again
            valid, errors = RobustIndexHandler.validate_index_files(index_dir)
            if valid:
                logger.info("[+] Indexes rebuilt successfully")
                return True, "Indexes rebuilt successfully"
            else:
                return False, f"Rebuild failed: {errors}"

        except Exception as e:
            logger.error(f"[X] Rebuild failed: {e}")
            return False, str(e)


class RobustSearchHandler:
    """Handles searches robustly with fallbacks."""

    @staticmethod
    def safe_search(
        search_func,
        query: str,
        top_k: int = 10,
        fallback_search_func=None,
        timeout: float = 30.0,
    ) -> Tuple[List[dict], Optional[str]]:
        """
        Execute search with timeout and fallback.

        Returns:
            Tuple of (results: List[dict], error: Optional[str])
        """
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Search exceeded {timeout}s timeout")

        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            results = search_func(query, top_k)
            signal.alarm(0)  # Cancel alarm

            if not results:
                logger.warning(f"No results for query: {query}")

                # Try fallback if available
                if fallback_search_func:
                    logger.info("Trying fallback search...")
                    results = fallback_search_func(query, top_k)

            return results, None

        except TimeoutError as e:
            logger.error(f"Search timeout: {e}")
            return [], str(e)
        except Exception as e:
            logger.error(f"Search failed: {e}")

            # Try fallback
            if fallback_search_func:
                try:
                    logger.info("Trying fallback search...")
                    results = fallback_search_func(query, top_k)
                    return results, f"Primary search failed, using fallback: {e}"
                except Exception as fb_error:
                    logger.error(f"Fallback also failed: {fb_error}")
                    return [], f"Primary and fallback both failed: {e}; {fb_error}"

            return [], str(e)

        finally:
            signal.alarm(0)  # Cancel any pending alarm
            signal.signal(signal.SIGALRM, old_handler)


class RobustConfigHandler:
    """Handles configuration robustly."""

    @staticmethod
    def load_config_safe(
        config_path: Path, defaults: dict = None
    ) -> Tuple[dict, Optional[str]]:
        """
        Load config with defaults fallback.

        Returns:
            Tuple of (config: dict, error: Optional[str])
        """
        defaults = defaults or {}

        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return defaults.copy(), None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix == ".json":
                    config = json.load(f)
                elif config_path.suffix in [".yaml", ".yml"]:
                    import yaml

                    config = yaml.safe_load(f) or {}
                else:
                    return (
                        defaults.copy(),
                        f"Unknown config format: {config_path.suffix}",
                    )

            # Merge with defaults
            result = {**defaults, **config}
            logger.info(f"[+] Loaded config: {config_path}")
            return result, None

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config: {e}")
            return defaults.copy(), f"Invalid JSON: {e}"
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return defaults.copy(), str(e)
