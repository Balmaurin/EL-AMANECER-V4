"""
Sheily MCP Enterprise - Optimization Engine
Sistema de optimización avanzada de dependencias
"""

import asyncio
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class OptimizationEngine:
    """Motor de optimización avanzada enterprise"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    async def optimize_dependencies(
        self, action: str, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Optimizar dependencias según acción especificada enterprise"""

        if action == "dedupe":
            return await self._deduplicate_packages(dry_run)
        elif action == "compact":
            return await self._compact_environment(dry_run)
        elif action == "cleanup":
            return await self._cleanup_unused(dry_run)
        else:
            return {
                "success": False,
                "error": f"Unknown optimization action: {action}",
                "action": action,
                "dry_run": dry_run,
                "removed_packages": 0,
                "deduped_packages": 0,
                "space_saved_mb": 0.0,
            }

    async def _deduplicate_packages(self, dry_run: bool) -> Dict[str, Any]:
        """Remove duplicate packages and optimize for size"""
        result = {
            "action": "dedupe",
            "dry_run": dry_run,
            "removed_packages": 0,
            "deduped_packages": 0,
            "space_saved_mb": 0.0,
            "errors": [],
        }

        try:
            # Analyze installed packages
            installed_packages = await self._get_installed_packages()

            # Find potential deduplications (same package with different case/names)
            duplicates = self._find_duplicate_packages(installed_packages)

            if duplicates and not dry_run:
                # Remove duplicates
                removed_count = await self._remove_duplicate_packages(duplicates)
                result["removed_packages"] = removed_count
                result["deduped_packages"] = len(duplicates)

            elif duplicates:
                # Dry run - just count
                result["deduped_packages"] = len(duplicates)

            # Estimate space savings
            if duplicates:
                result["space_saved_mb"] = len(duplicates) * 5.0  # Rough estimate

            logger.info(
                f"Deduplication completed: {result['deduped_packages']} duplicates found"
            )

        except Exception as e:
            result["errors"].append(f"Deduplication failed: {str(e)}")
            logger.error(f"Deduplication failed: {e}")

        return result

    async def _compact_environment(self, dry_run: bool) -> Dict[str, Any]:
        """Compact the environment by cleaning up cache and temporary files"""
        result = {
            "action": "compact",
            "dry_run": dry_run,
            "removed_packages": 0,
            "deduped_packages": 0,
            "space_saved_mb": 0.0,
            "errors": [],
        }

        try:
            # Clear pip cache
            space_saved_pip = await self._clear_pip_cache(dry_run)

            # Clear Python cache files
            space_saved_python = await self._clear_python_cache(dry_run)

            # Remove temporary files
            space_saved_temp = await self._clear_temp_files(dry_run)

            result["space_saved_mb"] = (
                space_saved_pip + space_saved_python + space_saved_temp
            )

            logger.info(f"Compact completed: {result['space_saved_mb']:.1f} MB saved")

        except Exception as e:
            result["errors"].append(f"Compact failed: {str(e)}")
            logger.error(f"Compact failed: {e}")

        return result

    async def _cleanup_unused(self, dry_run: bool) -> Dict[str, Any]:
        """Remove unused packages (advanced cleanup)"""
        result = {
            "action": "cleanup",
            "dry_run": dry_run,
            "removed_packages": 0,
            "deduped_packages": 0,
            "space_saved_mb": 0.0,
            "errors": [],
        }

        try:
            # Get installed packages
            installed_packages = await self._get_installed_packages()

            # Analyze requirements
            required_packages = await self._get_required_packages()

            # Find unused packages
            unused_packages = self._find_unused_packages(
                installed_packages, required_packages
            )

            if unused_packages and not dry_run:
                # Remove unused packages
                removed_count = await self._remove_unused_packages(unused_packages)
                result["removed_packages"] = removed_count

                # Estimate space saved
                result["space_saved_mb"] = removed_count * 8.0  # Rough estimate

            elif unused_packages:
                # Dry run - just count
                result["removed_packages"] = len(unused_packages)

            logger.info(
                f"Cleanup completed: {result['removed_packages']} unused packages found"
            )

        except Exception as e:
            result["errors"].append(f"Cleanup failed: {str(e)}")
            logger.error(f"Cleanup failed: {e}")

        return result

    async def _get_installed_packages(self) -> Dict[str, str]:
        """Get currently installed packages"""
        packages = {}

        try:
            import subprocess

            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                import json

                package_list = json.loads(result.stdout)
                packages = {pkg["name"]: pkg["version"] for pkg in package_list}

        except Exception as e:
            logger.warning(f"Failed to get installed packages: {e}")

        return packages

    async def _get_required_packages(self) -> Dict[str, str]:
        """Get packages listed in requirements files"""
        required = {}

        # Read requirements.txt
        requirements_file = self.root_dir / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            package_name = (
                                line.split()[0]
                                .split(">=")[0]
                                .split("==")[0]
                                .split("<")[0]
                            )
                            required[package_name.lower()] = line
            except Exception as e:
                logger.warning(f"Failed to read requirements.txt: {e}")

        # Read other requirements files in config/
        config_dir = self.root_dir / "config"
        if config_dir.exists():
            for req_file in config_dir.glob("requirements-*.txt"):
                try:
                    with open(req_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                package_name = (
                                    line.split()[0]
                                    .split(">=")[0]
                                    .split("==")[0]
                                    .split("<")[0]
                                )
                                required[package_name.lower()] = line
                except Exception:
                    continue

        return required

    def _find_duplicate_packages(self, installed_packages: Dict[str, str]) -> List[str]:
        """Find potentially duplicate packages"""
        # This is a simplified implementation
        # Real duplicate detection would be more complex
        seen_names = set()
        duplicates = []

        for name in installed_packages.keys():
            lower_name = name.lower()
            if lower_name in seen_names:
                # Simplified duplicate detection - packages with similar names
                for existing in seen_names:
                    if existing.replace("-", "").replace("_", "") != lower_name.replace(
                        "-", ""
                    ).replace("_", ""):
                        continue
                    if existing != lower_name:
                        duplicates.append(name)
                        break
            else:
                seen_names.add(lower_name)

        return duplicates

    def _find_unused_packages(
        self, installed: Dict[str, str], required: Dict[str, str]
    ) -> List[str]:
        """Find packages that are installed but not in requirements"""
        unused = []

        for package_name in installed.keys():
            package_lower = package_name.lower()

            # Skip if package is in requirements (exact match)
            if package_lower in required:
                continue

            # Skip common system packages that typically aren't in requirements
            system_packages = {
                "pip",
                "setuptools",
                "wheel",
                "pkg-resources",
                "distribute",
            }

            if package_lower not in system_packages:
                # Check if it might be a dependency (simplified check)
                # Real implementation would need dependency tree analysis
                unused.append(package_name)

        return unused

    async def _remove_duplicate_packages(self, duplicates: List[str]) -> int:
        """Remove duplicate packages"""
        removed_count = 0

        try:
            import subprocess

            for package in duplicates:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", package, "-y"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode == 0:
                    removed_count += 1

        except Exception as e:
            logger.warning(f"Failed to remove duplicate packages: {e}")

        return removed_count

    async def _remove_unused_packages(self, unused: List[str]) -> int:
        """Remove unused packages"""
        removed_count = 0

        try:
            import subprocess

            for package in unused:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", package, "-y"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode == 0:
                    removed_count += 1

        except Exception as e:
            logger.warning(f"Failed to remove unused packages: {e}")

        return removed_count

    async def _clear_pip_cache(self, dry_run: bool) -> float:
        """Clear pip cache and return space saved"""
        space_saved_mb = 0.0

        try:
            import subprocess

            # Get cache directory
            result = subprocess.run(
                [sys.executable, "-m", "pip", "cache", "dir"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                cache_dir = Path(result.stdout.strip())

                # Calculate space before clearing
                space_before = await self._calculate_directory_size(cache_dir)

                if not dry_run:
                    # Clear cache
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "cache", "purge"],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        space_after = await self._calculate_directory_size(cache_dir)
                        space_saved_mb = (space_before - space_after) / (1024 * 1024)
                else:
                    # Dry run - just estimate
                    space_saved_mb = (
                        space_before / (1024 * 1024) * 0.8
                    )  # Assume 80% can be cleared

        except Exception as e:
            logger.warning(f"Failed to clear pip cache: {e}")

        return space_saved_mb

    async def _clear_python_cache(self, dry_run: bool) -> float:
        """Clear Python __pycache__ directories"""
        space_saved_mb = 0.0

        try:
            import shutil
            import subprocess

            # Find and remove __pycache__ directories
            result = subprocess.run(
                [
                    "find",
                    str(self.root_dir),
                    "-type",
                    "d",
                    "-name",
                    "__pycache__",
                    "-print0",
                ],
                capture_output=True,
                text=False,
            )

            if result.returncode == 0:
                cache_dirs = result.stdout.decode("utf-8").split("\0")

                for cache_dir_str in cache_dirs:
                    if not cache_dir_str:
                        continue

                    cache_dir = Path(cache_dir_str.strip())

                    if cache_dir.exists():
                        size = await self._calculate_directory_size(cache_dir)
                        space_saved_mb += size / (1024 * 1024)

                        if not dry_run:
                            shutil.rmtree(cache_dir, ignore_errors=True)

            # Clear .pyc files
            result = subprocess.run(
                ["find", str(self.root_dir), "-name", "*.pyc", "-delete"],
                capture_output=True,
            )

        except Exception as e:
            logger.warning(f"Failed to clear Python cache: {e}")

        return space_saved_mb

    async def _clear_temp_files(self, dry_run: bool) -> float:
        """Clear temporary files"""
        space_saved_mb = 0.0

        try:
            import subprocess
            import tempfile

            # Clear system temp directory of Python-related files
            temp_dir = Path(tempfile.gettempdir())

            # Find Python temp files
            result = subprocess.run(
                ["find", str(temp_dir), "-name", "*pip*", "-type", "f", "-mtime", "+1"],
                capture_output=True,
            )

            if result.returncode == 0:
                temp_files = result.stdout.decode("utf-8").strip().split("\n")

                for temp_file in temp_files:
                    temp_file_path = Path(temp_file.strip())
                    if temp_file_path.exists():
                        size = temp_file_path.stat().st_size
                        space_saved_mb += size / (1024 * 1024)

                        if not dry_run:
                            temp_file_path.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Failed to clear temp files: {e}")

        return space_saved_mb

    async def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate directory size in bytes"""
        total_size = 0

        try:
            import subprocess

            if directory.exists():
                # Use find to calculate size (Unix/linux)
                result = subprocess.run(
                    [
                        "find",
                        str(directory),
                        "-type",
                        "f",
                        "-exec",
                        "stat",
                        "-f",
                        "%z",
                        "{}",
                        ";",
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    sizes = result.stdout.strip().split("\n")
                    total_size = sum(int(size) for size in sizes if size.isdigit())

        except Exception:
            # Fallback - simpler method
            try:
                if directory.exists():
                    result = subprocess.run(
                        ["du", "-sb", str(directory)], capture_output=True, text=True
                    )

                    if result.returncode == 0:
                        parts = result.stdout.strip().split()
                        if parts:
                            total_size = int(parts[0])
            except:
                pass

        return total_size
