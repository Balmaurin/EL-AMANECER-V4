"""
Sheily MCP Enterprise - Update Manager
Gestión inteligente de actualizaciones de dependencias
"""

import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pkg_resources

logger = logging.getLogger(__name__)


class UpdateManager:
    """Gestor inteligente de actualizaciones enterprise"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.requirements_file = self.root_dir / "requirements.txt"

    async def update_dependencies(
        self, safe_mode: bool = True, target: str = "", interactive: bool = False
    ) -> Dict[str, Any]:
        """Actualizar dependencias de forma inteligente"""

        result = {
            "updated_packages": 0,
            "skipped_packages": 0,
            "failed_packages": 0,
            "warnings": [],
            "errors": [],
            "breaking_changes": [],
            "safe_mode": safe_mode,
            "interactive_mode": interactive,
            "target_filter": target,
        }

        try:
            # Get current installed packages
            current_packages = await self._get_installed_packages()

            # Get requirements
            required_packages = await self._parse_requirements_file()

            # Determine update strategy
            if safe_mode:
                updates = await self._calculate_safe_updates(
                    current_packages, required_packages, target
                )
            else:
                updates = await self._calculate_full_updates(
                    current_packages, required_packages, target
                )

            # Execute updates
            update_result = await self._execute_updates(updates, interactive)

            # Merge results
            result.update(update_result)

        except Exception as e:
            result["errors"].append(f"Update process failed: {str(e)}")
            logger.error(f"Update manager failed: {e}")

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

    async def _parse_requirements_file(self) -> Dict[str, Dict[str, Any]]:
        """Parse requirements.txt for update specifications"""
        requirements = {}

        if not self.requirements_file.exists():
            logger.warning(f"Requirements file not found: {self.requirements_file}")
            return requirements

        try:
            with open(self.requirements_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    req = await self._parse_requirement_line(line)
                    if req:
                        requirements[req["name"]] = req

        except Exception as e:
            logger.error(f"Failed to parse requirements file: {e}")

        return requirements

    async def _parse_requirement_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single requirement line"""
        try:
            # Parse package name and version spec
            if ">=" in line or "==" in line or "<=" in line:
                # Extract package name (first word)
                package_name = line.split()[0]

                return {
                    "name": package_name.lower(),
                    "spec": line,
                    "original_line": line,
                }

            return None

        except Exception as e:
            logger.debug(f"Failed to parse requirement line '{line}': {e}")
            return None

    async def _calculate_safe_updates(
        self, current: Dict[str, str], required: Dict[str, Dict[str, Any]], target: str
    ) -> List[Dict[str, Any]]:
        """Calculate safe updates (patch and minor only)"""
        updates = []

        for pkg_name, current_version in current.items():
            # Apply target filter if specified
            if target and target != "all" and pkg_name.lower() not in target.lower():
                continue

            # Check if package has requirement specification
            if pkg_name.lower() in required:
                req_info = required[pkg_name.lower()]

                # Check if update is within safe ranges
                safe_version = await self._get_safe_update_version(
                    pkg_name, current_version
                )

                if safe_version and safe_version != current_version:
                    # Compare semantic versions
                    if self._is_semantic_update_safe(current_version, safe_version):
                        updates.append(
                            {
                                "package": pkg_name,
                                "current_version": current_version,
                                "new_version": safe_version,
                                "type": "safe_update",
                                "reason": f"Patch/minor update available ({current_version} -> {safe_version})",
                            }
                        )

        return updates

    async def _calculate_full_updates(
        self, current: Dict[str, str], required: Dict[str, Dict[str, Any]], target: str
    ) -> List[Dict[str, Any]]:
        """Calculate full updates (including breaking changes)"""
        updates = []

        for pkg_name, current_version in current.items():
            # Apply target filter
            if target and target != "all" and pkg_name.lower() not in target.lower():
                continue

            # Get latest available version
            latest_version = await self._get_latest_version(pkg_name)

            if latest_version and latest_version != current_version:
                update_type = (
                    "breaking"
                    if self._is_breaking_update(current_version, latest_version)
                    else "regular"
                )

                updates.append(
                    {
                        "package": pkg_name,
                        "current_version": current_version,
                        "new_version": latest_version,
                        "type": update_type,
                        "reason": f"Update available ({current_version} -> {latest_version})",
                    }
                )

        return updates

    async def _execute_updates(
        self, updates: List[Dict[str, Any]], interactive: bool
    ) -> Dict[str, Any]:
        """Execute the calculated updates"""
        result = {
            "updated_packages": 0,
            "skipped_packages": 0,
            "failed_packages": 0,
            "warnings": [],
            "errors": [],
            "breaking_changes": [],
        }

        for update in updates:
            package_name = update["package"]

            # Interactive confirmation for breaking changes
            if interactive and update["type"] == "breaking":
                if not await self._confirm_update(update):
                    result["skipped_packages"] += 1
                    continue

            # Execute update
            update_success = await self._update_single_package(
                package_name, update["new_version"]
            )

            if update_success:
                result["updated_packages"] += 1
                logger.info(
                    f"Successfully updated {package_name} to {update['new_version']}"
                )

                if update["type"] == "breaking":
                    result["breaking_changes"].append(
                        {
                            "package": package_name,
                            "old_version": update["current_version"],
                            "new_version": update["new_version"],
                        }
                    )
            else:
                result["failed_packages"] += 1
                result["errors"].append(f"Failed to update {package_name}")

        return result

    async def _update_single_package(
        self, package_name: str, target_version: str
    ) -> bool:
        """Update a single package"""
        try:
            import subprocess

            # Use pip install --upgrade to specific version
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    f"{package_name}=={target_version}",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            return result.returncode == 0

        except Exception as e:
            logger.warning(f"Failed to update {package_name}: {e}")
            return False

    async def _get_safe_update_version(
        self, package_name: str, current_version: str
    ) -> Optional[str]:
        """Get a safe update version (patch/minor only)"""
        try:
            # This would normally query PyPI or use pip-tools
            # For now, return None to indicate no safe update available
            # In a full implementation, this would check PyPI API
            return None

        except Exception:
            return None

    async def _get_latest_version(self, package_name: str) -> Optional[str]:
        """Get latest available version from PyPI"""
        try:
            # This would normally query PyPI API
            # For now, return None as we don't have this functionality
            # In production, this would use PyPI JSON API
            return None

        except Exception:
            return None

    def _is_semantic_update_safe(self, current: str, new: str) -> bool:
        """Check if semantic version update is safe (patch/minor only)"""
        try:
            current_parts = self._parse_version(current)
            new_parts = self._parse_version(new)

            # Safe if major version same and minor/patch increased
            return current_parts[0] == new_parts[0]

        except:
            return False

    def _is_breaking_update(self, current: str, new: str) -> bool:
        """Check if update is breaking (different major version)"""
        try:
            current_parts = self._parse_version(current)
            new_parts = self._parse_version(new)

            return current_parts[0] != new_parts[0]

        except:
            return False

    def _parse_version(self, version: str) -> tuple:
        """Parse version string into tuple"""
        try:
            # Simple version parsing
            parts = version.split(".")
            return tuple(int(x) for x in parts[:3])

        except:
            return (0, 0, 0)

    async def _confirm_update(self, update: Dict[str, Any]) -> bool:
        """Get user confirmation for update"""
        try:
            import sys

            print(f"\n⚠️ Breaking change detected:")
            print(f"  Package: {update['package']}")
            print(f"  Update: {update['current_version']} -> {update['new_version']}")
            print(f"  Reason: {update.get('reason', 'N/A')}")

            response = input("Continue with update? (y/N): ").strip().lower()
            return response in ["y", "yes"]

        except:
            return False
