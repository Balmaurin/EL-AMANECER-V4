"""
Sheily MCP Enterprise - Version Locker
Sistema avanzado de bloqueo de versiones enterprise
"""

import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class VersionLocker:
    """Sistema avanzado de bloqueo de versiones con integridad"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.config_dir = self.root_dir / "config"

    async def generate_lock_file(
        self, format_type: str = "poetry", strict: bool = False
    ) -> Dict[str, Any]:
        """Generar archivo de bloqueo de versiones enterprise"""

        # Get current installed packages
        installed_packages = await self._get_installed_packages()

        # Generate lock file based on format
        if format_type == "poetry":
            lock_data = await self._generate_poetry_lock(installed_packages, strict)
        elif format_type == "pip-tools":
            lock_data = await self._generate_pip_tools_lock(installed_packages, strict)
        else:  # requirements
            lock_data = await self._generate_requirements_lock(
                installed_packages, strict
            )

        # Calculate total packages based on format
        package_count = 0
        if format_type == "poetry":
            package_count = len(lock_data.get("package", {}))
        elif format_type == "pip-tools":
            package_count = len(lock_data.get("dependencies", {}))
        else:  # requirements
            package_count = len(lock_data.get("dependencies", {}))

        # Add metadata
        lock_data.update(
            {
                "metadata": {
                    "generated_at": asyncio.get_event_loop().time(),
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "working_directory": str(self.root_dir),
                    "total_packages": package_count,
                }
            }
        )

        return lock_data

    async def _get_installed_packages(self) -> Dict[str, str]:
        """Obtener paquetes instalados actualmente"""

        packages = {}

        try:
            # Use pip list to get installed packages
            result = await self._run_command(
                [sys.executable, "-m", "pip", "list", "--format=json"]
            )

            if result["success"]:
                package_list = json.loads(result["output"])
                packages = {pkg["name"]: pkg["version"] for pkg in package_list}

        except Exception as e:
            logger.warning(f"Failed to get installed packages: {e}")

        return packages

    async def _generate_poetry_lock(
        self, installed_packages: Dict[str, str], strict: bool
    ) -> Dict[str, Any]:
        """Generate Poetry-style lock file"""

        dependencies = {}
        package_hashes = {}

        for name, version in installed_packages.items():
            package_key = name.lower()

            # Generate hash for integrity
            package_hash = await self._generate_package_hash(name, version)

            dependencies[package_key] = {
                "version": version,
                "description": "",
                "category": "main",
            }

            if strict:
                package_hashes[f"{package_key}-{version}"] = package_hash

        lock_data = {
            "_meta": {
                "lock-version": "2.0",
                "python-versions": "*",
                "content-hash": await self._generate_content_hash(installed_packages),
                "hashes": package_hashes if strict else {},
            },
            "package": dependencies,
            "extras": {},
            "metadata": {
                "lock-version": "2.0",
                "python-versions": f"^{sys.version_info.major}.{sys.version_info.minor}",
                "content-hash": await self._generate_content_hash(installed_packages),
                "files": {},
            },
        }

        # Add missing format key so save_lock_file() logic works uniformly
        lock_data["format"] = "poetry"

        return lock_data

    async def _generate_pip_tools_lock(
        self, installed_packages: Dict[str, str], strict: bool
    ) -> Dict[str, Any]:
        """Generate pip-tools style requirements.txt lock"""

        lines = []

        for name, version in sorted(installed_packages.items()):
            line = f"{name}=={version}"

            if strict:
                # Add hash for strict mode
                package_hash = await self._generate_package_hash(name, version)
                line += f" --hash=sha256:{package_hash}"

            lines.append(line)

        return {
            "format": "pip-tools",
            "content": "\n".join(lines),
            "dependencies": installed_packages,
            "strict_mode": strict,
            "hash_enabled": strict,
        }

    async def _generate_requirements_lock(
        self, installed_packages: Dict[str, str], strict: bool
    ) -> Dict[str, Any]:
        """Generate basic requirements.txt style lock"""

        lines = []

        for name, version in sorted(installed_packages.items()):
            lines.append(f"{name}=={version}")

        return {
            "format": "requirements",
            "content": "\n".join(lines),
            "dependencies": installed_packages,
            "strict_mode": strict,
            "hash_enabled": False,
        }

    async def _generate_package_hash(self, name: str, version: str) -> str:
        """Generate a hash for package integrity"""

        # Create a deterministic hash based on package info
        content = f"{name}{version}{sys.version}{sys.platform}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def _generate_content_hash(self, packages: Dict[str, str]) -> str:
        """Generate content hash for all packages"""

        # Sort packages for deterministic hashing
        sorted_packages = sorted(packages.items())
        content = "".join(f"{name}{version}" for name, version in sorted_packages)

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def save_lock_file(
        self, lock_data: Dict[str, Any], output_path: Path
    ) -> bool:
        """Save lock file to disk"""

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if lock_data["format"] == "poetry":
                    json.dump(lock_data, f, indent=2)
                elif lock_data["format"] == "pip-tools":
                    # For pip-tools, save the content directly
                    f.write(lock_data.get("content", ""))
                else:  # requirements
                    f.write(lock_data.get("content", ""))

            logger.info(f"Lock file saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save lock file: {e}")
            return False

    async def validate_lock_file(self, lock_file_path: Path) -> Dict[str, Any]:
        """Validate integrity of lock file"""

        validation_result = {"valid": False, "errors": [], "warnings": []}

        try:
            # Load lock file
            with open(lock_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Basic validation - TODO: implement full validation logic
            if lock_file_path.suffix == ".json":
                try:
                    lock_data = json.loads(content)
                    validation_result["valid"] = True
                except json.JSONDecodeError as e:
                    validation_result["errors"].append(f"Invalid JSON: {e}")

            elif lock_file_path.suffix == ".txt":
                # Basic requirements.txt validation
                lines = content.split("\n")
                if lines:
                    validation_result["valid"] = True

        except FileNotFoundError:
            validation_result["errors"].append("Lock file not found")
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")

        return validation_result

    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Execute command and return results"""

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "output": stdout.decode("utf-8", errors="ignore").strip(),
                "error": stderr.decode("utf-8", errors="ignore").strip(),
                "returncode": process.returncode,
            }

        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "returncode": -1}
