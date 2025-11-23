"""
Sheily MCP Enterprise - Validation Engine
Sistema completo de validación de dependencias
"""

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationEngine:
    """Motor de validación completo"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    async def quick_validate(self) -> Dict[str, Any]:
        """Validación rápida: solo pip list"""
        return await self._validate_installed_packages(quick=True)

    async def full_validate(self) -> Dict[str, Any]:
        """Validación completa: pip check + requirements validation"""
        results = await self._validate_installed_packages(quick=False)

        # Add requirements validation if requirements.txt exists
        validation_results = await self._validate_requirements_alignment()
        results.update(validation_results)

        return results

    async def standard_validate(self) -> Dict[str, Any]:
        """Validación estándar: pip check básico"""
        return await self._validate_installed_packages(quick=False)

    async def auto_repair(self, issues: Dict[str, Any]) -> Dict[str, Any]:
        """Intentar reparaciones automáticas"""
        repairs = {
            "pip_check_fixes": 0,
            "requirements_alignment_fixes": 0,
            "total_repairs": 0,
        }

        try:
            # Try pip check --fix if available (pip-tools extension)
            result = await self._run_command(
                [sys.executable, "-m", "pip", "check", "--fix"]
            )
            if result["success"]:
                repairs["pip_check_fixes"] += 1

        except Exception as e:
            logger.warning(f"Auto-repair failed: {e}")

        repairs["total_repairs"] = sum(repairs.values())
        return repairs

    async def _validate_installed_packages(self, quick: bool = False) -> Dict[str, Any]:
        """Validar paquetes instalados usando pip"""

        results = {
            "python_packages": 0,
            "python_packages_list": [],
            "pip_check_ok": False,
            "issues_found": 0,
            "issues_details": [],
        }

        try:
            # Get installed packages
            list_result = await self._run_command(
                [sys.executable, "-m", "pip", "list", "--format=json"]
            )

            if list_result["success"]:
                packages = json.loads(list_result["output"])
                results["python_packages"] = len(packages)
                results["python_packages_list"] = [
                    {"name": pkg["name"], "version": pkg["version"]} for pkg in packages
                ]

            # Run pip check unless quick mode
            if not quick:
                check_result = await self._run_command(
                    [sys.executable, "-m", "pip", "check"]
                )

                results["pip_check_ok"] = check_result["success"]
                if not check_result["success"]:
                    results["issues_found"] += 1
                    results["issues_details"].append(
                        {"type": "pip_check_failed", "message": check_result["error"]}
                    )

        except Exception as e:
            results["issues_found"] += 1
            results["issues_details"].append(
                {"type": "validation_error", "message": str(e)}
            )

        return results

    async def _validate_requirements_alignment(self) -> Dict[str, Any]:
        """Validar alineación entre installed packages y requirements"""

        alignment_results = {
            "requirements_aligned": True,
            "alignment_issues": 0,
            "alignment_details": [],
            "requirements_found": False,
        }

        requirements_file = self.root_dir / "requirements.txt"

        if not requirements_file.exists():
            return alignment_results

        alignment_results["requirements_found"] = True

        try:
            # Get installed packages
            installed_result = await self._run_command(
                [sys.executable, "-m", "pip", "list", "--format=json"]
            )

            if not installed_result["success"]:
                return alignment_results

            installed_packages = {
                pkg["name"].lower(): pkg["version"]
                for pkg in json.loads(installed_result["output"])
            }

            # Parse requirements.txt
            required_packages = {}
            with open(requirements_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # Simple parsing - get package name
                    package_name = (
                        line.split()[0].split(">=")[0].split("==")[0].split("<")[0]
                    )
                    required_packages[package_name.lower()] = line

            # Check alignment
            for req_pkg, req_spec in required_packages.items():
                if req_pkg not in installed_packages:
                    alignment_results["requirements_aligned"] = False
                    alignment_results["alignment_issues"] += 1
                    alignment_results["alignment_details"].append(
                        {
                            "type": "missing_package",
                            "package": req_pkg,
                            "required": req_spec,
                        }
                    )

        except Exception as e:
            logger.warning(f"Requirements alignment check failed: {e}")

        return alignment_results

    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Ejecutar comando y retornar resultados"""

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
