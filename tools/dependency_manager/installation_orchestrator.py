"""
Sheily MCP Enterprise - Installation Orchestrator
Sistema inteligente de instalación de dependencias
"""

import asyncio
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class InstallationOrchestrator:
    """Orquestador inteligente de instalación"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.config_dir = self.root_dir / "config"
        self.frontend_dir = self.root_dir / "Frontend"

        # Dependency file mappings - INCLUDING MCP-ADK INTEGRATION
        self.dependency_files = {
            "core": self.root_dir / "requirements.txt",
            "dev": self.config_dir / "requirements-dev.txt",
            "rag": self.config_dir / "requirements-rag.txt",
            "ci": self.config_dir / "requirements-ci.txt",
            "frontend": self.frontend_dir / "package.json",
            # GOOGLE ADK INTEGRATION - Agregado automaticamente
            "mcp_adk": self.root_dir / "requirements_mcp_adk.txt",
        }

        # MCP-ADK Integration flags
        self.adk_integration_enabled = self._check_adk_integration_feasibility()
        self.adk_requirements_generated = False

    async def install_dependencies(
        self,
        target: str = "all",
        dry_run: bool = False,
        force: bool = False,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """Instalar dependencias por fases"""

        start_time = time.time()

        results = {
            "python_packages": 0,
            "frontend_packages": 0,
            "total_time": 0.0,
            "phases_completed": [],
            "errors": [],
            "warnings": [],
        }

        try:
            if dry_run:
                logger.info("Running in dry-run mode - no actual installations")

            # Install Python dependencies by category
            python_results = await self._install_python_dependencies(
                target, dry_run, force, parallel
            )
            results.update(python_results)
            results["phases_completed"].extend(
                python_results.get("phases_completed", [])
            )

            # Install Frontend dependencies if requested
            if target in ["frontend", "all"]:
                frontend_results = await self._install_frontend_dependencies(
                    dry_run, force
                )
                results["frontend_packages"] = frontend_results.get(
                    "installed_packages", 0
                )
                if frontend_results.get("completed"):
                    results["phases_completed"].append("frontend")
                if frontend_results.get("errors"):
                    results["errors"].extend(frontend_results["errors"])

        except Exception as e:
            results["errors"].append(f"Installation failed: {str(e)}")
            logger.error(f"Installation orchestrator failed: {e}")

        results["total_time"] = time.time() - start_time

        # Log final summary
        total_installed = results["python_packages"] + results["frontend_packages"]
        logger.info(
            f"Installation completed: {total_installed} packages in {results['total_time']:.1f}s"
        )

        return results

    async def _install_python_dependencies(
        self, target: str, dry_run: bool, force: bool, parallel: bool
    ) -> Dict[str, Any]:
        """Install Python dependencies by category"""

        python_results = {"python_packages": 0, "phases_completed": [], "errors": []}

        categories = self._get_target_categories(target)

        if parallel and len(categories) > 1:
            # Install in parallel
            tasks = []
            for category in categories:
                task = self._install_python_category(category, dry_run, force)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                category = categories[i]
                if isinstance(result, Exception):
                    python_results["errors"].append(f"{category}: {str(result)}")
                elif isinstance(result, dict) and result.get("success"):
                    python_results["python_packages"] += result.get(
                        "installed_count", 0
                    )
                    python_results["phases_completed"].append(category)
                    if result.get("errors"):
                        python_results["errors"].extend(result["errors"])
        else:
            # Install sequentially
            for category in categories:
                result = await self._install_python_category(category, dry_run, force)
                if result and result.get("success"):
                    python_results["python_packages"] += result.get(
                        "installed_count", 0
                    )
                    python_results["phases_completed"].append(category)
                    if result.get("errors"):
                        python_results["errors"].extend(result["errors"])
                elif result:
                    python_results["errors"].extend(result.get("errors", []))

        return python_results

    async def _install_python_category(
        self, category: str, dry_run: bool, force: bool
    ) -> Dict[str, Any]:
        """Install a specific category of Python dependencies"""

        file_path = self.dependency_files.get(category)
        if not file_path or not file_path.exists():
            return {
                "success": False,
                "installed_count": 0,
                "errors": [f"No {category} requirements file found"],
            }

        logger.info(f"Installing {category} Python dependencies from {file_path}")

        # Prepare pip install command
        cmd = [sys.executable, "-m", "pip", "install"]

        if force:
            cmd.append("--force-reinstall")
        else:
            cmd.append("--upgrade")

        if dry_run:
            cmd.extend(["--dry-run", "--quiet"])

        cmd.extend(["-r", str(file_path)])

        # Execute the installation
        result = await self._run_command(cmd)

        if result["success"] or dry_run:
            # Count packages (rough estimate from requirements file)
            installed_count = await self._count_packages_in_file(file_path)
            return {
                "success": True,
                "installed_count": installed_count if not dry_run else 0,
                "category": category,
                "errors": [],
            }
        else:
            return {
                "success": False,
                "installed_count": 0,
                "category": category,
                "errors": [result["error"]],
            }

    async def _install_frontend_dependencies(
        self, dry_run: bool, force: bool
    ) -> Dict[str, Any]:
        """Install frontend dependencies using npm"""

        frontend_results = {"installed_packages": 0, "completed": False, "errors": []}

        package_json = self.dependency_files.get("frontend")
        if not package_json or not package_json.exists():
            frontend_results["errors"].append("No package.json found")
            return frontend_results

        logger.info("Installing frontend dependencies...")

        # Change to frontend directory if it exists
        cwd = self.frontend_dir if self.frontend_dir.exists() else self.root_dir

        # Prepare npm install command
        cmd = ["npm", "install"]

        if force:
            cmd.append("--force")

        if dry_run:
            cmd.append("--dry-run")

        # Execute npm install
        result = await self._run_command(cmd, cwd=str(cwd))

        if result["success"] or dry_run:
            # Try to count installed packages from package.json
            try:
                with open(package_json, "r", encoding="utf-8") as f:
                    package_data = json.load(f)

                deps_count = len(package_data.get("dependencies", {}))
                dev_deps_count = len(package_data.get("devDependencies", {}))

                frontend_results["installed_packages"] = (
                    deps_count + dev_deps_count if not dry_run else 0
                )
                frontend_results["completed"] = True

            except Exception as e:
                frontend_results["errors"].append(f"Could not parse package.json: {e}")
        else:
            frontend_results["errors"].append(result["error"])

        return frontend_results

    def _get_target_categories(self, target: str) -> List[str]:
        """Get list of categories to install based on target"""

        category_map = {
            "core": ["core"],
            "ai": ["rag"],  # AI stuff is in rag requirements
            "mcp_adk": (
                ["mcp_adk"] if self.adk_integration_enabled else []
            ),  # GOOGLE ADK INTEGRATION
            "dev": ["dev"],
            "frontend": [],  # Handled separately
            "all": ["core", "dev", "rag", "ci"]
            + (["mcp_adk"] if self.adk_integration_enabled else []),
        }

        return category_map.get(target, [])

    async def _count_packages_in_file(self, file_path: Path) -> int:
        """Count packages in a requirements file"""

        count = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        count += 1
        except Exception:
            pass

        return count

    async def _run_command(self, cmd: List[str], cwd: str = None) -> Dict[str, Any]:
        """Execute command asynchronously"""

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "output": stdout.decode("utf-8", errors="ignore"),
                "error": stderr.decode("utf-8", errors="ignore"),
                "returncode": process.returncode,
            }

        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "error": f"Command not found: {cmd[0]}",
                "returncode": -1,
            }

        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "returncode": -1}

    # ============================================================================
    # MCP-ADK INTEGRATION METHODS
    # ============================================================================

    def _check_adk_integration_feasibility(self) -> bool:
        """
        Check if MCP-ADK integration is feasible and generate requirements file if needed

        Este método verifica si existe implementación MCP-ADK y genera
        automáticamente requirements_mcp_adk.txt si no existe.
        """
        try:
            # Check if MCP-ADK implementation exists
            adk_integration_file = self.root_dir / "sheily_core" / "adk_integration.py"

            if not adk_integration_file.exists():
                logger.info("MCP-ADK integration not found - integration disabled")
                return False

            # Check if MCP-ADK requirements file exists
            adk_requirements_file = self.root_dir / "requirements_mcp_adk.txt"

            if not adk_requirements_file.exists():
                logger.info("Generating requirements_mcp_adk.txt...")
                success = self._generate_adk_requirements_file(adk_requirements_file)
                if not success:
                    logger.warning("Failed to generate MCP-ADK requirements file")
                    return False

            logger.info("MCP-ADK integration feasibility: ENABLED")
            return True

        except Exception as e:
            logger.warning(f"MCP-ADK integration check failed: {e}")
            return False

    def _generate_adk_requirements_file(self, requirements_file: Path) -> bool:
        """
        Generate requirements_mcp_adk.txt with Google ADK dependencies
        """
        try:
            # MCP-ADK dependencies for Google ADK enterprise integration
            adk_dependencies = [
                "# GOOGLE ADK REQUIREMENTS - MCP-ADK ENTERPRISE INTEGRATION",
                "# Generated automatically by InstallationOrchestrator",
                "# Version: 18 de Noviembre 2025",
                "",
                "# Core Google ADK package (optional - functional with stubs)",
                "google-adk>=1.0.0",
                "",
                "# Essential dependencies for ADK functionality",
                "protobuf>=3.20.0",
                "grpcio>=1.50.0",
                "google-auth>=2.0.0",
                "google-cloud-storage>=2.0.0",
                "",
                "# AI/ML dependencies for ADK models integration",
                "transformers>=4.20.0",
                "torch>=1.12.0",
                "tensorflow>=2.10.0",
                "",
                "# Web UI dependencies for ADK debugging",
                "fastapi>=0.100.0",
                "uvicorn>=0.20.0",
                "websockets>=11.0.0",
                "",
                "# Streaming and multimodal dependencies",
                "opencv-python>=4.8.0",
                "pillow>=9.0.0",
                "speechrecognition>=3.10.0",
                "",
                "# Enterprise features",
                "kubernetes>=24.0.0",
                "docker>=6.0.0",
                "prometheus-client>=0.17.0",
                "",
                "# ADK CLI tools (fallback implementations available)",
                "# google-cloud-aiplatform>=1.0.0",
                "# google-cloud-functions>=1.10.0",
            ]

            # Write the requirements file
            requirements_file.parent.mkdir(parents=True, exist_ok=True)

            with open(requirements_file, "w", encoding="utf-8") as f:
                f.write("\n".join(adk_dependencies))
                f.write("\n")

            logger.info(f"Generated MCP-ADK requirements file: {requirements_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate ADK requirements file: {e}")
            return False

    async def install_mcp_adk_dependencies(
        self, force: bool = False, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Install MCP-ADK dependencies through the dependency management system

        Esta función integra Google ADK completamente con el sistema de dependencias MCP Enterprise.
        """
        results = {
            "adk_installed": False,
            "integration_active": self.adk_integration_enabled,
            "packages_installed": 0,
            "errors": [],
            "warnings": [],
        }

        if not self.adk_integration_enabled:
            results["warnings"].append("MCP-ADK integration not enabled")
            results["warnings"].append("Install MCP-ADK implementation first")
            return results

        try:
            logger.info("Installing MCP-ADK dependencies through dependency manager...")

            # Use existing installation logic for MCP-ADK category
            install_result = await self._install_python_category(
                "mcp_adk", dry_run, force
            )

            if install_result.get("success"):
                results["adk_installed"] = True
                results["packages_installed"] = install_result.get("installed_count", 0)
                logger.info(
                    f"MCP-ADK dependencies installed: {results['packages_installed']} packages"
                )
            else:
                results["errors"].extend(install_result.get("errors", []))

        except Exception as e:
            results["errors"].append(f"MCP-ADK installation failed: {str(e)}")

        return results

    def get_mcp_adk_integration_status(self) -> Dict[str, Any]:
        """
        Get MCP-ADK integration status for dependency management

        Returns comprehensive status of MCP-ADK integration within the dependency system.
        """
        status: Dict[str, Any] = {
            "adk_integration_enabled": self.adk_integration_enabled,
            "adk_requirements_file_exists": False,
            "google_adk_available": False,
            "mcp_control_active": self.adk_integration_enabled,
            "requirements_generated": self.adk_requirements_generated,
        }

        # Check requirements file existence
        adk_requirements_file = self.dependency_files.get("mcp_adk")
        if adk_requirements_file and adk_requirements_file.exists():
            status["adk_requirements_file_exists"] = True
            status["requirements_file_path"] = (
                str(adk_requirements_file) if adk_requirements_file else None
            )

            # Count packages in requirements
            try:
                with open(adk_requirements_file, "r", encoding="utf-8") as f:
                    package_count = sum(
                        1 for line in f if line.strip() and not line.startswith("#")
                    )
                status["adk_required_packages"] = package_count
            except Exception as e:
                status["requirements_error"] = str(e)

        # Check Google ADK availability - CATCH import error properly
        try:
            import google.adk  # type: ignore

            status["google_adk_available"] = True
            status["adk_version"] = getattr(google.adk, "__version__", "unknown")
        except ImportError:
            status["google_adk_available"] = False
            status["adk_availability_message"] = (
                "Google ADK not installed - MCP-ADK operates with functional stubs"
            )

        # MCP-ADK specific status
        status["mcp_orchestrator_present"] = (
            self.root_dir / "sheily_core" / "adk_integration.py"
        ).exists()
        status["cli_integration_present"] = (
            self.root_dir / "scripts" / "mcp_adk_cli.py"
        ).exists()

        return status
