"""
Sheily MCP Enterprise - Environment Manager
Gestión avanzada de entornos virtuales Python
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Gestor avanzado de entornos virtuales"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.envs_dir = self.root_dir / ".envs"

    async def create_environment(
        self, name: str, python_version: str = "3.11"
    ) -> Dict[str, Any]:
        """Crear entorno virtual"""
        env_path = self.envs_dir / name

        try:
            # Crear comando para crear venv
            cmd = [sys.executable, "-m", "venv", str(env_path)]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            await process.wait()

            if process.returncode == 0:
                return {
                    "success": True,
                    "path": str(env_path),
                    "python_version": python_version,
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to create environment: {process.returncode}",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_environments(self) -> Dict[str, Any]:
        """Listar entornos disponibles"""
        if not self.envs_dir.exists():
            return {}

        envs = {}
        for env_dir in self.envs_dir.iterdir():
            if env_dir.is_dir():
                env_info = await self._get_env_info(env_dir)
                envs[env_dir.name] = env_info

        return envs

    async def activate_environment(self, name: str) -> Dict[str, Any]:
        """Activar entorno virtual"""
        env_path = self.envs_dir / name
        activate_script = env_path / "bin" / "activate"  # Unix
        if not activate_script.exists():
            activate_script = env_path / "Scripts" / "activate"  # Windows

        if activate_script.exists():
            return {"success": True, "activate_script": str(activate_script)}
        else:
            return {"success": False, "error": "Activate script not found"}

    async def remove_environment(self, name: str) -> Dict[str, Any]:
        """Eliminar entorno virtual"""
        env_path = self.envs_dir / name

        if not env_path.exists():
            return {"success": False, "error": "Environment does not exist"}

        try:
            import shutil

            shutil.rmtree(env_path)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_env_info(self, env_path: Path) -> Dict[str, Any]:
        """Obtener información de un entorno"""
        python_exe = env_path / "bin" / "python"  # Unix
        if not python_exe.exists():
            python_exe = env_path / "Scripts" / "python.exe"  # Windows

        if python_exe.exists():
            try:
                # Obtener versión de Python
                process = await asyncio.create_subprocess_exec(
                    str(python_exe),
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, _ = await process.communicate()
                version = stdout.decode().strip()

                return {"path": str(env_path), "python_version": version}
            except:
                pass

        return {"path": str(env_path), "python_version": "unknown"}
