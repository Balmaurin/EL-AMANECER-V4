#!/usr/bin/env python3
"""
Sistema de Coordinación Funcional - IMPLEMENTACIÓN REAL
Transforma la arquitectura abstracta en un sistema operativo funcional conectado al hardware.
"""

import asyncio
import json
import logging
import os
import shutil
import time
import platform
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

# Intentar importar psutil para métricas reales de hardware
try:
    import psutil
except ImportError:
    psutil = None

from .agent_registry import AgentRegistry
from .multi_agent_system import CoordinatorAgent, MultiAgentSystem, SpecializedAgent, AgentRole


@dataclass
class WorkloadBalance:
    """Balance de carga de trabajo REAL basado en recursos del sistema"""

    agent_id: str
    current_load: float
    max_capacity: float
    available_capacity: float
    last_updated: datetime
    process_id: int  # ID del proceso real si aplica


class FunctionalCoordinatorAgent(CoordinatorAgent):
    """Coordinador que gestiona tareas basándose en métricas REALES del sistema"""

    def __init__(self):
        super().__init__("functional_coordinator", "Functional Coordinator")
        self.workload_balances: Dict[str, WorkloadBalance] = {}
        self.task_assignment_history = []
        self.optimization_enabled = True
        self.logger = logging.getLogger("Sheily.Coordinator")

    async def intelligent_task_distribution(
        self, tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Distribuir tareas basándose en carga real de CPU/Memoria"""
        assignment_results = []

        # Actualizar métricas del sistema antes de distribuir
        current_system_load = self._get_real_system_load()

        for task in tasks:
            # Si el sistema está saturado (>90%), pausar asignación
            if current_system_load > 90.0:
                assignment_results.append({
                    "task_id": task.get("id", "unknown"),
                    "status": "delayed",
                    "reason": "system_saturation_protection"
                })
                continue

            best_agent = await self._find_optimal_agent(task)

            if best_agent:
                assignment_result = await self._assign_task_optimally(task, best_agent)
                assignment_results.append(assignment_result)
            else:
                assignment_results.append(
                    {
                        "task_id": task.get("id", "unknown"),
                        "status": "failed",
                        "reason": "no_available_agent",
                    }
                )

        return {
            "total_tasks": len(tasks),
            "system_load_at_distribution": current_system_load,
            "assignments": assignment_results,
        }

    def _get_real_system_load(self) -> float:
        """Obtener carga real del sistema"""
        if psutil:
            return psutil.cpu_percent(interval=0.1)
        return 0.0

    async def _find_optimal_agent(self, task: Dict[str, Any]) -> Optional[str]:
        """Encontrar el agente óptimo basándose en especialización y carga"""
        task_type = task.get("type", "unknown")
        
        # Mapeo de tareas a especialidades
        target_specialty = None
        if "analysis" in task_type or "data" in task_type:
            target_specialty = "data_analysis"
        elif "maintenance" in task_type or "system" in task_type:
            target_specialty = "system_maintenance"
        elif "security" in task_type:
            target_specialty = "security"
        elif "code" in task_type or "engineering" in task_type:
            target_specialty = "code_engineering"

        # Buscar el agente especializado correcto
        for agent_id, balance in self.workload_balances.items():
            # En un sistema real, verificaríamos si el agente está vivo
            if target_specialty and target_specialty in agent_id:
                return agent_id

        # Fallback: devolver cualquier agente disponible con capacidad
        return list(self.workload_balances.keys())[0] if self.workload_balances else None

    async def _assign_task_optimally(
        self, task: Dict[str, Any], agent_id: str
    ) -> Dict[str, Any]:
        """Asignar tarea y registrar métricas reales"""
        try:
            # Registrar asignación
            assignment_record = {
                "task_id": task.get("id", "unknown"),
                "agent_id": agent_id,
                "assigned_at": datetime.now().isoformat(),
                "task_type": task.get("type", "unknown"),
                "system_cpu": self._get_real_system_load()
            }

            self.task_assignment_history.append(assignment_record)

            # Mantener historial limpio
            if len(self.task_assignment_history) > 1000:
                self.task_assignment_history = self.task_assignment_history[-1000:]

            return {
                "task_id": task.get("id", "unknown"),
                "status": "success",
                "assigned_to": agent_id,
                "assignment_time": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "task_id": task.get("id", "unknown"),
                "status": "failed",
                "reason": str(e),
            }

    async def rebalance_workload(self) -> Dict[str, Any]:
        """Rebalanceo real: No aplicable en monoproceso, pero monitoriza estado"""
        # En esta versión monorepo, "rebalancear" significa verificar salud del sistema
        load = self._get_real_system_load()
        status = "optimal" if load < 70 else "high_load"
        
        return {
            "rebalance_status": "monitored",
            "system_load": load,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }

    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Métricas reales del coordinador"""
        return {
            "active_agents": len(self.workload_balances),
            "total_tasks_history": len(self.task_assignment_history),
            "real_system_load": self._get_real_system_load(),
            "memory_usage": psutil.virtual_memory().percent if psutil else 0,
        }


class FunctionalSpecializedAgent(SpecializedAgent):
    """
    Agente Especializado REAL.
    Ejecuta operaciones verdaderas en el sistema de archivos y hardware.
    """

    def __init__(self, specialty: str):
        super().__init__(
            f"specialist_{specialty}",
            f"Real {specialty.replace('_', ' ').title()} Agent",
            AgentRole.SPECIALIST,
        )
        self.specialty = specialty
        self.specialized_functions = {}
        self._setup_specialized_functions()
        self.workspace_root = Path(os.getcwd())

    def _setup_specialized_functions(self):
        """Configurar funciones REALES según el tipo"""
        if self.specialty == "data_analysis":
            self.specialized_functions = {
                "analyze_file": self._analyze_file_real,
                "count_lines": self._count_lines_real,
                "scan_directory": self._scan_directory_real,
            }
        elif self.specialty == "system_maintenance":
            self.specialized_functions = {
                "cleanup_temp": self._cleanup_temp_files_real,
                "system_diagnostics": self._system_diagnostics_real,
                "backup_directory": self._backup_directory_real,
            }
        elif self.specialty == "security":
            self.specialized_functions = {
                "check_permissions": self._check_permissions_real,
                "scan_sensitive_files": self._scan_sensitive_files_real,
                "audit_processes": self._audit_processes_real,
            }
        elif self.specialty == "code_engineering":
            self.specialized_functions = {
                "read_code": self._read_code_real,
                "write_code": self._write_code_real,
                "apply_patch": self._apply_patch_real,
            }

    async def execute_specialized_task(
        self, task_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecutar tarea especializada real"""
        if task_name not in self.specialized_functions:
            return {
                "status": "error",
                "message": f"Unknown specialized task: {task_name}",
                "available_tasks": list(self.specialized_functions.keys()),
            }

        try:
            start_time = datetime.now()
            # Ejecución real de la función
            result = await self.specialized_functions[task_name](parameters)
            duration = (datetime.now() - start_time).total_seconds()

            self.update_performance_stats(duration, result.get("status") == "success")

            return {
                "status": "success",
                "task": task_name,
                "result": result,
                "duration": duration,
                "completed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "status": "error",
                "task": task_name,
                "message": str(e),
                "error_type": type(e).__name__,
                "completed_at": datetime.now().isoformat(),
            }

    # =========================================================================
    # FUNCIONES REALES: CODE ENGINEERING (AUTO-PROGRAMACIÓN)
    # =========================================================================
    async def _read_code_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Lee el contenido de un archivo de código"""
        file_path = params.get("path")
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return {
                "status": "success",
                "path": str(path),
                "content": content,
                "lines": len(content.splitlines())
            }
        except Exception as e:
            raise RuntimeError(f"Failed to read code: {e}")

    async def _write_code_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Escribe código en un archivo (CON BACKUP AUTOMÁTICO)"""
        file_path = params.get("path")
        content = params.get("content")
        force = params.get("force", False)
        
        if not file_path or content is None:
            raise ValueError("Path and content required")
            
        path = Path(file_path)
        
        # Seguridad: No escribir fuera del workspace
        if ".." in str(path) or (path.is_absolute() and not str(path).startswith(str(self.workspace_root))):
             return {"status": "error", "message": "Safety block: Cannot write outside workspace"}

        # Backup automático si el archivo existe
        backup_path = None
        if path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = path.with_suffix(f"{path.suffix}.bak_{timestamp}")
            shutil.copy2(path, backup_path)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            return {
                "status": "success",
                "path": str(path),
                "bytes_written": len(content),
                "backup_created": str(backup_path) if backup_path else None
            }
        except Exception as e:
            raise RuntimeError(f"Failed to write code: {e}")

    async def _apply_patch_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica un parche simple (reemplazo de texto)"""
        file_path = params.get("path")
        target = params.get("target")
        replacement = params.get("replacement")
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError("File not found")
            
        # Leer contenido
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if target not in content:
            return {"status": "failed", "message": "Target text not found in file"}
            
        # Crear backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = path.with_suffix(f"{path.suffix}.bak_{timestamp}")
        shutil.copy2(path, backup_path)
        
        # Aplicar parche
        new_content = content.replace(target, replacement)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        return {
            "status": "success",
            "path": str(path),
            "backup": str(backup_path),
            "message": "Patch applied successfully"
        }

    # =========================================================================
    # FUNCIONES REALES: DATA ANALYSIS
    # =========================================================================
    async def _analyze_file_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza un archivo real del sistema"""
        file_path = params.get("path")
        if not file_path:
            raise ValueError("Path parameter is required")
            
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stats = path.stat()
        
        return {
            "status": "success",
            "filename": path.name,
            "size_bytes": stats.st_size,
            "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "is_file": path.is_file(),
            "extension": path.suffix
        }

    async def _scan_directory_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Escanea un directorio real y lista contenidos"""
        dir_path = params.get("path", ".")
        path = Path(dir_path)
        
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory: {dir_path}")

        files = []
        dirs = []
        total_size = 0

        for item in path.iterdir():
            if item.is_file():
                files.append(item.name)
                total_size += item.stat().st_size
            elif item.is_dir():
                dirs.append(item.name)

        return {
            "status": "success",
            "directory": str(path.absolute()),
            "file_count": len(files),
            "dir_count": len(dirs),
            "total_size_bytes": total_size,
            "files": files[:50],  # Limitado para seguridad
            "subdirectories": dirs[:50]
        }

    async def _count_lines_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cuenta líneas en un archivo de texto real"""
        file_path = params.get("path")
        path = Path(file_path)
        
        if not path.exists() or not path.is_file():
            raise ValueError("Invalid file path")

        try:
            count = 0
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in f:
                    count += 1
            
            return {
                "status": "success",
                "file": path.name,
                "line_count": count
            }
        except Exception as e:
            raise RuntimeError(f"Error reading file: {e}")

    # =========================================================================
    # FUNCIONES REALES: SYSTEM MAINTENANCE
    # =========================================================================
    async def _system_diagnostics_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene diagnósticos reales del hardware"""
        if not psutil:
            return {"status": "error", "message": "psutil not installed"}

        return {
            "status": "success",
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "cpu_cores": psutil.cpu_count(),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "usage_percent": psutil.disk_usage('/').percent,
                "free_space": psutil.disk_usage('/').free
            },
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "platform": platform.system(),
            "platform_release": platform.release()
        }

    async def _cleanup_temp_files_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Limpia archivos temporales reales (con seguridad)"""
        target_dir = params.get("path", "./tmp")
        path = Path(target_dir)
        
        # Seguridad: No permitir borrar fuera del workspace o directorios críticos
        if ".." in str(path) or path.is_absolute() and not str(path).startswith(str(self.workspace_root)):
             # Permitir si está explícitamente en una lista blanca, si no, bloquear
             if not str(path).endswith("tmp") and not str(path).endswith("logs"):
                 return {"status": "error", "message": "Safety block: Cannot clean protected paths"}

        if not path.exists():
            return {"status": "skipped", "message": "Directory does not exist"}

        deleted_count = 0
        freed_bytes = 0

        try:
            for item in path.iterdir():
                if item.is_file() and item.suffix in ['.tmp', '.log', '.cache']:
                    size = item.stat().st_size
                    item.unlink()
                    deleted_count += 1
                    freed_bytes += size
            
            return {
                "status": "success",
                "deleted_files": deleted_count,
                "freed_bytes": freed_bytes,
                "target": str(path)
            }
        except Exception as e:
            return {"status": "partial_error", "message": str(e)}

    async def _backup_directory_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza un backup real comprimido de un directorio"""
        source = params.get("source")
        dest = params.get("dest", "./backups")
        
        if not source:
            raise ValueError("Source directory required")

        source_path = Path(source)
        dest_path = Path(dest)
        
        if not source_path.exists():
            raise FileNotFoundError("Source does not exist")

        # Crear directorio de destino si no existe
        dest_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_name = f"backup_{source_path.name}_{timestamp}"
        
        try:
            # Crear archivo zip real
            archive_path = shutil.make_archive(
                str(dest_path / archive_name),
                'zip',
                str(source_path)
            )
            
            return {
                "status": "success",
                "archive_path": archive_path,
                "size_bytes": os.path.getsize(archive_path),
                "source": str(source_path)
            }
        except Exception as e:
            raise RuntimeError(f"Backup failed: {e}")

    # =========================================================================
    # FUNCIONES REALES: SECURITY
    # =========================================================================
    async def _check_permissions_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Verifica permisos reales de un archivo"""
        path_str = params.get("path", ".")
        path = Path(path_str)
        
        if not path.exists():
            raise FileNotFoundError("Path not found")
            
        stat = path.stat()
        import stat as stat_lib
        
        return {
            "status": "success",
            "path": str(path),
            "mode_octal": oct(stat.st_mode)[-3:],
            "is_writable": os.access(path, os.W_OK),
            "is_readable": os.access(path, os.R_OK),
            "is_executable": os.access(path, os.X_OK),
            "owner_uid": stat.st_uid
        }

    async def _audit_processes_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Audita procesos reales en ejecución"""
        if not psutil:
            return {"status": "error", "message": "psutil required"}
            
        limit = params.get("limit", 10)
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_percent']):
            try:
                pinfo = proc.info
                processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        # Ordenar por uso de memoria
        processes.sort(key=lambda x: x['memory_percent'] or 0, reverse=True)
        
        return {
            "status": "success",
            "process_count": len(processes),
            "top_memory_consumers": processes[:limit]
        }

    async def _scan_sensitive_files_real(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Escanea en busca de archivos sensibles (claves, envs)"""
        root_dir = params.get("path", ".")
        sensitive_patterns = ['.env', '.pem', '.key', 'id_rsa', 'secrets']
        found_files = []
        
        path = Path(root_dir)
        
        # Escaneo superficial para evitar tardar demasiado
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    if any(pattern in item.name for pattern in sensitive_patterns):
                        found_files.append({
                            "name": item.name,
                            "path": str(item),
                            "size": item.stat().st_size
                        })
                if len(found_files) > 20: # Límite de seguridad
                    break
        except Exception as e:
            pass
            
        return {
            "status": "success",
            "sensitive_files_found": len(found_files),
            "files": found_files,
            "recommendation": "Ensure these files are in .gitignore" if found_files else "Clean"
        }


class FunctionalMultiAgentSystem(MultiAgentSystem):
    """Sistema multi-agente que coordina y ejecuta trabajo REAL"""

    def __init__(self):
        super().__init__()
        self.functional_coordinator = FunctionalCoordinatorAgent()
        self.specialized_agents: Dict[str, FunctionalSpecializedAgent] = {}
        self.system_metrics = {}
        self._initialize_specialized_agents()

    def _initialize_specialized_agents(self):
        """Inicializar agentes especializados reales"""
        specialties = ["data_analysis", "system_maintenance", "security", "code_engineering"]

        for specialty in specialties:
            agent = FunctionalSpecializedAgent(specialty)
            self.specialized_agents[specialty] = agent

    async def start_functional_system(self) -> Dict[str, Any]:
        """Iniciar sistema funcional completo"""
        startup_results = {}

        try:
            # Iniciar coordinador
            await self.functional_coordinator.start()
            startup_results["coordinator"] = "started"

            # Iniciar agentes especializados
            for specialty, agent in self.specialized_agents.items():
                await agent.start()
                startup_results[f"specialist_{specialty}"] = "started"

            # Configurar balances de carga iniciales
            for specialty, agent in self.specialized_agents.items():
                self.functional_coordinator.workload_balances[agent.agent_id] = (
                    WorkloadBalance(
                        agent_id=agent.agent_id,
                        current_load=0.0,
                        max_capacity=100.0, # 100% CPU
                        available_capacity=100.0,
                        last_updated=datetime.now(),
                        process_id=os.getpid()
                    )
                )

            startup_results["system_status"] = "fully_functional_real_mode"
            return startup_results

        except Exception as e:
            startup_results["error"] = str(e)
            return startup_results

    async def execute_distributed_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar tarea distribuida en el sistema"""
        task_type = task.get("type", "unknown")

        # Determinar qué agente especializado debe manejar la tarea
        specialist = None
        if "analysis" in task_type or "file" in task_type:
            specialist = self.specialized_agents.get("data_analysis")
        elif "maintenance" in task_type or "system" in task_type:
            specialist = self.specialized_agents.get("system_maintenance")
        elif "security" in task_type:
            specialist = self.specialized_agents.get("security")
        elif "code" in task_type or "engineering" in task_type:
            specialist = self.specialized_agents.get("code_engineering")

        if specialist:
            # Ejecutar tarea especializada
            return await specialist.execute_specialized_task(
                task.get("function", ""), task.get("parameters", {})
            )
        else:
            return {
                "status": "error", 
                "message": f"No specialist found for task type: {task_type}",
                "available_agents": list(self.specialized_agents.keys())
            }

    async def get_system_overview(self) -> Dict[str, Any]:
        """Obtener vista general del sistema funcional"""
        coordinator_metrics = self.functional_coordinator.get_coordination_metrics()

        specialist_status = {}
        for specialty, agent in self.specialized_agents.items():
            specialist_status[specialty] = {
                "agent_id": agent.agent_id,
                "is_active": agent.is_active,
                "available_functions": list(agent.specialized_functions.keys()),
                "performance_stats": agent.performance_stats,
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": "operational_real",
            "coordinator_metrics": coordinator_metrics,
            "specialists": specialist_status,
            "total_agents": len(self.specialized_agents) + 1,
        }


# CONVERTIR las clases base en sistema funcional
functional_coordinator = FunctionalCoordinatorAgent()
functional_multi_agent_system = FunctionalMultiAgentSystem()


# Funciones de utilidad
async def start_functional_coordination_system():
    """Iniciar sistema de coordinación funcional"""
    return await functional_multi_agent_system.start_functional_system()


async def get_coordination_system_status():
    """Obtener estado del sistema de coordinación"""
    return await functional_multi_agent_system.get_system_overview()


async def execute_coordinated_task(task: Dict[str, Any]):
    """Ejecutar tarea coordinada"""
    return await functional_multi_agent_system.execute_distributed_task(task)
