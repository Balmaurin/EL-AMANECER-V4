#!/usr/bin/env python3
"""
Dynamic Configuration Manager - Gestión Dinámica de Configuración MCP

Este módulo implementa un sistema avanzado de configuración dinámica
con hot-reload para el servidor MCP empresarial, permitiendo cambios
de configuración en tiempo real sin reiniciar el sistema.
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConfigSection:
    """Sección de configuración con metadata"""

    name: str
    data: Dict[str, Any]
    version: str = "1.0"
    last_modified: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    validators: List[Callable] = field(default_factory=list)
    hot_reload_enabled: bool = True


@dataclass
class ConfigChange:
    """Cambio de configuración registrado"""

    section: str
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "api"  # api, file, env
    user: Optional[str] = None


class DynamicConfigManager:
    """
    Gestor de configuración dinámica con hot-reload MCP.

    Permite modificar configuración en tiempo real y gestionar
    cambios de forma segura con validación y rollback.
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.configs: Dict[str, ConfigSection] = {}
        self.change_history: List[ConfigChange] = []
        self.watchers: Dict[str, List[Callable]] = {}

        # Hot reload
        self.file_monitoring = True
        self.monitor_thread = None
        self.file_check_interval = 5  # segundos
        self.last_file_checksums: Dict[str, str] = {}

        # MCP control
        self.mcp_enabled = True
        self.config_backups: Dict[str, Dict[str, Any]] = {}

        # Inicializar configuración por defecto
        self._initialize_default_config()

        logger.info("DynamicConfigManager inicializado")

    def _initialize_default_config(self):
        """Inicializar configuración por defecto"""
        default_configs = {
            "server": {
                "host": "0.0.0.0",
                "port": 8006,
                "workers": 4,
                "timeout": 30,
                "cors_origins": ["*"],
                "log_level": "INFO",
            },
            "agents": {
                "max_agents": 50,
                "coordination_interval": 10,
                "auto_restart": True,
                "learning_enabled": True,
                "emergency_response_time": 5,
            },
            "performance": {
                "monitoring_enabled": True,
                "collection_interval": 5,
                "cpu_warning_threshold": 70.0,
                "cpu_critical_threshold": 90.0,
                "memory_warning_threshold": 75.0,
                "memory_critical_threshold": 90.0,
                "auto_gc_enabled": True,
                "optimization_enabled": True,
            },
            "security": {
                "auth_enabled": True,
                "session_timeout": 3600,
                "max_login_attempts": 5,
                "password_min_length": 8,
                "encryption_enabled": True,
            },
            "cache": {
                "enabled": True,
                "max_size": 1000,
                "ttl": 3600,
                "eviction_policy": "lru",
                "compression_enabled": True,
            },
            "database": {
                "type": "sqlite",
                "path": "data/sheily.db",
                "pool_size": 10,
                "timeout": 30,
                "backup_enabled": True,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_enabled": True,
                "file_path": "logs/sheily.log",
                "max_file_size": 10485760,  # 10MB
                "backup_count": 5,
            },
        }

        for section_name, config_data in default_configs.items():
            self.configs[section_name] = ConfigSection(
                name=section_name,
                data=config_data.copy(),
                checksum=self._calculate_checksum(config_data),
            )

    async def start_file_monitoring(self) -> bool:
        """Iniciar monitoreo de archivos de configuración"""
        try:
            if not self.file_monitoring:
                return True

            # Cargar configuraciones existentes desde archivos
            await self._load_config_files()

            # Calcular checksums iniciales
            self._update_file_checksums()

            # Iniciar thread de monitoreo
            self.monitor_thread = threading.Thread(
                target=self._file_monitor_loop, daemon=True
            )
            self.monitor_thread.start()

            logger.info("Monitoreo de archivos de configuración iniciado")
            return True

        except Exception as e:
            logger.error(f"Error iniciando monitoreo de archivos: {e}")
            return False

    async def stop_file_monitoring(self) -> bool:
        """Detener monitoreo de archivos de configuración"""
        try:
            self.file_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)

            logger.info("Monitoreo de archivos de configuración detenido")
            return True

        except Exception as e:
            logger.error(f"Error deteniendo monitoreo de archivos: {e}")
            return False

    def _file_monitor_loop(self):
        """Loop de monitoreo de archivos"""
        while self.file_monitoring:
            try:
                self._check_file_changes()
                time.sleep(self.file_check_interval)
            except Exception as e:
                logger.error(f"Error en loop de monitoreo de archivos: {e}")
                time.sleep(1)

    def _check_file_changes(self):
        """Verificar cambios en archivos de configuración"""
        for config_file in self.config_dir.glob("*.json"):
            try:
                current_checksum = self._calculate_file_checksum(config_file)
                last_checksum = self.last_file_checksums.get(str(config_file))

                if current_checksum != last_checksum:
                    logger.info(f"Archivo de configuración cambiado: {config_file}")
                    asyncio.run(self._reload_config_file(config_file))
                    self.last_file_checksums[str(config_file)] = current_checksum

            except Exception as e:
                logger.error(f"Error verificando cambios en {config_file}: {e}")

    async def _reload_config_file(self, config_file: Path):
        """Recargar archivo de configuración"""
        try:
            section_name = config_file.stem

            with open(config_file, "r", encoding="utf-8") as f:
                new_config = json.load(f)

            # Validar configuración
            if await self._validate_config(section_name, new_config):
                # Crear backup
                if section_name in self.configs:
                    self.config_backups[section_name] = self.configs[
                        section_name
                    ].data.copy()

                # Actualizar configuración
                old_data = self.configs[section_name].data.copy()
                self.configs[section_name].data = new_config
                self.configs[section_name].last_modified = datetime.now()
                self.configs[section_name].checksum = self._calculate_checksum(
                    new_config
                )

                # Registrar cambios
                await self._register_config_changes(
                    section_name, old_data, new_config, "file"
                )

                # Notificar watchers
                await self._notify_watchers(section_name, old_data, new_config)

                logger.info(f"Configuración recargada desde archivo: {section_name}")

        except Exception as e:
            logger.error(f"Error recargando configuración desde {config_file}: {e}")

    async def _load_config_files(self):
        """Cargar configuraciones desde archivos"""
        for config_file in self.config_dir.glob("*.json"):
            try:
                await self._reload_config_file(config_file)
            except Exception as e:
                logger.warning(f"Error cargando configuración desde {config_file}: {e}")

    def _update_file_checksums(self):
        """Actualizar checksums de archivos"""
        for config_file in self.config_dir.glob("*.json"):
            try:
                self.last_file_checksums[str(config_file)] = (
                    self._calculate_file_checksum(config_file)
                )
            except Exception as e:
                logger.error(f"Error calculando checksum para {config_file}: {e}")

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calcular checksum de archivo"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calcular checksum de datos de configuración"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    # ========================================
    # MCP CONTROL ENDPOINTS
    # ========================================

    async def get_config_section(self, section_name: str) -> Dict[str, Any]:
        """Obtener sección de configuración vía MCP"""
        try:
            if section_name not in self.configs:
                return {
                    "error": f"Sección de configuración no encontrada: {section_name}"
                }

            section = self.configs[section_name]
            return {
                "section": section_name,
                "config": section.data,
                "version": section.version,
                "last_modified": section.last_modified.isoformat(),
                "checksum": section.checksum,
                "hot_reload_enabled": section.hot_reload_enabled,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(
                f"Error obteniendo sección de configuración {section_name}: {e}"
            )
            return {"error": f"Error interno del servidor: {str(e)}"}

    async def update_config_section(
        self, section_name: str, new_config: Dict[str, Any], user: Optional[str] = None
    ) -> Dict[str, Any]:
        """Actualizar sección de configuración vía MCP"""
        try:
            if section_name not in self.configs:
                return {
                    "error": f"Sección de configuración no encontrada: {section_name}"
                }

            # Validar configuración
            if not await self._validate_config(section_name, new_config):
                return {"error": "Configuración no válida"}

            # Crear backup
            section = self.configs[section_name]
            old_data = section.data.copy()
            self.config_backups[section_name] = old_data

            # Actualizar configuración
            section.data = new_config
            section.last_modified = datetime.now()
            section.checksum = self._calculate_checksum(new_config)

            # Registrar cambios
            await self._register_config_changes(
                section_name, old_data, new_config, "api", user
            )

            # Notificar watchers
            await self._notify_watchers(section_name, old_data, new_config)

            # Guardar en archivo si existe
            await self._save_config_to_file(section_name)

            logger.info(
                f"Configuración actualizada vía MCP: {section_name} por {user or 'unknown'}"
            )
            return {
                "success": True,
                "section": section_name,
                "message": "Configuración actualizada correctamente",
                "changes_count": len(
                    [c for c in self.change_history[-100:] if c.section == section_name]
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error actualizando configuración {section_name}: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}

    async def get_all_config(self) -> Dict[str, Any]:
        """Obtener toda la configuración vía MCP"""
        try:
            config_summary = {}
            for section_name, section in self.configs.items():
                config_summary[section_name] = {
                    "version": section.version,
                    "last_modified": section.last_modified.isoformat(),
                    "keys_count": len(section.data),
                    "checksum": section.checksum[:8],  # Solo primeros 8 caracteres
                }

            return {
                "config_sections": list(self.configs.keys()),
                "config_summary": config_summary,
                "total_sections": len(self.configs),
                "last_change": max(
                    (s.last_modified for s in self.configs.values()),
                    default=datetime.now(),
                ).isoformat(),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error obteniendo configuración completa: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}

    async def rollback_config(
        self, section_name: str, user: Optional[str] = None
    ) -> Dict[str, Any]:
        """Revertir configuración a backup anterior vía MCP"""
        try:
            if section_name not in self.config_backups:
                return {
                    "error": f"No hay backup disponible para la sección: {section_name}"
                }

            if section_name not in self.configs:
                return {
                    "error": f"Sección de configuración no encontrada: {section_name}"
                }

            # Obtener backup
            backup_data = self.config_backups[section_name]
            current_data = self.configs[section_name].data.copy()

            # Restaurar backup
            self.configs[section_name].data = backup_data
            self.configs[section_name].last_modified = datetime.now()
            self.configs[section_name].checksum = self._calculate_checksum(backup_data)

            # Registrar rollback
            rollback_change = ConfigChange(
                section=section_name,
                key="ROLLBACK",
                old_value=current_data,
                new_value=backup_data,
                source="api",
                user=user,
            )
            self.change_history.append(rollback_change)

            # Notificar watchers
            await self._notify_watchers(section_name, current_data, backup_data)

            # Guardar en archivo
            await self._save_config_to_file(section_name)

            logger.info(
                f"Configuración revertida vía MCP: {section_name} por {user or 'unknown'}"
            )
            return {
                "success": True,
                "section": section_name,
                "message": "Configuración revertida al backup anterior",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error revirtiendo configuración {section_name}: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}

    async def get_config_history(
        self, section_name: Optional[str] = None, limit: int = 50
    ) -> Dict[str, Any]:
        """Obtener historial de cambios de configuración vía MCP"""
        try:
            # Filtrar cambios por sección si se especifica
            if section_name:
                changes = [c for c in self.change_history if c.section == section_name]
            else:
                changes = self.change_history

            # Obtener cambios más recientes
            recent_changes = changes[-limit:]

            # Formatear para respuesta
            formatted_changes = []
            for change in recent_changes:
                formatted_changes.append(
                    {
                        "section": change.section,
                        "key": change.key,
                        "old_value": (
                            str(change.old_value)[:100] + "..."
                            if len(str(change.old_value)) > 100
                            else change.old_value
                        ),
                        "new_value": (
                            str(change.new_value)[:100] + "..."
                            if len(str(change.new_value)) > 100
                            else change.new_value
                        ),
                        "timestamp": change.timestamp.isoformat(),
                        "source": change.source,
                        "user": change.user,
                    }
                )

            return {
                "changes": formatted_changes,
                "total_changes": len(changes),
                "returned_changes": len(formatted_changes),
                "section_filter": section_name,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error obteniendo historial de configuración: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}

    async def validate_config(
        self, section_name: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validar configuración vía MCP"""
        try:
            is_valid, errors = await self._validate_config_detailed(
                section_name, config
            )

            return {
                "section": section_name,
                "valid": is_valid,
                "errors": errors,
                "warnings": [],  # Podríamos añadir validaciones de warning
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error validando configuración {section_name}: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}

    async def reload_config_files(self) -> Dict[str, Any]:
        """Recargar todos los archivos de configuración vía MCP"""
        try:
            reloaded_sections = []

            for config_file in self.config_dir.glob("*.json"):
                try:
                    await self._reload_config_file(config_file)
                    reloaded_sections.append(config_file.stem)
                except Exception as e:
                    logger.warning(f"Error recargando {config_file}: {e}")

            return {
                "success": True,
                "reloaded_sections": reloaded_sections,
                "total_reloaded": len(reloaded_sections),
                "message": f"Recargados {len(reloaded_sections)} archivos de configuración",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error recargando archivos de configuración: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}

    # ========================================
    # HELPERS Y UTILIDADES
    # ========================================

    async def _validate_config(self, section_name: str, config: Dict[str, Any]) -> bool:
        """Validar configuración básica"""
        try:
            is_valid, _ = await self._validate_config_detailed(section_name, config)
            return is_valid
        except Exception:
            return False

    async def _validate_config_detailed(
        self, section_name: str, config: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Validar configuración con detalle"""
        errors = []

        try:
            # Validaciones específicas por sección
            if section_name == "server":
                if not isinstance(config.get("port"), int) or not (
                    1000 <= config["port"] <= 65535
                ):
                    errors.append("Puerto debe ser un entero entre 1000 y 65535")

                if config.get("workers", 0) < 1:
                    errors.append("Número de workers debe ser al menos 1")

            elif section_name == "performance":
                thresholds = [
                    "cpu_warning_threshold",
                    "cpu_critical_threshold",
                    "memory_warning_threshold",
                    "memory_critical_threshold",
                ]

                for threshold in thresholds:
                    value = config.get(threshold)
                    if not isinstance(value, (int, float)) or not (0 <= value <= 100):
                        errors.append(f"{threshold} debe ser un número entre 0 y 100")

            elif section_name == "security":
                if config.get("password_min_length", 0) < 4:
                    errors.append("Longitud mínima de contraseña debe ser al menos 4")

                if config.get("max_login_attempts", 0) < 1:
                    errors.append("Máximo de intentos de login debe ser al menos 1")

            # Validaciones generales
            if not isinstance(config, dict):
                errors.append("Configuración debe ser un objeto JSON válido")

            # Ejecutar validadores personalizados
            if section_name in self.configs:
                for validator in self.configs[section_name].validators:
                    try:
                        validator_result = validator(config)
                        if validator_result:
                            errors.extend(validator_result)
                    except Exception as e:
                        errors.append(f"Error en validador personalizado: {str(e)}")

        except Exception as e:
            errors.append(f"Error general de validación: {str(e)}")

        return len(errors) == 0, errors

    async def _register_config_changes(
        self,
        section_name: str,
        old_data: Dict[str, Any],
        new_data: Dict[str, Any],
        source: str,
        user: Optional[str] = None,
    ):
        """Registrar cambios de configuración"""
        try:
            # Encontrar diferencias
            changes = []
            all_keys = set(old_data.keys()) | set(new_data.keys())

            for key in all_keys:
                old_value = old_data.get(key)
                new_value = new_data.get(key)

                if old_value != new_value:
                    changes.append(
                        ConfigChange(
                            section=section_name,
                            key=key,
                            old_value=old_value,
                            new_value=new_value,
                            source=source,
                            user=user,
                        )
                    )

            self.change_history.extend(changes)

            # Mantener límite de historial
            if len(self.change_history) > 1000:
                self.change_history = self.change_history[-1000:]

        except Exception as e:
            logger.error(f"Error registrando cambios de configuración: {e}")

    async def _notify_watchers(
        self, section_name: str, old_data: Dict[str, Any], new_data: Dict[str, Any]
    ):
        """Notificar watchers de cambios de configuración"""
        try:
            if section_name in self.watchers:
                for watcher in self.watchers[section_name]:
                    try:
                        await watcher(section_name, old_data, new_data)
                    except Exception as e:
                        logger.error(f"Error notificando watcher: {e}")
        except Exception as e:
            logger.error(f"Error notificando watchers para {section_name}: {e}")

    async def _save_config_to_file(self, section_name: str):
        """Guardar configuración en archivo"""
        try:
            if not self.file_monitoring:
                return

            config_file = self.config_dir / f"{section_name}.json"
            section = self.configs[section_name]

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(section.data, f, indent=2, ensure_ascii=False)

            # Actualizar checksum
            self.last_file_checksums[str(config_file)] = self._calculate_file_checksum(
                config_file
            )

        except Exception as e:
            logger.error(
                f"Error guardando configuración en archivo {section_name}: {e}"
            )

    def add_config_watcher(self, section_name: str, watcher: Callable):
        """Añadir watcher para cambios de configuración"""
        if section_name not in self.watchers:
            self.watchers[section_name] = []
        self.watchers[section_name].append(watcher)

    def remove_config_watcher(self, section_name: str, watcher: Callable):
        """Remover watcher de cambios de configuración"""
        if section_name in self.watchers:
            try:
                self.watchers[section_name].remove(watcher)
            except ValueError:
                pass

    async def get_config_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de configuración"""
        try:
            total_sections = len(self.configs)
            total_keys = sum(len(section.data) for section in self.configs.values())
            total_changes = len(self.change_history)

            sections_stats = {}
            for name, section in self.configs.items():
                sections_stats[name] = {
                    "keys_count": len(section.data),
                    "last_modified": section.last_modified.isoformat(),
                    "has_backup": name in self.config_backups,
                    "hot_reload_enabled": section.hot_reload_enabled,
                }

            return {
                "total_sections": total_sections,
                "total_config_keys": total_keys,
                "total_changes": total_changes,
                "file_monitoring_active": self.file_monitoring,
                "sections_stats": sections_stats,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de configuración: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}


# Instancia global del gestor de configuración dinámica
_dynamic_config_manager: Optional[DynamicConfigManager] = None


async def get_dynamic_config_manager() -> DynamicConfigManager:
    """Obtener instancia del gestor de configuración dinámica"""
    global _dynamic_config_manager

    if _dynamic_config_manager is None:
        _dynamic_config_manager = DynamicConfigManager()
        await _dynamic_config_manager.start_file_monitoring()

    return _dynamic_config_manager


async def cleanup_dynamic_config_manager():
    """Limpiar el gestor de configuración dinámica"""
    global _dynamic_config_manager

    if _dynamic_config_manager:
        await _dynamic_config_manager.stop_file_monitoring()
        _dynamic_config_manager = None
