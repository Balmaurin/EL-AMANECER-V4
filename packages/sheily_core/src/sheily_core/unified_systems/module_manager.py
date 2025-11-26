"""
🎯 MODULE MANAGEMENT SYSTEM - SHEILY AI UNIFIED SYSTEM
============================================================
Sistema unificado de gestión de módulos consolidado

Este módulo consolida 7 archivos separados en una arquitectura cohesiva:
- module_registry.py → ModuleRegistry
- module_scanner.py → ModuleScanner  
- module_initializer.py → ModuleInitializer
- module_integrator.py → (merged con ModuleInitializer)
- module_validator.py → ModuleValidator
- module_monitor.py → ModuleMonitor
- module_plugin_system.py → ModulePluginSystem

CARACTERÍSTICAS:
- Registro centralizado de módulos
- Descubrimiento automático de módulos
- Inicialización ordenada con dependencias
- Validación de salud y conformidad
- Monitoreo de rendimiento en tiempo real
- Sistema de plugins extensible

AUTORES: Sheily AI Team - Arquitectura Unificada v2.0
FECHA: 2025
"""

from __future__ import annotations
import asyncio
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================================================
# EXCEPTIONS - CUSTOM ERRORS
# ========================================================================
class ModuleManagementError(Exception):
    """Error base de gestión de módulos"""
    pass

class ModuleInitializationError(ModuleManagementError):
    """Error durante inicialización de módulo"""
    pass

class ModuleValidationError(ModuleManagementError):
    """Error durante validación de módulo"""
    pass

class ModuleDependencyError(ModuleManagementError):
    """Error de dependencias de módulo"""
    pass

# ========================================================================
# DATA STRUCTURES
# ========================================================================
@dataclass
class ModuleMetadata:
    """Metadatos de un módulo descubierto"""
    name: str
    path: Path
    module_type: str
    version: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    author: str = ""
    entry_point: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)

@dataclass
class ModuleHealthStatus:
    """Estado de salud de un módulo"""
    module_name: str
    is_healthy: bool = True
    status: str = "unknown"  # unknown, healthy, degraded, unhealthy
    last_check: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModulePerformanceMetrics:
    """Métricas de rendimiento de módulo"""
    module_name: str
    initialization_time: float = 0.0
    average_execution_time: float = 0.0
    total_calls: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)

@dataclass
class ModuleInitializationConfig:
    """Configuración de inicialización"""
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    fail_on_error: bool = False
    parallel_initialization: bool = True
    max_parallel_workers: int = 5
    log_level: str = "INFO"

# ========================================================================
# MODULE REGISTRY - CENTRALIZED MODULE TRACKING
# ========================================================================
class ModuleRegistry:
    """
    Registro centralizado de módulos del sistema
    
    Consolida funcionalidad de module_registry.py
    """
    
    def __init__(self):
        self._modules: Dict[str, Any] = {}
        self._metadata: Dict[str, ModuleMetadata] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._reverse_dependencies: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()
        
        logger.info("📝 ModuleRegistry inicializado")
    
    def register(
        self,
        module_name: str,
        module_instance: Any,
        metadata: Optional[ModuleMetadata] = None
    ) -> bool:
        """Registrar un módulo en el registro"""
        with self._lock:
            try:
                if module_name in self._modules:
                    logger.warning(f"⚠️ Módulo '{module_name}' ya registrado, sobrescribiendo")
                
                self._modules[module_name] = module_instance
                
                if metadata:
                    self._metadata[module_name] = metadata
                    
                    # Registrar dependencias
                    if metadata.dependencies:
                        self._dependencies[module_name] = set(metadata.dependencies)
                        
                        # Actualizar dependencias inversas
                        for dep in metadata.dependencies:
                            if dep not in self._reverse_dependencies:
                                self._reverse_dependencies[dep] = set()
                            self._reverse_dependencies[dep].add(module_name)
                
                logger.info(f"✅ Módulo registrado: {module_name}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error registrando módulo '{module_name}': {e}")
                return False
    
    def unregister(self, module_name: str) -> bool:
        """Desregistrar un módulo"""
        with self._lock:
            try:
                if module_name not in self._modules:
                    logger.warning(f"⚠️ Módulo '{module_name}' no encontrado")
                    return False
                
                # Verificar dependencias inversas
                if module_name in self._reverse_dependencies:
                    dependent_modules = self._reverse_dependencies[module_name]
                    if dependent_modules:
                        logger.error(
                            f"❌ No se puede desregistrar '{module_name}', "
                            f"módulos dependientes: {dependent_modules}"
                        )
                        return False
                
                # Eliminar
                del self._modules[module_name]
                self._metadata.pop(module_name, None)
                self._dependencies.pop(module_name, None)
                
                logger.info(f"🗑️ Módulo desregistrado: {module_name}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error desregistrando módulo '{module_name}': {e}")
                return False
    
    def get(self, module_name: str) -> Optional[Any]:
        """Obtener instancia de módulo"""
        return self._modules.get(module_name)
    
    def get_metadata(self, module_name: str) -> Optional[ModuleMetadata]:
        """Obtener metadatos de módulo"""
        return self._metadata.get(module_name)
    
    def get_all_modules(self) -> Dict[str, Any]:
        """Obtener todos los módulos registrados"""
        return self._modules.copy()
    
    def get_dependencies(self, module_name: str) -> Set[str]:
        """Obtener dependencias de un módulo"""
        return self._dependencies.get(module_name, set()).copy()
    
    def get_dependents(self, module_name: str) -> Set[str]:
        """Obtener módulos que dependen de este módulo"""
        return self._reverse_dependencies.get(module_name, set()).copy()
    
    def is_registered(self, module_name: str) -> bool:
        """Verificar si un módulo está registrado"""
        return module_name in self._modules
    
    def get_initialization_order(self) -> List[str]:
        """
        Obtener orden de inicialización respetando dependencias
        Usa ordenamiento topológico
        """
        with self._lock:
            result = []
            visited = set()
            temp_mark = set()
            
            def visit(module_name: str):
                if module_name in temp_mark:
                    raise ModuleDependencyError(
                        f"Dependencia circular detectada en: {module_name}"
                    )
                
                if module_name not in visited:
                    temp_mark.add(module_name)
                    
                    # Visitar dependencias primero
                    deps = self._dependencies.get(module_name, set())
                    for dep in deps:
                        if dep in self._modules:  # Solo si está registrado
                            visit(dep)
                    
                    temp_mark.remove(module_name)
                    visited.add(module_name)
                    result.append(module_name)
            
            # Visitar todos los módulos
            for module_name in list(self._modules.keys()):
                if module_name not in visited:
                    visit(module_name)
            
            return result

# ========================================================================
# MODULE SCANNER - AUTOMATIC MODULE DISCOVERY
# ========================================================================
class ModuleScanner:
    """
    Escaneo y descubrimiento automático de módulos
    
    Consolida funcionalidad de module_scanner.py
    """
    
    def __init__(self, base_paths: Optional[List[Path]] = None):
        self.base_paths = base_paths or []
        self.discovered_modules: List[ModuleMetadata] = []
        
        logger.info("🔍 ModuleScanner inicializado")
    
    async def scan_directory(
        self,
        directory: Path,
        pattern: str = "*.py",
        recursive: bool = True
    ) -> List[ModuleMetadata]:
        """Escanear directorio en busca de módulos"""
        try:
            directory = Path(directory)
            if not directory.exists():
                logger.warning(f"⚠️ Directorio no existe: {directory}")
                return []
            
            discovered = []
            
            # Buscar archivos
            if recursive:
                files = directory.rglob(pattern)
            else:
                files = directory.glob(pattern)
            
            for file_path in files:
                if file_path.is_file() and not file_path.name.startswith('_'):
                    metadata = await self._extract_module_metadata(file_path)
                    if metadata:
                        discovered.append(metadata)
                        self.discovered_modules.append(metadata)
            
            logger.info(f"🔍 Descubiertos {len(discovered)} módulos en {directory}")
            return discovered
            
        except Exception as e:
            logger.error(f"❌ Error escaneando directorio {directory}: {e}")
            return []
    
    async def _extract_module_metadata(self, file_path: Path) -> Optional[ModuleMetadata]:
        """Extraer metadatos de un archivo de módulo"""
        try:
            # Leer el archivo
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extraer información básica
            module_name = file_path.stem
            
            metadata = ModuleMetadata(
                name=module_name,
                path=file_path,
                module_type="python_module",
                description=self._extract_docstring(content),
                dependencies=self._extract_imports(content)
            )
            
            return metadata
            
        except Exception as e:
            logger.debug(f"No se pudo extraer metadata de {file_path}: {e}")
            return None
    
    def _extract_docstring(self, content: str) -> str:
        """Extraer docstring de módulo"""
        lines = content.split('\n')
        in_docstring = False
        docstring_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                in_docstring = True
                docstring_lines.append(stripped[3:])
                if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                    break
            elif in_docstring:
                if '"""' in stripped or "'''" in stripped:
                    docstring_lines.append(stripped[:stripped.find('"""')])
                    break
                docstring_lines.append(stripped)
        
        return ' '.join(docstring_lines).strip()
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extraer imports de módulo"""
        imports = []
        lines = content.split('\n')
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                # Extraer nombre del módulo
                if stripped.startswith('import '):
                    module = stripped.split()[1].split('.')[0]
                    imports.append(module)
                elif 'import' in stripped:
                    module = stripped.split()[1].split('.')[0]
                    imports.append(module)
        
        return list(set(imports))  # Eliminar duplicados
    
    def get_discovered_modules(self) -> List[ModuleMetadata]:
        """Obtener módulos descubiertos"""
        return self.discovered_modules.copy()

# ========================================================================
# MODULE INITIALIZER - MODULE LIFECYCLE MANAGEMENT
# ========================================================================
class ModuleInitializer:
    """
    Inicialización ordenada de módulos
    
    Consolida funcionalidad de module_initializer.py y module_integrator.py
    """
    
    def __init__(
        self,
        registry: ModuleRegistry,
        config: Optional[ModuleInitializationConfig] = None
    ):
        self.registry = registry
        self.config = config or ModuleInitializationConfig()
        self.initialized_modules: Set[str] = set()
        
        logger.info("🚀 ModuleInitializer inicializado")
    
    async def initialize_module(
        self,
        module_name: str,
        module_class: Type,
        init_args: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Inicializar un módulo individual"""
        try:
            logger.info(f"🔄 Inicializando módulo: {module_name}")
            
            # Verificar dependencias
            metadata = self.registry.get_metadata(module_name)
            if metadata and metadata.dependencies:
                for dep in metadata.dependencies:
                    if not self.registry.is_registered(dep):
                        raise ModuleDependencyError(
                            f"Dependencia faltante: {dep} para {module_name}"
                        )
            
            # Crear instancia
            start_time = time.time()
            init_args = init_args or {}
            
            if asyncio.iscoroutinefunction(module_class.__init__):
                instance = await module_class(**init_args)
            else:
                instance = module_class(**init_args)
            
            initialization_time = time.time() - start_time
            
            # Registrar
            self.registry.register(module_name, instance, metadata)
            self.initialized_modules.add(module_name)
            
            logger.info(
                f"✅ Módulo inicializado: {module_name} "
                f"({initialization_time:.2f}s)"
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando módulo '{module_name}': {e}")
            logger.debug(traceback.format_exc())
            
            if self.config.fail_on_error:
                raise ModuleInitializationError(f"Fallo en {module_name}: {e}")
            
            return False
    
    async def initialize_all(
        self,
        modules: Dict[str, Tuple[Type, Dict[str, Any]]]
    ) -> Dict[str, bool]:
        """
        Inicializar múltiples módulos en orden correcto
        
        Args:
            modules: Dict[module_name] = (module_class, init_args)
        """
        try:
            # Obtener orden de inicialización
            init_order = self.registry.get_initialization_order()
            
            results = {}
            
            if self.config.parallel_initialization:
                # Inicialización paralela (respetando dependencias por bloques)
                tasks = []
                for module_name in init_order:
                    if module_name in modules:
                        module_class, init_args = modules[module_name]
                        task = self.initialize_module(module_name, module_class, init_args)
                        tasks.append((module_name, task))
                
                for module_name, task in tasks:
                    result = await task
                    results[module_name] = result
            else:
                # Inicialización secuencial
                for module_name in init_order:
                    if module_name in modules:
                        module_class, init_args = modules[module_name]
                        result = await self.initialize_module(
                            module_name, module_class, init_args
                        )
                        results[module_name] = result
            
            success_count = sum(1 for v in results.values() if v)
            logger.info(
                f"🎉 Inicialización completada: "
                f"{success_count}/{len(results)} módulos exitosos"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en inicialización masiva: {e}")
            return {}

# ========================================================================
# MODULE VALIDATOR - HEALTH CHECKS & VALIDATION
# ========================================================================
class ModuleValidator:
    """
    Validación de módulos y health checks
    
    Consolida funcionalidad de module_validator.py
    """
    
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.health_status: Dict[str, ModuleHealthStatus] = {}
        
        logger.info("🏥 ModuleValidator inicializado")
    
    async def validate_module(self, module_name: str) -> ModuleHealthStatus:
        """Validar salud de un módulo"""
        try:
            status = ModuleHealthStatus(module_name=module_name)
            
            # Verificar que existe
            module = self.registry.get(module_name)
            if not module:
                status.is_healthy = False
                status.status = "unhealthy"
                status.errors.append("Módulo no encontrado en registro")
                return status
            
            # Verificar dependencias
            deps = self.registry.get_dependencies(module_name)
            for dep in deps:
                if not self.registry.is_registered(dep):
                    status.warnings.append(f"Dependencia faltante: {dep}")
                    status.status = "degraded"
            
            # Health check personalizado si existe
            if hasattr(module, 'health_check'):
                try:
                    if asyncio.iscoroutinefunction(module.health_check):
                        health_result = await module.health_check()
                    else:
                        health_result = module.health_check()
                    
                    if not health_result:
                        status.is_healthy = False
                        status.status = "unhealthy"
                        status.errors.append("Health check falló")
                except Exception as e:
                    status.warnings.append(f"Health check error: {e}")
            
            # Determinar status final
            if not status.errors:
                if not status.warnings:
                    status.status = "healthy"
                else:
                    status.status = "degraded"
            else:
                status.status = "unhealthy"
                status.is_healthy = False
            
            status.last_check = datetime.now()
            self.health_status[module_name] = status
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error validando módulo '{module_name}': {e}")
            status = ModuleHealthStatus(module_name=module_name)
            status.is_healthy = False
            status.status = "unhealthy"
            status.errors.append(str(e))
            return status
    
    async def validate_all(self) -> Dict[str, ModuleHealthStatus]:
        """Validar todos los módulos registrados"""
        results = {}
        modules = self.registry.get_all_modules()
        
        for module_name in modules.keys():
            status = await self.validate_module(module_name)
            results[module_name] = status
        
        healthy_count = sum(1 for s in results.values() if s.is_healthy)
        logger.info(
            f"🏥 Validación completada: "
            f"{healthy_count}/{len(results)} módulos saludables"
        )
        
        return results

# ========================================================================
# MODULE MONITOR - PERFORMANCE TRACKING
# ========================================================================
class ModuleMonitor:
    """
    Monitoreo de rendimiento de módulos
    
    Consolida funcionalidad de module_monitor.py
    """
    
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.metrics: Dict[str, ModulePerformanceMetrics] = {}
        self._monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info("📊 ModuleMonitor inicializado")
    
    def record_execution(
        self,
        module_name: str,
        execution_time: float,
        success: bool = True
    ):
        """Registrar ejecución de módulo"""
        if module_name not in self.metrics:
            self.metrics[module_name] = ModulePerformanceMetrics(module_name=module_name)
        
        metrics = self.metrics[module_name]
        metrics.total_calls += 1
        
        # Actualizar tiempo promedio
        metrics.average_execution_time = (
            (metrics.average_execution_time * (metrics.total_calls - 1) + execution_time)
            / metrics.total_calls
        )
        
        if not success:
            metrics.error_count += 1
    
    def get_metrics(self, module_name: str) -> Optional[ModulePerformanceMetrics]:
        """Obtener métricas de un módulo"""
        return self.metrics.get(module_name)
    
    def get_all_metrics(self) -> Dict[str, ModulePerformanceMetrics]:
        """Obtener métricas de todos los módulos"""
        return self.metrics.copy()
    
    async def start_monitoring(self, interval_seconds: float = 60.0):
        """Iniciar monitoreo continuo"""
        self._monitoring_active = True
        
        async def monitor_loop():
            while self._monitoring_active:
                await asyncio.sleep(interval_seconds)
                
                # Actualizar métricas
                modules = self.registry.get_all_modules()
                for module_name in modules.keys():
                    # Aquí se podrían agregar métricas de sistema
                    # (CPU, memoria, etc.) si están disponibles
                    pass
        
        self._monitor_task = asyncio.create_task(monitor_loop())
        logger.info("📊 Monitoreo continuo iniciado")
    
    async def stop_monitoring(self):
        """Detener monitoreo continuo"""
        self._monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("📊 Monitoreo continuo detenido")

# ========================================================================
# MODULE PLUGIN SYSTEM - EXTENSIBILITY
# ========================================================================
class ModulePluginBase(ABC):
    """Clase base para plugins de módulo"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Inicializar plugin"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Ejecutar funcionalidad del plugin"""
        pass

class ModulePluginSystem:
    """
    Sistema de plugins para extensibilidad
    
    Consolida funcionalidad de module_plugin_system.py
    """
    
    def __init__(self):
        self.registered_plugins: Dict[str, Type[ModulePluginBase]] = {}
        self.active_plugins: Dict[str, ModulePluginBase] = {}
        
        logger.info("🔌 ModulePluginSystem inicializado")
    
    def register_plugin(
        self,
        plugin_name: str,
        plugin_class: Type[ModulePluginBase]
    ) -> bool:
        """Registrar un plugin"""
        try:
            if plugin_name in self.registered_plugins:
                logger.warning(f"⚠️ Plugin '{plugin_name}' ya registrado")
                return False
            
            self.registered_plugins[plugin_name] = plugin_class
            logger.info(f"🔌 Plugin registrado: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error registrando plugin '{plugin_name}': {e}")
            return False
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Cargar y activar un plugin"""
        try:
            if plugin_name not in self.registered_plugins:
                logger.error(f"❌ Plugin '{plugin_name}' no registrado")
                return False
            
            plugin_class = self.registered_plugins[plugin_name]
            plugin_instance = plugin_class()
            
            if plugin_instance.initialize():
                self.active_plugins[plugin_name] = plugin_instance
                logger.info(f"✅ Plugin cargado: {plugin_name}")
                return True
            else:
                logger.error(f"❌ Fallo al inicializar plugin '{plugin_name}'")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error cargando plugin '{plugin_name}': {e}")
            return False
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Ejecutar un plugin activo"""
        if plugin_name not in self.active_plugins:
            logger.error(f"❌ Plugin '{plugin_name}' no está activo")
            return None
        
        try:
            return self.active_plugins[plugin_name].execute(*args, **kwargs)
        except Exception as e:
            logger.error(f"❌ Error ejecutando plugin '{plugin_name}': {e}")
            return None

# ========================================================================
# UNIFIED MODULE MANAGER - FACADE
# ========================================================================
class UnifiedModuleManager:
    """
    Gestor unificado que integra todos los componentes
    
    Facade pattern para facilitar el uso
    """
    
    def __init__(self, config: Optional[ModuleInitializationConfig] = None):
        self.config = config or ModuleInitializationConfig()
        
        # Inicializar componentes
        self.registry = ModuleRegistry()
        self.scanner = ModuleScanner()
        self.initializer = ModuleInitializer(self.registry, self.config)
        self.validator = ModuleValidator(self.registry)
        self.monitor = ModuleMonitor(self.registry)
        self.plugin_system = ModulePluginSystem()
        
        logger.info("✨ UnifiedModuleManager inicializado")
    
    async def discover_and_register(
        self,
        directories: List[Path],
        auto_initialize: bool = False
    ) -> int:
        """Descubrir y registrar módulos automáticamente"""
        total_discovered = 0
        
        for directory in directories:
            modules = await self.scanner.scan_directory(directory)
            total_discovered += len(modules)
            
            for metadata in modules:
                self.registry.register(metadata.name, None, metadata)
        
        logger.info(f"🔍 Total descubiertos y registrados: {total_discovered} módulos")
        return total_discovered
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Health check de todos los módulos"""
        results = await self.validator.validate_all()
        return {name: status.is_healthy for name, status in results.items()}
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        all_modules = self.registry.get_all_modules()
        health_results = await self.validator.validate_all()
        
        healthy_count = sum(1 for s in health_results.values() if s.is_healthy)
        
        return {
            "total_modules": len(all_modules),
            "healthy_modules": healthy_count,
            "unhealthy_modules": len(all_modules) - healthy_count,
            "active_plugins": len(self.plugin_system.active_plugins),
            "monitoring_active": self.monitor._monitoring_active,
        }

# ========================================================================
# DEMO & TESTING
# ========================================================================
async def demo_module_manager():
    """Demostración del sistema unificado"""
    logger.info("=" * 70)
    logger.info("🎯 UNIFIED MODULE MANAGER - DEMO")
    logger.info("=" * 70)
    
    # Crear manager
    manager = UnifiedModuleManager()
    
    # Simular descubrimiento
    logger.info("\n📁 Descubriendo módulos...")
    discovered = await manager.discover_and_register([Path("./packages")])
    logger.info(f"✅ Descubiertos: {discovered} módulos")
    
    # Health check
    logger.info("\n🏥 Ejecutando health checks...")
    health = await manager.health_check_all()
    logger.info(f"✅ Health check completado: {health}")
    
    # Stats
    logger.info("\n📊 Estadísticas del sistema:")
    stats = await manager.get_system_stats()
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")
    
    logger.info("\n✅ Demo completado")

if __name__ == "__main__":
    asyncio.run(demo_module_manager())
