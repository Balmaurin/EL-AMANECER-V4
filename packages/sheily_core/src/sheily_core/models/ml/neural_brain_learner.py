#!/usr/bin/env python3
"""
NEURAL BRAIN LEARNER - APRENDIZAJE AUTOMÁTICO DEL PROYECTO MCP
===============================================================

Esta extensión permite que el cerebro neuronal MCP aprenda automáticamente
toda la estructura, componentes, arquitectura y datos del proyecto actual.

Funcionalidades:
- Auto-discovery de todos los archivos y módulos MCP
- Análisis profundo de dependencias y relaciones
- Aprendizaje de configuraciones críticas
- Indexación inteligente de documentación
- Memoria neuronal del proyecto completo
- Optimizaciones basadas en estructura aprendida
"""

import asyncio
import hashlib
import json
import os
import re
import shelve
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Import opcional para evitar dependencias circulares
try:
    from ...core.mcp.mcp_neural_brain import MCPNeural, canonical_task_hash

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPNeural = None

    def canonical_task_hash(task_features):
        import hashlib
        import json

        task_json = json.dumps(task_features, sort_keys=True)
        return hashlib.sha256(task_json.encode()).hexdigest()


def set_env_vars_for_deep_scan():
    """Configura variables de entorno para escaneo profundo sin límites."""
    import os

    # Configuraciones para escaneo masivo
    os.environ["NB_LEARNER_MAX_FILES"] = "0"  # Sin límite
    os.environ["NB_LEARNER_SAVE_EVERY"] = "500"  # Batch saving más grande
    os.environ["NB_LEARNER_SAVE_VERBOSE"] = "1"  # Verbose saving
    os.environ["NB_LEARNER_SKIP_DIRS"] = ""  # No skip adicional


class ProjectKnowledgeGraph:
    """Grafos de conocimiento del proyecto MCP aprendido."""

    def __init__(self, storage_path: str = "project_knowledge.db"):
        self.storage_path = storage_path
        self.modules: Dict[str, Dict] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.configurations: Dict[str, Any] = {}
        self.documentation_index: Dict[str, str] = {}
        self.architectural_patterns: Dict[str, List[str]] = {}
        self._last_updated = None

        # Persistencia optimizada (batch) para grandes repositorios
        try:
            self._save_every = int(os.getenv("NB_LEARNER_SAVE_EVERY", "200") or 200)
        except Exception:
            self._save_every = 200
        self._pending_changes = 0
        # Verbosidad de guardado (por defecto silencioso, solo en force)
        try:
            self._verbose_save = str(
                os.getenv("NB_LEARNER_SAVE_VERBOSE", "0")
            ).strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            self._verbose_save = False

        self._load_knowledge()

    def _load_knowledge(self):
        try:
            if os.path.exists(self.storage_path):
                with shelve.open(self.storage_path) as db:
                    self.modules = dict(db.get("modules", {}))
                    self.dependencies = {
                        k: set(v) for k, v in db.get("dependencies", {}).items()
                    }
                    self.configurations = dict(db.get("configurations", {}))
                    self.documentation_index = dict(db.get("documentation_index", {}))
                    self.architectural_patterns = dict(
                        db.get("architectural_patterns", {})
                    )
                    self._last_updated = db.get("_last_updated")
                print(
                    f"✅ Conocimiento del proyecto cargado: {len(self.modules)} módulos"
                )
        except Exception as e:
            print(f"⚠️  Error cargando knowledge graph: {e}")

    def _save_knowledge(self, force: bool = False):
        """Guarda conocimiento con escritura en batch para reducir I/O."""
        try:
            self._pending_changes += 1
            if not force and self._pending_changes < max(1, self._save_every):
                return

            self._last_updated = datetime.utcnow()
            with shelve.open(self.storage_path) as db:
                db["modules"] = self.modules
                db["dependencies"] = {k: list(v) for k, v in self.dependencies.items()}
                db["configurations"] = self.configurations
                db["documentation_index"] = self.documentation_index
                db["architectural_patterns"] = self.architectural_patterns
                db["_last_updated"] = self._last_updated
            self._pending_changes = 0
            # Mensaje controlado: por defecto solo cuando es force o si está habilitada la verbosidad
            if force or self._verbose_save:
                print(
                    f"💾 Conocimiento del proyecto guardado: {len(self.modules)} módulos"
                )
        except Exception as e:
            print(f"❌ Error guardando knowledge graph: {e}")

    def add_module(self, path: str, module_info: Dict[str, Any]):
        """Añadir información de módulo al grafo de conocimiento."""
        self.modules[path] = {
            "path": path,
            "type": module_info.get("type", "unknown"),
            "dependencies": module_info.get("dependencies", []),
            "functions": module_info.get("functions", []),
            "classes": module_info.get("classes", []),
            "complexity_score": module_info.get("complexity_score", 0.0),
            "last_analyzed": datetime.utcnow().isoformat(),
            **module_info,
        }
        self.dependencies[path] = set(module_info.get("dependencies", []))
        self._save_knowledge()

    def update_configurations(self, config_data: Dict[str, Any]):
        """Actualizar configuraciones aprendidas."""
        self.configurations.update(config_data)
        self._save_knowledge()

    def index_documentation(self, doc_path: str, content_summary: str):
        """Indexar documentación."""
        self.documentation_index[doc_path] = content_summary
        self._save_knowledge()


class MCPProjectScanner:
    """Escáner inteligente del proyecto MCP completo."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.knowledge_graph = ProjectKnowledgeGraph()
        self.scanned_files: Set[str] = set()

        # Control de rendimiento para repositorios grandes
        try:
            self.max_files = int(os.getenv("NB_LEARNER_MAX_FILES", "0") or 0)
        except Exception:
            self.max_files = 0
        # Directorios a omitir (normalizados a minúsculas)
        default_skips = {
            ".git",
            "__pycache__",
            "node_modules",
            "venv",
            ".venv",
            "dist",
            "build",
            "htmlcov",
            "coverage",
            "logs",
            "security_logs_20251117125456",
            "security_logs_20251117161150",
        }
        env_skips = os.getenv("NB_LEARNER_SKIP_DIRS", "")
        if env_skips:
            default_skips.update(
                {s.strip().lower() for s in env_skips.split(",") if s.strip()}
            )
        self.skip_dirs = default_skips

        # Patrones de análisis
        self.python_patterns = {
            "function_def": re.compile(r"\bdef\s+(\w+)\s*\("),
            "class_def": re.compile(r"\bclass\s+(\w+)"),
            "import": re.compile(r"^(?:from\s+[\w.]+\s+)?import\s+([\w.,\s]+)"),
            "config_load": re.compile(r"config\.|settings\.|\.yml|\.yaml|\.json"),
        }

        self.config_files = [
            "config/**/*.yml",
            "config/**/*.yaml",
            "config/**/*.json",
            "*.yml",
            "*.yaml",
            "*.json",
            "conf/**/*.py",
            "settings/**/*.py",
        ]

        self.doc_files = [
            "docs/**/*.md",
            "docs/**/*.txt",
            "README*",
            "docs/**/*.rst",
            "**/readme*",
            "**/*.md",
        ]

    async def scan_project_complete(self) -> Dict[str, Any]:
        """Escaneo completo del proyecto MCP."""
        print(f"🔍 Iniciando escaneo completo del proyecto: {self.project_root}")

        # Escanear módulos Python
        await self._scan_python_modules()

        # Escanear configuraciones
        await self._scan_configurations()

        # Escanear documentación
        await self._scan_documentation()

        # Analizar dependencias
        await self._analyze_dependencies()

        # Detectar patrones arquitectónicos
        await self._detect_architectural_patterns()

        scan_results = {
            "total_modules": len(self.knowledge_graph.modules),
            "total_dependencies": sum(
                len(deps) for deps in self.knowledge_graph.dependencies.values()
            ),
            "config_files_found": len(self.knowledge_graph.configurations),
            "documentation_indexed": len(self.knowledge_graph.documentation_index),
            "architectural_patterns": self.knowledge_graph.architectural_patterns,
            "scan_completed": datetime.utcnow().isoformat(),
        }

        print("✅ Escaneo completo terminado:")
        print(f"   📦 Módulos encontrados: {scan_results['total_modules']}")
        print(f"   🔗 Dependencias analizadas: {scan_results['total_dependencies']}")
        print(f"   ⚙️  Configuraciones cargadas: {scan_results['config_files_found']}")
        print(f"   📚 Documentación indexada: {scan_results['documentation_indexed']}")

        return scan_results

    async def _scan_python_modules(self):
        """Escanea todos los módulos Python del proyecto."""
        print("🔧 Escaneando módulos Python...")

        processed = 0
        for py_file in self.project_root.rglob("*.py"):
            if py_file.name.startswith(".") or "test" in py_file.name.lower():
                continue

            # Omitir directorios grandes/no relevantes
            parts_lower = {p.lower() for p in py_file.parts}
            if parts_lower & self.skip_dirs:
                continue

            try:
                relative_path = str(py_file.relative_to(self.project_root))
                module_info = await self._analyze_python_file(py_file)

                if module_info:
                    self.knowledge_graph.add_module(relative_path, module_info)
                    self.scanned_files.add(relative_path)
                    processed += 1
                    # Progreso mínimo
                    if processed % 200 == 0:
                        print(f"   ⋯ Progreso: {processed} archivos analizados…")

                    # Limitar cantidad si se configuró
                    if self.max_files and processed >= self.max_files:
                        print(
                            f"⏳ Límite de análisis alcanzado: {processed}/{self.max_files}"
                        )
                        break

            except Exception as e:
                print(f"⚠️  Error analizando {py_file}: {e}")

        # Persistir cambios pendientes del batch
        self.knowledge_graph._save_knowledge(force=True)
        print(f"✅ Módulos escaneados: {len(self.knowledge_graph.modules)}")

    async def _analyze_python_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Análisis profundo de un archivo Python."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Extraer funciones
            functions = self.python_patterns["function_def"].findall(content)
            classes = self.python_patterns["class_def"].findall(content)

            # Extraer imports
            imports_raw = self.python_patterns["import"].findall(content)
            imports = []
            for imp in imports_raw:
                imports.extend(
                    [x.strip() for x in imp.replace(",", " ").split() if x.strip()]
                )

            # Calcular complejidad básica
            complexity_score = self._calculate_code_complexity(content)

            # Determinar tipo de módulo
            module_type = self._classify_module(file_path, content)

            return {
                "file_path": str(file_path),
                "type": module_type,
                "functions": functions,
                "classes": classes,
                "imports": list(set(imports)),
                "dependencies": imports,
                "line_count": len(content.splitlines()),
                "complexity_score": complexity_score,
                "size_kb": file_path.stat().st_size / 1024,
            }

        except Exception as e:
            return None

    def _calculate_code_complexity(self, content: str) -> float:
        """Calcula complejidad básica del código."""
        lines = content.splitlines()
        complexity = 0
        complexity += len([l for l in lines if "if " in l])
        complexity += len([l for l in lines if "for " in l or "while " in l])
        complexity += len([l for l in lines if "try:" in l])
        complexity += len([l for l in lines if "def " in l or "class " in l])

        # Normalizar por longitud del archivo
        normalized = min(1.0, complexity / max(1, len(lines) / 50))

        return round(normalized, 3)

    def _classify_module(self, file_path: Path, content: str) -> str:
        """Clasifica el tipo de módulo."""
        name = file_path.name.lower()

        if "agent" in name or "agent" in content:
            return "agent_module"
        elif "orchestrator" in name or "orchestrator" in content:
            return "orchestration_module"
        elif "config" in name or self.python_patterns["config_load"].search(content):
            return "configuration_module"
        elif "security" in name or "auth" in name:
            return "security_module"
        elif "logger" in name or "logging" in content:
            return "logging_module"
        elif "api" in name or "handler" in content:
            return "api_module"
        elif "database" in name or "db" in name:
            return "database_module"
        elif "neural" in name or "brain" in content:
            return "neural_module"
        else:
            return "utility_module"

    async def _scan_configurations(self):
        """Escanea archivos de configuración."""
        print("⚙️  Escaneando configuraciones...")

        for pattern in self.config_files:
            for config_file in self.project_root.glob(pattern):
                try:
                    if config_file.is_file():
                        config_data = await self._analyze_config_file(config_file)
                        if config_data:
                            relative_path = str(
                                config_file.relative_to(self.project_root)
                            )
                            self.knowledge_graph.update_configurations(
                                {relative_path: config_data}
                            )
                except Exception as e:
                    print(f"⚠️  Error analizando config {config_file}: {e}")

    async def _analyze_config_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Análisis básico de archivo de configuración."""
        try:
            ext = file_path.suffix.lower()
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            if ext == ".json":
                try:
                    data = json.loads(content)
                    return {
                        "type": "json_config",
                        "keys_count": len(data) if isinstance(data, dict) else "array",
                        "nested_level": self._deep_count(data),
                    }
                except:
                    pass

            elif ext in [".yml", ".yaml"]:
                return {
                    "type": "yaml_config",
                    "lines": len(content.splitlines()),
                    "estimated_keys": content.count(": "),
                    "has_secrets": bool(
                        re.search(r"password|token|secret", content, re.I)
                    ),
                }

            return {
                "type": "unknown_config",
                "size_kb": file_path.stat().st_size / 1024,
            }

        except Exception as e:
            return None

    def _deep_count(self, obj, depth=0) -> int:
        """Cuenta profundidad de estructura JSON/YAML."""
        if isinstance(obj, dict):
            return max(
                (self._deep_count(v, depth + 1) for v in obj.values()), default=depth
            )
        elif isinstance(obj, list):
            return max(
                (self._deep_count(item, depth + 1) for item in obj), default=depth
            )
        else:
            return depth

    async def _scan_documentation(self):
        """Escanea e indexa documentación."""
        print("📚 Escaneando documentación...")

        for pattern in self.doc_files:
            for doc_file in self.project_root.glob(pattern):
                try:
                    if doc_file.is_file():
                        doc_summary = await self._analyze_documentation(doc_file)
                        if doc_summary:
                            relative_path = str(doc_file.relative_to(self.project_root))
                            self.knowledge_graph.index_documentation(
                                relative_path, doc_summary
                            )
                except Exception as e:
                    print(f"⚠️  Error analizando doc {doc_file}: {e}")

    async def _analyze_documentation(self, file_path: Path) -> Optional[str]:
        """Análisis básico de documentación."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Extraer secciones principales
            lines = content.splitlines()
            sections = []
            current_section = ""

            for line in lines[:200]:  # Analizar primeras 200 líneas
                line = line.strip()
                if line.startswith("# "):
                    if current_section:
                        sections.append(current_section)
                    current_section = line[2:]
                elif current_section and len(current_section) < 500:
                    current_section += " " + line[:100]  # Limitar crecimiento

            if current_section:
                sections.append(current_section)

            summary = f"DOC: {', '.join(sections[:5])}"  # Top 5 secciones
            return summary[:500]  # Limitar a 500 chars

        except Exception as e:
            return None

    async def _analyze_dependencies(self):
        """Analiza dependencias entre módulos."""
        print("🔗 Analizando dependencias...")

        # Construir grafo de dependencias basadas en imports
        dependency_graph = {}

        for module_path, module_info in self.knowledge_graph.modules.items():
            module_name = Path(module_path).stem

            for dep in module_info.get("dependencies", []):
                dep_path = await self._resolve_import_to_path(dep, module_path)
                if dep_path and dep_path in self.knowledge_graph.modules:
                    if module_path not in dependency_graph:
                        dependency_graph[module_path] = set()
                    dependency_graph[module_path].add(dep_path)

        self.knowledge_graph.dependencies.update(dependency_graph)
        self.knowledge_graph._save_knowledge(force=True)

        print(
            f"✅ Dependencias analizadas: {len(self.knowledge_graph.dependencies)} relaciones"
        )

    async def _resolve_import_to_path(
        self, import_name: str, from_module: str
    ) -> Optional[str]:
        """Resuelve un import a un path relativo conocido."""
        # Lógica básica de resolución
        if import_name.startswith("."):
            from_path = Path(from_module).parent
            for part in import_name.split("."):
                if part:
                    from_path = from_path / part
            return str(from_path.with_suffix(".py"))
        else:
            # Import absoluto - buscar en módulos conocidos
            for known_path in self.knowledge_graph.modules:
                if (
                    known_path.endswith(import_name + ".py")
                    or f"{import_name}." in known_path
                ):
                    return known_path

        return None

    async def _detect_architectural_patterns(self):
        """Detecta patrones arquitectónicos del proyecto."""
        print("🏗️  Detectando patrones arquitectónicos...")

        patterns = {}

        # Patrones de arquitectura
        module_types = {}
        for module in self.knowledge_graph.modules.values():
            mtype = module.get("type", "unknown")
            module_types[mtype] = module_types.get(mtype, 0) + 1

        patterns["module_types"] = module_types

        # Patrones de dependencia
        circular_deps = []
        for module, deps in self.knowledge_graph.dependencies.items():
            for dep in deps:
                if (
                    dep in self.knowledge_graph.dependencies
                    and module in self.knowledge_graph.dependencies[dep]
                ):
                    circular_deps.append((module, dep))

        patterns["circular_dependencies"] = circular_deps

        # Patrones de configuración
        config_patterns = []
        for config, data in self.knowledge_graph.configurations.items():
            if isinstance(data, dict) and data.get("has_secrets"):
                config_patterns.append("security_configuration_detected")

        patterns["configuration_patterns"] = config_patterns

        self.knowledge_graph.architectural_patterns = patterns
        self.knowledge_graph._save_knowledge(force=True)

        print(f"✅ Patrones arquitectónicos detectados")


class NeuralBrainLearner:
    """Sistema de aprendizaje automático del cerebro neuronal MCP."""

    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.scanner = MCPProjectScanner(project_root)
        self.learning_memory: Dict[str, Any] = {}

    async def learn_project_deep_scan(self) -> Dict[str, Any]:
        """Escaneo profundo y exhaustivo de TODO el proyecto MCP."""
        print("🔬 CEREBRO NEURONAL MCP - ESCANEO PROFUNDO DEL PROYECTO")
        print("=" * 70)

        # Eliminar archivos de conocimiento previos para escaneo fresh
        knowledge_files = ["project_knowledge.db", "neural_brain_learning.json"]

        for kf in knowledge_files:
            try:
                if os.path.exists(kf):
                    os.remove(kf)
                    print(f"🧹 Eliminado {kf} para escaneo limpio")
            except Exception:
                pass

        # Crear nuevo scanner limpio
        self.scanner = MCPProjectScanner(self.project_root)

        # Configuración masiva de escaneo completo
        set_env_vars_for_deep_scan()

        print("🔍 Iniciando escaneo PROFUNDO - analizando TODOS los archivos...")
        scan_results = await self.scanner.scan_project_complete()

        # Escaneo adicional: Buscar archivos Python que no se hayan encontrado
        print("\n🔎 EJECUTANDO ESCANEO COMPLEMENTARIO DE VERIFICACIÓN...")
        complementary_findings = await self._complementary_scan()

        # Escaneo secundario: Archivos de configuración ocultos
        hidden_configs = await self._scan_hidden_configs()

        # Merge de resultados
        total_modules = len(self.scanner.knowledge_graph.modules)
        complementary_modules = len(complementary_findings.get("modules", {}))
        final_modules = total_modules + complementary_modules

        print(f"\n📊 RESULTADOS FINALES:")
        print(f"   📦 Módulos originales: {total_modules}")
        print(f"   🔍 Módulos complementarios: {complementary_modules}")
        print(f"   🎯 TOTAL MÓDULOS ENCONTRADOS: {final_modules}")

        if complementary_findings.get("modules"):
            # Añadir los módulos adicionales
            for path, module_info in complementary_findings["modules"].items():
                self.scanner.knowledge_graph.add_module(path, module_info)
            print("✅ Módulos complementarios añadidos al conocimiento")

        # Análisis profundo de componentes clave
        mcp_core_analysis = await self._analyze_mcp_core()

        # Aprendizaje de patrones de uso avanzado
        usage_patterns = await self._learn_usage_patterns_deep()

        # Optimizaciones sugeridas basadas en análisis profundo
        optimizations = await self._generate_deep_optimizations()

        learning_results = {
            "scan_completed": scan_results,
            "complementary_scan": complementary_findings,
            "hidden_configs": hidden_configs,
            "final_module_count": final_modules,
            "mcp_core_understanding": mcp_core_analysis,
            "learned_patterns": usage_patterns,
            "suggested_optimizations": optimizations,
            "deep_scan_timestamp": datetime.utcnow().isoformat(),
            "project_name": "Sheily MCP System",
            "total_files_learned": len(self.scanner.scanned_files),
        }

        # Guardar el aprendizaje profundo
        self._save_learning(learning_results)

        print(f"\n🎊 ESCANEO PROFUNDO COMPLETADO!")
        print(f"   🧠 Proyecto completamente analizado a profundidad")
        print(f"   📊 {learning_results['total_files_learned']} archivos escaneados")
        print(f"   🎯 {final_modules} MODULOS TOTALES ENCONTRADOS")
        print(
            f"   🏗️  Arquitectura comprendida: {len(scan_results['architectural_patterns'])} patrones"
        )
        print(f"   💡 Optimizaciones generadas: {len(optimizations)} sugerencias")

        return learning_results

    async def _complementary_scan(self) -> Dict[str, Any]:
        """Escaneo complementario para archivos que podrían haberse perdido."""
        print("🔎 Escaneo complementario activado...")

        found_modules = {}
        found_configs = {}
        found_docs = []

        # Buscar todos los directorios recursivamente
        all_dirs = []
        for root, dirs, files in os.walk(self.project_root):
            # Omitir directorios de sistema
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d.lower() not in self.scanner.skip_dirs
            ]

            relative_root = os.path.relpath(root, self.project_root)
            if relative_root != ".":
                all_dirs.append(relative_root)

        print(f"   🗂️  Directorios candidatos: {len(all_dirs)}")

        # Escanear archivos Python manualmente en cada directorio
        for relative_dir in all_dirs:
            dir_path = Path(self.project_root) / relative_dir
            try:
                for py_file in dir_path.glob("*.py"):
                    # Verificar si ya está escaneado
                    relative_path = os.path.relpath(py_file, self.project_root)

                    if relative_path in self.scanner.scanned_files:
                        continue  # Ya escaneado

                    # Analizar archivo nuevo
                    try:
                        module_info = await self.scanner._analyze_python_file(py_file)
                        if module_info:
                            found_modules[relative_path] = module_info
                            self.scanner.scanned_files.add(relative_path)
                            print(f"   ✅ Encontrado nuevo módulo: {relative_path}")
                    except Exception as e:
                        print(f"   ⚠️  Error analizando {relative_path}: {e}")

                    # Buscar archivos de configuración
                    for config_file in dir_path.glob("*.yml"):
                        config_path = os.path.relpath(config_file, self.project_root)
                        if (
                            config_path
                            not in self.scanner.knowledge_graph.configurations
                        ):
                            found_configs[config_path] = {
                                "type": "yaml_found",
                                "location": relative_dir,
                            }
                            print(f"   ⚙️  Encontrado config candidato: {config_path}")

            except PermissionError:
                continue  # Saltar directorios sin permisos
            except Exception as e:
                print(f"   ⚠️  Error escaneando {relative_dir}: {e}")

        return {
            "modules": found_modules,
            "configs": found_configs,
            "directories_scanned": len(all_dirs),
        }

    async def _scan_hidden_configs(self) -> Dict[str, Any]:
        """Escaneo de configuraciones ocultas o no estándares."""
        hidden_configs = {}

        # Patrones adicionales para archivos de configuración
        hidden_patterns = [
            "**/settings/**/*.py",
            "**/conf/**/*.ini",
            "**/*config*/**/*.toml",
            "**/.env*",
            "**/env/**/*.json",
            "**/secrets/**/*.yaml",
        ]

        for pattern in hidden_patterns:
            try:
                for config_file in Path(self.project_root).glob(pattern):
                    if config_file.is_file():
                        relative_path = os.path.relpath(config_file, self.project_root)
                        if (
                            relative_path
                            not in self.scanner.knowledge_graph.configurations
                        ):
                            hidden_configs[relative_path] = {
                                "type": "hidden_config",
                                "pattern": pattern,
                            }
            except Exception:
                continue

        return hidden_configs

    async def _learn_usage_patterns_deep(self) -> Dict[str, Any]:
        """Aprendizaje avanzado de patrones de uso."""
        patterns = {
            "task_types": [
                "security_audit",
                "ml_training",
                "data_processing",
                "config_management",
                "api_calls",
            ],
            "integration_points": [
                "n8n",
                "blockchain",
                "rag",
                "neural_brain",
                "docker",
                "kubernetes",
            ],
            "deployment_targets": [
                "enterprise",
                "distributed",
                "dockerized",
                "cloud-native",
            ],
            "monitoring_requirements": [
                "real_time",
                "comprehensive",
                "ai_driven",
                "enterprise_logging",
            ],
            "security_requirements": [
                "multi_layer",
                "zero_trust",
                "continuous_monitoring",
            ],
            "scalability_patterns": ["horizontal", "vertical", "microservices"],
        }

        return patterns

    async def _generate_deep_optimizations(self) -> List[str]:
        """Genera recomendaciones profundas basadas en análisis completo."""
        optimizations = []

        module_count = len(self.scanner.knowledge_graph.modules)

        # Optimizaciones basadas en escala
        if module_count > 1000:
            optimizations.append(
                "🏗️  Implement distributed module loading for better startup performance"
            )
            optimizations.append(
                "🔄 Add module dependency caching to reduce import overhead"
            )

        if module_count > 500:
            optimizations.append(
                "📦 Consider module federation for large-scale deployments"
            )

        # Análisis de dependencias
        circular_deps = self.scanner.knowledge_graph.architectural_patterns.get(
            "circular_dependencies", []
        )
        if circular_deps:
            optimizations.append(
                f"🔄 Break {len(circular_deps)} circular dependencies detected"
            )

        # Análisis de configuración
        configs = self.scanner.knowledge_graph.configurations
        if len(configs) > 20:
            optimizations.append(
                "⚙️  Implement configuration validation and schema enforcement"
            )

        # Análisis de módulos por tipo
        module_types = {}
        for module in self.scanner.knowledge_graph.modules.values():
            mtype = module.get("type", "unknown")
            module_types[mtype] = module_types.get(mtype, 0) + 1

        # Recomendaciones por tipo
        if module_types.get("security_module", 0) < 5:
            optimizations.append(
                "🔒 Expand security modules - minimum 5 security layers recommended"
            )

        if module_types.get("api_module", 0) > 10:
            optimizations.append(
                "🌐 Consider API gateway for managing multiple endpoints"
            )

        if module_types.get("neural_module", 0) < 2:
            optimizations.append(
                "🧠 Strengthen neural components - add more AI capabilities"
            )

        return optimizations

    async def learn_project_completely(self) -> Dict[str, Any]:
        """Aprendizaje completo del proyecto MCP."""
        print("🧠 CEREBRO NEURONAL MCP - APRENDIZAJE DEL PROYECTO COMPLETO")
        print("=" * 70)

        # Escaneo completo del proyecto
        scan_results = await self.scanner.scan_project_complete()

        # Análisis profundo de componentes clave
        mcp_core_analysis = await self._analyze_mcp_core()

        # Aprendizaje de patrones de uso
        usage_patterns = await self._learn_usage_patterns()

        # Optimizaciones sugeridas basadas en aprendizaje
        optimizations = await self._generate_quotations()

        learning_results = {
            "scan_completed": scan_results,
            "mcp_core_understanding": mcp_core_analysis,
            "learned_patterns": usage_patterns,
            "suggested_optimizations": optimizations,
            "learning_timestamp": datetime.utcnow().isoformat(),
            "project_name": "Sheily MCP System",
            "total_files_learned": len(self.scanner.scanned_files),
        }

        # Guardar el aprendizaje
        self._save_learning(learning_results)

        print("\n🎊 APRENDIZAJE COMPLETADO!")
        print(f"   🧠 Proyecto totalmente aprendido")
        print(f"   📊 {learning_results['total_files_learned']} archivos analizados")
        print(
            f"   🏗️  Arquitectura comprendida: {len(scan_results['architectural_patterns'])} patrones"
        )
        print(f"   💡 Optimizaciones generadas: {len(optimizations)} sugerencias")

        return learning_results

    async def _analyze_mcp_core(self) -> Dict[str, Any]:
        """Análisis profundo del core de MCP."""
        core_analysis = {
            "orchestration_modules": [],
            "agent_modules": [],
            "neural_components": [],
            "security_layers": [],
            "api_endpoints": [],
        }

        for module_path, module_info in self.scanner.knowledge_graph.modules.items():
            module_type = module_info.get("type")

            if module_type == "orchestration_module":
                core_analysis["orchestration_modules"].append(
                    {
                        "path": module_path,
                        "functions": module_info.get("functions", []),
                        "complexity": module_info.get("complexity_score"),
                    }
                )

            elif module_type == "agent_module":
                core_analysis["agent_modules"].append(
                    {
                        "path": module_path,
                        "classes": module_info.get("classes", []),
                        "functions": module_info.get("functions", []),
                    }
                )

            elif module_type == "neural_module":
                core_analysis["neural_components"].append(
                    {
                        "path": module_path,
                        "capabilities": (
                            "neural_brain_learning"
                            if "brain" in module_path
                            else "neural_support"
                        ),
                    }
                )

            elif module_type == "security_module":
                core_analysis["security_layers"].append(
                    {
                        "path": module_path,
                        "security_features": len(module_info.get("functions", [])),
                    }
                )

        return core_analysis

    async def _learn_usage_patterns(self) -> Dict[str, Any]:
        """Aprende patrones de uso del proyecto."""
        patterns = {
            "task_types": [
                "security_audit",
                "ml_training",
                "data_processing",
                "config_management",
            ],
            "integration_points": ["n8n", "blockchain", "rag", "neural_brain"],
            "deployment_targets": ["enterprise", "distributed", "dockerized"],
            "monitoring_requirements": ["real_time", "comprehensive", "ai_driven"],
        }

        return patterns

    async def _generate_quotations(self) -> List[str]:
        """Genera optimizaciones basadas en el aprendizaje."""
        optimizations = []

        # Análisis de dependencias
        circular_deps = self.scanner.knowledge_graph.architectural_patterns.get(
            "circular_dependencies", []
        )

        if circular_deps:
            optimizations.append(
                "🔄 Break circular dependencies detected in modules for better maintainability"
            )

        # Análisis de configuración
        configs = self.scanner.knowledge_graph.configurations

        if configs:
            optimizations.append(
                "⚙️  Centralize configuration management - detected custom config files across modules"
            )

        # Análisis de módulos neurales
        neural_modules = [
            m
            for m in self.scanner.knowledge_graph.modules.values()
            if m.get("type") == "neural_module"
        ]

        if not neural_modules:
            optimizations.append(
                "🧠 Integrate neural intelligence layer for adaptive decision making"
            )
        elif len(neural_modules) < 3:
            optimizations.append(
                "🧬 Expand neural components - currently basic intelligence layer"
            )

        return optimizations

    def _save_learning(self, learning_results: Dict[str, Any]):
        """Guarda los resultados del aprendizaje."""
        # Soporte de directorio de salida configurable
        output_dir = os.getenv("NB_LEARNER_OUTPUT_DIR", "reports").strip() or "reports"
        target_dir = Path(self.project_root) / output_dir
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Si falla creación, fallback a raíz
            target_dir = Path(self.project_root)
        learning_file = target_dir / "neural_brain_learning.json"

        try:
            with open(learning_file, "w", encoding="utf-8") as f:
                json.dump(
                    learning_results, f, indent=2, ensure_ascii=False, default=str
                )

            print(f"💾 Aprendizaje guardado en: {learning_file}")

        except Exception as e:
            print(f"❌ Error guardando aprendizaje: {e}")

    async def get_project_insights(self) -> Dict[str, Any]:
        """Obtiene insights aprendidos del proyecto."""
        output_dir = os.getenv("NB_LEARNER_OUTPUT_DIR", "reports").strip() or "reports"
        learning_file = (
            Path(self.project_root) / output_dir / "neural_brain_learning.json"
        )
        if not learning_file.exists():
            # fallback legacy ubicación
            legacy = Path(self.project_root) / "neural_brain_learning.json"
            if legacy.exists():
                learning_file = legacy

        if learning_file.exists():
            try:
                with open(learning_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error cargando insights: {e}")

        return {
            "error": "No learning data available - run learn_project_completely() first"
        }


# FUNCIONES DE UTILIDAD PARA INTEGRACIÓN
async def auto_learn_project(project_root: str = ".") -> NeuralBrainLearner:
    """Aprende automáticamente todo el proyecto MCP."""
    learner = NeuralBrainLearner(project_root)
    await learner.learn_project_completely()
    return learner


async def get_project_knowledge(project_root: str = ".") -> Dict[str, Any]:
    """Obtiene conocimiento aprendido del proyecto."""
    learner = NeuralBrainLearner(project_root)
    insights = await learner.get_project_insights()
    return insights


if __name__ == "__main__":
    # Demo del learner
    asyncio.run(auto_learn_project())
