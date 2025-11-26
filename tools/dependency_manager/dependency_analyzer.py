"""
Sheily MCP Enterprise - Dependency Analyzer
Sistema avanzado de análisis de dependencias del proyecto
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """Analizador avanzado de dependencias del proyecto"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.config_dir = self.root_dir / "config"
        self.frontend_dir = self.root_dir / "Frontend"

        # Escaneo completo recursivo del proyecto
        self.dependency_files = self._scan_project_dependencies()

    async def analyze_project(self, deep_analysis: bool = False) -> Dict[str, Any]:
        """Análisis completo de dependencias del proyecto"""

        logger.info("Starting comprehensive dependency analysis...")

        results = {
            "timestamp": asyncio.get_event_loop().time(),
            "project": "Sheily MCP Enterprise",
            "analysis_type": "deep" if deep_analysis else "standard",
            "total_dependency_files": len(self.dependency_files),
            "all_dependencies": {},
            "python_dependencies": {},
            "frontend_dependencies": {},
            "conflicts": [],
            "security_issues": [],
            "optimization_opportunities": [],
        }

        try:
            # Análisis COMPLETO DE TODOS los archivos de dependencias encontrados
            results["all_dependencies"] = await self._analyze_all_dependency_files()

            # Analizar dependencias Python por categorías
            results["python_dependencies"] = await self._analyze_python_dependencies(
                deep_analysis
            )

            # Analizar dependencias Frontend
            results["frontend_dependencies"] = (
                await self._analyze_frontend_dependencies()
            )

            # Detectar conflictos
            results["conflicts"] = await self._detect_dependency_conflicts()

            # Análisis de seguridad básico
            results["security_issues"] = await self._analyze_security_issues()

            # Oportunidades de optimización
            results["optimization_opportunities"] = (
                await self._analyze_optimization_opportunities()
            )

            logger.info(
                f"Dependency analysis completed successfully - found {results['total_dependency_files']} files"
            )

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            results["error"] = str(e)

        return results

    async def _analyze_python_dependencies(self, deep: bool = False) -> Dict[str, Any]:
        """Analizar dependencias Python por categoría"""

        python_deps = {
            "core": [],
            "dev": [],
            "ci": [],
            "rag": [],
            "total_packages": 0,
            "version_conflicts": [],
        }

        # Analizar requirements.txt principal
        core_deps = await self._parse_requirements_file(
            self.dependency_files["python_core"]
        )
        python_deps["core"] = core_deps
        python_deps["total_packages"] += len(core_deps)

        # Analizar requirements-dev.txt
        dev_deps = await self._parse_requirements_file(
            self.dependency_files["python_dev"]
        )
        python_deps["dev"] = dev_deps
        python_deps["total_packages"] += len(dev_deps)

        # Analizar requirements-ci.txt
        ci_deps = await self._parse_requirements_file(
            self.dependency_files["python_ci"]
        )
        python_deps["ci"] = ci_deps
        python_deps["total_packages"] += len(ci_deps)

        # Analizar requirements-rag.txt
        rag_deps = await self._parse_requirements_file(
            self.dependency_files["python_rag"]
        )
        python_deps["rag"] = rag_deps
        python_deps["total_packages"] += len(rag_deps)

        # Detectar conflictos básicos entre categorías
        python_deps["version_conflicts"] = await self._detect_python_conflicts(
            core_deps, dev_deps, ci_deps, rag_deps
        )

        return python_deps

    async def _analyze_frontend_dependencies(self) -> Dict[str, Any]:
        """Analizar dependencias Frontend"""

        frontend_deps = {
            "dependencies": {},
            "devDependencies": {},
            "total_packages": 0,
            "total_dev_packages": 0,
        }

        try:
            package_json = self.dependency_files["frontend"]
            if package_json.exists():
                with open(package_json, "r", encoding="utf-8") as f:
                    package_data = json.load(f)

                frontend_deps["dependencies"] = package_data.get("dependencies", {})
                frontend_deps["devDependencies"] = package_data.get(
                    "devDependencies", {}
                )

                frontend_deps["total_packages"] = len(frontend_deps["dependencies"])
                frontend_deps["total_dev_packages"] = len(
                    frontend_deps["devDependencies"]
                )

        except Exception as e:
            logger.warning(f"Error parsing package.json: {e}")

        return frontend_deps

    async def _parse_requirements_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parsear archivo requirements.txt"""

        dependencies = []

        if not file_path.exists():
            return dependencies

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parsear líneas de requirements
            for line in content.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                dep_info = await self._parse_requirement_line(line)
                if dep_info:
                    dependencies.append(dep_info)

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

        return dependencies

    async def _parse_requirement_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parsear una línea individual de requirements"""

        # Remover comentarios
        line = line.split("#")[0].strip()

        # Pattern para package==version o package>=version,<version
        pattern = r"^([a-zA-Z0-9\-_.]+)([>=<~!]+.*)?$"
        match = re.match(pattern, line)

        if match:
            package_name = match.group(1)
            version_spec = match.group(2) or ""

            return {
                "package": package_name,
                "version_spec": version_spec,
                "line": line,
                "category": await self._categorize_package(package_name),
            }

        return None

    async def _categorize_package(self, package_name: str) -> str:
        """Categorizar un paquete según su función"""

        package_lower = package_name.lower()

        # Categorías básicas extendidas
        if package_name in ["fastapi", "uvicorn", "starlette", "pydantic"]:
            return "web_framework"
        elif package_name in ["sqlalchemy", "psycopg2-binary", "alembic"]:
            return "database"
        elif any(
            ai_pkg in package_lower
            for ai_pkg in [
                "torch",
                "transformers",
                "sentence-transformers",
                "diffusers",
                "accelerate",
                "datasets",
                "tokenizers",
                "faiss",
                "chromadb",
                "hnswlib",
                "pinecone",
                "weaviate",
                "milvus",
                "qdrant",
                "annoy",
                "nmslib",
                "simd",
            ]
        ):
            return "ai_ml"
        elif any(
            cv_pkg in package_lower
            for cv_pkg in ["opencv", "pillow", "scikit-image", "albumentations"]
        ):
            return "computer_vision"
        elif any(
            nlp_pkg in package_lower
            for nlp_pkg in ["nltk", "spacy", "coreferee", "textblob"]
        ):
            return "nlp"
        elif package_name in [
            "pytest",
            "black",
            "isort",
            "flake8",
            "mypy",
            "bandit",
            "pre-commit",
        ]:
            return "development"
        elif package_name in [
            "cryptography",
            "pyjwt",
            "bcrypt",
            "python-json-logger",
            "structlog",
        ]:
            return "security"
        elif any(
            data_pkg in package_lower
            for data_pkg in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "matplotlib",
                "seaborn",
                "plotly",
            ]
        ):
            return "data_science"
        elif package_name in ["redis", "celery", "rabbitmq"]:
            return "message_queue"
        elif any(
            web_pkg in package_lower
            for web_pkg in ["requests", "httpx", "aiohttp", "websockets"]
        ):
            return "web_client"
        elif package_name in ["rich", "tqdm", "click", "python-dotenv"]:
            return "utilities"
        else:
            return "general"

    async def _detect_python_conflicts(
        self, core_deps: List, dev_deps: List, ci_deps: List, rag_deps: List
    ) -> List[Dict[str, Any]]:
        """Detectar conflictos básicos entre dependencias Python"""

        conflicts = []
        all_deps = {"core": core_deps, "dev": dev_deps, "ci": ci_deps, "rag": rag_deps}

        # Crear índice de paquetes por nombre
        package_index = {}
        for category, deps in all_deps.items():
            for dep in deps:
                pkg_name = dep["package"]
                if pkg_name not in package_index:
                    package_index[pkg_name] = []
                package_index[pkg_name].append(
                    {
                        "category": category,
                        "version_spec": dep["version_spec"],
                        "package_info": dep,
                    }
                )

        # Detectar paquetes con diferentes especificaciones de versión
        for pkg_name, instances in package_index.items():
            if len(instances) > 1:
                version_specs = [inst["version_spec"] for inst in instances]
                if len(set(version_specs)) > 1:  # Diferentes specs
                    conflicts.append(
                        {
                            "package": pkg_name,
                            "type": "version_spec_conflict",
                            "instances": instances,
                            "severity": "medium",
                        }
                    )

        return conflicts

    async def _detect_dependency_conflicts(self) -> List[Dict[str, Any]]:
        """Detectar conflictos generales de dependencias"""

        conflicts = []

        # Python vs Frontend conflicts básicos
        python_packages = await self._analyze_python_dependencies()
        frontend_packages = await self._analyze_frontend_dependencies()

        # Buscar conflictos básicos (esto se puede expandir mucho)
        python_pkg_names = set()
        for deps in python_packages.values():
            if isinstance(deps, list):
                python_pkg_names.update([d["package"] for d in deps])

        # No hay conflictos detectados aún (implementación básica)
        # En una implementación completa, aquí se verificarían:
        # - Version ranges incompatibles
        # - Dependency cycles
        # - Package name conflicts
        # - Environment-specific conflicts

        return conflicts

    async def _analyze_security_issues(self) -> List[Dict[str, Any]]:
        """Análisis básico de issues de seguridad"""

        # Implementación básica - en producción esto usaríamos:
        # - Safety: https://github.com/pyupio/safety
        # - Bandit: https://github.com/PyCQA/bandit
        # - NPM audit

        security_issues = []

        # Placeholder para futuras expansiones
        # Aquí se integrarían herramientas reales de seguridad

        return security_issues

    async def _analyze_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Análisis de oportunidades de optimización"""

        opportunities = []

        # Análisis básico de dependencias no utilizadas
        # En producción, esto se haría con herramientas como:
        # - pip-tools para optimización
        # - deptry para dependencias no utilizadas

        return opportunities

    async def _analyze_all_dependency_files(self) -> Dict[str, Any]:
        """Analizar TODOS los archivos de dependencias encontrados"""

        all_deps = {
            "by_type": {},
            "by_location": {},
            "totals": {
                "python_requirements": 0,
                "frontend_packages": 0,
                "other_dependencies": 0,
                "total_files": len(self.dependency_files),
            },
        }

        # Clasificar archivos por tipo y analizar contenido
        for key, file_path in self.dependency_files.items():
            if not file_path.exists():
                continue

            relative_path = file_path.relative_to(self.root_dir)
            file_type = self._classify_dependency_file(file_path)

            # Analizar contenido según tipo
            if file_type == "python_requirements":
                deps = await self._parse_requirements_file(file_path)
                all_deps["by_type"].setdefault("python_requirements", {}).setdefault(
                    key, deps
                )
                all_deps["totals"]["python_requirements"] += len(deps)

            elif file_type == "frontend_package":
                deps_info = await self._parse_package_json(file_path)
                all_deps["by_type"].setdefault("frontend_packages", {}).setdefault(
                    key, deps_info
                )
                all_deps["totals"]["frontend_packages"] += deps_info.get(
                    "total_packages", 0
                )

            else:
                all_deps["by_type"].setdefault("other", {}).setdefault(
                    key, {"file": str(relative_path)}
                )

            # Clasificar por ubicación/directorio
            location = str(relative_path.parent)
            all_deps["by_location"].setdefault(location, []).append(key)

        logger.info(
            f"Analyzed {len(self.dependency_files)} dependency files: {all_deps['totals']}"
        )
        return all_deps

    def _classify_dependency_file(self, file_path: Path) -> str:
        """Clasificar el tipo de archivo de dependencias"""
        name = file_path.name.lower()

        if name.startswith("requirements") and name.endswith(".txt"):
            return "python_requirements"
        elif name == "package.json":
            return "frontend_package"
        elif name == "package-lock.json":
            return "frontend_lock"
        elif name == "yarn.lock":
            return "yarn_lock"
        elif name == "pipfile":
            return "pipenv"
        elif name == "pipfile.lock":
            return "pipenv_lock"
        elif name == "pyproject.toml":
            return "poetry"
        elif name == "poetry.lock":
            return "poetry_lock"
        elif name == "setup.py":
            return "setup_py"
        else:
            return "other"

    async def _parse_package_json(self, file_path: Path) -> Dict[str, Any]:
        """Parsear package.json file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "dependencies": data.get("dependencies", {}),
                "devDependencies": data.get("devDependencies", {}),
                "total_packages": len(data.get("dependencies", {})),
                "total_dev_packages": len(data.get("devDependencies", {})),
                "name": data.get("name", ""),
                "version": data.get("version", ""),
            }
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")
            return {"error": str(e)}

    def _scan_project_dependencies(self) -> Dict[str, Path]:
        """Escaneo completo recursivo de archivos de dependencias"""

        logger.info("Scanning project for all dependency files...")

        dependency_files = {}

        # Archivos a buscar recursivamente
        dependency_patterns = [
            "**/requirements*.txt",
            "**/package.json",
            "**/package-lock.json",
            "**/yarn.lock",
            "**/Pipfile",
            "**/Pipfile.lock",
            "**/pyproject.toml",
            "**/poetry.lock",
            "**/setup.py",
        ]

        # Excluir directorios de venv, node_modules, etc.
        exclude_dirs = [
            ".venv",
            "venv",
            "env",
            "node_modules",
            "__pycache__",
            ".git",
            ".venv_sheily",
        ]

        # Escanear todos los archivos
        for pattern in dependency_patterns:
            for file_path in self.root_dir.rglob(pattern):
                # Verificar que no esté en directorios excluidos
                if not any(part in file_path.parts for part in exclude_dirs):
                    # Crear clave única para el archivo
                    file_name = file_path.name
                    relative_path = file_path.relative_to(self.root_dir)
                    key = f"{relative_path}".replace("/", "_").replace("\\", "_")

                    dependency_files[key] = file_path
                    logger.debug(f"Found dependency file: {relative_path}")

        # Definir archivos principales conocidos (varian gestión inteligente)
        dependency_files.update(
            {
                "python_core": self.root_dir / "requirements.txt",
                "python_dev": self.config_dir / "requirements-dev.txt",
                "python_ci": self.config_dir / "requirements-ci.txt",
                "python_rag": self.config_dir / "requirements-rag.txt",
                "frontend": self.frontend_dir / "package.json",
                "frontend_lock": self.frontend_dir / "package-lock.json",
            }
        )

        logger.info(f"Found {len(dependency_files)} dependency files in project")
        return dependency_files
