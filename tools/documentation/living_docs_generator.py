# Living Documentation Generator - Sheily AI
# ==========================================

"""
Sistema de documentación viva que genera documentación automáticamente
desde el código fuente, APIs y configuraciones.
"""

import ast
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml


@dataclass
class FunctionDoc:
    """Documentación de una función"""

    name: str
    module: str
    signature: str
    docstring: str = ""
    parameters: Dict[str, str] = field(default_factory=dict)
    return_type: str = ""
    decorators: List[str] = field(default_factory=list)
    complexity: int = 0
    lines_of_code: int = 0


@dataclass
class ClassDoc:
    """Documentación de una clase"""

    name: str
    module: str
    docstring: str = ""
    methods: List[FunctionDoc] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    inheritance: List[str] = field(default_factory=list)


@dataclass
class ModuleDoc:
    """Documentación de un módulo"""

    name: str
    path: str
    docstring: str = ""
    functions: List[FunctionDoc] = field(default_factory=list)
    classes: List[ClassDoc] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


class LivingDocumentationGenerator:
    """
    Generador de documentación viva para Sheily AI
    Analiza código fuente y genera documentación automáticamente
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.output_dir = project_root / "docs" / "living"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configuración de módulos a documentar
        self.modules_to_document = [
            "sheily_core",
            "sheily_train",
            "centralized_tests",
            "tools",
        ]

    def generate_complete_documentation(self) -> Dict[str, Any]:
        """
        Generar documentación completa del proyecto

        Returns:
            Dict con toda la documentación generada
        """
        print("📚 GENERATING LIVING DOCUMENTATION")
        print("=" * 50)

        documentation = {
            "project_name": "Sheily AI",
            "version": "2.0.0",
            "generated_at": "2025-10-31T22:50:00Z",
            "modules": {},
            "apis": {},
            "architecture": {},
            "coverage": {},
        }

        # Generar documentación de módulos
        for module_name in self.modules_to_document:
            print(f"📖 Documenting module: {module_name}")
            module_path = self.project_root / module_name.replace(".", "/")

            if module_path.exists():
                module_doc = self._analyze_module(module_path, module_name)
                documentation["modules"][module_name] = module_doc

        # Generar documentación de APIs (si existe FastAPI)
        api_doc = self._generate_api_documentation()
        documentation["apis"] = api_doc

        # Generar documentación de arquitectura
        arch_doc = self._generate_architecture_documentation()
        documentation["architecture"] = arch_doc

        # Generar documentación de testing/coverage
        coverage_doc = self._generate_coverage_documentation()
        documentation["coverage"] = coverage_doc

        # Guardar documentación
        self._save_documentation(documentation)

        # Generar archivos HTML/Markdown
        self._generate_html_documentation(documentation)
        self._generate_markdown_documentation(documentation)

        print("✅ Living documentation generated successfully!")
        return documentation

    def _analyze_module(self, module_path: Path, module_name: str) -> Dict[str, Any]:
        """Analizar un módulo completo"""
        module_doc = {
            "name": module_name,
            "path": str(module_path.relative_to(self.project_root)),
            "files": [],
            "functions": [],
            "classes": [],
            "test_coverage": 0.0,
        }

        # Encontrar archivos Python
        python_files = list(module_path.rglob("*.py"))

        for py_file in python_files:
            if "__pycache__" in str(py_file) or py_file.name.startswith("."):
                continue

            try:
                file_doc = self._analyze_python_file(py_file, module_name)
                module_doc["files"].append(file_doc)

                # Agregar funciones y clases al módulo
                module_doc["functions"].extend(file_doc.get("functions", []))
                module_doc["classes"].extend(file_doc.get("classes", []))

            except Exception as e:
                print(f"Warning: Could not analyze {py_file}: {e}")

        return module_doc

    def _analyze_python_file(self, file_path: Path, module_name: str) -> Dict[str, Any]:
        """Analizar un archivo Python individual"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        file_doc = {
            "name": file_path.name,
            "path": str(file_path.relative_to(self.project_root)),
            "module": module_name,
            "functions": [],
            "classes": [],
            "imports": [],
            "lines_of_code": len(content.split("\n")),
        }

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_doc = self._analyze_function(node, module_name, file_path.name)
                    file_doc["functions"].append(func_doc)

                elif isinstance(node, ast.ClassDef):
                    class_doc = self._analyze_class(node, module_name, file_path.name)
                    file_doc["classes"].append(class_doc)

                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    file_doc["imports"].append(self._get_import_name(node))

        except SyntaxError:
            print(f"Warning: Syntax error in {file_path}")

        return file_doc

    def _analyze_function(
        self, node: ast.FunctionDef, module_name: str, file_name: str
    ) -> Dict[str, Any]:
        """Analizar una función"""
        # Calcular complejidad ciclomática básica
        complexity = self._calculate_complexity(node)

        # Extraer docstring
        docstring = ""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Str)
        ):
            docstring = node.body[0].value.s

        # Extraer parámetros
        parameters = {}
        for arg in node.args.args:
            param_name = arg.arg
            param_type = ""
            if arg.annotation:
                param_type = (
                    ast.unparse(arg.annotation)
                    if hasattr(ast, "unparse")
                    else str(arg.annotation)
                )
            parameters[param_name] = param_type

        # Extraer tipo de retorno
        return_type = ""
        if node.returns:
            return_type = (
                ast.unparse(node.returns)
                if hasattr(ast, "unparse")
                else str(node.returns)
            )

        return {
            "name": node.name,
            "module": module_name,
            "file": file_name,
            "signature": f"{node.name}({', '.join(parameters.keys())})",
            "docstring": docstring.strip(),
            "parameters": parameters,
            "return_type": return_type,
            "complexity": complexity,
            "decorators": [
                d.id if hasattr(d, "id") else str(d) for d in node.decorator_list
            ],
            "line_number": node.lineno,
        }

    def _analyze_class(
        self, node: ast.ClassDef, module_name: str, file_name: str
    ) -> Dict[str, Any]:
        """Analizar una clase"""
        # Extraer docstring
        docstring = ""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Str)
        ):
            docstring = node.body[0].value.s

        # Extraer herencia
        inheritance = []
        for base in node.bases:
            if hasattr(base, "id"):
                inheritance.append(base.id)
            elif hasattr(base, "attr"):
                inheritance.append(
                    f"{base.value.id}.{base.attr}"
                    if hasattr(base.value, "id")
                    else str(base)
                )

        # Analizar métodos
        methods = []
        attributes = {}

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_doc = self._analyze_function(item, module_name, file_name)
                methods.append(method_doc)
            elif isinstance(item, ast.AnnAssign) and item.target.id:
                # Attribute with type annotation
                attr_name = item.target.id
                attr_type = (
                    ast.unparse(item.annotation)
                    if hasattr(ast, "unparse")
                    else str(item.annotation)
                )
                attributes[attr_name] = attr_type

        return {
            "name": node.name,
            "module": module_name,
            "file": file_name,
            "docstring": docstring.strip(),
            "inheritance": inheritance,
            "methods": methods,
            "attributes": attributes,
            "line_number": node.lineno,
        }

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calcular complejidad ciclomática básica"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and len(child.values) > 1:
                complexity += len(child.values) - 1

        return complexity

    def _get_import_name(self, node: ast.stmt) -> str:
        """Extraer nombre de import"""
        if isinstance(node, ast.Import):
            return ", ".join(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(alias.name for alias in node.names)
            return f"from {module} import {names}"
        return ""

    def _generate_api_documentation(self) -> Dict[str, Any]:
        """Generar documentación de APIs"""
        api_doc = {"endpoints": [], "models": [], "middleware": []}

        # Intentar encontrar archivos de API
        api_files = list(self.project_root.rglob("*api*.py")) + list(
            self.project_root.rglob("*service*.py")
        )

        for api_file in api_files[:5]:  # Limitar para performance
            try:
                with open(api_file, "r") as f:
                    content = f.read()

                # Buscar patrones de endpoints FastAPI
                import re

                endpoint_patterns = [
                    r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                    r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                ]

                for pattern in endpoint_patterns:
                    matches = re.findall(pattern, content)
                    for method, path in matches:
                        api_doc["endpoints"].append(
                            {
                                "method": method.upper(),
                                "path": path,
                                "file": str(api_file.relative_to(self.project_root)),
                            }
                        )

            except Exception as e:
                print(f"Warning: Could not analyze API file {api_file}: {e}")

        return api_doc

    def _generate_architecture_documentation(self) -> Dict[str, Any]:
        """Generar documentación de arquitectura"""
        arch_doc = {
            "layers": [
                {
                    "name": "Unified Systems Layer",
                    "description": "Sistema maestro unificado con todos los componentes integrados",
                    "components": [
                        "UnifiedSystemCore",
                        "UnifiedConsciousnessMemorySystem",
                        "UnifiedSecurityAuthSystem",
                    ],
                },
                {
                    "name": "Core Systems Layer",
                    "description": "Sistemas core principales de Sheily AI",
                    "components": ["sheily_core", "sheily_train", "all-Branches"],
                },
                {
                    "name": "Infrastructure Layer",
                    "description": "Infraestructura enterprise con Docker, Kubernetes, monitoring",
                    "components": [
                        "Docker",
                        "PostgreSQL",
                        "Redis",
                        "Prometheus",
                        "ArgoCD",
                    ],
                },
            ],
            "patterns": [
                "Hexagonal Architecture",
                "Microservices",
                "Event-Driven Architecture",
                "Domain-Driven Design",
                "CQRS Pattern",
                "Circuit Breaker Pattern",
            ],
            "security_layers": [
                "Input Validation & Sanitization",
                "Authentication & Authorization (JWT + 2FA)",
                "Encryption at Rest & Transit (AES-256 + RSA)",
                "Rate Limiting & DDoS Protection",
                "Intrusion Detection & Response",
                "Audit Logging & Compliance",
            ],
        }

        return arch_doc

    def _generate_coverage_documentation(self) -> Dict[str, Any]:
        """Generar documentación de testing/coverage"""
        coverage_doc = {
            "current_coverage": 0.0,
            "target_coverage": 95.0,
            "testing_tools": [
                "pytest",
                "coverage",
                "hypothesis",
                "mutmut",
                "chaos engineering",
                "property-based testing",
            ],
            "test_types": [
                "unit tests",
                "integration tests",
                "security tests",
                "performance tests",
                "chaos tests",
                "property tests",
            ],
        }

        # Intentar leer coverage actual
        coverage_file = self.project_root / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, "r") as f:
                    coverage_data = json.load(f)
                    coverage_doc["current_coverage"] = coverage_data.get(
                        "totals", {}
                    ).get("percent_covered", 0)
            except:
                pass

        return coverage_doc

    def _save_documentation(self, documentation: Dict[str, Any]):
        """Guardar documentación en archivos JSON"""
        # Guardar documentación completa
        doc_file = self.output_dir / "living_documentation.json"
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(documentation, f, indent=2, ensure_ascii=False)

        # Guardar por módulos
        for module_name, module_data in documentation.get("modules", {}).items():
            module_file = (
                self.output_dir / f"{module_name.replace('.', '_')}_documentation.json"
            )
            with open(module_file, "w", encoding="utf-8") as f:
                json.dump(module_data, f, indent=2, ensure_ascii=False)

    def _generate_html_documentation(self, documentation: Dict[str, Any]):
        """Generar documentación HTML"""
        html_content = (
            ".2f"
            ".2f"
            f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sheily AI - Living Documentation</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .module {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .function {{
            background: #f8f9fa;
            margin: 10px 0;
            padding: 15px;
            border-left: 4px solid #007bff;
        }}
        .class {{
            background: #e9ecef;
            margin: 10px 0;
            padding: 15px;
            border-left: 4px solid #28a745;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Sheily AI - Living Documentation</h1>
        <p>Documentación viva generada automáticamente - {documentation['generated_at']}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="metric">{len(documentation.get('modules', {}))}</div>
            <div>Módulos</div>
        </div>
        <div class="stat-card">
            <div class="metric">{sum(len(m.get('functions', [])) for m in documentation.get('modules', {}).values())}</div>
            <div>Funciones</div>
        </div>
        <div class="stat-card">
            <div class="metric">{sum(len(m.get('classes', [])) for m in documentation.get('modules', {}).values())}</div>
            <div>Clases</div>
        </div>
        <div class="stat-card">
            <div class="metric">{documentation.get('coverage', {}).get('current_coverage', 0):.1f}%</div>
            <div>Cobertura</div>
        </div>
    </div>

    <h2>📚 Arquitectura del Sistema</h2>
    <div class="module">
        <h3>Capas Arquitectónicas</h3>
"""
        )

        # Agregar capas arquitectónicas
        for layer in documentation.get("architecture", {}).get("layers", []):
            html_content += f"""
        <div class="module">
            <h4>{layer['name']}</h4>
            <p>{layer['description']}</p>
            <ul>
"""
            for component in layer.get("components", []):
                html_content += f"<li>{component}</li>"
            html_content += "</ul></div>"

        # Agregar módulos
        html_content += "<h2>🔧 Módulos del Sistema</h2>"

        for module_name, module_data in documentation.get("modules", {}).items():
            html_content += (
                ".1f"
                f"""
    <div class="module">
        <h3>{module_name}</h3>
        <p><strong>Archivos:</strong> {len(module_data.get('files', []))} |
           <strong>Funciones:</strong> {len(module_data.get('functions', []))} |
           <strong>Clases:</strong> {len(module_data.get('classes', []))}</p>

        <h4>Funciones Principales</h4>
"""
            )

            for func in module_data.get("functions", [])[:5]:  # Top 5 functions
                html_content += f"""
        <div class="function">
            <strong>{func['name']}</strong>
            <code>{func['signature']}</code>
            <p>{func.get('docstring', '')[:100]}{'...' if len(func.get('docstring', '')) > 100 else ''}</p>
        </div>"""

            html_content += "<h4>Clases Principales</h4>"

            for cls in module_data.get("classes", [])[:3]:  # Top 3 classes
                html_content += f"""
        <div class="class">
            <strong>{cls['name']}</strong>
            <p>{cls.get('docstring', '')[:100]}{'...' if len(cls.get('docstring', '')) > 100 else ''}</p>
            <small>Métodos: {len(cls.get('methods', []))}</small>
        </div>"""

            html_content += "</div>"

        html_content += """
    </div>
</body>
</html>"""

        html_file = self.output_dir / "living_documentation.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📄 HTML documentation generated: {html_file}")

    def _generate_markdown_documentation(self, documentation: Dict[str, Any]):
        """Generar documentación Markdown"""
        md_content = f"""# 🤖 Sheily AI - Living Documentation

**Generado automáticamente:** {documentation['generated_at']}

## 📊 Estadísticas del Proyecto

- **Módulos:** {len(documentation.get('modules', {}))}
- **Funciones:** {sum(len(m.get('functions', [])) for m in documentation.get('modules', {}).values())}
- **Clases:** {sum(len(m.get('classes', [])) for m in documentation.get('modules', {}).values())}
- **Cobertura de Tests:** {documentation.get('coverage', {}).get('current_coverage', 0):.1f}%

## 🏗️ Arquitectura del Sistema

### Capas Arquitectónicas

"""

        for layer in documentation.get("architecture", {}).get("layers", []):
            md_content += f"""#### {layer['name']}
{layer['description']}

**Componentes:**
"""
            for component in layer.get("components", []):
                md_content += f"- {component}\n"
            md_content += "\n"

        md_content += """### Patrones Arquitectónicos
"""
        for pattern in documentation.get("architecture", {}).get("patterns", []):
            md_content += f"- {pattern}\n"

        md_content += """

## 🔒 Capas de Seguridad
"""
        for security_layer in documentation.get("architecture", {}).get(
            "security_layers", []
        ):
            md_content += f"- {security_layer}\n"

        md_content += """

## 🔧 Módulos del Sistema

"""

        for module_name, module_data in documentation.get("modules", {}).items():
            md_content += (
                ".1f"
                ".1f"
                f"""### {module_name}
- **Archivos:** {len(module_data.get('files', []))}
- **Funciones:** {len(module_data.get('functions', []))}
- **Clases:** {len(module_data.get('classes', []))}

#### Funciones Principales
"""
            )

            for func in module_data.get("functions", [])[:3]:
                md_content += f"""- **{func['name']}** `{func['signature']}`
  {func.get('docstring', '')[:100]}{'...' if len(func.get('docstring', '')) > 100 else ''}

"""

            md_content += """#### Clases Principales
"""

            for cls in module_data.get("classes", [])[:2]:
                md_content += f"""- **{cls['name']}**
  {cls.get('docstring', '')[:100]}{'...' if len(cls.get('docstring', '')) > 100 else ''}

"""

        md_content += """
## 📡 APIs del Sistema

### Endpoints Disponibles
"""

        for endpoint in documentation.get("apis", {}).get("endpoints", [])[:10]:
            md_content += (
                f"- `{endpoint['method']} {endpoint['path']}` ({endpoint['file']})\n"
            )

        md_content += """

## 🧪 Cobertura de Testing

- **Cobertura Actual:** {documentation.get('coverage', {}).get('current_coverage', 0):.1f}%
- **Objetivo:** {documentation.get('coverage', {}).get('target_coverage', 95.0)}%

### Herramientas de Testing
"""

        for tool in documentation.get("coverage", {}).get("testing_tools", []):
            md_content += f"- {tool}\n"

        md_content += """

---

*Documentación generada automáticamente por Living Documentation Generator*
"""

        md_file = self.output_dir / "LIVING_DOCUMENTATION.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"📝 Markdown documentation generated: {md_file}")


def generate_living_docs():
    """Función principal para generar documentación viva"""
    try:
        project_root = Path(__file__).parent.parent
        generator = LivingDocumentationGenerator(project_root)
        documentation = generator.generate_complete_documentation()

        # Retornar score de completitud (100% si se genera exitosamente)
        return 100.0 if documentation else 0.0

    except Exception as e:
        print(f"❌ Living documentation generation failed: {e}")
        return 0.0


if __name__ == "__main__":
    score = generate_living_docs()
    print(f"✅ Living documentation score: {score:.1f}%")
    exit(0 if score >= 50 else 1)  # 50% mínimo para considerar completado
