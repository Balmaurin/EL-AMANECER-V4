#!/usr/bin/env python3
"""
Auto Documentation Generator - Sheily AI
========================================

Generador de documentación automática simple pero efectivo.
Crea documentación técnica básica a partir del código.
"""

import ast
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class AutoDocGenerator:
    """Generador automático de documentación"""

    def __init__(self, source_dirs: List[str] = None):
        self.source_dirs = source_dirs or ["sheily_core", "sheily_train"]
        self.docs_dir = Path("docs/auto_generated")
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def generate_docs(self) -> Dict[str, Any]:
        """Generar documentación completa"""
        print("📚 Generating Auto Documentation...")

        documentation = {"modules": {}, "classes": [], "functions": [], "summary": {}}

        total_modules = 0
        total_classes = 0
        total_functions = 0

        for source_dir in self.source_dirs:
            source_path = Path(source_dir)
            if not source_path.exists():
                continue

            print(f"  Processing {source_dir}...")

            for py_file in source_path.rglob("*.py"):
                if "__pycache__" in str(py_file) or py_file.name.startswith("."):
                    continue

                try:
                    module_doc = self._analyze_file(py_file)
                    if module_doc:
                        documentation["modules"][
                            str(py_file.relative_to("."))
                        ] = module_doc
                        total_modules += 1
                        total_classes += len(module_doc.get("classes", []))
                        total_functions += len(module_doc.get("functions", []))

                except Exception as e:
                    print(f"    Warning: Could not process {py_file}: {e}")

        documentation["summary"] = {
            "total_modules": total_modules,
            "total_classes": total_classes,
            "total_functions": total_functions,
            "generated_at": "2025-10-31",
        }

        # Guardar documentación
        self._save_docs(documentation)

        print("✅ Auto documentation generated!")
        return documentation

    def _analyze_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analizar un archivo Python"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            module_doc = {
                "name": file_path.stem,
                "path": str(file_path),
                "classes": [],
                "functions": [],
                "lines_of_code": len(content.split("\n")),
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node, file_path)
                    if class_info:
                        module_doc["classes"].append(class_info)

                elif isinstance(node, ast.FunctionDef) and not any(
                    isinstance(parent, ast.ClassDef)
                    for parent in self._get_parents(tree, node)
                ):
                    func_info = self._extract_function_info(node, file_path)
                    if func_info:
                        module_doc["functions"].append(func_info)

            return (
                module_doc if module_doc["classes"] or module_doc["functions"] else None
            )

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def _extract_class_info(
        self, node: ast.ClassDef, file_path: Path
    ) -> Dict[str, Any]:
        """Extraer información de una clase"""
        # Obtener docstring
        docstring = ""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Str)
        ):
            docstring = node.body[0].value.s

        # Obtener métodos
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item, file_path)
                if method_info:
                    methods.append(method_info)

        return {
            "name": node.name,
            "docstring": docstring.strip(),
            "methods": methods,
            "line_number": node.lineno,
        }

    def _extract_function_info(
        self, node: ast.FunctionDef, file_path: Path
    ) -> Dict[str, Any]:
        """Extraer información de una función"""
        # Obtener docstring
        docstring = ""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Str)
        ):
            docstring = node.body[0].value.s

        # Calcular complejidad básica
        complexity = 1  # base
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1

        # Extraer parámetros
        parameters = []
        for arg in node.args.args:
            param_name = arg.arg
            param_type = ""
            if arg.annotation:
                param_type = (
                    ast.unparse(arg.annotation)
                    if hasattr(ast, "unparse")
                    else str(arg.annotation)
                )
            parameters.append(
                f"{param_name}: {param_type}" if param_type else param_name
            )

        return {
            "name": node.name,
            "signature": f"{node.name}({', '.join(parameters)})",
            "docstring": docstring.strip(),
            "complexity": complexity,
            "parameters": parameters,
            "line_number": node.lineno,
        }

    def _get_parents(self, tree: ast.AST, node: ast.AST) -> List[ast.AST]:
        """Obtener nodos padre de un nodo"""
        parents = []
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                if child == node:
                    parents.append(parent)
        return parents

    def _save_docs(self, documentation: Dict[str, Any]):
        """Guardar documentación generada"""

        # Guardar documentación completa
        with open(
            self.docs_dir / "auto_documentation.json", "w", encoding="utf-8"
        ) as f:
            json.dump(documentation, f, indent=2, ensure_ascii=False)

        # Generar README de documentación
        readme_content = "# 🤖 Sheily AI - Auto Generated Documentation\n\n"
        readme_content += (
            f"**Generated:** {documentation['summary']['generated_at']}\n\n"
        )

        summary = documentation["summary"]
        readme_content += "## 📊 Summary\n\n"
        readme_content += f"- **Modules:** {summary['total_modules']}\n"
        readme_content += f"- **Classes:** {summary['total_classes']}\n"
        readme_content += f"- **Functions:** {summary['total_functions']}\n\n"

        # Agregar módulos principales
        readme_content += "## 📚 Modules\n\n"
        for module_path, module_info in list(documentation["modules"].items())[
            :10
        ]:  # Top 10
            readme_content += f"### {module_info['name']}\n\n"
            readme_content += f"- **Path:** {module_path}\n"
            readme_content += f"- **Classes:** {len(module_info.get('classes', []))}\n"
            readme_content += (
                f"- **Functions:** {len(module_info.get('functions', []))}\n"
            )
            readme_content += f"- **Lines:** {module_info.get('lines_of_code', 0)}\n\n"

            # Agregar algunas clases principales
            classes = module_info.get("classes", [])[:3]
            if classes:
                readme_content += "**Main Classes:**\n"
                for cls in classes:
                    readme_content += (
                        f"- `{cls['name']}`: {cls.get('docstring', '')[:50]}...\n"
                    )
                readme_content += "\n"

        readme_content += "---\n\n"
        readme_content += "*This documentation was auto-generated from source code*\n"

        with open(self.docs_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

        print(f"📄 Documentation saved to {self.docs_dir}")


def generate_auto_docs():
    """Función principal para generar documentación automática"""
    try:
        generator = AutoDocGenerator()
        docs = generator.generate_docs()

        summary = docs["summary"]
        print("\n📊 AUTO DOCUMENTATION SUMMARY")
        print("=" * 40)
        print(f"Modules: {summary['total_modules']}")
        print(f"Classes: {summary['total_classes']}")
        print(f"Functions: {summary['total_functions']}")
        print(f"Generated: {summary['generated_at']}")

        return True

    except Exception as e:
        print(f"❌ Auto documentation failed: {e}")
        return False


if __name__ == "__main__":
    success = generate_auto_docs()
    exit(0 if success else 1)
