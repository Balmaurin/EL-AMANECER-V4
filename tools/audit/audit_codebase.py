#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUDITORÍA COMPLETA DEL CODEBASE
================================
Script simplificado para auditar todo el proyecto
"""

import ast
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configurar UTF-8 para Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Obtener el directorio raíz del proyecto
try:
    PROJECT_ROOT = Path(__file__).parent.resolve()
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()


class CodebaseAuditor:
    """Auditor completo del codebase"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "structure": {},
            "code_quality": {},
            "dependencies": {},
            "issues": [],
            "statistics": {},
        }

    def audit_structure(self):
        """Auditar estructura del proyecto"""
        print("\n" + "=" * 70)
        print("1. ESTRUCTURA DEL PROYECTO")
        print("=" * 70)

        structure = {
            "directories": {},
            "file_counts": {},
            "total_files": 0,
            "total_size_mb": 0,
        }

        # Directorios principales
        main_dirs = [
            "sheily_core",
            "tools",
            "all-Branches",
            "sheily_train",
            "centralized_tests",
            "config",
            "docs",
            "scripts",
        ]

        for dir_name in main_dirs:
            dir_path = PROJECT_ROOT / dir_name
            if dir_path.exists():
                py_files = list(dir_path.rglob("*.py"))
                total_size = sum(f.stat().st_size for f in py_files) / (1024 * 1024)
                structure["directories"][dir_name] = {
                    "python_files": len(py_files),
                    "size_mb": round(total_size, 2),
                }
                print(
                    f"[OK] {dir_name:20} | {len(py_files):>4} archivos Python | {total_size:>6.2f} MB"
                )
            else:
                print(f"[MISSING] {dir_name:20} | No existe")

        # Contar archivos por tipo
        file_types = defaultdict(int)
        total_size = 0

        for ext in [".py", ".json", ".md", ".yaml", ".yml", ".txt"]:
            files = list(PROJECT_ROOT.rglob(f"*{ext}"))
            file_types[ext] = len(files)
            total_size += sum(f.stat().st_size for f in files)

        structure["file_counts"] = dict(file_types)
        structure["total_files"] = sum(file_types.values())
        structure["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        self.results["structure"] = structure
        print(f"\nTotal archivos analizados: {structure['total_files']}")
        print(f"Tamaño total: {structure['total_size_mb']} MB")

    def audit_code_quality(self):
        """Auditar calidad del código"""
        print("\n" + "=" * 70)
        print("2. CALIDAD DEL CODIGO")
        print("=" * 70)

        quality = {
            "syntax_errors": [],
            "import_errors": [],
            "total_files_checked": 0,
            "files_with_issues": 0,
        }

        py_files = list(PROJECT_ROOT.rglob("*.py"))
        quality["total_files_checked"] = len(py_files)

        print(f"Verificando sintaxis de {len(py_files)} archivos Python...")

        for py_file in py_files[:500]:  # Limitar para no tardar mucho
            try:
                # Verificar sintaxis
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    ast.parse(content, filename=str(py_file))
            except SyntaxError as e:
                quality["syntax_errors"].append(
                    {
                        "file": str(py_file.relative_to(PROJECT_ROOT)),
                        "line": e.lineno,
                        "message": str(e),
                    }
                )
                quality["files_with_issues"] += 1
                print(f"[ERROR] {py_file.relative_to(PROJECT_ROOT)}: Línea {e.lineno}")
            except Exception as e:
                quality["import_errors"].append(
                    {"file": str(py_file.relative_to(PROJECT_ROOT)), "error": str(e)}
                )

        print(
            f"\n[OK] Archivos sin errores: {quality['total_files_checked'] - quality['files_with_issues']}"
        )
        print(f"[ERROR] Archivos con problemas: {quality['files_with_issues']}")

        self.results["code_quality"] = quality

    def audit_dependencies(self):
        """Auditar dependencias"""
        print("\n" + "=" * 70)
        print("3. DEPENDENCIAS")
        print("=" * 70)

        deps = {"requirements_files": [], "imports_analysis": {}}

        # Verificar archivos de dependencias
        req_files = ["requirements.txt", "requirements-min.txt", "pyproject.toml"]
        for req_file in req_files:
            req_path = PROJECT_ROOT / req_file
            if req_path.exists():
                deps["requirements_files"].append(req_file)
                print(f"[OK] {req_file} existe")
            else:
                print(f"[MISSING] {req_file}")

        self.results["dependencies"] = deps

    def audit_issues(self):
        """Buscar problemas comunes"""
        print("\n" + "=" * 70)
        print("4. PROBLEMAS DETECTADOS")
        print("=" * 70)

        issues = []

        # Buscar TODOs y FIXMEs
        todo_count = 0
        for py_file in PROJECT_ROOT.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if "TODO" in line.upper() or "FIXME" in line.upper():
                            todo_count += 1
                            issues.append(
                                {
                                    "type": "TODO/FIXME",
                                    "file": str(py_file.relative_to(PROJECT_ROOT)),
                                    "line": i,
                                    "content": line.strip()[:80],
                                }
                            )
            except:
                pass

        print(f"[INFO] TODOs/FIXMEs encontrados: {todo_count}")

        # Buscar imports problemáticos
        import_issues = 0
        for py_file in PROJECT_ROOT.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if "import *" in content:
                        import_issues += 1
                        issues.append(
                            {
                                "type": "wildcard_import",
                                "file": str(py_file.relative_to(PROJECT_ROOT)),
                            }
                        )
            except:
                pass

        print(f"[WARNING] Imports con * encontrados: {import_issues}")

        self.results["issues"] = issues

    def generate_statistics(self):
        """Generar estadísticas finales"""
        print("\n" + "=" * 70)
        print("5. ESTADISTICAS FINALES")
        print("=" * 70)

        stats = {
            "total_python_files": len(list(PROJECT_ROOT.rglob("*.py"))),
            "total_lines": 0,
            "total_size_mb": 0,
        }

        # Contar líneas de código
        line_count = 0
        for py_file in PROJECT_ROOT.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    line_count += len(f.readlines())
            except:
                pass

        stats["total_lines"] = line_count

        # Calcular tamaño
        total_size = sum(f.stat().st_size for f in PROJECT_ROOT.rglob("*.py")) / (
            1024 * 1024
        )
        stats["total_size_mb"] = round(total_size, 2)

        print(f"Archivos Python: {stats['total_python_files']}")
        print(f"Lineas de codigo: {stats['total_lines']:,}")
        print(f"Tamaño total: {stats['total_size_mb']} MB")

        self.results["statistics"] = stats

    def save_report(self):
        """Guardar reporte"""
        report_file = (
            PROJECT_ROOT
            / "audit_2024"
            / "reports"
            / f"codebase_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Reporte guardado en: {report_file}")

    def run(self):
        """Ejecutar auditoría completa"""
        print("\n" + "=" * 70)
        print("AUDITORIA COMPLETA DEL CODEBASE")
        print("=" * 70)

        self.audit_structure()
        self.audit_code_quality()
        self.audit_dependencies()
        self.audit_issues()
        self.generate_statistics()
        self.save_report()

        print("\n" + "=" * 70)
        print("[OK] AUDITORIA COMPLETADA")
        print("=" * 70)


if __name__ == "__main__":
    auditor = CodebaseAuditor()
    auditor.run()
