#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

report_file = Path("audit_2024/reports/codebase_audit_20251107_031606.json")

with open(report_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("RESUMEN EJECUTIVO - AUDITORIA DEL CODEBASE")
print("=" * 70)
print(f"\nTimestamp: {data['timestamp']}")

print("\n1. ESTRUCTURA DEL PROYECTO:")
print(f"   Total archivos analizados: {data['structure']['total_files']:,}")
print(f"   Tamaño total: {data['structure']['total_size_mb']:.2f} MB")
print("\n   Directorios principales:")
for dir_name, info in data["structure"]["directories"].items():
    print(
        f"     - {dir_name:20} | {info['python_files']:>4} archivos | {info['size_mb']:>6.2f} MB"
    )

print("\n2. CALIDAD DE CODIGO:")
print(
    f"   Archivos Python verificados: {data['code_quality']['total_files_checked']:,}"
)
print(
    f"   Archivos sin errores: {data['code_quality']['total_files_checked'] - data['code_quality']['files_with_issues']:,}"
)
print(f"   Archivos con problemas: {data['code_quality']['files_with_issues']}")
if data["code_quality"]["syntax_errors"]:
    print(f"   Errores de sintaxis: {len(data['code_quality']['syntax_errors'])}")
    print("   Primeros errores:")
    for err in data["code_quality"]["syntax_errors"][:3]:
        print(f"     - {err['file']}: Línea {err['line']}")

print("\n3. DEPENDENCIAS:")
print(
    f"   Archivos de dependencias encontrados: {len(data['dependencies']['requirements_files'])}"
)
for req in data["dependencies"]["requirements_files"]:
    print(f"     - {req}")

print("\n4. PROBLEMAS DETECTADOS:")
todos = [i for i in data["issues"] if i["type"] == "TODO/FIXME"]
wildcards = [i for i in data["issues"] if i["type"] == "wildcard_import"]
print(f"   TODOs/FIXMEs encontrados: {len(todos):,}")
print(f"   Imports con * (wildcard): {len(wildcards):,}")

print("\n5. ESTADISTICAS FINALES:")
print(f"   Total archivos Python: {data['statistics']['total_python_files']:,}")
print(f"   Total líneas de código: {data['statistics']['total_lines']:,}")
print(f"   Tamaño total código: {data['statistics']['total_size_mb']:.2f} MB")

print("\n" + "=" * 70)
print("AUDITORIA COMPLETADA EXITOSAMENTE")
print("=" * 70)
