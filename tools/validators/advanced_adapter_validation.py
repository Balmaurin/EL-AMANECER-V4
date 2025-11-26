#!/usr/bin/env python3
"""
Sistema de Validación Avanzada de Adaptadores LoRA
-------------------------------------------------
Herramientas para validar adaptadores LoRA (configuración, modelo y estructura)
con generación de reportes y métricas agregadas.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Funciones base de validación
# ---------------------------------------------------------------------------


def validate_adapter_comprehensive(adapter_path):
    """Validación comprehensiva de adaptador"""
    results = {
        "valid": False,
        "score": 0,
        "max_score": 100,
        "checks": {},
        "recommendations": [],
    }

    try:
        # 1. Verificar archivos requeridos
        config_file = adapter_path / "adapter_config.json"
        model_file = adapter_path / "adapter_model.safetensors"

        if config_file.exists():
            results["checks"]["config_exists"] = True
            results["score"] += 20

            # Verificar contenido del config
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)

                if "base_model_name" in config:
                    results["checks"]["config_valid"] = True
                    results["score"] += 15
                else:
                    results["checks"]["config_incomplete"] = True
                    results["recommendations"].append(
                        "Config incompleto: falta base_model_name"
                    )
            except:
                results["checks"]["config_corrupted"] = True
                results["recommendations"].append("Config JSON corrupto")
        else:
            results["checks"]["config_missing"] = True
            results["recommendations"].append("Falta archivo adapter_config.json")

        # 2. Verificar modelo
        if model_file.exists():
            results["checks"]["model_exists"] = True
            results["score"] += 25

            size = model_file.stat().st_size
            if size >= 100000:  # 100KB mínimo
                results["checks"]["model_size_ok"] = True
                results["score"] += 20
            elif size >= 50000:  # 50KB aceptable
                results["checks"]["model_size_acceptable"] = True
                results["score"] += 10
                results["recommendations"].append(
                    f"Tamaño pequeño: {size/1024:.1f}KB (recomendado >100KB)"
                )
            else:
                results["checks"]["model_size_small"] = True
                results["recommendations"].append(
                    f"Tamaño muy pequeño: {size/1024:.1f}KB"
                )

            # Verificar extensión correcta
            if model_file.suffix == ".safetensors":
                results["checks"]["model_format_ok"] = True
                results["score"] += 10
            else:
                results["checks"]["model_format_wrong"] = True
                results["recommendations"].append("Formato de modelo incorrecto")
        else:
            results["checks"]["model_missing"] = True
            results["recommendations"].append("Falta archivo adapter_model.safetensors")

        # 3. Verificar estructura de directorio
        if adapter_path.exists() and adapter_path.is_dir():
            results["checks"]["directory_ok"] = True
            results["score"] += 10

        # Determinar validez
        results["valid"] = results["score"] >= 70  # 70% mínimo para ser válido

        return results

    except Exception as e:
        results["checks"]["error"] = str(e)
        results["recommendations"].append(f"Error durante validación: {e}")
        return results


def validate_all_adapters(base_path):
    """Validar todos los adaptadores en un directorio"""
    results = {"total": 0, "valid": 0, "invalid": 0, "details": {}}

    base_path = Path(base_path)
    if not base_path.exists():
        return results

    # Buscar todas las ramas
    for adapter_dir in base_path.iterdir():
        if adapter_dir.is_dir():
            branch_name = adapter_dir.name
            results["total"] += 1

            validation = validate_adapter_comprehensive(adapter_dir)
            results["details"][branch_name] = validation

            if validation["valid"]:
                results["valid"] += 1
            else:
                results["invalid"] += 1

    return results


def generate_validation_report(results, output_file):
    """Generar reporte de validación"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_adapters": results["total"],
            "valid_adapters": results["valid"],
            "invalid_adapters": results["invalid"],
            "success_rate": (
                results["valid"] / results["total"] * 100 if results["total"] > 0 else 0
            ),
        },
        "details": results["details"],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


class AdvancedValidator:
    """Validador avanzado para adaptadores LoRA.

    Ofrece métodos para validar un adaptador individual, validar todos los
    adaptadores en un directorio y generar un reporte JSON con estadísticas.
    """

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)

    def validate_adapter(self, adapter_path: str | Path):
        return validate_adapter_comprehensive(Path(adapter_path))

    def validate_all(self):
        return validate_all_adapters(self.base_path)

    def generate_report(self, output_file: str):
        results = self.validate_all()
        return generate_validation_report(results, output_file)


def main():
    """Función principal de validación (CLI)."""
    if len(sys.argv) != 2:
        print("Uso: python advanced_validation.py <directorio_adaptadores>")
        return 1

    base_path = sys.argv[1]
    print(f"VALIDACIÓN AVANZADA: {base_path}")

    # Ejecutar validación
    results = validate_all_adapters(base_path)

    # Generar reporte
    report_file = "audit_2024/reports/validation_report.json"
    report = generate_validation_report(results, report_file)

    # Mostrar resumen
    print("\nRESULTADOS DE VALIDACIÓN")
    print(f"Válidos: {results['valid']}/{results['total']}")
    print(f"Inválidos: {results['invalid']}/{results['total']}")
    print(f"Tasa de éxito: {report['summary']['success_rate']:.1f}%")
    print(f"Reporte generado: {report_file}")

    return 0 if results["valid"] > 0 else 1


if __name__ == "__main__":
    exit(main())
