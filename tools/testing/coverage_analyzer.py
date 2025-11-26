#!/usr/bin/env python3
"""
Coverage Analysis Tool - Sheily AI
==================================

Herramienta avanzada para analizar y mejorar la cobertura de testing.
Identifica archivos con baja cobertura y genera recomendaciones.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


class CoverageAnalyzer:
    """
    Analizador avanzado de cobertura de código
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.coverage_data: Dict[str, Any] = {}
        self.target_coverage = 95.0

    def run_coverage_analysis(self) -> Dict[str, Any]:
        """
        Ejecutar análisis completo de cobertura

        Returns:
            Dict con resultados del análisis
        """
        print("📊 SHEILY AI - COVERAGE ANALYSIS")
        print("=" * 50)

        try:
            # Ejecutar pytest con coverage
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "centralized_tests/",
                "--cov=sheily_core",
                "--cov=sheily_train",
                "--cov=all-Branches",
                "--cov-report=json",
                "--cov-report=html",
                "--cov-report=xml",
                "--cov-report=term-missing",
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutos
            )

            # Parsear resultados
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, "r") as f:
                    self.coverage_data = json.load(f)

            analysis = self._analyze_coverage_results(result, self.coverage_data)

            self._print_coverage_report(analysis)
            return analysis

        except subprocess.TimeoutExpired:
            print("⏰ Coverage analysis timed out")
            return {"error": "timeout"}
        except Exception as e:
            print(f"❌ Coverage analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_coverage_results(
        self, pytest_result: subprocess.CompletedProcess, coverage_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analizar resultados de cobertura"""

        analysis = {
            "overall_coverage": 0.0,
            "target_coverage": self.target_coverage,
            "coverage_gap": 0.0,
            "files_analyzed": 0,
            "files_low_coverage": [],
            "files_missing_tests": [],
            "recommendations": [],
        }

        if "totals" in coverage_data:
            totals = coverage_data["totals"]
            analysis["overall_coverage"] = totals.get("percent_covered", 0)
            analysis["files_analyzed"] = totals.get("num_statements", 0)

        analysis["coverage_gap"] = max(
            0, self.target_coverage - analysis["overall_coverage"]
        )

        # Analizar archivos individuales
        if "files" in coverage_data:
            for file_path, file_data in coverage_data["files"].items():
                coverage_percent = file_data.get("summary", {}).get(
                    "percent_covered", 0
                )

                # Archivos con baja cobertura (< 80%)
                if coverage_percent < 80:
                    analysis["files_low_coverage"].append(
                        {
                            "file": file_path,
                            "coverage": coverage_percent,
                            "lines_covered": file_data.get("summary", {}).get(
                                "covered_lines", 0
                            ),
                            "lines_total": file_data.get("summary", {}).get(
                                "num_statements", 0
                            ),
                        }
                    )

                # Archivos sin cobertura (0%)
                if coverage_percent == 0:
                    analysis["files_missing_tests"].append(file_path)

        # Generar recomendaciones
        analysis["recommendations"] = self._generate_coverage_recommendations(analysis)

        return analysis

    def _generate_coverage_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generar recomendaciones para mejorar cobertura"""
        recommendations = []

        if analysis["overall_coverage"] < self.target_coverage:
            gap = analysis["coverage_gap"]
            recommendations.append(
                f"CRÍTICO: Coverage {analysis['overall_coverage']:.1f}% below {self.target_coverage:.1f}% target. Add tests for uncovered code."
            )
        if analysis["files_low_coverage"]:
            recommendations.append(
                f"📝 PRIORIDAD: Mejorar cobertura en {len(analysis['files_low_coverage'])} archivos con baja cobertura"
            )

            # Mostrar top 5 archivos con peor cobertura
            worst_files = sorted(
                analysis["files_low_coverage"], key=lambda x: x["coverage"]
            )[:5]

            for file_info in worst_files:
                recommendations.append(
                    f"   • {file_info['file']}: {file_info['coverage']:.1f}% coverage"
                )

        if analysis["files_missing_tests"]:
            recommendations.append(
                f"🚨 CRÍTICO: {len(analysis['files_missing_tests'])} archivos sin tests"
            )

            for file_path in analysis["files_missing_tests"][:3]:  # Top 3
                recommendations.append(f"   • {file_path}")

        if analysis["overall_coverage"] >= 95:
            recommendations.append("🎉 ¡EXCELENTE! Cobertura objetivo alcanzada")

        return recommendations

    def _print_coverage_report(self, analysis: Dict[str, Any]):
        """Imprimir reporte de cobertura"""
        print("\n📈 COVERAGE ANALYSIS REPORT")
        print("=" * 50)

        overall = analysis["overall_coverage"]
        target = analysis["target_coverage"]
        gap = analysis["coverage_gap"]

        print(f"Overall Coverage: {overall:.1f}%")
        print(f"Target Coverage: {target:.1f}%")
        print(f"Coverage Gap: {gap:.1f}%")

        # Evaluar estado
        if overall >= target:
            print("🎉 STATUS: TARGET ACHIEVED!")
        elif overall >= 80:
            print("✅ STATUS: GOOD - Minor improvements needed")
        elif overall >= 60:
            print("⚠️ STATUS: FAIR - Significant improvements needed")
        else:
            print("❌ STATUS: POOR - Major testing effort required")

        print(f"\nFiles Analyzed: {analysis['files_analyzed']}")
        print(f"Files Low Coverage: {len(analysis['files_low_coverage'])}")
        print(f"Files Missing Tests: {len(analysis['files_missing_tests'])}")

        if analysis["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in analysis["recommendations"]:
                print(f"   {rec}")

        print("\n" + "=" * 50)


def run_coverage_analysis():
    """Función principal para ejecutar análisis de cobertura"""
    try:
        project_root = Path(__file__).parent.parent
        analyzer = CoverageAnalyzer(project_root)
        results = analyzer.run_coverage_analysis()

        # Retornar score de cobertura para validación
        return results.get("overall_coverage", 0.0)

    except Exception as e:
        print(f"❌ Coverage analysis failed: {e}")
        return 0.0


if __name__ == "__main__":
    score = run_coverage_analysis()
    # Exit con código basado en si alcanza el target mínimo (80%)
    success = score >= 80
    sys.exit(0 if success else 1)
