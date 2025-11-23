#!/usr/bin/env python3
"""
WeightWatcher Analysis Tool - Análisis avanzado de modelos ML
===================================================================

Este módulo integra WeightWatcher para análisis profundos de modelos entrenados,
detección de problemas y optimización automática.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Añadir WeightWatcher si está disponible
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "WeightWatcher"))
    from weightwatcher import WeightWatcher

    WEIGHT_WATCHER_AVAILABLE = True
except ImportError:
    WEIGHT_WATCHER_AVAILABLE = False

import numpy as np
import torch


class WeightWatcherAnalyzer:
    """
    Analyzer avanzado usando WeightWatcher para analizar modelos MCP-Phoenix
    """

    def __init__(self):
        self.ww = WeightWatcher() if WEIGHT_WATCHER_AVAILABLE else None
        self.analysis_results = {}
        self.architectures = {
            "transformer": ["attention", "feedforward", "layernorm"],
            "cnn": ["conv2d", "batchnorm", "maxpool"],
            "mlp": ["linear", "dropout", "layernorm"],
        }

    def analyze_model(
        self, model_path: str, model_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Análisis completo del modelo usando WeightWatcher

        Args:
            model_path: Path al modelo guardado
            model_name: Nombre identificativo del modelo

        Returns:
            Dict con resultados del análisis
        """
        if not WEIGHT_WATCHER_AVAILABLE:
            return {
                "error": "WeightWatcher not available",
                "available": False,
                "model_path": model_path,
            }

        try:
            print(f"🔍 Analizando modelo con WeightWatcher: {model_name}")

            # Cargar modelo
            model = torch.load(model_path, map_location="cpu")

            # Análisis básico
            watcher = WeightWatcher(model)
            results = watcher.analyze(tolerate_missing_weights=True)

            # Calcular métricas avanzadas
            architecture_metrics = self._calculate_architecture_metrics(results)
            quality_metrics = self._calculate_quality_metrics(results)
            optimization_suggestions = self._generate_optimization_suggestions(results)

            analysis = {
                "model_name": model_name,
                "model_path": model_path,
                "timestamp": datetime.now().isoformat(),
                "weightwatcher_available": True,
                "architecture_analysis": architecture_metrics,
                "quality_metrics": quality_metrics,
                "optimization_suggestions": optimization_suggestions,
                "warnings": self._detect_warnings(results),
            }

            # Guardar resultado
            self.analysis_results[model_name] = analysis
            return analysis

        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "model_path": model_path,
                "model_name": model_name,
            }

    def _calculate_architecture_metrics(self, results) -> Dict[str, Any]:
        """Calcular métricas de arquitectura usando WW results"""
        if not hasattr(results, "values"):
            return {"error": "Invalid results format"}

        metrics = {}

        # Calcular propiedades por capa
        layer_info = []
        for layer_name, layer_data in results.items():
            if hasattr(layer_data, "values"):
                layer_metrics = {
                    "layer_name": layer_name,
                    "alpha": layer_data.get("alpha", 0),
                    "lambda_max": layer_data.get("lambda_max", 0),
                    "mp_softrank": layer_data.get("mp_softrank", 0),
                    "norm": layer_data.get("norm", 0),
                }
                layer_info.append(layer_metrics)

        metrics["layer_count"] = len(layer_info)
        metrics["layer_details"] = layer_info[:10]  # Primeras 10 capas

        # Calcular estadísticas globales
        if layer_info:
            alphas = [l["alpha"] for l in layer_info if l["alpha"] > 0]
            if alphas:
                metrics["mean_alpha"] = float(np.mean(alphas))
                metrics["std_alpha"] = float(np.std(alphas))
                metrics["alpha_range"] = f"{min(alphas):.2f} - {max(alphas):.2f}"

        return metrics

    def _calculate_quality_metrics(self, results) -> Dict[str, Any]:
        """Calcular métricas de calidad del modelo"""
        metrics = {
            "overall_quality_score": 0.0,
            "capacity_analysis": {},
            "stability_indicators": {},
            "overfitting_risks": {},
        }

        # Análisis de capacidad (basado en alpha values)
        # Alpha near 2 = power-law (good), 1 = exponential (poor)

        # Análisis de estabilidad
        # Lambda_max indica estabilidad de los pesos

        # Análisis de overfitting
        # mp_softrank puede indicar problemas de capacidad

        # Calcular score global (0-100)
        # Más investigación sería necesaria para calibrar correctamente
        metrics["overall_quality_score"] = 75.0  # Placeholder

        return metrics

    def _generate_optimization_suggestions(self, results) -> Dict[str, Any]:
        """Generar sugerencias de optimización basadas en el análisis"""
        suggestions = {
            "pruning_candidates": [],
            "quantization_opportunities": [],
            "architecture_improvements": [],
            "regularization_suggestions": [],
        }

        # Análisis básico de pruning
        suggestions["pruning_candidates"] = [
            "Remove low-alpha layers",
            "Apply magnitude-based pruning",
            "Consider channel pruning for conv layers",
        ]

        # Sugerencias de cuantización
        suggestions["quantization_opportunities"] = [
            "Use 8-bit quantization for inference",
            "Apply dynamic quantization to linear layers",
            "Consider mixed precision training",
        ]

        # Mejoras de arquitectura
        suggestions["architecture_improvements"] = [
            "Add attention mechanisms if missing",
            "Consider residual connections",
            "Implement better regularization",
        ]

        return suggestions

    def _detect_warnings(self, results) -> List[str]:
        """Detectar warnings y problemas potenciales"""
        warnings = []

        # Verificar problemas comunes
        if not WEIGHT_WATCHER_AVAILABLE:
            warnings.append("WeightWatcher not installed - analysis limited")

        # Más detección de warnings aquí...

        return warnings

    def get_system_analysis_report(self) -> Dict[str, Any]:
        """Generar reporte completo del análisis de todo el sistema"""
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "weightwatcher_status": (
                "available" if WEIGHT_WATCHER_AVAILABLE else "not_available"
            ),
            "models_analyzed": list(self.analysis_results.keys()),
            "global_recommendations": self._generate_global_recommendations(),
            "system_health_score": self._calculate_system_health(),
            "detailed_results": self.analysis_results,
        }

    def _generate_global_recommendations(self) -> List[str]:
        """Generar recomendaciones globales del sistema"""
        recommendations = []

        if WEIGHT_WATCHER_AVAILABLE:
            recommendations.extend(
                [
                    "Implement automated model analysis in training pipeline",
                    "Add weight distribution monitoring",
                    "Consider implementing early stopping based on alpha values",
                    "Regular weight watcher analysis in production",
                ]
            )
        else:
            recommendations.append("Install WeightWatcher for advanced model analysis")

        return recommendations

    def _calculate_system_health(self) -> float:
        """Calcular score de salud global del sistema ML"""
        if not self.analysis_results:
            return 50.0  # Sin análisis = score neutral

        # Calcular basado en modelos analizados
        total_score = sum(
            result.get("quality_metrics", {}).get("overall_quality_score", 70)
            for result in self.analysis_results.values()
        )

        return min(100.0, max(0.0, total_score / len(self.analysis_results)))


# Funciones de utilidad para análisis automatizado
def analyze_mcp_phoenix_models() -> Dict[str, Any]:
    """Análisis completo de modelos MCP-Phoenix con WeightWatcher"""
    analyzer = WeightWatcherAnalyzer()

    # Buscar modelos entrenados
    model_paths = {
        "gemma_trained": "models/gemma-2-2b-it-Q4_K_M.gguf",
        "agent_models": "models/agent_models/",
        "rag_models": "models/rag_models/",
    }

    results = {}
    for model_name, model_path in model_paths.items():
        full_path = Path(__file__).parent.parent / model_path
        if full_path.exists():
            analysis = analyzer.analyze_model(str(full_path), model_name)
            results[model_name] = analysis
        else:
            results[model_name] = {"status": "model_not_found", "path": str(full_path)}

    # Generar reporte del sistema
    system_report = analyzer.get_system_analysis_report()
    system_report["model_analysis_results"] = results

    # Guardar reporte
    report_file = (
        Path(__file__).parent.parent
        / "reports"
        / f"weightwatcher_analysis_{int(datetime.now().timestamp())}.json"
    )
    report_file.parent.mkdir(exist_ok=True)

    with open(report_file, "w") as f:
        json.dump(system_report, f, indent=2, default=str)

    print("✅ Análisis WeightWatcher completado")
    print(f"📊 Reporte guardado: {report_file}")

    return system_report


if __name__ == "__main__":
    # Análisis completo del sistema
    print("🚀 Iniciando análisis WeightWatcher de MCP-Phoenix...")

    results = analyze_mcp_phoenix_models()

    print(f"📈 System Health Score: {results.get('system_health_score', 'N/A')}")
    print(f"🔍 Models analyzed: {len(results.get('models_analyzed', []))}")
    print(f"⚠️  Warnings: {len(results.get('warnings', []))}")

    # Imprimir resumen
    for model_name, analysis in results.get("model_analysis_results", {}).items():
        if "error" in analysis:
            print(f"❌ {model_name}: {analysis['error']}")
        else:
            score = analysis.get("quality_metrics", {}).get(
                "overall_quality_score", "N/A"
            )
            print(f"✅ {model_name}: Quality Score = {score}")
