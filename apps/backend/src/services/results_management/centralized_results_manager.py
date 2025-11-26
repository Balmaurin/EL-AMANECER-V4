#!/usr/bin/env python3
"""
Centralized Results Management System
=====================================

Sistema avanzado para gestión de resultados centralizados con:
- Análisis de logs estructurados
- Monitoreo de rendimiento
- Alertas automáticas
- Reportes de estado
- Limpieza automática de logs antiguos
"""

import gzip
import json
import logging
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configurar logging básico para este script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CentralizedResultsManager:
    """Gestor avanzado de resultados centralizados"""

    def __init__(self, results_dir: str = "centralized_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Configuración de retención de logs
        self.log_retention_days = 30
        self.max_log_size_mb = 100

    def analyze_log_file(self, log_file: str = "sheily_ai.log") -> Dict[str, Any]:
        """Analizar archivo de log completo"""
        log_path = self.results_dir / log_file

        if not log_path.exists():
            return {"error": f"Log file {log_file} does not exist"}

        analysis = {
            "file_info": self._get_file_info(log_path),
            "log_stats": self._analyze_log_content(log_path),
            "performance_metrics": self._extract_performance_metrics(log_path),
            "error_analysis": self._analyze_errors(log_path),
            "component_activity": self._analyze_component_activity(log_path),
        }

        return analysis

    def _get_file_info(self, log_path: Path) -> Dict[str, Any]:
        """Obtener información básica del archivo"""
        stat = log_path.stat()
        return {
            "path": str(log_path),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "exists": True,
        }

    def _analyze_log_content(self, log_path: Path) -> Dict[str, Any]:
        """Analizar contenido del log"""
        stats = {
            "total_lines": 0,
            "level_counts": Counter(),
            "component_counts": Counter(),
            "time_range": {"start": None, "end": None},
            "json_lines": 0,
            "text_lines": 0,
        }

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    stats["total_lines"] = line_num

                    # Intentar parsear como JSON
                    try:
                        log_entry = json.loads(line.strip())
                        stats["json_lines"] += 1

                        # Extraer información
                        level = log_entry.get("level", "UNKNOWN")
                        stats["level_counts"][level] += 1

                        if "context" in log_entry:
                            component = log_entry["context"].get("component", "unknown")
                            stats["component_counts"][component] += 1

                        # Rango de tiempo
                        timestamp = log_entry.get("timestamp")
                        if timestamp:
                            if (
                                not stats["time_range"]["start"]
                                or timestamp < stats["time_range"]["start"]
                            ):
                                stats["time_range"]["start"] = timestamp
                            if (
                                not stats["time_range"]["end"]
                                or timestamp > stats["time_range"]["end"]
                            ):
                                stats["time_range"]["end"] = timestamp

                    except json.JSONDecodeError:
                        stats["text_lines"] += 1

        except Exception as e:
            logger.error(f"Error analyzing log content: {e}")
            stats["error"] = str(e)

        return stats

    def _extract_performance_metrics(self, log_path: Path) -> Dict[str, Any]:
        """Extraer métricas de rendimiento de los logs"""
        metrics = {
            "operation_times": [],
            "slow_operations": [],
            "performance_warnings": 0,
            "performance_errors": 0,
        }

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        message = log_entry.get("message", "")

                        # Buscar tiempos de ejecución
                        time_match = re.search(r"(\d+\.\d+)s", message)
                        if time_match:
                            execution_time = float(time_match.group(1))
                            metrics["operation_times"].append(execution_time)

                            # Clasificar operaciones lentas
                            if execution_time > 5.0:
                                metrics["performance_errors"] += 1
                                metrics["slow_operations"].append(
                                    {
                                        "time": execution_time,
                                        "message": message,
                                        "timestamp": log_entry.get("timestamp"),
                                    }
                                )
                            elif execution_time > 1.0:
                                metrics["performance_warnings"] += 1

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Error extracting performance metrics: {e}")
            metrics["error"] = str(e)

        # Calcular estadísticas
        if metrics["operation_times"]:
            metrics["stats"] = {
                "avg_time": round(
                    sum(metrics["operation_times"]) / len(metrics["operation_times"]), 3
                ),
                "max_time": max(metrics["operation_times"]),
                "min_time": min(metrics["operation_times"]),
                "total_operations": len(metrics["operation_times"]),
            }

        return metrics

    def _analyze_errors(self, log_path: Path) -> Dict[str, Any]:
        """Analizar errores en los logs"""
        error_analysis = {
            "error_count": 0,
            "warning_count": 0,
            "critical_count": 0,
            "error_types": Counter(),
            "error_components": Counter(),
            "recent_errors": [],
        }

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        level = log_entry.get("level", "").upper()

                        if level == "ERROR":
                            error_analysis["error_count"] += 1
                            error_analysis["error_types"][
                                log_entry.get("message", "Unknown")
                            ] += 1

                            if "context" in log_entry:
                                component = log_entry["context"].get(
                                    "component", "unknown"
                                )
                                error_analysis["error_components"][component] += 1

                            # Mantener últimos 10 errores
                            if len(error_analysis["recent_errors"]) < 10:
                                error_analysis["recent_errors"].append(
                                    {
                                        "timestamp": log_entry.get("timestamp"),
                                        "message": log_entry.get("message"),
                                        "component": (
                                            component
                                            if "component" in locals()
                                            else "unknown"
                                        ),
                                    }
                                )

                        elif level == "WARNING":
                            error_analysis["warning_count"] += 1
                        elif level == "CRITICAL":
                            error_analysis["critical_count"] += 1

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Error analyzing errors: {e}")
            error_analysis["error"] = str(e)

        return error_analysis

    def _analyze_component_activity(self, log_path: Path) -> Dict[str, Any]:
        """Analizar actividad por componente"""
        activity = defaultdict(
            lambda: {"logs": 0, "errors": 0, "warnings": 0, "last_activity": None}
        )

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        timestamp = log_entry.get("timestamp")

                        if "context" in log_entry:
                            component = log_entry["context"].get("component", "unknown")
                            activity[component]["logs"] += 1

                            level = log_entry.get("level", "").upper()
                            if level == "ERROR":
                                activity[component]["errors"] += 1
                            elif level == "WARNING":
                                activity[component]["warnings"] += 1

                            # Actualizar última actividad
                            if timestamp and (
                                not activity[component]["last_activity"]
                                or timestamp > activity[component]["last_activity"]
                            ):
                                activity[component]["last_activity"] = timestamp

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Error analyzing component activity: {e}")
            activity["error"] = str(e)

        return dict(activity)

    def generate_report(self) -> Dict[str, Any]:
        """Generar reporte completo de resultados"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "log_analysis": self.analyze_log_file(),
            "system_health": self._assess_system_health(),
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _assess_system_health(self) -> Dict[str, Any]:
        """Evaluar salud del sistema basado en logs"""
        analysis = self.analyze_log_file()

        health = {"overall_status": "unknown", "score": 0, "issues": [], "metrics": {}}

        if "error" in analysis:
            health["overall_status"] = "error"
            health["issues"].append(f"Log analysis failed: {analysis['error']}")
            return health

        log_stats = analysis.get("log_stats", {})
        error_analysis = analysis.get("error_analysis", {})
        perf_metrics = analysis.get("performance_metrics", {})

        # Calcular score de salud
        score = 100

        # Penalizar errores
        error_rate = error_analysis.get("error_count", 0) / max(
            log_stats.get("total_lines", 1), 1
        )
        if error_rate > 0.1:  # >10% errores
            score -= 30
            health["issues"].append(f"High error rate: {error_rate:.1%}")
        elif error_rate > 0.05:  # >5% errores
            score -= 15
            health["issues"].append(f"Moderate error rate: {error_rate:.1%}")

        # Penalizar operaciones lentas
        slow_ops = len(perf_metrics.get("slow_operations", []))
        if slow_ops > 10:
            score -= 20
            health["issues"].append(f"Too many slow operations: {slow_ops}")

        # Verificar tamaño del log
        file_size_mb = analysis.get("file_info", {}).get("size_mb", 0)
        if file_size_mb > self.max_log_size_mb:
            score -= 10
            health["issues"].append(f"Log file too large: {file_size_mb}MB")

        health["score"] = max(0, score)

        if health["score"] >= 80:
            health["overall_status"] = "healthy"
        elif health["score"] >= 60:
            health["overall_status"] = "warning"
        else:
            health["overall_status"] = "critical"

        # Métricas adicionales
        health["metrics"] = {
            "total_logs": log_stats.get("total_lines", 0),
            "error_count": error_analysis.get("error_count", 0),
            "warning_count": error_analysis.get("warning_count", 0),
            "avg_operation_time": perf_metrics.get("stats", {}).get("avg_time", 0),
            "log_size_mb": file_size_mb,
        }

        return health

    def _generate_recommendations(self) -> List[str]:
        """Generar recomendaciones basadas en el análisis"""
        analysis = self.analyze_log_file()
        recommendations = []

        if "error" in analysis:
            recommendations.append("Fix log analysis errors before proceeding")
            return recommendations

        log_stats = analysis.get("log_stats", {})
        error_analysis = analysis.get("error_analysis", {})
        perf_metrics = analysis.get("performance_metrics", {})

        # Recomendaciones basadas en errores
        if error_analysis.get("error_count", 0) > 50:
            recommendations.append(
                "High error count detected - review error patterns and fix root causes"
            )

        # Recomendaciones de rendimiento
        if perf_metrics.get("stats", {}).get("avg_time", 0) > 2.0:
            recommendations.append(
                "Average operation time is high - optimize performance bottlenecks"
            )

        slow_ops = len(perf_metrics.get("slow_operations", []))
        if slow_ops > 5:
            recommendations.append(
                f"Found {slow_ops} slow operations - investigate and optimize"
            )

        # Recomendaciones de tamaño
        file_size_mb = analysis.get("file_info", {}).get("size_mb", 0)
        if file_size_mb > self.max_log_size_mb:
            recommendations.append(
                f"Log file size ({file_size_mb}MB) exceeds limit - consider log rotation"
            )

        # Recomendaciones de componentes
        component_activity = analysis.get("component_activity", {})
        inactive_components = [
            comp
            for comp, data in component_activity.items()
            if not data.get("last_activity")
        ]
        if inactive_components:
            recommendations.append(
                f"Components with no recent activity: {inactive_components}"
            )

        if not recommendations:
            recommendations.append("System appears healthy - continue monitoring")

        return recommendations

    def cleanup_old_logs(self) -> Dict[str, Any]:
        """Limpiar logs antiguos"""
        cleanup_results = {
            "files_processed": 0,
            "files_compressed": 0,
            "files_deleted": 0,
            "space_saved_mb": 0,
        }

        cutoff_date = datetime.now() - timedelta(days=self.log_retention_days)

        for log_file in self.results_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    # Comprimir archivos antiguos
                    compressed_path = log_file.with_suffix(f"{log_file.suffix}.gz")
                    with open(log_file, "rb") as f_in:
                        with gzip.open(compressed_path, "wb") as f_out:
                            f_out.writelines(f_in)

                    original_size = log_file.stat().st_size
                    log_file.unlink()  # Eliminar original

                    cleanup_results["files_compressed"] += 1
                    cleanup_results["space_saved_mb"] += original_size / (1024 * 1024)

                except Exception as e:
                    logger.error(f"Error compressing {log_file}: {e}")

            cleanup_results["files_processed"] += 1

        return cleanup_results


def main():
    """Función principal para gestión de resultados"""
    manager = CentralizedResultsManager()

    print("=== Analyzing centralized results ===")
    report = manager.generate_report()

    print("\n[*] Results Report:")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    # Ejecutar limpieza si es necesario
    health = report.get("system_health", {})
    if health.get("score", 0) < 80:
        print("\n[*] Running log cleanup...")
        cleanup = manager.cleanup_old_logs()
        print(f"[+] Cleanup completed: {cleanup}")


if __name__ == "__main__":
    main()
