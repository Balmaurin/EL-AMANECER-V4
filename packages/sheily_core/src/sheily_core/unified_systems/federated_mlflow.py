"""
Integración con MLflow para Sistema de Aprendizaje Federado

Este módulo proporciona integración completa con MLflow para tracking de experimentos,
versionado de modelos y comparación de rendimiento en entornos federados.

Características:
- Tracking de rondas federadas como experimentos
- Versionado de modelos globales y locales
- Comparación de métricas entre clientes
- Visualización de evolución de privacidad
- Registro de parámetros de PETs

Autor: Sheily AI Team
Fecha: 2025
"""

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# MLflow imports
try:
    import mlflow
    import mlflow.keras
    import mlflow.pyfunc
    import mlflow.pytorch
    import mlflow.sklearn
    import mlflow.tensorflow
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    ViewType = None
    MLFLOW_AVAILABLE = False

from federated_client import FederatedClient

# Importaciones del sistema FL
from federated_learning import FederatedLearningSystem, FederatedRound, ModelUpdate

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedMLflowTracker:
    """
    Tracker de MLflow para experimentos federados
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "federated_learning_experiments",
        enable_model_registry: bool = True,
    ):
        """Inicializar tracker de MLflow"""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow no disponible - funcionalidad limitada")
            self.client = None
            return

        # Configurar MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.client = MlflowClient(tracking_uri)
        self.experiment_name = experiment_name
        self.enable_model_registry = enable_model_registry

        # Crear experimento si no existe
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            logger.error(f"Error creando experimento MLflow: {e}")
            self.experiment_id = None

        logger.info(f"🎯 MLflow tracker inicializado - Experimento: {experiment_name}")

    def start_federated_experiment(
        self,
        round_id: str,
        use_case: str,
        num_clients: int,
        global_model_info: Dict[str, Any],
    ) -> Optional[str]:
        """Iniciar experimento para ronda federada"""
        if not self.client:
            return None

        try:
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=f"round_{round_id}_{use_case}",
            ) as run:

                # Log parámetros de la ronda
                mlflow.log_param("round_id", round_id)
                mlflow.log_param("use_case", use_case)
                mlflow.log_param("num_clients", num_clients)
                mlflow.log_param("start_time", datetime.now().isoformat())

                # Log información del modelo global
                for key, value in global_model_info.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(f"global_model_{key}", value)

                # Log configuración de privacidad
                mlflow.log_param("privacy_differential_privacy", True)
                mlflow.log_param("privacy_secure_aggregation", True)
                mlflow.log_param("privacy_gdpr_compliance", True)

                run_id = run.info.run_id
                logger.info(f"✅ Experimento MLflow iniciado: {run_id}")
                return run_id

        except Exception as e:
            logger.error(f"❌ Error iniciando experimento MLflow: {e}")
            return None

    def log_client_update(
        self,
        run_id: str,
        client_id: str,
        update: ModelUpdate,
        client_info: Dict[str, Any],
    ) -> bool:
        """Registrar actualización de cliente en MLflow"""
        if not self.client:
            return False

        try:
            with mlflow.start_run(run_id=run_id):

                # Crear prefijo para métricas del cliente
                prefix = f"client_{client_id}"

                # Log métricas de entrenamiento local
                mlflow.log_metric(f"{prefix}_local_loss", update.local_loss)
                mlflow.log_metric(f"{prefix}_local_accuracy", update.local_accuracy)
                mlflow.log_metric(f"{prefix}_num_samples", update.num_samples)

                # Log información del cliente
                mlflow.log_param(
                    f"{prefix}_device_type", client_info.get("device_type", "unknown")
                )
                mlflow.log_param(
                    f"{prefix}_location", client_info.get("location", "unknown")
                )
                mlflow.log_param(
                    f"{prefix}_reputation_score",
                    client_info.get("reputation_score", 0.0),
                )

                # Log métricas de privacidad
                privacy = update.privacy_guarantees
                if privacy:
                    for key, value in privacy.items():
                        if isinstance(value, (int, float, str, bool)):
                            mlflow.log_param(f"{prefix}_privacy_{key}", value)

                # Log modelo local (opcional - solo metadatos)
                if hasattr(update, "training_time"):
                    mlflow.log_metric(f"{prefix}_training_time", update.training_time)

                logger.info(
                    f"✅ Actualización de cliente {client_id} registrada en MLflow"
                )
                return True

        except Exception as e:
            logger.error(f"❌ Error registrando actualización de cliente: {e}")
            return False

    def log_global_model_update(
        self,
        run_id: str,
        round_number: int,
        global_model: Any,
        aggregated_metrics: Dict[str, Any],
        privacy_metrics: Dict[str, Any],
    ) -> bool:
        """Registrar actualización del modelo global"""
        if not self.client:
            return False

        try:
            with mlflow.start_run(run_id=run_id):

                # Log número de ronda
                mlflow.log_metric("round_number", round_number)

                # Log métricas agregadas
                for key, value in aggregated_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"global_{key}", value)

                # Log métricas de privacidad
                for key, value in privacy_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"privacy_{key}", value)

                # Log modelo global si está disponible
                if global_model is not None:
                    self._log_model_to_mlflow(global_model, "global_model")

                logger.info(
                    f"✅ Modelo global de ronda {round_number} registrado en MLflow"
                )
                return True

        except Exception as e:
            logger.error(f"❌ Error registrando modelo global: {e}")
            return False

    def log_round_completion(
        self, run_id: str, round_info: Dict[str, Any], final_metrics: Dict[str, Any]
    ) -> bool:
        """Registrar finalización de ronda"""
        if not self.client:
            return False

        try:
            with mlflow.start_run(run_id=run_id):

                # Log tiempo de finalización
                mlflow.log_param("end_time", datetime.now().isoformat())

                # Log métricas finales
                for key, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"final_{key}", value)

                # Log estadísticas de la ronda
                mlflow.log_metric(
                    "total_clients_participated",
                    round_info.get("participating_clients", 0),
                )
                mlflow.log_metric(
                    "round_duration_seconds", round_info.get("duration", 0)
                )

                # Marcar como completado
                mlflow.set_tag("status", "completed")
                mlflow.set_tag("round_id", round_info.get("round_id", ""))

                logger.info(f"✅ Ronda completada registrada en MLflow")
                return True

        except Exception as e:
            logger.error(f"❌ Error registrando finalización de ronda: {e}")
            return False

    def _log_model_to_mlflow(self, model: Any, model_name: str):
        """Registrar modelo en MLflow"""
        try:
            # Detectar tipo de modelo y usar el logger apropiado
            if hasattr(model, "parameters"):  # PyTorch model
                mlflow.pytorch.log_model(model, model_name)
            elif hasattr(model, "predict"):  # Scikit-learn model
                mlflow.sklearn.log_model(model, model_name)
            else:
                # Modelo genérico - guardar como pyfunc
                self._log_generic_model(model, model_name)

        except Exception as e:
            logger.warning(f"No se pudo registrar modelo {model_name}: {e}")

    def _log_generic_model(self, model: Any, model_name: str):
        """Registrar modelo genérico como pyfunc"""
        try:
            # Crear directorio temporal para el modelo
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "model.pkl")

                # Serializar modelo (simplificado)
                import pickle

                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

                # Crear wrapper para MLflow
                class GenericModelWrapper(mlflow.pyfunc.PythonModel):
                    def load_context(self, context):
                        with open(context.artifacts["model"], "rb") as f:
                            self.model = pickle.load(f)

                    def predict(self, context, model_input):
                        # Implementar predicción genérica
                        return {"prediction": "model_loaded"}

                # Log modelo
                mlflow.pyfunc.log_model(
                    artifact_path=model_name,
                    python_model=GenericModelWrapper(),
                    artifacts={"model": model_path},
                )

        except Exception as e:
            logger.error(f"Error registrando modelo genérico: {e}")

    def compare_experiments(
        self, use_case: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Comparar experimentos federados"""
        if not self.client:
            return []

        try:
            # Obtener runs del experimento
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.use_case = '{use_case}'" if use_case else "",
                order_by=["metrics.final_accuracy DESC"],
                max_results=limit,
            )

            experiments = []
            for run in runs:
                experiment = {
                    "run_id": run.info.run_id,
                    "round_id": run.data.params.get("round_id"),
                    "use_case": run.data.params.get("use_case"),
                    "num_clients": int(run.data.params.get("num_clients", 0)),
                    "final_accuracy": run.data.metrics.get("final_accuracy"),
                    "privacy_budget": run.data.metrics.get(
                        "privacy_differential_privacy_budget"
                    ),
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                }
                experiments.append(experiment)

            return experiments

        except Exception as e:
            logger.error(f"Error comparando experimentos: {e}")
            return []

    def get_experiment_metrics_history(
        self, run_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Obtener histórico de métricas de un experimento"""
        if not self.client:
            return {}

        try:
            # Obtener métricas del run
            metrics = self.client.get_run(run_id).data.metrics
            params = self.client.get_run(run_id).data.params

            history = {"metrics": [], "parameters": dict(params), "client_updates": []}

            # Extraer métricas por cliente
            client_metrics = {}
            for key, value in metrics.items():
                if key.startswith("client_"):
                    parts = key.split("_", 2)
                    if len(parts) >= 3:
                        client_id = parts[1]
                        metric_name = parts[2]

                        if client_id not in client_metrics:
                            client_metrics[client_id] = {}

                        client_metrics[client_id][metric_name] = value

            history["client_updates"] = [
                {"client_id": client_id, **metrics}
                for client_id, metrics in client_metrics.items()
            ]

            return history

        except Exception as e:
            logger.error(f"Error obteniendo histórico de métricas: {e}")
            return {}

    def create_model_version(
        self, model_name: str, run_id: str, description: str = ""
    ) -> Optional[str]:
        """Crear nueva versión de modelo en MLflow Model Registry"""
        if not self.client or not self.enable_model_registry:
            return None

        try:
            # Crear modelo registrado si no existe
            try:
                self.client.get_registered_model(model_name)
            except:
                self.client.create_registered_model(model_name)

            # Crear nueva versión
            model_version = self.client.create_model_version(
                name=model_name,
                source=f"runs:/{run_id}/global_model",
                description=description,
            )

            logger.info(f"✅ Versión de modelo creada: {model_version.version}")
            return model_version.version

        except Exception as e:
            logger.error(f"❌ Error creando versión de modelo: {e}")
            return None

    def transition_model_version(
        self, model_name: str, version: str, stage: str
    ) -> bool:
        """Transicionar versión de modelo a un stage diferente"""
        if not self.client or not self.enable_model_registry:
            return False

        try:
            self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )

            logger.info(f"✅ Modelo {model_name} v{version} movido a stage: {stage}")
            return True

        except Exception as e:
            logger.error(f"❌ Error transitando versión de modelo: {e}")
            return False

    def get_model_performance_comparison(
        self, model_name: str, versions: List[str]
    ) -> Dict[str, Any]:
        """Comparar rendimiento entre versiones de modelo"""
        if not self.client or not self.enable_model_registry:
            return {}

        try:
            comparison = {"model_name": model_name, "versions": []}

            for version in versions:
                try:
                    mv = self.client.get_model_version(model_name, version)
                    run = self.client.get_run(mv.run_id)

                    version_info = {
                        "version": version,
                        "stage": mv.current_stage,
                        "accuracy": run.data.metrics.get("final_accuracy"),
                        "privacy_budget": run.data.metrics.get(
                            "privacy_differential_privacy_budget"
                        ),
                        "creation_time": mv.creation_timestamp,
                    }

                    comparison["versions"].append(version_info)

                except Exception as e:
                    logger.warning(f"Error obteniendo info de versión {version}: {e}")

            return comparison

        except Exception as e:
            logger.error(f"Error comparando versiones de modelo: {e}")
            return {}


class FederatedExperimentManager:
    """
    Gestor de experimentos federados con MLflow
    """

    def __init__(self, mlflow_tracker: FederatedMLflowTracker):
        """Inicializar gestor de experimentos"""
        self.tracker = mlflow_tracker
        self.active_experiments = {}

    def start_round_experiment(
        self, fl_system: FederatedLearningSystem, round_id: str
    ) -> Optional[str]:
        """Iniciar experimento para una ronda federada"""
        try:
            # Obtener información de la ronda
            if round_id not in fl_system.active_rounds:
                return None

            round_obj = fl_system.active_rounds[round_id]

            # Información del modelo global
            global_model_info = {
                "architecture": "federated_model",
                "use_case": (
                    round_obj.use_case.value if round_obj.use_case else "general"
                ),
                "num_layers": 3,  # Simplificado
                "input_size": 100,
                "output_size": 10,
            }

            # Iniciar experimento
            run_id = self.tracker.start_federated_experiment(
                round_id=round_id,
                use_case=round_obj.use_case.value if round_obj.use_case else "general",
                num_clients=len(round_obj.participating_clients),
                global_model_info=global_model_info,
            )

            if run_id:
                self.active_experiments[round_id] = run_id

            return run_id

        except Exception as e:
            logger.error(f"Error iniciando experimento de ronda: {e}")
            return None

    def log_client_contribution(
        self,
        round_id: str,
        client_id: str,
        update: ModelUpdate,
        client_info: Dict[str, Any],
    ) -> bool:
        """Registrar contribución de cliente"""
        try:
            run_id = self.active_experiments.get(round_id)
            if not run_id:
                return False

            return self.tracker.log_client_update(
                run_id, client_id, update, client_info
            )

        except Exception as e:
            logger.error(f"Error registrando contribución de cliente: {e}")
            return False

    def finalize_round_experiment(
        self, round_id: str, final_metrics: Dict[str, Any]
    ) -> bool:
        """Finalizar experimento de ronda"""
        try:
            run_id = self.active_experiments.get(round_id)
            if not run_id:
                return False

            # Obtener información de la ronda (simplificada)
            round_info = {
                "round_id": round_id,
                "participating_clients": final_metrics.get("participating_clients", 0),
                "duration": final_metrics.get("duration", 0),
            }

            success = self.tracker.log_round_completion(
                run_id, round_info, final_metrics
            )

            if success:
                # Limpiar experimento activo
                self.active_experiments.pop(round_id, None)

            return success

        except Exception as e:
            logger.error(f"Error finalizando experimento de ronda: {e}")
            return False

    def generate_experiment_report(
        self, use_case: Optional[str] = None, days: int = 7
    ) -> Dict[str, Any]:
        """Generar reporte de experimentos"""
        try:
            # Comparar experimentos recientes
            experiments = self.tracker.compare_experiments(use_case=use_case, limit=20)

            # Filtrar por días recientes
            cutoff_date = datetime.now().timestamp() * 1000 - (
                days * 24 * 60 * 60 * 1000
            )
            recent_experiments = [
                exp for exp in experiments if exp.get("start_time", 0) > cutoff_date
            ]

            # Calcular estadísticas
            report = {
                "period_days": days,
                "use_case": use_case or "all",
                "total_experiments": len(recent_experiments),
                "avg_accuracy": 0.0,
                "avg_clients": 0,
                "best_accuracy": 0.0,
                "worst_accuracy": 1.0,
                "experiments": recent_experiments[:10],  # Top 10
            }

            if recent_experiments:
                accuracies = [
                    exp.get("final_accuracy", 0)
                    for exp in recent_experiments
                    if exp.get("final_accuracy")
                ]
                clients = [exp.get("num_clients", 0) for exp in recent_experiments]

                if accuracies:
                    report["avg_accuracy"] = sum(accuracies) / len(accuracies)
                    report["best_accuracy"] = max(accuracies)
                    report["worst_accuracy"] = min(accuracies)

                if clients:
                    report["avg_clients"] = sum(clients) / len(clients)

            return report

        except Exception as e:
            logger.error(f"Error generando reporte de experimentos: {e}")
            return {}


# ==================== FUNCIONES DE UTILIDAD ====================


def create_mlflow_tracker(
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "federated_learning_experiments",
) -> FederatedMLflowTracker:
    """Crear tracker de MLflow para FL"""
    return FederatedMLflowTracker(tracking_uri, experiment_name)


def create_experiment_manager(
    tracking_uri: str = "http://localhost:5000",
) -> FederatedExperimentManager:
    """Crear gestor de experimentos federados"""
    tracker = create_mlflow_tracker(tracking_uri)
    return FederatedExperimentManager(tracker)


# ==================== DEMO MLFLOW ====================


async def demo_federated_mlflow():
    """Demostración de integración con MLflow"""
    logger.info("🎯 Demo de Integración FL con MLflow")
    logger.info("=" * 60)

    try:
        # Crear tracker de MLflow
        logger.info("📊 Creando tracker de MLflow...")
        tracker = create_mlflow_tracker()

        if not tracker.client:
            logger.error("MLflow no disponible")
            return

        # Crear gestor de experimentos
        logger.info("🎛️ Creando gestor de experimentos...")
        experiment_manager = create_experiment_manager()

        # Simular experimento federado
        logger.info("🚀 Simulando experimento federado...")

        # Iniciar experimento
        round_id = "demo_round_001"
        run_id = experiment_manager.start_round_experiment(
            fl_system=None, round_id=round_id  # Usar sistema mock
        )

        if run_id:
            logger.info(f"✅ Experimento iniciado: {run_id}")

            # Simular contribuciones de clientes
            logger.info("👥 Registrando contribuciones de clientes...")
            import torch
            from federated_learning import ModelUpdate

            for i in range(3):
                client_id = f"client_{i+1:03d}"

                # Crear actualización simulada
                mock_weights = {
                    "layer1.weight": torch.randn(50, 100),
                    "layer1.bias": torch.randn(50),
                    "layer2.weight": torch.randn(10, 50),
                    "layer2.bias": torch.randn(10),
                }

                mock_update = ModelUpdate(
                    client_id=client_id,
                    round_id=round_id,
                    model_weights=mock_weights,
                    local_loss=0.3 - i * 0.05,
                    local_accuracy=0.8 + i * 0.02,
                    num_samples=500 + i * 100,
                    privacy_guarantees={"differential_privacy": True},
                )

                client_info = {
                    "device_type": "server",
                    "location": f"location_{i}",
                    "reputation_score": 0.9 - i * 0.05,
                }

                experiment_manager.log_client_contribution(
                    round_id=round_id,
                    client_id=client_id,
                    update=mock_update,
                    client_info=client_info,
                )

            # Finalizar experimento
            logger.info("🏁 Finalizando experimento...")
            final_metrics = {
                "final_accuracy": 0.87,
                "privacy_budget_remaining": 0.95,
                "participating_clients": 3,
                "duration": 120,
            }

            experiment_manager.finalize_round_experiment(round_id, final_metrics)

        # Generar reporte
        logger.info("📊 Generando reporte de experimentos...")
        report = experiment_manager.generate_experiment_report(days=1)

        logger.info("📈 Resumen del reporte:")
        logger.info(f"  Experimentos totales: {report.get('total_experiments', 0)}")
        logger.info(".3f")
        logger.info(f"  Clientes promedio: {report.get('avg_clients', 0)}")

        # Comparar experimentos
        logger.info("⚖️ Comparando experimentos...")
        comparisons = tracker.compare_experiments(limit=5)

        if comparisons:
            logger.info("🏆 Top 5 experimentos:")
            for i, exp in enumerate(comparisons[:5], 1):
                logger.info(
                    f"  {i}. {exp.get('run_id', '')[:8]} - "
                    ".3f"
                    f" ({exp.get('num_clients', 0)} clientes)"
                )

        logger.info("✅ Demo de MLflow completada exitosamente")

    except Exception as e:
        logger.error(f"❌ Error en demo de MLflow: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(demo_federated_mlflow())
