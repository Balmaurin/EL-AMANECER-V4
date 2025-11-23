"""
Integración entre Sistema de Aprendizaje Federado y Sistema Unificado de Entrenamiento

Este módulo proporciona una integración fluida entre el sistema de aprendizaje federado
y el sistema existente de entrenamiento unificado, permitiendo una transición gradual
y compatibilidad hacia arquitecturas federadas.

Características:
- Puente entre sistemas centralizados y federados
- Migración gradual de cargas de trabajo
- Métricas combinadas y monitoreo unificado
- Compatibilidad con APIs existentes
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from federated_learning import FederatedConfig, FederatedLearningSystem, UseCase

# Importaciones de sistemas existentes
from unified_learning_training_system import (
    TrainingConfig,
    UnifiedLearningTrainingSystem,
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuración de integración"""

    enable_federated_mode: bool = False
    federated_percentage: float = 0.0  # 0.0 = 100% centralizado, 1.0 = 100% federado
    hybrid_mode: bool = False  # Permitir ambos modos simultáneamente
    migration_strategy: str = "gradual"  # gradual, immediate, phased

    # Configuración de federated learning
    fl_config: Optional[FederatedConfig] = None

    # Configuración de sistema existente
    existing_config: Optional[TrainingConfig] = None


class FederatedIntegrationSystem:
    """
    Sistema de integración que combina aprendizaje centralizado y federado
    """

    def __init__(self, integration_config: IntegrationConfig):
        """Inicializar sistema de integración"""
        self.config = integration_config

        # Instancias de sistemas
        self.unified_system: Optional[UnifiedLearningTrainingSystem] = None
        self.federated_system: Optional[FederatedLearningSystem] = None

        # Estado de integración
        self.migration_phase = "initialization"
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Inicializar sistemas según configuración
        self._initialize_systems()

        logger.info("🔗 Sistema de Integración FL inicializado")

    def _initialize_systems(self):
        """Inicializar sistemas centralizado y federado"""
        try:
            # Sistema centralizado existente
            if self.config.existing_config:
                self.unified_system = UnifiedLearningTrainingSystem(
                    config=self.config.existing_config
                )
                logger.info("✅ Sistema centralizado inicializado")

            # Sistema federado
            if self.config.enable_federated_mode and self.config.fl_config:
                self.federated_system = FederatedLearningSystem(
                    config=self.config.fl_config
                )
                logger.info("✅ Sistema federado inicializado")

        except Exception as e:
            logger.error(f"❌ Error inicializando sistemas: {e}")

    async def start_integrated_training(
        self,
        model_name: str,
        dataset_path: str,
        use_case: Optional[UseCase] = None,
        force_federated: bool = False,
    ) -> Dict[str, Any]:
        """Iniciar entrenamiento integrado (centralizado o federado)"""

        # Determinar si usar modo federado
        use_federated = force_federated or (
            self.config.enable_federated_mode
            and self._should_use_federated_mode(use_case)
        )

        if use_federated and self.federated_system:
            # Usar sistema federado
            return await self._start_federated_training(
                model_name, dataset_path, use_case
            )
        elif self.unified_system:
            # Usar sistema centralizado
            return await self._start_centralized_training(model_name, dataset_path)
        else:
            raise RuntimeError("Ningún sistema de entrenamiento disponible")

    async def _start_federated_training(
        self, model_name: str, dataset_path: str, use_case: Optional[UseCase]
    ) -> Dict[str, Any]:
        """Iniciar entrenamiento federado"""
        try:
            logger.info(f"🎯 Iniciando entrenamiento federado para {model_name}")

            # Crear modelo basado en caso de uso
            global_model = self._create_model_for_use_case(use_case)

            # Actualizar configuración del sistema federado
            if self.federated_system:
                self.federated_system.global_model = global_model

                # Iniciar ronda federada
                round_id = await self.federated_system.start_federated_round(1)

                session_info = {
                    "session_id": round_id,
                    "type": "federated",
                    "model_name": model_name,
                    "dataset_path": dataset_path,
                    "use_case": use_case.value if use_case else None,
                    "start_time": self.federated_system.active_rounds[
                        round_id
                    ].start_time,
                    "status": "running",
                }

                self.active_sessions[round_id] = session_info

                logger.info(f"✅ Sesión federada iniciada: {round_id}")
                return session_info

        except Exception as e:
            logger.error(f"❌ Error iniciando entrenamiento federado: {e}")
            raise

    async def _start_centralized_training(
        self, model_name: str, dataset_path: str
    ) -> Dict[str, Any]:
        """Iniciar entrenamiento centralizado"""
        try:
            logger.info(f"🎯 Iniciando entrenamiento centralizado para {model_name}")

            if not self.unified_system:
                raise RuntimeError("Sistema centralizado no disponible")

            # Iniciar sesión de entrenamiento centralizado
            session_id = await self.unified_system.start_training_session(
                model_name=model_name,
                dataset_path=dataset_path,
                training_mode=self.unified_system.TrainingMode.FINE_TUNE,
            )

            session_info = {
                "session_id": session_id,
                "type": "centralized",
                "model_name": model_name,
                "dataset_path": dataset_path,
                "use_case": None,
                "status": "running",
            }

            self.active_sessions[session_id] = session_info

            logger.info(f"✅ Sesión centralizada iniciada: {session_id}")
            return session_info

        except Exception as e:
            logger.error(f"❌ Error iniciando entrenamiento centralizado: {e}")
            raise

    def _should_use_federated_mode(self, use_case: Optional[UseCase]) -> bool:
        """Determinar si debe usar modo federado"""
        if not self.config.enable_federated_mode:
            return False

        # Lógica de decisión basada en caso de uso y configuración
        if use_case in [UseCase.HEALTHCARE, UseCase.FINANCIAL_SERVICES]:
            # Casos de uso sensibles siempre usan FL si está disponible
            return True

        # Usar porcentaje configurado para otros casos
        import random

        return random.random() < self.config.federated_percentage

    def _create_model_for_use_case(self, use_case: Optional[UseCase]):
        """Crear modelo apropiado para el caso de uso"""
        try:
            import torch.nn as nn

            if use_case == UseCase.HEALTHCARE:
                # Modelo médico simple
                return nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 2))
            elif use_case == UseCase.SPEECH_RECOGNITION:
                # Modelo de voz
                return nn.Sequential(
                    nn.LSTM(40, 128, batch_first=True), nn.Linear(128, 10)
                )
            elif use_case == UseCase.AUTONOMOUS_TRANSPORT:
                # Modelo de visión
                return nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(16 * 8 * 8, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10),
                )
            else:
                # Modelo genérico
                return nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))

        except Exception as e:
            logger.error(f"❌ Error creando modelo: {e}")
            return None

    async def get_integrated_metrics(self) -> Dict[str, Any]:
        """Obtener métricas combinadas de ambos sistemas"""
        try:
            metrics = {
                "integration_mode": self.config.migration_strategy,
                "federated_percentage": self.config.federated_percentage,
                "migration_phase": self.migration_phase,
                "active_sessions": len(self.active_sessions),
                "centralized_metrics": {},
                "federated_metrics": {},
            }

            # Métricas del sistema centralizado
            if self.unified_system:
                metrics["centralized_metrics"] = self.unified_system.get_system_stats()

            # Métricas del sistema federado
            if self.federated_system:
                metrics["federated_metrics"] = (
                    self.federated_system.get_federated_metrics()
                )

            # Métricas de integración
            metrics["integration_stats"] = self._calculate_integration_stats()

            return metrics

        except Exception as e:
            logger.error(f"❌ Error obteniendo métricas integradas: {e}")
            return {}

    def _calculate_integration_stats(self) -> Dict[str, Any]:
        """Calcular estadísticas de integración"""
        try:
            total_sessions = len(self.active_sessions)
            federated_sessions = sum(
                1 for s in self.active_sessions.values() if s.get("type") == "federated"
            )

            return {
                "total_active_sessions": total_sessions,
                "federated_sessions": federated_sessions,
                "centralized_sessions": total_sessions - federated_sessions,
                "federated_ratio": (
                    federated_sessions / total_sessions if total_sessions > 0 else 0.0
                ),
                "migration_progress": self._calculate_migration_progress(),
            }

        except Exception as e:
            return {}

    def _calculate_migration_progress(self) -> float:
        """Calcular progreso de migración a FL"""
        try:
            if self.config.migration_strategy == "immediate":
                return 1.0 if self.config.enable_federated_mode else 0.0
            elif self.config.migration_strategy == "gradual":
                return self.config.federated_percentage
            else:
                # phased migration - calcular basado en fases completadas
                return 0.5  # Implementación simplificada

        except Exception:
            return 0.0

    async def migrate_workload_to_federated(
        self, workload_id: str, target_use_case: UseCase, num_clients: int = 5
    ) -> bool:
        """Migrar una carga de trabajo al sistema federado"""
        try:
            logger.info(f"🔄 Migrando workload {workload_id} a sistema federado")

            if not self.federated_system:
                logger.error("Sistema federado no disponible")
                return False

            # Registrar clientes simulados para la migración
            for i in range(num_clients):
                client_id = f"migrated_client_{workload_id}_{i}"
                public_key = f"key_{client_id}"

                await self.federated_system.register_client(
                    client_id=client_id,
                    public_key=public_key,
                    use_case=target_use_case,
                    device_type="server",
                    location=f"migrated_{i}",
                )

            # Actualizar fase de migración
            self.migration_phase = f"migrating_{workload_id}"

            logger.info(f"✅ Workload {workload_id} migrado exitosamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error migrando workload: {e}")
            return False

    async def compare_system_performance(self) -> Dict[str, Any]:
        """Comparar rendimiento entre sistemas centralizado y federado"""
        try:
            comparison = {
                "centralized_performance": {},
                "federated_performance": {},
                "comparison_metrics": {},
            }

            # Rendimiento centralizado
            if self.unified_system:
                centralized_stats = self.unified_system.get_system_stats()
                comparison["centralized_performance"] = {
                    "active_sessions": centralized_stats.get(
                        "active_training_sessions", 0
                    ),
                    "total_experiences": centralized_stats.get(
                        "total_learning_experiences", 0
                    ),
                    "performance_score": centralized_stats.get(
                        "learning_stats", {}
                    ).get("average_performance", 0.0),
                }

            # Rendimiento federado
            if self.federated_system:
                federated_stats = self.federated_system.get_federated_metrics()
                comparison["federated_performance"] = {
                    "active_clients": federated_stats.get("active_clients", 0),
                    "total_clients": federated_stats.get("total_clients", 0),
                    "gdpr_compliance_rate": federated_stats.get(
                        "gdpr_compliance_rate", 0.0
                    ),
                    "security_events": len(federated_stats.get("security_events", [])),
                }

            # Métricas de comparación
            comparison["comparison_metrics"] = {
                "privacy_advantage": self._calculate_privacy_advantage(comparison),
                "scalability_advantage": self._calculate_scalability_advantage(
                    comparison
                ),
                "performance_tradeoff": self._calculate_performance_tradeoff(
                    comparison
                ),
            }

            return comparison

        except Exception as e:
            logger.error(f"❌ Error comparando rendimiento: {e}")
            return {}

    def _calculate_privacy_advantage(self, comparison: Dict[str, Any]) -> float:
        """Calcular ventaja de privacidad del sistema federado"""
        try:
            gdpr_rate = comparison["federated_performance"].get(
                "gdpr_compliance_rate", 0.0
            )
            security_events = comparison["federated_performance"].get(
                "security_events", 0
            )

            # Puntaje de privacidad (0-1, donde 1 es mejor privacidad)
            privacy_score = gdpr_rate * (1.0 - min(security_events / 10.0, 1.0))
            return privacy_score

        except Exception:
            return 0.0

    def _calculate_scalability_advantage(self, comparison: Dict[str, Any]) -> float:
        """Calcular ventaja de escalabilidad del sistema federado"""
        try:
            federated_clients = comparison["federated_performance"].get(
                "active_clients", 0
            )
            centralized_sessions = comparison["centralized_performance"].get(
                "active_sessions", 0
            )

            # FL escala mejor con más participantes
            if federated_clients > 0:
                return min(federated_clients / max(centralized_sessions + 1, 1), 2.0)
            return 0.0

        except Exception:
            return 0.0

    def _calculate_performance_tradeoff(self, comparison: Dict[str, Any]) -> float:
        """Calcular trade-off de rendimiento entre sistemas"""
        try:
            centralized_perf = comparison["centralized_performance"].get(
                "performance_score", 0.0
            )
            federated_privacy = comparison["federated_performance"].get(
                "gdpr_compliance_rate", 0.0
            )

            # Trade-off: rendimiento vs privacidad
            return centralized_perf * 0.6 + federated_privacy * 0.4

        except Exception:
            return 0.0

    async def close(self):
        """Cerrar sistema de integración"""
        try:
            # Cerrar sistemas individuales
            if self.unified_system:
                self.unified_system.close()

            if self.federated_system:
                self.federated_system.close()

            logger.info("🔒 Sistema de integración cerrado")

        except Exception as e:
            logger.error(f"❌ Error cerrando sistema de integración: {e}")


# ==================== FUNCIONES DE UTILIDAD ====================


def create_gradual_migration_config(
    initial_federated_percentage: float = 0.2,
    target_federated_percentage: float = 0.8,
    migration_steps: int = 5,
) -> IntegrationConfig:
    """Crear configuración para migración gradual"""

    config = IntegrationConfig(
        enable_federated_mode=True,
        federated_percentage=initial_federated_percentage,
        hybrid_mode=True,
        migration_strategy="gradual",
    )

    # Configuración FL para migración gradual
    config.fl_config = FederatedConfig(
        num_clients=10,
        num_rounds=20,  # Rondas más cortas para migración
        local_epochs=3,
        secure_aggregation=True,
        gdpr_compliance=True,
    )

    # Configuración sistema existente
    config.existing_config = TrainingConfig(num_train_epochs=5, batch_size=16)

    return config


def create_immediate_migration_config() -> IntegrationConfig:
    """Crear configuración para migración inmediata a FL"""

    config = IntegrationConfig(
        enable_federated_mode=True,
        federated_percentage=1.0,
        hybrid_mode=False,
        migration_strategy="immediate",
    )

    # Configuración FL completa
    config.fl_config = FederatedConfig(
        num_clients=20,
        num_rounds=50,
        privacy_techniques=["differential_privacy", "secure_multiparty_computation"],
        secure_aggregation=True,
        gdpr_compliance=True,
        data_quality_checks=True,
    )

    return config


# ==================== DEMO DE INTEGRACIÓN ====================


async def demo_integration_system():
    """Demostración del sistema de integración"""
    logger.info("🔗 Demo del Sistema de Integración FL")
    logger.info("=" * 60)

    try:
        # Crear configuración de migración gradual
        integration_config = create_gradual_migration_config(
            initial_federated_percentage=0.5
        )

        # Crear sistema de integración
        integration_system = FederatedIntegrationSystem(integration_config)
        logger.info("✅ Sistema de integración creado")

        # Iniciar entrenamientos mixtos
        logger.info("\n🚀 Iniciando entrenamientos mixtos...")

        # Entrenamiento centralizado
        centralized_session = await integration_system.start_integrated_training(
            model_name="centralized_model",
            dataset_path="data/centralized_dataset",
            force_federated=False,
        )
        logger.info(f"📊 Sesión centralizada: {centralized_session['session_id']}")

        # Entrenamiento federado
        federated_session = await integration_system.start_integrated_training(
            model_name="federated_model",
            dataset_path="data/federated_dataset",
            use_case=UseCase.HEALTHCARE,
            force_federated=True,
        )
        logger.info(f"🎯 Sesión federada: {federated_session['session_id']}")

        # Obtener métricas integradas
        logger.info("\n📈 Métricas del sistema integrado:")
        metrics = await integration_system.get_integrated_metrics()

        logger.info(f"  Modo de integración: {metrics['integration_mode']}")
        logger.info(f"  Porcentaje federado: {metrics['federated_percentage']:.1%}")
        logger.info(f"  Sesiones activas: {metrics['active_sessions']}")

        if metrics["centralized_metrics"]:
            logger.info(
                f"  Sesiones centralizadas activas: {metrics['centralized_metrics'].get('active_training_sessions', 0)}"
            )

        if metrics["federated_metrics"]:
            logger.info(
                f"  Clientes FL activos: {metrics['federated_metrics'].get('active_clients', 0)}"
            )

        # Comparar rendimiento
        logger.info("\n⚖️ Comparación de rendimiento:")
        comparison = await integration_system.compare_system_performance()

        privacy_advantage = comparison.get("comparison_metrics", {}).get(
            "privacy_advantage", 0.0
        )
        scalability_advantage = comparison.get("comparison_metrics", {}).get(
            "scalability_advantage", 0.0
        )

        logger.info(".3f")
        logger.info(".3f")
        # Cerrar sistema
        await integration_system.close()

        logger.info("\n✅ Demo de integración completada exitosamente")

    except Exception as e:
        logger.error(f"❌ Error en demo de integración: {e}")


if __name__ == "__main__":
    asyncio.run(demo_integration_system())
