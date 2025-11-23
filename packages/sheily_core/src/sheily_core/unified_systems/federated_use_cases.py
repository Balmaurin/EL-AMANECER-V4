"""
Casos de Uso Avanzados para Aprendizaje Federado

Este módulo implementa casos de uso específicos para diferentes dominios:
- Servicios Financieros: Detección de fraude con máxima privacidad
- Dispositivos IoT: Procesamiento distribuido de sensores
- Integración con casos existentes (salud, voz, transporte)

Cada caso de uso incluye:
- Modelos especializados para el dominio
- Configuraciones de privacidad optimizadas
- Métricas específicas del dominio
- Validación de cumplimiento normativo

Autor: Sheily AI Team
Fecha: 2025
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from federated_client import ClientConfig, FederatedClient

# Importaciones del sistema FL
from federated_learning import (
    FederatedConfig,
    FederatedLearningSystem,
    ModelUpdate,
    UseCase,
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialUseCase(Enum):
    """Casos de uso específicos para servicios financieros"""

    FRAUD_DETECTION = "fraud_detection"
    CREDIT_SCORING = "credit_scoring"
    RISK_ASSESSMENT = "risk_assessment"
    ANOMALY_DETECTION = "anomaly_detection"


class IoTUseCase(Enum):
    """Casos de uso específicos para IoT"""

    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    ENERGY_OPTIMIZATION = "energy_optimization"
    ENVIRONMENTAL_MONITORING = "environmental_monitoring"
    SMART_CITY_ANALYTICS = "smart_city_analytics"


@dataclass
class FinancialConfig:
    """Configuración específica para casos de uso financiero"""

    use_case: FinancialUseCase
    gdpr_compliance_level: str = "maximum"  # maximum, high, standard
    data_sensitivity: str = "high"  # critical, high, medium, low
    regulatory_requirements: List[str] = field(
        default_factory=lambda: ["GDPR", "PSD2", "AML"]
    )
    privacy_budget: float = 0.1  # Muy restrictivo para finanzas
    max_clients_per_round: int = 10  # Limitado por regulación
    audit_trail: bool = True


@dataclass
class IoTConfig:
    """Configuración específica para casos de uso IoT"""

    use_case: IoTUseCase
    device_constraints: Dict[str, Any] = field(default_factory=dict)
    network_bandwidth: str = "low"  # high, medium, low
    power_constraints: str = "strict"  # none, moderate, strict
    update_frequency: str = "daily"  # realtime, hourly, daily, weekly
    edge_computing: bool = True
    federated_averaging: bool = True


class FinancialFederatedSystem(FederatedLearningSystem):
    """
    Sistema de aprendizaje federado especializado para servicios financieros
    """

    def __init__(self, config: FederatedConfig, financial_config: FinancialConfig):
        """Inicializar sistema financiero federado"""
        super().__init__(config)
        self.financial_config = financial_config
        self.audit_log = []
        self.compliance_monitor = ComplianceMonitor()

        # Configurar privacidad máxima para finanzas
        self._configure_financial_privacy()

        logger.info(
            f"💰 Sistema Financiero FL inicializado - Caso: {financial_config.use_case.value}"
        )

    def _configure_financial_privacy(self):
        """Configurar medidas de privacidad para finanzas"""
        # Privacidad diferencial muy estricta
        self.privacy_budget = self.financial_config.privacy_budget

        # Agregación segura obligatoria
        self.secure_aggregation = True

        # Logging de auditoría
        self.audit_enabled = self.financial_config.audit_trail

    async def validate_financial_compliance(
        self, client_id: str, data_info: Dict[str, Any]
    ) -> bool:
        """Validar cumplimiento normativo financiero"""
        try:
            # Verificar consentimiento explícito
            if not data_info.get("explicit_consent", False):
                logger.warning(f"❌ Cliente {client_id}: Consentimiento no explícito")
                return False

            # Verificar anonimización de datos
            if not data_info.get("data_anonymized", False):
                logger.warning(f"❌ Cliente {client_id}: Datos no anonimizados")
                return False

            # Verificar límites regulatorios
            if len(self.clients) >= self.financial_config.max_clients_per_round:
                logger.warning(f"❌ Cliente {client_id}: Límite de clientes excedido")
                return False

            # Registrar en auditoría
            if self.audit_enabled:
                self.audit_log.append(
                    {
                        "timestamp": datetime.now(),
                        "client_id": client_id,
                        "action": "compliance_check",
                        "result": "passed",
                        "data_info": data_info,
                    }
                )

            return True

        except Exception as e:
            logger.error(f"Error en validación financiera: {e}")
            return False

    def create_fraud_detection_model(self) -> Any:
        """Crear modelo de detección de fraude"""
        try:
            import torch.nn as nn

            class FraudDetectionModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(50, 100),  # Características financieras
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(100, 50),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(50, 2),  # Fraud/No fraud
                    )

                def forward(self, x):
                    return self.layers(x)

            return FraudDetectionModel()

        except Exception as e:
            logger.error(f"Error creando modelo de fraude: {e}")
            return None

    def create_credit_scoring_model(self) -> Any:
        """Crear modelo de credit scoring"""
        try:
            import torch.nn as nn

            class CreditScoringModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(30, 64),  # Variables crediticias
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),  # Credit score
                        nn.Sigmoid(),
                    )

                def forward(self, x):
                    return self.layers(x)

            return CreditScoringModel()

        except Exception as e:
            logger.error(f"Error creando modelo de crédito: {e}")
            return None

    async def process_financial_update(self, update: ModelUpdate) -> bool:
        """Procesar actualización financiera con validaciones extra"""
        try:
            # Validaciones financieras adicionales
            client = self.clients.get(update.client_id)
            if not client:
                return False

            # Verificar que el cliente tenga permisos financieros
            if (
                not hasattr(client, "financial_clearance")
                or not client.financial_clearance
            ):
                logger.warning(f"Cliente {update.client_id} sin permisos financieros")
                return False

            # Procesar actualización normalmente
            return await self.receive_client_update(update)

        except Exception as e:
            logger.error(f"Error procesando actualización financiera: {e}")
            return False

    def get_financial_metrics(self) -> Dict[str, Any]:
        """Obtener métricas específicas de finanzas"""
        base_metrics = self.get_federated_metrics()

        financial_metrics = {
            "financial_use_case": self.financial_config.use_case.value,
            "regulatory_compliance": self._calculate_compliance_score(),
            "audit_events": len(self.audit_log),
            "privacy_budget_used": 1.0 - self.privacy_budget,
            "data_sensitivity_level": self.financial_config.data_sensitivity,
            "active_regulatory_requirements": self.financial_config.regulatory_requirements,
        }

        return {**base_metrics, **financial_metrics}

    def _calculate_compliance_score(self) -> float:
        """Calcular puntaje de cumplimiento normativo"""
        try:
            # Factores de cumplimiento
            factors = {
                "gdpr_compliance": 0.9,
                "audit_trail": 1.0 if self.audit_enabled else 0.0,
                "secure_aggregation": 1.0 if self.secure_aggregation else 0.0,
                "privacy_budget": self.privacy_budget,
            }

            # Ponderación
            weights = {
                "gdpr_compliance": 0.4,
                "audit_trail": 0.2,
                "secure_aggregation": 0.2,
                "privacy_budget": 0.2,
            }

            compliance_score = sum(
                factors[key] * weights[key] for key in factors.keys()
            )

            return min(compliance_score, 1.0)

        except Exception:
            return 0.0


class IoTFederatedSystem(FederatedLearningSystem):
    """
    Sistema de aprendizaje federado especializado para dispositivos IoT
    """

    def __init__(self, config: FederatedConfig, iot_config: IoTConfig):
        """Inicializar sistema IoT federado"""
        super().__init__(config)
        self.iot_config = iot_config
        self.device_registry = {}
        self.network_optimizer = NetworkOptimizer()

        # Configurar para restricciones de IoT
        self._configure_iot_optimizations()

        logger.info(
            f"🔗 Sistema IoT FL inicializado - Caso: {iot_config.use_case.value}"
        )

    def _configure_iot_optimizations(self):
        """Configurar optimizaciones para IoT"""
        # Ajustar rondas según frecuencia
        frequency_settings = {
            "realtime": {"rounds_per_hour": 60, "local_epochs": 1},
            "hourly": {"rounds_per_hour": 1, "local_epochs": 3},
            "daily": {"rounds_per_hour": 1 / 24, "local_epochs": 5},
            "weekly": {"rounds_per_hour": 1 / (24 * 7), "local_epochs": 10},
        }

        settings = frequency_settings.get(
            self.iot_config.update_frequency, frequency_settings["daily"]
        )
        self.rounds_per_hour = settings["rounds_per_hour"]
        self.local_epochs = settings["local_epochs"]

        # Optimizaciones de red
        if self.iot_config.network_bandwidth == "low":
            self.compression_enabled = True
            self.quantization_bits = 8

    async def register_iot_device(
        self, client_id: str, device_info: Dict[str, Any], public_key: str
    ) -> bool:
        """Registrar dispositivo IoT con validaciones específicas"""
        try:
            # Validar capacidades del dispositivo
            if not self._validate_device_capabilities(device_info):
                logger.warning(f"Dispositivo {client_id} no cumple capacidades mínimas")
                return False

            # Registrar normalmente
            success = await self.register_client(
                client_id=client_id,
                public_key=public_key,
                use_case=UseCase.HEALTHCARE,  # Placeholder
                device_type=device_info.get("type", "iot"),
                location=device_info.get("location", "unknown"),
            )

            if success:
                # Registrar información específica de IoT
                self.device_registry[client_id] = {
                    "device_info": device_info,
                    "registered_at": datetime.now(),
                    "network_status": "active",
                    "power_level": device_info.get("power_level", 100),
                }

            return success

        except Exception as e:
            logger.error(f"Error registrando dispositivo IoT: {e}")
            return False

    def _validate_device_capabilities(self, device_info: Dict[str, Any]) -> bool:
        """Validar capacidades del dispositivo IoT"""
        try:
            # Verificar memoria mínima
            min_memory = self.iot_config.device_constraints.get("min_memory_mb", 32)
            device_memory = device_info.get("memory_mb", 0)

            if device_memory < min_memory:
                return False

            # Verificar conectividad
            if self.iot_config.network_bandwidth == "low":
                # Dispositivos con baja conectividad deben tener compresión
                if not device_info.get("supports_compression", False):
                    return False

            # Verificar restricciones de energía
            if self.iot_config.power_constraints == "strict":
                power_usage = device_info.get("power_usage_mw", 1000)
                if power_usage > 500:  # Máximo 500mW para dispositivos estrictos
                    return False

            return True

        except Exception:
            return False

    def create_predictive_maintenance_model(self) -> Any:
        """Crear modelo de mantenimiento predictivo para IoT"""
        try:
            import torch.nn as nn

            class PredictiveMaintenanceModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Modelo ligero para IoT
                    self.layers = nn.Sequential(
                        nn.Linear(20, 32),  # Sensores de máquina
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 3),  # Normal, Warning, Critical
                    )

                def forward(self, x):
                    return self.layers(x)

            return PredictiveMaintenanceModel()

        except Exception as e:
            logger.error(f"Error creando modelo IoT: {e}")
            return None

    def create_energy_optimization_model(self) -> Any:
        """Crear modelo de optimización energética"""
        try:
            import torch.nn as nn

            class EnergyOptimizationModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(15, 24),  # Variables energéticas
                        nn.ReLU(),
                        nn.Linear(24, 12),
                        nn.ReLU(),
                        nn.Linear(12, 1),  # Eficiencia energética
                        nn.Sigmoid(),
                    )

                def forward(self, x):
                    return self.layers(x)

            return EnergyOptimizationModel()

        except Exception as e:
            logger.error(f"Error creando modelo energético: {e}")
            return None

    async def optimize_network_transmission(
        self, client_id: str, model_update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimizar transmisión de red para IoT"""
        try:
            device_info = self.device_registry.get(client_id, {})

            # Aplicar compresión si es necesario
            if self.iot_config.network_bandwidth == "low":
                model_update = self.network_optimizer.compress_update(model_update)

            # Cuantización para dispositivos limitados
            if device_info.get("supports_quantization", False):
                model_update = self._quantize_model_update(model_update)

            return model_update

        except Exception as e:
            logger.error(f"Error optimizando transmisión: {e}")
            return model_update

    def _quantize_model_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Cuantizar actualización del modelo para eficiencia"""
        try:
            import torch

            quantized = {}
            for key, tensor in update.items():
                if isinstance(tensor, torch.Tensor):
                    # Cuantizar a 8 bits
                    quantized_tensor = torch.quantize_per_tensor(
                        tensor.float(), 1.0, 0, torch.qint8
                    )
                    quantized[key] = quantized_tensor
                else:
                    quantized[key] = tensor

            return quantized

        except Exception:
            return update

    def get_iot_metrics(self) -> Dict[str, Any]:
        """Obtener métricas específicas de IoT"""
        base_metrics = self.get_federated_metrics()

        # Calcular métricas de dispositivos
        total_devices = len(self.device_registry)
        active_devices = sum(
            1
            for d in self.device_registry.values()
            if d.get("network_status") == "active"
        )

        avg_power_level = (
            np.mean([d.get("power_level", 100) for d in self.device_registry.values()])
            if self.device_registry
            else 100
        )

        iot_metrics = {
            "iot_use_case": self.iot_config.use_case.value,
            "total_devices": total_devices,
            "active_devices": active_devices,
            "avg_power_level": avg_power_level,
            "network_bandwidth": self.iot_config.network_bandwidth,
            "update_frequency": self.iot_config.update_frequency,
            "edge_computing_enabled": self.iot_config.edge_computing,
        }

        return {**base_metrics, **iot_metrics}


class ComplianceMonitor:
    """
    Monitor de cumplimiento normativo para sistemas financieros
    """

    def __init__(self):
        """Inicializar monitor de cumplimiento"""
        self.compliance_rules = {
            "GDPR": self._check_gdpr_compliance,
            "PSD2": self._check_psd2_compliance,
            "AML": self._check_aml_compliance,
        }
        self.violations = []

    def check_compliance(
        self, data: Dict[str, Any], regulations: List[str]
    ) -> Dict[str, Any]:
        """Verificar cumplimiento con regulaciones especificadas"""
        results = {}

        for regulation in regulations:
            check_func = self.compliance_rules.get(regulation)
            if check_func:
                compliant, details = check_func(data)
                results[regulation] = {"compliant": compliant, "details": details}

                if not compliant:
                    self.violations.append(
                        {
                            "timestamp": datetime.now(),
                            "regulation": regulation,
                            "details": details,
                            "data": data,
                        }
                    )

        return results

    def _check_gdpr_compliance(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Verificar cumplimiento GDPR"""
        # Verificar consentimiento
        if not data.get("consent_given", False):
            return False, "Consentimiento no proporcionado"

        # Verificar anonimización
        if not data.get("data_anonymized", False):
            return False, "Datos no anonimizados"

        # Verificar retención de datos
        retention_days = data.get("retention_days", 0)
        if retention_days > 2555:  # Máximo 7 años para datos financieros
            return False, "Período de retención excesivo"

        return True, "Cumple con GDPR"

    def _check_psd2_compliance(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Verificar cumplimiento PSD2"""
        # Verificar autenticación fuerte
        if not data.get("strong_authentication", False):
            return False, "Autenticación fuerte requerida"

        # Verificar límites de transacción
        transaction_amount = data.get("transaction_amount", 0)
        if transaction_amount > 50000:  # Ejemplo de límite
            return False, "Monto de transacción excede límites"

        return True, "Cumple con PSD2"

    def _check_aml_compliance(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Verificar cumplimiento AML (Anti-Money Laundering)"""
        # Verificar KYC (Know Your Customer)
        if not data.get("kyc_completed", False):
            return False, "KYC no completado"

        # Verificar screening de sanciones
        if data.get("sanctions_hit", False):
            return False, "Cliente en lista de sanciones"

        return True, "Cumple con AML"


class NetworkOptimizer:
    """
    Optimizador de red para transmisiones IoT eficientes
    """

    def __init__(self):
        """Inicializar optimizador de red"""
        self.compression_algorithms = {
            "gzip": self._gzip_compress,
            "delta": self._delta_compress,
            "sparse": self._sparse_compress,
        }

    def compress_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Comprimir actualización del modelo"""
        try:
            # Elegir algoritmo basado en el tipo de datos
            if self._is_sparse_update(update):
                return self.compression_algorithms["sparse"](update)
            else:
                return self.compression_algorithms["gzip"](update)

        except Exception:
            return update

    def _gzip_compress(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Compresión GZIP básica"""
        try:
            import gzip
            import pickle

            # Serializar y comprimir
            serialized = pickle.dumps(update)
            compressed = gzip.compress(serialized)

            return {
                "compressed_data": compressed,
                "compression_type": "gzip",
                "original_size": len(serialized),
                "compressed_size": len(compressed),
            }

        except Exception:
            return update

    def _delta_compress(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Compresión delta (diferencias)"""
        # Implementación simplificada
        return update

    def _sparse_compress(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Compresión para actualizaciones dispersas"""
        # Implementación simplificada
        return update

    def _is_sparse_update(self, update: Dict[str, Any]) -> bool:
        """Determinar si la actualización es dispersa"""
        # Lógica simplificada
        return len(update) < 10


# ==================== FUNCIONES DE UTILIDAD ====================


def create_financial_fl_system(
    use_case: FinancialUseCase, num_clients: int = 20
) -> FinancialFederatedSystem:
    """Crear sistema FL para servicios financieros"""

    # Configuración base
    config = FederatedConfig(
        num_clients=num_clients,
        num_rounds=100,  # Más rondas para estabilidad financiera
        local_epochs=5,
        secure_aggregation=True,
        gdpr_compliance=True,
        data_quality_checks=True,
    )

    # Configuración financiera específica
    financial_config = FinancialConfig(
        use_case=use_case,
        gdpr_compliance_level="maximum",
        data_sensitivity="high",
        privacy_budget=0.05,  # Muy restrictivo
        audit_trail=True,
    )

    return FinancialFederatedSystem(config, financial_config)


def create_iot_fl_system(
    use_case: IoTUseCase, num_clients: int = 50
) -> IoTFederatedSystem:
    """Crear sistema FL para dispositivos IoT"""

    # Configuración base optimizada para IoT
    config = FederatedConfig(
        num_clients=num_clients,
        num_rounds=200,  # Muchas rondas para dispositivos distribuidos
        local_epochs=3,  # Épocas reducidas para ahorro de energía
        secure_aggregation=True,
        gdpr_compliance=True,
    )

    # Configuración IoT específica
    iot_config = IoTConfig(
        use_case=use_case,
        device_constraints={"min_memory_mb": 64, "min_cpu_cores": 1},
        network_bandwidth="low",
        power_constraints="moderate",
        update_frequency="daily",
        edge_computing=True,
    )

    return IoTFederatedSystem(config, iot_config)


# ==================== DEMOS ====================


async def demo_financial_system():
    """Demostración del sistema financiero federado"""
    logger.info("💰 Demo del Sistema Financiero Federado")
    logger.info("=" * 60)

    try:
        # Crear sistema de detección de fraude
        financial_system = create_financial_fl_system(
            use_case=FinancialUseCase.FRAUD_DETECTION, num_clients=5
        )

        logger.info("✅ Sistema financiero inicializado")

        # Simular registro de bancos/clientes
        bank_names = ["Bank_A", "Bank_B", "Bank_C", "Bank_D", "Bank_E"]

        for bank in bank_names:
            # Simular validación financiera
            data_info = {
                "explicit_consent": True,
                "data_anonymized": True,
                "retention_days": 2555,  # 7 años
                "kyc_completed": True,
                "sanctions_hit": False,
            }

            compliance_ok = await financial_system.validate_financial_compliance(
                bank, data_info
            )
            logger.info(
                f"🏦 Banco {bank} - Cumplimiento: {'✅' if compliance_ok else '❌'}"
            )

        # Obtener métricas financieras
        metrics = financial_system.get_financial_metrics()
        logger.info("📊 Métricas financieras:")
        logger.info(f"  Clientes activos: {metrics.get('active_clients', 0)}")
        logger.info(".3f")
        logger.info(f"  Eventos de auditoría: {metrics.get('audit_events', 0)}")

        logger.info("✅ Demo del sistema financiero completada")

    except Exception as e:
        logger.error(f"❌ Error en demo financiera: {e}")


async def demo_iot_system():
    """Demostración del sistema IoT federado"""
    logger.info("🔗 Demo del Sistema IoT Federado")
    logger.info("=" * 60)

    try:
        # Crear sistema de mantenimiento predictivo
        iot_system = create_iot_fl_system(
            use_case=IoTUseCase.PREDICTIVE_MAINTENANCE, num_clients=10
        )

        logger.info("✅ Sistema IoT inicializado")

        # Simular registro de dispositivos IoT
        devices = [
            {
                "id": "sensor_001",
                "type": "vibration_sensor",
                "memory_mb": 128,
                "power_usage_mw": 200,
            },
            {
                "id": "sensor_002",
                "type": "temperature_sensor",
                "memory_mb": 64,
                "power_usage_mw": 150,
            },
            {
                "id": "sensor_003",
                "type": "pressure_sensor",
                "memory_mb": 256,
                "power_usage_mw": 300,
            },
        ]

        for device in devices:
            device_info = {
                "type": device["type"],
                "memory_mb": device["memory_mb"],
                "power_usage_mw": device["power_usage_mw"],
                "supports_compression": True,
                "supports_quantization": True,
                "location": "factory_floor",
            }

            success = await iot_system.register_iot_device(
                client_id=device["id"],
                device_info=device_info,
                public_key=f"key_{device['id']}",
            )

            logger.info(
                f"📡 Dispositivo {device['id']} - Registro: {'✅' if success else '❌'}"
            )

        # Obtener métricas IoT
        metrics = iot_system.get_iot_metrics()
        logger.info("📊 Métricas IoT:")
        logger.info(f"  Dispositivos totales: {metrics.get('total_devices', 0)}")
        logger.info(f"  Dispositivos activos: {metrics.get('active_devices', 0)}")
        logger.info(".1f")
        logger.info(f"  Ancho de banda: {metrics.get('network_bandwidth', 'unknown')}")

        logger.info("✅ Demo del sistema IoT completada")

    except Exception as e:
        logger.error(f"❌ Error en demo IoT: {e}")


async def demo_compliance_monitor():
    """Demostración del monitor de cumplimiento"""
    logger.info("⚖️ Demo del Monitor de Cumplimiento")
    logger.info("=" * 60)

    try:
        monitor = ComplianceMonitor()

        # Datos de prueba
        test_data = {
            "consent_given": True,
            "data_anonymized": True,
            "retention_days": 2000,
            "strong_authentication": True,
            "transaction_amount": 25000,
            "kyc_completed": True,
            "sanctions_hit": False,
        }

        # Verificar cumplimiento
        regulations = ["GDPR", "PSD2", "AML"]
        results = monitor.check_compliance(test_data, regulations)

        logger.info("📋 Resultados de cumplimiento:")
        for regulation, result in results.items():
            status = "✅" if result["compliant"] else "❌"
            logger.info(f"  {regulation}: {status} - {result['details']}")

        logger.info(f"🚨 Violaciones totales: {len(monitor.violations)}")

        logger.info("✅ Demo del monitor de cumplimiento completada")

    except Exception as e:
        logger.error(f"❌ Error en demo de cumplimiento: {e}")


if __name__ == "__main__":

    async def run_all_demos():
        await demo_financial_system()
        print("\n" + "=" * 60 + "\n")
        await demo_iot_system()
        print("\n" + "=" * 60 + "\n")
        await demo_compliance_monitor()

    asyncio.run(run_all_demos())
