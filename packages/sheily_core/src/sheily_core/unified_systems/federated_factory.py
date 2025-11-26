"""
🎯 FEDERATED LEARNING FACTORY - SHEILY AI UNIFIED SYSTEM
============================================================
Factory Pattern para crear especializaciones del sistema FL

Este módulo implementa el patrón Factory para crear diferentes tipos
de sistemas FL basados en el núcleo unificado.

CARACTERÍSTICAS:
- Factory methods para diferentes casos de uso
- Builder pattern para configuraciones complejas
- Registry de sistemas disponibles
- Lazy loading de módulos especializados

AUTORES: Sheily AI Team - Arquitectura Unificada v2.0
FECHA: 2025
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, Any, Optional, Type, Union
from abc import ABC, abstractmethod

# Import unified core
from federated_core import (
    BaseFederatedSystem,
    BaseFederatedConfig,
    UseCase,
    FederatedLearningMode,
    DependencyManager,
    common_logger,
    create_healthcare_config,
    create_finance_config,
    create_iot_config,
)

# ========================================================================
# SYSTEM REGISTRY - TRACK AVAILABLE SPECIALIZATIONS
# ========================================================================
class FederatedSystemRegistry:
    """Registry de sistemas FL disponibles"""

    _systems: Dict[str, Type[BaseFederatedSystem]] = {}

    @classmethod
    def register(cls, system_type: str, system_class: Type[BaseFederatedSystem]):
        """Registrar un tipo de sistema"""
        cls._systems[system_type] = system_class
        common_logger.info(f"📝 Sistema '{system_type}' registrado: {system_class.__name__}")

    @classmethod
    def get_available_systems(cls) -> Dict[str, Type[BaseFederatedSystem]]:
        """Obtener sistemas disponibles"""
        return cls._systems.copy()

    @classmethod
    def is_available(cls, system_type: str) -> bool:
        """Verificar si un sistema está disponible"""
        return system_type in cls._systems

# ========================================================================
# CONFIG BUILDERS - CONSTRUCT COMPLEX CONFIGS
# ========================================================================
class FederatedConfigBuilder:
    """Builder pattern para configuraciones FL complejas"""

    def __init__(self, base_config: Optional[BaseFederatedConfig] = None):
        self.config = base_config or BaseFederatedConfig()

    def with_use_case(self, use_case: UseCase) -> 'FederatedConfigBuilder':
        """Configurar caso de uso"""
        self.config.use_case = use_case
        return self

    def with_privacy_level(self, level: str) -> 'FederatedConfigBuilder':
        """Configurar nivel de privacidad"""
        from federated_core import PrivacyTechnique, SecurityProtocol

        if level == "maximum":
            self.config.privacy_techniques = [
                PrivacyTechnique.HOMOMORPHIC_ENCRYPTION,
                PrivacyTechnique.DIFFERENTIAL_PRIVACY,
                PrivacyTechnique.SECURE_MULTIPARTY_COMPUTATION
            ]
            self.config.security_protocols = [
                SecurityProtocol.HOMOMORPHIC_ENCRYPTION,
                SecurityProtocol.MPC_ENCRYPTION,
            ]
        elif level == "high":
            self.config.privacy_techniques = [
                PrivacyTechnique.DIFFERENTIAL_PRIVACY,
                PrivacyTechnique.SECURE_MULTIPARTY_COMPUTATION
            ]
        elif level == "standard":
            self.config.privacy_techniques = [PrivacyTechnique.DIFFERENTIAL_PRIVACY]

        return self

    def with_performance_optimization(self, device_type: str) -> 'FederatedConfigBuilder':
        """Optimizar para tipo de dispositivo"""
        if device_type == "mobile":
            self.config.local_epochs = 2
            self.config.batch_size = 16
            self.config.communication_timeout = 60
        elif device_type == "iot":
            self.config.local_epochs = 1
            self.config.batch_size = 8
            self.config.client_selection_strategy = "data_quality"

        return self

    def with_scale(self, client_count: int) -> 'FederatedConfigBuilder':
        """Configurar escala de sistema"""
        self.config.num_clients = client_count

        # Ajustes basados en escala
        if client_count > 100:
            self.config.max_clients_per_round = min(20, client_count // 5)
            self.config.communication_timeout = 600  # Más tiempo para grandes redes
        elif client_count > 50:
            self.config.max_clients_per_round = min(15, client_count // 3)
        else:
            self.config.max_clients_per_round = min(8, client_count // 2)

        return self

    def build(self) -> BaseFederatedConfig:
        """Construir configuración final"""
        return self.config

# ========================================================================
# SPECIALIZED SYSTEM CLASSES - PLACEHOLDERS FOR ACTUAL IMPLEMENTATIONS
# ========================================================================
class HealthcareFLSystem(BaseFederatedSystem):
    """Sistema FL especializado para healthcare"""

    async def register_client(self, client_id: str, public_key: str, use_case: UseCase = UseCase.HEALTHCARE, **kwargs) -> bool:
        """Registro con validaciones específicas de healthcare"""
        # Validar cumplimiento HIPAA/GDPR
        compliance_checks = {
            "hipaa_compliance": True,
            "data_anonymization": True,
            "medical_ethics": True
        }

        client_info = BaseClientInfo(
            client_id=client_id,
            public_key=public_key,
            use_case=use_case,
            compliance_status=compliance_checks,
            device_type=kwargs.get('device_type', 'hospital_server'),
            location=kwargs.get('location')
        )

        self.clients[client_id] = client_info
        self.logger.info(f"🏥 Cliente healthcare registrado: {client_id}")
        return True

    async def start_federated_round(self, round_number: int) -> str:
        """Iniciar ronda con protocolos healthcare específicos"""
        import uuid
        from federated_core import BaseFederatedRound

        round_id = str(uuid.uuid4())

        # Solo hospitales/clínicas verificados
        eligible_clients = [
            cid for cid, client in self.clients.items()
            if client.use_case == UseCase.HEALTHCARE and client.compliance_status.get('hipaa_compliance', False)
        ]

        selected_clients = eligible_clients[:self.config.max_clients_per_round]

        federated_round = BaseFederatedRound(
            round_id=round_id,
            round_number=round_number,
            start_time=self.security_manager.initialize_keys()[1],  # Use current time
            participating_clients=selected_clients
        )

        self.active_rounds[round_id] = federated_round
        self.logger.info(f"🏥 Ronda healthcare iniciada: {round_id} con {len(selected_clients)} clientes")
        return round_id

    def validate_medical_data_quality(self, client_id: str) -> float:
        """Validación específica de calidad de datos médicos"""
        if client_id not in self.clients:
            return 0.0

        client = self.clients[client_id]

        # Lógica específica para datos médicos
        # Verificar diversidad de casos, calidad de anotaciones, etc.
        quality_score = self.data_quality_assessor.evaluate({
            "data_type": "medical_records",
            "hipaa_compliant": client.compliance_status.get('hipaa_compliance', False),
        })

        return quality_score

class FinanceFLSystem(BaseFederatedSystem):
    """Sistema FL especializado para finanzas"""

    async def register_client(self, client_id: str, public_key: str, use_case: UseCase = UseCase.FINANCE, **kwargs) -> bool:
        """Registro con validaciones específicas de finanzas"""
        # Validar cumplimiento financiero (PCI DSS, SOX, etc.)
        compliance_checks = {
            "pci_dss_compliance": True,
            "data_encryption": True,
            "audit_trail": True,
            "risk_assessment": True
        }

        client_info = BaseClientInfo(
            client_id=client_id,
            public_key=public_key,
            use_case=use_case,
            compliance_status=compliance_checks,
            device_type=kwargs.get('device_type', 'bank_server'),
            location=kwargs.get('location')
        )

        self.clients[client_id] = client_info
        self.logger.info(f"💰 Cliente finance registrado: {client_id}")
        return True

    async def start_federated_round(self, round_number: int) -> str:
        """Iniciar ronda con protocolos finance específicos"""
        import uuid
        from federated_core import BaseFederatedRound

        round_id = str(uuid.uuid4())

        # Solo instituciones financieras verificadas
        eligible_clients = [
            cid for cid, client in self.clients.items()
            if client.use_case == UseCase.FINANCE and client.compliance_status.get('pci_dss_compliance', False)
        ]

        selected_clients = eligible_clients[:self.config.max_clients_per_round]

        federated_round = BaseFederatedRound(
            round_id=round_id,
            round_number=round_number,
            participating_clients=selected_clients
        )

        self.active_rounds[round_id] = federated_round
        self.logger.info(f"💰 Ronda finance iniciada: {round_id} con {len(selected_clients)} clientes")
        return round_id

    def assess_financial_risk(self, client_id: str) -> Dict[str, Any]:
        """Evaluación específica de riesgo financiero"""
        if client_id not in self.clients:
            return {"risk_level": "unknown"}

        client = self.clients[client_id]

        # Lógica específica para evaluación de riesgo financiero
        # Basado en reputación, historial de transacciones, etc.
        risk_assessment = {
            "risk_level": "low" if client.reputation_score > 0.8 else "medium",
            "credit_score": "excellent" if client.successful_contributions > 10 else "good",
            "regulatory_compliance": client.compliance_status.get('pci_dss_compliance', False)
        }

        return risk_assessment

class IoTFLSystem(BaseFederatedSystem):
    """Sistema FL especializado para IoT"""

    async def register_client(self, client_id: str, public_key: str, use_case: UseCase = UseCase.IOT, **kwargs) -> bool:
        """Registro con validaciones específicas de IoT"""
        compliance_checks = {
            "device_authentication": True,
            "secure_boot": True,
            "over_the_air_updates": True
        }

        client_info = BaseClientInfo(
            client_id=client_id,
            public_key=public_key,
            use_case=use_case,
            compliance_status=compliance_checks,
            device_type=kwargs.get('device_type', 'iot_device'),
            location=kwargs.get('location')
        )

        self.clients[client_id] = client_info
        self.logger.info(f"🔗 Cliente IoT registrado: {client_id}")
        return True

    async def start_federated_round(self, round_number: int) -> str:
        """Iniciar ronda optimizada para IoT"""
        import uuid
        from federated_core import BaseFederatedRound

        round_id = str(uuid.uuid4())

        # Clientes IoT disponibles (basado en batería, conectividad)
        eligible_clients = [
            cid for cid, client in self.clients.items()
            if client.use_case == UseCase.IOT and client.reputation_score > 0.3  # Threshold para estabilidad
        ]

        # Limitar basado en recursos disponibles
        max_iot_clients = min(self.config.max_clients_per_round, len(eligible_clients))
        selected_clients = eligible_clients[:max_iot_clients]

        federated_round = BaseFederatedRound(
            round_id=round_id,
            round_number=round_number,
            participating_clients=selected_clients
        )

        self.active_rounds[round_id] = federated_round
        self.logger.info(f"🔗 Ronda IoT iniciada: {round_id} con {len(selected_clients)} dispositivos")
        return round_id

    def optimize_for_edge_computing(self, client_id: str) -> Dict[str, Any]:
        """Optimizaciones específicas para edge computing"""
        if client_id not in self.clients:
            return {}

        client = self.clients[client_id]

        # Configuraciones optimizadas para recursos limitados
        optimizations = {
            "quantization_level": "8bit",
            "compression_ratio": 0.7,
            "batch_size_adaptive": True,
            "edge_computing_optimized": client.device_type == "iot_device"
        }

        return optimizations

# ========================================================================
# MAIN FACTORY CLASS
# ========================================================================
class FederatedFactory:
    """Factory principal para sistemas FL"""

    def __init__(self):
        # Register available systems
        self._register_systems()

    def _register_systems(self):
        """Registrar sistemas disponibles"""
        FederatedSystemRegistry.register("healthcare", HealthcareFLSystem)
        FederatedSystemRegistry.register("finance", FinanceFLSystem)
        FederatedSystemRegistry.register("iot", IoTFLSystem)

        # Attempt to register specialized systems if available
        self._try_register_specialized_systems()

    def _try_register_specialized_systems(self):
        """Intentar registrar sistemas especializados (lazy loading)"""
        try:
            # Try to import and register learning system
            from federated_learning_refactored import FederatedLearningSystem
            FederatedSystemRegistry.register("learning", FederatedLearningSystem)
            common_logger.info("📚 Sistema learning registrado")
        except ImportError:
            common_logger.warning("⚠️ Sistema learning no disponible")

        try:
            # Try to import and register API system
            from federated_api_refactored import FederatedAPI
            FederatedSystemRegistry.register("api", FederatedAPI)
            common_logger.info("🌐 Sistema API registrado")
        except ImportError:
            common_logger.warning("⚠️ Sistema API no disponible")

    def create_system(
        self,
        system_type: str,
        config: Optional[BaseFederatedConfig] = None,
        **kwargs
    ) -> BaseFederatedSystem:
        """
        Crear instancia de sistema FL

        Args:
            system_type: Tipo de sistema ('healthcare', 'finance', 'iot', 'learning', 'api')
            config: Configuración base (si None, usa config por defecto)
            **kwargs: Parámetros adicionales del sistema

        Returns:
            Instancia del sistema FL solicitado
        """
        if not FederatedSystemRegistry.is_available(system_type):
            available = list(FederatedSystemRegistry.get_available_systems().keys())
            raise ValueError(f"Sistema '{system_type}' no disponible. Disponibles: {available}")

        # Use provided config or create default
        if config is None:
            config = self._create_default_config(system_type, **kwargs)

        # Get system class
        system_class = FederatedSystemRegistry.get_available_systems()[system_type]

        # Create and return system instance
        common_logger.info(f"🏭 Creando sistema '{system_type}': {system_class.__name__}")
        return system_class(config)

    def _create_default_config(self, system_type: str, **kwargs) -> BaseFederatedConfig:
        """Crear configuración por defecto basada en tipo de sistema"""
        # Extract parameters from kwargs
        num_clients = kwargs.get('num_clients', 10)

        if system_type == "healthcare":
            return create_healthcare_config(num_clients=num_clients)
        elif system_type == "finance":
            return create_finance_config(num_clients=num_clients)
        elif system_type == "iot":
            return create_iot_config(num_clients=num_clients)
        else:
            # Generic config for other systems
            config = BaseFederatedConfig()
            config.num_clients = num_clients
            return config

    def create_config_builder(self) -> FederatedConfigBuilder:
        """Crear builder para configuraciones complejas"""
        return FederatedConfigBuilder()

    def get_available_systems(self) -> Dict[str, Type[BaseFederatedSystem]]:
        """Obtener sistemas disponibles"""
        return FederatedSystemRegistry.get_available_systems()

    def validate_system_requirements(self, system_type: str) -> Dict[str, bool]:
        """Validar requerimientos del sistema"""
        requirements = {
            "torch": DependencyManager.get_torch(),
            "crypten": DependencyManager.get_crypten(),
            "networking": True,  # Basic networking available
        }
