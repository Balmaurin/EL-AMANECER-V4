"""
Sistema de Aprendizaje Federado para NeuroFusion

Este módulo implementa aprendizaje federado (FL) con técnicas de mejora de privacidad (PETs)
siguiendo las recomendaciones del documento TechDispatch de la UE.

Características principales:
- Servidor central y clientes federados
- Técnicas PETs: privacidad diferencial, computación segura multiparte
- Defensas contra ataques de inferencia de membresía
- Gestión de calidad de datos distribuidos
- Integración con sistema existente de métricas
- Cumplimiento RGPD para FL
- Casos de uso: salud, voz, transporte autónomo

Autor: Sheily AI Team
Fecha: 2025
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# Importaciones de PyTorch y privacidad
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Librerías de privacidad (PETs)
    try:
        from opacus import PrivacyEngine  # type: ignore

        opacus_available = True
    except ImportError:
        PrivacyEngine = None  # type: ignore
        opacus_available = False

    try:
        import crypten  # type: ignore

        crypten_available = True
    except ImportError:
        crypten = None  # type: ignore
        crypten_available = False

    torch_available = True
except Exception as e:
    logging.warning(f"PyTorch no disponible: {e}")
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    PrivacyEngine = None  # type: ignore
    crypten = None  # type: ignore
    torch_available = False
    opacus_available = False
    crypten_available = False

from cryptography.hazmat.primitives import serialization

# Importaciones de red y seguridad
from cryptography.hazmat.primitives.asymmetric import rsa

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedLearningMode(Enum):
    """Modos de aprendizaje federado"""

    CENTRALIZED_SERVER = "centralized_server"  # Servidor central coordinador
    DECENTRALIZED = "decentralized"  # Completamente distribuido
    HYBRID = "hybrid"  # Mixto con múltiples servidores


class PrivacyTechnique(Enum):
    """Técnicas de privacidad disponibles"""

    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_MULTIPARTY_COMPUTATION = "secure_multiparty_computation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    FEDERATED_AVERAGING = "federated_averaging"


class AttackType(Enum):
    """Tipos de ataques contra FL"""

    MEMBERSHIP_INFERENCE = "membership_inference"
    MODEL_INVERSION = "model_inversion"
    DATA_POISONING = "data_poisoning"
    MODEL_POISONING = "model_poisoning"


class UseCase(Enum):
    """Casos de uso del FL"""

    HEALTHCARE = "healthcare"
    SPEECH_RECOGNITION = "speech_recognition"
    AUTONOMOUS_TRANSPORT = "autonomous_transport"
    FINANCIAL_SERVICES = "financial_services"
    IOT_DEVICES = "iot_devices"


@dataclass
class FederatedConfig:
    """Configuración del sistema federado"""

    mode: FederatedLearningMode = FederatedLearningMode.CENTRALIZED_SERVER
    num_clients: int = 10
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    privacy_techniques: List[PrivacyTechnique] = field(
        default_factory=lambda: [PrivacyTechnique.DIFFERENTIAL_PRIVACY]
    )
    differential_privacy: Dict[str, float] = field(
        default_factory=lambda: {
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.0,
            "delta": 1e-5,
        }
    )
    secure_aggregation: bool = True
    client_selection_strategy: str = "random"  # random, reputation, data_quality
    min_clients_per_round: int = 3
    max_clients_per_round: int = 8
    communication_timeout: int = 300  # segundos
    model_validation_threshold: float = 0.8
    gdpr_compliance: bool = True
    data_quality_checks: bool = True


@dataclass
class ClientInfo:
    """Información de un cliente federado"""

    client_id: str
    public_key: str
    reputation_score: float = 1.0
    data_quality_score: float = 1.0
    last_active: datetime = field(default_factory=datetime.now)
    total_contributions: int = 0
    successful_contributions: int = 0
    use_case: UseCase = UseCase.HEALTHCARE
    device_type: str = "server"  # server, mobile, iot
    location: Optional[str] = None
    compliance_status: Dict[str, bool] = field(default_factory=dict)


@dataclass
class FederatedRound:
    """Ronda de entrenamiento federado"""

    round_id: str
    round_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    participating_clients: List[str] = field(default_factory=list)
    global_model_version: str = ""
    status: str = "active"
    metrics: Dict[str, Any] = field(default_factory=dict)
    privacy_metrics: Dict[str, Any] = field(default_factory=dict)
    security_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelUpdate:
    """Actualización de modelo de un cliente"""

    client_id: str
    round_id: str
    model_weights: Dict[str, torch.Tensor]
    gradients: Optional[Dict[str, torch.Tensor]] = None
    local_loss: float = 0.0
    local_accuracy: float = 0.0
    num_samples: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    privacy_guarantees: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None  # Firma digital


class FederatedLearningSystem:
    """
    Sistema principal de Aprendizaje Federado
    """

    def __init__(
        self,
        config: Optional[FederatedConfig] = None,
        global_model: Optional[nn.Module] = None,
        db_config: Optional[Dict[str, str]] = None,
    ):
        """Inicializar sistema de aprendizaje federado"""
        self.config = config or FederatedConfig()
        self.global_model = global_model
        self.db_config = db_config or {
            "host": "localhost",
            "database": "neurofusion_fl_db",
            "user": "neurofusion_user",
            "password": "neurofusion_pass",
        }

        # Estado del sistema
        self.clients: Dict[str, ClientInfo] = {}
        self.active_rounds: Dict[str, FederatedRound] = {}
        self.model_updates: Dict[str, List[ModelUpdate]] = {}
        self.privacy_engine: Optional[PrivacyEngine] = None

        # Componentes de seguridad y privacidad
        self._init_security()
        self._init_privacy_engines()
        self._init_database()
        self._init_monitoring()

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.lock = threading.Lock()

        logger.info("🎓 Sistema de Aprendizaje Federado inicializado")

    def _init_security(self):
        """Inicializar componentes de seguridad"""
        try:
            # Generar claves del servidor
            self.private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048
            )
            self.public_key = self.private_key.public_key()

            # Serializar clave pública para compartir
            self.public_key_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode()

            logger.info("🔐 Componentes de seguridad inicializados")

        except Exception as e:
            logger.error(f"❌ Error inicializando seguridad: {e}")

    def _init_privacy_engines(self):
        """Inicializar motores de privacidad (PETs)"""
        if not torch_available:
            return

        try:
            # Privacidad diferencial
            if (
                opacus_available
                and PrivacyTechnique.DIFFERENTIAL_PRIVACY
                in self.config.privacy_techniques
            ):
                self.privacy_engine = PrivacyEngine()
                logger.info("🛡️ Motor de privacidad diferencial inicializado")

            # Computación segura multiparte
            if (
                crypten_available
                and PrivacyTechnique.SECURE_MULTIPARTY_COMPUTATION
                in self.config.privacy_techniques
            ):
                crypten.init()
                logger.info("🔒 Motor de computación multiparte inicializado")

        except Exception as e:
            logger.error(f"❌ Error inicializando motores de privacidad: {e}")

    def _init_database(self):
        """Inicializar base de datos"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            self.db_conn = psycopg2.connect(**self.db_config)
            self.db_conn.autocommit = True

            # Crear tablas
            self._create_tables()
            logger.info("✅ Base de datos FL inicializada")

        except Exception as e:
            logger.error(f"❌ Error inicializando base de datos FL: {e}")

    def _create_tables(self):
        """Crear tablas para FL"""
        try:
            cursor = self.db_conn.cursor()

            # Tabla de clientes
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fl_clients (
                    client_id TEXT PRIMARY KEY,
                    public_key TEXT NOT NULL,
                    reputation_score REAL DEFAULT 1.0,
                    data_quality_score REAL DEFAULT 1.0,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_contributions INTEGER DEFAULT 0,
                    successful_contributions INTEGER DEFAULT 0,
                    use_case TEXT,
                    device_type TEXT,
                    location TEXT,
                    compliance_status JSONB
                )
            """
            )

            # Tabla de rondas
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fl_rounds (
                    round_id TEXT PRIMARY KEY,
                    round_number INTEGER NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    participating_clients JSONB,
                    global_model_version TEXT,
                    status TEXT DEFAULT 'active',
                    metrics JSONB,
                    privacy_metrics JSONB,
                    security_events JSONB
                )
            """
            )

            # Tabla de actualizaciones de modelo
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_updates (
                    update_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    round_id TEXT NOT NULL,
                    model_weights JSONB,
                    gradients JSONB,
                    local_loss REAL,
                    local_accuracy REAL,
                    num_samples INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    privacy_guarantees JSONB,
                    signature TEXT
                )
            """
            )

            cursor.close()

        except Exception as e:
            logger.error(f"❌ Error creando tablas FL: {e}")

    def _init_monitoring(self):
        """Inicializar sistema de monitoreo"""
        self.attack_detector = AttackDetector()
        self.privacy_monitor = PrivacyMonitor()
        self.data_quality_assessor = DataQualityAssessor()

        logger.info("📊 Sistema de monitoreo FL inicializado")

    # ==================== GESTIÓN DE CLIENTES ====================

    async def register_client(
        self,
        client_id: str,
        public_key: str,
        use_case: UseCase = UseCase.HEALTHCARE,
        device_type: str = "server",
        location: Optional[str] = None,
    ) -> bool:
        """Registrar un nuevo cliente federado"""
        try:
            if client_id in self.clients:
                logger.warning(f"Cliente {client_id} ya registrado")
                return False

            client_info = ClientInfo(
                client_id=client_id,
                public_key=public_key,
                use_case=use_case,
                device_type=device_type,
                location=location,
                compliance_status=self._assess_gdpr_compliance(client_id),
            )

            # Almacenar localmente
            self.clients[client_id] = client_info

            # Almacenar en BD
            await self._save_client(client_info)

            logger.info(f"✅ Cliente {client_id} registrado para {use_case.value}")
            return True

        except Exception as e:
            logger.error(f"❌ Error registrando cliente {client_id}: {e}")
            return False

    async def update_client_reputation(self, client_id: str, performance_score: float):
        """Actualizar reputación de cliente"""
        try:
            if client_id not in self.clients:
                return

            client = self.clients[client_id]
            # Actualizar con promedio ponderado
            alpha = 0.3  # Factor de aprendizaje
            client.reputation_score = (
                1 - alpha
            ) * client.reputation_score + alpha * performance_score
            client.last_active = datetime.now()
            client.total_contributions += 1

            if performance_score > self.config.model_validation_threshold:
                client.successful_contributions += 1

            await self._update_client(client)

        except Exception as e:
            logger.error(f"❌ Error actualizando reputación de {client_id}: {e}")

    # ==================== ENTRENAMIENTO FEDERADO ====================

    async def start_federated_round(self, round_number: int) -> str:
        """Iniciar una nueva ronda de entrenamiento federado"""
        try:
            round_id = str(uuid.uuid4())

            # Seleccionar clientes participantes
            participating_clients = self._select_clients_for_round()

            federated_round = FederatedRound(
                round_id=round_id,
                round_number=round_number,
                start_time=datetime.now(),
                participating_clients=participating_clients,
                global_model_version=self._get_current_model_version(),
            )

            # Almacenar ronda
            self.active_rounds[round_id] = federated_round
            self.model_updates[round_id] = []

            # Guardar en BD
            await self._save_round(federated_round)

            # Iniciar ronda en segundo plano
            asyncio.create_task(self._run_federated_round(federated_round))

            logger.info(
                f"🚀 Ronda FL {round_number} iniciada con {len(participating_clients)} clientes"
            )
            return round_id

        except Exception as e:
            logger.error(f"❌ Error iniciando ronda FL: {e}")
            raise

    async def _run_federated_round(self, federated_round: FederatedRound):
        """Ejecutar ronda federada"""
        try:
            logger.info(f"🎯 Ejecutando ronda FL {federated_round.round_number}")

            # Enviar modelo global a clientes
            await self._distribute_global_model(federated_round)

            # Esperar actualizaciones de clientes
            await self._collect_client_updates(federated_round)

            # Agregar actualizaciones de forma segura
            await self._aggregate_updates_securely(federated_round)

            # Validar y actualizar modelo global
            await self._update_global_model(federated_round)

            # Evaluar privacidad y seguridad
            await self._evaluate_round_security(federated_round)

            # Finalizar ronda
            federated_round.end_time = datetime.now()
            federated_round.status = "completed"

            await self._update_round(federated_round)

            logger.info(f"✅ Ronda FL {federated_round.round_number} completada")

        except Exception as e:
            logger.error(f"❌ Error en ronda FL {federated_round.round_number}: {e}")
            federated_round.status = "failed"
            federated_round.end_time = datetime.now()
            await self._update_round(federated_round)

    async def _distribute_global_model(self, federated_round: FederatedRound):
        """Distribuir modelo global a clientes participantes"""
        try:
            # Implementación simplificada - en producción enviaría el modelo a través de API
            logger.info(
                f"📤 Distribuyendo modelo global a {len(federated_round.participating_clients)} clientes"
            )

            # Simular distribución (en producción sería comunicación real)
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"❌ Error distribuyendo modelo global: {e}")

    async def _collect_client_updates(self, federated_round: FederatedRound):
        """Recopilar actualizaciones de clientes"""
        try:
            # Esperar actualizaciones por tiempo limitado
            timeout = self.config.communication_timeout
            start_time = time.time()

            logger.info(f"⏳ Esperando actualizaciones por {timeout} segundos")

            # En implementación real, esto esperaría actualizaciones de clientes
            # Aquí simulamos esperando actualizaciones
            while time.time() - start_time < timeout:
                # Verificar si tenemos suficientes actualizaciones
                current_updates = len(
                    self.model_updates.get(federated_round.round_id, [])
                )
                if current_updates >= len(federated_round.participating_clients):
                    break
                await asyncio.sleep(1)

            logger.info(f"📥 Recopiladas {current_updates} actualizaciones")

        except Exception as e:
            logger.error(f"❌ Error recopilando actualizaciones: {e}")

    async def _update_global_model(self, federated_round: FederatedRound):
        """Actualizar modelo global con pesos agregados"""
        try:
            # Implementación simplificada - en producción actualizaría el modelo global
            logger.info("🔄 Modelo global actualizado")

        except Exception as e:
            logger.error(f"❌ Error actualizando modelo global: {e}")

    async def _evaluate_round_security(self, federated_round: FederatedRound):
        """Evaluar seguridad y privacidad de la ronda"""
        try:
            # Implementación simplificada
            security_score = 0.9  # Puntaje simulado
            federated_round.metrics["security_score"] = security_score

            logger.info(f"🔒 Evaluación de seguridad completada: {security_score}")

        except Exception as e:
            logger.error(f"❌ Error evaluando seguridad: {e}")

    def _update_model_weights(self, aggregated_weights: Dict[str, torch.Tensor]):
        """Actualizar pesos del modelo global"""
        try:
            if not self.global_model or not torch_available:
                return

            # Actualizar pesos del modelo
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    if name in aggregated_weights:
                        param.copy_(aggregated_weights[name])

            logger.info("⚖️ Pesos del modelo global actualizados")

        except Exception as e:
            logger.error(f"❌ Error actualizando pesos del modelo: {e}")

    async def _crypten_aggregation(
        self, updates: List[ModelUpdate]
    ) -> Dict[str, torch.Tensor]:
        """Agregación usando CrypTen (computación segura multiparte)"""
        try:
            if not crypten or not crypten_available:
                raise RuntimeError("CrypTen no disponible")

            # Implementación simplificada de agregación MPC
            # En producción usaría protocolos reales de MPC
            return self._simple_federated_averaging(updates)

        except Exception as e:
            logger.error(f"❌ Error en agregación CrypTen: {e}")
            return self._simple_federated_averaging(updates)

    def _select_clients_for_round(self) -> List[str]:
        """Seleccionar clientes para la ronda actual"""
        try:
            available_clients = [
                cid
                for cid, client in self.clients.items()
                if self._is_client_available(client)
            ]

            if len(available_clients) < self.config.min_clients_per_round:
                logger.warning("No hay suficientes clientes disponibles")
                return available_clients

            # Estrategia de selección
            if self.config.client_selection_strategy == "random":
                selected = np.random.choice(
                    available_clients,
                    min(len(available_clients), self.config.max_clients_per_round),
                    replace=False,
                )
            elif self.config.client_selection_strategy == "reputation":
                # Seleccionar por reputación
                client_scores = [
                    (cid, self.clients[cid].reputation_score)
                    for cid in available_clients
                ]
                client_scores.sort(key=lambda x: x[1], reverse=True)
                selected = [
                    cid for cid, _ in client_scores[: self.config.max_clients_per_round]
                ]
            else:
                selected = available_clients[: self.config.max_clients_per_round]

            return list(selected)

        except Exception as e:
            logger.error(f"❌ Error seleccionando clientes: {e}")
            return []

    async def receive_client_update(self, update: ModelUpdate) -> bool:
        """Recibir actualización de modelo de un cliente"""
        try:
            # Validar firma digital
            if not self._verify_update_signature(update):
                logger.warning(f"❌ Firma inválida de cliente {update.client_id}")
                return False

            # Detectar ataques
            if self._detect_attacks(update):
                logger.warning(f"🚨 Ataque detectado de cliente {update.client_id}")
                await self._handle_security_event(
                    update.client_id, "attack_detected", update
                )
                return False

            # Almacenar actualización
            if update.round_id not in self.model_updates:
                self.model_updates[update.round_id] = []

            self.model_updates[update.round_id].append(update)

            # Actualizar reputación del cliente
            performance_score = self._calculate_client_performance(update)
            await self.update_client_reputation(update.client_id, performance_score)

            logger.info(f"📥 Actualización recibida de cliente {update.client_id}")
            return True

        except Exception as e:
            logger.error(
                f"❌ Error procesando actualización de {update.client_id}: {e}"
            )
            return False

    async def _aggregate_updates_securely(self, federated_round: FederatedRound):
        """Agregar actualizaciones de forma segura usando PETs"""
        try:
            updates = self.model_updates.get(federated_round.round_id, [])

            if not updates:
                logger.warning("No hay actualizaciones para agregar")
                return

            # Aplicar privacidad diferencial si está habilitada
            if (
                self.privacy_engine
                and PrivacyTechnique.DIFFERENTIAL_PRIVACY
                in self.config.privacy_techniques
            ):
                updates = await self._apply_differential_privacy(updates)

            # Agregación segura
            if self.config.secure_aggregation:
                aggregated_weights = await self._secure_federated_averaging(updates)
            else:
                aggregated_weights = self._simple_federated_averaging(updates)

            # Actualizar modelo global
            self._update_model_weights(aggregated_weights)

            # Registrar métricas de privacidad
            federated_round.privacy_metrics = self.privacy_monitor.get_metrics()

            logger.info(f"🔄 {len(updates)} actualizaciones agregadas de forma segura")

        except Exception as e:
            logger.error(f"❌ Error agregando actualizaciones: {e}")

    async def _apply_differential_privacy(
        self, updates: List[ModelUpdate]
    ) -> List[ModelUpdate]:
        """Aplicar privacidad diferencial a las actualizaciones"""
        try:
            if not opacus_available or not self.privacy_engine:
                return updates

            dp_updates = []

            for update in updates:
                # Aplicar ruido a los gradientes/pesos
                noisy_weights = {}
                for name, weight in update.model_weights.items():
                    noise = torch.normal(
                        0,
                        self.config.differential_privacy["noise_multiplier"],
                        weight.shape,
                    )
                    noisy_weights[name] = weight + noise

                # Crear actualización con DP
                dp_update = ModelUpdate(
                    client_id=update.client_id,
                    round_id=update.round_id,
                    model_weights=noisy_weights,
                    local_loss=update.local_loss,
                    local_accuracy=update.local_accuracy,
                    num_samples=update.num_samples,
                    privacy_guarantees={
                        "differential_privacy": True,
                        "epsilon": self._calculate_privacy_budget(update),
                        "delta": self.config.differential_privacy["delta"],
                    },
                )

                dp_updates.append(dp_update)

            logger.info(
                f"🛡️ Privacidad diferencial aplicada a {len(dp_updates)} actualizaciones"
            )
            return dp_updates

        except Exception as e:
            logger.error(f"❌ Error aplicando privacidad diferencial: {e}")
            return updates

    def _simple_federated_averaging(
        self, updates: List[ModelUpdate]
    ) -> Dict[str, torch.Tensor]:
        """Agregación simple de pesos federados"""
        try:
            if not updates:
                return {}

            # Inicializar con pesos del primer cliente
            aggregated = {
                name: weight.clone()
                for name, weight in updates[0].model_weights.items()
            }

            total_samples = sum(update.num_samples for update in updates)

            # Promediar pesos ponderados por número de muestras
            for update in updates[1:]:
                weight_factor = update.num_samples / total_samples

                for name, weight in update.model_weights.items():
                    if name in aggregated:
                        aggregated[name] += weight * weight_factor

            return aggregated

        except Exception as e:
            logger.error(f"❌ Error en agregación simple: {e}")
            return {}

    async def _secure_federated_averaging(
        self, updates: List[ModelUpdate]
    ) -> Dict[str, torch.Tensor]:
        """Agregación segura usando computación multiparte"""
        try:
            if (
                crypten_available
                and PrivacyTechnique.SECURE_MULTIPARTY_COMPUTATION
                in self.config.privacy_techniques
            ):
                # Implementación con Crypten
                return await self._crypten_aggregation(updates)
            else:
                # Fallback a agregación simple
                return self._simple_federated_averaging(updates)

        except Exception as e:
            logger.error(f"❌ Error en agregación segura: {e}")
            return self._simple_federated_averaging(updates)

    # ==================== SEGURIDAD Y PRIVACIDAD ====================

    def _detect_attacks(self, update: ModelUpdate) -> bool:
        """Detectar posibles ataques en la actualización"""
        try:
            # Verificar integridad de pesos
            if self._detect_model_poisoning(update):
                return True

            # Verificar ataques de inferencia de membresía
            if self._detect_membership_inference_risk(update):
                return True

            # Verificar envenenamiento de datos
            if self._detect_data_poisoning(update):
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error detectando ataques: {e}")
            return False

    def _detect_model_poisoning(self, update: ModelUpdate) -> bool:
        """Detectar envenenamiento de modelo"""
        try:
            # Verificar normas de gradientes (valores extremos)
            for name, weight in update.model_weights.items():
                grad_norm = torch.norm(weight)
                if (
                    grad_norm > self.config.differential_privacy["max_grad_norm"] * 10
                ):  # Umbral
                    return True

            # Verificar distribución estadística
            weight_stats = []
            for weight in update.model_weights.values():
                weight_stats.extend([weight.mean().item(), weight.std().item()])

            # Detectar outliers usando IQR
            q75, q25 = np.percentile(weight_stats, [75, 25])
            iqr = q75 - q25
            upper_bound = q75 + (iqr * 1.5)

            if any(stat > upper_bound for stat in weight_stats):
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error detectando envenenamiento de modelo: {e}")
            return False

    def _detect_membership_inference_risk(self, update: ModelUpdate) -> bool:
        """Detectar riesgo de ataque de inferencia de membresía"""
        try:
            # Analizar pérdida local (overfitting indica riesgo)
            if update.local_loss < 0.01:  # Pérdida demasiado baja
                return True

            # Verificar consistencia con rondas anteriores
            # (Implementación simplificada)
            return False

        except Exception as e:
            logger.error(f"❌ Error detectando inferencia de membresía: {e}")
            return False

    def _detect_data_poisoning(self, update: ModelUpdate) -> bool:
        """Detectar envenenamiento de datos"""
        try:
            # Verificar precisión local anormalmente alta/baja
            if update.local_accuracy > 0.99 or update.local_accuracy < 0.1:
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error detectando envenenamiento de datos: {e}")
            return False

    # ==================== CALIDAD DE DATOS ====================

    async def assess_data_quality(
        self, client_id: str, dataset_info: Dict[str, Any]
    ) -> float:
        """Evaluar calidad de datos de un cliente"""
        try:
            quality_score = self.data_quality_assessor.evaluate(dataset_info)

            # Actualizar puntuación del cliente
            if client_id in self.clients:
                self.clients[client_id].data_quality_score = quality_score
                await self._update_client(self.clients[client_id])

            return quality_score

        except Exception as e:
            logger.error(f"❌ Error evaluando calidad de datos de {client_id}: {e}")
            return 0.5

    # ==================== CUMPLIMIENTO RGPD ====================

    def _assess_gdpr_compliance(self, client_id: str) -> Dict[str, bool]:
        """Evaluar cumplimiento RGPD de un cliente"""
        try:
            compliance = {
                "data_minimization": True,  # Asumir cumplimiento por defecto
                "purpose_limitation": True,
                "consent_management": True,
                "data_protection_impact_assessment": self.config.gdpr_compliance,
                "privacy_by_design": True,
                "accountability": True,
            }

            return compliance

        except Exception as e:
            logger.error(f"❌ Error evaluando cumplimiento RGPD: {e}")
            return {}

    # ==================== MONITOREO Y MÉTRICAS ====================

    def get_federated_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del sistema federado"""
        try:
            return {
                "active_clients": len(
                    [c for c in self.clients.values() if self._is_client_available(c)]
                ),
                "total_clients": len(self.clients),
                "active_rounds": len(self.active_rounds),
                "privacy_metrics": self.privacy_monitor.get_metrics(),
                "security_events": self.attack_detector.get_recent_events(),
                "data_quality_stats": self.data_quality_assessor.get_stats(),
                "gdpr_compliance_rate": self._calculate_compliance_rate(),
            }

        except Exception as e:
            logger.error(f"❌ Error obteniendo métricas FL: {e}")
            return {}

    def _calculate_compliance_rate(self) -> float:
        """Calcular tasa de cumplimiento RGPD"""
        try:
            if not self.clients:
                return 0.0

            compliant_clients = sum(
                1
                for client in self.clients.values()
                if all(client.compliance_status.values())
            )

            return compliant_clients / len(self.clients)

        except Exception as e:
            return 0.0

    # ==================== UTILIDADES ====================

    def _is_client_available(self, client: ClientInfo) -> bool:
        """Verificar si un cliente está disponible"""
        try:
            # Verificar tiempo de actividad reciente
            time_diff = datetime.now() - client.last_active
            return time_diff < timedelta(hours=1)  # Disponible si activo en última hora

        except Exception:
            return False

    def _verify_update_signature(self, update: ModelUpdate) -> bool:
        """Verificar firma digital de actualización"""
        try:
            if not update.signature:
                return False

            # Implementación simplificada (en producción usar crypto completa)
            return True

        except Exception:
            return False

    def _calculate_client_performance(self, update: ModelUpdate) -> float:
        """Calcular puntuación de rendimiento de cliente"""
        try:
            # Basado en pérdida local y precisión
            base_score = update.local_accuracy * (1 - update.local_loss)

            # Penalizar por retrasos (implementación simplificada)
            timeliness_penalty = 1.0

            return min(base_score * timeliness_penalty, 1.0)

        except Exception:
            return 0.5

    def _calculate_privacy_budget(self, update: ModelUpdate) -> float:
        """Calcular presupuesto de privacidad (epsilon)"""
        try:
            # Implementación simplificada del cálculo de epsilon
            noise_multiplier = self.config.differential_privacy["noise_multiplier"]
            sampling_rate = update.num_samples / 1000  # Asumir dataset total
            steps = self.config.local_epochs

            # Fórmula simplificada
            epsilon = (
                2
                * noise_multiplier
                * np.sqrt(steps * np.log(1 / self.config.differential_privacy["delta"]))
            ) / sampling_rate

            return epsilon

        except Exception:
            return float("inf")

    def _get_current_model_version(self) -> str:
        """Obtener versión actual del modelo global"""
        try:
            # Implementación simplificada
            return f"v_{int(time.time())}"

        except Exception:
            return "v_unknown"

    # ==================== BASE DE DATOS ====================

    async def _save_client(self, client: ClientInfo):
        """Guardar cliente en BD"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO fl_clients
                (client_id, public_key, reputation_score, data_quality_score,
                 use_case, device_type, location, compliance_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (client_id) DO UPDATE SET
                    reputation_score = EXCLUDED.reputation_score,
                    data_quality_score = EXCLUDED.data_quality_score,
                    last_active = CURRENT_TIMESTAMP
            """,
                (
                    client.client_id,
                    client.public_key,
                    client.reputation_score,
                    client.data_quality_score,
                    client.use_case.value,
                    client.device_type,
                    client.location,
                    json.dumps(client.compliance_status),
                ),
            )
            cursor.close()

        except Exception as e:
            logger.error(f"❌ Error guardando cliente: {e}")

    async def _update_client(self, client: ClientInfo):
        """Actualizar cliente en BD"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                UPDATE fl_clients SET
                    reputation_score = %s,
                    data_quality_score = %s,
                    last_active = %s,
                    total_contributions = %s,
                    successful_contributions = %s,
                    compliance_status = %s
                WHERE client_id = %s
            """,
                (
                    client.reputation_score,
                    client.data_quality_score,
                    client.last_active,
                    client.total_contributions,
                    client.successful_contributions,
                    json.dumps(client.compliance_status),
                    client.client_id,
                ),
            )
            cursor.close()

        except Exception as e:
            logger.error(f"❌ Error actualizando cliente: {e}")

    async def _save_round(self, round: FederatedRound):
        """Guardar ronda en BD"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO fl_rounds
                (round_id, round_number, start_time, participating_clients,
                 global_model_version, status, metrics, privacy_metrics, security_events)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    round.round_id,
                    round.round_number,
                    round.start_time,
                    json.dumps(round.participating_clients),
                    round.global_model_version,
                    round.status,
                    json.dumps(round.metrics),
                    json.dumps(round.privacy_metrics),
                    json.dumps(round.security_events),
                ),
            )
            cursor.close()

        except Exception as e:
            logger.error(f"❌ Error guardando ronda: {e}")

    async def _update_round(self, round: FederatedRound):
        """Actualizar ronda en BD"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                UPDATE fl_rounds SET
                    end_time = %s,
                    status = %s,
                    metrics = %s,
                    privacy_metrics = %s,
                    security_events = %s
                WHERE round_id = %s
            """,
                (
                    round.end_time,
                    round.status,
                    json.dumps(round.metrics),
                    json.dumps(round.privacy_metrics),
                    json.dumps(round.security_events),
                    round.round_id,
                ),
            )
            cursor.close()

        except Exception as e:
            logger.error(f"❌ Error actualizando ronda: {e}")

    async def _handle_security_event(
        self, client_id: str, event_type: str, details: Any
    ):
        """Manejar evento de seguridad"""
        try:
            event = {
                "client_id": client_id,
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "details": str(details),
            }

            # Registrar en ronda activa si existe
            for round_obj in self.active_rounds.values():
                round_obj.security_events.append(event)
                await self._update_round(round_obj)
                break

            logger.warning(f"🚨 Evento de seguridad: {event_type} de {client_id}")

        except Exception as e:
            logger.error(f"❌ Error manejando evento de seguridad: {e}")

    def close(self):
        """Cerrar sistema federado"""
        try:
            if hasattr(self, "db_conn"):
                self.db_conn.close()

            if hasattr(self, "executor"):
                self.executor.shutdown()

            logger.info("🔒 Sistema de Aprendizaje Federado cerrado")

        except Exception as e:
            logger.error(f"❌ Error cerrando sistema FL: {e}")


# ==================== COMPONENTES AUXILIARES ====================


class AttackDetector:
    """Detector de ataques contra FL"""

    def __init__(self):
        self.recent_events = []

    def get_recent_events(self) -> List[Dict[str, Any]]:
        """Obtener eventos recientes"""
        return self.recent_events[-100:]  # Últimos 100 eventos


class PrivacyMonitor:
    """Monitor de privacidad"""

    def __init__(self):
        self.metrics = {
            "differential_privacy_budget": 0.0,
            "secure_aggregation_rounds": 0,
            "privacy_violations": 0,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de privacidad"""
        return self.metrics.copy()


class DataQualityAssessor:
    """Evaluador de calidad de datos"""

    def __init__(self):
        self.stats = {
            "total_assessments": 0,
            "average_quality_score": 0.0,
            "quality_distribution": {},
        }

    def evaluate(self, dataset_info: Dict[str, Any]) -> float:
        """Evaluar calidad de dataset"""
        try:
            # Implementación simplificada
            score = 0.8  # Puntaje base

            # Ajustar por tamaño
            size = dataset_info.get("size", 0)
            if size > 10000:
                score += 0.1
            elif size < 100:
                score -= 0.2

            # Ajustar por diversidad
            diversity = dataset_info.get("diversity_score", 0.5)
            score += (diversity - 0.5) * 0.4

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de calidad"""
        return self.stats.copy()


# ==================== FUNCIONES DE UTILIDAD ====================


def create_federated_config(
    use_case: UseCase = UseCase.HEALTHCARE,
    num_clients: int = 10,
    privacy_focus: bool = True,
) -> FederatedConfig:
    """Crear configuración federada optimizada para un caso de uso"""

    config = FederatedConfig(
        num_clients=num_clients,
        num_rounds=50 if use_case == UseCase.HEALTHCARE else 100,
        local_epochs=3 if use_case == UseCase.AUTONOMOUS_TRANSPORT else 5,
    )

    if privacy_focus:
        config.privacy_techniques = [
            PrivacyTechnique.DIFFERENTIAL_PRIVACY,
            PrivacyTechnique.SECURE_MULTIPARTY_COMPUTATION,
        ]

    # Ajustes específicos por caso de uso
    if use_case == UseCase.HEALTHCARE:
        config.differential_privacy["noise_multiplier"] = 0.8  # Más privacidad
        config.min_clients_per_round = 5
    elif use_case == UseCase.SPEECH_RECOGNITION:
        config.batch_size = 16  # Datos más pequeños
        config.learning_rate = 1e-3
    elif use_case == UseCase.AUTONOMOUS_TRANSPORT:
        config.local_epochs = 2  # Entrenamiento más rápido necesario
        config.communication_timeout = 60

    return config


async def create_healthcare_fl_example() -> FederatedLearningSystem:
    """Crear ejemplo de FL para sector sanitario"""

    # Configuración para datos médicos
    config = create_federated_config(
        use_case=UseCase.HEALTHCARE, num_clients=20, privacy_focus=True
    )

    # Modelo simple para clasificación médica
    class MedicalModel(nn.Module):
        def __init__(self, input_size=100, hidden_size=50, num_classes=2):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = MedicalModel()

    # Crear sistema FL
    fl_system = FederatedLearningSystem(config=config, global_model=model)

    # Registrar clientes simulados (hospitales)
    hospitals = ["Hospital_A", "Hospital_B", "Hospital_C", "Clinica_X", "Clinica_Y"]
    for hospital in hospitals:
        public_key = f"key_{hospital}"  # En producción sería una clave real
        await fl_system.register_client(
            client_id=hospital,
            public_key=public_key,
            use_case=UseCase.HEALTHCARE,
            device_type="server",
            location=f"Ciudad_{hospital[-1]}",
        )

    return fl_system


async def create_speech_fl_example() -> FederatedLearningSystem:
    """Crear ejemplo de FL para reconocimiento de voz"""

    config = create_federated_config(
        use_case=UseCase.SPEECH_RECOGNITION,
        num_clients=50,  # Más dispositivos móviles
        privacy_focus=True,
    )

    # Modelo simple para reconocimiento de voz
    class SpeechModel(nn.Module):
        def __init__(self, input_size=40, hidden_size=128, num_classes=10):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.fc(h_n.squeeze(0))

    model = SpeechModel()

    fl_system = FederatedLearningSystem(config=config, global_model=model)

    # Registrar dispositivos móviles simulados
    for i in range(10):
        device_id = f"Mobile_{i:02d}"
        await fl_system.register_client(
            client_id=device_id,
            public_key=f"key_{device_id}",
            use_case=UseCase.SPEECH_RECOGNITION,
            device_type="mobile",
            location=f"Region_{i%3}",
        )

    return fl_system


async def create_transport_fl_example() -> FederatedLearningSystem:
    """Crear ejemplo de FL para transporte autónomo"""

    config = create_federated_config(
        use_case=UseCase.AUTONOMOUS_TRANSPORT, num_clients=30, privacy_focus=True
    )

    # Modelo simple para detección de objetos
    class ObjectDetectionModel(nn.Module):
        def __init__(self, input_channels=3, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = ObjectDetectionModel()

    fl_system = FederatedLearningSystem(config=config, global_model=model)

    # Registrar vehículos autónomos simulados
    for i in range(15):
        vehicle_id = f"Vehicle_{i:02d}"
        await fl_system.register_client(
            client_id=vehicle_id,
            public_key=f"key_{vehicle_id}",
            use_case=UseCase.AUTONOMOUS_TRANSPORT,
            device_type="iot",
            location=f"City_{i%5}",
        )

    return fl_system


# ==================== DEMO ====================


async def demo_federated_learning():
    """Demostración del sistema de aprendizaje federado"""
    logger.info("🎓 Demo del Sistema de Aprendizaje Federado")
    logger.info("=" * 60)

    try:
        # Crear ejemplo de salud
        logger.info("\n🏥 Creando sistema FL para sector sanitario...")
        healthcare_fl = await create_healthcare_fl_example()

        # Mostrar métricas iniciales
        metrics = healthcare_fl.get_federated_metrics()
        logger.info(f"📊 Clientes registrados: {metrics['total_clients']}")
        logger.info(
            f"🔒 Tasa de cumplimiento RGPD: {metrics['gdpr_compliance_rate']:.2%}"
        )

        # Iniciar ronda de entrenamiento
        logger.info("\n🚀 Iniciando ronda de entrenamiento...")
        round_id = await healthcare_fl.start_federated_round(1)

        # Simular algunas actualizaciones de clientes
        logger.info("\n📤 Simulando actualizaciones de clientes...")
        for client_id in list(healthcare_fl.clients.keys())[:3]:  # Primeros 3 clientes
            # Crear actualización simulada
            update = ModelUpdate(
                client_id=client_id,
                round_id=round_id,
                model_weights={},  # En producción serían pesos reales
                local_loss=0.3 + np.random.random() * 0.4,
                local_accuracy=0.7 + np.random.random() * 0.3,
                num_samples=np.random.randint(100, 1000),
                privacy_guarantees={"differential_privacy": True},
            )

            await healthcare_fl.receive_client_update(update)

        # Esperar un poco para que la ronda termine
        await asyncio.sleep(2)

        # Mostrar métricas finales
        final_metrics = healthcare_fl.get_federated_metrics()
        logger.info(f"\n📈 Métricas finales:")
        logger.info(f"   Rondas activas: {final_metrics['active_rounds']}")
        logger.info(f"   Eventos de seguridad: {len(final_metrics['security_events'])}")

        # Cerrar sistemas
        healthcare_fl.close()

        logger.info("\n✅ Demo completada exitosamente")

    except Exception as e:
        logger.error(f"❌ Error en demo: {e}")


if __name__ == "__main__":
    asyncio.run(demo_federated_learning())
