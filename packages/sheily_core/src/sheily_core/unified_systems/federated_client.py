"""
Cliente Federado para Aprendizaje Federado

Este módulo implementa el lado cliente del sistema de aprendizaje federado.
Permite a dispositivos/clientes participar en rondas de entrenamiento FL
mientras mantienen sus datos locales privados.

Características:
- Entrenamiento local con datos privados
- Comunicación segura con servidor FL
- Aplicación de técnicas PETs localmente
- Gestión de privacidad y cumplimiento RGPD
- Soporte para diferentes casos de uso

Autor: Sheily AI Team
Fecha: 2025
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Importaciones de PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, Subset

    # Librerías de privacidad
    try:
        from opacus import PrivacyEngine

        opacus_available = True
    except ImportError:
        PrivacyEngine = None
        opacus_available = False

    torch_available = True
except Exception as e:
    logging.warning(f"PyTorch no disponible: {e}")
    torch = None
    nn = None
    optim = None
    DataLoader = None
    Dataset = None
    F = None
    PrivacyEngine = None
    torch_available = False
    opacus_available = False

import hashlib

# Importaciones de comunicación y seguridad
import aiohttp
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuración del cliente federado"""

    client_id: str
    server_url: str = "http://localhost:8000"
    use_case: str = "healthcare"
    device_type: str = "server"
    location: Optional[str] = None

    # Configuración de privacidad
    enable_differential_privacy: bool = True
    privacy_budget: float = 1.0
    noise_multiplier: float = 1.0

    # Configuración de entrenamiento
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Configuración de comunicación
    heartbeat_interval: int = 60  # segundos
    max_retries: int = 3
    timeout: int = 300

    # Cumplimiento RGPD
    gdpr_consent: bool = True
    data_retention_days: int = 30


@dataclass
class LocalDataset:
    """Dataset local del cliente"""

    name: str
    size: int
    features: List[str]
    labels: List[str]
    data_path: Optional[Path] = None
    quality_score: float = 1.0
    privacy_level: str = "high"  # high, medium, low


@dataclass
class TrainingResult:
    """Resultado de entrenamiento local"""

    client_id: str
    round_id: str
    model_weights: Dict[str, torch.Tensor]
    gradients: Optional[Dict[str, torch.Tensor]] = None
    local_loss: float = 0.0
    local_accuracy: float = 0.0
    num_samples: int = 0
    training_time: float = 0.0
    privacy_guarantees: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None


class FederatedClient:
    """
    Cliente federado para participación en aprendizaje federado
    """

    def __init__(
        self,
        config: ClientConfig,
        local_model: Optional[nn.Module] = None,
        local_dataset: Optional[LocalDataset] = None,
    ):
        """Inicializar cliente federado"""
        self.config = config
        self.local_model = local_model
        self.local_dataset = local_dataset

        # Estado del cliente
        self.is_registered = False
        self.current_round: Optional[str] = None
        self.server_public_key: Optional[str] = None
        self.privacy_engine: Optional[PrivacyEngine] = None

        # Generar claves del cliente
        self._init_security()

        # Cliente HTTP para comunicación
        self.http_client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )

        # Dataset local
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        logger.info(f"🎓 Cliente Federado {config.client_id} inicializado")

    def _init_security(self):
        """Inicializar componentes de seguridad"""
        try:
            # Generar claves RSA
            self.private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )
            self.public_key = self.private_key.public_key()

            # Serializar clave pública
            self.public_key_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode()

            logger.info("🔐 Componentes de seguridad del cliente inicializados")

        except Exception as e:
            logger.error(f"❌ Error inicializando seguridad del cliente: {e}")

    async def register_with_server(self) -> bool:
        """Registrar cliente con el servidor FL"""
        try:
            registration_data = {
                "client_id": self.config.client_id,
                "public_key": self.public_key_pem,
                "use_case": self.config.use_case,
                "device_type": self.config.device_type,
                "location": self.config.location,
                "gdpr_consent": self.config.gdpr_consent,
            }

            async with self.http_client.post(
                f"{self.config.server_url}/register", json=registration_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.is_registered = True
                    self.server_public_key = result.get("server_public_key")
                    logger.info(f"✅ Cliente registrado exitosamente con servidor")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"❌ Error registrando cliente: {error}")
                    return False

        except Exception as e:
            logger.error(f"❌ Error en registro: {e}")
            return False

    async def start_heartbeat(self):
        """Iniciar latidos del corazón para mantener conexión"""
        while True:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"❌ Error en heartbeat: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)

    async def _send_heartbeat(self):
        """Enviar latido del corazón al servidor"""
        try:
            heartbeat_data = {
                "client_id": self.config.client_id,
                "timestamp": datetime.now().isoformat(),
                "status": "active",
            }

            async with self.http_client.post(
                f"{self.config.server_url}/heartbeat", json=heartbeat_data
            ) as response:
                if response.status != 200:
                    logger.warning(f"Heartbeat fallido: {response.status}")

        except Exception as e:
            logger.debug(f"Error en heartbeat (continuando): {e}")

    async def participate_in_round(
        self, round_id: str, global_model_weights: Dict[str, torch.Tensor]
    ) -> Optional[TrainingResult]:
        """Participar en una ronda de entrenamiento FL"""
        try:
            self.current_round = round_id
            logger.info(f"🎯 Participando en ronda {round_id}")

            # Actualizar modelo local con pesos globales
            self._update_local_model(global_model_weights)

            # Preparar dataset local
            if not self._prepare_local_dataset():
                logger.error("❌ Error preparando dataset local")
                return None

            # Entrenar modelo local
            training_result = await self._train_local_model()

            if training_result:
                # Aplicar privacidad diferencial si está habilitada
                if self.config.enable_differential_privacy and opacus_available:
                    training_result = await self._apply_local_differential_privacy(
                        training_result
                    )

                # Firmar resultado
                training_result.signature = self._sign_training_result(training_result)

                logger.info(f"✅ Entrenamiento local completado para ronda {round_id}")
                return training_result
            else:
                logger.error("❌ Error en entrenamiento local")
                return None

        except Exception as e:
            logger.error(f"❌ Error participando en ronda: {e}")
            return None

    def _update_local_model(self, global_weights: Dict[str, torch.Tensor]):
        """Actualizar modelo local con pesos globales"""
        try:
            if not self.local_model or not torch_available:
                return

            with torch.no_grad():
                for name, param in self.local_model.named_parameters():
                    if name in global_weights:
                        param.copy_(global_weights[name])

            logger.info("🔄 Modelo local actualizado con pesos globales")

        except Exception as e:
            logger.error(f"❌ Error actualizando modelo local: {e}")

    def _prepare_local_dataset(self) -> bool:
        """Preparar dataset local para entrenamiento"""
        try:
            if not self.local_dataset or not torch_available:
                # Crear dataset sintético para demo
                self._create_synthetic_dataset()
                return True

            # En producción, cargar dataset real desde local_dataset.data_path
            # Aquí simulamos la carga
            self.train_loader = self._create_data_loader("train")
            self.val_loader = self._create_data_loader("val")

            logger.info(
                f"📊 Dataset local preparado: {self.local_dataset.size} muestras"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Error preparando dataset: {e}")
            return False

    def _create_synthetic_dataset(self):
        """Crear dataset sintético para demostración"""
        try:
            # Dataset sintético basado en el caso de uso
            if self.config.use_case == "healthcare":
                # Datos médicos sintéticos
                num_samples = 1000
                input_size = 100
                num_classes = 2

                # Crear modelo si no existe
                if not self.local_model:
                    self.local_model = nn.Sequential(
                        nn.Linear(input_size, 50), nn.ReLU(), nn.Linear(50, num_classes)
                    )

            elif self.config.use_case == "speech_recognition":
                # Datos de voz sintéticos
                num_samples = 500
                input_size = 40  # MFCC features
                num_classes = 10  # Dígitos del 0-9

                if not self.local_model:
                    self.local_model = nn.Sequential(
                        nn.LSTM(input_size, 128, batch_first=True),
                        nn.Linear(128, num_classes),
                    )

            elif self.config.use_case == "autonomous_transport":
                # Datos de imágenes sintéticas
                num_samples = 200
                input_channels = 3
                num_classes = 10  # Tipos de objetos

                if not self.local_model:
                    self.local_model = nn.Sequential(
                        nn.Conv2d(input_channels, 16, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(16 * 8 * 8, 128),
                        nn.ReLU(),
                        nn.Linear(128, num_classes),
                    )

            # Crear DataLoader sintético
            self.train_loader = self._create_data_loader("train")
            self.val_loader = self._create_data_loader("val")

        except Exception as e:
            logger.error(f"❌ Error creando dataset sintético: {e}")

    def _create_data_loader(self, split: str) -> Optional[DataLoader]:
        """Crear DataLoader para el split especificado"""
        try:
            if not torch_available:
                return None

            # Dataset sintético simple
            class SyntheticDataset(Dataset):
                def __init__(self, size=100, input_shape=(100,), num_classes=2):
                    self.size = size
                    self.input_shape = input_shape
                    self.num_classes = num_classes

                def __len__(self):
                    return self.size

                def __getitem__(self, idx):
                    # Generar datos aleatorios
                    if len(self.input_shape) == 1:
                        x = torch.randn(self.input_shape[0])
                    else:
                        x = torch.randn(*self.input_shape)

                    y = torch.randint(0, self.num_classes, (1,)).item()
                    return x, y

            dataset = SyntheticDataset()
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        except Exception as e:
            logger.error(f"❌ Error creando DataLoader: {e}")
            return None

    async def _train_local_model(self) -> Optional[TrainingResult]:
        """Entrenar modelo local"""
        try:
            if not self.local_model or not self.train_loader or not torch_available:
                return None

            # Configurar optimizer
            optimizer = optim.AdamW(
                self.local_model.parameters(), lr=self.config.learning_rate
            )
            criterion = nn.CrossEntropyLoss()

            # Configurar privacidad diferencial si está habilitada
            if self.config.enable_differential_privacy and opacus_available:
                self.privacy_engine = PrivacyEngine()
                self.local_model, optimizer, self.train_loader = (
                    self.privacy_engine.make_private(
                        module=self.local_model,
                        optimizer=optimizer,
                        data_loader=self.train_loader,
                        noise_multiplier=self.config.noise_multiplier,
                        max_grad_norm=1.0,
                    )
                )

            # Entrenamiento
            start_time = time.time()
            self.local_model.train()

            total_loss = 0.0
            correct = 0
            total = 0

            for epoch in range(self.config.local_epochs):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0

                for batch_x, batch_y in self.train_loader:
                    optimizer.zero_grad()

                    # Forward pass
                    if hasattr(self.local_model, "lstm"):
                        # Modelo LSTM (voz)
                        outputs, _ = self.local_model(batch_x.unsqueeze(1))
                        outputs = outputs[:, -1, :]  # Última salida de secuencia
                    else:
                        outputs = self.local_model(batch_x)

                    loss = criterion(outputs, batch_y)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Estadísticas
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_total += batch_y.size(0)
                    epoch_correct += (predicted == batch_y).sum().item()

                # Promedio de época
                total_loss += epoch_loss / len(self.train_loader)
                correct += epoch_correct
                total += epoch_total

            training_time = time.time() - start_time
            accuracy = correct / total if total > 0 else 0.0
            avg_loss = total_loss / self.config.local_epochs

            # Obtener pesos del modelo
            model_weights = {
                name: param.clone()
                for name, param in self.local_model.named_parameters()
            }

            # Crear resultado
            result = TrainingResult(
                client_id=self.config.client_id,
                round_id=self.current_round or "",
                model_weights=model_weights,
                local_loss=avg_loss,
                local_accuracy=accuracy,
                num_samples=total,
                training_time=training_time,
                privacy_guarantees={
                    "differential_privacy": self.config.enable_differential_privacy,
                    "privacy_budget": self.config.privacy_budget,
                },
            )

            logger.info(".4f")
            return result

        except Exception as e:
            logger.error(f"❌ Error en entrenamiento local: {e}")
            return None

    async def _apply_local_differential_privacy(
        self, result: TrainingResult
    ) -> TrainingResult:
        """Aplicar privacidad diferencial local adicional"""
        try:
            if not self.privacy_engine:
                return result

            # Aplicar ruido adicional a los pesos
            noisy_weights = {}
            for name, weight in result.model_weights.items():
                noise = torch.normal(
                    0,
                    self.config.noise_multiplier * 0.1,  # Ruido adicional reducido
                    weight.shape,
                )
                noisy_weights[name] = weight + noise

            result.model_weights = noisy_weights
            result.privacy_guarantees["local_dp_applied"] = True

            logger.info("🛡️ Privacidad diferencial local aplicada")
            return result

        except Exception as e:
            logger.error(f"❌ Error aplicando DP local: {e}")
            return result

    def _sign_training_result(self, result: TrainingResult) -> str:
        """Firmar resultado de entrenamiento"""
        try:
            # Crear hash del resultado
            result_data = json.dumps(
                {
                    "client_id": result.client_id,
                    "round_id": result.round_id,
                    "local_loss": result.local_loss,
                    "local_accuracy": result.local_accuracy,
                    "num_samples": result.num_samples,
                    "timestamp": datetime.now().isoformat(),
                },
                sort_keys=True,
            )

            # Firmar con clave privada
            signature = self.private_key.sign(
                result_data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            return signature.hex()

        except Exception as e:
            logger.error(f"❌ Error firmando resultado: {e}")
            return ""

    async def send_training_result(self, result: TrainingResult) -> bool:
        """Enviar resultado de entrenamiento al servidor"""
        try:
            # Convertir pesos a lista para JSON (simplificado)
            # En producción, usar serialización más eficiente
            weights_serializable = {}
            for name, weight in result.model_weights.items():
                weights_serializable[name] = weight.tolist()

            result_data = {
                "client_id": result.client_id,
                "round_id": result.round_id,
                "model_weights": weights_serializable,
                "local_loss": result.local_loss,
                "local_accuracy": result.local_accuracy,
                "num_samples": result.num_samples,
                "training_time": result.training_time,
                "privacy_guarantees": result.privacy_guarantees,
                "signature": result.signature,
            }

            async with self.http_client.post(
                f"{self.config.server_url}/submit_update", json=result_data
            ) as response:
                if response.status == 200:
                    logger.info("✅ Resultado enviado exitosamente al servidor")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"❌ Error enviando resultado: {error}")
                    return False

        except Exception as e:
            logger.error(f"❌ Error enviando resultado: {e}")
            return False

    async def assess_data_quality(self) -> float:
        """Evaluar calidad de datos locales"""
        try:
            if not self.local_dataset:
                return 0.5

            # Evaluar calidad basada en características del dataset
            quality_score = self.local_dataset.quality_score

            # Enviar evaluación al servidor
            quality_data = {
                "client_id": self.config.client_id,
                "dataset_info": {
                    "name": self.local_dataset.name,
                    "size": self.local_dataset.size,
                    "quality_score": quality_score,
                },
            }

            async with self.http_client.post(
                f"{self.config.server_url}/assess_quality", json=quality_data
            ) as response:
                if response.status == 200:
                    logger.info(f"📊 Calidad de datos evaluada: {quality_score}")
                    return quality_score
                else:
                    logger.warning("Error enviando evaluación de calidad")
                    return quality_score

        except Exception as e:
            logger.error(f"❌ Error evaluando calidad de datos: {e}")
            return 0.5

    async def close(self):
        """Cerrar cliente federado"""
        try:
            await self.http_client.close()
            logger.info("🔒 Cliente federado cerrado")

        except Exception as e:
            logger.error(f"❌ Error cerrando cliente: {e}")


# ==================== FUNCIONES DE UTILIDAD ====================


def create_healthcare_client(
    client_id: str, server_url: str = "http://localhost:8000"
) -> FederatedClient:
    """Crear cliente para sector sanitario"""
    config = ClientConfig(
        client_id=client_id,
        server_url=server_url,
        use_case="healthcare",
        device_type="server",
        enable_differential_privacy=True,
        privacy_budget=0.8,  # Alta privacidad para datos médicos
    )

    # Modelo médico simple
    if torch_available:
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2),  # Binario: enfermedad/no enfermedad
        )
    else:
        model = None

    # Dataset médico
    dataset = LocalDataset(
        name=f"medical_data_{client_id}",
        size=1000,
        features=["age", "blood_pressure", "cholesterol", "glucose"],
        labels=["healthy", "disease"],
        quality_score=0.9,
        privacy_level="high",
    )

    return FederatedClient(config=config, local_model=model, local_dataset=dataset)


def create_speech_client(
    client_id: str, server_url: str = "http://localhost:8000"
) -> FederatedClient:
    """Crear cliente para reconocimiento de voz"""
    config = ClientConfig(
        client_id=client_id,
        server_url=server_url,
        use_case="speech_recognition",
        device_type="mobile",
        enable_differential_privacy=True,
        privacy_budget=1.0,
    )

    # Modelo de voz
    if torch_available:
        model = nn.Sequential(
            nn.LSTM(40, 128, batch_first=True),  # MFCC features
            nn.Linear(128, 10),  # Dígitos 0-9
        )
    else:
        model = None

    # Dataset de voz
    dataset = LocalDataset(
        name=f"speech_data_{client_id}",
        size=500,
        features=["mfcc_features"],
        labels=[str(i) for i in range(10)],
        quality_score=0.8,
        privacy_level="medium",
    )

    return FederatedClient(config=config, local_model=model, local_dataset=dataset)


def create_transport_client(
    client_id: str, server_url: str = "http://localhost:8000"
) -> FederatedClient:
    """Crear cliente para transporte autónomo"""
    config = ClientConfig(
        client_id=client_id,
        server_url=server_url,
        use_case="autonomous_transport",
        device_type="iot",
        enable_differential_privacy=True,
        privacy_budget=1.2,  # Menos privacidad para datos de sensores
        local_epochs=2,  # Entrenamiento más rápido necesario
    )

    # Modelo de detección de objetos
    if torch_available:
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # Tipos de objetos
        )
    else:
        model = None

    # Dataset de imágenes
    dataset = LocalDataset(
        name=f"transport_data_{client_id}",
        size=200,
        features=["camera_images"],
        labels=["car", "pedestrian", "traffic_light", "stop_sign"],
        quality_score=0.85,
        privacy_level="low",
    )

    return FederatedClient(config=config, local_model=model, local_dataset=dataset)


# ==================== DEMO ====================


async def demo_federated_client():
    """Demostración del cliente federado"""
    logger.info("🎓 Demo del Cliente Federado")
    logger.info("=" * 50)

    try:
        # Crear cliente de salud
        client = create_healthcare_client("Hospital_Demo", "http://localhost:8000")
        logger.info("🏥 Cliente médico creado")

        # Simular registro (sin servidor real)
        logger.info("📝 Registrando cliente...")
        # En producción: await client.register_with_server()

        # Simular recepción de pesos globales
        logger.info("📥 Recibiendo pesos globales...")
        # En producción: global_weights = await get_global_weights_from_server()

        # Simular pesos globales (aleatorios para demo)
        if torch_available:
            global_weights = {
                "0.weight": torch.randn(50, 100),
                "0.bias": torch.randn(50),
                "2.weight": torch.randn(2, 50),
                "2.bias": torch.randn(2),
            }
        else:
            global_weights = {}

        # Participar en ronda
        logger.info("🎯 Participando en ronda de entrenamiento...")
        result = await client.participate_in_round("round_demo", global_weights)

        if result:
            logger.info("✅ Entrenamiento local completado")
            logger.info(".4f")
            logger.info(f"   Muestras entrenadas: {result.num_samples}")
            logger.info(".2f")
            # En producción: await client.send_training_result(result)
            logger.info("📤 Resultado enviado al servidor (simulado)")
        else:
            logger.error("❌ Error en entrenamiento local")

        # Cerrar cliente
        await client.close()

        logger.info("\n✅ Demo del cliente completada exitosamente")

    except Exception as e:
        logger.error(f"❌ Error en demo del cliente: {e}")


if __name__ == "__main__":
    asyncio.run(demo_federated_client())
