"""
API REST para Sistema de Aprendizaje Federado

Esta API proporciona endpoints REST para la gestión remota del sistema FL,
permitiendo a clientes federados registrarse, enviar actualizaciones y
recibir configuraciones de forma segura.

Endpoints principales:
- POST /register - Registro de clientes
- POST /submit_update - Envío de actualizaciones de modelo
- GET /round_status - Estado de rondas activas
- POST /heartbeat - Latidos del corazón de clientes
- GET /metrics - Métricas del sistema FL

Autor: Sheily AI Team
Fecha: 2025
"""

import asyncio
import hashlib
import json
import logging
import secrets
from datetime import datetime
from typing import Any, Dict, Optional

# FastAPI y dependencias
try:
    import uvicorn
    from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field, validator

    FASTAPI_AVAILABLE = True
except ImportError:
    # Fallback si FastAPI no está disponible
    FastAPI = None
    HTTPException = Exception
    BackgroundTasks = None
    Depends = None
    Security = None
    HTTPBearer = None
    HTTPAuthorizationCredentials = None
    CORSMiddleware = None
    BaseModel = object
    Field = lambda **kwargs: None
    validator = lambda f: f
    uvicorn = None
    FASTAPI_AVAILABLE = False

from federated_client import ClientConfig

# Importaciones del sistema FL
from federated_learning import FederatedConfig, FederatedLearningSystem, UseCase

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== MODELOS Pydantic ====================


class ClientRegistrationRequest(BaseModel):
    """Solicitud de registro de cliente"""

    client_id: str = Field(..., description="ID único del cliente")
    public_key: str = Field(..., description="Clave pública RSA en formato PEM")
    use_case: str = Field(
        ..., description="Caso de uso (healthcare, speech_recognition, etc.)"
    )
    device_type: str = Field(
        "server", description="Tipo de dispositivo (server, mobile, iot)"
    )
    location: Optional[str] = Field(None, description="Ubicación geográfica")
    gdpr_consent: bool = Field(True, description="Consentimiento RGPD")

    @validator("use_case")
    def validate_use_case(cls, v):
        valid_cases = [case.value for case in UseCase]
        if v not in valid_cases:
            raise ValueError(f"use_case debe ser uno de: {valid_cases}")
        return v


class ClientRegistrationResponse(BaseModel):
    """Respuesta de registro de cliente"""

    success: bool
    client_id: str
    server_public_key: str
    message: str
    registered_at: datetime


class ModelUpdateRequest(BaseModel):
    """Solicitud de actualización de modelo"""

    client_id: str
    round_id: str
    model_weights: Dict[str, Any]  # Pesos serializados
    local_loss: float
    local_accuracy: float
    num_samples: int
    training_time: float
    privacy_guarantees: Dict[str, Any] = Field(default_factory=dict)
    signature: str


class ModelUpdateResponse(BaseModel):
    """Respuesta de actualización de modelo"""

    success: bool
    client_id: str
    round_id: str
    message: str
    processed_at: datetime
    reputation_score: Optional[float] = None


class HeartbeatRequest(BaseModel):
    """Solicitud de latido del corazón"""

    client_id: str
    timestamp: datetime
    status: str = "active"


class HeartbeatResponse(BaseModel):
    """Respuesta de latido del corazón"""

    success: bool
    client_id: str
    server_time: datetime
    active_rounds: int


class RoundStatusResponse(BaseModel):
    """Respuesta de estado de ronda"""

    round_id: str
    round_number: int
    status: str
    participating_clients: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None


class MetricsResponse(BaseModel):
    """Respuesta de métricas del sistema"""

    timestamp: datetime
    active_clients: int
    total_clients: int
    active_rounds: int
    privacy_metrics: Dict[str, Any]
    security_events: int
    gdpr_compliance_rate: float


class StartRoundRequest(BaseModel):
    """Solicitud para iniciar nueva ronda"""

    round_number: Optional[int] = None
    force_start: bool = False


class StartRoundResponse(BaseModel):
    """Respuesta de inicio de ronda"""

    success: bool
    round_id: str
    round_number: int
    participating_clients: int
    message: str


# ==================== API PRINCIPAL ====================


class FederatedLearningAPI:
    """API REST para el sistema de aprendizaje federado"""

    def __init__(self, fl_system: Optional[FederatedLearningSystem] = None):
        """Inicializar API FL"""
        self.fl_system = fl_system or FederatedLearningSystem()
        self.app = None
        self.security = None

        if FASTAPI_AVAILABLE:
            self._create_app()
        else:
            logger.warning("FastAPI no disponible - API REST no inicializada")

    def _create_app(self):
        """Crear aplicación FastAPI"""
        self.app = FastAPI(
            title="Federated Learning API",
            description="API REST para gestión del sistema de aprendizaje federado",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Configurar CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # En producción, especificar orígenes permitidos
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Seguridad
        self.security = HTTPBearer()

        # Registrar rutas
        self._register_routes()

        logger.info("🚀 API REST de Aprendizaje Federado inicializada")

    def _register_routes(self):
        """Registrar todas las rutas de la API"""

        @self.app.post("/register", response_model=ClientRegistrationResponse)
        async def register_client(request: ClientRegistrationRequest):
            """Registrar un nuevo cliente federado"""
            try:
                # Convertir string de caso de uso a enum
                use_case_enum = UseCase(request.use_case)

                # Registrar cliente
                success = await self.fl_system.register_client(
                    client_id=request.client_id,
                    public_key=request.public_key,
                    use_case=use_case_enum,
                    device_type=request.device_type,
                    location=request.location,
                )

                if success:
                    return ClientRegistrationResponse(
                        success=True,
                        client_id=request.client_id,
                        server_public_key=self.fl_system.public_key_pem.decode(),
                        message="Cliente registrado exitosamente",
                        registered_at=datetime.now(),
                    )
                else:
                    raise HTTPException(
                        status_code=400, detail="Error registrando cliente"
                    )

            except Exception as e:
                logger.error(f"Error en registro de cliente: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/submit_update", response_model=ModelUpdateResponse)
        async def submit_model_update(
            request: ModelUpdateRequest,
            credentials: HTTPAuthorizationCredentials = Security(self.security),
        ):
            """Recibir actualización de modelo de un cliente"""
            try:
                # Verificar token (simplificado)
                if not self._verify_token(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Token inválido")

                # Convertir pesos de lista a tensores
                model_weights = self._deserialize_weights(request.model_weights)

                # Crear objeto ModelUpdate
                from federated_learning import ModelUpdate

                update = ModelUpdate(
                    client_id=request.client_id,
                    round_id=request.round_id,
                    model_weights=model_weights,
                    local_loss=request.local_loss,
                    local_accuracy=request.local_accuracy,
                    num_samples=request.num_samples,
                    training_time=request.training_time,
                    privacy_guarantees=request.privacy_guarantees,
                    signature=request.signature,
                )

                # Procesar actualización
                success = await self.fl_system.receive_client_update(update)

                if success:
                    # Obtener reputación actualizada
                    reputation = None
                    if request.client_id in self.fl_system.clients:
                        reputation = self.fl_system.clients[
                            request.client_id
                        ].reputation_score

                    return ModelUpdateResponse(
                        success=True,
                        client_id=request.client_id,
                        round_id=request.round_id,
                        message="Actualización procesada exitosamente",
                        processed_at=datetime.now(),
                        reputation_score=reputation,
                    )
                else:
                    raise HTTPException(
                        status_code=400, detail="Actualización rechazada"
                    )

            except Exception as e:
                logger.error(f"Error procesando actualización: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/heartbeat", response_model=HeartbeatResponse)
        async def client_heartbeat(request: HeartbeatRequest):
            """Recibir latido del corazón de cliente"""
            try:
                # En implementación real, actualizar estado del cliente
                active_rounds = len(self.fl_system.active_rounds)

                return HeartbeatResponse(
                    success=True,
                    client_id=request.client_id,
                    server_time=datetime.now(),
                    active_rounds=active_rounds,
                )

            except Exception as e:
                logger.error(f"Error procesando heartbeat: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/round_status", response_model=RoundStatusResponse)
        async def get_round_status(round_id: str):
            """Obtener estado de una ronda específica"""
            try:
                if round_id not in self.fl_system.active_rounds:
                    raise HTTPException(status_code=404, detail="Ronda no encontrada")

                round_obj = self.fl_system.active_rounds[round_id]

                return RoundStatusResponse(
                    round_id=round_obj.round_id,
                    round_number=round_obj.round_number,
                    status=round_obj.status,
                    participating_clients=len(round_obj.participating_clients),
                    start_time=round_obj.start_time,
                    estimated_completion=None,  # Calcular si es necesario
                )

            except Exception as e:
                logger.error(f"Error obteniendo estado de ronda: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/metrics", response_model=MetricsResponse)
        async def get_system_metrics():
            """Obtener métricas del sistema FL"""
            try:
                metrics = self.fl_system.get_federated_metrics()

                return MetricsResponse(
                    timestamp=datetime.now(),
                    active_clients=metrics.get("active_clients", 0),
                    total_clients=metrics.get("total_clients", 0),
                    active_rounds=metrics.get("active_rounds", 0),
                    privacy_metrics=metrics.get("privacy_metrics", {}),
                    security_events=len(metrics.get("security_events", [])),
                    gdpr_compliance_rate=metrics.get("gdpr_compliance_rate", 0.0),
                )

            except Exception as e:
                logger.error(f"Error obteniendo métricas: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/start_round", response_model=StartRoundResponse)
        async def start_new_round(request: StartRoundRequest):
            """Iniciar una nueva ronda de entrenamiento (solo admin)"""
            try:
                # En producción, verificar permisos de administrador
                round_number = (
                    request.round_number or len(self.fl_system.active_rounds) + 1
                )

                round_id = await self.fl_system.start_federated_round(round_number)

                participating_clients = 0
                if round_id in self.fl_system.active_rounds:
                    participating_clients = len(
                        self.fl_system.active_rounds[round_id].participating_clients
                    )

                return StartRoundResponse(
                    success=True,
                    round_id=round_id,
                    round_number=round_number,
                    participating_clients=participating_clients,
                    message=f"Ronda {round_number} iniciada exitosamente",
                )

            except Exception as e:
                logger.error(f"Error iniciando ronda: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health_check():
            """Endpoint de verificación de salud"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system": "Federated Learning API",
                "version": "1.0.0",
            }

    def _verify_token(self, token: str) -> bool:
        """Verificar token de autenticación (simplificado)"""
        # En producción, implementar verificación JWT o similar
        return len(token) > 10  # Verificación básica

    def _deserialize_weights(self, weights_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deserializar pesos de modelo de formato JSON"""
        try:
            import torch

            deserialized = {}
            for name, weights_list in weights_dict.items():
                # Convertir listas a tensores
                if isinstance(weights_list, list):
                    tensor = torch.tensor(weights_list)
                    deserialized[name] = tensor
                else:
                    deserialized[name] = weights_list

            return deserialized

        except Exception as e:
            logger.error(f"Error deserializando pesos: {e}")
            return weights_dict

    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Iniciar servidor API"""
        if not FASTAPI_AVAILABLE or not uvicorn:
            logger.error("FastAPI/uvicorn no disponibles")
            return

        logger.info(f"🚀 Iniciando servidor API en {host}:{port}")
        config = uvicorn.Config(self.app, host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()

    def get_app(self):
        """Obtener instancia de la aplicación FastAPI"""
        return self.app


# ==================== CLIENTE API ====================


class FederatedAPIClient:
    """Cliente para interactuar con la API REST del sistema FL"""

    def __init__(self, server_url: str = "http://localhost:8000"):
        """Inicializar cliente API"""
        self.server_url = server_url.rstrip("/")
        self.session = None
        self.auth_token = None

        if FASTAPI_AVAILABLE:
            import aiohttp

            self.session = aiohttp.ClientSession()
        else:
            logger.warning("aiohttp no disponible - cliente API limitado")

    async def register_client(self, client_config: ClientConfig) -> bool:
        """Registrar cliente con el servidor"""
        try:
            registration_data = {
                "client_id": client_config.client_id,
                "public_key": "simulated_public_key",  # En producción usar clave real
                "use_case": client_config.use_case,
                "device_type": client_config.device_type,
                "location": client_config.location,
                "gdpr_consent": client_config.gdpr_consent,
            }

            async with self.session.post(
                f"{self.server_url}/register", json=registration_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Cliente registrado: {result['message']}")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"❌ Error registrando cliente: {error}")
                    return False

        except Exception as e:
            logger.error(f"❌ Error en registro: {e}")
            return False

    async def submit_update(self, update: "ModelUpdate") -> bool:
        """Enviar actualización de modelo"""
        try:
            # Serializar pesos
            weights_serializable = {}
            for name, weight in update.model_weights.items():
                try:
                    weights_serializable[name] = weight.tolist()
                except:
                    weights_serializable[name] = str(weight)

            update_data = {
                "client_id": update.client_id,
                "round_id": update.round_id,
                "model_weights": weights_serializable,
                "local_loss": update.local_loss,
                "local_accuracy": update.local_accuracy,
                "num_samples": update.num_samples,
                "training_time": getattr(update, "training_time", 0.0),
                "privacy_guarantees": update.privacy_guarantees,
                "signature": update.signature or "",
            }

            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            async with self.session.post(
                f"{self.server_url}/submit_update", json=update_data, headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Actualización enviada: {result['message']}")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"❌ Error enviando actualización: {error}")
                    return False

        except Exception as e:
            logger.error(f"❌ Error enviando actualización: {e}")
            return False

    async def send_heartbeat(self, client_id: str) -> bool:
        """Enviar latido del corazón"""
        try:
            heartbeat_data = {
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "status": "active",
            }

            async with self.session.post(
                f"{self.server_url}/heartbeat", json=heartbeat_data
            ) as response:
                return response.status == 200

        except Exception as e:
            logger.debug(f"Error en heartbeat: {e}")
            return False

    async def get_round_status(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de ronda"""
        try:
            async with self.session.get(
                f"{self.server_url}/round_status", params={"round_id": round_id}
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None

        except Exception as e:
            logger.error(f"Error obteniendo estado de ronda: {e}")
            return None

    async def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Obtener métricas del sistema"""
        try:
            async with self.session.get(f"{self.server_url}/metrics") as response:
                if response.status == 200:
                    return await response.json()
                return None

        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}")
            return None

    async def close(self):
        """Cerrar cliente API"""
        if self.session:
            await self.session.close()


# ==================== FUNCIONES DE UTILIDAD ====================


def create_fl_api_server(
    config: Optional[FederatedConfig] = None,
) -> FederatedLearningAPI:
    """Crear servidor API para sistema FL"""
    fl_system = FederatedLearningSystem(config=config)
    return FederatedLearningAPI(fl_system)


def create_fl_api_client(
    server_url: str = "http://localhost:8000",
) -> FederatedAPIClient:
    """Crear cliente para API FL"""
    return FederatedAPIClient(server_url)


# ==================== DEMO API ====================


async def demo_federated_api():
    """Demostración de la API REST FL"""
    logger.info("🚀 Demo de API REST de Aprendizaje Federado")
    logger.info("=" * 60)

    try:
        # Crear servidor API
        logger.info("📡 Creando servidor API...")
        api_server = create_fl_api_server()

        if not api_server.app:
            logger.error("Servidor API no disponible")
            return

        # En un entorno real, iniciar servidor en background
        # asyncio.create_task(api_server.start_server())

        # Crear cliente API
        logger.info("📱 Creando cliente API...")
        api_client = create_fl_api_client()

        # Simular registro de cliente
        logger.info("📝 Registrando cliente vía API...")
        from federated_client import ClientConfig

        client_config = ClientConfig(
            client_id="api_test_client",
            server_url="http://localhost:8000",
            use_case="healthcare",
        )

        # En producción: success = await api_client.register_client(client_config)

        # Simular envío de actualización
        logger.info("📤 Enviando actualización vía API...")
        # Crear actualización simulada
        import torch
        from federated_learning import ModelUpdate

        mock_weights = {
            "layer1.weight": torch.randn(50, 100),
            "layer1.bias": torch.randn(50),
            "layer2.weight": torch.randn(2, 50),
            "layer2.bias": torch.randn(2),
        }

        mock_update = ModelUpdate(
            client_id="api_test_client",
            round_id="test_round",
            model_weights=mock_weights,
            local_loss=0.25,
            local_accuracy=0.85,
            num_samples=500,
            privacy_guarantees={"differential_privacy": True},
        )

        # En producción: success = await api_client.submit_update(mock_update)

        # Obtener métricas
        logger.info("📊 Obteniendo métricas vía API...")
        # En producción: metrics = await api_client.get_metrics()

        # Cerrar cliente
        await api_client.close()

        logger.info("✅ Demo de API completada exitosamente")

    except Exception as e:
        logger.error(f"❌ Error en demo de API: {e}")


if __name__ == "__main__":
    asyncio.run(demo_federated_api())
