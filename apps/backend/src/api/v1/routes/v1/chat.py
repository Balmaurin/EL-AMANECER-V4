"""
API endpoints para chat - FastAPI router
"""

import logging

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)

from .schemas import (
    ChatRequest,
    ChatResponse,
    MessageValidationRequest,
    MessageValidationResponse,
)

logger = logging.getLogger(__name__)

# Crear router
router = APIRouter()


def get_chat_service():
    """Dependencia para obtener el servicio de chat"""
    from ...main import app

    return app.state.chat_service


def get_conversation_service():
    """Dependencia para obtener el servicio de conversaciones"""
    from ...main import app

    return app.state.conversation_service


@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    conversation_id: str = Query(..., description="ID de la conversación"),
    chat_service=Depends(get_chat_service),
    conv_service=Depends(get_conversation_service),
):
    """
    Enviar un mensaje de chat y obtener respuesta

    - **conversation_id**: ID único de la conversación
    - **message**: Mensaje del usuario
    - **system_prompt**: Prompt del sistema (opcional)
    - **max_tokens**: Máximo tokens en respuesta (1-2048)
    - **temperature**: Temperatura de generación (0.0-2.0)
    - **top_p**: Top-p sampling (0.0-1.0)
    - **rag_top_k**: Documentos RAG a recuperar (1-20)
    - **rag_threshold**: Umbral similitud RAG (0.0-1.0)
    """
    try:
        # Validar mensaje
        validation = await chat_service.validate_message(request.message)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["error"])

        # Procesar mensaje
        response_data = await chat_service.process_message(
            conversation_id=conversation_id,
            user_message=request.message,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            rag_top_k=request.rag_top_k,
            rag_threshold=request.rag_threshold,
        )

        # Guardar mensajes en la conversación
        await conv_service.add_message(
            conversation_id=conversation_id,
            role="user",
            content=request.message,
        )

        await conv_service.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_data["response"],
            context_used=response_data["context_used"],
            context_length=response_data["context_length"],
        )

        return ChatResponse(**response_data)

    except Exception as e:
        logger.error(f"Error procesando mensaje: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.websocket("/ws/{conversation_id}")
async def chat_websocket(
    websocket: WebSocket,
    conversation_id: str,
    chat_service=Depends(get_chat_service),
    conv_service=Depends(get_conversation_service),
):
    """
    WebSocket para chat en tiempo real

    - **conversation_id**: ID de la conversación
    """
    await websocket.accept()
    logger.info(f"Cliente WebSocket conectado: {conversation_id}")

    try:
        while True:
            # Recibir mensaje del cliente
            data = await websocket.receive_json()
            user_message = data.get("message", "").strip()

            if not user_message:
                continue

            logger.info(
                f"Mensaje WebSocket recibido de {conversation_id}: {user_message[:50]}..."  # noqa: E501
            )

            # Validar mensaje
            validation = await chat_service.validate_message(user_message)
            if not validation["valid"]:
                await websocket.send_json(
                    {
                        "error": validation["error"],
                        "conversation_id": conversation_id,
                    }
                )
                continue

            # Procesar mensaje
            response_data = await chat_service.process_message(
                conversation_id=conversation_id,
                user_message=user_message,
                **data.get("params", {}),  # Parámetros adicionales
            )

            # Guardar mensajes
            await conv_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=user_message,
            )

            await conv_service.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=response_data["response"],
                context_used=response_data["context_used"],
                context_length=response_data["context_length"],
            )

            # Enviar respuesta
            await websocket.send_json(response_data)

    except WebSocketDisconnect:
        logger.info(f"Cliente WebSocket desconectado: {conversation_id}")
    except Exception as e:
        logger.error(f"Error en WebSocket {conversation_id}: {e}")
        try:
            await websocket.send_json(
                {
                    "error": "Error interno del servidor",
                    "conversation_id": conversation_id,
                }
            )
        except Exception:
            pass


@router.get("/info")
async def get_chat_info(chat_service=Depends(get_chat_service)):
    """
    Obtener información del servicio de chat
    """
    try:
        info = await chat_service.get_chat_info()
        return info
    except Exception as e:
        logger.error(f"Error obteniendo info de chat: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo información")


@router.post("/validate", response_model=MessageValidationResponse)
async def validate_message(
    request: MessageValidationRequest, chat_service=Depends(get_chat_service)
):
    """
    Validar un mensaje antes del procesamiento

    - **message**: Mensaje a validar
    """
    try:
        result = await chat_service.validate_message(request.message)
        return MessageValidationResponse(**result)
    except Exception as e:
        logger.error(f"Error validando mensaje: {e}")
        raise HTTPException(status_code=500, detail="Error de validación")
