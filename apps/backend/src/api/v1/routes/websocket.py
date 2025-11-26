"""
WebSocket API - Comunicación en tiempo real para chat
Extraído de backend/chat_server.py
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# Importar servicios necesarios (compatibilidad)
try:
    from apps.backend.src.core.llm import LLMFactory
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Configuración
CHROMA_DB_PATH = "./data/chroma_db"

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    message: str
    context: Optional[List[Dict]] = None

class WebSocketConnectionManager:
    """Gestor de conexiones WebSocket"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, conversation_id: str):
        await websocket.accept()
        self.active_connections[conversation_id] = websocket
        logger.info(f"Cliente WebSocket conectado: {conversation_id}")

    def disconnect(self, conversation_id: str):
        if conversation_id in self.active_connections:
            del self.active_connections[conversation_id]
            logger.info(f"Cliente WebSocket desconectado: {conversation_id}")

    async def send_message(self, conversation_id: str, message: dict):
        if conversation_id in self.active_connections:
            try:
                await self.active_connections[conversation_id].send_json(message)
            except Exception as e:
                logger.error(f"Error enviando mensaje a {conversation_id}: {e}")
                # Remover conexión si falla
                self.disconnect(conversation_id)

class KnowledgeBase:
    """Base de conocimientos para RAG"""

    def __init__(self):
        self.collection = None
        self._initialize_collection()

    def _initialize_collection(self):
        """Inicializar colección ChromaDB"""
        if not CHROMA_AVAILABLE:
            logger.warning("ChromaDB no disponible - knowledge base desactivada")
            return

        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.collection = chroma_client.get_or_create_collection(
                name="knowledge_base",
                metadata={"description": "Base de conocimientos para RAG"},
            )
            logger.info("✅ Knowledge base ChromaDB inicializada")
        except Exception as e:
            logger.error(f"Error inicializando ChromaDB: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generar embedding (versión simplificada)"""
        # Hash determinista del texto
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)

        # Generar vector pseudo-aleatorio determinista
        import numpy as np
        np.random.seed(hash_int % 2**32)
        embedding = np.random.normal(0, 1, 384).tolist()

        # Normalizar
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()

        return embedding

    async def add_knowledge(self, text: str, metadata: Dict = None) -> Dict:
        """Añadir conocimiento a la base"""
        try:
            if not self.collection:
                return {"success": False, "error": "Knowledge base no disponible"}

            if not text or not text.strip():
                return {"success": False, "error": "Texto requerido"}

            # Generar embedding
            embedding = self.generate_embedding(text)

            # Añadir a ChromaDB
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_id = f"doc_{timestamp}_{hash(text) % 10000}"

            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}],
                ids=[doc_id],
            )

            logger.info(f"Conocimiento añadido: {doc_id}")
            return {
                "success": True,
                "document_id": doc_id,
                "chunks_added": 1,
            }

        except Exception as e:
            logger.error(f"Error añadiendo conocimiento: {e}")
            return {"success": False, "error": str(e)}

    async def search_knowledge(self, query: str, limit: int = 5) -> Dict:
        """Buscar en la base de conocimientos"""
        try:
            if not self.collection or self.collection.count() == 0:
                return {"results": [], "query": query}

            # Generar embedding de consulta
            query_embedding = self.generate_embedding(query)

            # Buscar
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            # Formatear resultados
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    formatted_results.append({
                        "document": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": distance,
                        "score": 1 - distance,  # Score basado en similitud
                    })

            return {
                "results": formatted_results[:limit],
                "query": query,
                "total_found": len(formatted_results),
            }

        except Exception as e:
            logger.error(f"Error buscando conocimiento: {e}")
            return {"results": [], "query": query, "error": str(e)}

# Inicializar servicios
connection_manager = WebSocketConnectionManager()
knowledge_base = KnowledgeBase()

router = APIRouter()

@router.websocket("/chat/{conversation_id}")
async def chat_websocket(websocket: WebSocket, conversation_id: str):
    """WebSocket para chat en tiempo real"""
    await connection_manager.connect(websocket, conversation_id)

    try:
        while True:
            # Recibir mensaje del cliente
            data = await websocket.receive_json()
            user_message = data.get("message", "").strip()

            if not user_message:
                continue

            logger.info(f"Mensaje recibido de {conversation_id}: {user_message[:50]}...")

            # Procesar mensaje
            response_data = await process_chat_message(conversation_id, user_message)

            # Enviar respuesta
            await connection_manager.send_message(conversation_id, response_data)

    except WebSocketDisconnect:
        connection_manager.disconnect(conversation_id)
    except Exception as e:
        logger.error(f"Error en WebSocket {conversation_id}: {e}")
        connection_manager.disconnect(conversation_id)

async def process_chat_message(conversation_id: str, user_message: str) -> Dict:
    """Procesar mensaje de chat y generar respuesta"""
    try:
        # Recuperar contexto relevante (si está disponible)
        context = ""
        if knowledge_base.collection and knowledge_base.collection.count() > 0:
            search_results = await knowledge_base.search_knowledge(user_message, limit=3)
            if search_results.get("results"):
                context = "\n".join([r["document"] for r in search_results["results"]])
                logger.info(f"Contexto RAG recuperado: {len(context)} caracteres")

        # Generar respuesta usando sistema disponible
        response_text = await generate_chat_response(user_message, context)

        # Crear datos de respuesta
        timestamp = asyncio.get_event_loop().time()
        response_data = {
            "response": response_text,
            "conversation_id": conversation_id,
            "timestamp": timestamp,
            "context_used": len(context.strip()) > 0,
        }

        # Guardar conversación si es necesario
        # (Aquí podría integrarse con conversation service si existe)

        return response_data

    except Exception as e:
        logger.error(f"Error procesando mensaje de chat: {e}")
        return {
            "error": "Error interno del servidor",
            "conversation_id": conversation_id,
            "timestamp": asyncio.get_event_loop().time(),
        }

async def generate_chat_response(message: str, context: str = "") -> str:
    """Generar respuesta de chat usando sistemas disponibles"""
    try:
        # Intentar usar sistema polymorphic LLM primero si está disponible
        if LLM_AVAILABLE:
            try:
                logger.info("Intentando usar sistema polymorphic LLM...")
                llm = LLMFactory.create_llm({'provider': 'local'})  # Usa Local LLM por defecto
                ready = await llm.is_ready()
                if ready:
                    response = await llm.generate_response(message, context=context)
                    logger.info("✅ Respuesta exitosa del sistema polymorphic LLM")
                    return response
                else:
                    logger.warning("LLM no está ready")

            except Exception as e:
                logger.warning(f"Error con sistema polymorphic LLM: {e}")

        # Fallback: Sistema simplificado
        fallback_responses = [
            f"Entiendo tu mensaje: '{message[:50]}...'. Esta es una respuesta de fallback.",
            f"Gracias por tu mensaje. Como sistema en desarrollo, estoy procesando: {message[:30]}...",
            f"He recibido tu mensaje. Mi respuesta actual es limitada, pero puedo confirmar que entendí: {message[:40]}...",
        ]

        import random
        response = random.choice(fallback_responses)

        # Agregar contexto si está disponible
        if context:
            response += f"\n\nContexto relevante: {context[:100]}..."

        return response

    except Exception as e:
        logger.error(f"Error generando respuesta de chat: {e}")
        return "Lo siento, tuve un problema procesando tu mensaje. Por favor intenta de nuevo."

# Endpoint adicional para compatibilidad con API REST
@router.post("/chat/message")
async def send_chat_message(chat_msg: ChatMessage):
    """Endpoint REST para enviar mensaje de chat (compatibilidad)"""
    try:
        response_data = await process_chat_message("rest_api", chat_msg.message)
        return {
            "success": True,
            "response": response_data.get("response", "Error en respuesta"),
            "conversation_id": response_data.get("conversation_id"),
            "timestamp": response_data.get("timestamp"),
            "context_used": response_data.get("context_used", False),
        }
    except Exception as e:
        logger.error(f"Error en API REST de chat: {e}")
        raise HTTPException(status_code=500, detail="Error procesando mensaje")

@router.get("/websocket/connections")
async def get_websocket_connections():
    """Obtener información de conexiones WebSocket activas"""
    return {
        "active_connections": len(connection_manager.active_connections),
        "connections": list(connection_manager.active_connections.keys()),
    }

@router.get("/knowledge/stats")
async def get_knowledge_stats():
    """Obtener estadísticas de la base de conocimientos"""
    try:
        if knowledge_base.collection:
            count = knowledge_base.collection.count()
        else:
            count = 0

        return {
            "knowledge_base_available": knowledge_base.collection is not None,
            "documents_count": count,
            "chroma_available": CHROMA_AVAILABLE,
            "llm_available": LLM_AVAILABLE,
        }
    except Exception as e:
        logger.error(f"Error obteniendo stats de knowledge: {e}")
        return {"error": str(e)}
