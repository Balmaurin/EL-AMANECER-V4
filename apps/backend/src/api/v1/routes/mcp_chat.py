"""
MCP Enterprise Chat API - Chat Orquestado por MCP Master
=========================================================

Endpoint que usa el MCP Chat Orchestrator para coordinar
todos los sistemas del proyecto.
"""

import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from sheily_core.chat.mcp_chat_orchestrator import get_mcp_chat_orchestrator
from sheily_core.logger import get_logger

logger = get_logger("mcp_chat_api")

router = APIRouter(prefix="/mcp/chat", tags=["mcp-chat"])


class MCPChatRequest(BaseModel):
    """Solicitud de chat MCP"""
    message: str
    user_id: Optional[str] = "default"
    conversation_id: Optional[str] = None


class MCPChatResponseModel(BaseModel):
    """Respuesta de chat MCP"""
    query: str
    response: str
    agent_used: str
    confidence: float
    processing_time: float
    memories_accessed: int
    insights_used: list
    timestamp: str


@router.post("/message", response_model=MCPChatResponseModel)
async def send_mcp_chat_message(request: MCPChatRequest):
    """
    Enviar mensaje al sistema MCP Enterprise Chat
    
    Este endpoint orquesta automáticamente:
    - Finance Agent para análisis financiero
    - Quantitative Agent para trading/estrategias
    - Memory System para contexto histórico
    - Learning System para mejorar respuestas
    
    El MCP Master decide qué componente usar basado en la consulta.
    """
    try:
        logger.info(f"Procesando mensaje MCP de {request.user_id}")
        
        # Obtener orquestador
        orchestrator = get_mcp_chat_orchestrator()
        
        # Procesar consulta
        response = await orchestrator.process_query(
            query=request.message,
            user_id=request.user_id
        )
        
        # Convertir a modelo de respuesta
        return MCPChatResponseModel(
            query=response.query,
            response=response.response,
            agent_used=response.agent_used,
            confidence=response.confidence,
            processing_time=response.processing_time,
            memories_accessed=response.memories_accessed,
            insights_used=response.insights_used,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error procesando chat MCP: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en el sistema MCP: {str(e)}"
        )


@router.get("/status")
async def get_mcp_chat_status():
    """Obtener estado del sistema MCP Chat"""
    try:
        orchestrator = get_mcp_chat_orchestrator()
        
        # Asegurar que MCP Master esté inicializado
        await orchestrator._ensure_mcp_master()
        
        status = {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "mcp_master": orchestrator.mcp_master is not None,
                "finance_agent": bool(orchestrator.mcp_master and orchestrator.mcp_master.finance_agent),
                "quant_agent": bool(orchestrator.mcp_master and orchestrator.mcp_master.quant_agent),
                "memory_system": bool(orchestrator.mcp_master and hasattr(orchestrator.mcp_master, 'memory_core')),
            }
        }
        
        if orchestrator.mcp_master and hasattr(orchestrator.mcp_master, 'memory_core'):
            memory_count = len(orchestrator.mcp_master.memory_core.unified_memory.memories)
            status["memory_count"] = memory_count
        
        return status
    
    except Exception as e:
        logger.error(f"Error obteniendo estado MCP: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
