"""
Chat & AI API Routes
====================

Core AI orchestration endpoints for Sheily MCP.
Provides chat functionality, AI model interactions, and agent coordination.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel

from apps.backend.src.config.settings import settings
from apps.backend.src.core.database import get_db
from apps.backend.src.core.agent_orchestrator import AgentOrchestrator
from apps.backend.src.services.ai_service import AIService

router = APIRouter()

# Initialize AI services
ai_service = AIService()
agent_orchestrator = AgentOrchestrator()


# =================================
# REQUEST/RESPONSE MODELS
# =================================


class ChatMessage(BaseModel):
    """Chat message structure"""

    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Chat request structure"""

    message: str
    conversation_id: Optional[str] = None
    model: Optional[str] = "default-model"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    agent_id: Optional[str] = None
    use_rag: Optional[bool] = True
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response structure"""

    response: str
    conversation_id: str
    model_used: str
    tokens_used: int
    processing_time: float
    agent_used: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationInfo(BaseModel):
    """Conversation information"""

    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    agents_used: List[str]
    total_tokens: int
    tags: List[str]


# =================================
# CHAT ENDPOINTS
# =================================


@router.post("", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main chat endpoint for AI conversation.

    Supports streaming responses, RAG integration, and agent orchestration.

    - **message**: User's message text
    - **conversation_id**: Optional existing conversation ID
    - **model**: AI model to use (default: default-model)
    - **temperature**: Response creativity (0.0-2.0)
    - **max_tokens**: Maximum response length
    - **stream**: Enable streaming responses
    - **agent_id**: Specific agent to use
    - **use_rag**: Enable RAG knowledge retrieval
    - **context**: Additional context for the conversation
    """
    try:
        import time

        start_time = time.time()

        # Validate request parameters
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty",
            )

        # Generate conversation ID if not provided
        if not request.conversation_id:
            import uuid

            request.conversation_id = str(uuid.uuid4())[:8]

        # Process message through AI service
        response_data = await ai_service.process_message(
            message=request.message,
            conversation_id=request.conversation_id,
            model=request.model or "gemma2",
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 4096,
            agent_id=request.agent_id,
            use_rag=request.use_rag if request.use_rag is not None else True,
            context=request.context,
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Background task to update conversation metadata
        background_tasks.add_task(
            ai_service.update_conversation_metadata,
            conversation_id=request.conversation_id,
            processing_time=processing_time,
            tokens_used=response_data.get("tokens_used", 0),
        )

        return ChatResponse(
            response=response_data["response"],
            conversation_id=request.conversation_id,
            model_used=response_data.get("model_used", request.model or "gemma2"),
            tokens_used=response_data.get("tokens_used", 0),
            processing_time=round(processing_time, 3),
            agent_used=response_data.get("agent_used"),
            sources=response_data.get("sources"),
            metadata=response_data.get("metadata"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}",
        )


@router.get("/conversations", response_model=List[ConversationInfo])
async def list_conversations(
    skip: int = 0, limit: int = 50, agent_filter: Optional[str] = None
):
    """
    List user conversations with filtering options.

    - **skip**: Number of conversations to skip (pagination)
    - **limit**: Maximum conversations to return (1-100)
    - **agent_filter**: Filter by specific agent
    """
    try:
        if limit > 100:
            limit = 100
        elif limit < 1:
            limit = 1

        conversations = await ai_service.list_conversations(
            skip=skip, limit=limit, agent_filter=agent_filter
        )

        return conversations

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversations: {str(e)}",
        )


@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get detailed conversation history.

    - **conversation_id**: Unique conversation identifier
    """
    try:
        conversation = await ai_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

        return conversation

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation: {str(e)}",
        )


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its messages.

    - **conversation_id**: Unique conversation identifier
    """
    try:
        success = await ai_service.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or could not be deleted",
            )

        return {
            "message": "Conversation deleted successfully",
            "conversation_id": conversation_id,
            "timestamp": "2025-01-16T08:53:13Z",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}",
        )


@router.post("/conversation/{conversation_id}/title")
async def update_conversation_title(conversation_id: str, title: str):
    """
    Update conversation title.

    - **conversation_id**: Unique conversation identifier
    - **title**: New conversation title
    """
    try:
        if not title or len(title.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Title cannot be empty"
            )

        success = await ai_service.update_conversation_title(
            conversation_id=conversation_id, title=title.strip()
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

        return {
            "message": "Conversation title updated",
            "conversation_id": conversation_id,
            "title": title.strip(),
            "timestamp": "2025-01-16T08:53:13Z",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update conversation title: {str(e)}",
        )


# =================================
# AI MODEL ENDPOINTS
# =================================


@router.get("/models")
async def list_available_models():
    """List all available AI models with their capabilities."""
    try:
        models = await ai_service.list_available_models()
        return {
            "models": models,
            "total": len(models),
            "timestamp": "2025-01-16T08:53:13Z",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models: {str(e)}",
        )


@router.get("/model/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific AI model."""
    try:
        model_info = await ai_service.get_model_info(model_name)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found",
            )

        return model_info

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}",
        )


# =================================
# AGENT ORCHESTRATION ENDPOINTS
# =================================


@router.post("/agents/{agent_id}/execute")
async def execute_agent(
    agent_id: str, request: ChatRequest, background_tasks: BackgroundTasks
):
    """
    Execute a specific agent with a given task.

    - **agent_id**: ID of the agent to execute
    - **request**: Chat request with task details
    """
    try:
        import time

        start_time = time.time()

        # Validate agent exists
        agent = await agent_orchestrator.get_agent(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_id}' not found",
            )

        # Execute agent
        result = await agent_orchestrator.execute_agent(
            agent_id=agent_id,
            message=request.message,
            context=request.context,
            parameters={
                "model": request.model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
        )

        processing_time = time.time() - start_time

        # Background task to update agent metrics
        background_tasks.add_task(
            agent_orchestrator.update_agent_metrics,
            agent_id=agent_id,
            processing_time=processing_time,
            success=result.get("success", True),
        )

        return {
            "agent_id": agent_id,
            "result": result,
            "processing_time": round(processing_time, 3),
            "timestamp": "2025-01-16T08:53:13Z",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )


@router.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get current status and metrics for a specific agent."""
    try:
        agent = await agent_orchestrator.get_agent(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_id}' not found",
            )

        status_info = await agent_orchestrator.get_agent_status(agent_id)
        return {
            "agent_id": agent_id,
            "status": status_info,
            "timestamp": "2025-01-16T08:53:13Z",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent status: {str(e)}",
        )


# =================================
# STREAMING ENDPOINTS (Future Implementation)
# =================================


@router.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    Streaming chat endpoint (future implementation).

    This would return Server-Sent Events for real-time streaming.
    """
    # TODO: Implement streaming response
    return {
        "message": "Streaming endpoint - implementation pending",
        "conversation_id": request.conversation_id or "new",
        "timestamp": "2025-01-16T08:53:13Z",
    }


# =================================
# RAG ENDPOINTS
# =================================


@router.post("/rag/search")
async def search_knowledge_base(
    query: str, top_k: int = 10, conversation_id: Optional[str] = None
):
    """
    Search the knowledge base using RAG.

    - **query**: Search query text
    - **top_k**: Number of results to return (1-50)
    - **conversation_id**: Optional conversation context
    """
    try:
        if not query or len(query.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty"
            )

        if top_k < 1 or top_k > 50:
            top_k = min(max(top_k, 1), 50)

        results = await ai_service.search_knowledge_base(
            query=query.strip(), top_k=top_k, conversation_id=conversation_id
        )

        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "timestamp": "2025-01-16T08:53:13Z",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG search failed: {str(e)}",
        )
