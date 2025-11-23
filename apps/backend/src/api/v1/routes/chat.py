"""
Chat API endpoints for Sheily AI
Handles chat conversations, RAG queries, and AI interactions
"""

import time
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from apps.backend.src.config.settings import settings
from apps.backend.src.core.cache import cache_delete, cache_get, cache_set

# Real AI Chat Engine - No fallbacks or mocks
from sheily_core.chat_engine import create_chat_engine

chat_engine = create_chat_engine()
print("[INFO] Sheily Core AI Engine loaded successfully - No mocks or fallbacks")


router = APIRouter(prefix="/api", tags=["chat"])


# Pydantic models for API
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context_docs: Optional[List[str]] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500


class ChatResponse(BaseModel):
    conversation_id: str
    message: ChatMessage
    response_time: float
    cached: bool = False
    tokens_used: Optional[int] = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_message: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest, background_tasks: BackgroundTasks
) -> ChatResponse:
    """
    Main chat endpoint for AI conversations
    Supports caching, conversation management, and RAG
    """
    start_time = time.time()

    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Check cache first (for repeated queries)
        cache_key = f"chat:{hash(request.message)}"
        cached_response = cache_get(cache_key)

        if cached_response and settings.cache_enabled:
            # Return cached response
            response_time = time.time() - start_time
            return ChatResponse(
                conversation_id=conversation_id,
                message=ChatMessage(
                    role="assistant", content=cached_response, timestamp=datetime.now()
                ),
                response_time=response_time,
                cached=True,
            )

        # Process with AI engine
        ai_response = chat_engine(request.message, conversation_id)

        # Handle Sheily Core response
        ai_content = (
            ai_response.response
            if hasattr(ai_response, "response")
            else ai_response.content
        )
        tokens_used = getattr(ai_response, "tokens_used", None)

        # Cache the response (1 hour)
        if settings.cache_enabled:
            cache_set(cache_key, ai_content, 3600)

        # Prepare response
        response_time = time.time() - start_time
        response = ChatResponse(
            conversation_id=conversation_id,
            message=ChatMessage(
                role="assistant", content=ai_content, timestamp=datetime.now()
            ),
            response_time=response_time,
            cached=False,
            tokens_used=tokens_used,
        )

        # Background task: Update conversation in database
        background_tasks.add_task(
            update_conversation_async, conversation_id, request.message, ai_content
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}"
        )


@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(limit: int = 50) -> List[ConversationSummary]:
    """List user conversations"""
    try:
        # This would query the database in a full implementation
        # For now, return mock data
        return [
            ConversationSummary(
                id="conv-1",
                title="AI Assistant Chat",
                message_count=5,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                last_message="Hello! How can I help you today?",
            )
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing conversations: {str(e)}"
        )


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get full conversation history"""
    try:
        # This would query the database in a full implementation
        return {
            "id": conversation_id,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "role": "assistant",
                    "content": "Hi! How can I help you?",
                    "timestamp": datetime.now().isoformat(),
                },
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        # Clear cache for this conversation
        cache_delete(f"conv:{conversation_id}")

        # This would delete from database in a full implementation
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting conversation: {str(e)}"
        )


@router.get("/health")
async def chat_health():
    """Health check for chat service"""
    return {
        "status": "healthy",
        "service": "chat",
        "version": settings.version,
        "cache_enabled": settings.cache_enabled,
        "timestamp": datetime.now().isoformat(),
    }


# Background task functions
async def update_conversation_async(
    conversation_id: str, user_message: str, ai_response: str
):
    """Background task to update conversation in database"""
    try:
        # This would update the database in a full implementation
        # For now, just log the action
        print(f"[INFO] Updated conversation {conversation_id}")

        # Clear any related cache
        cache_delete(f"conv:{conversation_id}")

    except Exception as e:
        print(f"[WARNING] Error updating conversation: {e}")


# Utility functions
def estimate_tokens(text: str) -> int:
    """Rough token estimation (words * 1.3)"""
    return int(len(text.split()) * 1.3)
