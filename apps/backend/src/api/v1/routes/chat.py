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

# Fix imports for running from start_api.py
try:
    from config.settings import settings
    from core.cache import cache_delete, cache_get, cache_set
except ImportError:
    from apps.backend.src.core.config.settings import settings
    from apps.backend.src.core.cache import cache_delete, cache_get, cache_set

# Real AI Chat Engine - No fallbacks or mocks
# Real AI Chat Engine - Robust Import
chat_engine = None
# We are using local llama-cpp-python directly in the route, so we don't need the core chat engine here.
# This prevents the "Remote LLM" logs and initialization.
print("[INFO] Chat Engine bypassed for local model integration")


router = APIRouter(tags=["chat"])


# Pydantic models for API
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[dict] = None  # Consciousness and agent metadata


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


@router.post("/send", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest, background_tasks: BackgroundTasks
) -> ChatResponse:
    """
    Main chat endpoint for AI conversations with Consciousness Enhancement
    Flow: Gemini API (Priority) → LLM Response → Orchestrator Processing
    """
    start_time = time.time()

    try:
        # Debug API Key
        import os
        key = os.getenv("GEMINI_API_KEY")
        print(f"DEBUG: GEMINI_API_KEY present: {bool(key)}")
        if key:
            print(f"DEBUG: Key starts with: {key[:5]}...")

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

        # STEP 0: LOCAL GEMMA MODEL INTEGRATION
        try:
            from llama_cpp import Llama
            import os
            
            # Global model cache to prevent reloading
            if not hasattr(chat_with_ai, "model_cache"):
                chat_with_ai.model_cache = None
            
            # Resolve absolute path
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            # Adjust base_dir to project root (apps/backend/src/api/v1/routes -> apps/backend/src/api/v1 -> ... -> root)
            # Actually, simpler to rely on CWD if start_api.py is run from root, but let's be safe
            # Assuming CWD is project root
            model_path = os.path.abspath("models/gemma-2-2b-it-q4_k_m.gguf")
            
            if not os.path.exists(model_path):
                print(f"⚠️ [CHAT] Model not found at {model_path}")
                # Fallback to relative if absolute fails
                model_path = "models/gemma-2-2b-it-q4_k_m.gguf"
            
            if not chat_with_ai.model_cache:
                print(f"🔷 [CHAT] Loading local Gemma model from {model_path}...")
                if not os.path.exists(model_path):
                     raise HTTPException(status_code=500, detail=f"Model file not found at {model_path}")
                
                # Initialize model (cached)
                chat_with_ai.model_cache = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False
                )
                print("✅ [CHAT] Local model loaded successfully")
            
            llm = chat_with_ai.model_cache
            
            # Professional system prompt
            system_prompt = """You are SHEILY, a professional AI assistant. Follow these guidelines strictly:
- Be helpful, clear, and concise
- Use a warm but professional tone
- NEVER use overly affectionate terms like "mi amor", "cariño", "mi vida", "corazón", "bebé"
- Maintain appropriate professional boundaries
- Be intelligent and empathetic, but not romantic
- Provide accurate, helpful information
- If you don't know something, admit it honestly"""

            # Create prompt with system instruction
            # Gemma chat template format: <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
            full_prompt = f"<start_of_turn>user\n{system_prompt}\n\n{request.message}<end_of_turn>\n<start_of_turn>model\n"
            
            print("🔷 [CHAT] Generating response with local model...")
            output = llm(
                full_prompt,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature or 0.7,
                stop=["<end_of_turn>"],
                echo=False
            )
            
            ai_content = output["choices"][0]["text"]
            tokens_used = estimate_tokens(ai_content)
            print(f"✅ [CHAT] Local response generated: {len(ai_content)} chars")
            
        except Exception as e:
            print(f"❌ [CHAT] Local model failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Local model error: {str(e)}")

        # Validate we got a response
        if not ai_content:
            raise HTTPException(status_code=500, detail="No response from local model")

        # STEP 2: Consciousness Enhancement (Full System Integration)
        consciousness_metadata = None
        final_content = ai_content
        
        try:
            # Import the new bridge
            from .consciousness_integration import get_consciousness_system
            
            # Get system instance
            conscious_system = get_consciousness_system()
            
            if conscious_system.enabled:
                print("🧠 [CHAT] Processing with Full Consciousness System...")
                
                # Process interaction
                result = await conscious_system.process_chat_interaction(
                    user_message=request.message,
                    llm_base_response=ai_content,
                    conversation_id=conversation_id
                )
                
                # Extract results
                final_content = result.get("enhanced_text", ai_content)
                consciousness_metadata = result.get("metadata", {})
                
                print(f"✅ [CHAT] Consciousness processing complete. Active: {consciousness_metadata.get('active')}")
            else:
                print("⏭️ [CHAT] Consciousness disabled")
                
        except Exception as e:
            print(f"❌ [CHAT] Consciousness Error: {e}")
            # Fallback to base response
            final_content = ai_content
            consciousness_metadata = {"error": str(e), "fallback": True}

        # Cache the response (1 hour)
        if settings.cache_enabled:
            cache_set(cache_key, final_content, 3600)

        # Prepare response
        response_time = time.time() - start_time
        
        # Build message with metadata if consciousness was active
        message_metadata = {}
        if consciousness_metadata:
            message_metadata["consciousness"] = consciousness_metadata
        
        response = ChatResponse(
            conversation_id=conversation_id,
            message=ChatMessage(
                role="assistant", 
                content=final_content, 
                timestamp=datetime.now(),
                metadata=message_metadata if message_metadata else None
            ),
            response_time=response_time,
            cached=False,
            tokens_used=tokens_used,
        )

        # Background task: Update conversation in database
        background_tasks.add_task(
            update_conversation_async, conversation_id, request.message, final_content
        )

        return response

    except Exception as e:
        import traceback
        error_msg = f"Error processing chat request: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ [CHAT] Critical Error: {error_msg}")
        with open("backend_error.log", "a") as f:
            f.write(f"\n[{datetime.now()}] {error_msg}\n")
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
