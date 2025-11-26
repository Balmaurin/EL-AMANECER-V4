"""
FastAPI Application - Sheily AI MCP Enterprise
Main application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Sheily AI MCP Enterprise API",
    description="Next-generation AI Operating System with Consciousness, Blockchain, and Specialized Agents",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers individually to prevent cascade failure
# Chat Router
try:
    from apps.backend.src.api.v1.routes import chat
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    print("✅ Chat routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Chat routes could not be loaded: {e}")

# Health Router
try:
    from apps.backend.src.api.v1.routes import health
    app.include_router(health.router, prefix="/api/health", tags=["health"])
    print("✅ Health routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Health routes could not be loaded: {e}")

# RAG Router
try:
    from apps.backend.src.api.v1.routes import rag
    app.include_router(rag.router, prefix="/api/rag", tags=["rag"])
    print("✅ RAG routes loaded")
except Exception as e:
    print(f"⚠️ Warning: RAG routes could not be loaded: {e}")

# System Router
try:
    from apps.backend.src.api.v1.routes import system
    app.include_router(system.router, prefix="/api/system", tags=["system"])
    print("✅ System routes loaded")
except Exception as e:
    print(f"⚠️ Warning: System routes could not be loaded: {e}")

# Dashboard Router
try:
    from apps.backend.src.api.v1.routes import dashboard
    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
    print("✅ Dashboard routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Dashboard routes could not be loaded: {e}")

# Users Router
try:
    from apps.backend.src.api.v1.routes import users
    app.include_router(users.router, prefix="/api/users", tags=["users"])
    print("✅ Users routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Users routes could not be loaded: {e}")

# Auth Router
try:
    from apps.backend.src.api.v1.routes import auth
    app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
    print("✅ Auth routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Auth routes could not be loaded: {e}")

# Datasets Router
try:
    from apps.backend.src.api.v1.routes import datasets
    app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
    print("✅ Datasets routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Datasets routes could not be loaded: {e}")

# Blockchain Router
try:
    from apps.backend.src.api.v1.routes import blockchain
    app.include_router(blockchain.router, prefix="/api/blockchain", tags=["blockchain"])
    print("✅ Blockchain routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Blockchain routes could not be loaded: {e}")

# Marketplace Router
try:
    from apps.backend.src.api.v1.routes import marketplace
    app.include_router(marketplace.router, prefix="/api/marketplace", tags=["marketplace"])
    print("✅ Marketplace routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Marketplace routes could not be loaded: {e}")

# Conversations Router
try:
    from apps.backend.src.api.v1.routes import conversations
    app.include_router(conversations.router, prefix="/api/conversations", tags=["conversations"])
    print("✅ Conversations routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Conversations routes could not be loaded: {e}")

# Analytics Router
try:
    from apps.backend.src.api.v1.routes import analytics
    app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
    print("✅ Analytics routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Analytics routes could not be loaded: {e}")

# MCP Orchestration Router
try:
    from apps.backend.src.api.v1.routes import mcp_orchestration
    app.include_router(mcp_orchestration.router, prefix="/api", tags=["mcp-orchestration"])
    print("✅ MCP Orchestration routes loaded")
except Exception as e:
    print(f"⚠️ Warning: MCP Orchestration routes could not be loaded: {e}")


# WebSocket Router
try:
    from apps.backend.src.api.v1.routes import websocket
    app.include_router(websocket.router, prefix="/api/ws", tags=["websocket"])
    print("✅ WebSocket routes loaded")
except Exception as e:
    print(f"⚠️ Warning: WebSocket routes could not be loaded: {e}")

# Exercises Router
try:
    from apps.backend.src.api.v1.routes import exercises
    app.include_router(exercises.router, prefix="/api/exercises", tags=["exercises"])
    print("✅ Exercises routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Exercises routes could not be loaded: {e}")

# Knowledge Router
try:
    from apps.backend.src.api.v1.routes import knowledge
    app.include_router(knowledge.router, prefix="/api/knowledge", tags=["knowledge"])
    print("✅ Knowledge routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Knowledge routes could not be loaded: {e}")

# Community Router
try:
    from apps.backend.src.api.v1.routes import community
    app.include_router(community.router, prefix="/api/community", tags=["community"])
    print("✅ Community routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Community routes could not be loaded: {e}")

# Uploads Router
try:
    from apps.backend.src.api.v1.routes import uploads
    app.include_router(uploads.router, prefix="/api/uploads", tags=["uploads"])
    print("✅ Uploads routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Uploads routes could not be loaded: {e}")

# Vault Router
try:
    from apps.backend.src.api.v1.routes import vault
    app.include_router(vault.router, prefix="/api/vault", tags=["vault"])
    print("✅ Vault routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Vault routes could not be loaded: {e}")

# Service Registry Router (MCP Integration)
try:
    from apps.backend.src.api.v1.routes import service_registry
    app.include_router(service_registry.router, prefix="/api/mcp", tags=["mcp", "service-registry"])
    print("✅ MCP Service Registry routes loaded")
except Exception as e:
    print(f"⚠️ Warning: MCP Service Registry routes could not be loaded: {e}")

# Theory of Mind Router (Advanced - Levels 8-10)

try:
    from apps.backend.src.api.v1.routes import tom
    app.include_router(tom.router, tags=["Theory of Mind"])
    print("✅ Theory of Mind routes loaded (Levels 8-10)")
except Exception as e:
    print(f"⚠️ Warning: Theory of Mind routes could not be loaded: {e}")



print(f"DEBUG: App ID in __init__: {id(app)}")
print(f"DEBUG: Registered routes: {[r.path for r in app.routes]}")

@app.get("/")
async def root():
    return {
        "name": "Sheily AI MCP Enterprise API",
        "version": "3.0.0",
        "status": "operational",
        "features": [
            "Consciousness (IIT 4.0, GWT, FEP, SMH)",
            "Theory of Mind (Levels 8-10)",
            "Auto-Evolution",
            "Blockchain",
            "RAG",
            "Multi-Agent"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "chat": "/api/chat",
            "tom": "/api/v1/tom/status",
            "mcp_manifest": "/api/mcp/services/manifest",
            "mcp_services": "/api/mcp/services/list"
        },
        "achievements": {
            "tom_level": "8-9 (World's First Validated)",
            "papers": 36,
            "fidelity": "91.5%"
        }
    }

@app.get("/api")
async def api_root():
    return {
        "message": "Sheily AI MCP Enterprise API v3.0",
        "status": "online"
    }
