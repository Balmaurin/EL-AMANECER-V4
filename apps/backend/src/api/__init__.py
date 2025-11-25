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
    from api.v1.routes import chat
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    print("✅ Chat routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Chat routes could not be loaded: {e}")

# Health Router
try:
    from api.v1.routes import health
    app.include_router(health.router, prefix="/api/health", tags=["health"])
    print("✅ Health routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Health routes could not be loaded: {e}")

# RAG Router
try:
    from api.v1.routes import rag
    app.include_router(rag.router, prefix="/api/rag", tags=["rag"])
    print("✅ RAG routes loaded")
except Exception as e:
    print(f"⚠️ Warning: RAG routes could not be loaded: {e}")

# System Router
try:
    from api.v1.routes import system
    app.include_router(system.router, prefix="/api/system", tags=["system"])
    print("✅ System routes loaded")
except Exception as e:
    print(f"⚠️ Warning: System routes could not be loaded: {e}")

# Dashboard Router
try:
    from api.v1.routes import dashboard
    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
    print("✅ Dashboard routes loaded")
except Exception as e:
    print(f"⚠️ Warning: Dashboard routes could not be loaded: {e}")

# User Router
try:
    from api.v1.routes import user
    app.include_router(user.router, prefix="/api/user", tags=["user"])
    print("✅ User routes loaded")
except Exception as e:
    print(f"⚠️ Warning: User routes could not be loaded: {e}")


print(f"DEBUG: App ID in __init__: {id(app)}")
print(f"DEBUG: Registered routes: {[r.path for r in app.routes]}")

@app.get("/")
async def root():
    return {
        "name": "Sheily AI MCP Enterprise API",
        "version": "3.0.0",
        "status": "operational",
        "features": ["Consciousness", "Blockchain", "RAG", "Agents"],
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "chat": "/api/chat"
        }
    }

@app.get("/api")
async def api_root():
    return {
        "message": "Sheily AI MCP Enterprise API v3.0",
        "status": "online"
    }
