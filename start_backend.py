#!/usr/bin/env python3
"""
Servidor Backend Completo para EL-AMANECERV3
=============================================

Inicia FastAPI con MCP Chat simplificado inline.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Añadir sheily-core al path
current_dir = Path(__file__).parent.absolute()
sheily_core_path = current_dir / "packages" / "sheily_core" / "src"
sys.path.insert(0, str(sheily_core_path))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Crear app FastAPI
app = FastAPI(
    title="EL-AMANECERV3 Backend",
    description="Backend completo con MCP Enterprise Master",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check básico
@app.get("/")
def root():
    return {
        "status": "operational",
        "system": "EL-AMANECERV3",
        "backend": "FastAPI + MCP Enterprise"
    }

@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "services": ["mcp_chat", "dashboard", "consciousness"]
    }

# ==========================================
# MCP CHAT ENDPOINTS (REAL)
# ==========================================

# Importar router de MCP Chat Real
try:
    sys.path.append(str(current_dir / "apps" / "backend" / "src"))
    from api.v1.routes.mcp_chat import router as mcp_chat_router
    app.include_router(mcp_chat_router, prefix="/api/v1", tags=["mcp-chat"])
    print("\n" + "="*70)
    print("✅ ✅ ✅ MCP CHAT ROUTER REAL CARGADO EXITOSAMENTE ✅ ✅ ✅")
    print("="*70 + "\n")
    import sys
    sys.stdout.flush()
except Exception as e:
    import traceback
    print("\n" + "="*70)
    print(f"[ERROR] Error cargando MCP Chat router real:")
    print(f"{e}")
    traceback.print_exc()
    print("="*70)
    print("❌ Sistema NO puede arrancar sin el router de chat.")
    print("="*70 + "\n")
    import sys
    sys.stdout.flush()
    raise e

# Importar dashboard router si está disponible
try:
    sys.path.append(str(current_dir / "apps" / "backend" / "src"))
    from api.v1.routes.dashboard import router as dashboard_router
    app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["dashboard"])
    print("✅ Dashboard router cargado")
except Exception as e:
    print(f"⚠️ Dashboard router no disponible: {e}")

if __name__ == "__main__":
    print("\\n" + "=" * 70)
    print("  EL-AMANECERV3 - BACKEND SERVIDOR COMPLETO")
    print("=" * 70)
    print("")
    print("  🌐 Dashboard:     http://localhost:8001")
    print("  📚 API Docs:      http://localhost:8001/docs")
    print("  💬 MCP Chat:      POST http://localhost:8001/api/v1/mcp/chat/message")
    print("  🧠 Consciousness: http://localhost:8001/api/v1/dashboard/consciousness")
    print("")
    print("=" * 70)
    print("")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False
    )
