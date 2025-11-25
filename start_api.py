#!/usr/bin/env python3
"""
FastAPI Backend Starter
Launches the Sheily AI MCP Enterprise backend API server
"""
import sys
from pathlib import Path

# Add backend src and packages to path
project_root = Path(__file__).parent
backend_src = project_root / "apps" / "backend" / "src"
packages = project_root / "packages"

# Insert at beginning of path to prioritize local packages
sys.path.insert(0, str(backend_src))
sys.path.insert(0, str(packages / "sheily_core" / "src"))
sys.path.insert(0, str(packages / "rag_engine"))
sys.path.insert(0, str(packages / "consciousness" / "src"))

if __name__ == "__main__":
    import uvicorn
    
    # Debug import source
    try:
        import sheily_core
        print(f"DEBUG: sheily_core loaded from: {sheily_core.__file__}")
    except ImportError:
        print("DEBUG: Could not import sheily_core to check path")

    # Import the FastAPI app
    # FIX: Alias CONCIENCIA to conciencia to resolve case sensitivity/typo issues
    try:
        import conciencia
        sys.modules['CONCIENCIA'] = conciencia
        print("✅ Fixed 'CONCIENCIA' import alias")
    except ImportError:
        print("⚠️ Could not import 'conciencia' to fix alias")

    try:
        from api import app
        print("✅ FastAPI app loaded successfully")
        print(f"DEBUG: App ID in start_api: {id(app)}")
    except Exception as e:
        print(f"❌ Error loading FastAPI app: {e}")
        print("\nTrying to debug...")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("🚀 Starting Sheily AI MCP Enterprise API Server...")
    print("📍 Backend: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("🔗 Remote LLM: Using configured Gemini backend")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
