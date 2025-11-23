import sys
import asyncio
from pathlib import Path

# Configurar path absoluto y sub-paths críticos
ROOT_DIR = Path.cwd()
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "packages" / "sheily_core" / "src"))
sys.path.append(str(ROOT_DIR / "packages" / "rag_engine" / "src"))

# Importar el módulo de auto-mejora
try:
    from packages.rag_engine.src.core.mcp_auto_improvement import run_mcp_auto_improvement
    print("✅ Módulo de auto-mejora importado correctamente")
except ImportError as e:
    print(f"❌ Error importando módulo: {e}")
    # Intentar ruta alternativa si falla
    sys.path.append(str(ROOT_DIR / "packages" / "rag-engine" / "src"))
    from core.mcp_auto_improvement import run_mcp_auto_improvement

async def launch():
    print("\n🔥 INICIANDO PROTOCOLO DE AUTO-EVOLUCIÓN")
    print("========================================")
    # Ejecutar solo 1 iteración para la demo rápida
    await run_mcp_auto_improvement(full_cycle=True, iterations=1)

if __name__ == "__main__":
    asyncio.run(launch())
