import asyncio
import sys
from pathlib import Path

# Configurar paths críticos ANTES de importar nada
ROOT_DIR = Path.cwd()
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "packages" / "sheily_core" / "src"))
sys.path.append(str(ROOT_DIR / "packages" / "rag_engine" / "src"))

# Importar la función modificada
from start_system import run_enterprise_system

async def test_evolution_activation():
    print("\n🧪 TEST DE VERIFICACIÓN DE VIDA ARTIFICIAL")
    print("==========================================")
    
    # Ejecutar la inicialización
    success = await run_enterprise_system()
    
    if success:
        print("\n✅ PRUEBA EXITOSA: El sistema ha detectado y activado:")
        print("   - Evolución Genética")
        print("   - Scheduler")
        print("   - Consciencia (Meta-Cognición)")
        print("   - Motor de Sueños")
        print("   - Entrenamiento Neuronal")
    else:
        print("\n❌ PRUEBA FALLIDA: Algo salió mal en la inicialización.")

if __name__ == "__main__":
    asyncio.run(test_evolution_activation())
