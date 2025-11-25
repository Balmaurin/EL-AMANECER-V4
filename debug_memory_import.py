import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "packages" / "sheily_core" / "src"))

print("Intentando importar sheily_chat_memory_adapter...")
try:
    from sheily_core.chat.sheily_chat_memory_adapter import respond, vault
    print("✅ Importación exitosa")
except Exception as e:
    print(f"❌ Error de importación: {e}")
    import traceback
    traceback.print_exc()
