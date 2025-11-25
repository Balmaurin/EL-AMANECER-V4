import sys
from pathlib import Path

# Add paths exactly as start_api.py does
project_root = Path(__file__).parent
backend_src = project_root / "apps" / "backend" / "src"
packages = project_root / "packages"

sys.path.insert(0, str(backend_src))
sys.path.insert(0, str(packages / "sheily_core" / "src"))
sys.path.insert(0, str(packages / "rag_engine"))
sys.path.insert(0, str(packages / "consciousness" / "src"))

print("Paths added:")
for p in sys.path[:4]:
    print(f"  - {p}")

print("\nAttempting import...")
try:
    from sheily_core.chat.chat_engine import create_chat_engine
    print("✅ Import SUCCESS!")
except Exception as e:
    print(f"❌ Import FAILED: {e}")
    import traceback
    traceback.print_exc()
