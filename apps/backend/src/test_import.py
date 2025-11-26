import sys
import os
import traceback

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)
print(f"Project root: {project_root}\n")

print("=" * 60)
print("Testing: apps.backend.src.api.v1.routes.users")
print("=" * 60)

try:
    import apps.backend.src.api.v1.routes.users
    print("✅ SUCCESS")
except Exception as e:
    print(f"❌ FAILED: {e}\n")
    print("Full traceback:")
    traceback.print_exc()
