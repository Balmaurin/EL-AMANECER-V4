import sys
import os
import traceback

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

print("=" * 60)
print("Testing: apps.backend.src.core.auth.service")
print("=" * 60)

try:
    import apps.backend.src.core.auth.service
    print("✅ SUCCESS")
except Exception as e:
    print(f"❌ FAILED: {e}\n")
    print("Full traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing: apps.backend.src.services.ai.ai_service")
print("=" * 60)

try:
    import apps.backend.src.services.ai.ai_service
    print("✅ SUCCESS")
except Exception as e:
    print(f"❌ FAILED: {e}\n")
    print("Full traceback:")
    traceback.print_exc()
