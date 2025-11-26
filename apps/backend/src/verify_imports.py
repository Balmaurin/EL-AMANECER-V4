import sys
import os

# Add project root to path (apps/backend/src -> project root is 3 levels up)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)
print(f"📁 Project root: {project_root}")


def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error importing {module_name}: {e}")
        return False

modules_to_check = [
    "apps.backend.src.core.config.settings",
    "apps.backend.src.core.auth.service",
    "apps.backend.src.core.security.manager",
    "apps.backend.src.core.database.session",
    "apps.backend.src.services.ai.ai_service",
    "apps.backend.src.api.v1.routes.auth",
    "apps.backend.src.api.v1.routes.chat",
    "apps.backend.src.api.v1.routes.users",
    "apps.backend.src.api.v1.routes.datasets",
]

print("🔍 Verifying imports...")
success_count = 0
for module in modules_to_check:
    if check_import(module):
        success_count += 1

print(f"\n📊 Result: {success_count}/{len(modules_to_check)} modules imported successfully.")

if success_count == len(modules_to_check):
    print("🎉 All checks passed!")
    sys.exit(0)
else:
    print("⚠️ Some checks failed.")
    sys.exit(1)
