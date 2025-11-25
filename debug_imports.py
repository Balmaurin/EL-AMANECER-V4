import sys
from pathlib import Path
import traceback

# Setup paths exactly as in consciousness_integration.py
current_dir = Path("c:/Users/YO/Desktop/EL-AMANECERV3-main/apps/backend/src/api/v1/routes")
project_root = Path("c:/Users/YO/Desktop/EL-AMANECERV3-main")
packages_dir = project_root / "packages"

consciousness_pkg = packages_dir / "consciousness" / "src"
sheily_core_pkg = packages_dir / "sheily_core" / "src"
backend_src = project_root / "apps" / "backend" / "src"

sys.path.append(str(consciousness_pkg))
sys.path.append(str(sheily_core_pkg))
sys.path.append(str(backend_src))

print("Testing imports...")

try:
    print("1. Importing HumanEmotionalSystem...")
    from conciencia.modulos.human_emotions_system import HumanEmotionalSystem
    print("   SUCCESS")
except Exception as e:
    print(f"   FAILED: {e}")
    traceback.print_exc()

try:
    print("2. Importing ConsciousEnhancedOrchestrator...")
    from sheily_core.core.system.conscious_enhanced_orchestrator import get_conscious_enhanced_orchestrator
    print("   SUCCESS")
except Exception as e:
    print(f"   FAILED: {e}")
    traceback.print_exc()

try:
    print("3. Importing NeuralBrainLearner...")
    from sheily_core.models.ml.neural_brain_learner import NeuralBrainLearner
    print("   SUCCESS")
except Exception as e:
    print(f"   FAILED: {e}")
    traceback.print_exc()

try:
    print("4. Importing AgentOrchestrator...")
    from api.v1.routes.agent_orchestration import get_agent_orchestrator
    print("   SUCCESS")
except Exception as e:
    print(f"   FAILED: {e}")
    traceback.print_exc()

try:
    print("5. Importing LLMFactory...")
    from core.llm.llm_factory import LLMFactory
    print("   SUCCESS")
except Exception as e:
    print(f"   FAILED: {e}")
    traceback.print_exc()
