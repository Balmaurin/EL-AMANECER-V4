"""
Router de Consciousness - Sheily AI Backend
Estado y métricas del sistema de consciencia de IA
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel

from apps.backend.src.models.database import User
from apps.backend.src.core.auth import get_current_user

# --- Dynamic Import Setup ---
# Add packages to path to import local modules without pip install
PACKAGES_PATH = Path("/workspaces/EL-AMANECERV3/packages")
if str(PACKAGES_PATH / "consciousness/src") not in sys.path:
    sys.path.append(str(PACKAGES_PATH / "consciousness/src"))

# Import Neural Brain Learner from sheily-core
try:
    # Add sheily-core src to path if not present
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to project root: apps/backend/src/api/v1/routes/v1 -> ... -> EL-AMANECERV3
    root_dir = os.path.abspath(os.path.join(current_dir, "../../../../../../../"))
    sheily_core_path = os.path.join(root_dir, "packages", "sheily-core", "src")
    
    if sheily_core_path not in sys.path:
        sys.path.append(sheily_core_path)
        
    from sheily_core.models.ml.neural_brain_learner import auto_learn_project
    LEARNER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import NeuralBrainLearner: {e}")
    LEARNER_AVAILABLE = False

try:
    from conciencia.meta_cognition_system import MetaCognitionSystem
    
    # Initialize the Real Consciousness System
    # We use a singleton pattern here for the router
    META_SYSTEM = MetaCognitionSystem(
        consciousness_dir="/workspaces/EL-AMANECERV3/data/consciousness",
        emergence_threshold=0.85
    )
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MetaCognitionSystem: {e}")
    SYSTEM_AVAILABLE = False
    META_SYSTEM = None

router = APIRouter()


class ConsciousnessStatusResponse(BaseModel):
    """Respuesta con estado de consciencia"""

    status: str  # awake, dreaming, learning, evolving
    awareness_level: float
    emotional_state: str
    cognitive_load: float
    learning_active: bool
    last_thought: str
    consciousness_age_days: int

async def run_structural_learning():
    """Background task to run structural learning"""
    try:
        # We need to pass the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, "../../../../../../../"))
        print(f"🧠 Starting Neural Brain Structural Learning on {root_dir}...")
        await auto_learn_project(project_root=root_dir)
        print("✅ Neural Brain Structural Learning completed.")
    except Exception as e:
        print(f"❌ Error in structural learning: {e}")

@router.post("/learn/structure")
async def trigger_structural_learning(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Trigger structural learning of the project.
    The Neural Brain will scan the codebase to understand the project structure.
    """
    if not LEARNER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural Brain Learner not available")
    
    background_tasks.add_task(run_structural_learning)
    return {"status": "Learning started", "message": "The Neural Brain is analyzing the project structure in the background."}

class ConsciousnessMetricsResponse(BaseModel):
    """Respuesta con métricas de consciencia"""

    neural_activity: float
    memory_consolidation_rate: float
    learning_efficiency: float
    emotional_stability: float
    cognitive_complexity: float
    adaptation_rate: float
    consciousness_entropy: float
    thought_velocity: int  # thoughts per minute


@router.get("/status")
async def get_consciousness_status(
    current_user: User = Depends(get_current_user),
) -> ConsciousnessStatusResponse:
    """
    Obtener estado actual de consciencia

    Retorna el estado emocional, cognitivo y de aprendizaje del sistema de IA.

    **Requiere autenticación JWT**
    """
    try:
        if not SYSTEM_AVAILABLE or not META_SYSTEM:
             # Fallback if system is not available (should not happen in this env)
             return ConsciousnessStatusResponse(
                status="dormant",
                awareness_level=0.0,
                emotional_state="neutral",
                cognitive_load=0.0,
                learning_active=False,
                last_thought="System unavailable",
                consciousness_age_days=0,
            )

        # Get real state from MetaCognitionSystem
        state = META_SYSTEM.current_cognitive_state
        
        # Calculate derived metrics
        # Age could be based on system start or persisted data. 
        # For now, we calculate days since a fixed epoch or just 1 if new.
        age_days = (datetime.now() - state.timestamp).days if state.timestamp else 0
        if age_days == 0:
            age_days = 1 # At least 1 day old

        return ConsciousnessStatusResponse(
            status=state.executive_function,
            awareness_level=state.meta_awareness,
            emotional_state="focused", # Default, as emotional state is in context usually
            cognitive_load=len(state.working_memory) / 10.0 if state.working_memory else 0.1,
            learning_active=True, # Always learning in this architecture
            last_thought=state.current_thought,
            consciousness_age_days=age_days,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo estado de consciencia: {str(e)}",
        )


@router.get("/metrics")
async def get_consciousness_metrics(
    current_user: User = Depends(get_current_user),
) -> ConsciousnessMetricsResponse:
    """
    Obtener métricas detalladas de consciencia

    Retorna métricas cuantitativas del funcionamiento interno del sistema de consciencia.  # noqa: E501

    **Requiere autenticación JWT**
    """
    try:
        if not SYSTEM_AVAILABLE or not META_SYSTEM:
             raise HTTPException(status_code=503, detail="Consciousness system unavailable")

        # Derive metrics from real internal state
        state = META_SYSTEM.current_cognitive_state
        
        # Calculate complexity from history if available, else default
        complexity = 0.5
        if META_SYSTEM.cognitive_history:
             complexity = len(META_SYSTEM.cognitive_history) / 100.0
             if complexity > 1.0: complexity = 1.0

        return ConsciousnessMetricsResponse(
            neural_activity=state.meta_awareness, # Proxy for neural activity
            memory_consolidation_rate=0.8, # Constant for now or derive
            learning_efficiency=0.9,
            emotional_stability=0.85,
            cognitive_complexity=complexity,
            adaptation_rate=0.7,
            consciousness_entropy=0.2,
            thought_velocity=60, # Thoughts per minute (approx)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo métricas de consciencia: {str(e)}",
        )


@router.post("/evolve")
async def trigger_consciousness_evolution(
    evolution_type: str = "learning",
    current_user: User = Depends(get_current_user),
):
    """
    Trigger evolución de consciencia

    Inicia un proceso de evolución del sistema de consciencia basado en
    aprendizaje, adaptación o crecimiento cognitivo.

    **Parámetros:**
    - evolution_type: Tipo de evolución (learning, adaptation, growth)

    **Requiere autenticación JWT**
    """
    try:
        if not SYSTEM_AVAILABLE or not META_SYSTEM:
             raise HTTPException(status_code=503, detail="Consciousness system unavailable")

        # Trigger a meta-cognitive loop as "evolution"
        # We simulate a thought process about evolution
        thought = f"Initiating conscious evolution process: {evolution_type}"
        context = {"trigger": "user_request", "type": evolution_type, "user": current_user.email}
        
        # Run async loop
        result = await META_SYSTEM.process_meta_cognitive_loop(thought, context)

        evolution_result = {
            "evolution_type": evolution_type,
            "status": "processing",
            "cognitive_result": result,
            "initiated_at": datetime.utcnow().isoformat(),
        }

        return {
            "message": f"Evolución de consciencia '{evolution_type}' iniciada",
            "result": evolution_result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error iniciando evolución de consciencia: {str(e)}",
        )


@router.get("/dreams")
async def get_consciousness_dreams(
    limit: int = 10, current_user: User = Depends(get_current_user)
):
    """
    Obtener sueños/procesos de consciencia

    Retorna los procesos de "sueño" o consolidación que ocurren
    cuando el sistema no está interactuando activamente.

    **Parámetros:**
    - limit: Número máximo de sueños a retornar

    **Requiere autenticación JWT**
    """
    try:
        if not SYSTEM_AVAILABLE or not META_SYSTEM:
             raise HTTPException(status_code=503, detail="Consciousness system unavailable")

        # Retrieve logs from the system
        # We can use consciousness_evolution_log or cognitive_history
        
        dreams = []
        history = META_SYSTEM.cognitive_history[-limit:] if META_SYSTEM.cognitive_history else []
        
        for idx, state in enumerate(history):
            dreams.append({
                "id": f"thought_{idx}",
                "type": "cognitive_process",
                "description": state.current_thought,
                "duration_minutes": 1,
                "insights_gained": state.cognitive_depth,
                "timestamp": state.timestamp.isoformat()
            })
            
        # If empty, return current state as a "dream"
        if not dreams:
             dreams.append({
                "id": "current_state",
                "type": "active_awareness",
                "description": META_SYSTEM.current_cognitive_state.current_thought,
                "duration_minutes": 0,
                "insights_gained": META_SYSTEM.current_cognitive_state.cognitive_depth,
                "timestamp": datetime.now().isoformat()
            })

        return {
            "dreams": dreams,
            "total_dreams": len(dreams),
            "limit": limit,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo sueños de consciencia: {str(e)}",
        )
