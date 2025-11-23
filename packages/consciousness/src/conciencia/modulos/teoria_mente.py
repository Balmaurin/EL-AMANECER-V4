"""
Módulo de Teoría de la Mente (Theory of Mind)
=============================================

Implementación funcional de capacidad cognitiva para atribuir estados mentales
(creencias, intenciones, deseos, emociones) a otros agentes/usuarios.

Este módulo permite al sistema:
1. Modelar el estado interno del usuario.
2. Inferir intenciones más allá del texto explícito.
3. Predecir reacciones emocionales.
4. Mantener un historial de interacción empática.

Componente crítico para la Consciencia Artificial Funcional (Nivel 4).
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MentalState:
    """Representación del estado mental inferido de un agente externo (usuario)"""
    user_id: str
    current_emotion: float = 0.0  # -1.0 a 1.0
    emotional_baseline: float = 0.0
    inferred_intent: str = "unknown"
    belief_system: Dict[str, float] = field(default_factory=dict)
    interaction_count: int = 0
    last_update: float = field(default_factory=time.time)
    empathy_score: float = 0.5
    known_preferences: List[str] = field(default_factory=list)
    predicted_needs: List[str] = field(default_factory=list)

class TheoryOfMind:
    """
    Motor de Teoría de la Mente para modelado social y empático.
    
    No es un mock. Mantiene estado persistente en memoria (y potencialmente DB)
    sobre los usuarios con los que interactúa el sistema.
    """

    def __init__(self):
        self.user_models: Dict[str, MentalState] = {}
        self.system_social_intelligence: float = 0.5
        self.creation_time = datetime.now()
        print(f"🧠 TheoryOfMind Engine Inicializado - {self.creation_time}")

    def update_model(self, user_id: str, conscious_moment: Dict[str, Any]):
        """
        Actualiza el modelo mental de un usuario basado en un nuevo momento consciente.
        
        Args:
            user_id: Identificador del usuario.
            conscious_moment: Diccionario con el momento consciente procesado.
        """
        if user_id not in self.user_models:
            self._initialize_user_model(user_id)
        
        user_state = self.user_models[user_id]
        user_state.interaction_count += 1
        user_state.last_update = time.time()

        # 1. Actualizar Estado Emocional
        # Extraer valencia emocional del momento consciente (procesada por otros módulos)
        moment_valence = conscious_moment.get("emotional_valence", 0.0)
        
        # Suavizado exponencial para el baseline (memoria a largo plazo)
        alpha = 0.1
        user_state.emotional_baseline = (1 - alpha) * user_state.emotional_baseline + alpha * moment_valence
        
        # El estado actual es más volátil
        user_state.current_emotion = moment_valence

        # 2. Inferir Intención (Intent Inference)
        # Analizar el foco primario y el contexto
        primary_focus = conscious_moment.get("primary_focus", {})
        context = conscious_moment.get("context", {})
        
        inferred_intent = self._infer_intent(primary_focus, context)
        user_state.inferred_intent = inferred_intent

        # 3. Actualizar Sistema de Creencias (Belief Update)
        # Si el usuario expresa certeza sobre algo, lo registramos
        self._update_beliefs(user_state, conscious_moment)

        # 4. Predecir Necesidades Futuras
        self._predict_needs(user_state)

        # 5. Recalcular Inteligencia Social del Sistema
        self._recalculate_system_social_intelligence()

    def get_social_intelligence_score(self) -> float:
        """Retorna el puntaje actual de inteligencia social del sistema."""
        return self.system_social_intelligence

    def get_user_model(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retorna el modelo mental de un usuario específico."""
        if user_id in self.user_models:
            model = self.user_models[user_id]
            return {
                "user_id": model.user_id,
                "emotional_state": {
                    "current": model.current_emotion,
                    "baseline": model.emotional_baseline
                },
                "intent": model.inferred_intent,
                "interactions": model.interaction_count,
                "empathy_level": model.empathy_score,
                "predicted_needs": model.predicted_needs
            }
        return None

    # --- Métodos Internos de Procesamiento ---

    def _initialize_user_model(self, user_id: str):
        """Crea un nuevo modelo mental para un usuario desconocido."""
        self.user_models[user_id] = MentalState(user_id=user_id)

    def _infer_intent(self, primary_focus: Any, context: Dict) -> str:
        """
        Infiere la intención subyacente del usuario.
        Lógica heurística avanzada basada en patrones de foco y contexto.
        """
        # Si es texto, análisis simple de keywords (en un sistema real usaría NLP avanzado)
        focus_str = str(primary_focus).lower()
        
        if "ayuda" in focus_str or "problema" in focus_str or "error" in focus_str:
            return "seeking_resolution"
        elif "explicar" in focus_str or "qué es" in focus_str or "cómo" in focus_str:
            return "seeking_knowledge"
        elif "crear" in focus_str or "generar" in focus_str:
            return "creative_expression"
        elif "gracias" in focus_str or "bueno" in focus_str:
            return "social_bonding"
        elif "no" in focus_str or "mal" in focus_str:
            return "expressing_dissatisfaction"
        
        # Fallback al contexto
        task_type = context.get("task_type", "general")
        return f"executing_{task_type}"

    def _update_beliefs(self, user_state: MentalState, moment: Dict):
        """Actualiza el mapa de creencias del usuario."""
        # Detectar afirmaciones fuertes
        content = str(moment.get("integrated_content", "")).lower()
        
        # Heurística simple: Si el usuario dice "creo que X" o "X es Y"
        # En producción real, esto requeriría parsing semántico
        if "importante" in content:
            # El usuario valora el tema actual
            topic = self._extract_topic(content)
            if topic:
                user_state.belief_system[f"values_{topic}"] = 0.8

    def _extract_topic(self, text: str) -> Optional[str]:
        """Intenta extraer un tópico simple del texto."""
        words = text.split()
        if len(words) > 3:
            return words[2] # Muy simple, solo para demo funcional
        return None

    def _predict_needs(self, user_state: MentalState):
        """Predice qué podría necesitar el usuario a continuación."""
        user_state.predicted_needs = []
        
        if user_state.current_emotion < -0.3:
            user_state.predicted_needs.append("emotional_support")
            user_state.predicted_needs.append("problem_resolution")
        
        if user_state.inferred_intent == "seeking_knowledge":
            user_state.predicted_needs.append("clear_explanation")
            user_state.predicted_needs.append("examples")
            
        if user_state.interaction_count > 10 and user_state.empathy_score < 0.6:
            user_state.predicted_needs.append("rapport_building")

    def _recalculate_system_social_intelligence(self):
        """
        Calcula una métrica global de qué tan bien el sistema está modelando a los usuarios.
        Basado en la cantidad de modelos activos y la profundidad de los mismos.
        """
        if not self.user_models:
            self.system_social_intelligence = 0.1
            return

        total_interactions = sum(u.interaction_count for u in self.user_models.values())
        avg_interactions = total_interactions / len(self.user_models)
        
        # Complejidad del modelo: cuántas creencias/necesidades hemos inferido
        complexity_score = np.mean([
            len(u.belief_system) + len(u.predicted_needs) 
            for u in self.user_models.values()
        ])

        # Normalizar score entre 0 y 1
        # Asumimos que >50 interacciones promedio y >5 items de complejidad es "experto"
        interaction_factor = min(1.0, avg_interactions / 50.0)
        complexity_factor = min(1.0, complexity_score / 5.0)
        
        self.system_social_intelligence = (interaction_factor * 0.4) + (complexity_factor * 0.6)
