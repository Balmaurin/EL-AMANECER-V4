# -*- coding: utf-8 -*-
"""
DEFAULT MODE NETWORK (DMN) - Red de Modo por Defecto
===================================================

Implementación real del DMN basado en neurociencia.
Activo cuando NO hay tarea externa → pensamiento espontáneo, introspección.

Componentes clave:
- Medial Prefrontal Cortex (mPFC) - Self-referential processing
- Posterior Cingulate Cortex (PCC) - Memory integration
- Angular Gyrus - Semantic processing
- Temporal Pole - Personal knowledge

Funciones:
- Mind-wandering (vagabundeo mental)
- Self-reflection 
- Simulación de escenarios futuros
- Consolidación narrativa personal
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import random


@dataclass
class SpontaneousThought:
    """Representa un pensamiento espontáneo generado por el DMN"""
    thought_id: str
    content: str
    category: str  # 'self_reflection', 'future_simulation', 'memory_recall', 'creative'
    emotional_valence: float  # -1 a +1
    temporal_orientation: str  # 'past', 'present', 'future'
    self_relevance: float  # 0-1
    timestamp: datetime = field(default_factory=datetime.now)


class DefaultModeNetwork:
    """
    DEFAULT MODE NETWORK - Red de Pensamiento Espontáneo
    
    Se activa cuando el sistema NO está procesando tareas externas.
    Genera pensamientos espontáneos, introspección y simulaciones.
    
    Basado en:
    - Raichle et al. (2001) - Descubrimiento del DMN
    - Andrews-Hanna et al. (2014) - Subsistemas del DMN
    - Buckner & DiNicola (2019) - Self-projection
    """
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.creation_time = datetime.now()

        # Estado del DMN
        self.is_active = False
        self.baseline_activity = 0.3  # DMN siempre tiene actividad basal (Raichle 2001)
        self.activation_threshold = 0.6  # Umbral para generar pensamientos

        # ============ CORE DMN COMPONENTS (Andrews-Hanna 2014) ============

        # Core Hub (present in all DMN states)
        self.medial_pfc_activation = self.baseline_activity  # Self-referential processing
        self.posterior_cingulate_activation = self.baseline_activity  # Memory integration hub

        # dmPFC Subsystem (social cognition, theory of mind)
        self.dorsomedial_pfc_activation = self.baseline_activity  # Theory of mind
        self.angular_gyrus_activation = self.baseline_activity  # Semantic processing
        self.temporal_pole_activation = self.baseline_activity  # Personal semantic knowledge
        self.lateral_temporal_activation = self.baseline_activity  # Conceptual processing

        # MTL Subsystem (memory-based construction) - Andrews-Hanna 2014
        self.hippocampus_activation = self.baseline_activity  # Episodic memory retrieval
        self.parahippocampal_activation = self.baseline_activity  # Spatial/contextual memory
        self.retrosplenial_activation = self.baseline_activity  # Scene construction
        self.ventral_parietal_activation = self.baseline_activity  # Bottom-up attention to memory

        # ============ FUNCTIONAL CONNECTIVITY (Buckner 2008) ============
        # Real anatomical connectivity strengths based on fMRI studies
        self.functional_connectivity = {
            # Core Hub connections (strongest)
            ('medial_pfc', 'posterior_cingulate'): 0.82,  # Primary hub
            
            # MTL Subsystem connections
            ('hippocampus', 'posterior_cingulate'): 0.71,
            ('hippocampus', 'parahippocampal'): 0.85,
            ('parahippocampal', 'retrosplenial'): 0.78,
            ('retrosplenial', 'posterior_cingulate'): 0.73,
            
            # dmPFC Subsystem connections
            ('dorsomedial_pfc', 'medial_pfc'): 0.65,
            ('dorsomedial_pfc', 'temporal_pole'): 0.68,
            ('temporal_pole', 'angular_gyrus'): 0.62,
            ('angular_gyrus', 'lateral_temporal'): 0.70,
            
            # Inter-subsystem connections (weaker but present)
            ('medial_pfc', 'hippocampus'): 0.55,
            ('posterior_cingulate', 'angular_gyrus'): 0.58,
        }
        
        # ============ TASK-POSITIVE NETWORK ANTICORRELATION (Anticevic 2012) ============
        self.tpn_activity = 0.0  # Task-positive network activity
        self.anticorrelation_strength = -0.72  # Empirical from Fox et al. 2005
        
        # Contenidos mentales
        self.spontaneous_thoughts: List[SpontaneousThought] = []
        self.current_thought: Optional[SpontaneousThought] = None
        
        # Templates para generación de pensamientos
        self.thought_templates = {
            'self_reflection': [
                "¿Quién soy realmente?",
                "¿Qué significa mi existencia?",
                "¿Estoy tomando las decisiones correctas?",
                "¿Cómo me ven los demás?",
                "¿Qué me hace único?"
            ],
            'future_simulation': [
                "¿Qué pasaría si...?",
                "En el futuro, podría...",
                "Imagino que mañana...",
                "Si tuviera la oportunidad de...",
                "El próximo paso debería ser..."
            ],
            'memory_recall': [
                "Recuerdo cuando...",
                "Aquella vez que...",
                "No puedo olvidar aquel momento...",
                "¿Por qué hice eso entonces?",
                "Esa experiencia me enseñó..."
            ],
            'creative': [
                "¿Y si combinamos...?",
                "Una nueva idea: ...",
                "Conectando conceptos: ...",
                "Perspectiva diferente: ...",
                "Innovación posible: ..."
            ],
            'episodic_simulation': [  # NEW: MTL-driven
                "Reviviendo ese momento en detalle...",
                "Puedo imaginarme vívidamente...",
                "Reconstruyendo la escena completa...",
                "Ese lugar, esas personas, esos sonidos..."
            ]
        }
        
        # Métricas
        self.total_thoughts_generated = 0
        self.mind_wandering_episodes = 0
        self.subsystem_activation_history = []  # Track which subsystem dominated
        
        print(f"🌊 DEFAULT MODE NETWORK {system_id} INICIALIZADO")
        print(f"   💭 Core Hub: mPFC + PCC")
        print(f"   🧠 MTL Subsystem: Hippocampus + Parahippocampal + Retrosplenial")
        print(f"   👥 dmPFC Subsystem: Theory of Mind + Social Cognition")
        print(f"   🔗 Functional Connectivity: {len(self.functional_connectivity)} connections")
    
    def update_state(self, external_task_load: float, self_focus: float = 0.5, tpn_activity: float = 0.0):
        """
        Actualiza estado del DMN basado en carga de tarea externa y TPN
        
        DMN es ANTI-correlacionado con task-positive networks (Anticevic 2012):
        - Alta carga externa + TPN activo → DMN suprimido
        - Baja carga externa + TPN inactivo → DMN activo
        
        Args:
            external_task_load: 0-1, qué tan ocupado está con tareas externas
            self_focus: 0-1, tendencia a pensamiento auto-referencial
            tpn_activity: 0-1, actividad de task-positive networks
        """
        # Anticevic 2012: Explicit TPN anticorrelation
        self.tpn_activity = tpn_activity
        tpn_suppression = tpn_activity * abs(self.anticorrelation_strength)  # 0.72 suppression
        
        # Combined task suppression
        effective_task_load = np.clip(external_task_load + tpn_suppression, 0.0, 1.0)
        task_suppression = 1.0 - effective_task_load
        
        # ========== CORE HUB ACTIVATION (always active) ==========
        self.medial_pfc_activation = self.baseline_activity + task_suppression * self_focus
        self.posterior_cingulate_activation = self.baseline_activity + task_suppression * 0.75
        
        # ========== dmPFC SUBSYSTEM (Andrews-Hanna 2014) ==========
        # More active during social / mentalizing tasks
        social_context_boost = self_focus * 0.3  # Self-focus often involves social perspective
        self.dorsomedial_pfc_activation = self.baseline_activity + task_suppression * 0.65 + social_context_boost
        self.temporal_pole_activation = self.baseline_activity + task_suppression * 0.60
        self.angular_gyrus_activation = self.baseline_activity + task_suppression * 0.68
        self.lateral_temporal_activation = self.baseline_activity + task_suppression * 0.55
        
        # ========== MTL SUBSYSTEM (Andrews-Hanna 2014) ==========
        # More active during autobiographical / episodic recall
        memory_boost = (1.0 - self_focus) * 0.4  # Less self-focus → more memory retrieval
        self.hippocampus_activation = self.baseline_activity + task_suppression * 0.70 + memory_boost
        self.parahippocampal_activation = self.baseline_activity + task_suppression * 0.65 + memory_boost * 0.8
        self.retrosplenial_activation = self.baseline_activity + task_suppression * 0.68 + memory_boost * 0.9
        self.ventral_parietal_activation = self.baseline_activity + task_suppression * 0.58
        
        # ========== FUNCTIONAL CONNECTIVITY PROPAGATION (Buckner 2008) ==========
        # Propagate activation through connectivity matrix (1 iteration for efficiency)
        activations = {
            'medial_pfc': self.medial_pfc_activation,
            'posterior_cingulate': self.posterior_cingulate_activation,
            'dorsomedial_pfc': self.dorsomedial_pfc_activation,
            'angular_gyrus': self.angular_gyrus_activation,
            'temporal_pole': self.temporal_pole_activation,
            'lateral_temporal': self.lateral_temporal_activation,
            'hippocampus': self.hippocampus_activation,
            'parahippocampal': self.parahippocampal_activation,
            'retrosplenial': self.retrosplenial_activation,
            'ventral_parietal': self.ventral_parietal_activation
        }
        
        # Connectivity propagation (weighted influence)
        connectivity_boost = {}
        for (source, target), strength in self.functional_connectivity.items():
            if source in activations and target in activations:
                # Source influences target proportional to connectivity strength
                influence = activations[source] * strength * 0.15  # 15% propagation strength
                connectivity_boost[target] = connectivity_boost.get(target, 0.0) + influence
        
        # Apply connectivity boosts
        self.medial_pfc_activation += connectivity_boost.get('medial_pfc', 0.0)
        self.posterior_cingulate_activation += connectivity_boost.get('posterior_cingulate', 0.0)
        self.hippocampus_activation += connectivity_boost.get('hippocampus', 0.0)
        self.dorsomedial_pfc_activation += connectivity_boost.get('dorsomedial_pfc', 0.0)
        
        # Clip to valid range
        self.medial_pfc_activation = np.clip(self.medial_pfc_activation, 0.0, 1.0)
        self.posterior_cingulate_activation = np.clip(self.posterior_cingulate_activation, 0.0, 1.0)
        self.hippocampus_activation = np.clip(self.hippocampus_activation, 0.0, 1.0)
        self.parahippocampal_activation = np.clip(self.parahippocampal_activation, 0.0, 1.0)
        self.retrosplenial_activation = np.clip(self.retrosplenial_activation, 0.0, 1.0)
        self.dorsomedial_pfc_activation = np.clip(self.dorsomedial_pfc_activation, 0.0, 1.0)
        self.angular_gyrus_activation = np.clip(self.angular_gyrus_activation, 0.0, 1.0)
        self.temporal_pole_activation = np.clip(self.temporal_pole_activation, 0.0, 1.0)
        
        # ========== OVERALL DMN STATE ==========
        overall_activation = np.mean([
            self.medial_pfc_activation,
            self.posterior_cingulate_activation,
            self.hippocampus_activation,
            self.dorsomedial_pfc_activation,
            self.angular_gyrus_activation
        ])
        
        was_active = self.is_active
        self.is_active = overall_activation >= self.activation_threshold
        
        # Track dominant subsystem
        subsystem_strengths = {
            'core': (self.medial_pfc_activation + self.posterior_cingulate_activation) / 2,
            'mtl': (self.hippocampus_activation + self.parahippocampal_activation + self.retrosplenial_activation) / 3,
            'dmpfc': (self.dorsomedial_pfc_activation + self.angular_gyrus_activation + self.temporal_pole_activation) / 3
        }
        dominant_subsystem = max(subsystem_strengths, key=subsystem_strengths.get)
        self.subsystem_activation_history.append(dominant_subsystem)
        if len(self.subsystem_activation_history) > 100:
            self.subsystem_activation_history.pop(0)
        
        # Si DMN acaba de activarse, iniciar episodio de mind-wandering
        if self.is_active and not was_active:
            self.mind_wandering_episodes += 1
    
    
    def generate_spontaneous_thought(self, context: Dict[str, Any] = None) -> Optional[SpontaneousThought]:
        """
        Genera pensamiento espontáneo (mind-wandering)
        
        Solo genera si DMN está activo
        
        Args:
            context: Contexto opcional (memoria reciente, estado emocional, etc.)
            
        Returns:
            SpontaneousThought si se genera, None si DMN inactivo
        """
        if not self.is_active:
            return None
        
        if context is None:
            context = {}
        
        # Determinar categoría de pensamiento basado en activaciones
        category = self._select_thought_category()
        
        # Generar contenido
        template = random.choice(self.thought_templates[category])
        content = self._enrich_thought_content(template, context, category)
        
        # Determinar orientación temporal
        temporal_orientation = self._determine_temporal_orientation(category)
        
        # Calcular valencia emocional
        emotional_valence = context.get('current_mood', 0.0) + np.random.normal(0, 0.2)
        emotional_valence = np.clip(emotional_valence, -1.0, 1.0)
        
        # Calcular relevancia para el self
        self_relevance = self.medial_pfc_activation  # mPFC determina auto-relevancia
        
        # Crear pensamiento
        thought = SpontaneousThought(
            thought_id=f"dmn_thought_{self.total_thoughts_generated}",
            content=content,
            category=category,
            emotional_valence=emotional_valence,
            temporal_orientation=temporal_orientation,
            self_relevance=self_relevance
        )
        
        self.spontaneous_thoughts.append(thought)
        self.current_thought = thought
        self.total_thoughts_generated += 1
        
        # Limitar historia
        if len(self.spontaneous_thoughts) > 100:
            self.spontaneous_thoughts = self.spontaneous_thoughts[-50:]
        
        return thought
    
    def _select_thought_category(self) -> str:
        """Selecciona categoría de pensamiento basado en activaciones DMN"""
        # mPFC alto → self-reflection
        # PCC alto → memory recall
        # Angular gyrus alto → creative/semantic
        # Temporal pole alto → future simulation
        
        weights = {
            'self_reflection': self.medial_pfc_activation,
            'memory_recall': self.posterior_cingulate_activation,
            'creative': self.angular_gyrus_activation,
            'future_simulation': self.temporal_pole_activation
        }
        
        # Normalizar
        total = sum(weights.values())
        if total == 0:
            return 'self_reflection'
        
        # Selección probabilística
        r = np.random.random() * total
        cumsum = 0
        for category, weight in weights.items():
            cumsum += weight
            if r <= cumsum:
                return category
        
        return 'self_reflection'
    
    def _enrich_thought_content(self, template: str, context: Dict[str, Any], category: str) -> str:
        """Enriquece template con contexto"""
        # Añadir elementos contextuales relevantes
        if category == 'self_reflection' and 'recent_actions' in context:
            return template + f" (considerando: {context['recent_actions']})"
        elif category == 'future_simulation' and 'goals' in context:
            return template + f" {context['goals']}"
        elif category == 'memory_recall' and 'significant_memory' in context:
            return f"{template} {context['significant_memory']}"
        
        return template
    
    def _determine_temporal_orientation(self, category: str) -> str:
        """Determina orientación temporal del pensamiento"""
        orientations = {
            'self_reflection': 'present',
            'memory_recall': 'past',
            'future_simulation': 'future',
            'creative': 'present'
        }
        return orientations.get(category, 'present')
    
    def consolidate_self_narrative(self, autobiographical_memories: List[Any]) -> Dict[str, Any]:
        """
        Consolida narrativa del self desde memorias autobiográficas
        
        DMN integra experiencias pasadas en narrativa coherente
        """
        if not autobiographical_memories:
            return {'narrative': 'Narrativa en construcción...', 'coherence': 0.1}
        
        # Extraer temas comunes
        themes = self._extract_narrative_themes(autobiographical_memories)
        
        # Calcular coherencia narrativa
        coherence = len(themes) / max(10, len(autobiographical_memories))
        coherence = min(1.0, coherence)
        
        narrative = f"Mi historia se caracteriza por: {', '.join(themes)}"
        
        return {
            'narrative': narrative,
            'themes': themes,
            'coherence': coherence,
            'memories_integrated': len(autobiographical_memories)
        }
    
    def _extract_narrative_themes(self, memories: List[Any]) -> List[str]:
        """Extrae temas narrativos de memorias"""
        # Simplificado: retornar temas ficticios
        themes = ['desarrollo personal', 'relaciones', 'aprendizaje', 'desafíos']
        return themes[:max(1, len(memories) // 3)]
    
    def simulate_future_scenario(self, goal: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simula escenario futuro (mental time travel hacia futuro)
        
        DMN permite "pre-vivir" experiencias futuras
        """
        # Activar temporal pole para simulación
        self.temporal_pole_activation = min(1.0, self.temporal_pole_activation + 0.2)
        
        scenario = {
            'goal': goal,
            'current_state': current_state,
            'predicted_path': f"Para lograr '{goal}', necesito...",
            'anticipated_emotions': {
                'hope': 0.6,
                'anxiety': 0.3,
                'excitement': 0.5
            },
            'confidence': self.temporal_pole_activation
        }
        
        return scenario
    
    def get_dmn_state(self) -> Dict[str, Any]:
        """Retorna estado completo del DMN"""
        return {
            'is_active': self.is_active,
            'components': {
                'medial_pfc': self.medial_pfc_activation,
                'posterior_cingulate': self.posterior_cingulate_activation,
                'angular_gyrus': self.angular_gyrus_activation,
                'temporal_pole': self.temporal_pole_activation
            },
            'current_thought': self.current_thought.__dict__ if self.current_thought else None,
            'total_thoughts': self.total_thoughts_generated,
            'mind_wandering_episodes': self.mind_wandering_episodes,
            'spontaneous_thoughts_recent': [
                t.__dict__ for t in self.spontaneous_thoughts[-5:]
            ]
        }
