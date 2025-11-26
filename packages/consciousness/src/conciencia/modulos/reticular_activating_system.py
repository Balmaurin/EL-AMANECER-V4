# -*- coding: utf-8 -*-
"""
RETICULAR ACTIVATING SYSTEM (RAS) - Control de Arousal Global
============================================================

Implementación real del RAS como modulador del nivel de despertar cerebral.
Controla el arousal global que afecta a TODO el cerebro.

Funciones clave:
- Control de nivel vigilia/sueño
- Modulación de arousal por estímulos
- Broadcasting de nivel de activación
- Transiciones de estado consciente
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime
import numpy as np


class AscendingPathway:
    """
    Vía ascendente del RAS
    Cada neurotransmisor tiene su propio pathway
    """
    def __init__(self, neurotransmitter: str, baseline: float = 0.5):
        self.neurotransmitter = neurotransmitter
        self.activity_level = baseline
        self.baseline = baseline
        self.modulation_strength = 1.0
    
    def modulate(self, stimulus_intensity: float):
        """Modula actividad por estímulo"""
        delta = stimulus_intensity * self.modulation_strength * 0.1
        self.activity_level = np.clip(self.activity_level + delta, 0.0, 1.0)
    
    def decay_to_baseline(self, rate: float = 0.05):
        """Decaimiento hacia baseline"""
        delta = (self.baseline - self.activity_level) * rate
        self.activity_level += delta


class ReticularActivatingSystem:
    """
    RETICULAR ACTIVATING SYSTEM (RAS) - Real Implementation
    
    Sistema que controla el nivel global de despertar del cerebro.
    Determina si estás dormido, despierto, alerta o hiperalerta.
    
    Basado en:
    - Formación reticular del tronco encefálico
    - Vías ascendentes de activación
    - Neurotransmisores moduladores
    """
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.creation_time = datetime.now()
        
        # Estado de arousal global (0=deep sleep, 1=hyperarousal)
        self.global_arousal = 0.5
        
        # Vías ascendentes (cada neurotransmisor)
        self.ascending_pathways = {
            'norepinephrine': AscendingPathway('norepinephrine', baseline=0.5),  # Alerta
            'serotonin': AscendingPathway('serotonin', baseline=0.6),  # Estado de ánimo
            'dopamine': AscendingPathway('dopamine', baseline=0.5),  # Motivación
            'acetylcholine': AscendingPathway('acetylcholine', baseline=0.7),  # Atención
            'histamine': AscendingPathway('histamine', baseline=0.4),  # Vigilia
        }
        
        # Estado de consciencia
        self.consciousness_state = 'awake'  # 'deep_sleep', 'light_sleep', 'drowsy', 'awake', 'alert', 'hyperalert'
        
        # Umbralespara transiciones
        self.state_thresholds = {
            'deep_sleep': 0.0,
            'light_sleep': 0.2,
            'drowsy': 0.4,
            'awake': 0.6,
            'alert': 0.75,
            'hyperalert': 0.9
        }
        
        # Métricas
        self.arousal_history = []
        self.state_changes = 0
        
        print(f"⚡ RAS (RETICULAR ACTIVATING SYSTEM) {system_id} INICIALIZADO")
        print(f"   🌅 Estado inicial: {self.consciousness_state}")
        print(f"   📊 Arousal global: {self.global_arousal:.1%}")
    
    def process_stimulus(self, stimulus: Dict[str, Any]) -> float:
        """
        Procesa estímulo y ajusta arousal global
        
        Args:
            stimulus: Dict con información del estímulo
            
        Returns:
            Nuevo nivel de arousal
        """
        # Extraer características relevantes
        intensity = stimulus.get('intensity', 0.5)
        urgency = stimulus.get('urgency', 0.0)
        novelty = stimulus.get('novelty', 0.0)
        emotional_significance = abs(stimulus.get('emotional_valence', 0.0))
        
        # Calcular impacto en arousal
        arousal_impact = (
            intensity * 0.3 +
            urgency * 0.4 +
            novelty * 0.2 +
            emotional_significance * 0.1
        )
        
        # Modular pathways específicos
        if arousal_impact > 0.5:
            # Estímulo significativo → activar norepinefrina
            self.ascending_pathways['norepinephrine'].modulate(arousal_impact)
        
        if novelty > 0.6:
            # Novedad → activar dopamina
            self.ascending_pathways['dopamine'].modulate(novelty)
        
        if emotional_significance > 0.7:
            # Emoción fuerte → activar acetilcolina
            self.ascending_pathways['acetylcholine'].modulate(emotional_significance)
        
        # Calcular nuevo arousal global
        self._update_global_arousal()
        
        # Determinar nuevo estado de consciencia
        self._update_consciousness_state()
        
        # Guardar en historia
        self.arousal_history.append({
            'arousal': self.global_arousal,
            'state': self.consciousness_state,
            'timestamp': datetime.now()
        })
        
        # Limitar historia
        if len(self.arousal_history) > 1000:
            self.arousal_history = self.arousal_history[-500:]
        
        return self.global_arousal
    
    def _update_global_arousal(self):
        """Calcula arousal global desde pathways"""
        # Promediar actividad de pathways
        pathway_activities = [p.activity_level for p in self.ascending_pathways.values()]
        
        if pathway_activities:
            self.global_arousal = np.mean(pathway_activities)
        
        # Asegurar bounds
        self.global_arousal = np.clip(self.global_arousal, 0.0, 1.0)
    
    def _update_consciousness_state(self):
        """Determina estado de consciencia basado en arousal"""
        old_state = self.consciousness_state
        
        if self.global_arousal >= self.state_thresholds['hyperalert']:
            self.consciousness_state = 'hyperalert'
        elif self.global_arousal >= self.state_thresholds['alert']:
            self.consciousness_state = 'alert'
        elif self.global_arousal >= self.state_thresholds['awake']:
            self.consciousness_state = 'awake'
        elif self.global_arousal >= self.state_thresholds['drowsy']:
            self.consciousness_state = 'drowsy'
        elif self.global_arousal >= self.state_thresholds['light_sleep']:
            self.consciousness_state = 'light_sleep'
        else:
            self.consciousness_state = 'deep_sleep'
        
        if old_state != self.consciousness_state:
            self.state_changes += 1
            print(f"   ⚡ RAS: Transición {old_state} → {self.consciousness_state}")
    
    def decay_arousal(self, rate: float = 0.01):
        """
        Decaimiento natural de arousal (sin estímulos)
        Simula relajación gradual
        """
        for pathway in self.ascending_pathways.values():
            pathway.decay_to_baseline(rate)
        
        self._update_global_arousal()
        self._update_consciousness_state()
    
    def force_arousal_level(self, level: float, reason: str = "external"):
        """Fuerza un nivel de arousal específico"""
        self.global_arousal = np.clip(level, 0.0, 1.0)
        
        # Ajustar pathways para matching
        for pathway in self.ascending_pathways.values():
            pathway.activity_level = self.global_arousal
        
        self._update_consciousness_state()
        print(f"   ⚡ RAS: Arousal forzado a {level:.1%} ({reason})")
    
    def get_arousal_state(self) -> Dict[str, Any]:
        """Retorna estado completo del RAS"""
        return {
            'global_arousal': self.global_arousal,
            'consciousness_state': self.consciousness_state,
            'pathways': {
                name: p.activity_level 
                for name, p in self.ascending_pathways.items()
            },
            'state_changes': self.state_changes,
            'is_awake': self.consciousness_state in ['awake', 'alert', 'hyperalert']
        }
