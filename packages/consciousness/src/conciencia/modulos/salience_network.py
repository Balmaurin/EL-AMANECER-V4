# -*- coding: utf-8 -*-
"""
SALIENCE NETWORK - Red de Detección de Saliencia
===============================================

Implementación real de la Salience Network basada en neurociencia.
Detecta qué eventos son importantes y requieren atención.

Componentes clave:
- Anterior Insula - Interoception y awareness
- Anterior Cingulate Cortex (ACC) - Conflict monitoring
- Amygdala - Emotional salience

Funciones:
- Detectar eventos salientes
- Cambiar entre DMN y Task-Positive Network
- Orquestar atención hacia lo relevante
- Generar "surprise" signals
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


@dataclass
class SalientEvent:
    """Representa un evento detectado como saliente"""
    event_id: str
    content: Any
    saliency_score: float  # 0-1
    surprise_level: float  # 0-1, qué tan inesperado
    urgency: float  # 0-1
    sources: List[str]  # Qué componentes detectaron saliencia
    action_required: bool
    timestamp: datetime = field(default_factory=datetime.now)


class SalienceNetwork:
    """
    SALIENCE NETWORK - Detector de Importancia
    
    Orquesta cambios de estado mental:
    - De DMN (mind-wandering) a atención externa
    - De tarea A a tarea B más urgente
    - De procesamiento superficial a profundo
    
    Basado en:
    - Seeley et al. (2007) - Descubrimiento de Salience Network
    - Menon & Uddin (2010) - Switching mechanism
    - Craig (2009) - Insula y awareness
    """
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.creation_time = datetime.now()
        
        # Componentes de la red
        self.anterior_insula_activation = 0.0  # Interoceptive awareness
        self.anterior_cingulate_activation = 0.0  # Conflict/error detection
        self.amygdala_activation = 0.0  # Emotional salience
        
        # Estado de la red
        self.overall_salience = 0.0
        self.detection_threshold = 0.6  # Umbral para considerar algo saliente
        
        # Eventos detectados
        self.salient_events: List[SalientEvent] = []
        self.current_salient_event: Optional[SalientEvent] = None
        
        # Modelo predictivo (para surprise)
        self.predictions: Dict[str, float] = {}  # contexto -> valor esperado
        
        # Métricas
        self.total_detections = 0
        self.false_alarms = 0  # Eventos no tan salientes como parecían
        self.network_switches = 0  # Cuántas veces ordenó cambiar atención
        
        print(f"🎯 SALIENCE NETWORK {system_id} INICIALIZADA")
        print(f"   🔍 Detección de saliencia: activa")
        print(f"   ⚡ Threshold: {self.detection_threshold:.1%}")
    
    def detect_salient_events(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Optional[SalientEvent]:
        """
        Detecta eventos salientes en inputs
        
        Args:
            inputs: Señales sensoriales/cognitivas actuales
            context: Contexto (estado emocional, tareas, expectativas)
            
        Returns:
            SalientEvent si detecta algo importante, None si no
        """
        # === PASO 1: CALCULAR SALIENCIA DESDE MÚLTIPLES FUENTES ===
        
        # 1a. Insula: Saliencia interoceptiva (cambios corporales)
        interoceptive_salience = self._detect_interoceptive_salience(inputs, context)
        
        # 1b. ACC: Detección de conflicto/error
        conflict_salience = self._detect_conflict(inputs, context)
        
        # 1c. Amygdala: Saliencia emocional
        emotional_salience = self._detect_emotional_salience(inputs, context)
        
        # 1d. Surprise: Qué tan inesperado (prediction error)
        surprise = self._calculate_surprise(inputs, context)
        
        # === PASO 2: INTEGRAR FUENTES DE SALIENCIA ===
        sources_active = []
        salience_components = {}
        
        if interoceptive_salience > 0.3:
            sources_active.append('insula')
            salience_components['interoceptive'] = interoceptive_salience
        
        if conflict_salience > 0.3:
            sources_active.append('ACC')
            salience_components['conflict'] = conflict_salience
        
        if emotional_salience > 0.3:
            sources_active.append('amygdala')
            salience_components['emotional'] = emotional_salience
        
        if surprise > 0.4:
            sources_active.append('prediction_error')
            salience_components['surprise'] = surprise
        
        # Actualizar activaciones de componentes
        self.anterior_insula_activation = interoceptive_salience
        self.anterior_cingulate_activation = conflict_salience
        self.amygdala_activation = emotional_salience
        
        # Saliencia global ponderada
        self.overall_salience = (
            interoceptive_salience * 0.25 +
            conflict_salience * 0.30 +
            emotional_salience * 0.25 +
            surprise * 0.20
        )
        
        # === PASO 3: DECIDIR SI ES SALIENTE ===
        if self.overall_salience >= self.detection_threshold or len(sources_active) >= 2:
            # ¡EVENTO SALIENTE DETECTADO!
            self.total_detections += 1
            
            # Determinar urgencia
            urgency = self._calculate_urgency(salience_components, inputs)
            
            # Determinar si requiere acción
            action_required = urgency > 0.7 or emotional_salience > 0.8
            
            # Crear evento
            event = SalientEvent(
                event_id=f"salient_{self.total_detections}",
                content=inputs,
                saliency_score=self.overall_salience,
                surprise_level=surprise,
                urgency=urgency,
                sources=sources_active,
                action_required=action_required
            )
            
            self.salient_events.append(event)
            self.current_salient_event = event
            
            # Limitar historia
            if len(self.salient_events) > 100:
                self.salient_events = self.salient_events[-50:]
            
            return event
        
        return None
    
    def _detect_interoceptive_salience(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Detecta saliencia interoceptiva (cambios en estado corporal)
        
        Insula monitorea: dolor, temperatura, hambre, arousal, etc.
        """
        salience = 0.0
        
        # Cambios en arousal
        if 'arousal' in context:
            arousal_change = abs(context.get('arousal', 0.5) - context.get('previous_arousal', 0.5))
            salience += arousal_change * 0.5
        
        # Dolor o disconfort
        if 'pain' in inputs or 'discomfort' in inputs:
            salience += 0.8
        
        # Hambre/sed
        if 'hunger' in inputs or 'thirst' in inputs:
            salience += 0.6
        
        # Cambios en frecuencia cardíaca (simulated)
        if 'heart_rate_change' in inputs:
            salience += abs(inputs['heart_rate_change']) * 0.4
        
        return np.clip(salience, 0.0, 1.0)
    
    def _detect_conflict(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Detecta conflictos o errores (ACC specialization)
        
        ACC se activa cuando hay:
        - Múltiples respuestas competitivas
        - Errores detectados
        - Incertidumbre alta
        """
        conflict = 0.0
        
        # Múltiples señales contradictorias
        if isinstance(inputs, dict):
            values = [v for v in inputs.values() if isinstance(v, (int, float))]
            if len(values) >= 2:
                variance = np.var(values)
                conflict += min(1.0, variance * 2.0)
        
        # Error explícito
        if inputs.get('error_detected', False):
            conflict += 0.9
        
        # Incertidumbre
        if 'uncertainty' in inputs:
            conflict += inputs['uncertainty'] * 0.6
        
        # Conflicto con expectativa
        if 'expected' in context and 'actual' in inputs:
            diff = abs(context['expected'] - inputs.get('actual', 0))
            conflict += diff * 0.7
        
        return np.clip(conflict, 0.0, 1.0)
    
    def _detect_emotional_salience(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Detecta saliencia emocional (Amygdala specialization)
        
        Amygdala responde a:
        - Amenazas
        - Recompensas grandes
        - Eventos emocionalmente intensos
        """
        emotional = 0.0
        
        # Valencia emocional extrema
        if 'emotional_valence' in inputs:
            emotional += abs(inputs['emotional_valence']) * 0.8
        
        # Amenaza
        if inputs.get('threat_detected', False):
            emotional += 1.0
        
        # Recompensa grande
        if 'reward' in inputs and inputs['reward'] > 0.7:
            emotional += 0.8
        
        # Novedad emocional
        if inputs.get('novel_emotional_event', False):
            emotional += 0.7
        
        return np.clip(emotional, 0.0, 1.0)
    
    def _calculate_surprise(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Calcula nivel de sorpresa (prediction error)
        
        Surprise = |actual - expected|
        """
        surprise = 0.0
        
        # Si tenemos predicciones previas
        context_key = context.get('situation', 'default')
        
        if context_key in self.predictions:
            expected = self.predictions[context_key]
            
            # Calcular actual (simplificado)
            if isinstance(inputs, dict):
                actual_values = [v for v in inputs.values() if isinstance(v, (int, float))]
                if actual_values:
                    actual = np.mean(actual_values)
                    surprise = abs(actual - expected)
        
        # Novedad explícita
        if inputs.get('novelty', 0) > 0.6:
            surprise = max(surprise, inputs['novelty'])
        
        # Actualizar predicción para próxima vez
        if isinstance(inputs, dict):
            actual_values = [v for v in inputs.values() if isinstance(v, (int, float))]
            if actual_values:
                self.predictions[context_key] = np.mean(actual_values)
        
        return np.clip(surprise, 0.0, 1.0)
    
    def _calculate_urgency(self, salience_components: Dict[str, float], inputs: Dict[str, Any]) -> float:
        """Calcula urgencia del evento saliente"""
        urgency = 0.0
        
        # Urgency basada en componentes
        if 'emotional' in salience_components:
            urgency += salience_components['emotional'] * 0.4
        
        if 'conflict' in salience_components:
            urgency += salience_components['conflict'] * 0.3
        
        if 'surprise' in salience_components:
            urgency += salience_components['surprise'] * 0.3
        
        # Urgency explícita en input
        if 'urgency' in inputs:
            urgency = max(urgency, inputs['urgency'])
        
        return np.clip(urgency, 0.0, 1.0)
    
    def trigger_network_switch(self, from_network: str, to_network: str, reason: str) -> Dict[str, Any]:
        """
        Orquesta cambio de red neural activa
        
        Salience Network actúa como "switch operator":
        - DMN → Task-Positive (cuando detecta evento saliente)
        - Task A → Task B (cuando B más urgente)
        
        Args:
            from_network: Red actual (ej: 'DMN')
            to_network: Red objetivo (ej: 'task_positive')
            reason: Razón del switch
            
        Returns:
            Comando de switch
        """
        self.network_switches += 1
        
        switch_command = {
            'action': 'switch_network',
            'from': from_network,
            'to': to_network,
            'reason': reason,
            'salience': self.overall_salience,
            'timestamp': datetime.now(),
            'switch_number': self.network_switches
        }
        
        print(f"   🎯 Salience Network: Switch {from_network} → {to_network} ({reason})")
        
        return switch_command
    
    def get_salience_state(self) -> Dict[str, Any]:
        """Retorna estado completo de la Salience Network"""
        return {
            'overall_salience': self.overall_salience,
            'components': {
                'anterior_insula': self.anterior_insula_activation,
                'anterior_cingulate': self.anterior_cingulate_activation,
                'amygdala': self.amygdala_activation
            },
            'current_event': self.current_salient_event.__dict__ if self.current_salient_event else None,
            'total_detections': self.total_detections,
            'network_switches': self.network_switches,
            'recent_events': [
                e.__dict__ for e in self.salient_events[-5:]
            ],
            'predictions': self.predictions
        }
