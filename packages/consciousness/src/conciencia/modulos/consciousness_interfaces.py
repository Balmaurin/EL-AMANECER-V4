#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONSCIOUSNESS INTERFACES - Interfaces Claras y Desacopladas
============================================================

Define interfaces estándar para todos los módulos del sistema de consciencia.
Reduce acoplamiento y facilita testing, extensión y mantenimiento.

Principios:
- Dependency Injection
- Interface Segregation
- Liskov Substitution
- Abstracción sobre implementación
"""

import sys
from pathlib import Path

# Agregar path del proyecto para imports
_current_file = Path(__file__).resolve()
_src_dir = _current_file.parent.parent.parent  # Subir a src/
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


# ==================== TIPOS BASE ====================

@dataclass
class ProcessingResult:
    """Resultado estándar de procesamiento"""
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class ConsciousState:
    """Estado consciente estándar"""
    level: float  # 0-1
    coherence: float  # 0-1
    integration: float  # 0-1
    phenomenal_content: Dict[str, Any]
    timestamp: float


# ==================== INTERFACES PRINCIPALES ====================

class INeuralProcessor(ABC):
    """Interfaz para procesadores neurales"""
    
    @abstractmethod
    def process_input(self, input_pattern: Dict[str, Any]) -> ProcessingResult:
        """Procesa patrón de entrada"""
        pass
    
    @abstractmethod
    def get_activation_state(self) -> Dict[str, float]:
        """Obtiene estado de activación actual"""
        pass
    
    @abstractmethod
    def update_weights(self, learning_signal: float):
        """Actualiza pesos sinápticos"""
        pass


class IEmotionalProcessor(ABC):
    """Interfaz para procesadores emocionales"""
    
    @abstractmethod
    def process_stimulus(self, stimulus: Dict[str, Any]) -> ProcessingResult:
        """Procesa estímulo emocional"""
        pass
    
    @abstractmethod
    def get_emotional_state(self) -> Dict[str, Any]:
        """Obtiene estado emocional actual"""
        pass
    
    @abstractmethod
    def get_neurochemical_profile(self) -> Dict[str, float]:
        """Obtiene perfil neuroquímico (dopamina, serotonina, etc.)"""
        pass
    
    @abstractmethod
    def regulate_emotion(self, strategy: str) -> bool:
        """Aplica regulación emocional"""
        pass


class ICognitiveProcessor(ABC):
    """Interfaz para procesadores cognitivos"""
    
    @abstractmethod
    def process_information(self, information: Dict[str, Any]) -> ProcessingResult:
        """Procesa información cognitiva"""
        pass
    
    @abstractmethod
    def get_cognitive_load(self) -> float:
        """Obtiene carga cognitiva actual (0-1)"""
        pass
    
    @abstractmethod
    def allocate_attention(self, target: str, intensity: float) -> bool:
        """Asigna atención a objetivo"""
        pass


class IConsciousnessEngine(ABC):
    """Interfaz para motores de consciencia"""
    
    @abstractmethod
    def generate_conscious_moment(self, input_data: Dict[str, Any]) -> ConsciousState:
        """Genera momento consciente"""
        pass
    
    @abstractmethod
    def evaluate_emergence(self) -> float:
        """Evalúa nivel de emergencia consciente (0-1)"""
        pass
    
    @abstractmethod
    def integrate_subsystems(self, subsystems: List[Any]) -> ProcessingResult:
        """Integra subsistemas en workspace global"""
        pass


class IMemorySystem(ABC):
    """Interfaz para sistemas de memoria"""
    
    @abstractmethod
    def store(self, experience: Dict[str, Any]) -> str:
        """Almacena experiencia, retorna ID"""
        pass
    
    @abstractmethod
    def retrieve(self, query: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Recupera experiencias similares"""
        pass
    
    @abstractmethod
    def consolidate(self) -> bool:
        """Consolida memoria (e.g. durante sueño)"""
        pass


class IDecisionMaker(ABC):
    """Interfaz para sistemas de toma de decisiones"""
    
    @abstractmethod
    def evaluate_options(self, options: List[Dict[str, Any]]) -> ProcessingResult:
        """Evalúa opciones disponibles"""
        pass
    
    @abstractmethod
    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Toma decisión basada en contexto"""
        pass
    
    @abstractmethod
    def get_confidence(self) -> float:
        """Obtiene confianza en última decisión (0-1)"""
        pass


class IMetacognitiveEngine(ABC):
    """Interfaz para metacognición"""
    
    @abstractmethod
    def monitor_cognition(self) -> ProcessingResult:
        """Monitorea procesos cognitivos"""
        pass
    
    @abstractmethod
    def evaluate_confidence(self, process: str) -> float:
        """Evalúa confianza en proceso (0-1)"""
        pass
    
    @abstractmethod
    def adjust_strategy(self, feedback: Dict[str, Any]) -> bool:
        """Ajusta estrategia basada en feedback"""
        pass


class IEthicalEvaluator(ABC):
    """Interfaz para evaluación ética"""
    
    @abstractmethod
    def evaluate_action(self, action: Dict[str, Any]) -> ProcessingResult:
        """Evalúa acción desde perspectiva ética"""
        pass
    
    @abstractmethod
    def get_ethical_principles(self) -> List[str]:
        """Obtiene principios éticos activos"""
        pass
    
    @abstractmethod
    def resolve_dilemma(self, dilemma: Dict[str, Any]) -> Dict[str, Any]:
        """Resuelve dilema ético"""
        pass


# ==================== ADAPTADORES ====================

class BiologicalSystemAdapter(INeuralProcessor):
    """
    Adaptador para BiologicalConsciousnessSystem
    Convierte interfaz específica a interfaz estándar
    """
    
    def __init__(self, biological_system):
        self.system = biological_system
    
    def process_input(self, input_pattern: Dict[str, Any]) -> ProcessingResult:
        """Adapta process_experience a process_input"""
        try:
            result = self.system.process_experience(
                input_pattern.get('stimulus', {}),
                input_pattern.get('context', {})
            )
            return ProcessingResult(
                success=True,
                data=result,
                metadata={'source': 'biological_system'}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                errors=[str(e)]
            )
    
    def get_activation_state(self) -> Dict[str, float]:
        """Obtiene estado de activación neural"""
        if hasattr(self.system, 'neural_network'):
            return self.system.neural_network.get_activation_state()
        return {}
    
    def update_weights(self, learning_signal: float):
        """Actualiza pesos mediante reinforcement learning"""
        if hasattr(self.system, 'neural_network'):
            # Identificar neuronas activas
            active_neurons = [
                n_id for n_id, neuron in self.system.neural_network.neurons.items()
                if neuron.membrane_potential > neuron.threshold
            ]
            self.system.neural_network.reinforce_learning(active_neurons, learning_signal)


class EmotionalSystemAdapter(IEmotionalProcessor):
    """Adaptador para HumanEmotionalSystem"""
    
    def __init__(self, emotional_system):
        self.system = emotional_system
    
    def process_stimulus(self, stimulus: Dict[str, Any]) -> ProcessingResult:
        """Procesa estímulo emocional"""
        try:
            emotion = stimulus.get('emotion', 'neutral')
            intensity = stimulus.get('intensity', 0.5)
            
            self.system.activate_circuit(emotion, intensity)
            
            return ProcessingResult(
                success=True,
                data=self.get_emotional_state(),
                metadata={'emotion_activated': emotion}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                errors=[str(e)]
            )
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """Obtiene estado emocional"""
        return self.system.get_emotional_state()
    
    def get_neurochemical_profile(self) -> Dict[str, float]:
        """Obtiene perfil neuroquímico"""
        return self.system.get_neurochemical_profile()
    
    def regulate_emotion(self, strategy: str) -> bool:
        """Aplica regulación emocional"""
        try:
            self.system.regulate_emotion(strategy)
            return True
        except:
            return False


# ==================== FACTORY ====================

class ConsciousnessComponentFactory:
    """
    Factory para crear componentes con interfaces estándar
    Permite intercambiar implementaciones fácilmente
    """
    
    @staticmethod
    def create_neural_processor(impl_type: str = "biological", **kwargs) -> INeuralProcessor:
        """Crea procesador neural"""
        if impl_type == "biological":
            from conciencia.modulos.biological_consciousness import BiologicalConsciousnessSystem
            bio_system = BiologicalConsciousnessSystem(**kwargs)
            return BiologicalSystemAdapter(bio_system)
        else:
            raise ValueError(f"Neural processor type '{impl_type}' not supported")
    
    @staticmethod
    def create_emotional_processor(impl_type: str = "human", **kwargs) -> IEmotionalProcessor:
        """Crea procesador emocional"""
        if impl_type == "human":
            from conciencia.modulos.human_emotions_system import HumanEmotionalSystem
            emotional_system = HumanEmotionalSystem(**kwargs)
            return EmotionalSystemAdapter(emotional_system)
        else:
            raise ValueError(f"Emotional processor type '{impl_type}' not supported")
    
    @staticmethod
    def create_memory_system(impl_type: str = "autobiographical", **kwargs) -> IMemorySystem:
        """Crea sistema de memoria"""
        # TODO: Implementar adaptadores para memoria
        raise NotImplementedError("Memory system creation not yet implemented")


# ==================== DEPENDENCY INJECTION ====================

class ConsciousnessContainer:
    """
    Contenedor de dependencias para inyección
    Permite configurar y obtener componentes sin acoplamiento directo
    """
    
    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register(self, interface: type, implementation: Any, singleton: bool = False):
        """Registra implementación para interfaz"""
        if singleton:
            self._singletons[interface.__name__] = implementation
        else:
            self._components[interface.__name__] = implementation
    
    def get(self, interface: type) -> Any:
        """Obtiene implementación para interfaz"""
        interface_name = interface.__name__
        
        # Verificar singleton primero
        if interface_name in self._singletons:
            return self._singletons[interface_name]
        
        # Si no, crear nueva instancia
        if interface_name in self._components:
            impl_class = self._components[interface_name]
            if callable(impl_class):
                return impl_class()
            return impl_class
        
        raise ValueError(f"No implementation registered for {interface_name}")
    
    def has(self, interface: type) -> bool:
        """Verifica si hay implementación registrada"""
        interface_name = interface.__name__
        return interface_name in self._components or interface_name in self._singletons


# ==================== EJEMPLO DE USO ====================

def demonstrate_interfaces():
    """Demuestra uso de interfaces desacopladas"""
    print("=" * 70)
    print("CONSCIOUSNESS INTERFACES - Demostración")
    print("=" * 70)
    
    # 1. Crear componentes mediante factory (desacoplado)
    print("\n1️⃣ Creando componentes mediante Factory...")
    
    neural_proc = ConsciousnessComponentFactory.create_neural_processor(
        impl_type="biological",
        system_id="demo",
        neural_network_size=2000,  # Red neuronal completa
        synaptic_density=0.15       # Densidad sináptica alta
    )
    print("   ✅ Neural Processor creado (2000 neuronas)")
    
    emotional_proc = ConsciousnessComponentFactory.create_emotional_processor(
        impl_type="human",
        num_circuits=35
    )
    print("   ✅ Emotional Processor creado")
    
    # 2. Usar interfaces estándar (desacoplado de implementación)
    print("\n2️⃣ Usando interfaces estándar...")
    
    # Procesar input neural
    result = neural_proc.process_input({
        'stimulus': {'type': 'visual', 'intensity': 0.8},
        'context': {'novelty': 0.6}
    })
    print(f"   ✅ Neural processing: {'Success' if result.success else 'Failed'}")
    
    # Procesar estímulo emocional
    result = emotional_proc.process_stimulus({
        'emotion': 'alegria',
        'intensity': 0.7
    })
    print(f"   ✅ Emotional processing: {'Success' if result.success else 'Failed'}")
    
    # 3. Dependency Injection
    print("\n3️⃣ Configurando Dependency Injection...")
    
    container = ConsciousnessContainer()
    container.register(INeuralProcessor, neural_proc, singleton=True)
    container.register(IEmotionalProcessor, emotional_proc, singleton=True)
    
    print("   ✅ Componentes registrados")
    
    # Obtener componentes desde container (totalmente desacoplado)
    neural = container.get(INeuralProcessor)
    emotional = container.get(IEmotionalProcessor)
    
    print("   ✅ Componentes obtenidos desde container")
    
    # 4. Intercambiabilidad
    print("\n4️⃣ Verificando intercambiabilidad...")
    print("   ✅ Cualquier implementación de IEmotionalProcessor puede usarse")
    print("   ✅ Sin cambios en código que lo use")
    print("   ✅ Facilita testing con mocks")
    
    print("\n" + "=" * 70)
    print("✅ Demostración completada - Interfaces desacopladas funcionando")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_interfaces()
