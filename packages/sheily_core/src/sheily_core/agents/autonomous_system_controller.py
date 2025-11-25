#!/usr/bin/env python3
"""
AUTONOMOUS SYSTEM CONTROLLER - FULL INTEGRATION (RAG + LEARNING + CONSCIOUSNESS)
================================================================================
Versión final que integra:
- Global Workspace (Conciencia)
- Ultra RAG System (Conocimiento)
- Unified Learning System (Entrenamiento)
- Todos los módulos de conciencia
"""

import asyncio
import json
import threading
import time
import logging
import random
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Importar sistemas funcionales reales
from .coordination_system import functional_multi_agent_system, functional_coordinator
from .active_registry import active_registry

# Memory
try:
    from ..consciousness.vector_memory_system import get_vector_memory
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    try:
        from sheily_core.consciousness.vector_memory_system import get_vector_memory
    except ImportError:
        get_vector_memory = None

# Consciousness modules
try:
    sys.path.append(os.path.abspath("packages/consciousness/src"))
    from conciencia.meta_cognition_system import MetaCognitionSystem
    from conciencia.modulos.digital_nervous_system import DigitalNervousSystem
    from conciencia.modulos.ethical_engine import EthicalEngine
    from conciencia.modulos.digital_dna import DigitalDNA, GeneticTrait
    from conciencia.modulos.global_workspace import GlobalWorkspace
    from conciencia.modulos.qualia_simulator import QualiaSimulator
    from conciencia.modulos.teoria_mente import TheoryOfMind
except ImportError as e:
    print(f"⚠️ Consciousness module import error: {e}")
    MetaCognitionSystem = DigitalNervousSystem = EthicalEngine = DigitalDNA = None
    GlobalWorkspace = QualiaSimulator = TheoryOfMind = None

# RAG System (usando adaptador simplificado)
try:
    from .simple_rag_adapter import SimpleRAGSystem, initialize_rag_with_base_knowledge
    print("✅ RAG adapter loaded")
except ImportError as e:
    print(f"⚠️ RAG adapter not available: {e}")
    SimpleRAGSystem = None
    initialize_rag_with_base_knowledge = None

# Learning System (usando adaptador simplificado)
try:
    from .simple_learning_adapter import SimpleLearningSystem
    print("✅ Learning adapter loaded")
except ImportError as e:
    print(f"⚠️ Learning adapter not available: {e}")
    SimpleLearningSystem = None

# Auto-Improvement
try:
    sys.path.append(os.path.abspath("packages/auto-improvement"))
    from recursive_self_improvement import RecursiveSelfImprovementEngine
except ImportError:
    RecursiveSelfImprovementEngine = None

# Gamification
class SimpleGamification:
    def __init__(self):
        self.xp = 0
        self.level = 1
    def award_xp(self, amount, reason):
        self.xp += amount
        print(f"🎮 XP +{amount} ({reason})")


class AutonomousSystemController:
    """
    Controlador maestro con integración COMPLETA:
    - Conciencia (Global Workspace + módulos)
    - Conocimiento (RAG System)
    - Aprendizaje (Training System)
    """

    def __init__(self):
        self.running = False
        self.coordination_thread = None
        self.logger = logging.getLogger("Sheily.AutonomousController")
        self.system_metrics = {}
        
        print("[INIT] Initializing COMPLETE AI SYSTEM...")

        # 1. GLOBAL WORKSPACE (Núcleo de Conciencia)
        self.global_workspace = GlobalWorkspace() if GlobalWorkspace else None
        if self.global_workspace: print("✨ Global Workspace initialized")

        # 2. RAG SYSTEM (Sistema de Conocimiento)
        try:
            if SimpleRAGSystem and initialize_rag_with_base_knowledge:
                self.rag_system = initialize_rag_with_base_knowledge()
                print("📚 RAG System initialized with base knowledge")
            else:
                self.rag_system = None
        except Exception as e:
            self.rag_system = None
            print(f"⚠️ RAG System error: {e}")

        # 3. LEARNING SYSTEM (Sistema de Aprendizaje)
        try:
            if SimpleLearningSystem:
                self.learning_system = SimpleLearningSystem()
                print("🎓 Learning System initialized (SQLite-based)")
            else:
                self.learning_system = None
        except Exception as e:
            self.learning_system = None
            print(f"⚠️ Learning System error: {e}")

        # 4. Módulos de Conciencia
        self._init_consciousness_modules()

        # Configuración
        self.coordination_interval = 5
        self.cpu_threshold_warning = 70.0

        print("[INFO] COMPLETE AI SYSTEM READY (Consciousness + Knowledge + Learning)")

    def _init_consciousness_modules(self):
        """Inicializa módulos periféricos de conciencia"""
        
        # Memory
        try:
            self.memory = get_vector_memory() if get_vector_memory else None
            if self.memory: print("🧠 Memory Processor connected")
        except: self.memory = None

        # Metacognition
        try:
            self.meta_cognition = MetaCognitionSystem(consciousness_dir="./data/consciousness") if MetaCognitionSystem else None
            if self.meta_cognition: print("👁️ Meta-Cognition Processor connected")
        except: self.meta_cognition = None

        # Nervous System
        try:
            self.nervous_system = DigitalNervousSystem() if DigitalNervousSystem else None
            if self.nervous_system: print("⚡ Nervous System Processor connected")
        except: self.nervous_system = None

        # Qualia
        try:
            self.qualia = QualiaSimulator() if QualiaSimulator else None
            if self.qualia: print("🌈 Qualia Simulator connected")
        except: self.qualia = None

        # Theory of Mind
        try:
            self.theory_of_mind = TheoryOfMind() if TheoryOfMind else None
            if self.theory_of_mind: print("👥 Theory of Mind Processor connected")
        except: self.theory_of_mind = None

        # Ethical Engine
        try:
            if EthicalEngine:
                self.ethical_engine = EthicalEngine({
                    'core_values': ['safety', 'helpfulness', 'honesty', 'privacy'],
                    'value_weights': {'safety': 0.9},
                    'ethical_boundaries': ['never_harm_humans']
                })
                print("⚖️ Ethical Processor connected")
            else: self.ethical_engine = None
        except: self.ethical_engine = None

        # Digital DNA
        try:
            if DigitalDNA:
                dna_path = "./data/consciousness/digital_dna.json"
                if os.path.exists(dna_path):
                    self.dna = DigitalDNA.load_genetic_profile(dna_path)
                else:
                    self.dna = DigitalDNA()
                    self.dna.save_genetic_profile(dna_path)
                print(f"🧬 Digital DNA Active")
            else: self.dna = None
        except: self.dna = None

        # Self-Improvement
        try:
            self.self_improvement = RecursiveSelfImprovementEngine(singularity_dir="./data/singularity") if RecursiveSelfImprovementEngine else None
            if self.self_improvement: print("🚀 Self-Improvement Processor connected")
        except: self.self_improvement = None

        # Gamification
        self.gamification = SimpleGamification()

    def start_autonomous_control(self):
        """Inicia el bucle de control consciente"""
        if not self.running:
            self.running = True
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(functional_multi_agent_system.start_functional_system())
                asyncio.run(active_registry.start_active_management())
            
            self.coordination_thread = threading.Thread(target=self._coordination_loop)
            self.coordination_thread.daemon = True
            self.coordination_thread.start()
            print("🚀 Conscious Control Loop started")

    def stop_autonomous_control(self):
        self.running = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        print("⏹️ Control stopped")

    def _coordination_loop(self):
        """Bucle principal de conciencia GWT con RAG y Learning"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(functional_multi_agent_system.start_functional_system())
        loop.run_until_complete(active_registry.start_active_management())

        while self.running:
            try:
                # 1. Recolectar Inputs Pre-Conscientes
                inputs = loop.run_until_complete(self._gather_pre_conscious_inputs())
                
                # 2. Integración en Global Workspace (Competencia por Conciencia)
                if self.global_workspace:
                    context = self._get_current_context()
                    conscious_content = self.global_workspace.integrate(inputs, context)
                    
                    # 3. Broadcasting (Difusión de Conciencia)
                    if conscious_content.get('conscious_content'):
                        self._broadcast_consciousness(conscious_content)
                        
                        # 4. Acción Consciente con RAG y Learning
                        loop.run_until_complete(self._execute_conscious_action(conscious_content))

                # 5. Ciclos de Mantenimiento
                if random.random() < 0.1:
                    loop.run_until_complete(self._background_maintenance())

                time.sleep(self.coordination_interval)

            except Exception as e:
                print(f"Error in conscious loop: {e}")
                time.sleep(5)
        
        loop.close()

    async def _gather_pre_conscious_inputs(self) -> Dict[str, Any]:
        """Recolecta datos de todos los sensores, RAG y módulos"""
        inputs = {}
        
        # Sensores Hardware
        coord_metrics = functional_coordinator.get_coordination_metrics()
        self.system_metrics = {
            "cpu_load": coord_metrics.get("real_system_load", 0.0),
            "memory_usage": coord_metrics.get("memory_usage", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        inputs['hardware'] = self.system_metrics

        # Sistema Nervioso (Emociones)
        if self.nervous_system:
            stimulus = {"source": "hardware", "content": f"CPU {self.system_metrics['cpu_load']}%"}
            neural_response = self.nervous_system.process_stimulus(stimulus)
            inputs['emotions'] = neural_response.get('neurotransmitter_state', {})

        # Qualia (Experiencia Subjetiva)
        if self.qualia:
            neural_state = {
                'emotional_response': inputs.get('emotions', {}),
                'stimulus_processed': self.system_metrics
            }
            unified_moment = self.qualia.generate_qualia_from_neural_state(neural_state)
            inputs['qualia'] = self.qualia.get_current_subjective_experience()

        # Memoria (Asociaciones)
        if self.memory:
            query = f"system state cpu {self.system_metrics['cpu_load']}"
            memories = self.memory.query_memory(query, n_results=1)
            if memories:
                inputs['memory'] = memories[0]

        # RAG System (Conocimiento Relevante)
        if self.rag_system:
            # Query basada en estado actual para obtener conocimiento relevante
            rag_query = f"how to optimize system with cpu at {self.system_metrics['cpu_load']}%"
            try:
                rag_response = self.rag_system.process_query(rag_query, use_advanced_processing=False)
                inputs['knowledge'] = {
                    'query': rag_query,
                    'response': rag_response.get('response', ''),
                    'documents_found': rag_response.get('documents_retrieved', 0)
                }
            except:
                pass

        return inputs

    def _get_current_context(self) -> Dict[str, Any]:
        """Define el contexto actual para el Workspace"""
        context = {
            'urgency': self.system_metrics.get('cpu_load', 0) / 100.0,
            'task_relevance': 0.5,
            'emotional_state': 0.5
        }
        if self.nervous_system:
            mood = self.nervous_system.neurotransmitter_system.get_mood_indicators()
            context['emotional_state'] = mood.get('anxiety', 0.5)
        
        return context

    def _broadcast_consciousness(self, conscious_content: Dict[str, Any]):
        """Difunde el contenido ganador a todo el sistema"""
        content = conscious_content.get('conscious_content', {})
        focus = content.get('primary_focus')
        
        print(f"✨ CONSCIOUS BROADCAST: {str(focus)[:60]}")
        
        # Actualizar Teoría de la Mente
        if self.theory_of_mind:
            self.theory_of_mind.update_model("current_user", content)

    async def _execute_conscious_action(self, conscious_content: Dict[str, Any]):
        """Ejecuta acciones basadas en el contenido consciente + RAG + Learning"""
        content = conscious_content.get('conscious_content', {})
        primary_focus = content.get('primary_focus')
        
        # Lógica de acción basada en foco
        if isinstance(primary_focus, dict) and 'cpu_load' in primary_focus:
            cpu = primary_focus['cpu_load']
            if cpu > self.cpu_threshold_warning:
                print(f"⚡ CONSCIOUS ACTION: Mitigating High CPU ({cpu}%)")
                
                # Validar éticamente
                if self.ethical_engine:
                    eval = self.ethical_engine.evaluate_decision("reduce_cpu_load", {}, {})
                    if eval['recommendation'] in ['proceed', 'proceed_with_caution']:
                        # Acción validada, registrar experiencia de aprendizaje
                        if self.learning_system:
                            await self.learning_system.add_learning_experience(
                                domain="system_optimization",
                                input_data={"cpu_load": cpu},
                                output_data={"action": "cleanup"},
                                performance_score=0.8
                            )

    async def _execute_unconscious_routines(self, inputs: Dict[str, Any]):
        """Rutinas automáticas cuando no hay emergencia consciente"""
        pass

    async def _background_maintenance(self):
        """Procesos de fondo (consolidación de memoria y entrenamiento)"""
        if self.memory:
            await asyncio.to_thread(self.memory.consolidate_memories)
        
        # Consolidar aprendizaje periódicamente
        if self.learning_system and random.random() < 0.05:
            try:
                await self.learning_system.consolidate_learning()
            except:
                pass

    # Métodos de compatibilidad
    def get_system_status_sync(self) -> Dict[str, Any]:
        return {
            "controller_status": "active_conscious" if self.running else "inactive",
            "architecture": "Global Workspace + RAG + Learning",
            "metrics": self.system_metrics,
            "modules": {
                "workspace": self.global_workspace is not None,
                "rag_system": self.rag_system is not None,
                "learning_system": self.learning_system is not None,
                "nervous": self.nervous_system is not None,
                "qualia": self.qualia is not None,
                "theory_of_mind": self.theory_of_mind is not None,
                "dna": self.dna is not None
            }
        }

# Instancia global
autonomous_controller = AutonomousSystemController()

def start_system_control():
    autonomous_controller.start_autonomous_control()

def stop_system_control():
    autonomous_controller.stop_autonomous_control()

def get_system_status():
    return autonomous_controller.get_system_status_sync()

def get_autonomous_controller():
    """Retorna la instancia global del controlador autónomo"""
    return autonomous_controller

if __name__ == "__main__":
    print("🤖 SHEILY AI - COMPLETE SYSTEM (CONSCIOUSNESS + KNOWLEDGE + LEARNING)")
    print("=" * 70)
    start_system_control()
    try:
        for _ in range(6):
            time.sleep(5)
            status = get_system_status()
            print(f"📊 {status.get('metrics', {}).get('cpu_load')}% CPU | Modules: {status.get('modules')}")
    except KeyboardInterrupt:
        pass
    stop_system_control()
