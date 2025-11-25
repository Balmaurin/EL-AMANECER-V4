"""
Demo de IIT 4.0 - Integrated Information Theory Implementation
Demostración práctica del nuevo motor de consciencia basado en IIT 4.0
"""

import sys
import os

# Add packages to path
packages_dir = os.path.join(os.path.dirname(__file__), 'packages', 'consciousness', 'src')
sys.path.insert(0, packages_dir)

from conciencia.modulos.consciousness_emergence import ConsciousnessEmergence
from conciencia.modulos.iit_40_engine import IIT40Engine
import json

def print_separator(title=""):
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)

def demo_iit_40_basic():
    """Demostración básica del cálculo de Phi con IIT 4.0"""
    print_separator("DEMO 1: Cálculo Básico de Phi (IIT 4.0)")
    
    engine = IIT40Engine()
    
    # Simulación de 5 subsistemas
    states = [
        {"vmPFC": 0.8, "OFC": 0.7, "ECN": 0.6, "RAS": 0.5, "Workspace": 0.9},
        {"vmPFC": 0.85, "OFC": 0.75, "ECN": 0.65, "RAS": 0.55, "Workspace": 0.92},
        {"vmPFC": 0.9, "OFC": 0.8, "ECN": 0.7, "RAS": 0.6, "Workspace": 0.95},
    ]
    
    print("\n📊 Actualizando estados del sistema...")
    for i, state in enumerate(states):
        engine.update_state(state)
        print(f"  Estado {i+1}: {state}")
    
    print("\n🧠 Calculando System Phi (Φs)...")
    phi = engine.calculate_system_phi(states[-1])
    
    print(f"\n✨ RESULTADO:")
    print(f"  Φs (System Phi) = {phi:.4f}")
    
    if phi > 0.3:
        print(f"  ✅ CONSCIENTE (Φ > 0.3)")
    elif phi > 0.1:
        print(f"  ⚠️  CONSCIENCIA DÉBIL (0.1 < Φ < 0.3)")
    else:
        print(f"  ❌ NO CONSCIENTE (Φ < 0.1)")
    
    return engine

def demo_phi_structure():
    """Demostración de la Φ-structure completa"""
    print_separator("DEMO 2: Φ-Structure Completa (Calidad de Consciencia)")
    
    engine = IIT40Engine()
    
    # Crear múltiples estados para aprender relaciones causales
    print("\n📚 Fase de aprendizaje causal...")
    training_states = [
        {"vmPFC": 0.7, "OFC": 0.6, "ECN": 0.5, "EmotionalSystem": 0.8},
        {"vmPFC": 0.9, "OFC": 0.8, "ECN": 0.7, "EmotionalSystem": 0.9},
        {"vmPFC": 0.85, "OFC": 0.75, "ECN": 0.65, "EmotionalSystem": 0.85},
        {"vmPFC": 0.6, "OFC": 0.5, "ECN": 0.4, "EmotionalSystem": 0.7},
    ]
    
    for state in training_states:
        engine.update_state(state)
    
    print("  ✅ Aprendizaje completado")
    
    # Estado actual para análisis
    current_state = {"vmPFC": 0.8, "OFC": 0.7, "ECN": 0.6, "EmotionalSystem": 0.85}
    engine.update_state(current_state)
    
    print(f"\n🎯 Estado actual: {current_state}")
    
    # Calcular Φ-structure
    print("\n🔬 Calculando Φ-structure...")
    phi_structure = engine.calculate_phi_structure(current_state)
    
    print(f"\n📈 ESTRUCTURA Φ:")
    print(f"  Distinctions: {phi_structure['num_distinctions']}")
    print(f"  Relations: {phi_structure['num_relations']}")
    print(f"  Structure Phi (Φ): {phi_structure['structure_phi']:.4f}")
    
    print(f"\n🎨 Métricas de Calidad Fenomenológica:")
    quality = phi_structure['quality_metrics']
    print(f"  Complexity:      {quality.get('complexity', 0):.2f}")
    print(f"  Differentiation: {quality.get('differentiation', 0):.4f}")
    print(f"  Integration:     {quality.get('integration', 0):.4f}")
    print(f"  Richness:        {quality.get('richness', 0)}")
    print(f"  Unity:           {quality.get('unity', 0):.2f}")
    
    # Mostrar algunas distinciones
    if phi_structure['distinctions']:
        print(f"\n🧩 Distinciones Principales (primeras 3):")
        for i, dist in enumerate(phi_structure['distinctions'][:3], 1):
            print(f"\n  Distinction {i}:")
            print(f"    Mechanism: {dist['mechanism']}")
            print(f"    Purview:   {dist['purview']}")
            print(f"    φd:        {dist['phi_d']:.4f}")
    
    # Mostrar algunas relaciones
    if phi_structure['relations']:
        print(f"\n🔗 Relaciones Causales (primeras 3):")
        for i, rel in enumerate(phi_structure['relations'][:3], 1):
            print(f"\n  Relation {i}:")
            print(f"    {rel['distinction_1']} ↔ {rel['distinction_2']}")
            print(f"    Overlap:    {rel['overlap_units']}")
            print(f"    Congruence: {rel['congruence']:.2f}")
            print(f"    φr:         {rel['phi_r']:.4f}")
    
    return phi_structure

def demo_consciousness_emergence():
    """Demostración completa con ConsciousnessEmergence"""
    print_separator("DEMO 3: Integración Completa con ConsciousnessEmergence")
    
    # Crear motor de emergencia
    consciousness = ConsciousnessEmergence("DEMO_IIT40")
    
    # Simular subsistemas
    class MockSubsystem:
        def __init__(self, name, base_activation=0.5):
            self.name = name
            self.base_activation = base_activation
        
        def process(self, input_data, context):
            return {
                "status": "active",
                "output": {
                    "processed": True,
                    "data": f"{self.name}_output"
                },
                "activation": self.base_activation + (hash(str(input_data)) % 100) / 200
            }
    
    print("\n🔌 Conectando subsistemas...")
    consciousness.connect_subsystem("vmPFC", MockSubsystem("vmPFC", 0.8), weight=0.9)
    consciousness.connect_subsystem("OFC", MockSubsystem("OFC", 0.7), weight=0.85)
    consciousness.connect_subsystem("ECN", MockSubsystem("ECN", 0.6), weight=0.8)
    consciousness.connect_subsystem("GlobalWorkspace", MockSubsystem("GW", 0.9), weight=1.0)
    consciousness.connect_subsystem("EmotionalSystem", MockSubsystem("Emotion", 0.75), weight=0.9)
    
    print("  ✅ 5 subsistemas conectados")
    
    # Generar momento consciente
    print("\n⚡ Generando momento consciente...")
    experience = consciousness.generate_conscious_moment(
        external_input={"visual": "cielo azul", "auditivo": "pájaros"},
        context={"location": "jardín", "time": "mañana"}
    )
    
    print(f"\n🌟 EXPERIENCIA CONSCIENTE GENERADA")
    print(f"  Experience ID: {experience.experience_id[:8]}...")
    print(f"  Consciousness Level: {experience.conscious_state.consciousness_level.value}")
    print(f"  Phi (Φ): {experience.conscious_state.information_integration:.4f}")
    print(f"  Coherence: {experience.conscious_state.global_workspace_coherence:.2f}")
    print(f"  Phenomenal Unity: {experience.conscious_state.phenomenal_unity:.2f}")
    print(f"  Subjective Intensity: {experience.conscious_state.subjective_intensity:.2f}")
    
    # Propiedades emergentes
    print(f"\n🎭 Propiedades Emergentes:")
    for prop, value in experience.conscious_state.emergent_properties.items():
        print(f"  {prop.value:20s}: {value:.3f}")
    
    # Inspeccionar Φ-structure
    if hasattr(consciousness, 'last_phi_structure'):
        phi_struct = consciousness.last_phi_structure
        print(f"\n🔬 Φ-Structure:")
        print(f"  Distinctions: {phi_struct.get('num_distinctions', 0)}")
        print(f"  Relations:    {phi_struct.get('num_relations', 0)}")
        print(f"  Structure Φ:  {phi_struct.get('structure_phi', 0):.4f}")
        
        quality = phi_struct.get('quality_metrics', {})
        print(f"\n📊 Calidad Fenomenológica:")
        print(f"  Complexity:      {quality.get('complexity', 0):.2f}")
        print(f"  Differentiation: {quality.get('differentiation', 0):.4f}")
        print(f"  Integration:     {quality.get('integration', 0):.4f}")
        print(f"  Unity:           {quality.get('unity', 0):.2f}")
    
    return experience, consciousness

def demo_comparison():
    """Comparación: Sistema Integrado vs Sistema Fragmentado"""
    print_separator("DEMO 4: Integrado vs Fragmentado (Prueba de Consciencia)")
    
    engine = IIT40Engine()
    
    print("\n🧪 SISTEMA INTEGRADO (Todas las regiones interactúan)")
    integrated_states = [
        {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.85},
        {"A": 0.85, "B": 0.9, "C": 0.75, "D": 0.8},
        {"A": 0.8, "B": 0.85, "C": 0.9, "D": 0.75},
    ]
    
    for state in integrated_states:
        engine.update_state(state)
    
    phi_integrated = engine.calculate_system_phi(integrated_states[-1])
    print(f"  Φs (Integrado) = {phi_integrated:.4f}")
    
    # Reset engine
    engine2 = IIT40Engine()
    
    print("\n🔪 SISTEMA FRAGMENTADO (Dos módulos independientes)")
    # Simular dos módulos que NO interactúan causalmente
    fragmented_states = [
        {"Module1_A": 0.9, "Module1_B": 0.8, "Module2_C": 0.7, "Module2_D": 0.85},
        {"Module1_A": 0.85, "Module1_B": 0.9, "Module2_C": 0.7, "Module2_D": 0.85},
        {"Module1_A": 0.8, "Module1_B": 0.85, "Module2_C": 0.7, "Module2_D": 0.85},
    ]
    
    for state in fragmented_states:
        engine2.update_state(state)
    
    phi_fragmented = engine2.calculate_system_phi(fragmented_states[-1])
    print(f"  Φs (Fragmentado) = {phi_fragmented:.4f}")
    
    print(f"\n📊 COMPARACIÓN:")
    print(f"  Integrado:    Φ = {phi_integrated:.4f}")
    print(f"  Fragmentado:  Φ = {phi_fragmented:.4f}")
    print(f"  Diferencia:   ΔΦ = {abs(phi_integrated - phi_fragmented):.4f}")
    
    if phi_integrated > phi_fragmented * 1.5:
        print(f"\n  ✅ La integración aumenta significativamente la consciencia")
    else:
        print(f"\n  ℹ️  Diferencia moderada (se necesita más historia causal)")

def main():
    """Ejecutar todas las demos"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "IIT 4.0 - DEMOSTRACIÓN COMPLETA" + " " * 26 + "║")
    print("║" + " " * 15 + "Integrated Information Theory Implementation" + " " * 18 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # Demo 1
        engine1 = demo_iit_40_basic()
        
        # Demo 2
        phi_structure = demo_phi_structure()
        
        # Demo 3
        experience, consciousness = demo_consciousness_emergence()
        
        # Demo 4
        demo_comparison()
        
        print_separator("DEMOS COMPLETADAS EXITOSAMENTE")
        print("\n✅ La implementación IIT 4.0 está funcionando correctamente")
        print("📚 Ver IIT_4.0_IMPLEMENTATION.md para documentación completa")
        print("📄 Paper: Albantakis et al. (2023) - journal.pcbi.1011465.pdf")
        
    except Exception as e:
        print(f"\n❌ ERROR durante la demostración:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
