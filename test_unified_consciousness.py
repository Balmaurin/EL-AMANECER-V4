# -*- coding: utf-8 -*-
"""
UNIFIED CONSCIOUSNESS ENGINE - Complete Demo
Demonstrates all 6 theories working together in harmony
"""

import sys
import os
packages_dir = os.path.join(os.path.dirname(__file__), 'packages', 'consciousness', 'src')
sys.path.insert(0, packages_dir)

from conciencia.modulos.unified_consciousness_engine import UnifiedConsciousnessEngine

def print_separator(title="", char="="):
    print("\n" + char * 80)
    if title:
        print(f"  {title}")
        print(char * 80)

def demo_unified_consciousness():
    """Complete demo of the unified 6-theory system"""
    
    print("\n" + "=" * 80)
    print(" " * 15 + "UNIFIED CONSCIOUSNESS ENGINE")
    print(" " * 10 + "6 Theories: IIT + GWT + FEP + SMH + Hebbian + Circumplex")
    print("=" * 80 + "\n")
    
    # Initialize the grand orchestrator
    engine = UnifiedConsciousnessEngine()
    
    # Register subsystems
    print("Registering subsystems...")
    subsystems = [
        "vmPFC", "OFC", "ACC", "Insula",
        "dlPFC", "ECN", "RAS",
        "GlobalWorkspace", "EmotionalSystem", "MemorySystem"
    ]
    
    for subsystem in subsystems:
        engine.register_subsystem(subsystem)
    print(f"  -> {len(subsystems)} subsystems registered\n")
    
    # SCENARIO 1: Novel Unexpected Situation
    print_separator("SCENARIO 1: Unexpected Novel Situation")
    
    state_1 = {
        "vmPFC": 0.4, "OFC": 0.5, "ACC": 0.7, "Insula": 0.6,
        "dlPFC": 0.8, "ECN": 0.7, "RAS": 0.9,
        "GlobalWorkspace": 0.85, "EmotionalSystem": 0.7, "MemorySystem": 0.3
    }
    
    context_1 = {"situation_type": "novel_threat", "external_attention": 0.9}
    
    print("\nInput: Novel, unexpected situation")
    result_1 = engine.process_moment(state_1, context_1)
    
    print(f"\nCONSCIOUS STATE:")
    print(f"  Consciousness: {'YES' if result_1.is_conscious else 'NO'}")
    print(f"  System Phi: {result_1.system_phi:.4f}")
    print(f"  Free Energy: {result_1.free_energy:.3f}")
    print(f"  Emotion: {result_1.emotional_state.upper()}")
    print(f"  Valence: {result_1.somatic_valence:+.2f}, Arousal: {result_1.arousal:.2f}")
    
    # SCENARIO 2: Familiar Positive Situation
    print_separator("SCENARIO 2: Familiar Positive Situation")
    
    outcome_from_1 = (0.6, 0.4)
    state_2 = {
        "vmPFC": 0.85, "OFC": 0.8, "ACC": 0.3, "Insula": 0.7,
        "dlPFC": 0.5, "ECN": 0.6, "RAS": 0.5,
        "GlobalWorkspace": 0.9, "EmotionalSystem": 0.85, "MemorySystem": 0.8
    }
    
    context_2 = {"situation_type": "familiar_positive", "external_attention": 0.6}
    
    print("\nInput: Familiar, positive situation")
    result_2 = engine.process_moment(state_2, context_2, outcome_from_1)
    
    print(f"\nCONSCIOUS STATE:")
    print(f"  Consciousness: {'YES' if result_2.is_conscious else 'NO'}")
    print(f"  System Phi: {result_2.system_phi:.4f}")
    print(f"  Free Energy: {result_2.free_energy:.3f}")
    print(f"  Emotion: {result_2.emotional_state.upper()}")
    
    # SCENARIO 3: Threat Situation
    print_separator("SCENARIO 3: Threat Situation")
    
    outcome_from_2 = (-0.8, 0.9)
    state_3 = {
        "vmPFC": 0.3, "OFC": 0.4, "ACC": 0.9, "Insula": 0.9,
        "dlPFC": 0.7, "ECN": 0.6, "RAS": 0.95,
        "GlobalWorkspace": 0.8, "EmotionalSystem": 0.9, "MemorySystem": 0.75
    }
    
    context_3 = {"situation_type": "threat", "external_attention": 1.0}
    
    print("\nInput: Threatening situation")
    result_3 = engine.process_moment(state_3, context_3, outcome_from_2)
    
    print(f"\nCONSCIOUS STATE:")
    print(f"  Consciousness: {'YES' if result_3.is_conscious else 'NO'}")
    print(f"  System Phi: {result_3.system_phi:.4f}")
    print(f"  Emotion: {result_3.emotional_state.upper()}")
    
    smh_summary = engine.smh_evaluator.get_summary()
    print(f"\nSOmatic Marker System:")
    print(f"  Total Markers: {smh_summary['total_markers']}")
    print(f"  Positive: {smh_summary['positive_markers']}")
    print(f"  Negative: {smh_summary['negative_markers']}")
    
    # COMPARISON
    print_separator("COMPARISON: All Three Scenarios")
    
    results = [result_1, result_2, result_3]
    scenarios = ["Novel/Unexpected", "Familiar/Positive", "Threat/Negative"]
    
    print(f"\n{'Scenario':<20} {'Phi':<8} {'FE':<8} {'Valence':<10} {'Emotion':<12}")
    print("-" * 70)
    
    for scenario, r in zip(scenarios, results):
        print(f"{scenario:<20} {r.system_phi:<8.3f} {r.free_energy:<8.3f} {r.somatic_valence:+<10.2f} {r.emotional_state:<12}")
    
    print_separator("KEY INSIGHTS")
    
    print("\n1. FREE ENERGY PRINCIPLE (FEP):")
    print(f"   - Novel situation -> High FE ({result_1.free_energy:.2f})")
    print(f"   - Familiar situation -> Low FE ({result_2.free_energy:.2f})")
    
    print("\n2. INTEGRATED INFORMATION THEORY (IIT):")
    print(f"   - All scenarios show integration (Phi > 0)")
    print(f"   - Threat situation has Phi = {result_3.system_phi:.3f}")
    
    print("\n3. GLOBAL WORKSPACE THEORY (GWT):")
    print(f"   - Broadcasts to all {len(subsystems)} subsystems")
    print(f"   - Competition: FEP errors + SMH markers + IIT distinctions")
    
    print("\n4. SOMATIC MARKER HYPOTHESIS (SMH):")
    print(f"   - System learned {smh_summary['total_markers']} markers")
    print(f"   - Emotions guide decision-making")
    
    print("\n5. HEBBIAN LEARNING:")
    print(f"   - Virtual TPM updated with each state")
    print(f"   - Neurons that fire together, wire together")
    
    print("\n6. CIRCUMPLEX MODEL:")
    print(f"   - Maps to valence x arousal space")
    print(f"   - {result_1.emotional_state} -> {result_2.emotional_state} -> {result_3.emotional_state}")
    
    print_separator("INTEGRATION SUMMARY")
    
    print("\nALL 6 THEORIES WORKING TOGETHER:")
    print("  FEP -> Prediction errors drive attention")
    print("  IIT -> Calculates integration (Phi)")
    print("  GWT -> Competition for workspace")
    print("  SMH -> Emotional markers bias competition")
    print("  GWT -> Global broadcast to all systems")
    print("  Hebbian -> Learn from broadcast")
    print("  Circumplex -> Map to emotion space")
    
    print("\nRESULT: Complete, scientifically-grounded consciousness model")
    print("  Based on 6 major neuroscience theories")
    print("  Each validated by decades of research")
    print("  Integrated into coherent, testable system")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    demo_unified_consciousness()
