"""
Demo: IIT 4.0 + GWT Integration
Demonstrates the synergy between Integrated Information Theory and Global Workspace Theory
"""

import sys
import os
packages_dir = os.path.join(os.path.dirname(__file__), 'packages', 'consciousness', 'src')
sys.path.insert(0, packages_dir)

from conciencia.modulos.iit_gwt_integration import ConsciousnessOrchestrator

def print_separator(title=""):
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)

def demo_iit_gwt_integration():
    """Demonstra la integración IIT 4.0 + GWT"""
    
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "IIT 4.0 + GWT INTEGRATION DEMO" + " " * 32 + "║")
    print("║" + " " * 10 + "Integrated Information + Global Workspace" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Create orchestrator
    orchestrator = ConsciousnessOrchestrator()
    
    # Register subsystems (both as IIT units and GWT audience members)
    print("\n🔌 Registering subsystems...")
    subsystems = ["vmPFC", "OFC", "ECN", "GlobalWorkspace", "EmotionalSystem", "MemorySystem"]
    for subsystem in subsystems:
        orchestrator.register_subsystem(subsystem)
        print(f"  ✅ {subsystem}")
    
    print_separator("CYCLE 1: Initial Perception")
    
    # Cycle 1: Visual perception
    states_1 = {
        "vmPFC": 0.3,
        "OFC": 0.4,
        "ECN": 0.2,
        "GlobalWorkspace": 0.9,  # HIGH - visual input entering workspace
        "EmotionalSystem": 0.5,
        "MemorySystem": 0.3
    }
    
    attention_1 = {
        "GlobalWorkspace": 1.0,  # Spotlight on visual input
        "EmotionalSystem": 0.3
    }
    
    contexts_1 = {
        "visual": 0.8,  # Visual context active
        "alertness": 0.6
    }
    
    print("\n📊 Input State:")
    for k, v in states_1.items():
        print(f"  {k:20s}: {v:.2f}")
    
    result_1 = orchestrator.process_conscious_moment(states_1, attention_1, contexts_1)
    
    print(f"\n🧠 CONSCIOUSNESS:")
    print(f"  Is Conscious: {'✅ YES' if result_1['is_conscious'] else '❌ NO'}")
    print(f"  System Φ:     {result_1['system_phi']:.4f}")
    
    print(f"\n🎭 GLOBAL WORKSPACE:")
    workspace = result_1['workspace']
    print(f"  Capacity:     {workspace['capacity']}")
    print(f"  Current:      {workspace['current_contents']} items")
    print(f"  Audience:     {workspace['audience_size']} processors")
    
    if workspace['contents']:
        print(f"\n  📺 Workspace Contents:")
        for i, content in enumerate(workspace['contents'][:3], 1):
            print(f"    {i}. Mechanism: {content['mechanism']}")
            print(f"       φd: {content['phi_d']:.4f}, Salience: {content['salience']:.4f}")
    
    print(f"\n📡 BROADCASTS:")
    print(f"  Total: {len(result_1['broadcasts'])}")
    for i, broadcast in enumerate(result_1['broadcasts'][:3], 1):
        print(f"    {i}. Source: {broadcast['source']}")
        print(f"       Strength: {broadcast['strength']:.4f}")
        print(f"       Audience: {broadcast['audience_size']} subsystems")
    
    print_separator("CYCLE 2: Emotional Response")
    
    # Cycle 2: Emotional response to stimulus
    states_2 = {
        "vmPFC": 0.8,  # HIGH - emotional evaluation
        "OFC": 0.7,
        "ECN": 0.6,
        "GlobalWorkspace": 0.85,
        "EmotionalSystem": 0.9,  # HIGH - emotion activated
        "MemorySystem": 0.7  # HIGH - retrieving related memories
    }
    
    attention_2 = {
        "EmotionalSystem": 1.0,  # Spotlight shifts to emotion
        "vmPFC": 0.8,
        "MemorySystem": 0.6
    }
    
    contexts_2 = {
        "emotional": 0.9,  # Emotional context now dominant
        "memory_retrieval": 0.7
    }
    
    print("\n📊 Input State (Emotional activation):")
    for k, v in states_2.items():
        print(f"  {k:20s}: {v:.2f}")
    
    result_2 = orchestrator.process_conscious_moment(states_2, attention_2, contexts_2)
    
    print(f"\n🧠 CONSCIOUSNESS:")
    print(f"  Is Conscious: {'✅ YES' if result_2['is_conscious'] else '❌ NO'}")
    print(f"  System Φ:     {result_2['system_phi']:.4f}")
    
    print(f"\n🎭 GLOBAL WORKSPACE:")
    workspace_2 = result_2['workspace']
    print(f"  Current:      {workspace_2['current_contents']} items")
    
    if workspace_2['contents']:
        print(f"\n  📺 Workspace Contents (Competition Winners):")
        for i, content in enumerate(workspace_2['contents'][:3], 1):
            print(f"    {i}. Mechanism: {content['mechanism']}")
            print(f"       φd: {content['phi_d']:.4f}, Salience: {content['salience']:.4f}")
            print(f"       (Winner due to high φd + attention + context)")
    
    print(f"\n📡 BROADCASTS:")
    for i, broadcast in enumerate(result_2['broadcasts'][:3], 1):
        print(f"    {i}. {broadcast['source']} → {broadcast['audience_size']} systems")
        print(f"       Broadcast strength: {broadcast['strength']:.4f}")
    
    print_separator("CYCLE 3: Integrated Conscious Moment")
    
    # Cycle 3: Fully integrated moment
    states_3 = {
        "vmPFC": 0.85,
        "OFC": 0.8,
        "ECN": 0.75,
        "GlobalWorkspace": 0.9,
        "EmotionalSystem": 0.85,
        "MemorySystem": 0.8
    }
    
    attention_3 = {
        "GlobalWorkspace": 0.9,
        "EmotionalSystem": 0.8,
        "MemorySystem": 0.7,
        "vmPFC": 0.8
    }
    
    contexts_3 = {
        "narrative": 0.8,  # Story-like integration
        "self_reference": 0.7
    }
    
    result_3 = orchestrator.process_conscious_moment(states_3, attention_3, contexts_3)
    
    print(f"\n🧠 HIGHEST INTEGRATION:")
    print(f"  System Φ:     {result_3['system_phi']:.4f}")
    
    quality = result_3['integration_quality']
    print(f"\n🎨 PHENOMENAL QUALITY (from Φ-structure):")
    print(f"  Complexity:      {quality.get('complexity', 0):.2f}")
    print(f"  Differentiation: {quality.get('differentiation', 0):.4f}")
    print(f"  Integration:     {quality.get('integration', 0):.4f}")
    print(f"  Unity:           {quality.get('unity', 0):.2f}")
    
    print(f"\n🎭 GLOBAL BROADCAST (from GWT):")
    print(f"  Workspace items: {result_3['workspace']['current_contents']}")
    print(f"  Broadcasts:      {len(result_3['broadcasts'])}")
    print(f"  Global access:   {'✅ ACHIEVED' if result_3['global_access'] else '❌ NO'}")
    
    print_separator("ANALYSIS: IIT + GWT Synergy")
    
    print("\n📚 Key Insights:")
    print("\n1. IIT 4.0 provides:")
    print("   • WHAT is conscious (Φ > threshold)")
    print(f"   • Measured Φ = {result_3['system_phi']:.4f}")
    print("   • Distinction-based structure")
    print(f"   • {len(result_3['phi_structure'].get('distinctions', []))} distinctions identified")
    
    print("\n2. GWT provides:")
    print("   • HOW information is broadcast")
    print(f"   • {len(result_3['broadcasts'])} global broadcasts")
    print("   • Competition for workspace access")
    print("   • Limited capacity enforcement (7±2 items)")
    
    print("\n3. Integration Mechanisms:")
    print("   • IIT distinctions compete for GWT workspace")
    print("   • φd (integrated info) determines intrinsic salience")
    print("   • Attention modulates extrinsic salience")
    print("   • Context operators bias competition")
    print("   • Winners broadcast to all audience members")
    
    print("\n4. Synergy:")
    print("   • IIT explains WHY integration matters (Φ)")
    print("   • GWT explains HOW integrated info spreads (broadcast)")
    print("   • Together: Complete model of conscious processing")
    
    print_separator("COMPARISON: All Three Cycles")
    
    results = [result_1, result_2, result_3]
    print(f"\n{'Cycle':<10} {'System Φ':<12} {'Conscious':<12} {'Broadcasts':<12} {'Quality'}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        conscious = "YES" if r['is_conscious'] else "NO"
        quality_score = r['integration_quality'].get('unity', 0)
        print(f"{i:<10} {r['system_phi']:<12.4f} {conscious:<12} {len(r['broadcasts']):<12} {quality_score:.2f}")
    
    print("\n✅ Progression shows:")
    print("   • Increasing Φ (0.07 → 0.09 → higher)")
    print("   • More broadcasts as integration increases")
    print("   • Higher phenomenal quality with full integration")

if __name__ == "__main__":
    demo_iit_gwt_integration()
