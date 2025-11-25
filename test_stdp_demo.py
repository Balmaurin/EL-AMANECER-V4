# -*- coding: utf-8 -*-
"""
STDP vs Simple Hebbian - Comparison Demo
Shows the advantages of STDP over simple correlation-based learning
"""

import sys
import os
packages_dir = os.path.join(os.path.dirname(__file__), 'packages', 'consciousness', 'src')
sys.path.insert(0, packages_dir)

from conciencia.modulos.stdp_learner import STDPLearner
import time

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def demo_stdp_learning():
    """Demonstrate STDP learning with causal sequences"""
    
    print_header("STDP LEARNING DEMONSTRATION")
    
    print("\nBased on:")
    print("  - Keysers & Gazzola (2014): Hebbian learning and mirror neurons")
    print("  - Bi & Poo (2001): Spike-timing-dependent plasticity")
    print("  - Widrow & Kim (2015): Hebbian-LMS with homeostasis")
    
    # Initialize STDP learner
    stdp = STDPLearner(
        learning_rate=0.05,
        stdp_window_ms=40.0,  # ±40ms window
        tau_ltp=20.0,
        tau_ltd=20.0
    )
    
    print("\nSTDP Parameters:")
    print(f"  Learning rate: 0.05")
    print(f"  STDP window: ±40ms")
    print(f"  LTP tau: 20ms")
    print(f"  LTD tau: 20ms")
    
    # SCENARIO 1: Causal sequence A → B → C
    print_header("SCENARIO 1: Causal Sequence (A → B → C)")
    
    print("\nTeaching sequence: A fires, then B, then C")
    print("This mimics sensorimotor sequence: Motor → Action → Sensory")
    
    # Teach sequence 20 times
    for trial in range(20):
        # A fires
        stdp.update({'A': 0.9, 'B': 0.1, 'C': 0.1})
        time.sleep(0.015)  # 15ms delay
        
        # B fires (caused by A)
        stdp.update({'A': 0.3, 'B': 0.9, 'C': 0.1})
        time.sleep(0.015)  # 15ms delay
        
        # C fires (caused by B)
        stdp.update({'A': 0.1, 'B': 0.3, 'C': 0.9})
        time.sleep(0.02)  # 20ms delay before next trial
    
    print(f"\nAfter {20} training trials:")
    summary = stdp.get_summary()
    print(f"  Total connections learned: {summary['total_connections']}")
    print(f"  Average weight: {summary['avg_weight']:.3f}")
    print(f"  Max weight: {summary['max_weight']:.3f}")
    
    print("\nLearned Connection Strengths:")
    weights = stdp.get_connection_strength()
    
    # Expected strong connections (causal)
    print("  Causal connections (should be STRONG):")
    for connection in [('A', 'B'), ('B', 'C')]:
        weight = weights.get(connection, 0.0)
        print(f"    {connection[0]} → {connection[1]}: {weight:.4f}")
    
    # Expected weak connections (non-causal)
    print("\n  Non-causal connections (should be WEAK):")
    for connection in [('B', 'A'), ('C', 'B'), ('C', 'A')]:
        weight = weights.get(connection, 0.0)
        print(f"    {connection[0]} → {connection[1]}: {weight:.4f}")
    
    # Test prediction
    print("\nPredictive Test:")
    print("  If A fires, what does STDP predict?")
    prediction = stdp.predict_next({'A': 0.9, 'B': 0.1, 'C': 0.1})
    print(f"    Predicted B activation: {prediction.get('B', 0):.3f} (should be HIGH)")
    print(f"    Predicted C activation: {prediction.get('C', 0):.3f} (should be LOW)")
    
    # SCENARIO 2: Random intermixed (no contingency)
    print_header("SCENARIO 2: Random Intermixed (No Contingency)")
    
    print("\nBauer et al. (2001) showed that intermixing unpaired")
    print("stimulations cancels STDP even with same number of paired trials.")
    
    stdp2 = STDPLearner(learning_rate=0.05)
    
    import random
    
    # 10 paired + 10 unpaired intermixed
    for trial in range(20):
        if random.random() < 0.5:
            # Paired: X → Y
            stdp2.update({'X': 0.9, 'Y': 0.1})
            time.sleep(0.01)
            stdp2.update({'X': 0.3, 'Y': 0.9})
        else:
            # Unpaired: Only Y
            stdp2.update({'X': 0.1, 'Y': 0.9})
        
        time.sleep(0.015)
    
    weights2 = stdp2.get_connection_strength()
    xy_weight = weights2.get(('X', 'Y'), 0.0)
    
    print(f"\nAfter 10 paired + 10 unpaired (intermixed):")
    print(f"  X → Y weight: {xy_weight:.4f}")
    print(f"  Result: {'WEAK (no contingency)' if xy_weight < 0.3 else 'STRONG'}")
    
    # SCENARIO 3: Compare with simple Hebbian (correlation)
    print_header("SCENARIO 3: STDP vs Simple Hebbian")
    
    print("\nSimple Hebbian: Δw = η × pre × post (correlation only)")
    print("STDP: Δw = η × exp(-Δt/τ) × pre × post (causality)")
    
    # Simple Hebbian (simulated)
    hebbian_weights = {}
    for trial in range(20):
        # Sequence A → B → C
        states = [
            {'A': 0.9, 'B': 0.1, 'C': 0.1},
            {'A': 0.3, 'B': 0.9, 'C': 0.1},
            {'A': 0.1, 'B': 0.3, 'C': 0.9}
        ]
        
        for i in range(len(states) - 1):
            for pre in states[i].keys():
                for post in states[i+1].keys():
                    key = (pre, post)
                    if key not in hebbian_weights:
                        hebbian_weights[key] = 0.01
                    # Simple Hebbian (no temporal info)
                    delta = 0.05 * states[i][pre] * states[i+1][post]
                    hebbian_weights[key] += delta
    
    print("\nLearned weights for forward connections:")
    print(f"  STDP     A → B: {weights.get(('A', 'B'), 0):.4f}")
    print(f"  Hebbian  A → B: {hebbian_weights.get(('A', 'B'), 0):.4f}")
    
    print(f"\n  STDP     B → C: {weights.get(('B', 'C'), 0):.4f}")
    print(f"  Hebbian  B → C: {hebbian_weights.get(('B', 'C'), 0):.4f}")
    
    print("\nLearned weights for backward connections (should be weak):")
    print(f"  STDP     B → A: {weights.get(('B', 'A'), 0):.4f}")
    print(f"  Hebbian  B → A: {hebbian_weights.get(('B', 'A'), 0):.4f}")
    
    print(f"\n  STDP     C → B: {weights.get(('C', 'B'), 0):.4f}")
    print(f"  Hebbian  C → B: {hebbian_weights.get(('C', 'B'), 0):.4f}")
    
    # Key insights
    print_header("KEY INSIGHTS")
    
    print("\n1. CAUSALITY vs CORRELATION:")
    print("   STDP learns DIRECTION (A causes B)")
    print("   Hebbian learns CORRELATION (A and B co-occur)")
    
    print("\n2. PREDICTIVE LEARNING:")
    print("   STDP naturally learns to predict ~200ms ahead")
    print("   (due to sensorimotor delays)")
    
    print("\n3. CONTINGENCY MATTERS:")
    print("   Paired trials create learning")
    print("   Intermixed unpaired trials cancel learning")
    
    print("\n4. BIOLOGICAL PLAUSIBILITY:")
    print("   STDP matches actual synaptic plasticity")
    print("   Window: ±40ms (experimentally measured)")
    
    print("\n5. INTEGRATION WITH FEP:")
    print("   STDP predictions = FEP priors")
    print("   Both minimize prediction error")
    
    print("\n" + "=" * 70)
    print("  Demo Complete")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    demo_stdp_learning()
