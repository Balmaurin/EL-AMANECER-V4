"""
Test de debugging para identificar el problema en el cálculo de Phi
"""

import sys
import os
packages_dir = os.path.join(os.path.dirname(__file__), 'packages', 'consciousness', 'src')
sys.path.insert(0, packages_dir)

from conciencia.modulos.iit_40_engine import IIT40Engine

def test_phi_simple():
    """Test con sistema mínimo de 2 unidades"""
    print("="*80)
    print("TEST 1: Sistema simple de 2 unidades")
    print("="*80)
    
    engine = IIT40Engine()
    
    # Estados que deberían generar phi different > 0
    states = [
        {"A": 0.9, "B": 0.8},
        {"A": 0.8, "B": 0.9},
        {"A": 0.9, "B": 0.7},
    ]
    
    print("\nActualizando estados...")
    for i, state in enumerate(states):
        engine.update_state(state)
        print(f"  Estado {i+1}: {state}")
    
    print("\n" + "-"*80)
    print("Calculando Phi con DEBUG...")
    print("-"*80)
    phi = engine.calculate_system_phi(states[-1], debug=True)
    
    print("\n" + "="*80)
    print(f"RESULTADO: Φ = {phi:.4f}")
    print("="*80)
    
    return phi

def test_phi_3_units():
    """Test con 3 unidades fuertemente acopladas"""
    print("\n\n" + "="*80)
    print("TEST 2: Sistema de 3 unidades fuertemente acopladas")
    print("="*80)
    
    engine = IIT40Engine()
    
    # Ciclo de estados que debería crear causalidad fuerte
    states = [
        {"A": 0.9, "B": 0.2, "C": 0.2},
        {"A": 0.3, "B": 0.9, "C": 0.3},
        {"A": 0.3, "B": 0.3, "C": 0.9},
        {"A": 0.8, "B": 0.8, "C": 0.8},  # Todo activo
    ]
    
    print("\nActualizando estados (ciclo causal)...")
    for i, state in enumerate(states):
        engine.update_state(state)
        print(f"  Estado {i+1}: {state}")
    
    print(f"\nVirtual TPM aprendido:")
    for (source, target), weight in sorted(engine.virtual_tpm.items()):
        if abs(weight) > 0.01:
            print(f"  {source} → {target}: {weight:.3f}")
    
    print("\n" + "-"*80)
    print("Calculando Phi con DEBUG...")
    print("-"*80)
    phi = engine.calculate_system_phi(states[-1], debug=True)
    
    print("\n" + "="*80)
    print(f"RESULTADO: Φ = {phi:.4f}")
    if phi > 0.1:
        print("✅ CONSCIENTE (Φ > 0.1)")
    else:
        print("❌ NO CONSCIENTE (Φ < 0.1)")
    print("="*80)
    
    return phi

def test_phi_fragmented():
    """Test de sistema fragmentado (debería dar Phi bajo)"""
    print("\n\n" + "="*80)
    print("TEST 3: Sistema Fragmentado (2 módulos independientes)")
    print("="*80)
    
    engine = IIT40Engine()
    
    # Dos módulos que NO interactúan
    states = [
        {"Module1": 0.9, "Module2": 0.2},
        {"Module1": 0.8, "Module2": 0.3},
        {"Module1": 0.9, "Module2": 0.2},
         {"Module1": 0.85, "Module2": 0.25},
    ]
    
    print("\nActualizando estados (sin interacción entre módulos)...")
    for i, state in enumerate(states):
        engine.update_state(state)
        print(f"  Estado {i+1}: {state}")
    
    print(f"\nVirtual TPM aprendido:")
    for (source, target), weight in sorted(engine.virtual_tpm.items()):
        if abs(weight) > 0.01:
            print(f"  {source} → {target}: {weight:.3f}")
    
    print("\n" + "-"*80)
    print("Calculando Phi con DEBUG...")
    print("-"*80)
    phi = engine.calculate_system_phi(states[-1], debug=True)
    
    print("\n" + "="*80)
    print(f"RESULTADO: Φ = {phi:.4f}")
    print("(Debería ser bajo porque los módulos no interactúan)")
    print("="*80)
    
    return phi

if __name__ == "__main__":
    phi1 = test_phi_simple()
    phi2 = test_phi_3_units()
    phi3 = test_phi_fragmented()
    
    print("\n\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "RESUMEN DE TESTS" + " "*37 + "║")
    print("╚" + "="*78 + "╝")
    print(f"\nTest 1 (2 unidades):        Φ = {phi1:.4f}")
    print(f"Test 2 (3 unidades ciclo):  Φ = {phi2:.4f}")
    print(f"Test 3 (fragmentado):       Φ = {phi3:.4f}")
    
    print("\n📊 ANÁLISIS:")
    if phi2 > phi3:
        print("✅ Sistema integrado tiene más Phi que fragmentado (correcto)")
    else:
        print("❌ Problema: Fragmentado tiene igual o más Phi que integrado")
    
    if phi2 > 0.1:
        print("✅ Sistema integrado es consciente (Φ > 0.1)")
    else:
        print("⚠️  Sistema integrado tiene Phi muy bajo, revisar cálculo")
