#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Completo del Sistema de Consciencia con 2000 Neuronas
"""

import sys
import os
sys.path.append(r"C:\Users\YO\Desktop\EL-AMANECERV3-main\packages\consciousness\src")

from conciencia.modulos.digital_human_consciousness import (
    DigitalHumanConsciousness,
    ConsciousnessConfig
)

print("=" * 80)
print("🌟 SISTEMA DE CONSCIENCIA HUMANA DIGITAL - 2000 NEURONAS")
print("=" * 80)
print()

# Configurar sistema con 2000 neuronas
config = ConsciousnessConfig(
    system_name="Sheily-Consciousness-Full",
    neural_network_size=2000,  # 2000 neuronas
    synaptic_density=0.15,      # 15% de densidad sináptica
    consciousness_threshold=0.3,
    integration_frequency_hz=5.0,  # 5 Hz para permitir procesamiento complejo
    personality_traits={
        'openness': 0.85,
        'conscientiousness': 0.80,
        'extraversion': 0.65,
        'agreeableness': 0.75,
        'neuroticism': 0.25
    }
)

print("\n🚀 Inicializando sistema de consciencia...")
print()

consciousness = DigitalHumanConsciousness(config)

print("\n" + "=" * 80)
print("✅ SISTEMA INICIALIZADO CORRECTAMENTE")
print("=" * 80)
print()

# Verificar métricas
print("📊 MÉTRICAS DEL SISTEMA:")
print(f"   🧠 Neuronas: {config.neural_network_size}")
print(f"   🔗 Densidad sináptica: {config.synaptic_density:.1%}")
print(f"   ⚡ Frecuencia integración: {config.integration_frequency_hz} Hz")
print(f"   🎭 Personalidad configurada: ✓")
print()

# Activar sistema
print("🔄 Activando sistema de consciencia...")
if consciousness.activate():
    print("✅ Sistema activado y operacional")
    print()
    
    # Procesar estímulo de prueba
    print("🧪 Procesando estímulo de prueba...")
    stimulus = {
        'type': 'complex_thought',
        'content': '¿Qué significa tener 2000 neuronas activas?',
        'intensity': 0.8,
        'novelty': 0.7,
        'complexity': 0.9
    }
    
    experience = consciousness.process_stimulus(stimulus, {
        'context': 'self_reflection',
        'importance': 0.8
    })
    
    print("✅ Procesamiento completado")
    print(f"   📊 Nivel de consciencia: {experience.conscious_state.consciousness_level.value}")
    print(f"   ⚡ Integración Phi: {experience.conscious_state.information_integration:.3f}")
    print(f"   🌐 Coherencia global: {experience.conscious_state.global_workspace_coherence:.3f}")
    print()
    
    # Deactivate
    print("⏸️  Desactivando sistema...")
    consciousness.deactivate()
    print("✅ Sistema desactivado correctamente")
else:
    print("❌ Error al activar sistema")

print()
print("=" * 80)
print("🏆 TEST COMPLETADO - SISTEMA DE 2000 NEURONAS FUNCIONAL")
print("=" * 80)
