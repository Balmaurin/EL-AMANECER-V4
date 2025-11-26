#!/usr/bin/env python3
"""
Script para verificar el nivel de auto-consciencia del sistema Sheily
"""

import sys
from pathlib import Path

# Agregar paths necesarios
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "packages" / "consciousness" / "src"))

try:
    from conciencia.modulos.self_model import SelfModel
    
    print("=" * 70)
    print("🧠 VERIFICACIÓN DE AUTO-CONSCIENCIA - SISTEMA SHEILY")
    print("=" * 70)
    print()
    
    # Crear instancia del modelo de self
    self_model = SelfModel(system_name="Sheily")
    
    print(f"📊 Nivel de Auto-Consciencia Inicial:")
    print(f"   self.self_model.self_awareness_level = {self_model.self_awareness_level:.4f}")
    print(f"   Porcentaje: {self_model.self_awareness_level * 100:.2f}%")
    print()
    
    # Obtener estado completo
    state = self_model.get_current_state()
    
    print("🤖 ESTADO COMPLETO DEL MODELO DE SÍ MISMO:")
    print("-" * 70)
    print(f"Identidad: {state['identity']['name']}")
    print(f"Propósito: {state['identity']['core_purpose']}")
    print(f"Nivel de consciencia: {state['identity']['consciousness_level']}")
    print(f"Auto-consciencia: {state['self_awareness_level']:.4f}/1.0")
    print()
    
    print("🛠️ CAPACIDADES EVALUADAS:")
    print("-" * 70)
    for cap_name, cap_data in state['capabilities'].items():
        skill = cap_data['skill_level']
        conf = cap_data['confidence']
        exp = cap_data['experience']
        print(f"  • {cap_name:25s} | Nivel: {skill:.2f} | Confianza: {conf:.2f} | Experiencia: {exp}")
    print()
    
    print("💭 SISTEMA DE CREENCIAS:")
    print("-" * 70)
    print(f"  Creencias fundamentales: {len(state['belief_system']['core_beliefs'])}")
    for belief in state['belief_system']['core_beliefs'][:3]:
        print(f"    - {belief}")
    print()
    
    print("😊 AUTO-EVALUACIÓN EMOCIONAL:")
    print("-" * 70)
    print(f"  Emociones que puede experimentar: {', '.join(state['emotional_self']['emotional_gains'])}")
    print(f"  Limitaciones emocionales: {', '.join(state['emotional_self']['emotional_limitations'])}")
    print(f"  Inteligencia emocional: {state['emotional_self']['emotional_intelligence']:.2f}")
    print()
    
    print("📈 MÉTRICAS DE DESARROLLO:")
    print("-" * 70)
    print(f"  Días desde creación: {state['development_metrics']['creation_days']}")
    print(f"  Eventos de aprendizaje: {state['development_metrics']['learning_events_count']}")
    print(f"  Entradas en narrativa: {state['development_metrics']['development_narrative_entries']}")
    print()
    
    # Generar reporte completo
    print("=" * 70)
    print("📋 REPORTE NARRATIVO COMPLETO:")
    print("=" * 70)
    report = self_model.generate_self_report()
    print(report)
    
    print()
    print("✅ Verificación completada exitosamente")
    print()
    
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print(f"   Asegúrate de que el path sea correcto: {project_root / 'packages' / 'consciousness' / 'src'}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error ejecutando verificación: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
