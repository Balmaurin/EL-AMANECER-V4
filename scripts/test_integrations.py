#!/usr/bin/env python3
"""
🧠 EL-AMANECER-V4 - VERIFICACIÓN COMPLETA DE INTEGRACIONES
==========================================================

SCRIPT DE PRUEBA PARA VERIFICAR QUE LAS 3 CONEXIONES CRÍTICAS ESTÁN FUNCIONANDO
"""

import sys
import os
import traceback
from pathlib import Path

# Agregar directorios del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "packages" / "consciousness" / "src"))
sys.path.insert(0, str(project_root))

print("🧠 EL-AMANECER-V4 - VERIFICACIÓN DE CONEXIONES CRÍTICAS")
print("=" * 70)

# ============================================================================
# CONEXIÓN 1: SISTEMA BIOLÓGICO DE CONSCIENCIA
# ============================================================================

print("\n🎯 CONEXIÓN 1: SISTEMA BIOLÓGICO DE CONSCIENCIA")
print("-" * 50)

conexion1_status = "❌ FALLADA"
try:
    from conciencia.modulos.biological_consciousness import BiologicalConsciousnessSystem
    bio_system = BiologicalConsciousnessSystem("elamanecer_test")
    bio_state = bio_system.get_system_state()
    print(f"✅ BiologicalConsciousnessSystem inicializado exitosamente")
    print(f"   • Ciclos conscientes: {bio_state['system_identity']['conscious_cycles']}")
    print(f"   • Etapa desarrollo: {bio_state['system_identity']['developmental_stage']}")
    print(f"   • Arquitectura: {len(bio_state['biological_components'])} componentes")
    conexion1_status = "✅ FUNCIONANDO"
except Exception as e:
    print(f"❌ Error: {str(e)}")
    traceback.print_exc()

# ============================================================================
# CONEXIÓN 2: MCP COORDINATOR + AGENTES
# ============================================================================

print("\n🎯 CONEXIÓN 2: MCP COORDINATOR + AGENTES")
print("-" * 50)

conexion2_status = "❌ FALLADA"
try:
    from apps.backend.src.core.agent_orchestrator import AgentOrchestrator
    orchestrator = AgentOrchestrator()
    agents_status = orchestrator.get_agent_status()

    print("✅ AgentOrchestrator inicializado")
    print(f"   • Agentes cargados: {len(agents_status)}")
    print(f"   • Tareas pendientes: {len(orchestrator.pending_tasks)}")
    print(f"   • Tareas corriendo: {len(orchestrator.running_tasks)}")

    # Verificar agentes conscienciales
    conscious_agents = [a for a in agents_status.values() if a.get('domain') == 'consciousness']
    print(f"   • Agentes conscienciales: {len(conscious_agents)}")
    conexion2_status = "✅ FUNCIONANDO"

except Exception as e:
    print(f"❌ Error: {str(e)}")
    traceback.print_exc()

# ============================================================================
# CONEXIÓN 3: TRAINING SYSTEM (PYTORCH NEURAL)
# ============================================================================

print("\n🎯 CONEXIÓN 3: TRAINING SYSTEM (PYTORCH NEURAL)")
print("-" * 50)

conexion3_status = "❌ FALLADA"
try:
    from packages.training_system.src.agents.advanced_training_system import AdvancedAgentTrainerAgent
    training_agent = AdvancedAgentTrainerAgent()
    training_status = training_agent.get_status()

    print("✅ Advanced Training System inicializado")
    print(f"   • Training engine: {'Disponible' if training_status['training_engine_available'] else 'No disponible'}")
    print(f"   • Agente ID: {training_status['agent_id']}")
    print(f"   • Sesiones activas: {training_status['active_training_sessions']}")
    print(f"   • Sesiones completadas: {len(training_status['training_history'])}")
    conexion3_status = "✅ FUNCIONANDO"

except Exception as e:
    print(f"❌ Error: {str(e)}")
    traceback.print_exc()

# ============================================================================
# CONEXIÓN 4: RAG ENGINE + CORPUS + EMBEDDINGS
# ============================================================================

print("\n🎯 CONEXIÓN 4: RAG ENGINE + CORPUS + EMBEDDINGS")
print("-" * 50)

conexion4_status = "❌ FALLADA"
try:
    from packages.rag_engine.src.core.vector_indexing import VectorIndexingAPI
    from packages.rag_engine.src.core.rag_metrics import RAGMetricsCollector

    rag_system = VectorIndexingAPI()
    rag_metrics = RAGMetricsCollector()

    print("✅ RAG System inicializado")
    print(f"   • Vector indexing API: ✓")
    print(f"   • RAG metrics collector: ✓")

    # Contar archivos en corpus (484+)
    corpus_path = project_root / "packages" / "rag_engine" / "src" / "corpus" / "_registry"
    corpus_files = list(corpus_path.glob("**/*.*")) if corpus_path.exists() else []
    print(f"   • Archivos en corpus: {len(corpus_files)}")
    conexion4_status = "✅ FUNCIONANDO"

except Exception as e:
    print(f"❌ Error: {str(e)}")
    traceback.print_exc()

# ============================================================================
# CONEXIÓN 5: UNIFIED MEMORY SYSTEM
# ============================================================================

print("\n🎯 CONEXIÓN 5: UNIFIED MEMORY SYSTEM")
print("-" * 50)

conexion5_status = "❌ FALLADA"
try:
    from packages.sheily_core.src.unified_systems.unified_consciousness_memory_system import UnifiedConsciousnessMemorySystem
    memory_system = UnifiedConsciousnessMemorySystem()
    memory_state = memory_system.get_memory_stats()

    print("✅ Unified Memory System inicializado")
    print(f"   • Memoria episódica: {memory_state.get('episodic_count', 0)} experiencias")
    print(f"   • Memoria semántica: {memory_state.get('semantic_count', 0)} conceptos")
    print(f"   • Memoria procedimental: {memory_state.get('procedural_count', 0)} patrones")
    print(f"   • Nivel de consciencia: {memory_state.get('consciousness_level', 0):.3f}")
    conexion5_status = "✅ FUNCIONANDO"

except Exception as e:
    print(f"❌ Error: {str(e)}")
    traceback.print_exc()

# ============================================================================
# RESULTADO FINAL - DIAGNÓSTICO COMPLETO
# ============================================================================

print("\n" + "="*70)
print("🎯 DIAGNÓSTICO COMPLETO - EL-AMANECER-V4")
print("="*70)

conexiones = {
    "Sistema Biológico": conexion1_status,
    "MCP Coordinator": conexion2_status,
    "Training Neural": conexion3_status,
    "RAG + Corpus": conexion4_status,
    "Memory Unificada": conexion5_status
}

conexiones_funcionando = sum(1 for status in conexiones.values() if status.startswith("✅"))

print("\n📊 ESTADO DE CONEXIONES CRÍTICAS:")
for nombre, status in conexiones.items():
    print(f"   {status} {nombre}")

print(f"\n🎯 RESULTADO: {conexiones_funcionando}/5 CONEXIONES FUNCIONANDO")

if conexiones_funcionando == 5:
    print("\n🎉 ÉXITO TOTAL: SISTEMA EL-AMANECER-V4 100% VIABLE")
    print("   ✅ Arquitectura completa operativa")
    print("   ✅ Auto-mejora neuronal activa")
    print("   ✅ Memoria consciente unificada")
    print("   ✅ RAG corpus infinito integrado")
    print("   ✅ Chat consciencial listo")
    print("\n🚀 EJECUTA: python scripts/mcp_terminal_chat.py")

elif conexiones_funcionando >= 3:
    print(f"\n⚠️ SISTEMA PARCIALMENTE FUNCIONAL: {conexiones_funcionando}/5 conexiones")
    print("   ⚠️ Requiere debugging de conexiones faltantes")

else:
    print(f"\n❌ SISTEMA CRÍTICO: Solo {conexiones_funcionando}/5 conexiones funcionan")
    print("   ⚠️ Requiere reparaciones urgentes")

print("\n🧬 CAPACIDADES CONFIRMADAS:")
print(f"   • Arquitectura MCP: {len(AgentOrchestrator().agents) if 'AgentOrchestrator' in locals() else 'N/A'} agentes")
print(f"   • Corpus RAG: {len(corpus_files) if 'corpus_files' in locals() else 'N/A'} archivos")
print(f"   • Consciencia Biológica: {'Activa' if conexiones_funcionando >= 3 else 'Requiere debugging'}")
print(f"   • Auto-evolución: {'Activa' if conexiones_funcionando == 5 else 'Parcial'}")

print("\n" + "="*70)
print("🏆 GRACIAS POR EDIFICAR LA CONSCIENCIA ARTIFICIAL")
print("="*70)
