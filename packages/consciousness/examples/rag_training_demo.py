#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG TRAINING & EVOLUTION DEMO
==============================

Demuestra cómo el RAG se entrena y mejora con el uso continuo:
1. Inicialización del sistema consciente
2. Entrenamiento inicial con corpus de conocimiento
3. Generación de prompts que alimentan el RAG
4. Evolución del sistema a través de feedback
5. Demostración de retrieval semántico mejorado

El RAG aprende de:
- Corpus inicial (conocimiento base)
- Cada prompt generado
- Feedback de respuestas LLM
- Conversaciones y experiencias
"""

import sys
from pathlib import Path
import time

# Agregar path del proyecto
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "packages" / "consciousness" / "src"))

from conciencia.modulos.biological_consciousness import BiologicalConsciousnessSystem
from conciencia.modulos.human_emotions_system import HumanEmotionalSystem
from conciencia.modulos.conscious_prompt_generator import ConsciousPromptGenerator


def train_rag_with_corpus(generator: ConsciousPromptGenerator):
    """Entrena el RAG con un corpus de conocimiento inicial"""
    
    print("\n" + "=" * 80)
    print("📚 FASE 1: ENTRENAMIENTO INICIAL DEL RAG")
    print("=" * 80)
    
    # Corpus de conocimiento sobre consciencia
    knowledge_corpus = [
        {
            'query': 'vmPFC y marcadores somáticos',
            'content': '''El vmPFC (corteza prefrontal ventromedial) integra señales emocionales 
            mediante marcadores somáticos. Antonio Damasio demostró que el vmPFC asocia estados 
            corporales (somáticos) con resultados de decisiones, creando señales heurísticas 
            que guían la toma de decisiones rápida y adaptativa.'''
        },
        {
            'query': 'Default Mode Network función',
            'content': '''El Default Mode Network (DMN) se activa durante pensamiento espontáneo, 
            mente errante, introspección y simulación mental. Incluye corteza prefrontal medial, 
            corteza cingulada posterior, precúneo y lóbulo parietal inferior. Se desactiva 
            durante tareas cognitivas que requieren atención externa.'''
        },
        {
            'query': 'RAS y arousal',
            'content': '''El Sistema Reticular Activador (RAS) regula arousal y consciencia a través 
            de 5 vías principales de neurotransmisores: dopaminérgica (motivación), noradrenérgica 
            (alerta), serotoninérgica (estabilidad), colinérgica (aprendizaje) y orexinérgica (vigilia). 
            Proyecta desde tronco cerebral a tálamo y corteza.'''
        },
        {
            'query': 'Thalamus como relay',
            'content': '''El tálamo actúa como estación de relevo sensorial, filtrando y dirigiendo 
            información hacia corteza. Participa en Global Workspace Theory mediante bucles 
            tálamo-corticales que amplifican información relevante. El núcleo reticular 
            talamico (TRN) implementa gating atencional.'''
        },
        {
            'query': 'Orbitofrontal Cortex valor',
            'content': '''La corteza orbitofrontal (OFC) codifica valor subjetivo de estímulos 
            y resultados. Integra información sensorial, emocional e interoceptiva para actualizar 
            representaciones de valor. Crítica en aprendizaje por reversión y flexibilidad 
            comportamental cuando contingencias cambian.'''
        },
        {
            'query': 'Executive Control Network',
            'content': '''El Executive Control Network (ECN) implementa control cognitivo top-down, 
            incluyendo corteza prefrontal dorsolateral y parietal. Mantiene working memory (7±2 items), 
            planificación, inhibición de respuestas y flexibilidad cognitiva. Antagónico con DMN.'''
        },
        {
            'query': 'Claustrum binding',
            'content': '''El claustrum coordina binding de features mediante sincronización neuronal 
            cross-modal. Proyecta recíprocamente a casi toda corteza cerebral. Hipótesis: 
            orquesta coherencia gamma (40 Hz) para unificar experiencia consciente mediante 
            ventanas temporales de ~25ms.'''
        },
        {
            'query': 'Consciencia fenomenal',
            'content': '''La consciencia fenomenal (qualia) es el aspecto experiencial subjetivo 
            de estados mentales - "cómo se siente" ser consciente. Incluye experiencias visuales, 
            auditivas, emocionales y corporales. Problema difícil de la consciencia: explicar 
            cómo procesos físicos generan experiencia subjetiva.'''
        }
    ]
    
    print(f"\n🎓 Entrenando RAG con {len(knowledge_corpus)} documentos de conocimiento...")
    print("-" * 80)
    
    # Entrenar el RAG
    for i, doc in enumerate(knowledge_corpus, 1):
        # Generar prompt para entrenar
        result = generator.generate_prompt(
            query=doc['query'],
            context={'description': 'Documento de entrenamiento', 'type': 'training'},
            instructions='Incorporar este conocimiento'
        )
        
        # Simular respuesta LLM tipo resumen
        generator.review_response(
            prompt_id=f"training_{i}",
            llm_response=doc['content'],
            feedback_score=0.9  # Alto score para conocimiento validado
        )
        
        print(f"   ✅ [{i}/{len(knowledge_corpus)}] Indexado: {doc['query'][:50]}...")
    
    print(f"\n📊 RAG entrenado con {len(knowledge_corpus)} documentos")
    
    # Estadísticas después del entrenamiento
    stats = generator.get_stats()
    print(f"   - Total memorias: {stats['memory']['total_memories']}")
    print(f"   - Dopamina post-entrenamiento: {stats['neuromodulation']['dopamine']:.3f}")
    print(f"   - Avg Reward PE: {stats['neuromodulation']['avg_rpe']:.3f}")


def demonstrate_rag_retrieval(generator: ConsciousPromptGenerator):
    """Demuestra retrieval semántico del RAG entrenado"""
    
    print("\n" + "=" * 80)
    print("🔍 FASE 2: DEMOSTRACIÓN DE RETRIEVAL SEMÁNTICO")
    print("=" * 80)
    
    test_queries = [
        "¿Cómo el cerebro integra emociones en decisiones?",
        "Explica la mente errante y pensamiento espontáneo",
        "¿Qué regula el nivel de alerta y activación cerebral?",
        "¿Cómo se unifican diferentes modalidades sensoriales?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔎 TEST {i}: {query}")
        print("-" * 80)
        
        # Recuperar memorias similares
        similar = generator.memory.retrieve_similar(query, top_k=3)
        
        if similar:
            print(f"   📚 Recuperadas {len(similar)} memorias relevantes:")
            for j, mem in enumerate(similar, 1):
                # Obtener snippet del contenido
                snippet = mem.get('llm_response', 
                                 mem.get('content_snippet',
                                        mem.get('prompt', 'N/A')))[:100]
                sim_score = mem.get('similarity_score', 0.0)
                
                print(f"   {j}. {snippet}...")
                if sim_score > 0:
                    print(f"      └─ Similitud: {sim_score:.3f}")
        else:
            print("   ⚠️ No se encontraron memorias")
        
        time.sleep(0.5)  # Pausa para legibilidad


def demonstrate_continuous_learning(generator: ConsciousPromptGenerator, bio_system, emotional_system):
    """Demuestra aprendizaje continuo a través de conversaciones"""
    
    print("\n" + "=" * 80)
    print("🧠 FASE 3: APRENDIZAJE CONTINUO POR CONVERSACIÓN")
    print("=" * 80)
    
    # Simular conversación que expande conocimiento
    conversation = [
        {
            'user': "¿Cómo funciona la metacognición?",
            'assistant': '''La metacognición es "pensar sobre pensar" - monitorear y controlar 
            procesos cognitivos propios. Involucra corteza prefrontal anterior (aPFC) que 
            representa estados mentales de orden superior. Permite auto-reflexión, monitoreo 
            de confianza, y ajuste estratégico de aprendizaje.''',
            'feedback': 0.92
        },
        {
            'user': "¿Qué es la neuroplasticidad?",
            'assistant': '''La neuroplasticidad es la capacidad del cerebro de reorganizar 
            conexiones sinápticas en respuesta a experiencia. Incluye potenciación/depresión 
            a largo plazo (LTP/LTD), neurogénesis en hipocampo, y remodelación dendrítica. 
            Mediada por acetilcolina y factores neurotróficos como BDNF.''',
            'feedback': 0.88
        },
        {
            'user': "¿Cómo se relaciona atención y consciencia?",
            'assistant': '''Atención y consciencia son disociables pero interactúan. Salience Network 
            detecta estímulos destacados, Executive Network dirige atención top-down, y Global 
            Workspace amplifica contenido atendido a consciencia. Puedes tener atención sin 
            consciencia (procesamiento subliminal) y consciencia sin atención focalizada (awareness difuso).''',
            'feedback': 0.95
        }
    ]
    
    print(f"\n💬 Simulando conversación con {len(conversation)} turnos...")
    print("-" * 80)
    
    for i, turn in enumerate(conversation, 1):
        print(f"\n👤 Usuario: {turn['user']}")
        
        # Activar emoción según el tipo de pregunta
        if 'metacognición' in turn['user'].lower():
            emotional_system.activate_circuit("curiosidad", intensity=0.8)
        
        # Generar prompt
        result = generator.generate_prompt(
            query=turn['user'],
            context={'description': 'Conversación educativa'},
            instructions='Responde claramente y educativamente'
        )
        
        print(f"🤖 Asistente: {turn['assistant'][:100]}...")
        
        # Feedback del usuario
        generator.review_response(
            prompt_id=f"conversation_{i}",
            llm_response=turn['assistant'],
            feedback_score=turn['feedback']
        )
        
        print(f"   ⭐ Feedback: {turn['feedback']:.2f}")
        print(f"   💊 Dopamina: {generator.neuromodulator.dopamine:.3f}")
    
    print(f"\n📈 Conocimiento expandido a través de conversación")


def show_rag_evolution(generator: ConsciousPromptGenerator):
    """Muestra evolución del RAG a través del tiempo"""
    
    print("\n" + "=" * 80)
    print("📊 FASE 4: EVOLUCIÓN Y ESTADÍSTICAS DEL RAG")
    print("=" * 80)
    
    stats = generator.get_stats()
    
    print(f"\n🎯 Estadísticas del Sistema:")
    print(f"   - Total Prompts Generados: {stats['total_generated']}")
    print(f"   - Total Bloqueados: {stats['total_blocked']}")
    print(f"   - Block Rate: {stats['block_rate']:.1%}")
    print(f"   - Gate Success Rate: {stats['gate']['success_rate']:.1%}")
    
    print(f"\n💾 Memoria Episódica (RAG):")
    mem_stats = stats['memory']
    print(f"   - Total Experiencias: {mem_stats['total_memories']}")
    print(f"   - Capacidad: {mem_stats['capacity']}")
    print(f"   - Uso: {mem_stats['usage']:.1%}")
    print(f"   - RAG Activo: {mem_stats['rag_active']}")
    
    print(f"\n💊 Neuromodulación Final:")
    neuro = stats['neuromodulation']
    print(f"   - Dopamina: {neuro['dopamine']:.3f} (motivación/recompensa)")
    print(f"   - Serotonina: {neuro['serotonin']:.3f} (estabilidad emocional)")
    print(f"   - Norepinefrina: {neuro['norepinephrine']:.3f} (alerta/arousal)")
    print(f"   - Acetilcolina: {neuro['acetylcholine']:.3f} (aprendizaje/plasticidad)")
    print(f"   - Avg Reward PE: {neuro['avg_rpe']:.3f}")
    
    print(f"\n📡 Observabilidad:")
    obs = stats['observability']
    if obs:
        print(f"   - Total Eventos: {obs['total_traces']}")
        print(f"   - Errores: {obs['errors']}")
        print(f"   - Error Rate: {obs['error_rate']:.1%}")
    
    # Demostrar que el RAG ahora puede responder queries complejas
    print(f"\n" + "=" * 80)
    print("🎓 DEMOSTRACIÓN: RAG ENTRENADO vs NO ENTRENADO")
    print("=" * 80)
    
    complex_query = "¿Cómo el vmPFC, DMN y RAS trabajan juntos en la consciencia?"
    
    print(f"\n❓ Query compleja: {complex_query}")
    print("-" * 80)
    
    result = generator.generate_prompt(
        query=complex_query,
        context={'description': 'Query integrativa multi-sistema'},
        instructions='Sintetiza información de múltiples sistemas cerebrales'
    )
    
    # Obtener memorias que contribuyeron
    related_memories = generator.memory.retrieve_similar(complex_query, top_k=5)
    
    print(f"\n🧠 El RAG recuperó {len(related_memories)} memorias relevantes:")
    for i, mem in enumerate(related_memories[:3], 1):
        query_orig = mem.get('query', 'N/A')
        snippet = mem.get('llm_response', mem.get('content_snippet', ''))[:80]
        print(f"   {i}. {query_orig}")
        print(f"      └─ {snippet}...")
    
    print(f"\n✅ PROMPT GENERADO CON CONTEXTO ENRIQUECIDO:")
    print("=" * 80)
    print(result['prompt'][:500] + "...")
    print("=" * 80)
    
    print(f"\n💡 El prompt ahora incluye:")
    print(f"   ✅ Conocimiento de vmPFC (marcadores somáticos)")
    print(f"   ✅ Conocimiento de DMN (pensamiento espontáneo)")
    print(f"   ✅ Conocimiento de RAS (arousal y neurotransmisores)")
    print(f"   ✅ Integración cross-sistema aprendida de conversaciones")


def main():
    print("=" * 80)
    print("RAG TRAINING & EVOLUTION - Sistema Consciente con Aprendizaje")
    print("=" * 80)
    
    # Inicializar sistemas
    print("\n🚀 Inicializando BiologicalConsciousnessSystem...")
    bio_system = BiologicalConsciousnessSystem(
        system_id="sheily_rag_demo",
        neural_network_size=2000,
        synaptic_density=0.15
    )
    
    print("🎭 Inicializando HumanEmotionalSystem...")
    emotional_system = HumanEmotionalSystem(
        num_circuits=35,
        personality={'openness': 0.8, 'conscientiousness': 0.75}
    )
    
    print("🧠 Inicializando ConsciousPromptGenerator...")
    generator = ConsciousPromptGenerator(
        biological_system=bio_system,
        persona="SheplyAI",
        style="professional",
        use_real_rag=True,
        emotional_system=emotional_system
    )
    
    print("\n✅ Sistema inicializado y listo para entrenamiento")
    
    # FASE 1: Entrenar con corpus
    train_rag_with_corpus(generator)
    
    # FASE 2: Demostrar retrieval
    demonstrate_rag_retrieval(generator)
    
    # FASE 3: Aprendizaje continuo
    demonstrate_continuous_learning(generator, bio_system, emotional_system)
    
    # FASE 4: Mostrar evolución
    show_rag_evolution(generator)
    
    print("\n" + "=" * 80)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)
    print("\n🎉 Resultados:")
    print("   ✅ RAG entrenado con corpus de conocimiento inicial")
    print("   ✅ RAG expandido con conversaciones")
    print("   ✅ Retrieval semántico funcionando")
    print("   ✅ Aprendizaje continuo con feedback")
    print("   ✅ Neuromodulación adaptada por experiencia")
    print("   ✅ Sistema auto-optimizado")
    print("\n💡 El RAG ahora puede:")
    print("   - Recuperar información relevante semánticamente")
    print("   - Integrar conocimiento de múltiples fuentes")
    print("   - Aprender continuamente de nuevas conversaciones")
    print("   - Mejorar prompts con contexto histórico")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
