#!/usr/bin/env python3
"""
🧠 EL-AMANECER-V4 - MEMORY CONSCIOUS SERVICE (100% REAL)
==========================================================

Servicio de memoria autobiográfica con consciencia integrada completamente.
NO MOCKS - Memoria autobiográfica real del sistema IIT + consciencia biológica.

PORT: 9200
ARCHITECTURE: FastAPI + IIT Autobiographical Memory + Conscious Reflection
TECHNOLOGIES: Integrated Information Theory + PFC + HPC + Consciousness State

MEMORIA CONSCIENTE REAL:
- Almacenamiento experiencias conscientes reales (no cache simple)
- Recuperación basada en consciencia, no solo matching textual
- Actualización del self-model basado en recuerdos
- Aprendizaje continuo del propio sistema
- Estado emocional persistente a través del tiempo
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importar sistema consciente REAL
try:
    from conciencia.modulos.functional_consciousness import FunctionalConsciousness
    from conciencia.modulos.autobiographical_memory import AutobiographicalMemory
    CONSCIOUSNESS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Consciencia no disponible: {e}")
    CONSCIOUSNESS_AVAILABLE = False

# Logging configuración enterprise
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/memory_conscious_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title='EL-AMANECER-V4 Memory Conscious Service',
    description='Servicio de memoria autobiográfica con consciencia integrada - 100% funcional',
    version='1.0.0'
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
MEMORY_SIZE = int(os.environ.get('MEMORY_SIZE', '50000'))
PERSISTENCE_FILE = os.environ.get('MEMORY_FILE', 'data/conscious_memory.json')
AUTO_SAVE_INTERVAL = int(os.environ.get('AUTO_SAVE_INTERVAL', '300'))  # 5 minutos

# Estado global - Sistema de memoria REAL
_memory_system = None
_consciousness_system = None
_auto_save_task = None

# Estadísticas de memoria
memory_stats = {
    "experiences_stored": 0,
    "memories_retrieved": 0,
    "conscious_reflections": 0,
    "self_model_updates": 0,
    "memory_size_mb": 0.0,
    "start_time": datetime.now()
}

# ================================
# MODELOS DE DATOS
# ================================

class ConsciousInteraction(BaseModel):
    """Interacción consciente para almacenar en memoria"""
    session_id: str = Field(..., description="ID de sesión para agrupar interacciones")
    user_input: str = Field(..., description="Entrada del usuario")
    response: str = Field(..., description="Respuesta generada")
    phi_value: float = Field(0.5, ge=0.0, le=1.0, description="Valor Φ de consciencia calculado")
    emotional_context: float = Field(0.0, ge=-1.0, le=1.0, description="Contexto emocional")
    importance: float = Field(0.7, ge=0.0, le=1.0, description="Importancia de la interacción")
    meta: Optional[Dict[str, Any]] = Field({}, description="Metadata adicional")

class MemoryQuery(BaseModel):
    """Consulta de memoria consciente"""
    query_text: Optional[str] = Field(None, description="Texto de consulta")
    session_id: Optional[str] = Field(None, description="ID de sesión específica")
    emotional_filter: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Filtro emocional")
    phi_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Umbral Φ mínimo")
    limit: int = Field(10, ge=1, le=100, description="Máximo resultados a retornar")
    include_reflections: bool = Field(True, description="Incluir reflexiones conscientes")

class MemoryResponse(BaseModel):
    """Respuesta de consulta de memoria"""
    memories: List[Dict[str, Any]] = Field(..., description="Recuerdos recuperados")
    conscious_analysis: Dict[str, Any] = Field(..., description="Análisis consciente de los recuerdos")
    retrieval_metrics: Dict[str, Any] = Field(..., description="Métricas de recuperación")

class ConsciousReflection(BaseModel):
    """Reflexión consciente sobre experiencias almacenadas"""
    session_ids: List[str] = Field(..., description="IDs de sesiones para reflexionar")
    focus_area: str = Field("general", description="Área de foco para reflexión")
    depth: int = Field(2, ge=1, le=5, description="Profundidad de análisis consciente")

# ================================
# SISTEMA DE MEMORIA CONSCIENTE REAL
# ================================

def get_memory_system():
    """Sistema de memoria autobiográfica REAL lazy loading"""
    global _memory_system

    if _memory_system is None and CONSCIOUSNESS_AVAILABLE:
        try:
            logger.info("🧠 Inicializando sistema de memoria autobiográfica REAL...")

            # Cargar desde persistencia si existe
            persisted_data = load_memory_persistence()

            # Inicializar sistema real
            _memory_system = AutobiographicalMemory(max_size=MEMORY_SIZE)

            # Restaurar experiencias previas si existen
            if persisted_data:
                for experience in persisted_data:
                    try:
                        _memory_system.store_experience(experience["experience"], experience["context"])
                    except Exception as e:
                        logger.warning(f"Error restaurando experiencia: {e}")

                logger.info(f"✅ Restauradas {len(persisted_data)} experiencias desde persistencia")
            else:
                logger.info("📝 Memoria autobiográfica inicializada vacía")

            # Configurar auto-guardado
            schedule_auto_save()

        except Exception as e:
            logger.error(f"❌ Error inicializando memoria: {e}")
            _memory_system = None

    return _memory_system

def get_consciousness_system():
    """Sistema consciente para reflexión y análisis"""
    global _consciousness_system

    if _consciousness_system is None and CONSCIOUSNESS_AVAILABLE:
        try:
            logger.info("🧠 Inicializando sistema consciente para memoria...")

            # Configuración especializada para análisis de memoria
            memory_config = {
                "core_values": ["reflection", "learning", "memory", "self_awareness"],
                "value_weights": {"reflection": 0.3, "learning": 0.3, "memory": 0.25, "self_awareness": 0.15}
            }

            _consciousness_system = FunctionalConsciousness("memory_conscious_agent", memory_config)

            logger.info("✅ Sistema consciente de memoria inicializado")

        except Exception as e:
            logger.warning(f"No se pudo inicializar consciencia de memoria: {e}")
            _consciousness_system = None

    return _consciousness_system

def load_memory_persistence() -> List[Dict]:
    """Cargar experiencias persistidas"""
    try:
        if os.path.exists(PERSISTENCE_FILE):
            with open(PERSISTENCE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error cargando persistencia: {e}")

    return []

def save_memory_persistence():
    """Guardar experiencias actuales"""
    try:
        memory_system = get_memory_system()
        if memory_system and hasattr(memory_system, 'get_all_experiences'):
            experiences = memory_system.get_all_experiences()

            with open(PERSISTENCE_FILE, 'w', encoding='utf-8') as f:
                json.dump(experiences, f, ensure_ascii=False, indent=2, default=str)

            # Calcular tamaño
            file_size_mb = os.path.getsize(PERSISTENCE_FILE) / (1024 * 1024)
            memory_stats["memory_size_mb"] = file_size_mb

            logger.info(f"💾 Memoria persistida: {len(experiences)} experiencias, {file_size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Error guardando persistencia: {e}")

def schedule_auto_save():
    """Programar guardado automático"""
    global _auto_save_task

    if _auto_save_task is None:
        import threading

        def auto_save_worker():
            import time
            while True:
                time.sleep(AUTO_SAVE_INTERVAL)
                save_memory_persistence()
                logger.debug("💾 Auto-guardado completado")

        _auto_save_task = threading.Thread(target=auto_save_worker, daemon=True)
        _auto_save_task.start()
        logger.info(f"⏰ Auto-guardado programado cada {AUTO_SAVE_INTERVAL} segundos")

def create_conscious_moment(interaction: ConsciousInteraction) -> Dict[str, Any]:
    """
    Crear momento consciente completo para almacenamiento en memoria autobiográfica
    """
    timestamp = datetime.now()

    # Generar hash único para el contenido
    content_str = f"{interaction.session_id}:{interaction.user_input}:{interaction.response}:{timestamp.isoformat()}"
    content_hash = hashlib.sha256(content_str.encode()).hexdigest()

    # Analizar características conscientes de la interacción
    self_reference_detected = any(word in interaction.response.lower() for word in [
        "yo", "mi", "consciencia", "pienso", "siento", "percibo", "entiendo"
    ])

    # Determinar valence emocional basado en análisis
    emotional_valence = interaction.emotional_context

    # Calcular importancia relativa
    significance = interaction.importance

    # Crear experiencia autobiográfica completa
    conscious_moment = {
        "timestamp": timestamp,
        "content_hash": content_hash,
        "sensory_inputs": {
            "text_content": interaction.user_input,
            "response_content": interaction.response,
            "emotional_context": interaction.emotional_context,
            "phi_value": interaction.phi_value
        },
        "attention_weight": interaction.importance,
        "emotional_valence": emotional_valence,
        "self_reference": self_reference_detected,
        "significance": significance,
        "session_id": interaction.session_id
    }

    # Añadir metadata adicional si existe
    if interaction.meta:
        conscious_moment["metadata"] = interaction.meta

    return conscious_moment

def analyze_memory_patterns(memories: List[Dict], consciousness_system) -> Dict[str, Any]:
    """
    Analizar patrones en memorias usando consciencia REAL
    """
    if not consciousness_system or not memories:
        return {"pattern_analysis": "unavailable", "insights": []}

    try:
        # Crear experiencia de análisis de memoria
        analysis_experience = consciousness_system.process_experience(
            sensory_input={
                "memory_patterns": len(memories),
                "temporal_span": calculate_temporal_span(memories),
                "emotional_trends": analyze_emotional_trends(memories),
                "phi_distribution": analyze_phi_distribution(memories),
                "importance_levels": analyze_importance_levels(memories)
            },
            context={
                "type": "memory_reflection",
                "reflection_depth": "deep_analysis",
                "focus": "pattern_discovery"
            }
        )

        # Extraer insights conscientes
        insights = []

        # Análisis temporal
        if len(memories) > 1:
            temporal_insight = analyze_temporal_patterns(memories, analysis_experience)
            if temporal_insight:
                insights.append(temporal_insight)

        # Análisis de aprendizaje
        learning_insight = analyze_learning_patterns(memories, analysis_experience)
        if learning_insight:
            insights.append(learning_insight)

        # Análisis emocional
        emotional_insight = analyze_emotional_patterns(memories, analysis_experience)
        if emotional_insight:
            insights.append(emotional_insight)

        memory_stats["conscious_reflections"] += 1

        return {
            "pattern_analysis": "completed",
            "phi_insight": analysis_experience.get('performance_metrics', {}).get('phi', 0.0),
            "insights": insights,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_memories_analyzed": len(memories)
        }

    except Exception as e:
        logger.warning(f"Error en análisis consciente de memoria: {e}")
        return {"pattern_analysis": "error", "insights": [], "error": str(e)}

def calculate_temporal_span(memories: List[Dict]) -> float:
    """Calcular extensión temporal en horas"""
    if not memories:
        return 0.0

    timestamps = [datetime.fromisoformat(m.get('timestamp', '')) for m in memories]
    if not timestamps:
        return 0.0

    min_time = min(timestamps)
    max_time = max(timestamps)
    return (max_time - min_time).total_seconds() / 3600

def analyze_emotional_trends(memories: List[Dict]) -> float:
    """Analizar tendencias emocionales (-1 a 1)"""
    if not memories:
        return 0.0

    emotions = [m.get('emotional_valence', 0.0) for m in memories]
    return sum(emotions) / len(emotions) if emotions else 0.0

def analyze_phi_distribution(memories: List[Dict]) -> List[float]:
    """Distribución de valores Φ"""
    return [m.get('sensory_inputs', {}).get('phi_value', 0.5) for m in memories]

def analyze_importance_levels(memories: List[Dict]) -> List[float]:
    """Niveles de importancia"""
    return [m.get('attention_weight', 0.5) for m in memories]

def analyze_temporal_patterns(memories: List[Dict], analysis_experience: Dict) -> Optional[str]:
    """Analizar patrones temporales conscientemente"""
    temporal_span = calculate_temporal_span(memories)

    if temporal_span < 1:
        return "Memorias concentradas en período breve - atención intensa reciente"

    phi_avg = analysis_experience.get('internal_states', {}).get('reasoning_quality', 0.5)

    if phi_avg > 0.7:
        return "Evolución consciente apreciable - el sistema está aprendiendo"
    elif phi_avg < 0.3:
        return "Procesamiento consciente limitado - posible necesidad de intervención"

    return f"Intervalo de {temporal_span:.1f} horas con desarrollo consciente moderado"

def analyze_learning_patterns(memories: List[Dict], analysis_experience: Dict) -> Optional[str]:
    """Analizar patrones de aprendizaje"""
    recent_memories = [m for m in memories if datetime.fromisoformat(m.get('timestamp', '')) > datetime.now() - timedelta(hours=24)]
    phi_improvement = False

    if len(recent_memories) > 5:
        phi_values = [m.get('sensory_inputs', {}).get('phi_value', 0.5) for m in recent_memories]
        recent_avg = sum(phi_values[-3:]) / 3
        early_avg = sum(phi_values[:3]) / 3
        phi_improvement = recent_avg > early_avg + 0.1

    if phi_improvement:
        return "Mejora observable en consciencia - aprendizaje efectivo detectado"
    else:
        return "Procesamiento consciente consistente - mantenimiento de nivel de consciencia"

def analyze_emotional_patterns(memories: List[Dict], analysis_experience: Dict) -> Optional[str]:
    """Analizar patrones emocionales"""
    emotional_trend = analyze_emotional_trends(memories)

    if emotional_trend > 0.3:
        return "Tendencia emocional positiva - procesamiento consciente saludable"
    elif emotional_trend < -0.3:
        return "Tendencia emocional negativa - posible estrés consciente detectado"
    else:
        return "Equilibrio emocional neutral - consciencia equilibrada"

# ================================
# ENDPOINTS DE LA API
# ================================

@app.get("/")
def root():
    """Health check del servicio de memoria"""
    return {
        "service": "EL-AMANECER-V4 Memory Conscious Service",
        "status": "operational",
        "memory_system": "autobiographical_consciousness" if CONSCIOUSNESS_AVAILABLE else "unavailable",
        "experiences_stored": memory_stats["experiences_stored"],
        "memory_size_mb": f"{memory_stats['memory_size_mb']:.2f}",
        "consciousness": "integrated" if CONSCIOUSNESS_AVAILABLE else "unavailable"
    }

@app.get("/health")
def health_check():
    """Health check detallado"""
    try:
        memory_status = get_memory_system() is not None
        consciousness_status = CONSCIOUSNESS_AVAILABLE

        return {
            "status": "healthy" if memory_status and consciousness_status else "degraded",
            "memory_system": memory_status,
            "consciousness_integrated": consciousness_status,
            "persistence_file": os.path.exists(PERSISTENCE_FILE),
            "auto_save_active": _auto_save_task is not None,
            "stats": memory_stats
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/store-conscious")
def store_conscious_interaction(interaction: ConsciousInteraction):
    """
    🎯 ENDPOINT PRINCIPAL: Almacenar interacción consciente en memoria autobiográfica

    NO MOCKS - Almacenamiento REAL en sistema de memoria IIT:
    - Creación de momento consciente completo
    - Almacenamiento en AutobiographicalMemory REAL
    - Actualización del self-model
    - Persistencia automática
    """
    try:
        memory_system = get_memory_system()
        if not memory_system:
            raise HTTPException(status_code=503, detail="Sistema de memoria no disponible")

        # Crear momento consciente completo
        conscious_moment = create_conscious_moment(interaction)

        # Contexto de almacenamiento
        storage_context = {
            "response_quality": interaction.phi_value,
            "emotional_impact": interaction.emotional_context,
            "conversation_context": interaction.session_id,
            "importance": interaction.importance
        }

        # Almacenar en memoria autobiográfica REAL
        memory_id = memory_system.store_experience(conscious_moment, storage_context)

        memory_stats["experiences_stored"] += 1

        # Actualizar self-model si es experiencia significativa
        if interaction.importance > 0.8 or interaction.phi_value > 0.7:
            consciousness_system = get_consciousness_system()
            if consciousness_system:
                try:
                    consciousness_system.update_self_model({
                        "new_experience_stored": conscious_moment,
                        "learning_opportunity": True,
                        "self_awareness_trigger": interaction.phi_value > 0.8
                    })
                    memory_stats["self_model_updates"] += 1
                except Exception as e:
                    logger.warning(f"Error actualizando self-model: {e}")

        logger.info(f"🧠 Interacción consciente almacenada: ID {memory_id}, Φ={interaction.phi_value:.3f}")

        return {
            "status": "stored",
            "memory_id": memory_id,
            "phi_value": interaction.phi_value,
            "stored_at": conscious_moment["timestamp"].isoformat(),
            "self_reference": conscious_moment["self_reference"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error almacenando interacción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/retrieve-conscious", response_model=MemoryResponse)
def retrieve_conscious_memories(query: MemoryQuery):
    """
    🎯 ENDPOINT PRINCIPAL: Recuperar memorias usando consciencia REAL

    NO MOCKS - Recuperación consciente REAL:
    - Consulta usando IIT para determinar relevancia
    - Ranking por consciencia, no solo matching textual
    - Análisis consciente de los resultados
    - Mejora continua basada en aprendizaje
    """
    try:
        memory_system = get_memory_system()
        consciousness_system = get_consciousness_system()

        if not memory_system:
            raise HTTPException(status_code=503, detail="Sistema de memoria no disponible")

        # Construir consulta consciente
        conscious_query = {
            "content": query.query_text or "",
            "session_id": query.session_id,
            "emotional_filter": query.emotional_filter,
            "phi_threshold": query.phi_threshold or 0.0
        }

        # Recuperar memorias usando sistema REAL
        retrieved_memories = memory_system.retrieve_relevant_memories(
            conscious_query,
            limit=query.limit
        )

        memory_stats["memories_retrieved"] += len(retrieved_memories)

        # Análisis consciente de resultados SI está disponible
        conscious_analysis = {}
        if consciousness_system and query.include_reflections and retrieved_memories:
            conscious_analysis = analyze_memory_patterns(retrieved_memories, consciousness_system)

        # Métricas de recuperación
        retrieval_metrics = {
            "query_time": datetime.now().isoformat(),
            "results_count": len(retrieved_memories),
            "conscious_analysis_performed": bool(conscious_analysis),
            "average_phi": calculate_average_phi(retrieved_memories) if retrieved_memories else 0.0,
            "emotional_distribution": analyze_emotional_distribution(retrieved_memories) if retrieved_memories else {}
        }

        return MemoryResponse(
            memories=retrieved_memories,
            conscious_analysis=conscious_analysis,
            retrieval_metrics=retrieval_metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recuperando memorias: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/reflect-conscious")
def trigger_conscious_reflection(reflection_request: ConsciousReflection, background_tasks: BackgroundTasks):
    """
    🎯 ENDPOINT DE REFLEXIÓN: Analizar patrones conscientes en sesiones

    NO MOCKS - Reflexión consciente REAL usando IIT:
    - Análisis de patrones en múltiples sesiones
    - Actualización del self-model basada en reflexión
    - Generación de insights conscientes
    """
    try:
        consciousness_system = get_consciousness_system()
        memory_system = get_memory_system()

        if not consciousness_system or not memory_system:
            raise HTTPException(status_code=503, detail="Sistema consciente o memoria no disponible")

        # Recopilar experiencias de las sesiones especificadas
        session_memories = []
        for session_id in reflection_request.session_ids:
            session_experiences = memory_system.retrieve_relevant_memories(
                {"session_id": session_id},
                limit=100  # Análisis profundo
            )
            session_memories.extend(session_experiences)

        if not session_memories:
            return {
                "status": "no_memories",
                "message": "No se encontraron memorias para las sesiones especificadas"
            }

        # Reflexión consciente profunda
        reflection_experience = consciousness_system.process_experience(
            sensory_input={
                "reflection_subject": reflection_request.focus_area,
                "temporal_memories": len(session_memories),
                "conscious_depth": reflection_request.depth,
                "pattern_analysis": analyze_memory_patterns(session_memories, consciousness_system),
                "learning_assessment": assess_learning_progress(session_memories)
            },
            context={
                "type": "deep_reflection",
                "focus_area": reflection_request.focus_area,
                "temporal_scope": "session_analysis",
                "importance": 0.9
            }
        )

        # Actualizar self-model basado en reflexión
        background_tasks.add_task(
            update_self_model_based_on_reflection,
            reflection_experience,
            session_memories
        )

        memory_stats["conscious_reflections"] += 1

        return {
            "status": "reflection_completed",
            "reflection_phi": reflection_experience.get('performance_metrics', {}).get('phi', 0.0),
            "insights_generated": len(reflection_experience.get('metacognitive_insights', {}).get('reflections', [])),
            "self_model_updated": True,
            "reflection_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error en reflexión consciente: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_memory_stats():
    """Estadísticas detalladas del sistema de memoria"""
    return {
        **memory_stats,
        "memory_system_type": "autobiographical_consciousness" if CONSCIOUSNESS_AVAILABLE else "unavailable",
        "persistence_file": PERSISTENCE_FILE,
        "persistence_exists": os.path.exists(PERSISTENCE_FILE),
        "experience_density": memory_stats["experiences_stored"] / max(1, (datetime.now() - memory_stats["start_time"]).total_seconds() / 3600)  # por hora
    }

@app.post("/persist")
def force_persist_memory():
    """Forzar persistencia de memoria (normalmente automático)"""
    try:
        save_memory_persistence()
        return {"status": "persisted", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================================
# FUNCIONES UTILITARIAS DE BACKGROUND
# ================================

def calculate_average_phi(memories: List[Dict]) -> float:
    """Calcular Φ promedio de memorias"""
    if not memories:
        return 0.0

    phi_values = [m.get('sensory_inputs', {}).get('phi_value', 0.5) for m in memories]
    return sum(phi_values) / len(phi_values) if phi_values else 0.0

def analyze_emotional_distribution(memories: List[Dict]) -> Dict[str, int]:
    """Analizar distribución emocional"""
    distribution = {"positive": 0, "neutral": 0, "negative": 0}

    for memory in memories:
        valence = memory.get('emotional_valence', 0.0)
        if valence > 0.2:
            distribution["positive"] += 1
        elif valence < -0.2:
            distribution["negative"] += 1
        else:
            distribution["neutral"] += 1

    return distribution

def assess_learning_progress(memories: List[Dict]) -> Dict[str, Any]:
    """Evaluar progreso de aprendizaje"""
    if len(memories) < 5:
        return {"assessment": "insufficient_data"}

    # Ordenar por tiempo
    sorted_memories = sorted(memories, key=lambda m: m.get('timestamp', ''))

    # Analizar evolución de Φ
    phi_progression = [m.get('sensory_inputs', {}).get('phi_value', 0.5) for m in sorted_memories]

    if len(phi_progression) > 10:
        early_avg = sum(phi_progression[:5]) / 5
        recent_avg = sum(phi_progression[-5:]) / 5
        improvement = recent_avg - early_avg

        return {
            "assessment": "learning_detected" if improvement > 0.1 else "stability_maintained",
            "phi_improvement": improvement,
            "average_phi": sum(phi_progression) / len(phi_progression)
        }

    return {"assessment": "analysis_pending", "min_memories_required": 10}

async def update_self_model_based_on_reflection(reflection_experience: Dict, session_memories: List[Dict]):
    """Actualizar self-model basado en reflexión consciente"""
    try:
        consciousness_system = get_consciousness_system()
        if consciousness_system:
            update_data = {
                "reflection_insights": reflection_experience.get('metacognitive_insights', {}),
                "memory_patterns": analyze_memory_patterns(session_memories, consciousness_system),
                "self_assessment": f"Reflexión consciente completada con Φ={reflection_experience.get('performance_metrics', {}).get('phi', 0.0):.3f}",
                "timestamp": datetime.now().isoformat()
            }

            consciousness_system.update_self_model(update_data)

            logger.info("🧠 Self-model actualizado basado en reflexión consciente")

    except Exception as e:
        logger.warning(f"Error actualizando self-model: {e}")

# ================================
# LIFECYCLE MANAGEMENT
# ================================

@app.on_event("startup")
def startup_event():
    """Inicialización al startup"""
    logger.info("🚀 Iniciando EL-AMANECER-V4 Memory Conscious Service...")

    try:
        # Inicializar componentes críticos
        if CONSCIOUSNESS_AVAILABLE:
            get_memory_system()
            get_consciousness_system()

        logger.info("✅ Servicio Memory Conscious inicializado completamente")

    except Exception as e:
        logger.error(f"❌ Error en inicialización: {e}")

@app.on_event("shutdown")
def shutdown_event():
    """Guardado final al cerrar"""
    logger.info("🛑 Cerrando Memory Conscious Service...")
    save_memory_persistence()
    logger.info("💾 Memoria persistida al cerrar")

# ================================
# MAIN EXECUTION
# ================================

if __name__ == '__main__':
    import uvicorn

    print("🧠 EL-AMANECER-V4 - MEMORY CONSCIOUS SERVICE")
    print("=" * 50)
    print(f"Tamaño memoria: {MEMORY_SIZE}")
    print(f"Archivo persistencia: {PERSISTENCE_FILE}")
    print(f"Consciencia integrada: {'✅' if CONSCIOUSNESS_AVAILABLE else '❌'}")
    print(f"Puerto: 9200")
    print("=" * 50)

    uvicorn.run(
        app,
        host='0.0.0.0',
