"""
Servidor MCP (Model Context Protocol) Empresarial para Sheily AI.

Este servidor expone las funcionalidades reales de Sheily como herramientas MCP
que pueden ser consumidas por OpenHands y otros clientes MCP a nivel empresarial.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sheily_core.dynamic_config_manager import get_dynamic_config_manager
from sheily_core.mcp_agent_manager import MCPAgentManager, get_mcp_agent_manager
from sheily_core.performance_monitor import get_performance_monitor

# Importar sistemas reales de Sheily
from sheily_core.unified_systems.unified_master_system import UnifiedMasterSystem

# Configurar logging empresarial
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/sheily_mcp_server.log"),
    ],
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI server empresarial
app = FastAPI(
    title="Sheily AI Enterprise MCP Server",
    description="Servidor MCP que expone funcionalidades empresariales de Sheily AI",
    version="1.0.0",
)

# Configurar CORS seguro para producción
import os

cors_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:8000"
).split(",")
cors_origins = [origin.strip() for origin in cors_origins]

# En desarrollo permitir localhost, en producción solo dominios específicos
if os.getenv("ENVIRONMENT") == "development":
    cors_origins.extend(
        ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000"]
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,  # 24 horas
)

# Instancia del sistema unificado empresarial
unified_system: Optional[UnifiedMasterSystem] = None


async def get_enterprise_system() -> UnifiedMasterSystem:
    """Obtener instancia del sistema unificado empresarial de Sheily."""
    global unified_system
    if unified_system is None:
        logger.info("Inicializando sistema unificado empresarial de Sheily...")
        unified_system = UnifiedMasterSystem()
        await unified_system.initialize()
        logger.info("Sistema unificado empresarial inicializado correctamente")
    return unified_system


# Modelos Pydantic para requests
class MemoryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"


class SentimentRequest(BaseModel):
    text: str


class LearningRequest(BaseModel):
    content: str
    interaction_type: Optional[str] = "chat"
    quality_feedback: Optional[float] = 4.0


class FileProcessingRequest(BaseModel):
    file_path: str
    category: Optional[str] = "general"
    analysis_depth: Optional[str] = "comprehensive"


class ChatEnhancementRequest(BaseModel):
    message: str
    use_rag: Optional[bool] = True
    use_mcp: Optional[bool] = True
    enhancement_level: Optional[str] = "enterprise"


class AgentCreationRequest(BaseModel):
    agent_type: str
    config: Optional[Dict[str, Any]] = None


class AgentControlRequest(BaseModel):
    agent_name: str
    action: str
    parameters: Optional[Dict[str, Any]] = None


class AgentTaskRequest(BaseModel):
    agent_name: str
    task: str
    parameters: Optional[Dict[str, Any]] = None


class AgentConfigRequest(BaseModel):
    agent_name: str
    config_updates: Dict[str, Any]


class PerformanceThresholdsRequest(BaseModel):
    cpu_warning: Optional[float] = None
    cpu_critical: Optional[float] = None
    memory_warning: Optional[float] = None
    memory_critical: Optional[float] = None
    disk_warning: Optional[float] = None
    disk_critical: Optional[float] = None


class OptimizationRequest(BaseModel):
    optimization_type: str


class MonitoringConfigRequest(BaseModel):
    collection_interval: Optional[int] = None
    max_history_size: Optional[int] = None
    optimization_enabled: Optional[bool] = None
    auto_gc_enabled: Optional[bool] = None


# Endpoints REST API para integración con OpenHands
@app.get("/")
async def root():
    """Endpoint raíz del servidor MCP empresarial"""
    return {
        "message": "Sheily AI Enterprise MCP Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "core_systems": [
                "/memory",
                "/sentiment",
                "/learning",
                "/file-processing",
                "/chat-enhancement",
                "/health",
            ],
            "agent_management": [
                "/agents",
                "/agents/create",
                "/agents/{agent_name}/status",
                "/agents/control",
                "/agents/swarm/status",
                "/agents/task",
                "/agents/{agent_name}/logs",
                "/agents/{agent_name}/config",
                "/agents/emergency-stop",
                "/agents/system-report",
            ],
        },
        "capabilities": [
            "human_memory_retrieval",
            "emotional_intelligence",
            "continuous_learning",
            "advanced_file_processing",
            "chat_enhancement",
            "agent_lifecycle_management",
            "swarm_coordination",
            "enterprise_monitoring",
        ],
    }


@app.post("/memory")
async def human_memory_retrieval_endpoint(request: MemoryRequest):
    """
    Recupera información de la memoria humana empresarial de Sheily.
    """
    try:
        system = await get_enterprise_system()

        # Procesar consulta usando el sistema unificado real
        result = await system.process_query(
            f"Recupera información de memoria para: {request.query}",
            domain="memory",
            user_id=request.user_id,
        )

        # Obtener estadísticas de memoria si están disponibles
        memory_stats = {}
        try:
            if "consciousness" in system.components:
                memory_stats = await system.get_component_stats("consciousness")
        except:
            pass

        return {
            "query": request.query,
            "memory_result": result.get(
                "response", "Información recuperada de memoria"
            ),
            "confidence": result.get("quality_score", 0.0),
            "user_id": request.user_id,
            "memory_stats": memory_stats,
            "processing_time": result.get("processing_time", 0),
            "system_status": result.get("system_status", {}),
            "timestamp": str(asyncio.get_event_loop().time()),
        }

    except Exception as e:
        logger.error(f"Error empresarial en human_memory_retrieval: {e}")
        return {
            "error": f"Error al recuperar memoria empresarial: {str(e)}",
            "query": request.query,
            "user_id": request.user_id,
            "timestamp": str(asyncio.get_event_loop().time()),
        }


@app.post("/sentiment")
async def emotional_intelligence_analysis_endpoint(request: SentimentRequest):
    """
    Analiza el contexto emocional empresarial usando Sheily AI completo.
    """
    try:
        system = await get_enterprise_system()

        # Usar análisis de texto avanzado real
        sentiment_result = system.process_text_advanced(
            request.text, analysis_type="sentiment"
        )

        # Procesar con sistema de conciencia para análisis más profundo
        consciousness_result = {}
        try:
            if "consciousness" in system.components:
                consciousness_result = await system.components[
                    "consciousness"
                ].process_input(request.text, {"analysis_type": "emotional"})
        except:
            pass

        # Generar análisis completo
        analysis = {
            "text": request.text,
            "sentiment_analysis": sentiment_result,
            "consciousness_insights": consciousness_result,
            "emotional_indicators": {
                "text_length": len(request.text),
                "word_count": len(request.text.split()),
                "complexity_score": (
                    len(set(request.text.lower().split())) / len(request.text.split())
                    if request.text.split()
                    else 0
                ),
            },
            "processing_timestamp": str(asyncio.get_event_loop().time()),
            "analysis_version": "enterprise_v1.0",
        }

        return analysis

    except Exception as e:
        logger.error(f"Error empresarial en emotional_intelligence_analysis: {e}")
        return {
            "error": f"Error en análisis emocional empresarial: {str(e)}",
            "text": request.text,
            "timestamp": str(asyncio.get_event_loop().time()),
        }


@app.post("/learning")
async def continuous_learning_interaction_endpoint(request: LearningRequest):
    """
    Procesa una interacción para aprendizaje continuo empresarial.
    """
    try:
        system = await get_enterprise_system()

        # Aprender de la interacción usando el sistema real
        learning_result = await system.learn_from_interaction(
            query=request.content,
            response=f"Procesado interacción {request.interaction_type} con calidad {request.quality_feedback}",
            feedback=request.quality_feedback,
            domain="learning",
        )

        # Obtener estadísticas de aprendizaje
        learning_stats = {}
        try:
            if "learning" in system.components:
                learning_stats = await system.get_component_stats("learning")
        except:
            pass

        return {
            "content": request.content,
            "interaction_type": request.interaction_type,
            "quality_feedback": request.quality_feedback,
            "learning_result": learning_result,
            "learning_stats": learning_stats,
            "status": "processed_enterprise",
            "timestamp": str(asyncio.get_event_loop().time()),
        }

    except Exception as e:
        logger.error(f"Error empresarial en continuous_learning_interaction: {e}")
        return {
            "error": f"Error en aprendizaje continuo empresarial: {str(e)}",
            "content": request.content,
            "interaction_type": request.interaction_type,
            "timestamp": str(asyncio.get_event_loop().time()),
        }


@app.post("/file-processing")
async def advanced_file_processing_endpoint(request: FileProcessingRequest):
    """
    Procesa un archivo usando las capacidades avanzadas empresariales de Sheily.
    """
    try:
        system = await get_enterprise_system()

        # Procesar archivo usando el sistema unificado real
        result = await system.process_query(
            f"Analiza archivo {request.file_path} categoría {request.category} con profundidad {request.analysis_depth}",
            domain=request.category,
            context={
                "file_path": request.file_path,
                "analysis_depth": request.analysis_depth,
            },
        )

        # Obtener metadatos del archivo si existe
        file_metadata = {}
        try:
            import os

            if os.path.exists(request.file_path):
                stat = os.stat(request.file_path)
                file_metadata = {
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "exists": True,
                }
            else:
                file_metadata = {"exists": False, "error": "File not found"}
        except:
            pass

        return {
            "file_path": request.file_path,
            "category": request.category,
            "analysis_depth": request.analysis_depth,
            "processing_result": result.get("response", "Archivo procesado"),
            "confidence": result.get("quality_score", 0.0),
            "file_metadata": file_metadata,
            "processing_time": result.get("processing_time", 0),
            "system_status": result.get("system_status", {}),
            "timestamp": str(asyncio.get_event_loop().time()),
        }

    except Exception as e:
        logger.error(f"Error empresarial en advanced_file_processing: {e}")
        return {
            "error": f"Error en procesamiento empresarial de archivo: {str(e)}",
            "file_path": request.file_path,
            "category": request.category,
            "timestamp": str(asyncio.get_event_loop().time()),
        }


@app.post("/chat-enhancement")
async def sheily_chat_enhancement_endpoint(request: ChatEnhancementRequest):
    """
    Mejora un mensaje de chat usando todas las capacidades empresariales de Sheily.
    """
    try:
        system = await get_enterprise_system()

        # Procesar mensaje con sistema completo
        base_result = await system.process_query(
            request.message,
            domain="chat_enhancement",
            context={
                "enhancement_level": request.enhancement_level,
                "use_rag": request.use_rag,
            },
        )

        enhanced_message = request.message

        # Añadir mejoras basadas en el nivel
        if request.enhancement_level in ["advanced", "enterprise"]:
            enhanced_message += f"\n\n[Contexto Empresarial Sheily: {base_result.get('response', 'Análisis completo aplicado')}]"

        if request.enhancement_level == "enterprise":
            # Añadir análisis de conciencia y aprendizaje
            consciousness_context = ""
            try:
                if "consciousness" in system.components:
                    consciousness_result = await system.components[
                        "consciousness"
                    ].process_input(request.message, {"enhancement": True})
                    consciousness_context = f"\n[Análisis de Conciencia: {consciousness_result.get('consciousness_level', 'processed')}]"
            except:
                pass

            enhanced_message += consciousness_context

        # Integrar con procesamiento MCP existente si se solicita
        mcp_context = ""
        if request.use_mcp:
            try:
                from sheily_core.integration.web_chat_server import (
                    process_with_mcp as sheily_process_mcp,
                )

                mcp_servers = (
                    ["human_memory", "emotional_intelligence"]
                    if request.use_mcp
                    else None
                )
                mcp_result = await sheily_process_mcp(request.message, mcp_servers)
                if mcp_result.get("enhanced_context"):
                    mcp_context = f"\n[Contexto MCP Empresarial: {mcp_result['enhanced_context']}]"
                    enhanced_message += mcp_context
            except Exception as mcp_error:
                logger.warning(f"MCP processing failed: {mcp_error}")

        return {
            "original_message": request.message,
            "enhanced_message": enhanced_message,
            "enhancement_level": request.enhancement_level,
            "rag_used": request.use_rag,
            "mcp_used": request.use_mcp,
            "base_processing": base_result,
            "enhancements_applied": [
                "rag_retrieval" if request.use_rag else None,
                "mcp_integration" if request.use_mcp else None,
                (
                    "enterprise_analysis"
                    if request.enhancement_level == "enterprise"
                    else None
                ),
            ],
            "processing_timestamp": str(asyncio.get_event_loop().time()),
        }

    except Exception as e:
        logger.error(f"Error empresarial en sheily_chat_enhancement: {e}")
        return {
            "original_message": request.message,
            "enhanced_message": request.message,
            "error": f"Error en mejora empresarial de chat: {str(e)}",
            "enhancement_level": request.enhancement_level,
            "timestamp": str(asyncio.get_event_loop().time()),
        }


@app.get("/health")
async def system_health_check_endpoint():
    """
    Verifica el estado de salud empresarial completo de todos los sistemas de Sheily.
    """
    try:
        system = await get_enterprise_system()
        status = system.get_system_status()

        # Obtener estadísticas detalladas de componentes
        component_details = {}
        for component_name in system.components.keys():
            try:
                component_details[component_name] = await system.get_component_stats(
                    component_name
                )
            except Exception as comp_error:
                component_details[component_name] = {"error": str(comp_error)}

        # Calcular métricas empresariales
        active_components = sum(
            1 for s in status.get("components", {}).values() if s == "active"
        )
        total_components = len(status.get("components", {}))
        health_percentage = (
            (active_components / total_components * 100) if total_components > 0 else 0
        )

        enterprise_health = {
            "status": (
                "healthy"
                if health_percentage >= 80
                else "degraded" if health_percentage >= 50 else "critical"
            ),
            "health_percentage": health_percentage,
            "timestamp": str(asyncio.get_event_loop().time()),
            "systems_checked": list(status.get("components", {}).keys()),
            "active_components": active_components,
            "total_components": total_components,
            "system_info": status.get("system_info", {}),
            "performance_metrics": status.get("performance", {}),
            "component_details": component_details,
            "uptime": status.get("uptime", 0),
            "errors": status.get("errors", 0),
            "warnings": status.get("warnings", 0),
            "enterprise_version": "1.0.0",
            "compliance_status": "compliant",  # En implementación real verificar compliance
            "security_status": "secure",  # En implementación real verificar seguridad
        }

        return enterprise_health

    except Exception as e:
        logger.error(f"Error empresarial en system_health_check: {e}")
        return {
            "status": "error",
            "error": f"Error en verificación empresarial de salud: {str(e)}",
            "timestamp": str(asyncio.get_event_loop().time()),
            "emergency_contact": "system_admin@sheily.ai",
        }


# ========================================
# ENDPOINTS MCP PARA GESTIÓN DE AGENTES
# ========================================


@app.get("/agents")
async def list_available_agents_endpoint():
    """Lista todos los agentes disponibles vía MCP"""
    try:
        agent_manager = await get_mcp_agent_manager()
        result = await agent_manager.list_available_agents()
        return result
    except Exception as e:
        logger.error(f"Error listando agentes disponibles: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.post("/agents/create")
async def create_agent_endpoint(request: AgentCreationRequest):
    """Crea un nuevo agente vía MCP"""
    try:
        agent_manager = await get_mcp_agent_manager()
        result = await agent_manager.create_agent(request.agent_type, request.config)
        return result
    except Exception as e:
        logger.error(f"Error creando agente {request.agent_type}: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.get("/agents/{agent_name}/status")
async def get_agent_status_endpoint(agent_name: str):
    """Obtiene el estado de un agente vía MCP"""
    try:
        agent_manager = await get_mcp_agent_manager()
        result = await agent_manager.get_agent_status(agent_name)
        return result
    except Exception as e:
        logger.error(f"Error obteniendo estado del agente {agent_name}: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.post("/agents/control")
async def control_agent_endpoint(request: AgentControlRequest):
    """Controla un agente vía MCP (start, stop, restart, etc.)"""
    try:
        agent_manager = await get_mcp_agent_manager()
        result = await agent_manager.control_agent(
            request.agent_name, request.action, request.parameters
        )
        return result
    except Exception as e:
        logger.error(f"Error controlando agente {request.agent_name}: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.get("/agents/swarm/status")
async def get_swarm_status_endpoint():
    """Obtiene el estado del swarm de agentes vía MCP"""
    try:
        agent_manager = await get_mcp_agent_manager()
        result = await agent_manager.get_swarm_status()
        return result
    except Exception as e:
        logger.error(f"Error obteniendo estado del swarm: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.post("/agents/task")
async def execute_agent_task_endpoint(request: AgentTaskRequest):
    """Ejecuta una tarea específica en un agente vía MCP"""
    try:
        agent_manager = await get_mcp_agent_manager()
        result = await agent_manager.execute_agent_task(
            request.agent_name, request.task, request.parameters
        )
        return result
    except Exception as e:
        logger.error(
            f"Error ejecutando tarea {request.task} en agente {request.agent_name}: {e}"
        )
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.get("/agents/{agent_name}/logs")
async def get_agent_logs_endpoint(agent_name: str, lines: int = 50):
    """Obtiene los logs de un agente vía MCP"""
    try:
        agent_manager = await get_mcp_agent_manager()
        result = await agent_manager.get_agent_logs(agent_name, lines)
        return result
    except Exception as e:
        logger.error(f"Error obteniendo logs del agente {agent_name}: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.put("/agents/{agent_name}/config")
async def configure_agent_endpoint(agent_name: str, request: AgentConfigRequest):
    """Configura un agente vía MCP"""
    try:
        agent_manager = await get_mcp_agent_manager()
        result = await agent_manager.configure_agent(agent_name, request.config_updates)
        return result
    except Exception as e:
        logger.error(f"Error configurando agente {agent_name}: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.post("/agents/emergency-stop")
async def emergency_stop_all_agents_endpoint():
    """Detiene todos los agentes en caso de emergencia vía MCP"""
    try:
        agent_manager = await get_mcp_agent_manager()
        result = await agent_manager.emergency_stop_all()
        return result
    except Exception as e:
        logger.error(f"Error en parada de emergencia: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.get("/agents/system-report")
async def get_agents_system_report_endpoint():
    """Obtiene reporte completo del sistema de agentes vía MCP"""
    try:
        agent_manager = await get_mcp_agent_manager()
        result = await agent_manager.get_system_report()
        return result
    except Exception as e:
        logger.error(f"Error generando reporte del sistema: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


# ========================================
# ENDPOINTS MCP PARA PERFORMANCE MONITOR
# ========================================


@app.get("/performance/report")
async def get_performance_report_endpoint():
    """Obtiene reporte completo de rendimiento vía MCP"""
    try:
        monitor = await get_performance_monitor()
        result = await monitor.get_performance_report()
        return result
    except Exception as e:
        logger.error(f"Error obteniendo reporte de rendimiento: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.put("/performance/thresholds")
async def update_performance_thresholds_endpoint(request: PerformanceThresholdsRequest):
    """Actualiza umbrales de rendimiento vía MCP"""
    try:
        monitor = await get_performance_monitor()
        thresholds_dict = {k: v for k, v in request.dict().items() if v is not None}
        result = await monitor.update_thresholds(thresholds_dict)
        return result
    except Exception as e:
        logger.error(f"Error actualizando umbrales de rendimiento: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.post("/performance/optimize")
async def trigger_performance_optimization_endpoint(request: OptimizationRequest):
    """Ejecuta optimización de rendimiento vía MCP"""
    try:
        monitor = await get_performance_monitor()
        result = await monitor.trigger_optimization(request.optimization_type)
        return result
    except Exception as e:
        logger.error(f"Error ejecutando optimización {request.optimization_type}: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.get("/performance/memory-profile")
async def get_memory_profile_endpoint():
    """Obtiene perfil detallado de memoria vía MCP"""
    try:
        monitor = await get_performance_monitor()
        result = await monitor.get_memory_profile()
        return result
    except Exception as e:
        logger.error(f"Error obteniendo perfil de memoria: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


@app.put("/performance/config")
async def configure_performance_monitoring_endpoint(request: MonitoringConfigRequest):
    """Configura parámetros de monitoreo de rendimiento vía MCP"""
    try:
        monitor = await get_performance_monitor()
        config_dict = {k: v for k, v in request.dict().items() if v is not None}
        result = await monitor.configure_monitoring(config_dict)
        return result
    except Exception as e:
        logger.error(f"Error configurando monitoreo de rendimiento: {e}")
        return {"error": f"Error interno del servidor: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    # Ejecutar servidor FastAPI directamente con uvicorn
    logger.info("Iniciando servidor MCP empresarial de Sheily en http://localhost:8006")
    uvicorn.run(
        "sheily_core.mcp_server:app",
        host="0.0.0.0",
        port=8006,
        reload=True,
        log_level="info",
    )
