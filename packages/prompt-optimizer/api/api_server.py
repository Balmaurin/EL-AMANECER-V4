#!/usr/bin/env python3
"""
API REST para el Sistema Universal de Optimización de Prompts
Usa FastAPI para endpoints rápidos y documentados.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..universal_prompt_optimizer import (
    HttpAdapter,
    LlamaCppAdapter,
    OpenAIAdapter,
    UniversalAutoImprovingPromptSystem,
)

logger = logging.getLogger(__name__)

# Configurar logging para API
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Universal Prompt Optimizer API",
    description="Sistema automático de mejora de prompts para cualquier LLM",
    version="1.0.0",
)

# Configurar CORS para integración web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo global del sistema (singleton pattern)
system_instances: Dict[str, UniversalAutoImprovingPromptSystem] = {}


class OptimizeRequest(BaseModel):
    """Modelo para solicitud de optimización"""

    prompt: str = Field(..., description="El prompt a optimizar")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Contexto adicional"
    )
    max_iterations: Optional[int] = Field(
        3, ge=1, le=10, description="Iteraciones máximas de optimización"
    )
    model_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Config del LLM"
    )


class GenerateRequest(BaseModel):
    """Modelo para solicitud de generación"""

    user_query: str = Field(
        ..., description="Query del usuario para generar respuesta mejorada"
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Contexto personalizado"
    )


class BenchmarkRequest(BaseModel):
    """Modelo para benchmark"""

    prompts: List[str] = Field(..., description="Lista de prompts a evaluar")
    metrics: Optional[List[str]] = Field(["score"], description="Métricas a evaluar")


class EvaluationResponse(BaseModel):
    """Respuesta de evaluación"""

    original_prompt: str
    optimized_prompt: str
    score: float
    metrics: Dict[str, float]
    reasoning: str
    improvements: List[str]
    iterations: int
    technique_used: str
    processing_time: float


class GenerationResponse(BaseModel):
    """Respuesta de generación"""

    response: str
    safe: bool = True
    warnings: Optional[List[str]] = []


class BenchmarkResponse(BaseModel):
    """Respuesta de benchmark"""

    results: List[EvaluationResponse]
    summary: Dict[str, Any]


class SystemStatus(BaseModel):
    """Estado del sistema"""

    status: str = "operational"
    available_models: List[str]
    version: str = "1.0.0"
    uptime: str


@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Estado del sistema"""
    available_models = list(system_instances.keys())
    if not available_models:
        available_models = ["No models loaded"]

    return SystemStatus(
        available_models=available_models, uptime="Operational since deployment"
    )


@app.post("/optimize", response_model=EvaluationResponse)
async def optimize_prompt(request: OptimizeRequest) -> EvaluationResponse:
    """Optimizar un prompt automáticamente"""

    # Seleccionar modelo (por defecto usa el primero disponible)
    system = _get_system_instance()

    start_time = datetime.now()

    try:
        # Optimizar el prompt
        result = await system.optimize_prompt(
            original_prompt=request.prompt,
            context=request.context,
            max_iterations=request.max_iterations,
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return EvaluationResponse(
            original_prompt=result.original_prompt,
            optimized_prompt=result.optimized_prompt,
            score=result.evaluation.score,
            metrics=result.evaluation.metrics,
            reasoning=result.evaluation.reasoning,
            improvements=result.evaluation.improvements,
            iterations=result.iterations,
            technique_used=result.technique_used,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Error en optimización: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerationResponse)
async def generate_response(request: GenerateRequest) -> GenerationResponse:
    """Generar respuesta optimizada para una query de usuario"""

    system = _get_system_instance()

    try:
        respuesta = await system.generate_response(request.user_query)

        # Aquí podrías integrar los guardrails de seguridad si los tienes
        return GenerationResponse(
            response=respuesta, safe=True  # Por defecto asumimos seguro
        )

    except Exception as e:
        logger.error(f"Error generando respuesta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_prompt_only(request: OptimizeRequest) -> EvaluationResponse:
    """Solo evaluar un prompt sin optimizarlo"""

    system = _get_system_instance()

    try:
        evaluation = await system.evaluator.evaluate_prompt(request.prompt)

        return EvaluationResponse(
            original_prompt=request.prompt,
            optimized_prompt=request.prompt,  # Sin cambios
            score=evaluation.score,
            metrics=evaluation.metrics,
            reasoning=evaluation.reasoning,
            improvements=evaluation.improvements,
            iterations=0,
            technique_used="evaluation_only",
            processing_time=0.0,
        )

    except Exception as e:
        logger.error(f"Error evaluando prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """Ejecutar benchmark en lote de prompts"""

    system = _get_system_instance()
    results = []
    total_score = 0

    for prompt in request.prompts:
        try:
            evaluation = await system.evaluator.evaluate_prompt(prompt)
            results.append(
                EvaluationResponse(
                    original_prompt=prompt,
                    optimized_prompt=prompt,
                    score=evaluation.score,
                    metrics=evaluation.metrics,
                    reasoning=evaluation.reasoning,
                    improvements=evaluation.improvements,
                    iterations=0,
                    technique_used="benchmark",
                    processing_time=0.0,
                )
            )
            total_score += evaluation.score

        except Exception as e:
            logger.error(f"Error en benchmark de prompt: {e}")
            continue

    summary = {
        "total_prompts": len(request.prompts),
        "successful_evaluations": len(results),
        "average_score": total_score / len(results) if results else 0,
        "metrics_evaluated": request.metrics or ["score"],
    }

    return BenchmarkResponse(results=results, summary=summary)


def _get_system_instance() -> UniversalAutoImprovingPromptSystem:
    """Obtener instancia del sistema (crear si no existe)"""

    # Por defecto usa Llama si está disponible
    model_key = "llama_default"

    if model_key not in system_instances:
        try:
            # Intentar crear con Llama
            llm = LlamaCppAdapter("models/llama-3.2-3b-q4.gguf")
            system_instances[model_key] = UniversalAutoImprovingPromptSystem(llm)
            logger.info("Sistema inicializado con Llama 3.2 3B")

        except Exception as e:
            logger.error(f"Error inicializando Llama: {e}")
            # Fallback - necesitarías tener OpenAI key o similar
            raise HTTPException(
                status_code=503,
                detail="No se pudo inicializar ningún modelo LLM. Verifica configuración.",
            )

    return system_instances[model_key]


@app.on_event("startup")
async def startup_event():
    """Inicialización en startup"""
    logger.info("🚀 Iniciando API del Universal Prompt Optimizer")
    # Aquí podrías cargar modelos pesados o inicializar conexiones


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza en shutdown"""
    logger.info("⏹️ Apagando API del Universal Prompt Optimizer")
    system_instances.clear()


# Para ejecutar directamente: python -m api.api_server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
