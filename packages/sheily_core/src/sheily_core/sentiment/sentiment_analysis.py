#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Análisis de Sentimientos en Tiempo Real - Sheily AI
============================================================

API avanzada para análisis de sentimientos y emociones:
- Análisis de sentimientos en múltiples idiomas
- Detección de emociones complejas (alegría, tristeza, ira, miedo, sorpresa, disgusto)
- Análisis de toxicidad y contenido inapropiado
- Métricas de confianza y explicabilidad
- Procesamiento en tiempo real con caché inteligente
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from sheily_core.cache.smart_cache import cached, get_cache

logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Etiquetas de sentimiento"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EmotionLabel(Enum):
    """Etiquetas de emociones"""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Resultado del análisis de sentimientos"""

    label: SentimentLabel
    confidence: float
    score: float  # Score raw del modelo

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return {
            "label": self.label.value,
            "confidence": self.confidence,
            "score": self.score,
        }


@dataclass
class EmotionResult:
    """Resultado del análisis de emociones"""

    emotions: List[Tuple[EmotionLabel, float]]  # Lista de (emoción, confianza)
    dominant_emotion: EmotionLabel
    confidence: float

    def to_dict(self) -> Dict[str, Union[str, float, List]]:
        return {
            "emotions": [
                {"emotion": e.value, "confidence": c} for e, c in self.emotions
            ],
            "dominant_emotion": self.dominant_emotion.value,
            "confidence": self.confidence,
        }


@dataclass
class ToxicityResult:
    """Resultado del análisis de toxicidad"""

    is_toxic: bool
    confidence: float
    categories: List[Tuple[str, float]]  # Lista de (categoría, score)

    def to_dict(self) -> Dict[str, Union[bool, float, List]]:
        return {
            "is_toxic": self.is_toxic,
            "confidence": self.confidence,
            "categories": [
                {"category": cat, "score": score} for cat, score in self.categories
            ],
        }


@dataclass
class ComprehensiveAnalysis:
    """Análisis completo de sentimientos y emociones"""

    sentiment: SentimentResult
    emotion: EmotionResult
    toxicity: ToxicityResult
    text_length: int
    processing_time_ms: float
    language: str = "es"

    def to_dict(self) -> Dict[str, Union[str, int, float, Dict]]:
        return {
            "sentiment": self.sentiment.to_dict(),
            "emotion": self.emotion.to_dict(),
            "toxicity": self.toxicity.to_dict(),
            "text_length": self.text_length,
            "processing_time_ms": self.processing_time_ms,
            "language": self.language,
        }


class SentimentAnalysisAPI:
    """API completa de análisis de sentimientos"""

    def __init__(self):
        self.cache = get_cache()

        # Modelos para análisis de sentimientos
        self.sentiment_model = None
        self.emotion_model = None
        self.toxicity_model = None

        # Configuración
        self.max_text_length = 512
        self.confidence_threshold = 0.6

        # Inicializar modelos de forma lazy
        self._models_loaded = False

    async def _load_models(self):
        """Cargar modelos de ML de forma lazy"""
        if self._models_loaded:
            return

        try:
            logger.info("Cargando modelos de análisis de sentimientos...")

            # Modelo de sentimientos multilingüe
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                return_all_scores=True,
            )

            # Modelo de emociones
            self.emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
            )

            # Modelo de toxicidad
            self.toxicity_model = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                return_all_scores=True,
            )

            self._models_loaded = True
            logger.info("✅ Modelos de análisis cargados exitosamente")

        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            raise

    def _normalize_sentiment(self, model_output: List[Dict]) -> SentimentResult:
        """Normalizar salida del modelo de sentimientos"""
        # El modelo devuelve scores para 1-5 estrellas, convertimos a positivo/negativo/neutral
        scores = {item["label"]: item["score"] for item in model_output}

        # Calcular score compuesto
        positive_score = (scores.get("5 stars", 0) + scores.get("4 stars", 0)) / 2
        negative_score = (scores.get("1 star", 0) + scores.get("2 stars", 0)) / 2
        neutral_score = scores.get("3 stars", 0)

        # Determinar etiqueta dominante
        max_score = max(positive_score, negative_score, neutral_score)

        if max_score == positive_score:
            label = SentimentLabel.POSITIVE
        elif max_score == negative_score:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        return SentimentResult(label=label, confidence=max_score, score=max_score)

    def _normalize_emotion(self, model_output: List[Dict]) -> EmotionResult:
        """Normalizar salida del modelo de emociones"""
        emotions = []
        for item in model_output:
            label = item["label"].lower()
            confidence = item["score"]

            # Mapear a nuestras etiquetas de emoción
            if "joy" in label:
                emotion = EmotionLabel.JOY
            elif "sadness" in label or "sad" in label:
                emotion = EmotionLabel.SADNESS
            elif "anger" in label or "angry" in label:
                emotion = EmotionLabel.ANGER
            elif "fear" in label:
                emotion = EmotionLabel.FEAR
            elif "surprise" in label:
                emotion = EmotionLabel.SURPRISE
            elif "disgust" in label:
                emotion = EmotionLabel.DISGUST
            else:
                emotion = EmotionLabel.NEUTRAL

            emotions.append((emotion, confidence))

        # Ordenar por confianza descendente
        emotions.sort(key=lambda x: x[1], reverse=True)

        return EmotionResult(
            emotions=emotions,
            dominant_emotion=emotions[0][0] if emotions else EmotionLabel.NEUTRAL,
            confidence=emotions[0][1] if emotions else 0.0,
        )

    def _normalize_toxicity(self, model_output: List[Dict]) -> ToxicityResult:
        """Normalizar salida del modelo de toxicidad"""
        # Buscar la categoría más relevante (toxic vs non-toxic)
        toxic_score = 0.0
        categories = []

        for item in model_output:
            label = item["label"].lower()
            score = item["score"]
            categories.append((label, score))

            if "toxic" in label:
                toxic_score = max(toxic_score, score)

        return ToxicityResult(
            is_toxic=toxic_score > self.confidence_threshold,
            confidence=toxic_score,
            categories=categories,
        )

    def _preprocess_text(self, text: str) -> str:
        """Preprocesar texto para análisis"""
        # Limitar longitud
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length]

        # Limpiar texto básico
        text = text.strip()
        if not text:
            text = "texto vacío"

        return text

    @cached("sentiment_analysis", ttl=3600)
    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analizar sentimientos de un texto"""
        await self._load_models()
        text = self._preprocess_text(text)

        try:
            result = self.sentiment_model(text)
            return self._normalize_sentiment(
                result[0] if isinstance(result, list) and result else result
            )
        except Exception as e:
            logger.error(f"Error analizando sentimiento: {e}")
            return SentimentResult(
                label=SentimentLabel.NEUTRAL, confidence=0.0, score=0.0
            )

    @cached("emotion_analysis", ttl=3600)
    async def analyze_emotion(self, text: str) -> EmotionResult:
        """Analizar emociones de un texto"""
        await self._load_models()
        text = self._preprocess_text(text)

        try:
            result = self.emotion_model(text)
            return self._normalize_emotion(
                result[0] if isinstance(result, list) and result else result
            )
        except Exception as e:
            logger.error(f"Error analizando emociones: {e}")
            return EmotionResult(
                emotions=[], dominant_emotion=EmotionLabel.NEUTRAL, confidence=0.0
            )

    @cached("toxicity_analysis", ttl=1800)
    async def analyze_toxicity(self, text: str) -> ToxicityResult:
        """Analizar toxicidad de un texto"""
        await self._load_models()
        text = self._preprocess_text(text)

        try:
            result = self.toxicity_model(text)
            return self._normalize_toxicity(
                result[0] if isinstance(result, list) and result else result
            )
        except Exception as e:
            logger.error(f"Error analizando toxicidad: {e}")
            return ToxicityResult(is_toxic=False, confidence=0.0, categories=[])

    async def analyze_comprehensive(self, text: str) -> ComprehensiveAnalysis:
        """Análisis completo de sentimientos, emociones y toxicidad"""
        import time

        start_time = time.time()

        text = self._preprocess_text(text)

        # Ejecutar análisis en paralelo
        sentiment_task = self.analyze_sentiment(text)
        emotion_task = self.analyze_emotion(text)
        toxicity_task = self.analyze_toxicity(text)

        sentiment, emotion, toxicity = await asyncio.gather(
            sentiment_task, emotion_task, toxicity_task
        )

        processing_time = (time.time() - start_time) * 1000  # ms

        return ComprehensiveAnalysis(
            sentiment=sentiment,
            emotion=emotion,
            toxicity=toxicity,
            text_length=len(text),
            processing_time_ms=processing_time,
        )

    async def batch_analyze(self, texts: List[str]) -> List[ComprehensiveAnalysis]:
        """Analizar múltiples textos en lote"""
        tasks = [self.analyze_comprehensive(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del sistema de análisis"""
        cache_metrics = self.cache.get_metrics()

        return {
            "models_loaded": self._models_loaded,
            "cache_metrics": cache_metrics,
            "configuration": {
                "max_text_length": self.max_text_length,
                "confidence_threshold": self.confidence_threshold,
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del sistema de análisis"""
        health = {
            "status": "healthy",
            "models_loaded": self._models_loaded,
            "cache_healthy": False,
        }

        try:
            cache_health = await self.cache.health_check()
            health["cache_healthy"] = cache_health["status"] == "healthy"
        except:
            pass

        if not health["models_loaded"]:
            health["status"] = "degraded"

        return health


# Instancia global
_sentiment_api_instance = None


def get_sentiment_api() -> SentimentAnalysisAPI:
    """Obtener instancia global de la API de sentimientos"""
    global _sentiment_api_instance
    if _sentiment_api_instance is None:
        _sentiment_api_instance = SentimentAnalysisAPI()
    return _sentiment_api_instance


# API endpoints para FastAPI
class SentimentRequest(BaseModel):
    text: str


class BatchSentimentRequest(BaseModel):
    texts: List[str]


class SentimentResponse(BaseModel):
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None


async def analyze_sentiment_endpoint(text: str) -> Dict[str, Any]:
    """Endpoint para análisis de sentimientos"""
    try:
        api = get_sentiment_api()
        result = await api.analyze_sentiment(text)
        return {"success": True, "data": result.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def analyze_emotion_endpoint(text: str) -> Dict[str, Any]:
    """Endpoint para análisis de emociones"""
    try:
        api = get_sentiment_api()
        result = await api.analyze_emotion(text)
        return {"success": True, "data": result.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def analyze_comprehensive_endpoint(text: str) -> Dict[str, Any]:
    """Endpoint para análisis completo"""
    try:
        api = get_sentiment_api()
        result = await api.analyze_comprehensive(text)
        return {"success": True, "data": result.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def batch_analyze_endpoint(texts: List[str]) -> Dict[str, Any]:
    """Endpoint para análisis en lote"""
    try:
        api = get_sentiment_api()
        results = await api.batch_analyze(texts)
        return {"success": True, "data": [r.to_dict() for r in results]}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Alias for compatibility
SentimentAnalyzer = SentimentAnalysisAPI
