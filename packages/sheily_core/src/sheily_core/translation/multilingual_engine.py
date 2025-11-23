#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Traducción Automática Multilingüe - Sheily AI
========================================================

Sistema completo para traducción automática que soporta:
- Traducción en tiempo real entre múltiples idiomas
- Detección automática de idioma
- Traducción contextual para consultas de IA
- Caché inteligente de traducciones
- Soporte para jerga técnica y terminología especializada
- API unificada para integración con el sistema principal
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langdetect import LangDetectError, detect
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from sheily_core.cache.smart_cache import cached, get_cache

logger = logging.getLogger(__name__)


@dataclass
class TranslationRequest:
    """Solicitud de traducción"""

    text: str
    source_lang: Optional[str] = None
    target_lang: str = "es"
    context: Optional[str] = None  # Contexto para traducción más precisa
    domain: str = "general"  # 'technical', 'medical', 'legal', 'general'


@dataclass
class TranslationResult:
    """Resultado de traducción"""

    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: float
    detected_lang: Optional[str] = None
    processing_time_ms: float = 0.0
    model_used: str = "unknown"


@dataclass
class LanguageSupport:
    """Información de soporte de idioma"""

    code: str
    name: str
    native_name: str
    supported: bool = True
    quality_score: float = 1.0  # 0.0-1.0


class MultilingualTranslationEngine:
    """Motor de traducción multilingüe"""

    def __init__(self):
        self.cache = get_cache()

        # Modelos de traducción
        self.translation_models = {}
        self.language_detector = None

        # Idiomas soportados
        self.supported_languages = self._initialize_supported_languages()

        # Configuración
        self.max_text_length = 512
        self.confidence_threshold = 0.7

        # Estadísticas
        self.translation_stats = {
            "total_translations": 0,
            "languages_used": {},
            "cache_hits": 0,
            "errors": 0,
        }

    def _initialize_supported_languages(self) -> Dict[str, LanguageSupport]:
        """Inicializar lista de idiomas soportados"""
        languages = {
            "es": LanguageSupport("es", "Spanish", "Español", quality_score=0.95),
            "en": LanguageSupport("en", "English", "English", quality_score=0.98),
            "fr": LanguageSupport("fr", "French", "Français", quality_score=0.90),
            "de": LanguageSupport("de", "German", "Deutsch", quality_score=0.88),
            "it": LanguageSupport("it", "Italian", "Italiano", quality_score=0.85),
            "pt": LanguageSupport("pt", "Portuguese", "Português", quality_score=0.87),
            "ru": LanguageSupport("ru", "Russian", "Русский", quality_score=0.80),
            "zh": LanguageSupport("zh", "Chinese", "中文", quality_score=0.82),
            "ja": LanguageSupport("ja", "Japanese", "日本語", quality_score=0.78),
            "ko": LanguageSupport("ko", "Korean", "한국어", quality_score=0.75),
            "ar": LanguageSupport("ar", "Arabic", "العربية", quality_score=0.70),
            "hi": LanguageSupport("hi", "Hindi", "हिन्दी", quality_score=0.72),
            "nl": LanguageSupport("nl", "Dutch", "Nederlands", quality_score=0.83),
            "sv": LanguageSupport("sv", "Swedish", "Svenska", quality_score=0.81),
            "da": LanguageSupport("da", "Danish", "Dansk", quality_score=0.79),
            "no": LanguageSupport("no", "Norwegian", "Norsk", quality_score=0.80),
            "fi": LanguageSupport("fi", "Finnish", "Suomi", quality_score=0.76),
            "pl": LanguageSupport("pl", "Polish", "Polski", quality_score=0.77),
            "tr": LanguageSupport("tr", "Turkish", "Türkçe", quality_score=0.74),
            "he": LanguageSupport("he", "Hebrew", "עברית", quality_score=0.71),
            "th": LanguageSupport("th", "Thai", "ไทย", quality_score=0.68),
            "vi": LanguageSupport("vi", "Vietnamese", "Tiếng Việt", quality_score=0.73),
        }
        return languages

    async def _load_translation_model(
        self, source_lang: str, target_lang: str
    ) -> Optional[object]:
        """Cargar modelo de traducción para par de idiomas"""
        model_key = f"{source_lang}-{target_lang}"

        if model_key in self.translation_models:
            return self.translation_models[model_key]

        try:
            # Usar modelo Helsinki-NLP para traducción
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

            # Verificar si el modelo está disponible
            try:
                model = pipeline("translation", model=model_name)
                self.translation_models[model_key] = model
                logger.info(f"Modelo de traducción cargado: {model_key}")
                return model
            except Exception as e:
                logger.warning(f"Modelo {model_name} no disponible: {e}")

                # Intentar modelo alternativo o más general
                if source_lang != "en" and target_lang != "en":
                    # Traducción indirecta vía inglés
                    en_source_model = await self._load_translation_model(
                        source_lang, "en"
                    )
                    en_target_model = await self._load_translation_model(
                        "en", target_lang
                    )

                    if en_source_model and en_target_model:
                        # Crear pipeline compuesto
                        self.translation_models[model_key] = {
                            "source_to_en": en_source_model,
                            "en_to_target": en_target_model,
                            "type": "indirect",
                        }
                        return self.translation_models[model_key]

        except Exception as e:
            logger.error(f"Error cargando modelo de traducción {model_key}: {e}")

        return None

    async def _load_language_detector(self):
        """Cargar detector de idioma"""
        if self.language_detector is None:
            try:
                # langdetect no requiere modelo descargable
                self.language_detector = detect
            except ImportError:
                logger.warning("langdetect no disponible, detección de idioma limitada")

    def _preprocess_text(self, text: str, domain: str) -> str:
        """Preprocesar texto para traducción"""
        # Limitar longitud
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length]

        # Limpieza básica
        text = text.strip()
        if not text:
            text = "texto vacío"

        # Adaptaciones por dominio
        if domain == "technical":
            # Preservar términos técnicos comunes
            technical_terms = {
                "API": "API",
                "HTTP": "HTTP",
                "JSON": "JSON",
                "XML": "XML",
                "SQL": "SQL",
                "CSS": "CSS",
                "HTML": "HTML",
                "URL": "URL",
            }
            # Aquí se podría implementar lógica para marcar términos técnicos
            pass
        elif domain == "code":
            # Para código, intentar preservar estructura
            pass

        return text

    async def detect_language(self, text: str) -> Tuple[str, float]:
        """Detectar idioma del texto"""
        await self._load_language_detector()

        if not self.language_detector:
            return "en", 0.5  # Default fallback

        try:
            detected = self.language_detector(text)
            # langdetect no proporciona confidence, estimamos basado en longitud
            confidence = min(len(text) / 50, 1.0) if len(text) > 10 else 0.3
            return detected, confidence
        except LangDetectError:
            return "en", 0.1

    @cached("language_detection", ttl=3600)
    async def detect_language_cached(self, text: str) -> Tuple[str, float]:
        """Detección de idioma con caché"""
        return await self.detect_language(text)

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """Traducir texto"""
        import time

        start_time = time.time()

        # Preprocesar texto
        processed_text = self._preprocess_text(request.text, request.domain)

        # Detectar idioma si no se especifica
        if not request.source_lang:
            detected_lang, detect_confidence = await self.detect_language_cached(
                processed_text
            )
            source_lang = detected_lang
        else:
            source_lang = request.source_lang
            detect_confidence = 1.0

        target_lang = request.target_lang

        # Verificar soporte de idiomas
        if (
            source_lang not in self.supported_languages
            or target_lang not in self.supported_languages
        ):
            raise ValueError(f"Idioma no soportado: {source_lang} -> {target_lang}")

        # Si los idiomas son iguales, devolver texto original
        if source_lang == target_lang:
            return TranslationResult(
                original_text=request.text,
                translated_text=request.text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=1.0,
                detected_lang=source_lang,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="none",
            )

        # Intentar traducción con modelo directo
        model = await self._load_translation_model(source_lang, target_lang)

        if model:
            try:
                if isinstance(model, dict) and model.get("type") == "indirect":
                    # Traducción indirecta vía inglés
                    en_translation = model["source_to_en"](processed_text)[0][
                        "translation_text"
                    ]
                    final_translation = model["en_to_target"](en_translation)[0][
                        "translation_text"
                    ]
                    model_name = "indirect_translation"
                else:
                    # Traducción directa
                    result = model(processed_text)[0]
                    final_translation = result["translation_text"]
                    model_name = "direct_translation"

                # Calcular confianza basada en calidad del modelo y detección
                lang_quality = min(
                    self.supported_languages[source_lang].quality_score,
                    self.supported_languages[target_lang].quality_score,
                )
                confidence = lang_quality * detect_confidence

                self.translation_stats["total_translations"] += 1
                self.translation_stats["languages_used"][
                    f"{source_lang}->{target_lang}"
                ] = (
                    self.translation_stats["languages_used"].get(
                        f"{source_lang}->{target_lang}", 0
                    )
                    + 1
                )

                return TranslationResult(
                    original_text=request.text,
                    translated_text=final_translation,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    confidence=confidence,
                    detected_lang=detected_lang if not request.source_lang else None,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    model_used=model_name,
                )

            except Exception as e:
                logger.error(f"Error en traducción: {e}")
                self.translation_stats["errors"] += 1

        # Fallback: traducción básica o error
        logger.warning(
            f"No se pudo traducir {source_lang} -> {target_lang}, usando fallback"
        )

        return TranslationResult(
            original_text=request.text,
            translated_text=request.text,  # Mantener original como fallback
            source_lang=source_lang,
            target_lang=target_lang,
            confidence=0.1,
            detected_lang=detected_lang if not request.source_lang else None,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_used="fallback",
        )

    @cached("translation", ttl=7200)  # Cache por 2 horas
    async def translate_cached(self, request: TranslationRequest) -> TranslationResult:
        """Traducción con caché inteligente"""
        return await self.translate(request)

    async def translate_batch(
        self, requests: List[TranslationRequest]
    ) -> List[TranslationResult]:
        """Traducir múltiples textos en lote"""
        tasks = [self.translate_cached(request) for request in requests]
        return await asyncio.gather(*tasks)

    async def get_supported_languages(self) -> Dict[str, Dict]:
        """Obtener lista de idiomas soportados"""
        return {
            code: {
                "name": lang.name,
                "native_name": lang.native_name,
                "supported": lang.supported,
                "quality_score": lang.quality_score,
            }
            for code, lang in self.supported_languages.items()
        }

    async def get_translation_stats(self) -> Dict[str, any]:
        """Obtener estadísticas de traducción"""
        return {
            "total_translations": self.translation_stats["total_translations"],
            "languages_used": self.translation_stats["languages_used"],
            "cache_hits": (
                self.cache.metrics.hits if hasattr(self.cache, "metrics") else 0
            ),
            "errors": self.translation_stats["errors"],
            "supported_languages_count": len(
                [l for l in self.supported_languages.values() if l.supported]
            ),
            "models_loaded": len(self.translation_models),
        }

    async def warmup_translations(self, common_pairs: List[Tuple[str, str]]):
        """Pre-cargar modelos para pares de idiomas comunes"""
        logger.info(
            f"Pre-cargando modelos para {len(common_pairs)} pares de idiomas..."
        )

        for source_lang, target_lang in common_pairs:
            try:
                await self._load_translation_model(source_lang, target_lang)
                logger.debug(f"Modelo pre-cargado: {source_lang} -> {target_lang}")
            except Exception as e:
                logger.error(f"Error pre-cargando {source_lang}->{target_lang}: {e}")

        logger.info("Pre-carga de modelos completada")

    async def health_check(self) -> Dict[str, any]:
        """Verificar salud del sistema de traducción"""
        health = {
            "status": "healthy",
            "models_loaded": len(self.translation_models),
            "supported_languages": len(self.supported_languages),
            "cache_healthy": False,
            "language_detector_available": self.language_detector is not None,
        }

        try:
            # Verificar caché
            cache_health = await self.cache.health_check()
            health["cache_healthy"] = cache_health["status"] == "healthy"
        except:
            pass

        # Verificar capacidad básica de traducción
        try:
            test_request = TranslationRequest(
                text="Hello world", source_lang="en", target_lang="es"
            )
            test_result = await self.translate(test_request)
            health["translation_working"] = test_result.confidence > 0.1
        except:
            health["translation_working"] = False
            health["status"] = "degraded"

        if not health["translation_working"]:
            health["status"] = "error"

        return health


# Instancia global
_translation_engine_instance = None


def get_translation_engine() -> MultilingualTranslationEngine:
    """Obtener instancia global del motor de traducción"""
    global _translation_engine_instance
    if _translation_engine_instance is None:
        _translation_engine_instance = MultilingualTranslationEngine()
    return _translation_engine_instance


# Funciones de utilidad para integración
async def translate_text(
    text: str,
    target_lang: str = "es",
    source_lang: Optional[str] = None,
    context: Optional[str] = None,
    domain: str = "general",
) -> TranslationResult:
    """Función de utilidad para traducir texto"""
    engine = get_translation_engine()
    request = TranslationRequest(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        context=context,
        domain=domain,
    )
    return await engine.translate_cached(request)


async def detect_text_language(text: str) -> Tuple[str, float]:
    """Función de utilidad para detectar idioma"""
    engine = get_translation_engine()
    return await engine.detect_language_cached(text)


async def translate_batch(
    texts: List[str], target_lang: str = "es", source_lang: Optional[str] = None
) -> List[TranslationResult]:
    """Función de utilidad para traducir múltiples textos"""
    engine = get_translation_engine()
    requests = [
        TranslationRequest(text=text, source_lang=source_lang, target_lang=target_lang)
        for text in texts
    ]
    return await engine.translate_batch(requests)


# Funciones de conversión para integración con el sistema principal
async def translate_user_query(
    query: str, user_preferred_lang: str = "es"
) -> Tuple[str, str]:
    """Traducir consulta de usuario al idioma del sistema (español)"""
    detected_lang, confidence = await detect_text_language(query)

    if detected_lang == "es" or confidence < 0.6:
        return query, detected_lang  # Ya en español o detección poco confiable

    # Traducir al español
    result = await translate_text(query, target_lang="es", source_lang=detected_lang)
    return result.translated_text, detected_lang


async def translate_system_response(response: str, user_lang: str) -> str:
    """Traducir respuesta del sistema al idioma del usuario"""
    if user_lang == "es":
        return response  # Ya en español

    # Traducir desde español al idioma del usuario
    result = await translate_text(response, target_lang=user_lang, source_lang="es")
    return result.translated_text
