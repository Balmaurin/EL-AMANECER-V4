#!/usr/bin/env python3
"""
Text Processor - Procesador Avanzado de Texto

Este módulo implementa un procesador avanzado de texto con capacidades de:
- Análisis de sentimientos
- Extracción de entidades
- Extracción de frases clave
- Análisis de texto completo
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TextAnalysisResult:
    """Resultado del análisis de texto"""

    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    key_phrases: List[str]
    language: str
    confidence: float
    processing_time: float


class TextProcessor:
    """Procesador avanzado de texto"""

    def __init__(self):
        """Inicializar procesador de texto"""
        self.initialized = True
        logger.info("TextProcessor inicializado")

    def analyze_text(self, text: str) -> TextAnalysisResult:
        """Analizar texto completo"""
        import time

        start_time = time.time()

        # Análisis básico (puede ser mejorado con modelos reales)
        sentiment = self.analyze_sentiment(text)
        entities = self.extract_entities(text)
        key_phrases = self.extract_key_phrases(text)
        language = self.detect_language(text)

        processing_time = time.time() - start_time

        return TextAnalysisResult(
            sentiment=sentiment,
            entities=entities,
            key_phrases=key_phrases,
            language=language,
            confidence=0.85,
            processing_time=processing_time,
        )

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analizar sentimiento del texto"""
        # Implementación básica - puede mejorarse con modelos reales
        text_lower = text.lower()

        positive_words = [
            "bueno",
            "excelente",
            "genial",
            "fantástico",
            "perfecto",
            "bien",
            "positivo",
        ]
        negative_words = ["malo", "terrible", "horrible", "pesimo", "negativo", "mal"]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            sentiment = "positive"
            score = min(1.0, 0.5 + (positive_count * 0.1))
        elif negative_count > positive_count:
            sentiment = "negative"
            score = max(0.0, 0.5 - (negative_count * 0.1))
        else:
            sentiment = "neutral"
            score = 0.5

        return {"sentiment": sentiment, "score": score, "confidence": 0.8}

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades del texto"""
        entities = []

        # Detección básica de entidades (puede mejorarse con NER real)
        words = text.split()

        for i, word in enumerate(words):
            # Detectar posibles nombres propios (capitalizados)
            if word.istitle() and len(word) > 1:
                entities.append(
                    {
                        "text": word,
                        "type": "PERSON",
                        "start": text.find(word),
                        "end": text.find(word) + len(word),
                        "confidence": 0.7,
                    }
                )

            # Detectar números
            if word.isdigit():
                entities.append(
                    {
                        "text": word,
                        "type": "NUMBER",
                        "start": text.find(word),
                        "end": text.find(word) + len(word),
                        "confidence": 0.9,
                    }
                )

        return entities

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extraer frases clave"""
        # Implementación básica - extraer frases de 2-4 palabras
        words = text.split()
        key_phrases = []

        for i in range(len(words) - 1):
            # Frases de 2 palabras
            if i < len(words) - 1:
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 5:  # Evitar frases muy cortas
                    key_phrases.append(phrase)

            # Frases de 3 palabras
            if i < len(words) - 2:
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(phrase) > 8:
                    key_phrases.append(phrase)

        # Limitar a las 5 frases más largas
        key_phrases.sort(key=len, reverse=True)
        return key_phrases[:5]

    def detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        # Detección básica por palabras clave
        spanish_words = [
            "el",
            "la",
            "los",
            "las",
            "un",
            "una",
            "y",
            "es",
            "son",
            "está",
        ]
        english_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of"]

        text_lower = text.lower()
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)

        if spanish_count > english_count:
            return "es"
        elif english_count > spanish_count:
            return "en"
        else:
            return "unknown"

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del procesador"""
        return {
            "initialized": self.initialized,
            "capabilities": [
                "sentiment_analysis",
                "entity_extraction",
                "key_phrase_extraction",
                "language_detection",
            ],
            "version": "1.0.0",
        }
