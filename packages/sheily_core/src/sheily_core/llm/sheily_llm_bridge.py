#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sheily_llm_bridge.py
====================
Simple bridge to LLM functionality for chat system
"""

import logging
from typing import Optional

from sheily_core.llm_engine.real_llm_engine import create_real_llm_engine

logger = logging.getLogger(__name__)

# Global engine instance
_engine = None


def _get_engine():
    """Get or create LLM engine instance"""
    global _engine
    if _engine is None:
        try:
            _engine = create_real_llm_engine()
        except Exception as e:
            logger.warning(f"Failed to create real LLM engine: {e}")
            _engine = None
    return _engine


def call_llm(prompt: str) -> str:
    """
    Call LLM with prompt and return response string

    Args:
        prompt: The input prompt

    Returns:
        str: The LLM response
    """
    engine = _get_engine()
    if engine and engine.is_available():
        try:
            result = engine.generate_response(prompt)
            if result.get("success"):
                return result["response"]
            else:
                logger.error(f"LLM generation failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")

    # Fallback response
    return "Lo siento, no puedo procesar tu solicitud en este momento. ¿Puedes intentarlo de nuevo?"
