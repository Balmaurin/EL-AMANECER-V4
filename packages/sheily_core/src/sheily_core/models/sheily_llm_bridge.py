#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sheily_llm_bridge.py
====================
Puente con llama.cpp (modelo Llama 3.2 GGUF).
"""

from __future__ import annotations

import shlex
import subprocess
import textwrap
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "gguf" / "llama-3.2.gguf"
LLAMA_BIN = (
    Path(__file__).resolve().parents[2] / "llama.cpp" / "build" / "bin" / "llama-cli"
)


def call_llm(
    prompt: str,
    n_predict: int = 128,
    temp: float = 0.7,
    top_p: float = 0.9,
    threads: int = 4,
) -> str:
    if not MODEL_PATH.exists():
        return f"[ERROR] Modelo no encontrado: {MODEL_PATH}"
    prompt = textwrap.dedent(prompt).strip().replace('"', '\\"')
    cmd = f'{LLAMA_BIN} --model "{MODEL_PATH}" --n-predict {n_predict} --temp {temp} --top-p {top_p} --threads {threads} --ctx-size 2048 --no-warmup --prompt "{prompt}"'
    try:
        result = subprocess.run(
            shlex.split(cmd), capture_output=True, text=True, timeout=180
        )
        return (result.stdout or "").strip() or "[sin respuesta]"
    except Exception as e:
        return f"[ERROR llama.cpp] {e}"


class SheilyLLMBridge:
    """Puente para interactuar con modelos LLM locales"""
    
    def __init__(self, model_path: str = None, llama_bin: str = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.llama_bin = Path(llama_bin) if llama_bin else LLAMA_BIN
    
    def generate(self, prompt: str, n_predict: int = 128, temp: float = 0.7, 
                 top_p: float = 0.9, threads: int = 4) -> str:
        """Generar respuesta usando el modelo LLM"""
        return call_llm(prompt, n_predict, temp, top_p, threads)
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Versión asíncrona de generate"""
        return self.generate(prompt, **kwargs)

