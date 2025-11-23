#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_chat_sheily.py
===================
Interfaz principal de conversación con Sheily (memoria híbrida + modelo local).
"""

from __future__ import annotations

import logging
import sys

from sheily_core.chat import sheily_chat_memory_adapter as chat

logger = logging.getLogger(__name__)


def main():
    logger.info("💬 Chat Sheily con memoria híbrida (chat + documentos)")
    logger.info("Comandos:")
    logger.info(' - "Sheily memoriza / guarda / aprende: <texto|ruta>"')
    logger.info(' - "borra este cacho: <fragmento>" o "borra: <fragmento>"')
    logger.info(' - "borra lo relacionado con: <tema>"')
    logger.info(' - "salir" para terminar.\n')
    while True:
        try:
            msg = input("Tú: ").strip()
        except EOFError:
            break
        if not msg:
            continue
        if msg.lower() in {"salir", "exit", "quit"}:
            logger.info("Sheily: Hasta pronto 💫")
            break
        resp = chat.respond(msg)
        logger.info(f"Sheily: {resp}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nSheily: sesión terminada.")
        sys.exit(0)
