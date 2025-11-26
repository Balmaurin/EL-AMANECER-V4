import os

from fastapi import Header, HTTPException


def require_api_key(x_api_key: str = Header(None)):
    expected = os.getenv("RAG_API_KEY")
    must = os.getenv("RAG_require_api_key", "true").lower() == "true"
    if must and not expected:
        raise HTTPException(500, "API key requerida pero no configurada (RAG_API_KEY)")
    if must and (x_api_key is None or x_api_key != expected):
        raise HTTPException(401, "Invalid or missing API key")
