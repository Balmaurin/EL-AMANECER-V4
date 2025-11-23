"""
Datasets API - Gestión de datasets de entrenamiento
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Mock data para datasets
MOCK_DATASETS = [
    {
        "id": 1,
        "exercise_type": "yesno",
        "num_questions": 20,
        "accuracy": 85,
        "tokens_earned": 30,
        "timestamp": "2025-11-11T10:00:00Z",
        "data": {
            "correct": 17,
            "incorrect": 3,
            "total": 20,
            "answers": [
                {
                    "question": 0,
                    "userAnswer": True,
                    "correctAnswer": True,
                    "isCorrect": True,
                },
                {
                    "question": 1,
                    "userAnswer": False,
                    "correctAnswer": False,
                    "isCorrect": True,
                },
                # ... más respuestas
            ],
        },
    },
    {
        "id": 2,
        "exercise_type": "truefalse",
        "num_questions": 20,
        "accuracy": 90,
        "tokens_earned": 35,
        "timestamp": "2025-11-11T11:30:00Z",
        "data": {"correct": 18, "incorrect": 2, "total": 20, "answers": []},
    },
]


@router.get("/")
async def get_datasets():
    """Obtener todos los datasets"""
    return {"datasets": MOCK_DATASETS, "total": len(MOCK_DATASETS)}


@router.get("/{dataset_id}")
async def get_dataset_detail(dataset_id: int):
    """Obtener detalle de un dataset específico"""
    dataset = next((d for d in MOCK_DATASETS if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    return dataset


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int):
    """Eliminar un dataset"""
    global MOCK_DATASETS
    MOCK_DATASETS = [d for d in MOCK_DATASETS if d["id"] != dataset_id]
    return {"message": "Dataset eliminado"}


@router.get("/{dataset_id}/export")
async def export_dataset(dataset_id: int):
    """Exportar dataset como JSON"""
    dataset = next((d for d in MOCK_DATASETS if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    return {"dataset": dataset, "exported_at": datetime.now().isoformat()}
