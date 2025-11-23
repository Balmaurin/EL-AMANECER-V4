"""
Exercises API - Endpoints para ejercicios y datasets
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from apps.backend.src.models.database import get_db, Exercise, Dataset, User
from apps.backend.src.core.auth import get_current_user

# Import fine-tuning system
try:
    import sys
    import os
    # Add the project root to the python path to find packages
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to project root: apps/backend/src/api/v1/routes/v1 -> ... -> EL-AMANECERV3
    root_dir = os.path.abspath(os.path.join(current_dir, "../../../../../../../"))
    
    # Add sheily-core src to path
    sheily_core_path = os.path.join(root_dir, "packages", "sheily-core", "src")
    if sheily_core_path not in sys.path:
        sys.path.append(sheily_core_path)
    
    from sheily_core.models.ml.qora_fine_tuning import QLoRAFinetuningPipeline
    TRAINING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import QLoRAFinetuningPipeline: {e}. Fine-tuning will be disabled.")
    QLoRAFinetuningPipeline = None
    TRAINING_AVAILABLE = False


router = APIRouter()

class ExerciseResult(BaseModel):
    exercise_type: str
    answers: List[Dict]
    correct: int
    incorrect: int
    total_tokens: int

class ExerciseDatasetResponse(BaseModel):
    id: int
    user_id: int
    exercise_type: str
    data: Optional[Dict]
    tokens_earned: float
    timestamp: str
    num_questions: int = 0
    accuracy: float = 0.0

async def train_on_exercise_dataset(dataset_data: Dict):
    """Background task to trigger fine-tuning"""
    if not TRAINING_AVAILABLE:
        return
    
    try:
        # Prepare data for the pipeline
        # The pipeline looks for JSON files in data/audit_results
        # We will create a synthetic audit file so the pipeline picks it up
        
        audit_dir = "data/audit_results"
        os.makedirs(audit_dir, exist_ok=True)
        
        interactions = []
        for answer in dataset_data.get("answers", []):
            if answer.get("isCorrect"):
                interactions.append({
                    "query": answer.get("question", ""),
                    "response": answer.get("correctAnswer", ""),
                    "quality_score": 1.0,
                    "source": "exercise_dataset",
                    "category": dataset_data.get("exercise_type")
                })
        
        if interactions:
            filename = f"exercise_audit_{int(datetime.now().timestamp())}.json"
            filepath = os.path.join(audit_dir, filename)
            
            audit_content = {
                "timestamp": datetime.now().isoformat(),
                "source": "exercise_system",
                "interactions": interactions,
                "audit_details": {
                    "dataset_id": dataset_data.get("id"),
                    "exercise_type": dataset_data.get("exercise_type")
                }
            }
            
            with open(filepath, "w") as f:
                json.dump(audit_content, f, indent=2)
            
            # Trigger the pipeline
            pipeline = QLoRAFinetuningPipeline()
            await pipeline.execute_automated_fine_tuning_cycle()
            
    except Exception as e:
        logging.error(f"Error in background training: {e}")

@router.post("/submit", response_model=Dict[str, Any])
async def submit_exercise(
    exercise: ExerciseResult, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Enviar resultados de ejercicio"""
    try:
        # 1. Save Exercise Record
        db_exercise = Exercise(
            user_id=current_user.id,
            exercise_type=exercise.exercise_type,
            question_count=len(exercise.answers),
            correct_answers=exercise.correct,
            incorrect_answers=exercise.incorrect,
            accuracy_percentage=(exercise.correct / len(exercise.answers) * 100) if exercise.answers else 0,
            tokens_earned=exercise.total_tokens,
            answers_data=exercise.answers,
            completed=True,
            status="completed",
            completed_at=datetime.utcnow()
        )
        db.add(db_exercise)
        
        # 2. Create Dataset Record
        dataset_data = {
            "exercise_type": exercise.exercise_type,
            "answers": exercise.answers,
            "correct": exercise.correct,
            "incorrect": exercise.incorrect,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        db_dataset = Dataset(
            user_id=current_user.id,
            name=f"Dataset {exercise.exercise_type} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            description=f"Generated from {exercise.exercise_type} exercise",
            exercise_type=exercise.exercise_type,
            total_questions=len(exercise.answers),
            accuracy=(exercise.correct / len(exercise.answers) * 100) if exercise.answers else 0,
            tokens_earned=exercise.total_tokens,
            dataset_metadata=dataset_data,
            is_public=False
        )
        db.add(db_dataset)
        
        # 3. Update User Tokens
        current_user.sheily_tokens += exercise.total_tokens
        current_user.experience_points += int(exercise.total_tokens * 10)
        
        db.commit()
        db.refresh(db_dataset)

        # 4. Trigger Background Training
        if TRAINING_AVAILABLE:
            background_tasks.add_task(train_on_exercise_dataset, dataset_data)

        return {
            "message": "Exercise submitted successfully",
            "dataset_id": db_dataset.id,
            "tokens_earned": exercise.total_tokens,
            "new_balance": current_user.sheily_tokens
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting exercise: {str(e)}",
        )

@router.get("/datasets", response_model=Dict[str, Any])
async def get_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Listar todos los datasets generados por el usuario"""
    datasets = db.query(Dataset).filter(Dataset.user_id == current_user.id).order_by(Dataset.created_at.desc()).all()
    
    result = []
    for ds in datasets:
        result.append({
            "id": ds.id,
            "user_id": ds.user_id,
            "exercise_type": ds.exercise_type,
            "data": ds.dataset_metadata,
            "tokens_earned": ds.tokens_earned,
            "timestamp": ds.created_at.isoformat(),
            "num_questions": ds.total_questions,
            "accuracy": ds.accuracy
        })
        
    return {"datasets": result, "total": len(result)}
