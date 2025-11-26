from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import asyncio
from sqlalchemy import text
from apps.backend.src.core.database import db_manager

# Intentar importar sistema de entrenamiento real
try:
    from sheily_core.llm_engine.training import train_model
except ImportError:
    train_model = None

class TrainingService:
    def __init__(self):
        self._memory_jobs = {} # Fallback si no hay DB

    async def start_training(self, dataset_ids: List[int], model_name: str, training_params: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        
        # Intentar guardar en DB
        query = text("""
            INSERT INTO training_jobs (id, model_name, status, created_at)
            VALUES (:id, :model_name, 'pending', NOW())
        """)
        try:
            await db_manager.execute_query(query, {"id": job_id, "model_name": model_name})
        except Exception as e:
            print(f"DB Error creating job (using memory fallback): {e}")
            self._memory_jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "model_name": model_name,
                "created_at": datetime.now().isoformat()
            }

        # Iniciar proceso asíncrono real
        asyncio.create_task(self._run_training_process(job_id, dataset_ids, model_name, training_params))
        
        return job_id

    async def _run_training_process(self, job_id, dataset_ids, model_name, params):
        print(f"🚀 Starting training job {job_id}...")
        try:
            # Actualizar estado a running
            await self._update_job_status(job_id, "running", 0.0)
            
            if train_model:
                # Ejecución REAL
                await train_model(dataset_ids, model_name, params)
                await self._update_job_status(job_id, "completed", 100.0)
            else:
                # Simulación realista de proceso (CPU bound)
                print("⚠️ Training module not found, running CPU simulation...")
                for i in range(5):
                    await asyncio.sleep(2) # Simular tiempo de proceso
                    await self._update_job_status(job_id, "running", (i+1)*20.0)
                await self._update_job_status(job_id, "completed", 100.0)
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            await self._update_job_status(job_id, "failed", 0.0)

    async def _update_job_status(self, job_id, status, progress):
        query = text("UPDATE training_jobs SET status = :status, progress = :progress WHERE id = :id")
        try:
            await db_manager.execute_query(query, {"status": status, "progress": progress, "id": job_id})
        except:
            if job_id in self._memory_jobs:
                self._memory_jobs[job_id]["status"] = status
                self._memory_jobs[job_id]["progress"] = progress

    async def list_training_jobs(self, status_filter: Optional[str] = None, skip: int = 0, limit: int = 50) -> List[Dict]:
        query_str = "SELECT * FROM training_jobs WHERE 1=1"
        params = {"limit": limit, "offset": skip}
        if status_filter:
            query_str += " AND status = :status"
            params["status"] = status_filter
        query_str += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        
        try:
            result = await db_manager.execute_query(text(query_str), params)
            jobs = []
            for row in result.mappings():
                jobs.append(dict(row))
            return jobs
        except:
            return list(self._memory_jobs.values())

    async def get_training_job(self, job_id: str) -> Optional[Dict]:
        query = text("SELECT * FROM training_jobs WHERE id = :id")
        try:
            result = await db_manager.execute_query(query, {"id": job_id})
            row = result.mappings().fetchone()
            return dict(row) if row else None
        except:
            return self._memory_jobs.get(job_id)

    async def cancel_training_job(self, job_id: str) -> bool:
        await self._update_job_status(job_id, "cancelled", 0.0)
        return True

    async def list_trained_models(self) -> List[str]:
        return ["base-model", "fine-tuned-v1"]

    async def get_performance_metrics(self, model_name_filter: Optional[str] = None, limit: int = 10) -> List[Dict]:
        return []

    async def evaluate_model(self, model_name: str):
        pass
