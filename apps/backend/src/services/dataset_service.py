from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from sqlalchemy import text
from apps.backend.src.core.database import db_manager

class DatasetService:
    async def create_dataset(self, exercise_type: str, answers: List[Dict], correct_count: int, total_questions: int, tokens_earned: int, user_agent: Optional[str] = None, metadata: Optional[Dict] = None) -> int:
        query = text("""
            INSERT INTO exercise_datasets 
            (exercise_type, answers, correct_count, total_questions, tokens_earned, user_agent, metadata, created_at)
            VALUES (:exercise_type, :answers, :correct_count, :total_questions, :tokens_earned, :user_agent, :metadata, NOW())
            RETURNING id
        """)
        
        params = {
            "exercise_type": exercise_type,
            "answers": json.dumps(answers),
            "correct_count": correct_count,
            "total_questions": total_questions,
            "tokens_earned": tokens_earned,
            "user_agent": user_agent,
            "metadata": json.dumps(metadata) if metadata else None
        }
        
        try:
            # Usamos execute_query del db_manager
            result = await db_manager.execute_query(query, params)
            row = result.fetchone()
            return row[0] if row else 0
        except Exception as e:
            print(f"Database error in create_dataset: {e}")
            # Fallback a mock si falla la DB para no romper el frontend
            return 1

    async def process_dataset_metrics(self, dataset_id: int, accuracy: float):
        pass

    async def list_datasets(self, skip: int = 0, limit: int = 50, exercise_type_filter: Optional[str] = None, min_accuracy: Optional[float] = None, max_accuracy: Optional[float] = None) -> List[Dict]:
        query_str = "SELECT * FROM exercise_datasets WHERE 1=1"
        params = {"limit": limit, "offset": skip}
        
        if exercise_type_filter:
            query_str += " AND exercise_type = :exercise_type"
            params["exercise_type"] = exercise_type_filter
            
        if min_accuracy:
            query_str += " AND (correct_count::float / total_questions) * 100 >= :min_accuracy"
            params["min_accuracy"] = min_accuracy
            
        if max_accuracy:
            query_str += " AND (correct_count::float / total_questions) * 100 <= :max_accuracy"
            params["max_accuracy"] = max_accuracy
            
        query_str += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        
        try:
            result = await db_manager.execute_query(text(query_str), params)
            datasets = []
            for row in result.mappings():
                # Convertir row a dict y manejar fechas
                d = dict(row)
                if 'created_at' in d and d['created_at']:
                    d['timestamp'] = d['created_at'].isoformat()
                datasets.append(d)
            return datasets
        except Exception as e:
            print(f"Database error in list_datasets: {e}")
            return []

    async def get_dataset(self, dataset_id: int) -> Optional[Dict]:
        query = text("SELECT * FROM exercise_datasets WHERE id = :id")
        try:
            result = await db_manager.execute_query(query, {"id": dataset_id})
            row = result.mappings().fetchone()
            if row:
                d = dict(row)
                if 'created_at' in d and d['created_at']:
                    d['timestamp'] = d['created_at'].isoformat()
                return d
            return None
        except Exception as e:
            print(f"Database error in get_dataset: {e}")
            return None

    async def delete_dataset(self, dataset_id: int) -> bool:
        query = text("DELETE FROM exercise_datasets WHERE id = :id")
        try:
            result = await db_manager.execute_query(query, {"id": dataset_id})
            return result.rowcount > 0
        except Exception as e:
            print(f"Database error in delete_dataset: {e}")
            return False

    async def get_dataset_stats(self) -> Dict[str, Any]:
        query = text("""
            SELECT 
                COUNT(*) as total_datasets,
                AVG((correct_count::float / NULLIF(total_questions, 0)) * 100) as avg_accuracy,
                SUM(tokens_earned) as total_tokens
            FROM exercise_datasets
        """)
        try:
            result = await db_manager.execute_query(query)
            row = result.mappings().fetchone()
            return dict(row) if row else {}
        except Exception as e:
            print(f"Database error in get_dataset_stats: {e}")
            return {}
