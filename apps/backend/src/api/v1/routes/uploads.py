"""
Uploads API - Gestión de subida de archivos
Extraído de tools/dashboard_backend.py
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from apps.backend.src.security import SecurityError, sanitize_filename, validate_command_args, validate_timeout


class UploadResult(BaseModel):
    filename: str
    size: int
    type: str
    path: str
    message: str
    tokens_earned: int
    total_tokens: int
    training: Dict = None


class Database:
    """Helper de base de datos para uploads"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        import sqlite3
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Inicializar base de datos de uploads"""
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            # Usuario por defecto
            conn.execute(
                """
                INSERT OR IGNORE INTO users (username, tokens) VALUES (?, ?)
            """,
                ("default_user", 100),
            )
            conn.commit()

    def get_user_tokens(self, user_id: int = 1) -> int:
        """Obtener tokens del usuario"""
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT tokens FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            return result[0] if result else 0

    def update_user_tokens(self, user_id: int, tokens: int, reason: str):
        """Actualizar tokens del usuario"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE users SET tokens = tokens + ? WHERE id = ?", (tokens, user_id)
            )
            conn.commit()
            logging.info(f"User {user_id} tokens updated by {tokens}: {reason}")


# Inicializar base de datos
db = Database(Path("data/sheily_dashboard.db"))
db.init_db()

# Importar sistema de entrenamiento automático (opcional)
try:
    from tools.auto_training_system import train_on_uploaded_file
    TRAINING_AVAILABLE = True
    logging.info("✅ Sistema de entrenamiento para uploads disponible")
except ImportError:
    TRAINING_AVAILABLE = False
    logging.info("ℹ️ Sistema de entrenamiento para uploads no disponible")

# Crear directorios necesarios
Path("data/uploads").mkdir(exist_ok=True)
Path("data/datasets").mkdir(exist_ok=True)

router = APIRouter()


@router.post("/files", response_model=List[UploadResult])
async def upload_files(
    files: List[UploadFile] = File(...),
):
    """Subir archivos para entrenamiento"""
    try:
        uploaded_files = []
        total_tokens = 0
        training_tasks = []

        allowed_extensions = {".md", ".txt", ".jsonl", ".pdf", ".xml"}

        for file in files:
            # Validar y sanitizar nombre de archivo
            try:
                safe_filename = sanitize_filename(file.filename)
            except SecurityError as se:
                logging.warning(f"Archivo rechazado por seguridad: {file.filename} - {se}")
                continue

            file_ext = Path(safe_filename).suffix.lower()

            if file_ext not in allowed_extensions:
                logging.warning(f"Tipo de archivo no permitido: {file_ext}")
                continue

            # Guardar archivo
            file_path = (
                Path("data/uploads")
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_filename}"
            )
            content = await file.read()

            with open(file_path, "wb") as f:
                f.write(content)

            uploaded_files.append(
                {
                    "filename": safe_filename,
                    "size": len(content),
                    "type": file_ext,
                    "path": str(file_path),
                }
            )

            # Calcular tokens por archivo
            tokens_earned = min(len(content) // 100, 50)  # Máximo 50 tokens por archivo
            total_tokens += tokens_earned

            # Preparar entrenamiento automático si está disponible
            if TRAINING_AVAILABLE and file_ext in [".txt", ".md", ".jsonl"]:
                try:
                    # Determinar tipo de archivo para el entrenamiento
                    file_type = file_ext[1:]  # Remover el punto
                    training_tasks.append(
                        {
                            "filename": safe_filename,
                            "training_started": True,
                            "training_status": "pending",
                            "file_path": file_path,
                            "file_type": file_type,
                        }
                    )
                    logging.info(
                        f"🚀 Entrenamiento automático preparado para archivo: {safe_filename}"
                    )
                except Exception as e:
                    logging.error(
                        f"Error preparando entrenamiento para {safe_filename}: {e}"
                    )
                    training_tasks.append(
                        {
                            "filename": safe_filename,
                            "training_started": False,
                            "training_status": "error",
                            "error": str(e),
                        }
                    )

        # Actualizar tokens del usuario
        if total_tokens > 0:
            db.update_user_tokens(1, total_tokens, "Subida de archivos")

        response_data = []

        # Procesar archivos subidos
        for uploaded_file in uploaded_files:
            file_result = {
                "filename": uploaded_file["filename"],
                "size": uploaded_file["size"],
                "type": uploaded_file["type"],
                "path": uploaded_file["path"],
                "message": "Archivo procesado exitosamente",
                "tokens_earned": min(uploaded_file["size"] // 100, 50),
                "total_tokens": db.get_user_tokens(),
                "training": None,
            }

            # Buscar training info para este archivo
            training_info = next(
                (t for t in training_tasks if t["filename"] == uploaded_file["filename"]),
                None
            )
            if training_info:
                file_result["training"] = {
                    "started": training_info["training_started"],
                    "status": training_info["training_status"],
                    "error": training_info.get("error"),
                }

                # Ejecutar entrenamiento si está preparado
                if training_info["training_started"]:
                    try:
                        await train_on_uploaded_file(
                            training_info["file_path"],
                            training_info["file_type"]
                        )
                        file_result["training"]["status"] = "completed"
                        logging.info(
                            f"✅ Entrenamiento completado para: {uploaded_file['filename']}"
                        )
                    except Exception as e:
                        file_result["training"]["status"] = "error"
                        file_result["training"]["error"] = str(e)
                        logging.error(
                            f"❌ Error en entrenamiento para {uploaded_file['filename']}: {e}"
                        )

            response_data.append(UploadResult(**file_result))

        return response_data

    except Exception as e:
        logging.error(f"Error subiendo archivos: {e}")
        raise HTTPException(status_code=500, detail="Error procesando archivos")


@router.get("/stats")
async def get_upload_stats() -> Dict:
    """Obtener estadísticas de uploads"""
    try:
        upload_dir = Path("data/uploads")
        if not upload_dir.exists():
            return {"total_files": 0, "total_size": 0, "files": []}

        files = []
        total_size = 0

        for file_path in upload_dir.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "type": file_path.suffix[1:] if file_path.suffix else "unknown",
                })
                total_size += stat.st_size

        return {
            "total_files": len(files),
            "total_size": total_size,
            "files": files,
            "training_available": TRAINING_AVAILABLE,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {e}")


@router.delete("/files/{filename}")
async def delete_uploaded_file(filename: str):
    """Eliminar archivo subido"""
    try:
        # Sanitizar nombre de archivo para seguridad
        try:
            safe_filename = sanitize_filename(filename)
        except SecurityError as se:
            raise HTTPException(status_code=400, detail=f"Nombre de archivo inválido: {se}")

        file_path = Path("data/uploads") / safe_filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Archivo no encontrado")

        file_path.unlink()

        return {"success": True, "message": f"Archivo {safe_filename} eliminado"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando archivo: {e}")
