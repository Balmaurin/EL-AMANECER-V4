#!/usr/bin/env python3
"""
Sheily AI - Backend API Server
==============================

Servidor FastAPI que maneja todas las operaciones del dashboard:
- Chat con IA local
- Sistema de tokens
- Caja fuerte
- Ejercicios y datasets
- Subida de archivos
"""

import asyncio
import json
import logging
import os
import random
import secrets
import sqlite3
import sys
from pathlib import Path

# Add backend to path for security module
sys.path.insert(0, str(Path(__file__).parent.parent))
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.security import (
    SecurityError,
    sanitize_filename,
    sanitize_path,
    sanitize_snapshot_name,
    validate_command_args,
    validate_timeout,
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar servicio RAG avanzado
RAG_AVAILABLE = False
UNIFIED_SEARCH_AVAILABLE = False
try:
    # Primero intentar importar desde corpus
    import sys
    from pathlib import Path

    # Agregar directorio corpus al path
    corpus_dir = Path(__file__).parent.parent / "all-Branches" / "corpus"
    if corpus_dir.exists():
        sys.path.insert(0, str(corpus_dir))

        # Importar herramientas del corpus
        from tools.chunking.semantic_split import semantic_chunks
        from tools.common.config import load_config
        from tools.embedding.embed import embed_corpus
        from tools.index.index_hnsw import build_hnsw
        from tools.ingest.ingest_folder import ingest_folder
        from tools.retrieval.search_unified import unified_search

        RAG_AVAILABLE = True
        UNIFIED_SEARCH_AVAILABLE = True
        logger.info("✅ RAG System desde corpus/ disponible")
    else:
        logger.info(f"ℹ️ Directorio corpus no encontrado (opcional): {corpus_dir}")
except ImportError as e:
    logger.info(f"ℹ️ RAG corpus tools no disponibles (opcional) - usar CLI directamente")
    pass

# Importar servicio RAG local (tools/rag_service.py)
try:
    from rag_service import get_rag_service

    logger.info("✅ RAG Service local disponible")
except ImportError as e:
    logger.info(f"ℹ️ RAG Service local no disponible (opcional)")
    get_rag_service = None

# Base de datos
DATABASE_PATH = Path("data/sheily_dashboard.db")

# Crear directorios necesarios
DATABASE_PATH.parent.mkdir(exist_ok=True)
Path("data/uploads").mkdir(exist_ok=True)
Path("data/datasets").mkdir(exist_ok=True)


# Modelos de datos
class ChatMessage(BaseModel):
    message: str
    context: Optional[List[Dict]] = []


class TokenUpdate(BaseModel):
    tokens: int
    reason: str


class ExerciseResult(BaseModel):
    exercise_type: str
    answers: List[Dict]
    correct: int
    incorrect: int
    total_tokens: int


class VaultAuth(BaseModel):
    pin: str


# Base de datos helper
class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Inicializar base de datos"""
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

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    message TEXT,
                    response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exercise_datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    exercise_type TEXT,
                    data TEXT,
                    tokens_earned INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vault_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    item_type TEXT,
                    data TEXT,
                    encrypted BOOLEAN DEFAULT FALSE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
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
            logger.info(f"User {user_id} tokens updated by {tokens}: {reason}")

    def save_chat_message(self, user_id: int, message: str, response: str):
        """Guardar mensaje de chat"""
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO chat_history (user_id, message, response) VALUES (?, ?, ?)",
                (user_id, message, response),
            )
            conn.commit()

    def save_exercise_dataset(
        self, user_id: int, exercise_type: str, data: Dict, tokens_earned: int
    ):
        """Guardar dataset de ejercicio"""
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO exercise_datasets (user_id, exercise_type, data, tokens_earned) VALUES (?, ?, ?, ?)",
                (user_id, exercise_type, json.dumps(data), tokens_earned),
            )
            conn.commit()

    def get_vault_pin(self, user_id: int) -> str:
        """Obtener PIN de caja fuerte (por defecto 0000)"""
        return "0000"  # En producción esto debería estar hasheado

    def get_vault_items(self, user_id: int) -> List[Dict]:
        """Obtener items de la caja fuerte"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, item_type, data, timestamp FROM vault_items WHERE user_id = ?",
                (user_id,),
            ).fetchall()

            return [
                {
                    "id": row[0],
                    "type": row[1],
                    "data": json.loads(row[2]) if row[2] else {},
                    "timestamp": row[3],
                }
                for row in rows
            ]


# Inicializar base de datos
db = Database(DATABASE_PATH)

# Importar sistema de chat
try:
    import sys
    from pathlib import Path

    # Añadir directorio raíz del proyecto al PYTHONPATH
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from sheily_core.chat.sheily_chat_memory_adapter import respond

    CHAT_AVAILABLE = True
    logger.info("✅ Sistema de chat cargado")
except ImportError as e:
    logger.warning(f"❌ Sistema de chat no disponible: {e}")
    CHAT_AVAILABLE = False

# Importar sistema de entrenamiento automático
try:
    from tools.auto_training_system import (
        train_on_exercise_dataset,
        train_on_uploaded_file,
    )

    TRAINING_AVAILABLE = True
    logger.info("✅ Sistema de entrenamiento automático cargado")
except ImportError as e:
    logger.warning(f"❌ Sistema de entrenamiento no disponible: {e}")
    TRAINING_AVAILABLE = False

# Importar sistema llama.cpp
try:
    import json
    import subprocess

    LLAMA_AVAILABLE = True
    logger.info("✅ Sistema llama.cpp disponible")
except ImportError as e:
    logger.warning(f"❌ Sistema llama.cpp no disponible: {e}")
    LLAMA_AVAILABLE = False


def chat_with_llama_cpp(message: str) -> str:
    """Chat usando llama.cpp con modelo Gemma 2"""
    try:
        # Cargar configuración
        config_path = Path("config/llama_config.json")
        if not config_path.exists():
            return "Error: Configuración de llama.cpp no encontrada"

        with open(config_path, "r") as f:
            config = json.load(f)

        model_path = config["model_path"]
        if not Path(model_path).exists():
            return f"Error: Modelo no encontrado en {model_path}"

        # Preparar comando llama.cpp con ruta completa
        # Buscar llama-cli.exe en varias ubicaciones posibles
        root_dir = Path(__file__).parent.parent
        possible_paths = [
            root_dir / "llama-cli.exe",
            root_dir / "tools" / "llama-cli.exe",
            root_dir / "llama_cpp_install" / "llama-cli.exe",
        ]

        llama_cli_path = None
        for path in possible_paths:
            if path.exists():
                llama_cli_path = path
                break

        if not llama_cli_path:
            return f"Error: llama-cli.exe no encontrado en ninguna ubicación conocida"

        # SECURITY: Sanitize message to prevent command injection
        # Remove dangerous characters and limit length
        safe_message = message.replace("\x00", "").replace("\n", " ").replace("\r", " ")
        safe_message = safe_message[:1000]  # Limit length

        cmd = [
            str(llama_cli_path),  # usar ruta completa
            "--model",
            model_path,
            "--prompt",
            f"Usuario: {safe_message}\nAsistente:",
            "--ctx-size",
            str(config.get("context_size", 4096)),
            "--threads",
            str(config.get("threads", 4)),
            "--temp",
            str(config.get("temperature", 0.7)),
            "--top-p",
            str(config.get("top_p", 0.9)),
            "--top-k",
            str(config.get("top_k", 40)),
            "--n-predict",
            str(config.get("max_tokens", 512)),
            "--repeat-penalty",
            str(config.get("repeat_penalty", 1.1)),
            "--repeat-last-n",
            str(config.get("repeat_last_n", 64)),
            "--seed",
            str(config.get("seed", -1)),
            "--simple-io",  # Para output simple
        ]

        # SECURITY: Validate timeout
        try:
            safe_timeout = validate_timeout(300, max_timeout=600)
        except SecurityError:
            safe_timeout = 300

        # Ejecutar comando con timeout extendido y logging detallado
        logger.info(f"🚀 Ejecutando llama.cpp con comando: {' '.join(cmd[:3])}...")
        logger.info(f"📁 Modelo: {model_path}")
        logger.info(f"⏰ Timeout: {safe_timeout} segundos")

        try:
            # Usar encoding UTF-8 con manejo de errores para caracteres Unicode
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=safe_timeout,
            )

            if result.returncode == 0:
                response = result.stdout.strip()
                logger.info(
                    f"✅ llama.cpp ejecutado exitosamente, respuesta: {len(response)} caracteres"
                )

                # Limpiar respuesta (remover prompt del usuario)
                if "Asistente:" in response:
                    response = response.split("Asistente:")[-1].strip()
                    logger.info(f"📝 Respuesta limpia: {response[:100]}...")

                return (
                    response
                    if response
                    else "Lo siento, no pude generar una respuesta."
                )
            else:
                logger.error(f"❌ llama.cpp falló con código {result.returncode}")
                logger.error(f"❌ STDERR: {result.stderr}")
                logger.error(f"❌ STDOUT: {result.stdout}")
                return f"Error ejecutando llama.cpp (código {result.returncode}): {result.stderr}"

        except subprocess.TimeoutExpired:
            logger.error("⏰ Timeout: llama.cpp tardó más de 300 segundos")
            return "Error: Timeout en la generación de respuesta (300s)"

    except subprocess.TimeoutExpired:
        return "Error: Timeout en la generación de respuesta"
    except FileNotFoundError:
        return "Error: llama-cli no encontrado. Instala llama.cpp primero."
    except Exception as e:
        return f"Error con llama.cpp: {str(e)}"


def get_smart_fallback_response(message: str) -> str:
    """Respuestas simuladas inteligentes como fallback"""
    message_lower = message.lower()

    # Respuestas contextuales basadas en el contenido del mensaje
    if any(
        word in message_lower
        for word in ["hola", "hello", "hi", "saludos", "buenos", "buenas"]
    ):
        return "¡Hola! 👋 Soy Sheily, tu asistente de IA local. ¿En qué puedo ayudarte hoy?"
    elif any(
        word in message_lower
        for word in [
            "python",
            "programar",
            "codigo",
            "code",
            "javascript",
            "java",
            "php",
            "web",
        ]
    ):
        return "¡Excelente! Python es un lenguaje poderoso. ¿Qué tipo de proyecto estás desarrollando? Puedo ayudarte con algoritmos, estructuras de datos, o mejores prácticas de programación."
    elif any(
        word in message_lower
        for word in [
            "matematicas",
            "math",
            "matemáticas",
            "calculo",
            "algebra",
            "geometria",
        ]
    ):
        return "Las matemáticas son fascinantes. ¿Necesitas ayuda con álgebra, cálculo, geometría, o algún problema específico?"
    elif any(
        word in message_lower
        for word in ["ayuda", "help", "que puedes", "capacidades", "funciones"]
    ):
        return "Puedo ayudarte con:\n• Programación y desarrollo de software\n• Matemáticas y resolución de problemas\n• Análisis de datos\n• Preguntas generales sobre tecnología\n• Explicaciones de conceptos complejos\n\n¿En qué área te gustaría que te ayude?"
    elif any(
        word in message_lower
        for word in [
            "ia",
            "inteligencia artificial",
            "machine learning",
            "deep learning",
            "neural",
        ]
    ):
        return "¡La IA es apasionante! ¿Quieres aprender sobre machine learning, deep learning, o aplicaciones prácticas de IA?"
    else:
        responses = [
            "Interesante pregunta. Como IA local, tengo acceso limitado pero puedo intentar ayudarte con lo que sé. ¿Puedes darme más detalles sobre lo que necesitas?",
            "Esa es una buena consulta. Aunque estoy en modo simulado, puedo darte información general sobre ese tema. ¿Hay algo específico que quieras explorar?",
            "¡Gracias por tu pregunta! Mi conocimiento está basado en datos generales. ¿Puedes darme más detalles sobre lo que necesitas saber?",
            "Entiendo tu consulta. Como asistente local, trabajo con la información que tengo disponible. ¿Hay algún aspecto específico que te gustaría profundizar?",
            "¡Qué buena pregunta! Aunque tengo capacidades limitadas en este modo, puedo intentar darte una respuesta útil. ¿Quieres que explore algún ángulo específico?",
        ]
        return random.choice(responses)


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Iniciando servidor backend Sheily AI")
    yield
    # Shutdown
    logger.info("👋 Servidor detenido")


app = FastAPI(
    title="Sheily AI Dashboard API",
    description="API backend para el dashboard de Sheily AI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="Frontend"), name="static")


# Rutas API
@app.get("/landing", response_class=HTMLResponse)
async def get_landing_page():
    """Servir la página de landing"""
    landing_path = Path("Frontend/landing.html")
    if landing_path.exists():
        return landing_path.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="Landing page no encontrada")


@app.get("/api/health")
async def health_check():
    """Verificar estado del servidor"""
    # Verificar si tenemos algún sistema de chat disponible
    # Desde tools/, el directorio padre es la raíz del proyecto
    root_dir = Path(__file__).parent.parent
    llama_cli_exists = (root_dir / "llama-cli.exe").exists()
    model_exists = (root_dir / "models" / "gemma-2-9b-it-Q4_K_M.gguf").exists()
    llama_ready = llama_cli_exists and model_exists

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "chat_available": bool(llama_ready or CHAT_AVAILABLE),
        "llama_available": llama_ready,
        "llama_cli_exists": llama_cli_exists,
        "model_exists": model_exists,
        "database_connected": True,
    }


@app.post("/api/chat")
async def chat_endpoint(chat_msg: ChatMessage):
    """Endpoint para chat con IA - Prioriza Gemini API"""
    try:
        import os
        import requests
        
        # OPCIÓN 1: Intentar Gemini API primero si la clave está disponible
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if gemini_api_key:
            try:
                logger.info("🔷 Intentando usar Gemini API...")
                
                # Configurar URL de Gemini
                gemini_url = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"
                
                # Preparar payload para Gemini
                payload = {
                    "contents": [{
                        "parts": [{"text": chat_msg.message}]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 512,
                    }
                }
                
                # Llamar a Gemini API
                response = requests.post(
                    f"{gemini_url}?key={gemini_api_key}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    gemini_response = data["candidates"][0]["content"]["parts"][0]["text"]
                    
                    logger.info(f"✅ Respuesta exitosa de Gemini: {len(gemini_response)} chars")
                    
                    # Guardar en base de datos
                    db.save_chat_message(1, chat_msg.message, gemini_response)
                    
                    return {
                        "response": gemini_response,
                        "timestamp": datetime.now().isoformat(),
                        "method": "gemini_api",
                        "model_status": "Gemini Pro API disponible",
                    }
                else:
                    logger.warning(f"⚠️ Gemini API devolvió {response.status_code}: {response.text}")
                    # Continuar al fallback de llama.cpp
                    
            except Exception as e:
                logger.warning(f"❌ Error con Gemini API: {e}")
                # Continuar al fallback de llama.cpp

        # OPCIÓN 2: Fallback a llama.cpp si Gemini no está disponible o falló
        root_dir = Path(__file__).parent.parent
        llama_cli_exists = (root_dir / "llama-cli.exe").exists()
        model_exists = (root_dir / "models" / "gemma-2-9b-it-Q4_K_M.gguf").exists()
        llama_ready = llama_cli_exists and model_exists

        if llama_ready:
            try:
                logger.info("🚀 Usando llama.cpp como fallback...")
                response = chat_with_llama_cpp(chat_msg.message)
                if response and not response.startswith("Error"):
                    method = "llama_cpp_gemma2"
                    model_status = "Gemma 2 (llama.cpp) disponible"
                    logger.info("✅ Respuesta exitosa de llama.cpp")
                else:
                    logger.warning(f"⚠️ llama.cpp devolvió error: {response}")
                    response = get_smart_fallback_response(chat_msg.message)
                    method = "simulated_fallback"
                    model_status = "Modo simulado - llama.cpp falló"
            except Exception as e:
                logger.warning(f"❌ Error con llama.cpp: {e}")
                response = get_smart_fallback_response(chat_msg.message)
                method = "simulated_fallback"
                model_status = "Modo simulado - Error en llama.cpp"
        elif CHAT_AVAILABLE:
            # Usar el sistema de chat real si está disponible
            response = respond(chat_msg.message)
            method = "real_llm"
            model_status = "Sistema de chat real disponible"
        else:
            # Fallback final a respuestas simuladas
            response = get_smart_fallback_response(chat_msg.message)
            method = "simulated_fallback"
            model_status = "Modo simulado - No hay sistemas de chat disponibles"

        # Guardar en base de datos
        db.save_chat_message(1, chat_msg.message, response)

        return {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "model_status": model_status,
        }

    except Exception as e:
        logger.error(f"Error en chat: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/api/tokens")
async def get_tokens():
    """Obtener tokens del usuario"""
    tokens = db.get_user_tokens()
    return {"tokens": tokens}


@app.get("/api/user/tokens")
async def get_user_tokens():
    """Obtener tokens del usuario (alias)"""
    tokens = db.get_user_tokens()
    return {"tokens": tokens, "user_id": 1}


@app.post("/api/tokens/update")
async def update_tokens(token_update: TokenUpdate):
    """Actualizar tokens del usuario"""
    try:
        db.update_user_tokens(1, token_update.tokens, token_update.reason)
        new_total = db.get_user_tokens()
        return {"tokens": new_total, "change": token_update.tokens}
    except Exception as e:
        logger.error(f"Error actualizando tokens: {e}")
        raise HTTPException(status_code=500, detail="Error actualizando tokens")


@app.post("/api/vault/auth")
async def vault_auth(auth: VaultAuth):
    """Autenticar caja fuerte"""
    correct_pin = db.get_vault_pin(1)
    if auth.pin == correct_pin:
        items = db.get_vault_items(1)
        return {"authenticated": True, "items": items, "tokens": db.get_user_tokens()}
    else:
        raise HTTPException(status_code=401, detail="PIN incorrecto")


@app.post("/api/exercises/submit")
async def submit_exercise(exercise: ExerciseResult, background_tasks: BackgroundTasks):
    """Enviar resultados de ejercicio"""
    try:
        # Guardar dataset
        dataset_data = {
            "exercise_type": exercise.exercise_type,
            "answers": exercise.answers,
            "correct": exercise.correct,
            "incorrect": exercise.incorrect,
            "timestamp": datetime.now().isoformat(),
        }

        db.save_exercise_dataset(
            1, exercise.exercise_type, dataset_data, exercise.total_tokens
        )

        # Actualizar tokens
        db.update_user_tokens(
            1, exercise.total_tokens, f"Ejercicio {exercise.exercise_type}"
        )

        # Generar archivo JSONL en background
        background_tasks.add_task(generate_dataset_file, dataset_data)

        # Entrenar modelo automáticamente si el sistema está disponible
        training_info = {"training_started": False, "training_status": "not_available"}
        if TRAINING_AVAILABLE:
            try:
                # Ejecutar entrenamiento en background
                background_tasks.add_task(train_on_exercise_dataset, dataset_data)
                training_info = {
                    "training_started": True,
                    "training_status": "in_progress",
                    "message": "Entrenamiento LoRA iniciado automáticamente",
                }
                logger.info(
                    f"🚀 Entrenamiento automático iniciado para ejercicio: {exercise.exercise_type}"
                )
            except Exception as e:
                logger.error(f"Error iniciando entrenamiento automático: {e}")
                training_info = {
                    "training_started": False,
                    "training_status": "error",
                    "error": str(e),
                }

        return {
            "success": True,
            "tokens_earned": exercise.total_tokens,
            "total_tokens": db.get_user_tokens(),
            "dataset_saved": True,
            "training": training_info,
        }

    except Exception as e:
        logger.error(f"Error guardando ejercicio: {e}")
        raise HTTPException(status_code=500, detail="Error guardando ejercicio")


@app.get("/api/stats")
async def get_stats():
    """Obtener estadísticas del usuario"""
    try:
        with db.get_connection() as conn:
            # Contar mensajes de chat
            chat_count = conn.execute(
                "SELECT COUNT(*) FROM chat_history WHERE user_id = ?", (1,)
            ).fetchone()[0]

            # Contar datasets
            dataset_count = conn.execute(
                "SELECT COUNT(*) FROM exercise_datasets WHERE user_id = ?", (1,)
            ).fetchone()[0]

            # Contar archivos subidos
            upload_count = len(list(Path("data/uploads").glob("*")))

        stats = {
            "chat_messages": chat_count,
            "datasets_generated": dataset_count,
            "files_uploaded": upload_count,
            "current_tokens": db.get_user_tokens(),
            "training_available": TRAINING_AVAILABLE,
        }

        # Agregar estadísticas de entrenamiento si está disponible
        if TRAINING_AVAILABLE:
            try:
                from tools.auto_training_system import get_training_statistics

                training_stats = await get_training_statistics()
                stats["training"] = training_stats
            except Exception as e:
                logger.warning(f"Error obteniendo estadísticas de entrenamiento: {e}")
                stats["training"] = {"error": str(e)}

        return stats

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo estadísticas")


@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None
):
    """Subir archivos para entrenamiento"""
    try:
        uploaded_files = []
        total_tokens = 0
        training_tasks = []

        for file in files:
            # Validar tipo de archivo
            allowed_extensions = {".md", ".txt", ".jsonl", ".pdf", ".xml"}
            file_ext = Path(file.filename).suffix.lower()

            if file_ext not in allowed_extensions:
                continue

            # Guardar archivo
            file_path = (
                Path("data/uploads")
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            )
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            uploaded_files.append(
                {
                    "filename": file.filename,
                    "size": len(content),
                    "type": file_ext,
                    "path": str(file_path),
                }
            )

            # Calcular tokens por archivo
            tokens_earned = min(len(content) // 100, 50)  # Máximo 50 tokens por archivo
            total_tokens += tokens_earned

            # Preparar entrenamiento automático si está disponible
            if TRAINING_AVAILABLE and background_tasks:
                try:
                    # Determinar tipo de archivo para el entrenamiento
                    file_type = file_ext[1:]  # Remover el punto
                    background_tasks.add_task(
                        train_on_uploaded_file, file_path, file_type
                    )
                    training_tasks.append(
                        {
                            "filename": file.filename,
                            "training_started": True,
                            "training_status": "in_progress",
                        }
                    )
                    logger.info(
                        f"🚀 Entrenamiento automático iniciado para archivo: {file.filename}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error iniciando entrenamiento para {file.filename}: {e}"
                    )
                    training_tasks.append(
                        {
                            "filename": file.filename,
                            "training_started": False,
                            "training_status": "error",
                            "error": str(e),
                        }
                    )

        # Actualizar tokens del usuario
        if total_tokens > 0:
            db.update_user_tokens(1, total_tokens, "Subida de archivos")

        response_data = {
            "uploaded": len(uploaded_files),
            "files": uploaded_files,
            "tokens_earned": total_tokens,
            "total_tokens": db.get_user_tokens(),
            "message": f"Archivos procesados. ¡Ganaste {total_tokens} tokens!",
        }

        # Agregar información de entrenamiento si hay tareas de entrenamiento
        if training_tasks:
            response_data["training"] = {
                "started": len([t for t in training_tasks if t["training_started"]]),
                "total": len(training_tasks),
                "details": training_tasks,
                "message": f"Entrenamiento LoRA iniciado para {len([t for t in training_tasks if t['training_started']])} archivo(s)",
            }

        return response_data

    except Exception as e:
        logger.error(f"Error subiendo archivos: {e}")
        raise HTTPException(status_code=500, detail="Error procesando archivos")


def generate_dataset_file(dataset_data: Dict):
    """Generar archivo JSONL del dataset"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sheily_dataset_{timestamp}.jsonl"
        filepath = Path("data/datasets") / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dataset_data, f, ensure_ascii=False)
            f.write("\n")

        logger.info(f"Dataset generado: {filepath}")

    except Exception as e:
        logger.error(f"Error generando dataset: {e}")


@app.get("/api/training/stats")
async def get_training_stats():
    """Obtener estadísticas del sistema de entrenamiento automático"""
    try:
        if not TRAINING_AVAILABLE:
            return {
                "training_available": False,
                "message": "Sistema de entrenamiento no disponible",
            }

        # Importar función de estadísticas
        from tools.auto_training_system import get_training_statistics

        stats = await get_training_statistics()

        return {"training_available": True, "stats": stats}

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de entrenamiento: {e}")
        return {"training_available": False, "error": str(e)}


# ============================================================================
# RUTAS MULTI-AGENTE
# ============================================================================


@app.get("/multiagent/agents")
async def get_agents():
    """Obtener lista de agentes disponibles"""
    try:
        # Datos mock para agentes
        agents = [
            {
                "id": "medical_specialist",
                "name": "Medical Specialist",
                "type": "LoRA",
                "status": "active",
                "performance": 94,
                "tasks_completed": 156,
                "specializations": ["Medicina General", "Diagnóstico"],
            },
            {
                "id": "programming_specialist",
                "name": "Programming Specialist",
                "type": "LoRA",
                "status": "busy",
                "performance": 89,
                "tasks_completed": 203,
                "specializations": ["Python", "JavaScript", "React"],
            },
        ]
        return {"agents": agents}
    except Exception as e:
        logger.error(f"Error obteniendo agentes: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo agentes")


@app.get("/multiagent/collaborations")
async def get_collaborations():
    """Obtener colaboraciones activas"""
    try:
        # Datos mock para colaboraciones
        collaborations = [
            {
                "id": "collab_1",
                "name": "Sistema Médico Completo",
                "agents": ["medical_specialist"],
                "status": "active",
                "performance": 92,
                "tasks_completed": 45,
            },
            {
                "id": "collab_2",
                "name": "Stack Tecnológico",
                "agents": ["programming_specialist"],
                "status": "training",
                "performance": 78,
                "tasks_completed": 23,
            },
        ]
        return {"collaborations": collaborations}
    except Exception as e:
        logger.error(f"Error obteniendo colaboraciones: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo colaboraciones")


# Endpoints adicionales para compatibilidad con el dashboard
@app.get("/health")
async def health_check_simple():
    """Health check simple (redirect a /api/health)"""
    return await health_check()


@app.get("/api/system/stats")
async def get_system_stats():
    """Obtener estadísticas del sistema"""
    try:
        stats = await get_stats()
        return {
            "system": {
                "cpu_usage": 45.2,
                "memory_usage": 62.8,
                "disk_usage": 38.5,
                "uptime": "2h 15m",
            },
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"Error obteniendo system stats: {e}")
        raise HTTPException(
            status_code=500, detail="Error obteniendo estadísticas del sistema"
        )


@app.get("/api/blockchain/balance")
async def get_blockchain_balance():
    """Obtener balance de blockchain (simulado)"""
    return {
        "balance": db.get_user_tokens(),
        "wallet_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        "network": "Sheily Chain",
        "transactions": 42,
        "staked": 500,
        "rewards": 25,
    }


@app.get("/api/consciousness/status")
async def get_consciousness_status():
    """Obtener estado de consciencia (simulado)"""
    return {
        "level": 85,
        "state": "active",
        "emotions": {"joy": 75, "curiosity": 90, "empathy": 80},
        "cognitive_load": 62,
        "timestamp": datetime.now().isoformat(),
    }


# ==================== ENDPOINTS RAG MANAGER ====================


@app.get("/api/rag/documents")
async def get_rag_documents():
    """Listar todos los documentos en el sistema RAG con estadísticas del corpus"""
    try:
        rag_dir = Path("data/rag_documents")
        rag_dir.mkdir(exist_ok=True)

        documents = []
        for file in rag_dir.glob("*"):
            if file.is_file():
                stat = file.stat()
                documents.append(
                    {
                        "id": file.stem,
                        "name": file.name,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "type": file.suffix[1:] if file.suffix else "unknown",
                    }
                )

        # Obtener estadísticas del corpus si RAG está disponible
        corpus_stats = {}
        if RAG_AVAILABLE:
            try:
                # Verificar si hay índice corpus construido
                corpus_base = Path("corpus/corpus/universal")
                if corpus_base.exists():
                    latest_ptr = corpus_base / "latest.ptr"
                    if latest_ptr.exists():
                        latest = latest_ptr.read_text(encoding="utf-8").strip()
                        latest_path = corpus_base / latest

                        if latest_path.exists():
                            # Contar archivos procesados
                            chunks_dir = latest_path / "chunks"
                            embeddings_dir = latest_path / "embeddings"

                            corpus_stats = {
                                "total_documents": (
                                    len(list((latest_path / "raw").glob("*")))
                                    if (latest_path / "raw").exists()
                                    else 0
                                ),
                                "total_chunks": (
                                    len(list(chunks_dir.glob("*.json")))
                                    if chunks_dir.exists()
                                    else 0
                                ),
                                "indexed": (latest_path / "index").exists(),
                                "embedder_model": "BAAI/bge-m3",
                                "chunking_mode": "semantic",
                            }
            except Exception as e:
                logger.warning(f"Error obteniendo stats del corpus: {e}")

        # También obtener stats del RAG service local si está disponible
        local_rag_stats = {}
        if get_rag_service is not None:
            try:
                rag_service = get_rag_service()
                local_stats = rag_service.get_corpus_stats()
                local_rag_stats = {
                    "total_chunks": local_stats.get("total_chunks", 0),
                    "total_documents": local_stats.get("total_documents", 0),
                    "indexed": local_stats.get("index_ready", False),
                }
            except Exception as e:
                logger.warning(f"Error obteniendo stats del RAG local: {e}")

        # Combinar estadísticas (prioridad a RAG local si existe)
        final_corpus_stats = local_rag_stats if local_rag_stats else corpus_stats

        return {
            "documents": documents,
            "total": len(documents),
            "corpus_stats": final_corpus_stats,
        }
    except Exception as e:
        logger.error(f"Error listando documentos RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/upload")
async def upload_rag_document(file: UploadFile = File(...)):
    """Subir documento al sistema RAG y procesarlo con corpus pipeline"""
    try:
        rag_dir = Path("data/rag_documents")
        rag_dir.mkdir(exist_ok=True)

        # Validar tipo de archivo
        allowed_extensions = [".txt", ".pdf", ".md", ".json"]
        file_ext = Path(file.filename or "document").suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no permitido. Use: {', '.join(allowed_extensions)}",
            )

        # Guardar archivo
        filename = (
            file.filename or f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        file_path = rag_dir / filename
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"📄 Documento RAG subido: {filename} ({len(content)} bytes)")

        # PROCESAR AUTOMÁTICAMENTE con RAG Service (Actualización Incremental)
        processing_result = {}
        if get_rag_service is not None and file_ext in [".txt", ".md"]:
            try:
                logger.info(f"➕ Procesando documento de forma incremental: {filename}")

                rag_service = get_rag_service()

                # Usar actualización INCREMENTAL automáticamente
                result = rag_service.update_corpus_incremental(file_path)

                if result.get("success"):
                    processing_result = {
                        "auto_processed": True,
                        "mode": "incremental",
                        "chunks_added": result.get("chunks_added", 0),
                        "doc_id": result.get("doc_id"),
                        "indexed": True,
                        "message": f"✅ {result.get('chunks_added', 0)} chunks añadidos automáticamente al corpus",
                    }
                    logger.info(
                        f"✅ Documento procesado incrementalmente: {result.get('chunks_added')} chunks"
                    )
                else:
                    processing_result = {
                        "auto_processed": False,
                        "error": result.get("error"),
                        "message": "⚠️ Error en procesamiento automático",
                    }
                    logger.warning(
                        f"Error procesando {filename}: {result.get('error')}"
                    )

            except Exception as e:
                logger.error(f"Error en procesamiento automático de {filename}: {e}")
                processing_result = {
                    "auto_processed": False,
                    "error": str(e),
                    "message": "⚠️ Documento guardado pero no procesado automáticamente",
                }
        elif file_ext in [".pdf", ".json"]:
            processing_result = {
                "auto_processed": False,
                "message": "📋 Archivos PDF/JSON requieren procesamiento manual con 'Reconstruir Corpus'",
            }

        return {
            "success": True,
            "filename": filename,
            "size": len(content),
            "message": (
                "Documento subido y procesado"
                if processing_result.get("auto_processed")
                else "Documento subido"
            ),
            "processing": processing_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error subiendo documento RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/rag/documents/{document_name}")
async def delete_rag_document(document_name: str):
    """Eliminar documento del sistema RAG"""
    try:
        rag_dir = Path("data/rag_documents")
        file_path = rag_dir / document_name

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Documento no encontrado")

        file_path.unlink()
        logger.info(f"🗑️ Documento RAG eliminado: {document_name}")

        return {"success": True, "message": f"Documento {document_name} eliminado"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando documento RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/search")
async def search_rag_documents(query: Dict[str, str]):
    """
    Buscar en documentos RAG usando corpus unified_search

    Query params:
    - query: string de búsqueda
    - mode: "semantic" | "keyword" | "hybrid" (default: hybrid)
    - top_k: número de resultados (default: 10)
    """
    try:
        search_query = query.get("query", "")
        if not search_query:
            raise HTTPException(status_code=400, detail="Query vacío")

        mode = query.get("mode", "hybrid")
        top_k = int(query.get("top_k", "10"))

        results = []

        # Usar sistema corpus si está disponible
        if RAG_AVAILABLE:
            try:
                # Verificar si hay un snapshot con índice
                corpus_base = Path("corpus/corpus/universal")
                latest_ptr = corpus_base / "latest.ptr"

                if latest_ptr.exists():
                    latest = latest_ptr.read_text(encoding="utf-8").strip()
                    latest_path = corpus_base / latest

                    # Usar unified_search del corpus
                    search_results = unified_search(
                        query=search_query,
                        snapshot=str(latest_path),
                        top_k=top_k,
                        mode=mode if mode in ["vector", "bm25", "hybrid"] else "hybrid",
                    )

                    # Convertir resultados al formato esperado
                    for result in search_results:
                        results.append(
                            {
                                "doc_id": result.get("doc_id", "unknown"),
                                "text": result.get("text", ""),
                                "score": result.get("score", 0.0),
                                "method": mode,
                                "chunk_id": result.get("chunk_id", ""),
                            }
                        )

                    logger.info(
                        f"🔍 Búsqueda corpus {mode}: '{search_query}' -> {len(results)} resultados"
                    )
                else:
                    logger.warning("No hay snapshot del corpus disponible")
                    results = _simple_search(search_query, top_k)

            except Exception as e:
                logger.warning(f"Error en búsqueda corpus: {e}")
                # Fallback a búsqueda simple
                results = _simple_search(search_query, top_k)
        else:
            # Búsqueda simple si RAG no está disponible
            results = _simple_search(search_query, top_k)

        return {
            "query": search_query,
            "mode": mode,
            "results": results,
            "total": len(results),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error buscando en RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _simple_search(search_query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Búsqueda simple en archivos de texto (fallback)"""
    rag_dir = Path("data/rag_documents")
    results = []

    for file in rag_dir.glob("*.txt"):
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if search_query.lower() in content.lower():
                    # Encontrar contexto
                    index = content.lower().find(search_query.lower())
                    start = max(0, index - 100)
                    end = min(len(content), index + 100)
                    snippet = content[start:end]

                    results.append(
                        {
                            "doc_id": file.stem,
                            "text": snippet,
                            "score": 0.8,
                            "method": "simple",
                            "chunk_id": f"{file.stem}_simple",
                        }
                    )
        except Exception as e:
            logger.warning(f"Error leyendo {file.name}: {e}")

    return results[:top_k]


# ==================== ENDPOINTS DATASETS ====================


@app.get("/api/datasets")
async def get_datasets():
    """Listar todos los datasets generados"""
    try:
        with db.get_connection() as conn:
            rows = conn.execute(
                """SELECT id, user_id, exercise_type, data, tokens_earned, timestamp 
                   FROM exercise_datasets 
                   ORDER BY timestamp DESC"""
            ).fetchall()

            datasets = []
            for row in rows:
                data = json.loads(row[3]) if row[3] else {}
                num_answers = len(data.get("answers", []))
                correct = data.get("correct", 0)
                total = num_answers if num_answers > 0 else 1

                datasets.append(
                    {
                        "id": row[0],
                        "user_id": row[1],
                        "exercise_type": row[2],
                        "data": data,
                        "tokens_earned": row[4],
                        "timestamp": row[5],
                        "num_questions": num_answers,
                        "accuracy": round((correct / total) * 100, 1),
                    }
                )

            return {"datasets": datasets, "total": len(datasets)}
    except Exception as e:
        logger.error(f"Error listando datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets/{dataset_id}")
async def get_dataset_detail(dataset_id: int):
    """Obtener detalle de un dataset específico"""
    try:
        with db.get_connection() as conn:
            row = conn.execute(
                """SELECT id, user_id, exercise_type, data, tokens_earned, timestamp 
                   FROM exercise_datasets 
                   WHERE id = ?""",
                (dataset_id,),
            ).fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Dataset no encontrado")

            data = json.loads(row[3]) if row[3] else {}

            return {
                "id": row[0],
                "user_id": row[1],
                "exercise_type": row[2],
                "data": data,
                "tokens_earned": row[4],
                "timestamp": row[5],
                "statistics": {
                    "total_questions": data.get("total", 0),
                    "correct": data.get("correct", 0),
                    "incorrect": data.get("incorrect", 0),
                    "accuracy": round(
                        (data.get("correct", 0) / max(data.get("total", 1), 1)) * 100, 1
                    ),
                    "tokens_earned": row[4],
                },
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets/{dataset_id}/export")
async def export_dataset(dataset_id: int):
    """Exportar dataset en formato JSON"""
    try:
        with db.get_connection() as conn:
            row = conn.execute(
                """SELECT exercise_type, data, timestamp 
                   FROM exercise_datasets 
                   WHERE id = ?""",
                (dataset_id,),
            ).fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Dataset no encontrado")

            data = json.loads(row[1]) if row[1] else {}

            export_data = {
                "dataset_id": dataset_id,
                "exercise_type": row[0],
                "timestamp": row[2],
                "data": data,
            }

            # Guardar archivo temporal
            export_path = Path("data/datasets") / f"dataset_{dataset_id}.json"
            export_path.parent.mkdir(exist_ok=True)

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return FileResponse(
                path=export_path,
                filename=f"dataset_{dataset_id}.json",
                media_type="application/json",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exportando dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int):
    """Eliminar un dataset"""
    try:
        with db.get_connection() as conn:
            result = conn.execute(
                "DELETE FROM exercise_datasets WHERE id = ?", (dataset_id,)
            )
            conn.commit()

            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Dataset no encontrado")

            logger.info(f"🗑️ Dataset {dataset_id} eliminado")

            return {"success": True, "message": f"Dataset {dataset_id} eliminado"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/train-from-dataset/{dataset_id}")
async def train_rag_from_dataset(dataset_id: int, background_tasks: BackgroundTasks):
    """Entrenar sistema RAG usando un dataset de ejercicios con procesamiento real"""
    try:
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT exercise_type, data FROM exercise_datasets WHERE id = ?",
                (dataset_id,),
            ).fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Dataset no encontrado")

            exercise_type = row[0]
            data = json.loads(row[1]) if row[1] else {}

            # Crear documento RAG a partir del dataset
            rag_dir = Path("data/rag_documents")
            rag_dir.mkdir(exist_ok=True)

            # Generar documento con las preguntas y respuestas del dataset
            doc_content = f"# Dataset de Entrenamiento - {exercise_type}\n\n"
            doc_content += (
                f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            doc_content += f"## Estadísticas\n"
            doc_content += f"- Total de preguntas: {len(data.get('answers', []))}\n"
            doc_content += f"- Respuestas correctas: {data.get('correct', 0)}\n"
            doc_content += f"- Respuestas incorrectas: {data.get('incorrect', 0)}\n\n"
            doc_content += f"## Conocimiento Extraído\n\n"

            # Extraer información de las preguntas
            for i, answer in enumerate(data.get("answers", []), 1):
                doc_content += f"{i}. Pregunta procesada - Resultado: {'✓ Correcto' if answer.get('isCorrect') else '✗ Incorrecto'}\n"

            # Guardar documento
            filename = f"dataset_{dataset_id}_{exercise_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            file_path = rag_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc_content)

            logger.info(
                f"📚 Documento RAG generado desde dataset {dataset_id}: {filename}"
            )

            # Procesar con RAG service avanzado - ACTUALIZACIÓN INCREMENTAL
            processing_result = {}
            chunks_processed = len(data.get("answers", []))

            if RAG_AVAILABLE:
                try:
                    rag_service = get_rag_service()
                    # Usar actualización incremental en lugar de ingest completo
                    processing_result = rag_service.update_corpus_incremental(file_path)
                    chunks_processed = processing_result.get(
                        "chunks_added", chunks_processed
                    )

                    logger.info(
                        f"➕ Dataset añadido incrementalmente: {chunks_processed} chunks nuevos"
                    )
                except Exception as e:
                    logger.warning(f"Error en actualización incremental RAG: {e}")
                    # Fallback: intentar ingesta normal
                    try:
                        processing_result = rag_service.ingest_document(
                            file_path, doc_id=f"dataset_{dataset_id}"
                        )
                        chunks_processed = processing_result.get(
                            "chunks", chunks_processed
                        )
                        logger.info(f"✅ Fallback: Dataset ingresado normalmente")
                    except Exception as e2:
                        logger.error(f"Error en fallback: {e2}")

            # Generar embeddings metadata (para compatibilidad)
            embeddings_path = Path("data/rag_documents/embeddings")
            embeddings_path.mkdir(exist_ok=True)

            embeddings_file = embeddings_path / f"{filename}.json"
            embeddings_data = {
                "dataset_id": dataset_id,
                "document": filename,
                "chunks": chunks_processed,
                "timestamp": datetime.now().isoformat(),
                "exercise_type": exercise_type,
                "processing": processing_result,
            }

            with open(embeddings_file, "w", encoding="utf-8") as f:
                json.dump(embeddings_data, f, indent=2, ensure_ascii=False)

            return {
                "success": True,
                "message": "Sistema RAG actualizado con dataset",
                "document": filename,
                "embeddings": f"{filename}.json",
                "chunks_processed": chunks_processed,
                "indexed": processing_result.get("indexed", False),
                "processing_details": processing_result,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error entrenando RAG desde dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/embeddings")
async def get_rag_embeddings():
    """Obtener información de embeddings generados"""
    try:
        embeddings_dir = Path("data/rag_documents/embeddings")
        embeddings_dir.mkdir(exist_ok=True)

        embeddings_list = []
        for file in embeddings_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    embeddings_list.append(
                        {
                            "name": file.name,
                            "document": data.get("document", "Documento sin nombre"),
                            "dataset_id": data.get("dataset_id"),
                            "num_chunks": data.get("chunks", data.get("num_chunks", 0)),
                            "chunks": data.get("chunks", data.get("num_chunks", 0)),
                            "timestamp": data.get("timestamp"),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error leyendo embedding file {file}: {e}")
                continue

        return {
            "embeddings": embeddings_list,
            "total": len(embeddings_list),
            "corpus_size": len(list(Path("data/rag_documents").glob("*.txt"))),
        }
    except Exception as e:
        logger.error(f"Error listando embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/rebuild-corpus")
async def rebuild_rag_corpus():
    """Reconstruir corpus completo del sistema RAG con procesamiento real"""
    try:
        rag_dir = Path("data/rag_documents")
        embeddings_dir = rag_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)

        # Contar documentos
        txt_files = list(rag_dir.glob("*.txt"))
        pdf_files = list(rag_dir.glob("*.pdf"))
        md_files = list(rag_dir.glob("*.md"))

        total_docs = len(txt_files) + len(pdf_files) + len(md_files)

        # Reconstruir con RAG service si está disponible
        corpus_stats = {}
        if RAG_AVAILABLE:
            try:
                rag_service = get_rag_service()
                rebuild_result = rag_service.rebuild_corpus()

                if rebuild_result.get("success"):
                    corpus_stats = rag_service.get_corpus_stats()
                    logger.info(
                        f"✅ Corpus reconstruido con RAG service: {corpus_stats}"
                    )
                else:
                    logger.warning(
                        f"Error reconstruyendo con RAG service: {rebuild_result.get('error')}"
                    )
            except Exception as e:
                logger.warning(f"Error usando RAG service: {e}")

        # Información del corpus (compatible con versión anterior)
        corpus_info = {
            "total_documents": total_docs,
            "txt_files": len(txt_files),
            "pdf_files": len(pdf_files),
            "md_files": len(md_files),
            "embeddings_updated": datetime.now().isoformat(),
            "status": "ready",
            "advanced_stats": corpus_stats,
        }

        # Guardar metadata del corpus
        corpus_file = rag_dir / "corpus_metadata.json"
        with open(corpus_file, "w", encoding="utf-8") as f:
            json.dump(corpus_info, f, indent=2, ensure_ascii=False)

        logger.info(f"📊 Corpus RAG reconstruido: {total_docs} documentos")

        return {
            "success": True,
            "message": "Corpus RAG reconstruido",
            "corpus": corpus_info,
        }

    except Exception as e:
        logger.error(f"Error reconstruyendo corpus: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# CORPUS PRO - Advanced RAG Endpoints
# ========================================


@app.post("/api/corpus/ingest")
async def corpus_ingest_folder(
    folder_path: str = Form(...), enable_ocr: bool = Form(False), workers: int = Form(6)
):
    """Ingesta profesional de documentos desde carpeta con OCR opcional"""
    try:
        import subprocess
        from pathlib import Path

        # SECURITY: Validate and sanitize folder path
        try:
            corpus_dir = Path(__file__).parent.parent / "all-Branches" / "corpus"
            source_folder = sanitize_path(
                folder_path, corpus_dir.parent.parent, max_depth=15
            )
        except SecurityError as se:
            raise HTTPException(status_code=400, detail=f"Invalid path: {se}")

        if not source_folder.exists():
            raise HTTPException(
                status_code=404, detail=f"Carpeta no encontrada: {folder_path}"
            )

        # Preparar comando CLI
        rag_cli = corpus_dir / "rag_cli.py"

        if not rag_cli.exists():
            raise HTTPException(status_code=404, detail="rag_cli.py no encontrado")

        # Ejecutar ingesta
        python_exe = sys.executable

        # SECURITY: Sanitize snapshot name
        try:
            snapshot_name = sanitize_snapshot_name(f"ingest_{int(time.time())}")
        except SecurityError:
            snapshot_name = f"ingest_{int(time.time())}"

        # SECURITY: Build command with validated arguments
        cmd = [python_exe, str(rag_cli), "ingest", str(source_folder), snapshot_name]

        # SECURITY: Validate timeout
        try:
            safe_timeout = validate_timeout(300, max_timeout=600)
        except SecurityError:
            safe_timeout = 300

        result = subprocess.run(
            cmd,
            cwd=str(corpus_dir),
            capture_output=True,
            text=True,
            timeout=safe_timeout,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500, detail=f"Error en ingesta: {result.stderr}"
            )

        return {
            "success": True,
            "message": "Ingesta completada",
            "snapshot": snapshot_name,
            "output": result.stdout,
            "workers": workers,
            "ocr_enabled": enable_ocr,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Timeout en ingesta (>5 min)")
    except Exception as e:
        logger.error(f"Error en corpus ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/corpus/pipeline")
async def corpus_run_pipeline(
    snapshot: str = Form(""),
    enterprise_mode: bool = Form(False),
    incremental: bool = Form(True),
    force_rebuild: bool = Form(False),
):
    """Ejecutar pipeline de RAG (incremental por defecto)"""
    try:
        import subprocess

        corpus_dir = Path(__file__).parent.parent / "all-Branches" / "corpus"
        rag_cli = corpus_dir / "rag_cli.py"

        if not rag_cli.exists():
            raise HTTPException(status_code=404, detail="rag_cli.py no encontrado")

        python_exe = sys.executable

        # SECURITY: Sanitize snapshot name if provided
        safe_snapshot = ""
        if snapshot:
            try:
                safe_snapshot = sanitize_snapshot_name(snapshot)
            except SecurityError as se:
                raise HTTPException(
                    status_code=400, detail=f"Invalid snapshot name: {se}"
                )

        # Construir comando base
        if enterprise_mode:
            cmd = (
                [python_exe, str(rag_cli), "pipeline_ent", safe_snapshot]
                if safe_snapshot
                else [python_exe, str(rag_cli), "pipeline_ent"]
            )
        else:
            cmd = (
                [python_exe, str(rag_cli), "pipeline", safe_snapshot]
                if safe_snapshot
                else [python_exe, str(rag_cli), "pipeline"]
            )

        # Agregar flags de modo incremental
        if force_rebuild:
            cmd.append("--force")
        elif not incremental:
            cmd.append("--full")
        # Por defecto es --incremental (no hace falta agregarlo)

        # Ejecutar en background (async)
        process = subprocess.Popen(
            cmd,
            cwd=str(corpus_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        mode_desc = "enterprise" if enterprise_mode else "standard"
        if force_rebuild:
            mode_desc += " (force rebuild)"
        elif incremental:
            mode_desc += " (incremental)"
        else:
            mode_desc += " (full)"

        return {
            "success": True,
            "message": "Pipeline iniciado en background",
            "pid": process.pid,
            "mode": mode_desc,
            "snapshot": snapshot or "auto",
            "incremental": incremental and not force_rebuild,
        }
    except Exception as e:
        logger.error(f"Error iniciando pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/corpus/pipeline/status")
async def corpus_pipeline_status():
    """Obtener estado actual del pipeline en ejecución"""
    try:
        corpus_dir = Path(__file__).parent.parent / "all-Branches" / "corpus"
        base_path = corpus_dir / "corpus" / "universal"

        # Buscar snapshot latest
        latest_ptr = base_path / "latest.ptr"
        if latest_ptr.exists():
            snapshot_name = latest_ptr.read_text(encoding="utf-8").strip()
            status_file = base_path / snapshot_name / ".pipeline_status.json"
        else:
            # Buscar snapshots y obtener el más reciente
            snapshots = sorted(
                [d for d in base_path.iterdir() if d.is_dir()], reverse=True
            )
            if not snapshots:
                return {"running": False, "step": "none", "status": "no_snapshot"}
            status_file = snapshots[0] / ".pipeline_status.json"

        if not status_file.exists():
            return {"running": False, "step": "none", "status": "not_started"}

        # Leer estado
        import json

        status_data = json.loads(status_file.read_text(encoding="utf-8"))

        # Determinar si está corriendo
        running = status_data.get("status") == "running"
        completed = status_data.get("step") == "completed"
        failed = status_data.get("status") == "failed"

        return {
            "running": running,
            "completed": completed,
            "failed": failed,
            "step": status_data.get("step", "unknown"),
            "status": status_data.get("status", "unknown"),
            "timestamp": status_data.get("timestamp", ""),
            "error": status_data.get("error", ""),
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado del pipeline: {e}")
        return {"running": False, "step": "error", "status": "error", "error": str(e)}


@app.get("/api/corpus/search")
async def corpus_advanced_search(
    q: str = Query(..., min_length=2),
    mode: str = Query("hybrid", regex="^(hybrid|bm25|dense|expanded|graph)$"),
    top_k: int = Query(10, ge=1, le=50),
    enable_rerank: bool = Query(True),
    enable_crag: bool = Query(False),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
):
    """Búsqueda avanzada con múltiples modos y características"""
    try:
        # Usar unified_search del corpus
        if not UNIFIED_SEARCH_AVAILABLE:
            raise HTTPException(status_code=503, detail="Unified search no disponible")

        corpus_dir = Path(__file__).parent.parent / "all-Branches" / "corpus"
        base_path = corpus_dir / "corpus" / "universal"

        # Buscar snapshot latest
        latest_ptr = base_path / "latest.ptr"
        if latest_ptr.exists():
            snapshot_name = latest_ptr.read_text(encoding="utf-8").strip()
            snapshot_path = base_path / snapshot_name
        else:
            # Buscar snapshot más reciente
            snapshots = sorted(
                [d for d in base_path.iterdir() if d.is_dir()], reverse=True
            )
            if not snapshots:
                raise HTTPException(
                    status_code=404, detail="No hay snapshots disponibles"
                )
            snapshot_path = snapshots[0]

        # Ejecutar búsqueda
        results = unified_search(
            branch="universal", base=snapshot_path, query=q, top_k=top_k, mode=mode
        )

        # Aplicar filtro de score mínimo
        if min_score > 0:
            results = [r for r in results if r.get("score", 0) >= min_score]

        return {
            "success": True,
            "query": q,
            "mode": mode,
            "results": results[:top_k],
            "total": len(results),
            "rerank_enabled": enable_rerank,
            "crag_enabled": enable_crag,
            "snapshot": snapshot_path.name,
        }
    except Exception as e:
        logger.error(f"Error en búsqueda avanzada: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/corpus/info")
async def corpus_get_info(snapshot: str = Query("")):
    """Obtener información del corpus"""
    try:
        corpus_dir = Path(__file__).parent.parent / "all-Branches" / "corpus"
        base_path = corpus_dir / "corpus" / "universal"

        # SECURITY: Sanitize snapshot name to prevent path traversal
        if snapshot:
            try:
                safe_snapshot = sanitize_snapshot_name(snapshot)
                snapshot_path = base_path / safe_snapshot
            except SecurityError as se:
                raise HTTPException(
                    status_code=400, detail=f"Invalid snapshot name: {se}"
                )
        else:
            latest_ptr = base_path / "latest.ptr"
            if latest_ptr.exists():
                # Leer con UTF-8-SIG para manejar BOM automáticamente
                snapshot_name = latest_ptr.read_text(encoding="utf-8-sig").strip()
                # SECURITY: Validate snapshot name from file
                try:
                    safe_snapshot_name = sanitize_snapshot_name(snapshot_name)
                    snapshot_path = base_path / safe_snapshot_name
                except SecurityError:
                    logger.error(
                        f"Invalid snapshot name in latest.ptr: {snapshot_name}"
                    )
                    return {
                        "success": False,
                        "message": "Invalid snapshot configuration",
                    }
            else:
                snapshots = sorted(
                    [d for d in base_path.iterdir() if d.is_dir()], reverse=True
                )
                if not snapshots:
                    return {"success": False, "message": "No hay snapshots"}
                snapshot_path = snapshots[0]

        if not snapshot_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Snapshot no encontrado: {snapshot}"
            )

        # Contar archivos
        raw_count = (
            len(list((snapshot_path / "raw").glob("*")))
            if (snapshot_path / "raw").exists()
            else 0
        )
        cleaned_count = (
            len(list((snapshot_path / "cleaned").glob("*")))
            if (snapshot_path / "cleaned").exists()
            else 0
        )
        chunks_count = (
            len(list((snapshot_path / "chunks").glob("*.jsonl")))
            if (snapshot_path / "chunks").exists()
            else 0
        )

        # Contar embeddings (buscar tanto .npy como .parquet)
        embeddings_count = 0
        if (snapshot_path / "embeddings").exists():
            emb_files = list((snapshot_path / "embeddings").glob("*.npy")) + list(
                (snapshot_path / "embeddings").glob("*.parquet")
            )
            embeddings_count = len(emb_files)

        # Verificar índices (buscar archivos reales)
        index_dir = snapshot_path / "index"
        indices = {}
        if index_dir.exists():
            indices = {
                "hnsw": (index_dir / "hnsw.idx").exists()
                or (index_dir / "hnsw.bin").exists(),
                "faiss": (index_dir / "faiss.index").exists(),
                "bm25": (index_dir / "bm25").exists(),
                "raptor": (index_dir / "raptor").exists(),
                "graph": (index_dir / "graph").exists(),
            }

        return {
            "success": True,
            "snapshot": snapshot_path.name,
            "stats": {
                "raw_documents": raw_count,
                "cleaned_documents": cleaned_count,
                "chunks": chunks_count,
                "embeddings": embeddings_count,
            },
            "indices": indices,
            "path": str(snapshot_path),
        }
    except Exception as e:
        logger.error(f"Error obteniendo info del corpus: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/corpus/doctor")
async def corpus_run_doctor():
    """Ejecutar diagnóstico del sistema RAG"""
    try:
        import subprocess

        corpus_dir = Path(__file__).parent.parent / "all-Branches" / "corpus"
        rag_cli = corpus_dir / "rag_cli.py"

        if not rag_cli.exists():
            raise HTTPException(status_code=404, detail="rag_cli.py no encontrado")

        python_exe = sys.executable

        result = subprocess.run(
            [python_exe, str(rag_cli), "doctor"],
            cwd=str(corpus_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )

        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr if result.returncode != 0 else None,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Timeout en doctor (>60s)")
    except Exception as e:
        logger.error(f"Error ejecutando doctor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/corpus/snapshots")
async def corpus_list_snapshots():
    """Listar todos los snapshots disponibles"""
    try:
        corpus_dir = Path(__file__).parent.parent / "all-Branches" / "corpus"
        base_path = corpus_dir / "corpus" / "universal"

        if not base_path.exists():
            return {"success": True, "snapshots": []}

        # Leer latest pointer
        latest_ptr = base_path / "latest.ptr"
        latest_name = ""
        if latest_ptr.exists():
            latest_name = latest_ptr.read_text(encoding="utf-8").strip()

        # Listar snapshots
        snapshots = []
        for item in sorted(base_path.iterdir(), reverse=True):
            if item.is_dir() and item.name not in ["latest", ".git"]:
                # Obtener info básica
                raw_count = (
                    len(list((item / "raw").glob("*")))
                    if (item / "raw").exists()
                    else 0
                )
                chunks_count = (
                    len(list((item / "chunks").glob("*.jsonl")))
                    if (item / "chunks").exists()
                    else 0
                )

                snapshots.append(
                    {
                        "name": item.name,
                        "is_latest": item.name == latest_name,
                        "created": datetime.fromtimestamp(
                            item.stat().st_ctime
                        ).isoformat(),
                        "documents": raw_count,
                        "chunks": chunks_count,
                        "path": str(item),
                    }
                )

        return {
            "success": True,
            "snapshots": snapshots,
            "latest": latest_name,
            "total": len(snapshots),
        }
    except Exception as e:
        logger.error(f"Error listando snapshots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/corpus/snapshot/create")
async def corpus_create_snapshot(name: str = Form("")):
    """Crear nuevo snapshot"""
    try:
        import subprocess

        corpus_dir = Path(__file__).parent.parent / "all-Branches" / "corpus"
        base_path = corpus_dir / "corpus" / "universal"
        base_path.mkdir(parents=True, exist_ok=True)

        # Generar nombre si no se proporciona
        if not name:
            safe_name = f"snapshot_{int(time.time())}"
        else:
            # SECURITY: Sanitize snapshot name
            try:
                safe_name = sanitize_snapshot_name(name)
            except SecurityError as se:
                raise HTTPException(
                    status_code=400, detail=f"Invalid snapshot name: {se}"
                )

        snapshot_path = base_path / safe_name

        # Crear estructura
        for subdir in [
            "raw",
            "cleaned",
            "chunks",
            "embeddings",
            "index",
            "datasets",
            "eval",
        ]:
            (snapshot_path / subdir).mkdir(parents=True, exist_ok=True)

        # Actualizar pointer
        latest_ptr = base_path / "latest.ptr"
        latest_ptr.write_text(safe_name, encoding="utf-8")

        return {
            "success": True,
            "message": "Snapshot creado",
            "snapshot": safe_name,
            "path": str(snapshot_path),
        }
    except Exception as e:
        logger.error(f"Error creando snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/corpus/snapshot/activate")
async def corpus_activate_snapshot(name: str = Form(...)):
    """Activar snapshot existente"""
    try:
        # SECURITY: Sanitize snapshot name
        try:
            safe_name = sanitize_snapshot_name(name)
        except SecurityError as se:
            raise HTTPException(status_code=400, detail=f"Invalid snapshot name: {se}")

        corpus_dir = Path(__file__).parent.parent / "all-Branches" / "corpus"
        base_path = corpus_dir / "corpus" / "universal"

        snapshot_path = base_path / safe_name
        if not snapshot_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Snapshot no encontrado: {name}"
            )

        # Actualizar pointer
        latest_ptr = base_path / "latest.ptr"
        latest_ptr.write_text(safe_name, encoding="utf-8")

        return {"success": True, "message": "Snapshot activado", "snapshot": safe_name}
    except Exception as e:
        logger.error(f"Error activando snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/corpus/metrics")
async def corpus_get_metrics():
    """Obtener métricas del sistema Corpus"""
    try:
        metrics = {
            "rag_service_available": RAG_AVAILABLE,
            "unified_search_available": UNIFIED_SEARCH_AVAILABLE,
            "timestamp": datetime.now().isoformat(),
        }

        # Métricas de RAG service si está disponible
        if RAG_AVAILABLE:
            try:
                rag_service = get_rag_service()
                corpus_stats = rag_service.get_corpus_stats()
                metrics["corpus"] = corpus_stats
            except Exception as e:
                logger.warning(f"Error obteniendo stats de corpus: {e}")

        # Métricas de archivos
        rag_dir = Path("data/rag_documents")
        if rag_dir.exists():
            metrics["local_files"] = {
                "txt": len(list(rag_dir.glob("*.txt"))),
                "pdf": len(list(rag_dir.glob("*.pdf"))),
                "md": len(list(rag_dir.glob("*.md"))),
            }

        return {"success": True, "metrics": metrics}
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/corpus/feature/toggle")
async def corpus_toggle_feature(
    feature: str = Form(..., regex="^(rerank|crag|cache)$"), enabled: bool = Form(...)
):
    """Activar/desactivar características del corpus"""
    try:
        corpus_dir = Path(__file__).parent.parent / "all-Branches" / "corpus"
        config_file = corpus_dir / "config" / "universal.yaml"

        if not config_file.exists():
            raise HTTPException(status_code=404, detail="Config no encontrado")

        import yaml

        # Leer config
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Modificar según feature
        if feature == "rerank":
            config.setdefault("retrieval", {}).setdefault("rerank", {})[
                "enabled"
            ] = enabled
        elif feature == "crag":
            config.setdefault("retrieval", {}).setdefault("crag", {})[
                "enabled"
            ] = enabled
        elif feature == "cache":
            config.setdefault("cache", {})["enabled"] = enabled

        # Guardar config
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)

        return {
            "success": True,
            "message": f"Feature '{feature}' {'activada' if enabled else 'desactivada'}",
            "feature": feature,
            "enabled": enabled,
        }
    except Exception as e:
        logger.error(f"Error toggleando feature: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "dashboard_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
