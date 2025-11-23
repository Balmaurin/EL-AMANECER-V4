"""
Vault API - Caja fuerte para tokens y datos sensibles
Extraído de tools/dashboard_backend.py
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from apps.backend.src.security import SecurityError, validate_command_args, validate_timeout


class VaultAuth(BaseModel):
    pin: str


class VaultItem(BaseModel):
    id: int
    item_type: str
    data: dict
    timestamp: str


class Database:
    """Helper de base de datos para vault"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Inicializar base de datos de vault"""
        with self.get_connection() as conn:
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


# Inicializar base de datos con ruta correcta
data_dir = Path(__file__).parent.parent.parent / "data"
data_dir.mkdir(exist_ok=True)
db = Database(data_dir / "vault.db")

router = APIRouter()


@router.post("/auth", response_model=Dict)
async def vault_auth(auth: VaultAuth):
    """Autenticar caja fuerte"""
    try:
        correct_pin = db.get_vault_pin(1)
        if auth.pin == correct_pin:
            items = db.get_vault_items(1)
            return {"authenticated": True, "items": items}
        else:
            raise HTTPException(status_code=401, detail="PIN incorrecto")
    except SecurityError as se:
        raise HTTPException(status_code=403, detail=f"Security error: {se}")


@router.get("/items", response_model=List[VaultItem])
async def get_vault_items():
    """Obtener todos los items de la caja fuerte"""
    try:
        items = db.get_vault_items(1)
        return [
            VaultItem(
                id=item["id"],
                item_type=item["type"],
                data=item["data"],
                timestamp=item["timestamp"],
            )
            for item in items
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo items: {e}")


@router.post("/items")
async def add_vault_item(item_type: str, data: Dict):
    """Añadir item a la caja fuerte"""
    try:
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO vault_items (user_id, item_type, data) VALUES (?, ?, ?)",
                (1, item_type, json.dumps(data)),
            )
            conn.commit()

            return {"success": True, "message": "Item añadido a la caja fuerte"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error añadiendo item: {e}")


@router.delete("/items/{item_id}")
async def delete_vault_item(item_id: int):
    """Eliminar item de la caja fuerte"""
    try:
        with db.get_connection() as conn:
            result = conn.execute(
                "DELETE FROM vault_items WHERE id = ? AND user_id = ?", (item_id, 1)
            )
            conn.commit()

            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Item no encontrado")

            return {"success": True, "message": f"Item {item_id} eliminado"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando item: {e}")
