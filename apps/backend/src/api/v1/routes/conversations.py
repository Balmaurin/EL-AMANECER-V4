"""
Conversations API - Gestión de conversaciones de chat
Extraído de backend/chat_server.py y sistemas de conversación
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ConversationCreate(BaseModel):
    user_id: str
    title: str = "Nueva Conversación"
    system_prompt: Optional[str] = None

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    is_active: Optional[bool] = None

class MessageCreate(BaseModel):
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    metadata: Optional[Dict] = None

router = APIRouter()

# Database path (same as main_api.py)
DB_PATH = "gamified_database.db"

# Helper function to get database connection
def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)

# Helper function to init conversations table
def init_conversations_table():
    """Initialize conversations table if it doesn't exist"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                system_prompt TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                message_count INTEGER DEFAULT 0
            )
        ''')

        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')

        conn.commit()
        conn.close()

    except Exception as e:
        logger.error(f"Error initializing conversations table: {e}")
        raise

@router.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    init_conversations_table()

@router.post("/conversations")
async def create_conversation(conv: ConversationCreate):
    """Crear una nueva conversación"""
    try:
        conv_id = f"conv_{int(datetime.now().timestamp() * 1000)}"

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO conversations (id, user_id, title, system_prompt, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            conv_id,
            conv.user_id,
            conv.title,
            conv.system_prompt,
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        return {
            "conversation_id": conv_id,
            "user_id": conv.user_id,
            "title": conv.title,
            "created_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@router.get("/conversations")
async def list_conversations(
    user_id: str = Query(..., description="User ID to filter conversations"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of conversations"),
    offset: int = Query(0, ge=0, description="Number of conversations to skip")
):
    """Listar conversaciones de un usuario"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, title, created_at, updated_at, message_count, is_active
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        ''', (user_id, limit, offset))

        rows = cursor.fetchall()
        conn.close()

        conversations = []
        for row in rows:
            conversations.append({
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": row[4],
                "is_active": bool(row[5])
            })

        return {"conversations": conversations, "count": len(conversations)}

    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Obtener detalles de una conversación específica"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Get conversation info
        cursor.execute('''
            SELECT id, user_id, title, system_prompt, created_at, updated_at, message_count, is_active
            FROM conversations
            WHERE id = ?
        ''', (conversation_id,))

        conv_row = cursor.fetchone()
        if not conv_row:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get messages
        cursor.execute('''
            SELECT role, content, metadata, created_at
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
        ''', (conversation_id,))

        message_rows = cursor.fetchall()
        conn.close()

        messages = []
        for row in message_rows:
            messages.append({
                "role": row[0],
                "content": row[1],
                "metadata": row[2],
                "created_at": row[3]
            })

        return {
            "conversation": {
                "id": conv_row[0],
                "user_id": conv_row[1],
                "title": conv_row[2],
                "system_prompt": conv_row[3],
                "created_at": conv_row[4],
                "updated_at": conv_row[5],
                "message_count": conv_row[6],
                "is_active": bool(conv_row[7])
            },
            "messages": messages
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")

@router.put("/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, update: ConversationUpdate):
    """Actualizar una conversación"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Build update query dynamically
        update_fields = []
        values = []

        if update.title is not None:
            update_fields.append("title = ?")
            values.append(update.title)

        if update.is_active is not None:
            update_fields.append("is_active = ?")
            values.append(1 if update.is_active else 0)

        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_fields.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(conversation_id)

        query = f'''
            UPDATE conversations
            SET {", ".join(update_fields)}
            WHERE id = ?
        '''

        cursor.execute(query, values)
        conn.commit()
        conn.close()

        return {"updated": conversation_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update conversation: {str(e)}")

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Eliminar una conversación y todos sus mensajes"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Delete messages first (foreign key constraint)
        cursor.execute('DELETE FROM conversation_messages WHERE conversation_id = ?', (conversation_id,))

        # Delete conversation
        cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"deleted": conversation_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")

@router.post("/messages")
async def create_message(message: MessageCreate):
    """Añadir un mensaje a una conversación"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Verify conversation exists
        cursor.execute('SELECT user_id FROM conversations WHERE id = ?', (message.conversation_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Insert message
        cursor.execute('''
            INSERT INTO conversation_messages (conversation_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            message.conversation_id,
            message.role,
            message.content,
            str(message.metadata) if message.metadata else None,
            datetime.now().isoformat()
        ))

        # Update conversation message count and updated_at
        cursor.execute('''
            UPDATE conversations
            SET message_count = message_count + 1, updated_at = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), message.conversation_id))

        conn.commit()
        conn.close()

        return {"message_id": cursor.lastrowid, "conversation_id": message.conversation_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create message: {str(e)}")

@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of messages"),
    offset: int = Query(0, ge=0, description="Number of messages to skip")
):
    """Obtener mensajes de una conversación"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Verify conversation exists
        cursor.execute('SELECT COUNT(*) FROM conversations WHERE id = ?', (conversation_id,))
        if cursor.fetchone()[0] == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get messages
        cursor.execute('''
            SELECT role, content, metadata, created_at
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            LIMIT ? OFFSET ?
        ''', (conversation_id, limit, offset))

        rows = cursor.fetchall()
        conn.close()

        messages = []
        for row in rows:
            messages.append({
                "role": row[0],
                "content": row[1],
                "metadata": row[2],
                "created_at": row[3]
            })

        return {"messages": messages, "count": len(messages)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@router.get("/stats")
async def get_conversation_stats(user_id: Optional[str] = Query(None, description="User ID to filter stats")):
    """Obtener estadísticas de conversaciones"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        if user_id:
            cursor.execute('''
                SELECT COUNT(*) as total_conversations,
                       SUM(message_count) as total_messages,
                       AVG(message_count) as avg_messages_per_conv,
                       COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_conversations
                FROM conversations
                WHERE user_id = ?
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT COUNT(*) as total_conversations,
                       SUM(message_count) as total_messages,
                       AVG(message_count) as avg_messages_per_conv,
                       COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_conversations
                FROM conversations
            ''')

        row = cursor.fetchone()
        conn.close()

        return {
            "total_conversations": row[0] or 0,
            "total_messages": row[1] or 0,
            "avg_messages_per_conversation": round(row[2] or 0, 2),
            "active_conversations": row[3] or 0
        }

    except Exception as e:
        logger.error(f"Error getting conversation stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
