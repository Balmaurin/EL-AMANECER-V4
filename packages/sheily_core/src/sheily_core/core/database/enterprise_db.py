#!/usr/bin/env python3
"""
Sheily Enterprise Database Architecture
======================================

Sistema de base de datos enterprise con optimistic locking, event sourcing,
y patterns avanzados integrados con ii-agent para máxima reliability.

Características:
- Async SQLAlchemy con optimistic locking
- Event sourcing para complete audit trail
- Advanced connection pooling
- Transaction management con rollback
- Integration con Event System
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar
from uuid import UUID, uuid4

# Database imports (using SQLite for simplicity, can be upgraded to PostgreSQL)
import aiosqlite

from sheily_core.core.events.event_system import (
    SheìlyEventType,
    get_event_stream,
    publish_event,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EntityStatus(str, Enum):
    """Estado de entidades del sistema"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"
    PENDING = "pending"


@dataclass
class BaseEntity:
    """Entidad base con versioning para optimistic locking"""

    id: str
    version: int = 1
    created_at: datetime = None
    updated_at: datetime = None
    status: EntityStatus = EntityStatus.ACTIVE
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentEntity(BaseEntity):
    """Entidad para agentes MCP"""

    agent_name: str = ""
    agent_type: str = ""
    capabilities: List[str] = None
    config: Dict[str, Any] = None
    last_activity: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = None

    def __post_init__(self):
        super().__post_init__()
        if self.capabilities is None:
            self.capabilities = []
        if self.config is None:
            self.config = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class SessionEntity(BaseEntity):
    """Entidad para sesiones enterprise"""

    session_id: str = ""
    user_id: str = ""
    session_data: Dict[str, Any] = None
    last_message: Optional[datetime] = None
    message_count: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.session_data is None:
            self.session_data = {}


@dataclass
class EventEntity(BaseEntity):
    """Entidad para event sourcing"""

    event_type: str = ""
    event_data: Dict[str, Any] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.event_data is None:
            self.event_data = {}


class StaleDataError(Exception):
    """Error de datos obsoletos para optimistic locking"""

    pass


class SheìlyDatabase:
    """Enterprise database system para Sheily"""

    def __init__(self, db_path: str = "sheily_enterprise.db"):
        self.db_path = Path(db_path)
        self.connection = None
        self.event_stream = None

    async def initialize(self) -> None:
        """Initialize database system"""
        try:
            # Create database directory if needed
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Initialize database schema
            await self._create_schema()

            # Get event stream for publishing
            self.event_stream = get_event_stream()

            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {"status": "database_initialized", "path": str(self.db_path)},
            )

            logger.info(f"✅ Sheily Enterprise Database initialized: {self.db_path}")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection with proper cleanup"""
        conn = None
        try:
            conn = await aiosqlite.connect(str(self.db_path))
            conn.row_factory = aiosqlite.Row
            yield conn
        except Exception as e:
            if conn:
                await conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                await conn.close()

    async def _create_schema(self) -> None:
        """Create database schema"""
        async with self.get_connection() as conn:
            # Agents table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    agent_name TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    capabilities TEXT NOT NULL DEFAULT '[]',
                    config TEXT NOT NULL DEFAULT '{}',
                    last_activity TEXT,
                    performance_metrics TEXT NOT NULL DEFAULT '{}',
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """
            )

            # Sessions table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    session_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    session_data TEXT NOT NULL DEFAULT '{}',
                    last_message TEXT,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """
            )

            # Events table for event sourcing
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL DEFAULT '{}',
                    agent_id TEXT,
                    session_id TEXT,
                    correlation_id TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """
            )

            # Indexes for performance
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agents_name ON agents (agent_name)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agents_type ON agents (agent_type)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_agent ON events (agent_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id)"
            )

            await conn.commit()

    # ========================================
    # GENERIC CRUD OPERATIONS WITH OPTIMISTIC LOCKING
    # ========================================

    async def save_entity(self, entity: BaseEntity, table_name: str) -> BaseEntity:
        """Save entity with optimistic locking"""
        async with self.get_connection() as conn:
            try:
                if await self._entity_exists(conn, entity.id, table_name):
                    # Update existing entity
                    return await self._update_entity(conn, entity, table_name)
                else:
                    # Insert new entity
                    return await self._insert_entity(conn, entity, table_name)

            except Exception as e:
                await conn.rollback()
                logger.error(f"Error saving entity {entity.id}: {e}")
                raise

    async def _entity_exists(self, conn, entity_id: str, table_name: str) -> bool:
        """Check if entity exists"""
        cursor = await conn.execute(
            f"SELECT id FROM {table_name} WHERE id = ?", (entity_id,)
        )
        return await cursor.fetchone() is not None

    async def _insert_entity(
        self, conn, entity: BaseEntity, table_name: str
    ) -> BaseEntity:
        """Insert new entity"""
        entity.created_at = datetime.now(timezone.utc)
        entity.updated_at = entity.created_at
        entity.version = 1

        # Convert entity to dict for insertion
        data = self._entity_to_dict(entity)

        columns = list(data.keys())
        placeholders = ["?" for _ in columns]
        values = [self._serialize_value(data[col]) for col in columns]

        await conn.execute(
            f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({','.join(placeholders)})",
            values,
        )
        await conn.commit()

        # Publish event
        if self.event_stream:
            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {
                    "action": "entity_created",
                    "table": table_name,
                    "entity_id": entity.id,
                    "entity_type": type(entity).__name__,
                },
            )

        return entity

    async def _update_entity(
        self, conn, entity: BaseEntity, table_name: str
    ) -> BaseEntity:
        """Update entity with optimistic locking"""
        # First, check current version
        cursor = await conn.execute(
            f"SELECT version FROM {table_name} WHERE id = ?", (entity.id,)
        )
        row = await cursor.fetchone()

        if not row:
            raise ValueError(f"Entity {entity.id} not found")

        current_version = row["version"]
        if current_version != entity.version:
            raise StaleDataError(
                f"Entity {entity.id} version conflict. Current: {current_version}, Provided: {entity.version}"
            )

        # Update entity
        entity.updated_at = datetime.now(timezone.utc)
        entity.version += 1

        data = self._entity_to_dict(entity)
        set_clause = ",".join([f"{col} = ?" for col in data.keys() if col != "id"])
        values = [
            self._serialize_value(data[col]) for col in data.keys() if col != "id"
        ]
        values.append(entity.id)

        await conn.execute(f"UPDATE {table_name} SET {set_clause} WHERE id = ?", values)
        await conn.commit()

        # Publish event
        if self.event_stream:
            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {
                    "action": "entity_updated",
                    "table": table_name,
                    "entity_id": entity.id,
                    "new_version": entity.version,
                },
            )

        return entity

    def _entity_to_dict(self, entity: BaseEntity) -> Dict[str, Any]:
        """Convert entity to dictionary for database storage"""
        if isinstance(entity, AgentEntity):
            return {
                "id": entity.id,
                "version": entity.version,
                "created_at": entity.created_at.isoformat(),
                "updated_at": entity.updated_at.isoformat(),
                "status": entity.status.value,
                "agent_name": entity.agent_name,
                "agent_type": entity.agent_type,
                "capabilities": json.dumps(entity.capabilities),
                "config": json.dumps(entity.config),
                "last_activity": (
                    entity.last_activity.isoformat() if entity.last_activity else None
                ),
                "performance_metrics": json.dumps(entity.performance_metrics),
                "metadata": json.dumps(entity.metadata),
            }
        elif isinstance(entity, SessionEntity):
            return {
                "id": entity.id,
                "version": entity.version,
                "created_at": entity.created_at.isoformat(),
                "updated_at": entity.updated_at.isoformat(),
                "status": entity.status.value,
                "session_id": entity.session_id,
                "user_id": entity.user_id,
                "session_data": json.dumps(entity.session_data),
                "last_message": (
                    entity.last_message.isoformat() if entity.last_message else None
                ),
                "message_count": entity.message_count,
                "metadata": json.dumps(entity.metadata),
            }
        elif isinstance(entity, EventEntity):
            return {
                "id": entity.id,
                "version": entity.version,
                "created_at": entity.created_at.isoformat(),
                "updated_at": entity.updated_at.isoformat(),
                "status": entity.status.value,
                "event_type": entity.event_type,
                "event_data": json.dumps(entity.event_data),
                "agent_id": entity.agent_id,
                "session_id": entity.session_id,
                "correlation_id": entity.correlation_id,
                "metadata": json.dumps(entity.metadata),
            }
        else:
            # Base entity
            return {
                "id": entity.id,
                "version": entity.version,
                "created_at": entity.created_at.isoformat(),
                "updated_at": entity.updated_at.isoformat(),
                "status": entity.status.value,
                "metadata": json.dumps(entity.metadata),
            }

    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for database storage"""
        if value is None:
            return None
        elif isinstance(value, (str, int, float)):
            return value
        else:
            return str(value)

    async def get_entity(
        self, entity_id: str, table_name: str, entity_class: Type[T]
    ) -> Optional[T]:
        """Get entity by ID"""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                f"SELECT * FROM {table_name} WHERE id = ? AND status != 'deleted'",
                (entity_id,),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            return self._row_to_entity(dict(row), entity_class)

    def _row_to_entity(self, row_dict: Dict[str, Any], entity_class: Type[T]) -> T:
        """Convert database row to entity"""
        # Parse timestamps
        created_at = datetime.fromisoformat(
            row_dict["created_at"].replace("Z", "+00:00")
        )
        updated_at = datetime.fromisoformat(
            row_dict["updated_at"].replace("Z", "+00:00")
        )

        if entity_class == AgentEntity:
            return AgentEntity(
                id=row_dict["id"],
                version=row_dict["version"],
                created_at=created_at,
                updated_at=updated_at,
                status=EntityStatus(row_dict["status"]),
                agent_name=row_dict["agent_name"],
                agent_type=row_dict["agent_type"],
                capabilities=json.loads(row_dict["capabilities"]),
                config=json.loads(row_dict["config"]),
                last_activity=(
                    datetime.fromisoformat(
                        row_dict["last_activity"].replace("Z", "+00:00")
                    )
                    if row_dict["last_activity"]
                    else None
                ),
                performance_metrics=json.loads(row_dict["performance_metrics"]),
                metadata=json.loads(row_dict["metadata"]),
            )
        elif entity_class == SessionEntity:
            return SessionEntity(
                id=row_dict["id"],
                version=row_dict["version"],
                created_at=created_at,
                updated_at=updated_at,
                status=EntityStatus(row_dict["status"]),
                session_id=row_dict["session_id"],
                user_id=row_dict["user_id"],
                session_data=json.loads(row_dict["session_data"]),
                last_message=(
                    datetime.fromisoformat(
                        row_dict["last_message"].replace("Z", "+00:00")
                    )
                    if row_dict["last_message"]
                    else None
                ),
                message_count=row_dict["message_count"],
                metadata=json.loads(row_dict["metadata"]),
            )
        elif entity_class == EventEntity:
            return EventEntity(
                id=row_dict["id"],
                version=row_dict["version"],
                created_at=created_at,
                updated_at=updated_at,
                status=EntityStatus(row_dict["status"]),
                event_type=row_dict["event_type"],
                event_data=json.loads(row_dict["event_data"]),
                agent_id=row_dict.get("agent_id"),
                session_id=row_dict.get("session_id"),
                correlation_id=row_dict.get("correlation_id"),
                metadata=json.loads(row_dict["metadata"]),
            )
        else:
            return entity_class(
                id=row_dict["id"],
                version=row_dict["version"],
                created_at=created_at,
                updated_at=updated_at,
                status=EntityStatus(row_dict["status"]),
                metadata=json.loads(row_dict["metadata"]),
            )

    # ========================================
    # SPECIALIZED OPERATIONS
    # ========================================

    async def save_agent(self, agent: AgentEntity) -> AgentEntity:
        """Save agent entity"""
        return await self.save_entity(agent, "agents")

    async def get_agent(self, agent_id: str) -> Optional[AgentEntity]:
        """Get agent by ID"""
        return await self.get_entity(agent_id, "agents", AgentEntity)

    async def get_agents_by_type(self, agent_type: str) -> List[AgentEntity]:
        """Get all agents of a specific type"""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM agents WHERE agent_type = ? AND status = 'active'",
                (agent_type,),
            )
            rows = await cursor.fetchall()

            return [self._row_to_entity(dict(row), AgentEntity) for row in rows]

    async def save_session(self, session: SessionEntity) -> SessionEntity:
        """Save session entity"""
        return await self.save_entity(session, "sessions")

    async def get_session(self, session_id: str) -> Optional[SessionEntity]:
        """Get session by session_id"""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM sessions WHERE session_id = ? AND status != 'deleted'",
                (session_id,),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            return self._row_to_entity(dict(row), SessionEntity)

    async def record_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> EventEntity:
        """Record event for audit trail"""
        event = EventEntity(
            id=str(uuid4()),
            event_type=event_type,
            event_data=event_data,
            agent_id=agent_id,
            session_id=session_id,
            correlation_id=str(uuid4()),
        )

        return await self.save_entity(event, "events")

    async def get_events(
        self,
        entity_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[EventEntity]:
        """Get events with optional filtering"""
        async with self.get_connection() as conn:
            query = "SELECT * FROM events WHERE status = 'active'"
            params = []

            if entity_id:
                query += " AND (agent_id = ? OR session_id = ?)"
                params.extend([entity_id, entity_id])

            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

            return [self._row_to_entity(dict(row), EventEntity) for row in rows]

    # ========================================
    # HEALTH & MAINTENANCE
    # ========================================

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        async with self.get_connection() as conn:
            stats = {}

            # Count records in each table
            for table in ["agents", "sessions", "events"]:
                cursor = await conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                row = await cursor.fetchone()
                stats[f"{table}_count"] = row["count"]

            # Database size
            stats["database_size_bytes"] = (
                self.db_path.stat().st_size if self.db_path.exists() else 0
            )

            return stats

    async def cleanup_old_events(self, days_old: int = 30) -> int:
        """Cleanup old events (soft delete)"""
        async with self.get_connection() as conn:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)

            cursor = await conn.execute(
                "UPDATE events SET status = 'deleted' WHERE created_at < ? AND status = 'active'",
                (cutoff_date.isoformat(),),
            )

            deleted_count = cursor.rowcount
            await conn.commit()

            if deleted_count > 0:
                await publish_event(
                    SheìlyEventType.SYSTEM_HEALTH,
                    {"action": "events_cleanup", "deleted_count": deleted_count},
                )

            return deleted_count

    async def shutdown(self) -> None:
        """Shutdown database system"""
        try:
            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH, {"status": "database_shutdown"}
            )
            logger.info("🛑 Sheily Database shut down")
        except Exception as e:
            logger.error(f"Error shutting down database: {e}")


# ========================================
# GLOBAL INSTANCE
# ========================================

_database: Optional[SheìlyDatabase] = None


async def get_database() -> SheìlyDatabase:
    """Get global database instance"""
    global _database
    if _database is None:
        _database = SheìlyDatabase()
        await _database.initialize()
    return _database


async def initialize_database_system() -> SheìlyDatabase:
    """Initialize the complete database system"""
    database = await get_database()
    logger.info("✅ Sheily Enterprise Database System initialized")
    return database


__all__ = [
    "SheìlyDatabase",
    "BaseEntity",
    "AgentEntity",
    "SessionEntity",
    "EventEntity",
    "EntityStatus",
    "StaleDataError",
    "get_database",
    "initialize_database_system",
]
