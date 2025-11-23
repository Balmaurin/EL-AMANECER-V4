"""
Sheily MCP Enterprise - Database Controller
Gestión completa de PostgreSQL y esquemas de datos

Controla:
- Conexiones de base de datos
- Migraciones de esquemas
- Gestión de datos
- Health checks
- Backup/restoration
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg

logger = logging.getLogger(__name__)


class DatabaseController:
    """Controlador completo de base de datos PostgreSQL"""

    def __init__(self, root_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.root_dir = Path(root_dir)
        self.backend_dir = root_dir / "backend"
        self.migrations_dir = self.backend_dir / "migrations"

        # Database configuration
        self.db_config = config or {
            "host": "localhost",
            "port": 5433,  # Development default
            "database": "sheily_ai_dev",
            "user": "sheily_dev",
            "password": "dev_password",
            "schema": "public",
            "connection_pool": {"min_size": 5, "max_size": 20},
        }

        # Connection pools
        self.pool: Optional[asyncpg.Pool] = None
        self.connected = False

    async def initialize(self) -> bool:
        """Inicializa la conexión a la base de datos"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                min_size=self.db_config["connection_pool"]["min_size"],
                max_size=self.db_config["connection_pool"]["max_size"],
            )
            self.connected = True
            logger.info("✅ Database connection initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            self.connected = False
            return False

    async def shutdown(self) -> None:
        """Cierra las conexiones de la base de datos"""
        if self.pool:
            await self.pool.close()
            self.connected = False
            logger.info("Database connection closed.")

    async def health_check(self) -> Dict[str, Any]:
        """Verifica la salud de la base de datos"""
        if not self.connected or not self.pool:
            return {
                "healthy": False,
                "status": "not_connected",
                "error": "Database not connected",
            }

        try:
            async with self.pool.acquire() as conn:
                # Basic health check
                result = await conn.fetchone("SELECT 1 as health_check")

                # Get database info
                db_info = await conn.fetchone(
                    """
                    SELECT
                        current_database() as database_name,
                        current_schema() as schema_name,
                        pg_size_pretty(pg_database_size(current_database())) as size
                """
                )

                # Check active connections
                connections = await conn.fetchone(
                    """
                    SELECT count(*) as active_connections
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                """
                )

                return {
                    "healthy": True,
                    "status": "online",
                    "database_info": {
                        "name": db_info["database_name"],
                        "schema": db_info["schema_name"],
                        "size": db_info["size"],
                    },
                    "connections": connections["active_connections"],
                    "pool_stats": {
                        "pool_size": len(self.pool._holders),
                        "used": self.pool._used,
                    },
                }

        except Exception as e:
            return {"healthy": False, "status": "error", "error": str(e)}

    async def get_schema_status(self) -> Dict[str, Any]:
        """Estado completo del esquema de la base de datos"""
        if not self.connected:
            return {"error": "Database not connected"}

        try:
            async with self.pool.acquire() as conn:
                # Get all tables
                tables_result = await conn.fetch(
                    """
                    SELECT schemaname, tablename, tableowner,
                           pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as table_size
                    FROM pg_tables
                    WHERE schemaname = $1
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """,
                    self.db_config["schema"],
                )

                # Get all indexes
                indexes_result = await conn.fetch(
                    """
                    SELECT schemaname, tablename, indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = $1
                """,
                    self.db_config["schema"],
                )

                # Get migration status
                migration_status = await self._get_migration_status(conn)

                return {
                    "schema": self.db_config["schema"],
                    "tables": [dict(row) for row in tables_result],
                    "total_tables": len(tables_result),
                    "indexes": [dict(row) for row in indexes_result],
                    "total_indexes": len(indexes_result),
                    "migrations": migration_status,
                }

        except Exception as e:
            return {"error": str(e)}

    async def run_migrations(self, direction: str = "up") -> Dict[str, Any]:
        """Ejecuta las migraciones de base de datos"""
        try:
            if not self.migrations_dir.exists():
                return {"error": "Migrations directory not found"}

            # Check for migration files
            migration_files = list(self.migrations_dir.glob("*.sql"))
            migration_files.sort()

            results = {
                "direction": direction,
                "executed_migrations": [],
                "failed_migrations": [],
                "total_migrations": len(migration_files),
            }

            async with self.pool.acquire() as conn:
                for migration_file in migration_files:
                    try:
                        # Read migration SQL
                        with open(migration_file, "r", encoding="utf-8") as f:
                            migration_sql = f.read()

                        # Execute migration
                        if direction == "up" and "_down" not in migration_file.name:
                            await conn.execute(migration_sql)
                            results["executed_migrations"].append(migration_file.name)
                        elif direction == "down" and "_down" in migration_file.name:
                            await conn.execute(migration_sql)
                            results["executed_migrations"].append(migration_file.name)

                    except Exception as e:
                        results["failed_migrations"].append(
                            {"file": migration_file.name, "error": str(e)}
                        )

            return results

        except Exception as e:
            return {"error": str(e)}

    async def execute_query(
        self, query: str, params: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Ejecuta una consulta SQL segura"""
        if not self.connected:
            return {"error": "Database not connected"}

        try:
            async with self.pool.acquire() as conn:
                if params:
                    result = await conn.fetch(query, *params)
                else:
                    result = await conn.fetch(query)

                return {
                    "success": True,
                    "affected_rows": len(result) if result else 0,
                    "columns": result[0].keys() if result else [],
                    "data": [dict(row) for row in result] if result else [],
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def create_backup(self, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """Crea un backup de la base de datos"""
        try:
            backup_file = (
                backup_path or f"backup_{int(asyncio.get_event_loop().time())}.sql"
            )

            cmd = [
                "pg_dump",
                "-h",
                self.db_config["host"],
                "-p",
                str(self.db_config["port"]),
                "-U",
                self.db_config["user"],
                "-d",
                self.db_config["database"],
                "-f",
                backup_file,
                "--no-password",
                "--format=custom",  # Binary format for better compression
            ]

            # Set password environment
            env = os.environ.copy()
            env["PGPASSWORD"] = self.db_config["password"]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                return {
                    "success": True,
                    "backup_file": backup_file,
                    "format": "custom",
                    "size": (
                        os.path.getsize(backup_file)
                        if os.path.exists(backup_file)
                        else 0
                    ),
                }
            else:
                return {"success": False, "error": stderr.decode("utf-8")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """Restaura un backup de la base de datos"""
        try:
            if not os.path.exists(backup_path):
                return {"error": f"Backup file not found: {backup_path}"}

            cmd = [
                "pg_restore",
                "-h",
                self.db_config["host"],
                "-p",
                str(self.db_config["port"]),
                "-U",
                self.db_config["user"],
                "-d",
                self.db_config["database"],
                "--clean",  # Clean (drop) database objects before recreating them
                "--if-exists",  # Use IF EXISTS when issuing DROP commands
                backup_path,
            ]

            env = os.environ.copy()
            env["PGPASSWORD"] = self.db_config["password"]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await result.communicate()

            return {
                "success": result.returncode == 0,
                "backup_file": backup_path,
                "output": (
                    stdout.decode("utf-8")
                    if result.returncode == 0
                    else stderr.decode("utf-8")
                ),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def optimize_tables(self) -> Dict[str, Any]:
        """Optimiza las tablas de la base de datos"""
        if not self.connected:
            return {"error": "Database not connected"}

        try:
            async with self.pool.acquire() as conn:
                # VACUUM ANALYZE all tables
                await conn.execute("VACUUM ANALYZE")

                # Reindex system catalogs
                await conn.execute("REINDEX SYSTEM")

                return {
                    "success": True,
                    "operation": "VACUUM ANALYZE + REINDEX",
                    "message": "Database optimization completed",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Métricas de rendimiento de la base de datos"""
        if not self.connected:
            return {"error": "Database not connected"}

        try:
            async with self.pool.acquire() as conn:
                # Query performance metrics
                slow_queries = await conn.fetch(
                    """
                    SELECT query, mean_time, calls, total_time
                    FROM pg_stat_statements
                    ORDER BY mean_time DESC
                    LIMIT 10
                """
                )

                # Table bloat
                table_bloat = await conn.fetch(
                    """
                    SELECT schemaname, tablename,
                           pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                           pg_size_pretty(pg_table_size(schemaname||'.'||tablename)) as data_size
                    FROM pg_tables
                    WHERE schemaname = $1
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                    LIMIT 10
                """,
                    self.db_config["schema"],
                )

                # Connection stats
                conn_stats = await conn.fetchone(
                    """
                    SELECT count(*) as active_connections,
                           sum(case when state = 'idle' then 1 else 0 end) as idle_connections,
                           sum(case when state = 'active' then 1 else 0 end) as active_connections
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                """
                )

                return {
                    "slow_queries": [dict(row) for row in slow_queries],
                    "largest_tables": [dict(row) for row in table_bloat],
                    "connection_stats": dict(conn_stats) if conn_stats else {},
                    "cache_hit_ratio": await self._get_cache_hit_ratio(conn),
                }

        except Exception as e:
            return {"error": str(e)}

    async def _get_cache_hit_ratio(self, conn) -> float:
        """Calcula el ratio de aciertos de cache"""
        try:
            result = await conn.fetchone(
                """
                SELECT
                    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read))::float as cache_hit_ratio
                FROM pg_statio_user_tables
            """
            )
            return (
                float(result["cache_hit_ratio"])
                if result and result["cache_hit_ratio"]
                else 0.0
            )
        except Exception:
            return 0.0

    async def _get_migration_status(self, conn) -> Dict[str, Any]:
        """Obtiene el estado de migraciones"""
        try:
            # Check if migration table exists
            migration_table_exists = await conn.fetchone(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = $1
                    AND table_name = 'alembic_version'
                )
            """,
                self.db_config["schema"],
            )

            if migration_table_exists and migration_table_exists[0]:
                current_version = await conn.fetchone(
                    "SELECT version_num FROM alembic_version"
                )

                return {
                    "system": "Alembic",
                    "current_version": (
                        current_version["version_num"] if current_version else "none"
                    ),
                    "migration_table_exists": True,
                }
            else:
                return {
                    "system": "None detected",
                    "current_version": "unknown",
                    "migration_table_exists": False,
                }

        except Exception as e:
            return {"system": "Error", "error": str(e), "migration_table_exists": False}

    async def manage_connections(
        self, action: str, params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Gestiona conexiones activas"""
        if not self.connected:
            return {"error": "Database not connected"}

        try:
            if action == "list":
                return await self._list_active_connections()
            elif action == "kill":
                if not params or "pid" not in params:
                    return {"error": "PID required for kill action"}
                return await self._kill_connection(params["pid"])
            elif action == "stats":
                return await self._connection_stats()
            else:
                return {"error": f"Unknown connection action: {action}"}

        except Exception as e:
            return {"error": str(e)}

    async def _list_active_connections(self) -> Dict[str, Any]:
        """Lista conexiones activas"""
        async with self.pool.acquire() as conn:
            connections = await conn.fetch(
                """
                SELECT pid, usename, client_addr, client_port,
                       state, query_start, state_change
                FROM pg_stat_activity
                WHERE datname = current_database()
                ORDER BY query_start DESC NULLS LAST
            """
            )

            return {
                "connections": [dict(row) for row in connections],
                "total_connections": len(connections),
            }

    async def _kill_connection(self, pid: int) -> Dict[str, Any]:
        """Elimina una conexión específica"""
        async with self.pool.acquire() as conn:
            await conn.execute("SELECT pg_terminate_backend($1)", pid)

            return {
                "success": True,
                "terminated_pid": pid,
                "message": f"Connection {pid} terminated successfully",
            }

    async def _connection_stats(self) -> Dict[str, Any]:
        """Estadísticas de conexiones"""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchone(
                """
                SELECT
                    COUNT(*) as total_connections,
                    COUNT(CASE WHEN state = 'active' THEN 1 END) as active_connections,
                    COUNT(CASE WHEN state = 'idle' THEN 1 END) as idle_connections,
                    COUNT(CASE WHEN state_change < NOW() - INTERVAL '5 minutes' THEN 1 END) as long_running
                FROM pg_stat_activity
                WHERE datname = current_database()
            """
            )

            return dict(stats) if stats else {}

    async def get_database_logs(self, lines: int = 100) -> Dict[str, Any]:
        """Obtiene logs de la base de datos"""
        try:
            # This would need to be customized based on PostgreSQL log location
            # Typically /var/log/postgresql/postgresql-*.log
            return {
                "message": "Database logs access not implemented - requires system configuration",
                "suggestion": "Configure PostgreSQL logging in postgresql.conf",
            }
        except Exception as e:
            return {"error": str(e)}


import os
