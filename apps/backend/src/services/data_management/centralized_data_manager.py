#!/usr/bin/env python3
"""
Centralized Data Management System
==================================

Sistema avanzado para gestión de datos centralizados con:
- Validación de integridad de bases de datos
- Backups automáticos
- Esquemas de datos
- Monitoreo de estado
- Recuperación de datos
"""

import json
import logging
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CentralizedDataManager:
    """Gestor avanzado de datos centralizados"""

    def __init__(self, data_dir: str = "centralized_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Definir esquemas de bases de datos
        self.schemas = {
            "enterprise_metrics": {
                "enterprise_metrics": """
                    CREATE TABLE IF NOT EXISTS enterprise_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metric_name TEXT NOT NULL,
                        metric_value REAL,
                        component TEXT,
                        metadata TEXT
                    )
                """,
                "enterprise_alerts": """
                    CREATE TABLE IF NOT EXISTS enterprise_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        alert_type TEXT NOT NULL,
                        severity TEXT CHECK(severity IN ('low', 'medium', 'high', 'critical')),
                        message TEXT,
                        component TEXT,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                """,
            },
            "neuro_user_v2_memory": {
                "memories": """
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL,
                        memory_type TEXT,
                        content TEXT,
                        context TEXT,
                        importance REAL DEFAULT 0.5
                    )
                """,
                "memory_state": """
                    CREATE TABLE IF NOT EXISTS memory_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT UNIQUE,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                        state_data TEXT
                    )
                """,
            },
            "test_user_memory": {
                "memories": """
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL,
                        memory_type TEXT,
                        content TEXT,
                        context TEXT,
                        importance REAL DEFAULT 0.5
                    )
                """,
                "memory_state": """
                    CREATE TABLE IF NOT EXISTS memory_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT UNIQUE,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                        state_data TEXT
                    )
                """,
            },
        }

    def validate_databases(self) -> Dict[str, Any]:
        """Validar integridad de todas las bases de datos"""
        results = {}

        for db_name in [
            "enterprise_metrics.db",
            "neuro_user_v2_memory.db",
            "test_user_memory.db",
        ]:
            db_path = self.data_dir / db_name
            results[db_name] = self._validate_single_database(
                db_path, db_name.replace(".db", "")
            )

        return results

    def _validate_single_database(
        self, db_path: Path, schema_name: str
    ) -> Dict[str, Any]:
        """Validar una base de datos individual"""
        result = {
            "exists": db_path.exists(),
            "integrity_check": None,
            "tables": [],
            "row_counts": {},
            "schema_compliance": False,
        }

        if not db_path.exists():
            return result

        try:
            conn = sqlite3.connect(str(db_path))

            # Verificar integridad
            integrity_result = conn.execute("PRAGMA integrity_check").fetchone()
            result["integrity_check"] = integrity_result[0] == "ok"

            # Obtener tablas
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            result["tables"] = [t[0] for t in tables]

            # Contar filas por tabla
            for table in result["tables"]:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                result["row_counts"][table] = count

            # Validar esquema
            if schema_name in self.schemas:
                expected_tables = set(self.schemas[schema_name].keys())
                actual_tables = set(result["tables"])
                result["schema_compliance"] = expected_tables.issubset(actual_tables)

            conn.close()

        except Exception as e:
            logger.error(f"Error validating {db_path}: {e}")
            result["error"] = str(e)

        return result

    def create_backup(self, backup_dir: str = "backups") -> str:
        """Crear backup completo de datos centralizados"""
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"centralized_data_backup_{timestamp}"

        try:
            shutil.copytree(self.data_dir, backup_path)
            logger.info(f"Backup creado: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Error creando backup: {e}")
            raise

    def validate_json_files(self) -> Dict[str, Any]:
        """Validar archivos JSON"""
        results = {}

        json_files = ["local_memory.json"]
        for json_file in json_files:
            file_path = self.data_dir / json_file
            results[json_file] = self._validate_json_file(file_path)

        return results

    def _validate_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Validar archivo JSON individual"""
        result = {
            "exists": file_path.exists(),
            "valid_json": False,
            "size": 0,
            "structure": None,
        }

        if not file_path.exists():
            return result

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                result["valid_json"] = True
                result["size"] = file_path.stat().st_size
                result["structure"] = self._analyze_json_structure(data)
        except Exception as e:
            logger.error(f"Error validating JSON {file_path}: {e}")
            result["error"] = str(e)

        return result

    def _analyze_json_structure(self, data: Any) -> Dict[str, Any]:
        """Analizar estructura de datos JSON"""
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys()),
                "nested_objects": {
                    k: self._analyze_json_structure(v)
                    for k, v in data.items()
                    if isinstance(v, (dict, list))
                },
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "element_types": list(set(type(item).__name__ for item in data)),
            }
        else:
            return {"type": type(data).__name__}

    def ensure_schemas(self) -> Dict[str, Any]:
        """Asegurar que todas las bases de datos tengan los esquemas correctos"""
        results = {}

        for db_name, schema in self.schemas.items():
            db_path = self.data_dir / f"{db_name}.db"
            results[db_name] = self._ensure_schema(db_path, schema)

        return results

    def _ensure_schema(self, db_path: Path, schema: Dict[str, str]) -> Dict[str, Any]:
        """Asegurar esquema de una base de datos"""
        result = {"success": False, "created_tables": [], "errors": []}

        try:
            conn = sqlite3.connect(str(db_path))

            for table_name, create_sql in schema.items():
                try:
                    conn.execute(create_sql)
                    result["created_tables"].append(table_name)
                except Exception as e:
                    result["errors"].append(f"{table_name}: {e}")

            conn.commit()
            conn.close()
            result["success"] = len(result["errors"]) == 0

        except Exception as e:
            result["errors"].append(f"Connection error: {e}")

        return result

    def get_data_summary(self) -> Dict[str, Any]:
        """Obtener resumen completo del estado de los datos"""
        return {
            "timestamp": datetime.now().isoformat(),
            "database_validation": self.validate_databases(),
            "json_validation": self.validate_json_files(),
            "schema_compliance": self.ensure_schemas(),
        }


def main():
    """Función principal para gestión de datos"""
    manager = CentralizedDataManager()

    print("=== Centralized Data Validation ===")
    summary = manager.get_data_summary()

    print("\n[*] Data Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Crear backup si todo está bien
    if all(
        db["integrity_check"]
        for db in summary["database_validation"].values()
        if db["exists"]
    ):
        print("\n[*] Creating backup...")
        backup_path = manager.create_backup()
        print(f"[+] Backup created: {backup_path}")
    else:
        print("\n[!] Some databases have integrity issues. Backup not created.")


if __name__ == "__main__":
    main()
