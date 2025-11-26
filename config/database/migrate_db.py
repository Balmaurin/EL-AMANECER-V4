#!/usr/bin/env python3
"""
MIGRACIÓN FORZADA DB - Sheily MCP System
Forzar migración de tabla exercises a estructura correcta
"""

import os
import sqlite3

# Database path - configuración independiente
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "gamified_database.db")


def force_migrate():
    """Forzar migración de tabla exercises"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        print("🔄 Verificando estructura de tabla exercises...")

        # Verificar si existe tabla
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='exercises'"
        )
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            print("✨ Creando tabla exercises nueva...")
            cursor.execute(
                """
                CREATE TABLE exercises (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT UNIQUE NOT NULL,
                    exercise_type TEXT NOT NULL,
                    num_answers INTEGER NOT NULL,
                    correct INTEGER NOT NULL,
                    incorrect INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    answers_json TEXT
                )
            """
            )
            print("✅ Tabla exercises creada exitosamente")
        else:
            # Verificar si tiene la columna dataset_id
            cursor.execute("PRAGMA table_info(exercises)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]

            if "dataset_id" not in column_names:
                print("⚠️ Tabla antigua detectada - migrando a estructura limpia...")

                # SOLUCIÓN DEFINITIVA: Recrear tabla desde cero
                cursor.execute("DROP TABLE exercises")

                cursor.execute(
                    """
                    CREATE TABLE exercises (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dataset_id TEXT UNIQUE NOT NULL,
                        exercise_type TEXT NOT NULL,
                        num_answers INTEGER NOT NULL,
                        correct INTEGER NOT NULL,
                        incorrect INTEGER NOT NULL,
                        accuracy REAL NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        answers_json TEXT
                    )
                """
                )

                print("✅ Migración forzada completada - tabla limpia creada")
            else:
                print("✅ Tabla ya tiene estructura correcta")

        # Verificar estructura final
        cursor.execute("PRAGMA table_info(exercises)")
        final_columns = cursor.fetchall()
        print(f"📊 Estructura final de tabla exercises:")
        for col in final_columns:
            print(f"   - {col[1]}: {col[2]} {'UNIQUE' if col[5] else ''}")

        conn.commit()
        print("✅ Migración forzada finalizada exitosamente!")

    except Exception as e:
        print(f"❌ Error en migración forzada: {e}")
        conn.rollback()

    finally:
        conn.close()


if __name__ == "__main__":
    force_migrate()
