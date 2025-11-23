#!/usr/bin/env python3
"""Update user balance to 10000 SHEILYS and create table if needed"""
import sqlite3
from datetime import datetime

conn = sqlite3.connect("gamified_database.db")
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS user_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL DEFAULT 'default_user',
        balance INTEGER NOT NULL DEFAULT 10000 CHECK(balance >= 0),
        level INTEGER NOT NULL DEFAULT 1 CHECK(level >= 1),
        total_exercises INTEGER NOT NULL DEFAULT 0 CHECK(total_exercises >= 0),
        accuracy_rate REAL NOT NULL DEFAULT 0.0 CHECK(accuracy_rate >= 0 AND accuracy_rate <= 100),
        streak_days INTEGER NOT NULL DEFAULT 0 CHECK(streak_days >= 0),
        updated_at TEXT NOT NULL,
        UNIQUE(user_id)
    )
"""
)

print("✅ Tabla user_tokens verificada/creada")

# Check if user exists
cursor.execute(
    "SELECT id, balance FROM user_tokens WHERE user_id = ?", ("default_user",)
)
existing = cursor.fetchone()

if existing:
    # Update existing user
    cursor.execute(
        "UPDATE user_tokens SET balance = 10000 WHERE user_id = ?", ("default_user",)
    )
    print(f"✅ Balance actualizado de {existing[1]} a 10000 SHEILYS")
else:
    # Insert new user
    cursor.execute(
        """
        INSERT INTO user_tokens (user_id, balance, level, total_exercises, accuracy_rate, streak_days, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        ("default_user", 10000, 1, 0, 0.0, 0, datetime.now().isoformat()),
    )
    print("✅ Nuevo usuario creado con 10000 SHEILYS")

conn.commit()

# Verify
cursor.execute(
    "SELECT user_id, balance, level FROM user_tokens WHERE user_id = ?",
    ("default_user",),
)
result = cursor.fetchone()
if result:
    print(f"✅ Usuario: {result[0]}, Balance: {result[1]} SHEILYS, Nivel: {result[2]}")
else:
    print("❌ Usuario no encontrado")

conn.close()
