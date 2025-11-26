#!/usr/bin/env python3
"""
GAMIFIED DATABASE INITIALIZATION MODULE
Enterprise-grade database initialization for consciousness and gamification systems
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional


def initialize_database(db_path: str = "data/gamified_database.db") -> sqlite3.Connection:
    """
    Initialize the gamified database with all required tables for enterprise tests
    
    Args:
        db_path: Path to the database file
        
    Returns:
        sqlite3.Connection: Connected database instance
        
    Raises:
        sqlite3.Error: If database initialization fails
    """
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    try:
        # Enable WAL mode for better concurrent access
        cursor.execute("PRAGMA journal_mode = WAL")
        
        # Set busy timeout to handle concurrent access
        cursor.execute("PRAGMA busy_timeout = 5000")
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # ============================================================
        # CONSCIOUSNESS STATE PERSISTENCE TABLES
        # ============================================================
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consciousness_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phi_value REAL NOT NULL,
                attention_focus TEXT NOT NULL,
                emotional_state TEXT NOT NULL,
                working_memory TEXT,
                timestamp TEXT NOT NULL,
                confidence_level REAL NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consciousness_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                importance REAL NOT NULL,
                phi_context REAL NOT NULL,
                timestamp TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ============================================================
        # TRANSACTION SAFETY TABLES
        # ============================================================
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consciousness_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phi_value REAL NOT NULL,
                operation_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                success INTEGER DEFAULT 1,
                rollbacks INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES consciousness_transactions(id)
            )
        """)
        
        # ============================================================
        # PERFORMANCE METRICS TABLES
        # ============================================================
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consciousness_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                phi_value REAL NOT NULL,
                response_time REAL NOT NULL,
                timestamp TEXT NOT NULL,
                memory_usage REAL DEFAULT 0,
                is_cached INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                success INTEGER DEFAULT 1,
                response_time REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ============================================================
        # GAMIFICATION TABLES
        # ============================================================
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT UNIQUE NOT NULL,
                exercise_type TEXT NOT NULL,
                num_answers INTEGER NOT NULL,
                correct INTEGER NOT NULL,
                incorrect INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                total_tokens INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                answers_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                exercise_id INTEGER NOT NULL,
                score REAL NOT NULL,
                completed_at TEXT NOT NULL,
                time_spent INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (exercise_id) REFERENCES exercises(id)
            )
        """)
        
        # ============================================================
        # INDEXES FOR PERFORMANCE
        # ============================================================
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_consciousness_states_timestamp 
            ON consciousness_states(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_consciousness_memories_type 
            ON consciousness_memories(type, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_consciousness_metrics_session 
            ON consciousness_metrics(session_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_performance_agent 
            ON agent_performance(agent_id, start_time)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exercises_dataset 
            ON exercises(dataset_id)
        """)
        
        # Commit all changes
        conn.commit()
        
        return conn
        
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Database initialization failed: {e}")


def get_database_connection(db_path: str = "data/gamified_database.db") -> sqlite3.Connection:
    """
    Get or create database connection
    
    Args:
        db_path: Path to the database file
        
    Returns:
        sqlite3.Connection: Connected database instance
    """
    if not os.path.exists(db_path):
        return initialize_database(db_path)
    
    return sqlite3.connect(db_path, check_same_thread=False)


def verify_database_integrity(db_path: str = "data/gamified_database.db") -> dict:
    """
    Verify database integrity and return status
    
    Args:
        db_path: Path to the database file
        
    Returns:
        dict: Database integrity status
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    integrity_status = {
        "database_exists": True,
        "tables": [],
        "indexes": [],
        "integrity_check": False
    }
    
    try:
        # Check database integrity
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        integrity_status["integrity_check"] = (result[0] == "ok")
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        integrity_status["tables"] = [row[0] for row in cursor.fetchall()]
        
        # Get all indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        integrity_status["indexes"] = [row[0] for row in cursor.fetchall()]
        
    finally:
        conn.close()
    
    return integrity_status


if __name__ == "__main__":
    # Test database initialization
    print("🔧 Initializing gamified database...")
    conn = initialize_database("data/gamified_database.db")
    print("✅ Database initialized successfully!")
    
    # Verify integrity
    status = verify_database_integrity("data/gamified_database.db")
    print(f"\n📊 Database Status:")
    print(f"   Tables: {len(status['tables'])}")
    print(f"   Indexes: {len(status['indexes'])}")
    print(f"   Integrity: {'✅ OK' if status['integrity_check'] else '❌ FAILED'}")
    
    conn.close()
