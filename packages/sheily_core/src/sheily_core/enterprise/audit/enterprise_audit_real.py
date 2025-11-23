#!/usr/bin/env python3
"""
Enterprise Real Audit - Solo componentes reales del proyecto
===========================================================

Auditoría práctica que solo usa módulos y endpoints reales existentes:
- Backend FastAPI `/health`, `/api/stats`, `/api/rag/quickdemo`
- SQLite local `gamified_database.db` tablas reales
- Corpus local y archivos RAG

Resultado: diccionario homogéneo con puntuación y resumen.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests


@dataclass
class AuditSection:
    name: str
    status: str  # healthy|warning|critical
    message: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


class MCPAuditReal:
    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8001")
        self.results: Dict[str, Any] = {
            "audit_id": f"real_{int(time.time())}",
            "auditor": "MCPAuditReal",
            "started_at": datetime.now().isoformat(),
            "sections": {},
        }

    def _add(self, section: AuditSection) -> None:
        self.results["sections"][section.name] = section.to_dict()

    async def _check_backend(self) -> None:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            if r.status_code == 200:
                self._add(
                    AuditSection(
                        name="backend_health",
                        status="healthy",
                        message="API /health OK",
                        details=(
                            r.json()
                            if r.headers.get("content-type", "").startswith(
                                "application/json"
                            )
                            else {"body": r.text}
                        ),
                    )
                )
            else:
                self._add(
                    AuditSection(
                        name="backend_health",
                        status="warning" if r.status_code < 500 else "critical",
                        message=f"/health HTTP {r.status_code}",
                        details={},
                    )
                )
        except Exception as e:
            self._add(
                AuditSection(
                    name="backend_health",
                    status="critical",
                    message=f"Error: {e}",
                    details={},
                )
            )

    async def _check_stats(self) -> None:
        try:
            r = requests.get(f"{self.base_url}/api/stats", timeout=15)
            if r.status_code == 200:
                self._add(
                    AuditSection(
                        name="backend_stats",
                        status="healthy",
                        message="/api/stats OK",
                        details=r.json(),
                    )
                )
            else:
                self._add(
                    AuditSection(
                        name="backend_stats",
                        status="warning" if r.status_code < 500 else "critical",
                        message=f"/api/stats HTTP {r.status_code}",
                        details={},
                    )
                )
        except Exception as e:
            self._add(
                AuditSection(
                    name="backend_stats",
                    status="critical",
                    message=f"Error: {e}",
                    details={},
                )
            )

    async def _check_rag(self) -> None:
        # quickdemo
        try:
            r = requests.get(f"{self.base_url}/api/rag/quickdemo", timeout=20)
            ok = r.status_code == 200
        except Exception:
            ok = False
        # search
        try:
            s = requests.post(
                f"{self.base_url}/api/rag/search",
                json={"query": "aprendizaje"},
                timeout=20,
            )
            ok = ok and (s.status_code == 200)
            details = {
                "quickdemo": getattr(r, "status_code", None),
                "search": getattr(s, "status_code", None),
            }
        except Exception as e:
            details = {"error": str(e)}
            ok = False
        self._add(
            AuditSection(
                name="rag_endpoints",
                status="healthy" if ok else "warning",
                message=(
                    "RAG endpoints verificados" if ok else "RAG endpoints con issues"
                ),
                details=details,
            )
        )

    async def _check_sqlite(self) -> None:
        try:
            import sqlite3

            conn = sqlite3.connect("gamified_database.db", timeout=5)
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS rag_autodata (id INTEGER PRIMARY KEY AUTOINCREMENT, exercise_id INTEGER, question_text TEXT, user_answer TEXT, correct_answer TEXT, is_correct BOOLEAN, category TEXT, difficulty TEXT, timestamp TEXT)"
            )
            cur.execute("SELECT COUNT(*) FROM rag_autodata")
            rows = cur.fetchone()[0]
            conn.close()
            self._add(
                AuditSection(
                    name="sqlite_db",
                    status="healthy",
                    message="SQLite OK",
                    details={"rag_autodata_rows": rows},
                )
            )
        except Exception as e:
            self._add(
                AuditSection(
                    name="sqlite_db",
                    status="critical",
                    message=f"SQLite error: {e}",
                    details={},
                )
            )

    async def _check_corpus(self) -> None:
        docs = 0
        for d in [Path("corpus/corpus"), Path("corpus/sample_data"), Path("data")]:
            if d.exists():
                for fp in d.rglob("*"):
                    if fp.is_file() and fp.suffix.lower() in {".txt", ".md", ".json"}:
                        docs += 1
        self._add(
            AuditSection(
                name="rag_corpus",
                status="healthy" if docs > 0 else "warning",
                message=f"Documentos: {docs}",
                details={"documents": docs},
            )
        )

    async def run(self) -> Dict[str, Any]:
        await self._check_backend()
        await self._check_stats()
        await self._check_rag()
        await self._check_sqlite()
        await self._check_corpus()

        # Simple scoring: 20 points per healthy section
        healthy = sum(
            1 for s in self.results["sections"].values() if s["status"] == "healthy"
        )
        score = healthy * 20
        self.results.update(
            {
                "finished_at": datetime.now().isoformat(),
                "success": True,
                "overall_health_score": score,
            }
        )
        return self.results


async def run_real_audit() -> Dict[str, Any]:
    auditor = MCPAuditReal()
    return await auditor.run()
