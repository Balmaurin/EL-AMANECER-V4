#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sheily_memory_vault.py
======================
Simple memory vault for chat system
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SimpleMemoryVault:
    """Simple in-memory vault for chat functionality"""

    def __init__(self):
        self.memories = {}
        self.db_path = Path("data/chat_memories.json")
        self.db_path.parent.mkdir(exist_ok=True)
        self._load_memories()

    def _load_memories(self):
        """Load memories from file"""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load memories: {e}")
                self.memories = {}
        else:
            self.memories = {}

    def _save_memories(self):
        """Save memories to file"""
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")

    def remember_chunked(
        self, text: str, user_id: str, origin: str = "chat", meta: Dict = None
    ) -> List[str]:
        """Store text in memory chunks"""
        if user_id not in self.memories:
            self.memories[user_id] = []

        chunk_id = f"{user_id}_{len(self.memories[user_id])}_{origin}"
        memory = {
            "id": chunk_id,
            "text": text,
            "origin": origin,
            "meta": meta or {},
            "score": 1.0,  # Simple relevance score
        }

        self.memories[user_id].append(memory)
        self._save_memories()
        return [chunk_id]

    def forget_related(
        self, user_id: str, query: str, threshold: float = 0.5, top_k: int = 10
    ) -> int:
        """Remove memories related to query"""
        if user_id not in self.memories:
            return 0

        # Simple text matching for related content
        to_remove = []
        query_lower = query.lower()

        for memory in self.memories[user_id]:
            if query_lower in memory["text"].lower():
                to_remove.append(memory)

        for memory in to_remove:
            self.memories[user_id].remove(memory)

        self._save_memories()
        return len(to_remove)

    def forget_exact_fragment(self, user_id: str, fragment: str, top_k: int = 5) -> int:
        """Remove exact or similar fragments"""
        if user_id not in self.memories:
            return 0

        to_remove = []
        fragment_lower = fragment.lower()

        for memory in self.memories[user_id]:
            if fragment_lower in memory["text"].lower():
                to_remove.append(memory)

        for memory in to_remove[:top_k]:
            self.memories[user_id].remove(memory)

        self._save_memories()
        return len(to_remove[:top_k])

    def search_semantic(self, query: str, user_id: str, top_k: int = 5) -> List[Dict]:
        """Simple semantic search using text matching"""
        if user_id not in self.memories:
            return []

        results = []
        query_lower = query.lower()

        for memory in self.memories[user_id]:
            text_lower = memory["text"].lower()
            if query_lower in text_lower:
                score = 1.0  # Simple scoring
                results.append(
                    {
                        "text": memory["text"],
                        "score": score,
                        "id": memory["id"],
                        "meta": memory["meta"],
                    }
                )

        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# Global instance
vault = SimpleMemoryVault()
