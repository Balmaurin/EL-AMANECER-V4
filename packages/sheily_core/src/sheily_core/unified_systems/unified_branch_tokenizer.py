#!/usr/bin/env python3
"""
Unified Branch Tokenizer - Sistema de Tokenización de Ramas Unificado

Este módulo implementa un sistema avanzado de tokenización de ramas
para el procesamiento inteligente de estructuras jerárquicas y ramificaciones.

Autor: Unified Systems Team
Fecha: 2025-11-12
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .unified_system_core import SystemConfig

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BranchToken:
    """Token de rama con metadatos completos"""

    token_id: str
    branch_name: str
    token_value: str
    token_type: str  # 'word', 'symbol', 'number', 'special'
    position: int
    depth: int
    parent_token: Optional[str] = None
    child_tokens: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "token_id": self.token_id,
            "branch_name": self.branch_name,
            "token_value": self.token_value,
            "token_type": self.token_type,
            "position": self.position,
            "depth": self.depth,
            "parent_token": self.parent_token,
            "child_tokens": self.child_tokens,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BranchToken":
        """Crear desde diccionario"""
        data_copy = data.copy()
        data_copy["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data_copy)


@dataclass
class BranchConfig:
    """Configuración del tokenizador de ramas"""

    max_branch_depth: int = 10
    max_tokens_per_branch: int = 1000
    enable_caching: bool = True
    cache_size: int = 10000
    token_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "word": r"[a-zA-Z_][a-zA-Z0-9_]*",
            "number": r"\d+(\.\d+)?",
            "symbol": r"[+\-*/=<>!&|^%~]",
            "special": r"[\[\](){},.;:]",
        }
    )
    reserved_words: Set[str] = field(
        default_factory=lambda: {
            "if",
            "else",
            "for",
            "while",
            "def",
            "class",
            "import",
            "from",
            "return",
            "break",
            "continue",
            "try",
            "except",
            "finally",
        }
    )


class UnifiedBranchTokenizer:
    """
    Tokenizador unificado de ramas para procesamiento inteligente
    de estructuras jerárquicas y ramificaciones del código.
    """

    def __init__(self, config: Optional[BranchConfig] = None):
        """Inicializar tokenizador de ramas"""
        self.config = config or BranchConfig()
        self.branches: Dict[str, List[BranchToken]] = {}
        self.token_cache: Dict[str, List[BranchToken]] = {}
        self.branch_metadata: Dict[str, Dict[str, Any]] = {}

        # Compilar patrones regex
        self.compiled_patterns = {}
        for token_type, pattern in self.config.token_patterns.items():
            self.compiled_patterns[token_type] = re.compile(pattern)

        logger.info("🎯 Unified Branch Tokenizer inicializado")

    def tokenize_branch(
        self, branch_name: str, content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tokenizar una rama completa

        Args:
            branch_name: Nombre de la rama
            content: Contenido a tokenizar (opcional)

        Returns:
            Información completa de la tokenización
        """
        try:
            # Usar contenido del branch_name si no se proporciona content
            if content is None:
                content = branch_name

            # Verificar caché
            if self.config.enable_caching and branch_name in self.token_cache:
                cached_result = self._get_cached_result(branch_name)
                if cached_result:
                    return cached_result

            # Tokenizar contenido
            tokens = self._tokenize_content(branch_name, content)

            # Construir jerarquía
            token_tree = self._build_token_hierarchy(tokens)

            # Calcular métricas
            metrics = self._calculate_branch_metrics(tokens)

            # Crear resultado
            result = {
                "branch_name": branch_name,
                "tokens": [token.to_dict() for token in tokens],
                "token_tree": token_tree,
                "metrics": metrics,
                "token_count": len(tokens),
                "depth": max((t.depth for t in tokens), default=0),
                "processed_at": datetime.now().isoformat(),
            }

            # Almacenar en caché
            if self.config.enable_caching:
                self._cache_result(branch_name, result)

            # Almacenar en memoria
            self.branches[branch_name] = tokens

            return result

        except Exception as e:
            logger.error(f"Error tokenizando rama {branch_name}: {e}")
            return {
                "branch_name": branch_name,
                "error": str(e),
                "tokens": [],
                "token_tree": {},
                "metrics": {},
                "token_count": 0,
                "depth": 0,
            }

    def _tokenize_content(self, branch_name: str, content: str) -> List[BranchToken]:
        """
        Tokenizar contenido usando patrones regex
        """
        tokens = []
        position = 0
        token_id_counter = 0

        # Dividir en líneas para manejar jerarquía
        lines = content.split("\n")

        for line_num, line in enumerate(lines):
            line_tokens = self._tokenize_line(line, branch_name, line_num)
            for token in line_tokens:
                token.position = position
                token.token_id = f"{branch_name}_{token_id_counter}"
                tokens.append(token)
                token_id_counter += 1
                position += 1

        return tokens

    def _tokenize_line(
        self, line: str, branch_name: str, line_num: int
    ) -> List[BranchToken]:
        """Tokenizar una línea individual"""
        tokens = []
        remaining = line.strip()

        # Calcular profundidad basada en indentación
        depth = len(line) - len(line.lstrip())

        while remaining:
            token_found = False

            # Intentar cada patrón
            for token_type, pattern in self.compiled_patterns.items():
                match = pattern.match(remaining)
                if match:
                    token_value = match.group(0)

                    # Crear token
                    token = BranchToken(
                        token_id="",  # Se asignará después
                        branch_name=branch_name,
                        token_value=token_value,
                        token_type=token_type,
                        position=0,  # Se asignará después
                        depth=depth,
                        metadata={
                            "line_number": line_num,
                            "is_reserved": token_value in self.config.reserved_words,
                            "length": len(token_value),
                        },
                    )

                    tokens.append(token)
                    remaining = remaining[len(token_value) :].lstrip()
                    token_found = True
                    break

            if not token_found:
                # Token desconocido, tratar como especial
                char = remaining[0]
                token = BranchToken(
                    token_id="",
                    branch_name=branch_name,
                    token_value=char,
                    token_type="unknown",
                    position=0,
                    depth=depth,
                    metadata={"line_number": line_num, "unknown_char": True},
                )
                tokens.append(token)
                remaining = remaining[1:].lstrip()

        return tokens

    def _build_token_hierarchy(self, tokens: List[BranchToken]) -> Dict[str, Any]:
        """Construir jerarquía de tokens"""
        hierarchy = {"root": []}

        for i, token in enumerate(tokens):
            # Determinar padre basado en profundidad
            parent_id = None
            for j in range(i - 1, -1, -1):
                if tokens[j].depth < token.depth:
                    parent_id = tokens[j].token_id
                    break

            token.parent_token = parent_id

            # Añadir a hijos del padre
            if parent_id:
                parent_token = next(
                    (t for t in tokens if t.token_id == parent_id), None
                )
                if parent_token:
                    parent_token.child_tokens.append(token.token_id)

            # Añadir a jerarquía
            depth_key = f"depth_{token.depth}"
            if depth_key not in hierarchy:
                hierarchy[depth_key] = []
            hierarchy[depth_key].append(token.token_id)

        return hierarchy

    def _calculate_branch_metrics(self, tokens: List[BranchToken]) -> Dict[str, Any]:
        """Calcular métricas de la rama"""
        if not tokens:
            return {}

        token_types = {}
        depths = {}
        reserved_words = 0

        for token in tokens:
            # Contar tipos
            token_types[token.token_type] = token_types.get(token.token_type, 0) + 1

            # Contar profundidades
            depths[token.depth] = depths.get(token.depth, 0) + 1

            # Contar palabras reservadas
            if token.metadata.get("is_reserved", False):
                reserved_words += 1

        return {
            "token_types": token_types,
            "depth_distribution": depths,
            "reserved_words_count": reserved_words,
            "avg_token_length": sum(len(t.token_value) for t in tokens) / len(tokens),
            "max_depth": max((t.depth for t in tokens), default=0),
            "unique_tokens": len(set(t.token_value for t in tokens)),
        }

    def _get_cached_result(self, branch_name: str) -> Optional[Dict[str, Any]]:
        """Obtener resultado del caché"""
        if branch_name in self.token_cache:
            cached = self.token_cache[branch_name]
            # Verificar si el caché no está expirado (simplificado)
            return cached
        return None

    def _cache_result(self, branch_name: str, result: Dict[str, Any]):
        """Almacenar resultado en caché"""
        if len(self.token_cache) >= self.config.cache_size:
            # Eliminar entrada más antigua (simplificado)
            oldest_key = next(iter(self.token_cache))
            del self.token_cache[oldest_key]

        self.token_cache[branch_name] = result

    def get_branch_info(self, branch_name: str) -> Dict[str, Any]:
        """
        Obtener información completa de una rama

        Args:
            branch_name: Nombre de la rama

        Returns:
            Información de la rama
        """
        if branch_name not in self.branches:
            # Tokenizar si no existe
            self.tokenize_branch(branch_name)

        tokens = self.branches.get(branch_name, [])

        return {
            "branch_name": branch_name,
            "exists": branch_name in self.branches,
            "token_count": len(tokens),
            "depth": max((t.depth for t in tokens), default=0) if tokens else 0,
            "cached": branch_name in self.token_cache,
            "last_processed": self.branch_metadata.get(branch_name, {}).get(
                "last_processed"
            ),
        }

    def compare_branches(self, branch1: str, branch2: str) -> Dict[str, Any]:
        """
        Comparar dos ramas tokenizadas

        Args:
            branch1: Nombre de la primera rama
            branch2: Nombre de la segunda rama

        Returns:
            Comparación detallada
        """
        info1 = self.get_branch_info(branch1)
        info2 = self.get_branch_info(branch2)

        # Tokenizar si es necesario
        tokens1 = self.branches.get(branch1, [])
        tokens2 = self.branches.get(branch2, [])

        # Calcular similitudes
        similarity = self._calculate_similarity(tokens1, tokens2)

        return {
            "branch1": info1,
            "branch2": info2,
            "similarity_score": similarity,
            "comparison": {
                "token_count_diff": info1["token_count"] - info2["token_count"],
                "depth_diff": info1["depth"] - info2["depth"],
                "structure_similarity": similarity.get("structure", 0),
            },
        }

    def _calculate_similarity(
        self, tokens1: List[BranchToken], tokens2: List[BranchToken]
    ) -> Dict[str, float]:
        """Calcular similitud entre dos listas de tokens"""
        if not tokens1 or not tokens2:
            return {"overall": 0.0, "structure": 0.0, "content": 0.0}

        # Similitud de estructura (basada en tipos de token)
        types1 = [t.token_type for t in tokens1]
        types2 = [t.token_type for t in tokens2]

        structure_similarity = len(set(types1) & set(types2)) / len(
            set(types1) | set(types2)
        )

        # Similitud de contenido (basada en valores de token)
        values1 = set(t.token_value for t in tokens1)
        values2 = set(t.token_value for t in tokens2)

        content_similarity = len(values1 & values2) / len(values1 | values2)

        # Similitud general
        overall_similarity = (structure_similarity + content_similarity) / 2

        return {
            "overall": overall_similarity,
            "structure": structure_similarity,
            "content": content_similarity,
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        total_branches = len(self.branches)
        total_tokens = sum(len(tokens) for tokens in self.branches.values())
        cached_branches = len(self.token_cache)

        return {
            "total_branches": total_branches,
            "total_tokens": total_tokens,
            "cached_branches": cached_branches,
            "cache_hit_rate": (
                cached_branches / total_branches if total_branches > 0 else 0
            ),
            "avg_tokens_per_branch": (
                total_tokens / total_branches if total_branches > 0 else 0
            ),
            "config": {
                "max_branch_depth": self.config.max_branch_depth,
                "max_tokens_per_branch": self.config.max_tokens_per_branch,
                "cache_enabled": self.config.enable_caching,
                "cache_size": self.config.cache_size,
            },
        }

    async def cleanup(self):
        """Limpiar recursos"""
        self.branches.clear()
        self.token_cache.clear()
        self.branch_metadata.clear()
        logger.info("🧹 Unified Branch Tokenizer limpiado")


# Alias para compatibilidad con código existente
VocabBuilder20Branches = UnifiedBranchTokenizer


# Función de compatibilidad para código existente
def tokenize_branch(branch_name: str) -> Dict[str, Any]:
    """Función de compatibilidad para tokenizar rama"""
    tokenizer = UnifiedBranchTokenizer()
    return tokenizer.tokenize_branch(branch_name)


if __name__ == "__main__":
    # Demo del tokenizador
    async def demo():
        tokenizer = UnifiedBranchTokenizer()

        # Ejemplo de código Python
        sample_code = """
def hello_world():
    if True:
        print("Hello, World!")
        return "success"
    else:
        return "error"
"""

        result = tokenizer.tokenize_branch("demo_branch", sample_code)

        print("🎯 Unified Branch Tokenizer Demo")
        print("=" * 50)
        print(f"Branch: {result['branch_name']}")
        print(f"Tokens: {result['token_count']}")
        print(f"Depth: {result['depth']}")
        print(f"Metrics: {result['metrics']}")

        # Mostrar algunos tokens
        print("\n📝 Sample Tokens:")
        for token in result["tokens"][:10]:
            print(f"  {token['token_type']}: {token['token_value']}")

        # Estadísticas del sistema
        stats = tokenizer.get_system_stats()
        print(f"\n📊 System Stats: {stats}")

    asyncio.run(demo())
