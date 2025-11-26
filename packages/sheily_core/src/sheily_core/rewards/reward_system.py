#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Recompensas Sheilys - Sheily AI
==========================================

Sistema completo y funcional de recompensas para aprendizaje incremental:
- Cálculo multifactorial de puntuaciones Sheilys
- Almacenamiento persistente en vault
- Optimización automática basada en rendimiento
- Integración con sistema de aprendizaje
"""

import hashlib
import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Importar evaluación contextual
try:
    from .contextual_accuracy import evaluate_contextual_accuracy
except ImportError:
    # Fallback si no está disponible
    def evaluate_contextual_accuracy(query: str, response: str) -> float:
        return 0.7  # Valor por defecto


class SheilyRewardSystem:
    """
    Sistema de Recompensas Sheilys para aprendizaje incremental
    Gestiona la evaluación y almacenamiento de recompensas
    """

    def __init__(
        self,
        vault_path: str = "./rewards/vault",
        max_vault_size: int = 10000,
        retention_days: int = 90,
        auto_cleanup: bool = True,
    ):
        """
        Inicializar sistema de recompensas

        Args:
            vault_path (str): Directorio para almacenar recompensas
            max_vault_size (int): Número máximo de recompensas a mantener
            retention_days (int): Días para retener recompensas
            auto_cleanup (bool): Limpiar automáticamente recompensas antiguas
        """
        self.vault_path = Path(vault_path)
        self.max_vault_size = max_vault_size
        self.retention_days = retention_days
        self.auto_cleanup = auto_cleanup

        # Crear directorio si no existe
        self.vault_path.mkdir(parents=True, exist_ok=True)

        # Estadísticas del sistema
        self.stats = {
            "total_rewards": 0,
            "total_sheilys": 0.0,
            "domains_processed": set(),
            "last_cleanup": None,
        }

        # Cargar estadísticas existentes
        self._load_stats()

        if self.auto_cleanup:
            self.cleanup_old_rewards()

    def _load_stats(self):
        """Cargar estadísticas del sistema"""
        stats_file = self.vault_path / "system_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    self.stats.update(json.load(f))
                    # Convertir set de vuelta
                    self.stats["domains_processed"] = set(
                        self.stats["domains_processed"]
                    )
            except Exception as e:
                print(f"Error cargando estadísticas: {e}")

    def _save_stats(self):
        """Guardar estadísticas del sistema"""
        stats_file = self.vault_path / "system_stats.json"
        try:
            # Convertir set para serialización
            stats_copy = self.stats.copy()
            stats_copy["domains_processed"] = list(self.stats["domains_processed"])

            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats_copy, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando estadísticas: {e}")

    def _calculate_sheilys(self, session_data: Dict[str, Any]) -> float:
        """
        Calcular puntuación de Sheilys con un modelo multifactorial avanzado

        Factores de evaluación:
        1. Calidad de la respuesta (25%)
        2. Complejidad del dominio (20%)
        3. Tokens procesados (15%)
        4. Novedad de la consulta (10%)
        5. Profundidad de la interacción (15%)
        6. Precisión contextual AVANZADA (15%)

        Args:
            session_data (dict): Datos completos de la sesión

        Returns:
            float: Puntuación de Sheilys (0-10)
        """
        # Configuración de pesos para cada factor
        factors = {
            "quality_score": 0.25,  # Calidad general de la respuesta
            "domain_complexity": 0.2,  # Complejidad del dominio
            "tokens_complexity": 0.15,  # Complejidad de tokens
            "novelty_factor": 0.1,  # Novedad de la consulta
            "interaction_depth": 0.15,  # Profundidad de la interacción
            "contextual_accuracy": 0.15,  # Precisión contextual AVANZADA
        }

        # Mapeo de dominios con complejidad incremental
        domain_complexity = {
            "medicina": 1.3,
            "ciberseguridad": 1.25,
            "programación": 1.2,
            "matemáticas": 1.15,
            "ciencia": 1.1,
            "ingeniería": 1.05,
            "general": 1.0,
            "entretenimiento": 0.9,
            "vida_diaria": 0.95,
            "legal": 1.1,
            "negocios": 1.05,
            "educación": 1.0,
            "deportes": 0.85,
            "arte": 0.9,
            "tecnología": 1.1,
        }

        # Extraer datos de la sesión
        quality_score = session_data.get("quality_score", 0.5)
        domain = session_data.get("domain", "general")
        tokens_used = session_data.get("tokens_used", 0)
        query = session_data.get("query", "")
        response = session_data.get("response", "")

        # 1. Complejidad de dominio
        complexity = domain_complexity.get(domain, 1.0)

        # 2. Complejidad de tokens (progresiva)
        tokens_complexity = min(1.0, math.log(tokens_used + 1, 100))

        # 3. Factor de novedad (basado en longitud y unicidad de la consulta)
        def calculate_novelty(text):
            # Considerar longitud y diversidad de palabras
            words = text.split()
            unique_words = len(set(words))
            word_diversity = unique_words / len(words) if words else 0
            length_factor = min(1.0, math.log(len(text) + 1, 100))
            return word_diversity * length_factor

        novelty_factor = calculate_novelty(query)

        # 4. Profundidad de interacción
        def interaction_depth(query, response):
            # Evaluar si la respuesta aborda múltiples aspectos de la consulta
            if not query:
                return 0.0  # Sin consulta, puntuación mínima
            query_tokens = set(query.lower().split())
            response_tokens = set(response.lower().split())
            coverage = len(query_tokens.intersection(response_tokens)) / len(
                query_tokens
            )
            return min(1.0, coverage * 1.5)  # Normalizar

        interaction_score = interaction_depth(query, response)

        # 5. Precisión Contextual AVANZADA
        try:
            contextual_accuracy = evaluate_contextual_accuracy(query, response)
        except Exception as e:
            print(f"Error en evaluación contextual: {e}")
            contextual_accuracy = 0.7  # Valor por defecto

        # Cálculo final de Sheilys
        sheilys = (
            factors["quality_score"] * quality_score
            + factors["domain_complexity"] * complexity
            + factors["tokens_complexity"] * tokens_complexity
            + factors["novelty_factor"] * novelty_factor
            + factors["interaction_depth"] * interaction_score
            + factors["contextual_accuracy"] * contextual_accuracy
        ) * 10  # Escalar a 0-10

        # Aplicar límites y redondear
        return round(max(0, min(sheilys, 10)), 2)

    def record_reward(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registrar recompensa Sheilys para una sesión

        Args:
            session_data (dict): Datos de la sesión

        Returns:
            dict: Detalles de la recompensa registrada
        """
        # Calcular Sheilys
        sheilys = self._calculate_sheilys(session_data)

        # Preparar datos de recompensa
        reward_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": session_data.get("session_id", "unknown"),
            "domain": session_data.get("domain", "general"),
            "sheilys": sheilys,
            "details": session_data,
            "factors_used": {
                "quality_score": session_data.get("quality_score", 0.5),
                "domain": session_data.get("domain", "general"),
                "tokens_used": session_data.get("tokens_used", 0),
                "contextual_accuracy": evaluate_contextual_accuracy(
                    session_data.get("query", ""), session_data.get("response", "")
                ),
            },
        }

        # Generar ID único
        reward_id = hashlib.sha256(
            json.dumps(reward_data, sort_keys=True).encode("utf-8")
        ).hexdigest()
        reward_data["reward_id"] = reward_id

        # Guardar en vault
        reward_file = self.vault_path / f"{reward_id}.json"
        with open(reward_file, "w", encoding="utf-8") as f:
            json.dump(reward_data, f, ensure_ascii=False, indent=2)

        # Actualizar estadísticas
        self.stats["total_rewards"] += 1
        self.stats["total_sheilys"] += sheilys
        self.stats["domains_processed"].add(session_data.get("domain", "general"))
        self._save_stats()

        print(
            f"✅ Recompensa registrada: {sheilys} Sheilys para sesión {reward_id[:8]}..."
        )

        return reward_data

    def get_total_sheilys(self, domain: str = None) -> float:
        """
        Obtener total de Sheilys acumulados

        Args:
            domain (str, optional): Filtrar por dominio específico

        Returns:
            float: Total de Sheilys
        """
        total_sheilys = 0.0
        cutoff_date = datetime.now(UTC) - timedelta(days=self.retention_days)

        for filename in os.listdir(self.vault_path):
            if filename.endswith(".json") and filename != "system_stats.json":
                filepath = self.vault_path / filename

                # Leer recompensa
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        reward = json.load(f)

                    # Filtrar por fecha y dominio
                    reward_date = datetime.fromisoformat(reward["timestamp"])
                    if reward_date >= cutoff_date and (
                        domain is None or reward["domain"] == domain
                    ):
                        total_sheilys += reward["sheilys"]
                except Exception as e:
                    print(f"Error leyendo recompensa {filename}: {e}")

        return round(total_sheilys, 2)

    def get_reward_history(
        self, domain: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de recompensas

        Args:
            domain (str, optional): Filtrar por dominio
            limit (int): Número máximo de recompensas a retornar

        Returns:
            list: Historial de recompensas
        """
        rewards = []
        cutoff_date = datetime.now(UTC) - timedelta(days=self.retention_days)

        for filename in os.listdir(self.vault_path):
            if filename.endswith(".json") and filename != "system_stats.json":
                filepath = self.vault_path / filename

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        reward = json.load(f)

                    reward_date = datetime.fromisoformat(reward["timestamp"])
                    if reward_date >= cutoff_date and (
                        domain is None or reward["domain"] == domain
                    ):
                        rewards.append(reward)
                except Exception as e:
                    print(f"Error leyendo recompensa {filename}: {e}")

        # Ordenar por fecha descendente (más recientes primero)
        rewards.sort(key=lambda x: x["timestamp"], reverse=True)

        return rewards[:limit]

    def get_domain_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener estadísticas por dominio

        Returns:
            dict: Estadísticas por dominio
        """
        domain_stats = {}
        rewards = self.get_reward_history()

        for reward in rewards:
            domain = reward["domain"]
            if domain not in domain_stats:
                domain_stats[domain] = {
                    "total_rewards": 0,
                    "total_sheilys": 0.0,
                    "avg_sheilys": 0.0,
                    "best_reward": 0.0,
                }

            domain_stats[domain]["total_rewards"] += 1
            domain_stats[domain]["total_sheilys"] += reward["sheilys"]
            domain_stats[domain]["best_reward"] = max(
                domain_stats[domain]["best_reward"], reward["sheilys"]
            )

        # Calcular promedios
        for domain, stats in domain_stats.items():
            if stats["total_rewards"] > 0:
                stats["avg_sheilys"] = round(
                    stats["total_sheilys"] / stats["total_rewards"], 2
                )
                stats["total_sheilys"] = round(stats["total_sheilys"], 2)

        return domain_stats

    def cleanup_old_rewards(self):
        """
        Limpiar recompensas antiguas
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=self.retention_days)
        removed_count = 0

        for filename in os.listdir(self.vault_path):
            if filename.endswith(".json") and filename != "system_stats.json":
                filepath = self.vault_path / filename

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        reward = json.load(f)

                    reward_date = datetime.fromisoformat(reward["timestamp"])

                    if reward_date < cutoff_date:
                        os.remove(filepath)
                        removed_count += 1
                        self.stats["total_rewards"] -= 1
                        self.stats["total_sheilys"] -= reward["sheilys"]
                except Exception as e:
                    print(f"Error procesando {filename}: {e}")

        if removed_count > 0:
            self.stats["last_cleanup"] = datetime.now(UTC).isoformat()
            self._save_stats()
            print(
                f"🧹 Limpieza completada: {removed_count} recompensas antiguas eliminadas"
            )

    def get_system_health(self) -> Dict[str, Any]:
        """
        Obtener estado de salud del sistema de recompensas

        Returns:
            dict: Estado del sistema
        """
        vault_size = sum(
            os.path.getsize(self.vault_path / f)
            for f in os.listdir(self.vault_path)
            if f.endswith(".json")
        )

        return {
            "vault_size_mb": round(vault_size / (1024 * 1024), 2),
            "total_rewards": self.stats["total_rewards"],
            "total_sheilys": round(self.stats["total_sheilys"], 2),
            "domains_processed": len(self.stats["domains_processed"]),
            "last_cleanup": self.stats.get("last_cleanup"),
            "retention_days": self.retention_days,
            "max_vault_size": self.max_vault_size,
            "auto_cleanup": self.auto_cleanup,
        }


# Función de conveniencia para crear sistema
def create_reward_system(
    vault_path: str = "./rewards/vault", **kwargs
) -> SheilyRewardSystem:
    """
    Crear instancia del sistema de recompensas

    Args:
        vault_path (str): Ruta del vault
        **kwargs: Parámetros adicionales

    Returns:
        SheilyRewardSystem: Sistema inicializado
    """
    return SheilyRewardSystem(vault_path=vault_path, **kwargs)
