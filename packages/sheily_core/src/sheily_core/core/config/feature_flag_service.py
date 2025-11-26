#!/usr/bin/env python3
"""
Feature Flag Service - Servicio de Feature Flags

Este módulo implementa gestión de feature flags con capacidades de:
- Activación/desactivación de características
- Configuración por usuario
- Grupos de características
- Métricas de uso
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeatureFlagService:
    """Servicio de feature flags"""

    def __init__(self):
        """Inicializar servicio de feature flags"""
        self.flags = {}  # flag_name -> config
        self.user_overrides = {}  # user_id -> {flag_name -> enabled}
        self.groups = {}  # group_name -> list of flags

        # Flags por defecto
        self._initialize_default_flags()

        self.initialized = True
        logger.info("FeatureFlagService inicializado")

    def _initialize_default_flags(self):
        """Inicializar flags por defecto"""
        self.flags = {
            "advanced_ai": {
                "enabled": True,
                "description": "Características avanzadas de IA",
                "rollout_percentage": 100,
                "created_at": time.time(),
            },
            "experimental_features": {
                "enabled": False,
                "description": "Características experimentales",
                "rollout_percentage": 10,
                "created_at": time.time(),
            },
            "beta_testing": {
                "enabled": True,
                "description": "Funcionalidades en beta",
                "rollout_percentage": 50,
                "created_at": time.time(),
            },
        }

    def create_flag(self, flag_name: str, config: Dict[str, Any]) -> bool:
        """Crear un nuevo feature flag"""
        if flag_name in self.flags:
            return False

        config["created_at"] = time.time()
        self.flags[flag_name] = config
        logger.info(f"Feature flag creado: {flag_name}")
        return True

    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Verificar si un flag está habilitado para un usuario"""
        if flag_name not in self.flags:
            return False

        flag_config = self.flags[flag_name]

        # Verificar override de usuario
        if (
            user_id
            and user_id in self.user_overrides
            and flag_name in self.user_overrides[user_id]
        ):
            return self.user_overrides[user_id][flag_name]

        # Verificar si está habilitado globalmente
        if not flag_config.get("enabled", False):
            return False

        # Verificar porcentaje de rollout
        rollout_percentage = flag_config.get("rollout_percentage", 100)
        if rollout_percentage < 100:
            # Usar user_id para determinar si está en el rollout
            if user_id:
                user_hash = hash(user_id) % 100
                if user_hash >= rollout_percentage:
                    return False

        # Verificar condiciones contextuales
        if context and "conditions" in flag_config:
            if not self._check_conditions(context, flag_config["conditions"]):
                return False

        return True

    def _check_conditions(
        self, context: Dict[str, Any], conditions: Dict[str, Any]
    ) -> bool:
        """Verificar condiciones contextuales"""
        for key, expected_value in conditions.items():
            if key not in context:
                return False

            actual_value = context[key]
            if isinstance(expected_value, dict):
                # Condiciones complejas
                if "operator" in expected_value:
                    operator = expected_value["operator"]
                    value = expected_value["value"]

                    if operator == "equals" and actual_value != value:
                        return False
                    elif operator == "in" and actual_value not in value:
                        return False
                else:
                    return False
            else:
                # Comparación simple
                if actual_value != expected_value:
                    return False

        return True

    def set_user_override(self, user_id: str, flag_name: str, enabled: bool) -> bool:
        """Establecer override para un usuario específico"""
        if flag_name not in self.flags:
            return False

        if user_id not in self.user_overrides:
            self.user_overrides[user_id] = {}

        self.user_overrides[user_id][flag_name] = enabled
        logger.info(f"Override establecido para {user_id}: {flag_name} = {enabled}")
        return True

    def update_flag_config(self, flag_name: str, config: Dict[str, Any]) -> bool:
        """Actualizar configuración de un flag"""
        if flag_name not in self.flags:
            return False

        # Mantener created_at original
        original_created = self.flags[flag_name].get("created_at")
        config["created_at"] = original_created
        config["updated_at"] = time.time()

        self.flags[flag_name].update(config)
        logger.info(f"Flag actualizado: {flag_name}")
        return True

    def create_group(self, group_name: str, flag_names: List[str]) -> bool:
        """Crear un grupo de flags"""
        if group_name in self.groups:
            return False

        # Verificar que todos los flags existen
        for flag_name in flag_names:
            if flag_name not in self.flags:
                return False

        self.groups[group_name] = flag_names.copy()
        logger.info(f"Grupo creado: {group_name} con {len(flag_names)} flags")
        return True

    def get_group_flags(
        self, group_name: str, user_id: Optional[str] = None
    ) -> Dict[str, bool]:
        """Obtener estado de todos los flags en un grupo"""
        if group_name not in self.groups:
            return {}

        result = {}
        for flag_name in self.groups[group_name]:
            result[flag_name] = self.is_enabled(flag_name, user_id)

        return result

    def get_all_flags(self) -> Dict[str, Dict[str, Any]]:
        """Obtener todos los flags con su configuración"""
        return self.flags.copy()

    def get_flag_stats(self, flag_name: str) -> Dict[str, Any]:
        """Obtener estadísticas de un flag"""
        if flag_name not in self.flags:
            return {"error": "Flag no encontrado"}

        flag_config = self.flags[flag_name]

        # Contar usuarios con overrides
        override_count = sum(
            1
            for user_overrides in self.user_overrides.values()
            if flag_name in user_overrides
        )

        return {
            "flag_name": flag_name,
            "enabled": flag_config.get("enabled", False),
            "rollout_percentage": flag_config.get("rollout_percentage", 100),
            "user_overrides": override_count,
            "description": flag_config.get("description", ""),
            "created_at": flag_config.get("created_at", 0),
            "updated_at": flag_config.get("updated_at", 0),
        }

    def cleanup_old_overrides(self, max_age_days: int = 30) -> int:
        """Limpiar overrides antiguos (simulado)"""
        # En una implementación real, esto eliminaría overrides expirados
        cleaned_count = 0
        logger.info(f"Limpiados {cleaned_count} overrides antiguos")
        return cleaned_count

    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        total_flags = len(self.flags)
        enabled_flags = sum(
            1 for flag in self.flags.values() if flag.get("enabled", False)
        )
        total_groups = len(self.groups)
        total_overrides = sum(
            len(overrides) for overrides in self.user_overrides.values()
        )

        return {
            "initialized": self.initialized,
            "total_flags": total_flags,
            "enabled_flags": enabled_flags,
            "total_groups": total_groups,
            "total_user_overrides": total_overrides,
            "capabilities": [
                "flag_management",
                "user_overrides",
                "group_management",
                "rollout_control",
            ],
            "version": "1.0.0",
        }
