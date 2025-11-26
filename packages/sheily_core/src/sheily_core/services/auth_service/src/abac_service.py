#!/usr/bin/env python3
"""
ABAC Service - Attribute-Based Access Control Service

Este módulo implementa control de acceso basado en atributos (ABAC) con capacidades de:
- Políticas basadas en atributos de usuario, recurso y entorno
- Evaluación dinámica de permisos
- Gestión flexible de políticas
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ABACService:
    """Servicio de control de acceso basado en atributos"""

    def __init__(self):
        """Inicializar servicio ABAC"""
        self.policies = []
        self.initialized = True
        logger.info("ABACService inicializado")

    def add_policy(self, policy: Dict[str, Any]) -> bool:
        """Añadir una nueva política ABAC"""
        required_fields = ["name", "effect", "conditions"]
        for field in required_fields:
            if field not in policy:
                return False

        policy["id"] = len(self.policies)
        self.policies.append(policy)
        logger.info(f"Política ABAC añadida: {policy['name']}")
        return True

    def evaluate_policy(
        self,
        subject: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Evaluar si una acción está permitida según las políticas ABAC"""
        context = context or {}

        # Atributos del sujeto (simulados)
        subject_attrs = self._get_subject_attributes(subject)

        # Atributos del recurso (simulados)
        resource_attrs = self._get_resource_attributes(resource)

        # Atributos del entorno
        environment_attrs = {
            "time": context.get("time", "business_hours"),
            "location": context.get("location", "internal"),
            "device": context.get("device", "trusted"),
            "action": action,
        }

        # Evaluar todas las políticas aplicables
        for policy in self.policies:
            if self._policy_applies(
                policy, subject_attrs, resource_attrs, environment_attrs
            ):
                effect = policy["effect"]
                if effect == "allow":
                    logger.info(f"Acceso permitido por política: {policy['name']}")
                    return True
                elif effect == "deny":
                    logger.info(f"Acceso denegado por política: {policy['name']}")
                    return False

        # Por defecto, denegar acceso
        logger.info("Acceso denegado: ninguna política aplicable")
        return False

    def _policy_applies(
        self,
        policy: Dict[str, Any],
        subject_attrs: Dict[str, Any],
        resource_attrs: Dict[str, Any],
        environment_attrs: Dict[str, Any],
    ) -> bool:
        """Verificar si una política se aplica a la solicitud"""
        conditions = policy.get("conditions", {})

        # Verificar condiciones del sujeto
        subject_conditions = conditions.get("subject", {})
        if not self._check_conditions(subject_attrs, subject_conditions):
            return False

        # Verificar condiciones del recurso
        resource_conditions = conditions.get("resource", {})
        if not self._check_conditions(resource_attrs, resource_conditions):
            return False

        # Verificar condiciones del entorno
        environment_conditions = conditions.get("environment", {})
        if not self._check_conditions(environment_attrs, environment_conditions):
            return False

        return True

    def _check_conditions(
        self, attributes: Dict[str, Any], conditions: Dict[str, Any]
    ) -> bool:
        """Verificar si los atributos cumplen las condiciones"""
        for attr_name, expected_value in conditions.items():
            if attr_name not in attributes:
                return False

            actual_value = attributes[attr_name]

            # Soporte para diferentes tipos de comparación
            if isinstance(expected_value, dict):
                # Condiciones complejas
                if "operator" in expected_value:
                    operator = expected_value["operator"]
                    value = expected_value["value"]

                    if operator == "equals" and actual_value != value:
                        return False
                    elif operator == "not_equals" and actual_value == value:
                        return False
                    elif operator == "in" and actual_value not in value:
                        return False
                    elif operator == "not_in" and actual_value in value:
                        return False
                else:
                    return False
            else:
                # Comparación simple
                if actual_value != expected_value:
                    return False

        return True

    def _get_subject_attributes(self, subject: str) -> Dict[str, Any]:
        """Obtener atributos del sujeto (simulados)"""
        # En producción, esto vendría de una base de datos de usuarios
        return {
            "id": subject,
            "role": "user",  # admin, user, guest
            "department": "engineering",
            "clearance_level": "confidential",
            "status": "active",
        }

    def _get_resource_attributes(self, resource: str) -> Dict[str, Any]:
        """Obtener atributos del recurso (simulados)"""
        # En producción, esto vendría de una base de datos de recursos
        return {
            "id": resource,
            "type": "document",  # document, api, system
            "classification": "internal",
            "owner": "admin",
            "sensitivity": "medium",
        }

    def remove_policy(self, policy_id: int) -> bool:
        """Eliminar una política"""
        if 0 <= policy_id < len(self.policies):
            removed_policy = self.policies.pop(policy_id)
            logger.info(f"Política eliminada: {removed_policy['name']}")
            return True
        return False

    def list_policies(self) -> List[Dict[str, Any]]:
        """Listar todas las políticas"""
        return self.policies.copy()

    def get_policy(self, policy_id: int) -> Optional[Dict[str, Any]]:
        """Obtener una política específica"""
        if 0 <= policy_id < len(self.policies):
            return self.policies[policy_id].copy()
        return None

    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        return {
            "initialized": self.initialized,
            "total_policies": len(self.policies),
            "capabilities": [
                "policy_evaluation",
                "attribute_based_access",
                "dynamic_permissions",
            ],
            "version": "1.0.0",
        }
