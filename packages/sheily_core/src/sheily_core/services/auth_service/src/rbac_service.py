#!/usr/bin/env python3
"""
RBAC Service - Role-Based Access Control Service

Este módulo implementa control de acceso basado en roles (RBAC) con capacidades de:
- Gestión de roles y permisos
- Asignación de roles a usuarios
- Verificación de permisos
"""

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RBACService:
    """Servicio de control de acceso basado en roles"""

    def __init__(self):
        """Inicializar servicio RBAC"""
        self.roles = {}  # role_name -> permissions
        self.user_roles = {}  # user_id -> set of roles
        self.role_hierarchy = {}  # role -> parent roles

        # Roles por defecto
        self._initialize_default_roles()

        self.initialized = True
        logger.info("RBACService inicializado")

    def _initialize_default_roles(self):
        """Inicializar roles por defecto"""
        self.roles = {
            "admin": {"*"},  # Todos los permisos
            "manager": {"read", "write", "manage_users"},
            "user": {"read", "write"},
            "guest": {"read"},
        }

        self.role_hierarchy = {
            "admin": set(),
            "manager": {"user"},
            "user": {"guest"},
            "guest": set(),
        }

    def add_role(self, role_name: str, permissions: Set[str]) -> bool:
        """Añadir un nuevo rol"""
        if role_name in self.roles:
            return False

        self.roles[role_name] = permissions.copy()
        self.role_hierarchy[role_name] = set()
        logger.info(f"Rol añadido: {role_name}")
        return True

    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Asignar un rol a un usuario"""
        if role_name not in self.roles:
            return False

        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()

        self.user_roles[user_id].add(role_name)
        logger.info(f"Rol {role_name} asignado a usuario {user_id}")
        return True

    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revocar un rol de un usuario"""
        if user_id in self.user_roles and role_name in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role_name)
            logger.info(f"Rol {role_name} revocado de usuario {user_id}")
            return True
        return False

    def check_permission(self, user_id: str, role: str, permission: str) -> bool:
        """Verificar si un usuario tiene un permiso"""
        if user_id not in self.user_roles:
            return False

        user_roles = self.user_roles[user_id]

        # Verificar roles directos e heredados
        all_user_roles = set()
        for user_role in user_roles:
            all_user_roles.add(user_role)
            all_user_roles.update(self._get_parent_roles(user_role))

        # Verificar permisos
        for user_role in all_user_roles:
            if user_role in self.roles:
                role_permissions = self.roles[user_role]
                if "*" in role_permissions or permission in role_permissions:
                    return True

        return False

    def _get_parent_roles(self, role: str) -> Set[str]:
        """Obtener roles padre (herencia)"""
        if role not in self.role_hierarchy:
            return set()
        return self.role_hierarchy[role]

    def add_role_relationship(self, child_role: str, parent_role: str) -> bool:
        """Añadir relación de herencia entre roles"""
        if child_role not in self.roles or parent_role not in self.roles:
            return False

        if child_role not in self.role_hierarchy:
            self.role_hierarchy[child_role] = set()

        self.role_hierarchy[child_role].add(parent_role)
        logger.info(f"Relación añadida: {child_role} -> {parent_role}")
        return True

    def get_user_roles(self, user_id: str) -> Set[str]:
        """Obtener roles de un usuario"""
        return self.user_roles.get(user_id, set()).copy()

    def get_role_permissions(self, role_name: str) -> Set[str]:
        """Obtener permisos de un rol"""
        return self.roles.get(role_name, set()).copy()

    def list_roles(self) -> List[str]:
        """Listar todos los roles"""
        return list(self.roles.keys())

    def list_users(self) -> List[str]:
        """Listar todos los usuarios con roles"""
        return list(self.user_roles.keys())

    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        total_users = len(self.user_roles)
        total_roles = len(self.roles)
        total_assignments = sum(len(roles) for roles in self.user_roles.values())

        return {
            "initialized": self.initialized,
            "total_users": total_users,
            "total_roles": total_roles,
            "total_assignments": total_assignments,
            "capabilities": [
                "role_management",
                "permission_checking",
                "role_inheritance",
            ],
            "version": "1.0.0",
        }
