#!/usr/bin/env python3
"""
Vault Integration Client for Sheily AI
======================================

Cliente enterprise para integración con HashiCorp Vault.
Proporciona gestión segura de secrets con caching y rotación automática.
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import jwt
from cryptography.hazmat.primitives import serialization


@dataclass
class VaultConfig:
    """Configuración de Vault"""

    address: str = "https://vault.sheily-ai.com:8200"
    mount_point: str = "secret"
    kv_version: str = "v2"
    auth_method: str = "kubernetes"  # kubernetes, jwt, token
    role: str = "sheily-ai-api"
    jwt_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    ca_cert_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    cache_ttl: int = 300  # 5 minutes
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class VaultSecret:
    """Secreto de Vault con metadata"""

    path: str
    data: Dict[str, Any]
    version: int = 0
    created_time: str = ""
    deletion_time: str = ""
    destroyed: bool = False
    cached_at: float = field(default_factory=time.time)


class VaultClient:
    """
    Cliente enterprise para HashiCorp Vault
    Incluye caching, rotación automática y manejo de errores robusto
    """

    def __init__(self, config: Optional[VaultConfig] = None):
        self.config = config or VaultConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._token: Optional[str] = None
        self._token_expires: float = 0
        self._cache: Dict[str, VaultSecret] = {}
        self._logger = logging.getLogger(__name__)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self):
        """Inicializar cliente de Vault"""
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(verify_ssl=True, limit=10, limit_per_host=5)
        )

        # Autenticar según el método configurado
        if self.config.auth_method == "kubernetes":
            await self._authenticate_kubernetes()
        elif self.config.auth_method == "jwt":
            await self._authenticate_jwt()
        elif self.config.auth_method == "token":
            self._token = os.getenv("VAULT_TOKEN")

        if not self._token:
            raise ValueError(
                f"Failed to authenticate with Vault using {self.config.auth_method}"
            )

        self._logger.info("Vault client initialized successfully")

    async def close(self):
        """Cerrar conexiones"""
        if self._session:
            await self._session.close()

    async def get_secret(self, path: str, force_refresh: bool = False) -> VaultSecret:
        """
        Obtener secreto de Vault con caching inteligente

        Args:
            path: Path del secreto (sin mount point)
            force_refresh: Forzar refresh desde Vault

        Returns:
            VaultSecret: Secreto con metadata
        """
        full_path = f"{self.config.mount_point}/data/{path}"

        # Check cache first
        if not force_refresh and path in self._cache:
            cached = self._cache[path]
            if time.time() - cached.cached_at < self.config.cache_ttl:
                self._logger.debug(f"Returning cached secret for {path}")
                return cached

        # Fetch from Vault
        for attempt in range(self.config.retry_attempts):
            try:
                secret = await self._fetch_secret_from_vault(full_path)
                self._cache[path] = secret
                return secret

            except Exception as e:
                self._logger.warning(f"Attempt {attempt + 1} failed for {path}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))

        raise Exception(
            f"Failed to fetch secret {path} after {self.config.retry_attempts} attempts"
        )

    async def put_secret(self, path: str, data: Dict[str, Any]) -> bool:
        """
        Almacenar secreto en Vault

        Args:
            path: Path del secreto
            data: Datos del secreto

        Returns:
            bool: True si se almacenó correctamente
        """
        full_path = f"{self.config.mount_point}/data/{path}"

        payload = {"data": data}

        for attempt in range(self.config.retry_attempts):
            try:
                async with self._session.post(
                    f"{self.config.address}/v1/{full_path}",
                    json=payload,
                    headers={"X-Vault-Token": self._token},
                ) as response:
                    if response.status == 200:
                        # Invalidate cache
                        self._cache.pop(path, None)
                        self._logger.info(f"Secret stored successfully at {path}")
                        return True
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Vault API error: {response.status} - {error_text}"
                        )

            except Exception as e:
                self._logger.warning(
                    f"Attempt {attempt + 1} failed to store {path}: {e}"
                )
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))

        return False

    async def rotate_credentials(self, database_role: str) -> Dict[str, str]:
        """
        Rotar credenciales de base de datos dinámicamente

        Args:
            database_role: Role de base de datos en Vault

        Returns:
            Dict con nuevas credenciales
        """
        for attempt in range(self.config.retry_attempts):
            try:
                async with self._session.get(
                    f"{self.config.address}/v1/database/creds/{database_role}",
                    headers={"X-Vault-Token": self._token},
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        creds = data.get("data", {})
                        self._logger.info(
                            f"Credentials rotated for role {database_role}"
                        )
                        return creds
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to rotate credentials: {error_text}")

            except Exception as e:
                self._logger.warning(
                    f"Attempt {attempt + 1} failed to rotate credentials: {e}"
                )
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))

        raise Exception(f"Failed to rotate credentials for {database_role}")

    async def _authenticate_kubernetes(self):
        """Autenticación usando service account de Kubernetes"""
        try:
            # Leer JWT token
            with open(self.config.jwt_path, "r") as f:
                jwt_token = f.read().strip()

            # Leer CA certificate
            with open(self.config.ca_cert_path, "r") as f:
                ca_cert = f.read()

            # Autenticar con Vault
            payload = {"role": self.config.role, "jwt": jwt_token}

            async with self._session.post(
                f"{self.config.address}/v1/auth/kubernetes/login", json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self._token = data["auth"]["client_token"]
                    self._token_expires = time.time() + data["auth"]["lease_duration"]
                    self._logger.info(
                        "Authenticated with Vault using Kubernetes service account"
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"Kubernetes auth failed: {error_text}")

        except Exception as e:
            raise Exception(f"Kubernetes authentication failed: {e}")

    async def _authenticate_jwt(self):
        """Autenticación usando JWT (para servicios externos)"""
        try:
            import time

            import jwt

            # Leer JWT token de config o environment
            jwt_token = self.config.jwt_token or os.getenv("VAULT_JWT_TOKEN")
            if not jwt_token:
                raise Exception("JWT token not provided for Vault authentication")

            # Autenticar con Vault
            payload = {"role": self.config.role, "jwt": jwt_token}

            async with self._session.post(
                f"{self.config.address}/v1/auth/jwt/login", json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self._token = data["auth"]["client_token"]
                    self._token_expires = time.time() + data["auth"]["lease_duration"]
                    self._logger.info("Authenticated with Vault using JWT")
                else:
                    error_text = await response.text()
                    raise Exception(f"JWT auth failed: {error_text}")

        except Exception as e:
            raise Exception(f"JWT authentication failed: {e}")

    async def _fetch_secret_from_vault(self, path: str) -> VaultSecret:
        """Fetch secreto desde Vault API"""
        async with self._session.get(
            f"{self.config.address}/v1/{path}", headers={"X-Vault-Token": self._token}
        ) as response:
            if response.status == 200:
                data = await response.json()

                if self.config.kv_version == "v2":
                    secret_data = data.get("data", {}).get("data", {})
                    metadata = data.get("data", {}).get("metadata", {})
                else:  # v1
                    secret_data = data.get("data", {})
                    metadata = {}

                secret = VaultSecret(
                    path=path,
                    data=secret_data,
                    version=metadata.get("version", 0),
                    created_time=metadata.get("created_time", ""),
                    deletion_time=metadata.get("deletion_time", ""),
                    destroyed=metadata.get("destroyed", False),
                )

                return secret
            else:
                error_text = await response.text()
                raise Exception(
                    f"Failed to fetch secret: {response.status} - {error_text}"
                )


# Función de utilidad para aplicaciones
async def get_vault_secret(path: str, key: Optional[str] = None) -> Any:
    """
    Función de utilidad para obtener secretos de Vault

    Args:
        path: Path del secreto
        key: Key específica dentro del secreto (opcional)

    Returns:
        Secreto o valor específico
    """
    config = VaultConfig()
    async with VaultClient(config) as client:
        secret = await client.get_secret(path)

        if key:
            return secret.data.get(key)
        else:
            return secret.data


# Función para rotar credenciales de BD
async def rotate_database_credentials(role: str = "sheily-ai-app") -> Dict[str, str]:
    """Rotar credenciales de base de datos"""
    config = VaultConfig()
    async with VaultClient(config) as client:
        return await client.rotate_credentials(role)


if __name__ == "__main__":
    # Demo básico
    async def demo():
        async with VaultClient() as client:
            # Obtener secreto de base de datos
            db_secret = await client.get_secret("sheily-ai/database")
            print(f"Database config: {db_secret.data}")

            # Rotar credenciales
            new_creds = await client.rotate_credentials("sheily-ai-app")
            print(f"New credentials: {new_creds}")

    asyncio.run(demo())
