#!/usr/bin/env python3
"""
Enterprise Security Bootstrap for Sheily AI Applications
========================================================

Script de inicialización que configura:
- Autenticación con HashiCorp Vault
- Obtención de secrets desde Vault
- Configuración de mTLS
- Validación de comunicaciones seguras

Este script debe ejecutarse al inicio de cada aplicación enterprise.
"""

import asyncio
import json
import logging
import os
import ssl
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

# Import local vault client
sys.path.append(str(Path(__file__).parent))
from vault_client import VaultClient, VaultConfig


class EnterpriseSecurityBootstrap:
    """
    Bootstrap de seguridad enterprise para aplicaciones Sheily AI
    Configura Vault, mTLS y validación de comunicaciones seguras
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vault_client: Optional[VaultClient] = None
        self.mtls_context: Optional[ssl.SSLContext] = None
        self.secrets: Dict[str, Any] = {}

    async def initialize_security(self) -> Dict[str, Any]:
        """
        Inicializar completamente la seguridad enterprise

        Returns:
            Dict con configuración de seguridad
        """
        self.logger.info("Initializing enterprise security bootstrap...")

        security_config = {
            "vault_connected": False,
            "mtls_configured": False,
            "secrets_loaded": False,
            "security_validation": False,
            "errors": [],
        }

        try:
            # 1. Inicializar conexión con Vault
            await self._initialize_vault()
            security_config["vault_connected"] = True
            self.logger.info("✓ Vault connection established")

            # 2. Cargar secrets críticos
            await self._load_critical_secrets()
            security_config["secrets_loaded"] = True
            self.logger.info("✓ Critical secrets loaded")

            # 3. Configurar mTLS
            await self._configure_mtls()
            security_config["mtls_configured"] = True
            self.logger.info("✓ mTLS configured")

            # 4. Validar configuración de seguridad
            await self._validate_security_configuration()
            security_config["security_validation"] = True
            self.logger.info("✓ Security configuration validated")

            self.logger.info("🎉 Enterprise security bootstrap completed successfully!")

        except Exception as e:
            error_msg = f"Security bootstrap failed: {str(e)}"
            self.logger.error(error_msg)
            security_config["errors"].append(error_msg)

            # En caso de error crítico, intentar modo degradado
            await self._enter_degraded_mode()

        return security_config

    async def _initialize_vault(self):
        """Inicializar conexión con HashiCorp Vault"""
        try:
            vault_config = VaultConfig()

            # Override con variables de entorno si existen
            if os.getenv("VAULT_ADDR"):
                vault_config.address = os.getenv("VAULT_ADDR")
            if os.getenv("VAULT_ROLE"):
                vault_config.role = os.getenv("VAULT_ROLE")

            self.vault_client = VaultClient(vault_config)
            await self.vault_client.__aenter__()

            # Verificar conectividad
            test_secret = await self.vault_client.get_secret("sheily-ai/test")
            self.logger.debug("Vault connectivity test successful")

        except Exception as e:
            raise Exception(f"Failed to initialize Vault connection: {e}")

    async def _load_critical_secrets(self):
        """Cargar secrets críticos para funcionamiento de la aplicación"""
        critical_secrets = [
            "sheily-ai/database",
            "sheily-ai/api",
            "sheily-ai/redis",
            "sheily-ai/external",
        ]

        for secret_path in critical_secrets:
            try:
                secret = await self.vault_client.get_secret(secret_path)
                self.secrets[secret_path] = secret.data

                # Set environment variables for legacy compatibility
                for key, value in secret.data.items():
                    env_key = f"{secret_path.replace('/', '_').replace('-', '_').upper()}_{key.upper()}"
                    os.environ[env_key] = str(value)

            except Exception as e:
                self.logger.warning(f"Failed to load secret {secret_path}: {e}")
                # Continue with other secrets

        # Verificar que tenemos al menos los secrets básicos
        if not any("database" in key for key in self.secrets.keys()):
            raise Exception("Critical database secrets not available")

    async def _configure_mtls(self):
        """Configurar mTLS para comunicaciones seguras"""
        try:
            # En Kubernetes con Istio, los certificados son inyectados automáticamente
            # Este código es para configuración manual o validación

            # Verificar que tenemos acceso a certificados mTLS
            cert_paths = [
                "/etc/istio/certs/cert-chain.pem",
                "/etc/istio/certs/key.pem",
                "/etc/istio/certs/root-cert.pem",
            ]

            certs_available = all(Path(path).exists() for path in cert_paths)

            if certs_available:
                # Configurar SSL context para mTLS
                self.mtls_context = ssl.create_default_context(
                    purpose=ssl.Purpose.CLIENT_AUTH
                )
                self.mtls_context.load_cert_chain(
                    certfile=cert_paths[0], keyfile=cert_paths[1]
                )
                self.mtls_context.load_verify_locations(cafile=cert_paths[2])
                self.mtls_context.verify_mode = ssl.CERT_REQUIRED
                self.mtls_context.check_hostname = True

                self.logger.info("mTLS certificates configured successfully")
            else:
                self.logger.warning(
                    "mTLS certificates not found - running in development mode"
                )
                # En desarrollo, crear contexto SSL básico
                self.mtls_context = ssl.create_default_context()

        except Exception as e:
            raise Exception(f"Failed to configure mTLS: {e}")

    async def _validate_security_configuration(self):
        """Validar que toda la configuración de seguridad está correcta"""
        validation_errors = []

        # 1. Validar Vault connectivity
        try:
            test_secret = await self.vault_client.get_secret("sheily-ai/test")
        except:
            validation_errors.append("Vault connectivity validation failed")

        # 2. Validar secrets críticos
        required_secrets = ["database", "api", "redis"]
        for secret_type in required_secrets:
            if not any(secret_type in key for key in self.secrets.keys()):
                validation_errors.append(
                    f"Required secret type '{secret_type}' not loaded"
                )

        # 3. Validar mTLS configuration
        if not self.mtls_context:
            validation_errors.append("mTLS context not configured")

        # 4. Validar configuración de aplicación
        required_env_vars = ["JWT_SECRET", "DATABASE_URL", "REDIS_URL"]

        for env_var in required_env_vars:
            if not os.getenv(env_var):
                validation_errors.append(
                    f"Required environment variable '{env_var}' not set"
                )

        if validation_errors:
            raise Exception(
                f"Security validation failed: {', '.join(validation_errors)}"
            )

        self.logger.info("Security configuration validation passed")

    async def _enter_degraded_mode(self):
        """Entrar en modo degradado cuando falla la configuración completa"""
        self.logger.warning("Entering degraded security mode")

        # Configurar valores por defecto para desarrollo/testing
        os.environ.setdefault("JWT_SECRET", "development_jwt_secret_not_secure")
        os.environ.setdefault("DATABASE_URL", "sqlite:///development.db")
        os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

        # Crear contexto SSL básico
        self.mtls_context = ssl.create_default_context()
        self.mtls_context.check_hostname = False
        self.mtls_context.verify_mode = ssl.CERT_NONE

        self.logger.warning("Running in degraded mode - NOT SECURE FOR PRODUCTION")

    async def get_secret(self, path: str, key: Optional[str] = None) -> Any:
        """Obtener secreto desde Vault (con cache local)"""
        if not self.vault_client:
            raise Exception("Vault client not initialized")

        if path in self.secrets:
            secret_data = self.secrets[path]
        else:
            secret = await self.vault_client.get_secret(path)
            secret_data = secret.data
            self.secrets[path] = secret_data

        if key:
            return secret_data.get(key)
        return secret_data

    def get_mtls_context(self) -> ssl.SSLContext:
        """Obtener contexto SSL para mTLS"""
        if not self.mtls_context:
            raise Exception("mTLS context not configured")
        return self.mtls_context

    async def rotate_database_credentials(self) -> Dict[str, str]:
        """Rotar credenciales de base de datos"""
        if not self.vault_client:
            raise Exception("Vault client not initialized")

        return await self.vault_client.rotate_credentials("sheily-ai-app")


# Instancia global para la aplicación
security_bootstrap: Optional[EnterpriseSecurityBootstrap] = None


async def initialize_enterprise_security() -> EnterpriseSecurityBootstrap:
    """
    Inicializar seguridad enterprise para la aplicación
    Debe llamarse al inicio de cada aplicación Sheily AI
    """
    global security_bootstrap

    if security_bootstrap is None:
        security_bootstrap = EnterpriseSecurityBootstrap()
        await security_bootstrap.initialize_security()

    return security_bootstrap


def get_security_bootstrap() -> EnterpriseSecurityBootstrap:
    """Obtener instancia global del bootstrap de seguridad"""
    if security_bootstrap is None:
        raise Exception(
            "Security bootstrap not initialized. Call initialize_enterprise_security() first"
        )
    return security_bootstrap


# Función de compatibilidad para aplicaciones legacy
async def get_vault_secret(path: str, key: Optional[str] = None) -> Any:
    """Función de compatibilidad para obtener secrets"""
    bootstrap = await initialize_enterprise_security()
    return await bootstrap.get_secret(path, key)


if __name__ == "__main__":
    # Demo de inicialización de seguridad
    async def demo():
        print("🔐 Initializing Enterprise Security for Sheily AI...")

        bootstrap = await initialize_enterprise_security()

        # Obtener un secreto de ejemplo
        try:
            db_config = await bootstrap.get_secret("sheily-ai/database")
            print(f"✓ Database configuration loaded: {list(db_config.keys())}")
        except Exception as e:
            print(f"⚠️ Could not load database config (expected in demo): {e}")

        print("✅ Enterprise Security Bootstrap Complete!")

    asyncio.run(demo())
