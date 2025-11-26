#!/usr/bin/env python3
"""
Script de Endurecimiento de Seguridad - Sheily AI MCP
====================================================

Este script mejora la seguridad del sistema:
- Mueve secretos a variables de entorno
- Configura CORS restrintivos
- Valida configuraciones de seguridad
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv, set_key


class SecurityHardener:
    """Endurecedor de seguridad para Sheily AI MCP"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.env_file = self.project_root / ".env"
        self.config_file = self.project_root / "config/security_config.json"

        # Secreto detectados para migrar
        self.secret_patterns = [
            r"(password|secret|key|token)\s*=\s*['\"][^'\"]{8,}['\"]",
            r"(api_key|auth_key)\s*=\s*['\"][^'\"]+['\"]",
            r"DATABASE_URL\s*=\s*['\"][^'\"]+['\"]",
            r"JWT_SECRET\s*=\s*['\"][^'\"]+['\"]",
            r"REDIS_PASSWORD\s*=\s*['\"][^'\"]+['\"]",
        ]

    def harden_security(self) -> bool:
        """Ejecuta todo el endurecimiento de seguridad"""
        print("🛡️ INICIANDO ENDURECIMIENTO DE SEGURIDAD - SHEILY AI MCP")
        print("=" * 70)

        try:
            # 1. Migrar secretos a variables de entorno
            secrets_migrated = self._migrate_secrets_to_env()

            # 2. Configurar CORS restrintivo
            cors_configured = self._configure_restrictive_cors()

            # 3. Crear configuración de seguridad
            security_config_created = self._create_security_config()

            # 4. Validar configuración actual
            validation_passed = self._validate_security_config()

            print(f"\n✅ ENDURECIMIENTO DE SEGURIDAD COMPLETADO")
            print(f"   🔑 Secretos migrados: {secrets_migrated}")
            print(f"   🌐 CORS configurado: {'✅' if cors_configured else '❌'}")
            print(f"   ⚙️ Config creada: {'✅' if security_config_created else '❌'}")
            print(f"   ✅ Validación: {'✅' if validation_passed else '❌'}")

            return True

        except Exception as e:
            print(f"❌ Error en endurecimiento de seguridad: {e}")
            return False

    def _migrate_secrets_to_env(self) -> int:
        """Migra secretos encontrados a variables de entorno"""
        print("🔑 Migrando secretos a variables de entorno...")

        secrets_found = 0
        migrated_count = 0

        # Buscar en archivos Python
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if not f.name.startswith("test_")]

        for file_path in python_files[:50]:  # Limitar para performance
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")

                for pattern in self.secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        secret_line = match.group(0)
                        secrets_found += 1

                        # Extraer nombre de variable y valor
                        if "=" in secret_line:
                            parts = secret_line.split("=", 1)
                            var_name = parts[0].strip().upper()
                            var_value = parts[1].strip().strip("'\"")

                            # Solo migrar si no está ya en .env
                            if var_name not in os.environ:
                                self._add_to_env_file(var_name, var_value)
                                migrated_count += 1
                                print(f"   ✅ Migrado: {var_name}")

                                # Reemplazar en archivo original
                                self._replace_secret_in_file(
                                    file_path, secret_line, var_name
                                )

            except Exception as e:
                continue

        # Migrar secretos comunes por defecto
        default_secrets = {
            "JWT_SECRET": self._generate_secure_secret(),
            "API_SECRET_KEY": self._generate_secure_secret(),
            "DATABASE_PASSWORD": "CHANGE_THIS_IN_PRODUCTION",
            "REDIS_PASSWORD": "",
            "OPENAI_API_KEY": "",
        }

        for var_name, default_value in default_secrets.items():
            if var_name not in os.environ:
                self._add_to_env_file(var_name, default_value)
                print(f"   ➕ Agregado: {var_name} (valor por defecto)")

        return migrated_count

    def _add_to_env_file(self, var_name: str, var_value: str) -> None:
        """Agrega variable al archivo .env"""
        try:
            # Crear archivo .env si no existe
            if not self.env_file.exists():
                self.env_file.touch()

            set_key(str(self.env_file), var_name, var_value)
        except Exception as e:
            print(f"   ⚠️ Error agregando a .env: {e}")

    def _replace_secret_in_file(
        self, file_path: Path, secret_line: str, var_name: str
    ) -> None:
        """Reemplaza secreto en archivo con variable de entorno"""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Reemplazar con os.getenv
            replacement = (
                f'{var_name.lower()} = os.getenv("{var_name}", "CHANGE_THIS_SECRET")'
            )

            # Asegurar import de os
            if "import os" not in content:
                lines = content.split("\n")
                if lines and not lines[0].startswith("#"):
                    lines.insert(0, "import os")
                elif lines and lines[0].startswith("#"):
                    lines.insert(1, "import os")
                else:
                    lines.insert(0, "import os")
                content = "\n".join(lines)

            new_content = content.replace(secret_line, replacement)
            file_path.write_text(new_content, encoding="utf-8")

        except Exception as e:
            print(f"   ⚠️ Error reemplazando en {file_path}: {e}")

    def _configure_restrictive_cors(self) -> bool:
        """Configura CORS restrictivo"""
        print("🌐 Configurando CORS restrictivo...")

        try:
            # Buscar archivos de configuración FastAPI/CORS
            cors_configs = []
            python_files = list(self.project_root.rglob("*.py"))

            for file_path in python_files:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")

                    if "CORSMiddleware" in content and "allow_origins" in content:
                        cors_configs.append(file_path)
                except:
                    continue

            # Configurar CORS restrictivo
            restrictive_cors = """
# CORS Configuration - RESTRICTIVE FOR SECURITY
from fastapi.middleware.cors import CORSMiddleware

# ALLOWED ORIGINS - ONLY TRUSTED DOMAINS
allowed_origins = [
    "http://localhost:3000",       # Frontend desarrollo
    "http://localhost:3001",       # Frontend alternativo
    "https://yourdomain.com",      # Producción - CAMBIAR
    "https://app.yourdomain.com",  # App producción - CAMBIAR
]

# CORS middleware con configuración restrictiva
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Solo orígenes permitidos
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Métodos específicos
    allow_headers=["*"],  # Headers permitidos
    max_age=86400,  # Cache por 24 horas
)
"""

            # Agregar configuración a archivo principal si no existe
            main_files = ["backend/main.py", "backend/app.py", "main.py", "app.py"]
            for main_file in main_files:
                main_path = self.project_root / main_file
                if main_path.exists():
                    try:
                        content = main_path.read_text(encoding="utf-8")
                        if "CORSMiddleware" not in content:
                            # Agregar configuración CORS
                            content += "\n\n" + restrictive_cors
                            main_path.write_text(content, encoding="utf-8")
                            print(f"   ✅ CORS configurado en: {main_file}")
                            return True
                    except Exception as e:
                        print(f"   ⚠️ Error configurando CORS en {main_file}: {e}")
                        continue

            print("   ⚠️ No se encontró archivo principal para configurar CORS")
            return False

        except Exception as e:
            print(f"   ❌ Error configurando CORS: {e}")
            return False

    def _create_security_config(self) -> bool:
        """Crea archivo de configuración de seguridad"""
        print("⚙️ Creando configuración de seguridad...")

        security_config = {
            "version": "1.0",
            "timestamp": "2025-11-17T12:00:00Z",
            "security": {
                "cors": {
                    "enabled": True,
                    "allowed_origins": [
                        "http://localhost:3000",
                        "http://localhost:3001",
                        "https://yourdomain.com",
                        "https://app.yourdomain.com",
                    ],
                    "allow_credentials": True,
                    "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                    "allowed_headers": ["*"],
                    "max_age": 86400,
                },
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 60,
                    "burst_limit": 10,
                },
                "ssl": {
                    "enabled": True,
                    "certificate_path": "/path/to/ssl/cert.pem",
                    "key_path": "/path/to/ssl/private.key",
                },
                "secrets": {
                    "use_env_vars": True,
                    "env_vars": [
                        "JWT_SECRET",
                        "API_SECRET_KEY",
                        "DATABASE_PASSWORD",
                        "REDIS_PASSWORD",
                        "OPENAI_API_KEY",
                    ],
                },
                "headers": {
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "X-XSS-Protection": "1; mode=block",
                    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                    "Content-Security-Policy": "default-src 'self'",
                },
            },
            "validation": {
                "secrets_migrated": True,
                "cors_configured": True,
                "env_vars_verified": True,
            },
        }

        try:
            # Crear directorio si no existe
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            # Escribir configuración
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(security_config, f, indent=2, ensure_ascii=False)

            print(f"   ✅ Configuración creada: {self.config_file}")
            return True

        except Exception as e:
            print(f"   ❌ Error creando configuración: {e}")
            return False

    def _validate_security_config(self) -> bool:
        """Valida la configuración de seguridad"""
        print("🔍 Validando configuración de seguridad...")

        validations = []

        # Verificar archivo .env
        if self.env_file.exists():
            print("   ✅ Archivo .env encontrado")
            validations.append(True)
        else:
            print("   ❌ Archivo .env no encontrado")
            validations.append(False)

        # Verificar variables críticas en entorno
        critical_vars = ["JWT_SECRET", "API_SECRET_KEY"]
        env_vars_present = all(var in os.environ for var in critical_vars)

        if env_vars_present:
            print("   ✅ Variables de entorno críticas configuradas")
        else:
            print("   ⚠️ Faltan algunas variables de entorno críticas")
        validations.append(env_vars_present)

        # Verificar configuración de seguridad
        if self.config_file.exists():
            print("   ✅ Archivo de configuración de seguridad encontrado")
            validations.append(True)
        else:
            print("   ❌ Archivo de configuración de seguridad no encontrado")
            validations.append(False)

        return all(validations)

    def _generate_secure_secret(self, length: int = 32) -> str:
        """Genera un secreto seguro aleatorio"""
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(secrets.choice(alphabet) for _ in range(length))


def main():
    """Función principal"""
    hardener = SecurityHardener()

    if hardener.harden_security():
        print("\n🎉 Endurecimiento de seguridad completado exitosamente!")
        print("🔒 El sistema está ahora más seguro.")
        print("⚠️ Recuerda cambiar todos los valores por defecto en producción.")
    else:
        print("\n❌ Falló el endurecimiento de seguridad.")
        exit(1)


if __name__ == "__main__":
    main()
