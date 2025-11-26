#!/usr/bin/env python3
"""
Sheily Enterprise Unified Configuration System
============================================

Sistema unificado de configuración y deployment para la arquitectura
híbrida que combina patterns de ii-agent con Sheily intelligence.

Características:
- Environment-specific configurations
- Deployment automation
- Configuration validation
- Hot-reload capabilities
- Security configuration
- Distributed configuration management
- Multi-environment support
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

from sheily_core.core.analytics.analytics_system import get_analytics_system
from sheily_core.core.events.event_system import (
    SheìlyEventType,
    get_event_stream,
    publish_event,
)

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Environments supported"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"


class ConfigType(str, Enum):
    """Configuration types"""

    SYSTEM = "system"
    DATABASE = "database"
    SECURITY = "security"
    ANALYTICS = "analytics"
    AGENTS = "agents"
    TOOLS = "tools"
    COMMUNICATION = "communication"
    MIDDLEWARE = "middleware"
    DEPLOYMENT = "deployment"


@dataclass
class DatabaseConfig:
    """Database configuration"""

    database_url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    echo_pool: bool = False
    enable_audit: bool = True
    backup_enabled: bool = True
    backup_interval_hours: int = 6


@dataclass
class SecurityConfig:
    """Security configuration"""

    secret_key: str
    jwt_secret: str
    jwt_expiration_hours: int = 24
    bcrypt_rounds: int = 12
    rate_limit_per_minute: int = 60
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    cors_origins: List[str] = field(default_factory=list)
    allowed_hosts: List[str] = field(default_factory=list)
    security_headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.secret_key:
            self.secret_key = secrets.token_urlsafe(32)
        if not self.jwt_secret:
            self.jwt_secret = secrets.token_urlsafe(32)


@dataclass
class AnalyticsConfig:
    """Analytics configuration"""

    enabled: bool = True
    metrics_retention_days: int = 30
    alert_retention_days: int = 90
    export_enabled: bool = True
    export_interval_hours: int = 24
    real_time_monitoring: bool = True
    performance_monitoring: bool = True
    billing_enabled: bool = False
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentsConfig:
    """Agents configuration"""

    max_agents: int = 100
    agent_timeout_seconds: int = 300
    heartbeat_interval_seconds: int = 30
    auto_restart_failed: bool = True
    load_balancing_enabled: bool = True
    agent_coordination_enabled: bool = True
    neural_network_config: Dict[str, Any] = field(default_factory=dict)
    specialized_agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.neural_network_config:
            self.neural_network_config = {
                "hidden_layers": [512, 256, 128],
                "activation": "relu",
                "optimizer": "adam",
                "learning_rate": 0.001,
            }


@dataclass
class ToolsConfig:
    """Tools configuration"""

    enabled: bool = True
    max_concurrent_executions: int = 50
    default_timeout_seconds: int = 30
    permission_checking_enabled: bool = True
    tool_discovery_enabled: bool = True
    custom_tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_categories: List[str] = field(
        default_factory=lambda: [
            "development",
            "data_analysis",
            "communication",
            "file_management",
            "system_admin",
            "ai_ml",
            "security",
            "monitoring",
            "productivity",
            "integration",
        ]
    )


@dataclass
class CommunicationConfig:
    """Communication configuration"""

    socketio_enabled: bool = True
    max_connections: int = 1000
    heartbeat_interval_seconds: int = 30
    disconnect_timeout_seconds: int = 60
    room_cleanup_interval_minutes: int = 10
    message_rate_limit: int = 100
    cors_allowed_origins: List[str] = field(default_factory=list)
    redis_url: Optional[str] = None  # For scaling Socket.IO across instances


@dataclass
class MiddlewareConfig:
    """Middleware configuration"""

    security_enabled: bool = True
    rate_limiting_enabled: bool = True
    monitoring_enabled: bool = True
    exception_handling_enabled: bool = True
    request_logging_enabled: bool = True
    response_compression_enabled: bool = True
    cors_enabled: bool = True
    security_headers_enabled: bool = True


@dataclass
class DeploymentConfig:
    """Deployment configuration"""

    environment: Environment
    debug: bool = False
    log_level: str = "INFO"
    workers: int = 1
    bind_host: str = "0.0.0.0"
    bind_port: int = 8000
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    health_check_enabled: bool = True
    metrics_endpoint_enabled: bool = True
    auto_reload: bool = False


@dataclass
class SheìlyConfiguration:
    """Main Sheily enterprise configuration"""

    environment: Environment
    database: DatabaseConfig
    security: SecurityConfig
    analytics: AnalyticsConfig
    agents: AgentsConfig
    tools: ToolsConfig
    communication: CommunicationConfig
    middleware: MiddlewareConfig
    deployment: DeploymentConfig

    # Meta information
    config_version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    config_hash: Optional[str] = None

    def update_hash(self) -> None:
        """Update configuration hash"""
        self.updated_at = datetime.now(timezone.utc)
        config_str = json.dumps(asdict(self), default=str, sort_keys=True)
        self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str, indent=indent)

    def to_yaml(self) -> str:
        """Convert to YAML string"""
        return yaml.dump(self.to_dict(), default_flow_style=False)


class ConfigurationManager:
    """Manager central de configuraciones enterprise"""

    def __init__(self):
        self.config: Optional[SheìlyConfiguration] = None
        self.config_file_path: Optional[Path] = None
        self.environment: Environment = Environment.DEVELOPMENT
        self.config_watchers: Set[Callable] = set()
        self.hot_reload_enabled: bool = False
        self.event_stream = None
        self.analytics = None

        # File watching
        self._watch_task: Optional[asyncio.Task] = None
        self._last_file_mtime: Optional[float] = None
        self._running = False

    async def initialize(
        self,
        config_file: Optional[str] = None,
        environment: Optional[Environment] = None,
    ) -> None:
        """Initialize configuration manager"""
        try:
            # Set environment
            if environment:
                self.environment = environment
            else:
                self.environment = Environment(os.getenv("SHEILY_ENV", "development"))

            # Load configuration
            if config_file:
                await self.load_from_file(config_file)
            else:
                # Look for environment-specific config
                await self._load_default_config()

            # Initialize event system integration
            self.event_stream = get_event_stream()
            self.analytics = await get_analytics_system()

            # Start file watching if enabled
            if self.hot_reload_enabled and self.config_file_path:
                await self.start_hot_reload()

            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {
                    "status": "configuration_manager_initialized",
                    "environment": self.environment.value,
                    "config_hash": self.config.config_hash if self.config else None,
                },
            )

            logger.info(
                f"✅ Configuration Manager initialized for {self.environment.value}"
            )

        except Exception as e:
            logger.error(f"Error initializing configuration manager: {e}")
            raise

    async def _load_default_config(self) -> None:
        """Load default configuration for environment"""
        config_paths = [
            f"config/sheily_{self.environment.value}.yaml",
            f"config/sheily_{self.environment.value}.yml",
            f"config/sheily_{self.environment.value}.json",
            "config/sheily.yaml",
            "config/sheily.yml",
            "config/sheily.json",
            "sheily_config.yaml",
            "sheily_config.yml",
            "sheily_config.json",
        ]

        for config_path in config_paths:
            if os.path.exists(config_path):
                await self.load_from_file(config_path)
                return

        # No config file found, create default
        logger.info(
            f"No configuration file found, creating default configuration for {self.environment.value}"
        )
        await self.create_default_config()

    async def load_from_file(self, file_path: str) -> None:
        """Load configuration from file"""
        try:
            file_path = Path(file_path)
            self.config_file_path = file_path

            if not file_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse based on extension
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(content)
            elif file_path.suffix.lower() == ".json":
                config_data = json.loads(content)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {file_path.suffix}"
                )

            # Create configuration object
            self.config = await self._create_config_from_dict(config_data)
            self.config.update_hash()

            # Update file modification time
            self._last_file_mtime = file_path.stat().st_mtime

            # Validate configuration
            await self.validate_config()

            logger.info(f"✅ Configuration loaded from {file_path}")

        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            raise

    async def _create_config_from_dict(
        self, config_data: Dict[str, Any]
    ) -> SheìlyConfiguration:
        """Create configuration object from dictionary"""
        try:
            # Handle environment
            env_str = config_data.get("environment", self.environment.value)
            environment = Environment(env_str)

            # Create sub-configurations
            database_config = DatabaseConfig(**config_data.get("database", {}))
            security_config = SecurityConfig(**config_data.get("security", {}))
            analytics_config = AnalyticsConfig(**config_data.get("analytics", {}))
            agents_config = AgentsConfig(**config_data.get("agents", {}))
            tools_config = ToolsConfig(**config_data.get("tools", {}))
            communication_config = CommunicationConfig(
                **config_data.get("communication", {})
            )
            middleware_config = MiddlewareConfig(**config_data.get("middleware", {}))
            deployment_config = DeploymentConfig(
                environment=environment, **config_data.get("deployment", {})
            )

            return SheìlyConfiguration(
                environment=environment,
                database=database_config,
                security=security_config,
                analytics=analytics_config,
                agents=agents_config,
                tools=tools_config,
                communication=communication_config,
                middleware=middleware_config,
                deployment=deployment_config,
            )

        except Exception as e:
            logger.error(f"Error creating configuration object: {e}")
            raise

    async def create_default_config(self) -> None:
        """Create default configuration"""
        try:
            # Environment-specific defaults
            if self.environment == Environment.PRODUCTION:
                database_url = os.getenv(
                    "DATABASE_URL", "sqlite:///./sheily_production.db"
                )
                debug = False
                log_level = "WARNING"
                workers = 4
            elif self.environment == Environment.STAGING:
                database_url = os.getenv(
                    "DATABASE_URL", "sqlite:///./sheily_staging.db"
                )
                debug = False
                log_level = "INFO"
                workers = 2
            else:  # development, testing, local
                database_url = os.getenv(
                    "DATABASE_URL", "sqlite:///./sheily_development.db"
                )
                debug = True
                log_level = "DEBUG"
                workers = 1

            # Create configuration with environment-specific defaults
            self.config = SheìlyConfiguration(
                environment=self.environment,
                database=DatabaseConfig(database_url=database_url),
                security=SecurityConfig(
                    secret_key=os.getenv("SECRET_KEY", ""),
                    jwt_secret=os.getenv("JWT_SECRET", ""),
                ),
                analytics=AnalyticsConfig(),
                agents=AgentsConfig(),
                tools=ToolsConfig(),
                communication=CommunicationConfig(),
                middleware=MiddlewareConfig(),
                deployment=DeploymentConfig(
                    environment=self.environment,
                    debug=debug,
                    log_level=log_level,
                    workers=workers,
                ),
            )

            self.config.update_hash()

            logger.info(
                f"✅ Created default configuration for {self.environment.value}"
            )

        except Exception as e:
            logger.error(f"Error creating default configuration: {e}")
            raise

    async def save_to_file(
        self, file_path: Optional[str] = None, format: str = "yaml"
    ) -> None:
        """Save current configuration to file"""
        try:
            if not self.config:
                raise ValueError("No configuration to save")

            # Determine file path
            if file_path:
                save_path = Path(file_path)
            elif self.config_file_path:
                save_path = self.config_file_path
            else:
                save_path = Path(f"config/sheily_{self.environment.value}.{format}")

            # Create directory if needed
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Update hash before saving
            self.config.update_hash()

            # Generate content
            if format.lower() in ["yaml", "yml"]:
                content = self.config.to_yaml()
            elif format.lower() == "json":
                content = self.config.to_json()
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Write file
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.config_file_path = save_path
            self._last_file_mtime = save_path.stat().st_mtime

            logger.info(f"✅ Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    async def validate_config(self) -> List[str]:
        """Validate current configuration"""
        errors = []

        if not self.config:
            errors.append("No configuration loaded")
            return errors

        try:
            # Database validation
            if not self.config.database.database_url:
                errors.append("Database URL is required")

            # Security validation
            if not self.config.security.secret_key:
                errors.append("Security secret key is required")

            if not self.config.security.jwt_secret:
                errors.append("JWT secret is required")

            # Port validation
            port = self.config.deployment.bind_port
            if not (1 <= port <= 65535):
                errors.append(f"Invalid port number: {port}")

            # SSL validation
            if self.config.deployment.ssl_enabled:
                if not self.config.deployment.ssl_cert_path:
                    errors.append("SSL certificate path required when SSL is enabled")
                if not self.config.deployment.ssl_key_path:
                    errors.append("SSL key path required when SSL is enabled")

            # Environment-specific validations
            if self.config.environment == Environment.PRODUCTION:
                if self.config.deployment.debug:
                    errors.append("Debug mode should be disabled in production")
                if self.config.deployment.auto_reload:
                    errors.append("Auto-reload should be disabled in production")

            if errors:
                logger.warning(
                    f"Configuration validation found {len(errors)} issues: {errors}"
                )
            else:
                logger.info("✅ Configuration validation passed")

            return errors

        except Exception as e:
            error_msg = f"Error during configuration validation: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return errors

    # ========================================
    # HOT RELOAD
    # ========================================

    async def start_hot_reload(self) -> None:
        """Start hot reload file watching"""
        if not self.config_file_path:
            logger.warning("Cannot start hot reload: no configuration file path")
            return

        self.hot_reload_enabled = True
        self._running = True
        self._watch_task = asyncio.create_task(self._file_watch_loop())
        logger.info(f"🔄 Hot reload enabled for {self.config_file_path}")

    async def stop_hot_reload(self) -> None:
        """Stop hot reload file watching"""
        self.hot_reload_enabled = False
        self._running = False

        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Hot reload disabled")

    async def _file_watch_loop(self) -> None:
        """File watching loop for hot reload"""
        logger.info("Started configuration file watching")

        while self._running:
            try:
                if self.config_file_path and self.config_file_path.exists():
                    current_mtime = self.config_file_path.stat().st_mtime

                    if (
                        self._last_file_mtime is not None
                        and current_mtime > self._last_file_mtime
                    ):

                        logger.info("🔄 Configuration file changed, reloading...")

                        try:
                            old_hash = self.config.config_hash if self.config else None
                            await self.load_from_file(str(self.config_file_path))
                            new_hash = self.config.config_hash if self.config else None

                            if old_hash != new_hash:
                                await self._notify_config_change()
                                logger.info("✅ Configuration reloaded successfully")

                        except Exception as e:
                            logger.error(f"Error reloading configuration: {e}")

                await asyncio.sleep(2)  # Check every 2 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in file watch loop: {e}")
                await asyncio.sleep(5)

    async def _notify_config_change(self) -> None:
        """Notify watchers of configuration change"""
        try:
            # Notify event system
            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {
                    "status": "configuration_reloaded",
                    "environment": (
                        self.config.environment.value if self.config else "unknown"
                    ),
                    "config_hash": self.config.config_hash if self.config else None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Notify registered watchers
            for watcher in self.config_watchers:
                try:
                    if asyncio.iscoroutinefunction(watcher):
                        await watcher(self.config)
                    else:
                        watcher(self.config)
                except Exception as e:
                    logger.error(f"Error in config watcher: {e}")

        except Exception as e:
            logger.error(f"Error notifying configuration change: {e}")

    def add_config_watcher(self, watcher: Callable) -> None:
        """Add configuration change watcher"""
        self.config_watchers.add(watcher)

    def remove_config_watcher(self, watcher: Callable) -> None:
        """Remove configuration change watcher"""
        self.config_watchers.discard(watcher)

    # ========================================
    # DEPLOYMENT HELPERS
    # ========================================

    async def generate_deployment_script(self, target_environment: Environment) -> str:
        """Generate deployment script for target environment"""
        if not self.config:
            raise ValueError("No configuration loaded")

        script_lines = [
            "#!/bin/bash",
            "# Sheily Enterprise Deployment Script",
            f"# Generated for environment: {target_environment.value}",
            f"# Generated at: {datetime.now(timezone.utc).isoformat()}",
            "",
            "set -e",
            "",
            "# Environment variables",
            f"export SHEILY_ENV={target_environment.value}",
            f'export DATABASE_URL="{self.config.database.database_url}"',
            f'export SECRET_KEY="{self.config.security.secret_key[:10]}..."',
            "",
            "# Create necessary directories",
            "mkdir -p logs",
            "mkdir -p data",
            "mkdir -p config",
            "",
            "# Install dependencies",
            "pip install -r requirements.txt",
            "",
            "# Database setup",
            "python -m sheily_core.database.setup_db",
            "",
            "# Start application",
            f"gunicorn -w {self.config.deployment.workers} "
            + f"-b {self.config.deployment.bind_host}:{self.config.deployment.bind_port} "
            + f"--log-level {self.config.deployment.log_level.lower()} "
            + "sheily_core.main:app",
            "",
        ]

        return "\n".join(script_lines)

    async def generate_docker_compose(self) -> str:
        """Generate Docker Compose configuration"""
        if not self.config:
            raise ValueError("No configuration loaded")

        compose_config = {
            "version": "3.8",
            "services": {
                "sheily": {
                    "build": ".",
                    "ports": [f"{self.config.deployment.bind_port}:8000"],
                    "environment": {
                        "SHEILY_ENV": self.config.environment.value,
                        "DATABASE_URL": self.config.database.database_url,
                        "SECRET_KEY": "${SECRET_KEY}",
                        "JWT_SECRET": "${JWT_SECRET}",
                    },
                    "volumes": [
                        "./data:/app/data",
                        "./logs:/app/logs",
                        "./config:/app/config",
                    ],
                    "restart": "unless-stopped",
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3,
                    },
                }
            },
        }

        # Add database service if using PostgreSQL
        if "postgresql" in self.config.database.database_url:
            compose_config["services"]["postgres"] = {
                "image": "postgres:13",
                "environment": {
                    "POSTGRES_DB": "${POSTGRES_DB:-sheily}",
                    "POSTGRES_USER": "${POSTGRES_USER:-sheily}",
                    "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}",
                },
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
                "restart": "unless-stopped",
            }
            compose_config["volumes"] = {"postgres_data": {}}

        # Add Redis if Socket.IO scaling is enabled
        if self.config.communication.redis_url:
            compose_config["services"]["redis"] = {
                "image": "redis:6-alpine",
                "restart": "unless-stopped",
                "volumes": ["redis_data:/data"],
            }
            if "volumes" not in compose_config:
                compose_config["volumes"] = {}
            compose_config["volumes"]["redis_data"] = {}

        return yaml.dump(compose_config, default_flow_style=False)

    # ========================================
    # CONFIGURATION ACCESS
    # ========================================

    def get_config(self) -> Optional[SheìlyConfiguration]:
        """Get current configuration"""
        return self.config

    def get_database_config(self) -> Optional[DatabaseConfig]:
        """Get database configuration"""
        return self.config.database if self.config else None

    def get_security_config(self) -> Optional[SecurityConfig]:
        """Get security configuration"""
        return self.config.security if self.config else None

    def get_deployment_config(self) -> Optional[DeploymentConfig]:
        """Get deployment configuration"""
        return self.config.deployment if self.config else None

    def update_config_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update specific configuration section"""
        try:
            if not self.config:
                return False

            if hasattr(self.config, section):
                section_obj = getattr(self.config, section)
                for key, value in updates.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

                self.config.update_hash()
                logger.info(f"✅ Updated configuration section: {section}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating configuration section {section}: {e}")
            return False


# ========================================
# GLOBAL INSTANCE
# ========================================

_config_manager: Optional[ConfigurationManager] = None


async def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
        await _config_manager.initialize()
    return _config_manager


async def get_config() -> Optional[SheìlyConfiguration]:
    """Get current configuration"""
    config_manager = await get_config_manager()
    return config_manager.get_config()


async def initialize_configuration_system(
    config_file: Optional[str] = None,
    environment: Optional[Environment] = None,
    hot_reload: bool = False,
) -> ConfigurationManager:
    """Initialize the complete configuration system"""
    config_manager = await get_config_manager()

    if config_file or environment:
        await config_manager.initialize(config_file, environment)

    if hot_reload:
        await config_manager.start_hot_reload()

    logger.info("✅ Sheily Enterprise Configuration System initialized")
    return config_manager


__all__ = [
    "Environment",
    "ConfigType",
    "SheìlyConfiguration",
    "ConfigurationManager",
    "DatabaseConfig",
    "SecurityConfig",
    "AnalyticsConfig",
    "AgentsConfig",
    "ToolsConfig",
    "CommunicationConfig",
    "MiddlewareConfig",
    "DeploymentConfig",
    "get_config_manager",
    "get_config",
    "initialize_configuration_system",
]
