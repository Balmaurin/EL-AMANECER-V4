#!/usr/bin/env python3
"""
Sheily MCP Enterprise - Dependency Management CLI
Sistema de línea de comandos para gestión avanzada de dependencias

Uso típico:
    cline deps analyze          # Análisis completo
    cline deps install-core     # Instalar dependencias base
    cline deps lock             # Generar archivos de bloqueo
    cline deps security-scan    # Escaneo de seguridad
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

from .agent_orchestrator import AgentOrchestrator
from .cicd_integrator import CIIntegrationEngine
from .database_controller import DatabaseController

# Importar componentes del sistema
from .dependency_analyzer import DependencyAnalyzer
from .environment_manager import EnvironmentManager

# COMPONENTES PARA CONTROL ABSOLUTO DEL PROYECTO
from .infrastructure_manager import InfrastructureManager
from .installation_orchestrator import InstallationOrchestrator
from .optimization_engine import OptimizationEngine
from .security_scanner import SecurityScanner
from .service_manager import ServiceManager
from .update_manager import UpdateManager
from .validation_engine import ValidationEngine
from .version_locker import VersionLocker


class DependencyCLI:
    """Interface de línea de comandos para gestión de dependencias"""

    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.root_dir / "config"
        self.dependency_manager = None

        # Inicializar componentes del sistema
        print("🚀 Inicializando Sheily MCP Enterprise - Sistema de Control Absoluto...")

        self.analyzer = DependencyAnalyzer(self.root_dir)
        self.env_manager = EnvironmentManager(self.root_dir)
        self.installer = InstallationOrchestrator(self.root_dir)
        self.locker = VersionLocker(self.root_dir)
        self.updater = UpdateManager(self.root_dir)
        self.validator = ValidationEngine(self.root_dir)
        self.security = SecurityScanner(self.root_dir)
        self.optimizer = OptimizationEngine(self.root_dir)

        # COMPONENTES NUEVOS para CONTROL ABSOLUTO
        self.infrastructure = InfrastructureManager(self.root_dir)
        self.database = None  # Se inicializará cuando se use
        self.agents = AgentOrchestrator(self.root_dir, self.infrastructure)
        self.cicd = None  # Se inicializará cuando se use
        self.service_manager = ServiceManager(
            self.root_dir, self.infrastructure, self.database, self.agents
        )

        print("✅ Sistema MCP Enterprise inicializado exitosamente!")
        print("🎯 CONTROL ABSOLUTO del proyecto establecido")

    def create_parser(self) -> argparse.ArgumentParser:
        """Crear parser principal de argumentos"""
        parser = argparse.ArgumentParser(
            description="Sheily MCP Enterprise - Dependency Management System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Ejemplos de uso:
  cline deps analyze                     # Análisis completo de dependencias
  cline deps install-core               # Instalar dependencias base
  cline deps env create --name dev      # Crear entorno virtual
  cline deps lock                       # Generar archivos de bloqueo
  cline deps security-scan              # Escaneo de vulnerabilidades
  cline deps update-safe                # Actualizaciones seguras

Para más información: cline deps --help
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

        # Comando analyze
        analyze_parser = subparsers.add_parser(
            "analyze", help="Analizar dependencias del proyecto"
        )
        analyze_parser.add_argument(
            "--format",
            choices=["json", "table", "html"],
            default="table",
            help="Formato de salida",
        )
        analyze_parser.add_argument("--output", "-o", help="Archivo de salida")
        analyze_parser.add_argument(
            "--deep",
            action="store_true",
            help="Análisis profundo incluyendo dependencias transitorias",
        )

        # Comando install
        install_parser = subparsers.add_parser("install", help="Instalar dependencias")
        install_parser.add_argument(
            "target",
            nargs="?",
            choices=["core", "ai", "frontend", "dev", "all"],
            default="all",
            help="Qué instalar",
        )
        install_parser.add_argument(
            "--dry-run", action="store_true", help="Simular instalación sin ejecutar"
        )
        install_parser.add_argument(
            "--force", action="store_true", help="Forzar reinstallación"
        )
        install_parser.add_argument(
            "--parallel", action="store_true", help="Instalación en paralelo"
        )

        # Comando env
        env_parser = subparsers.add_parser("env", help="Gestionar entornos virtuales")
        env_subparsers = env_parser.add_subparsers(dest="env_command")

        # env create
        env_create = env_subparsers.add_parser("create", help="Crear entorno virtual")
        env_create.add_argument("--name", required=True, help="Nombre del entorno")
        env_create.add_argument("--python", default="3.11", help="Versión de Python")

        # env list
        env_subparsers.add_parser("list", help="Listar entornos")

        # env activate
        env_activate = env_subparsers.add_parser("activate", help="Activar entorno")
        env_activate.add_argument("name", help="Nombre del entorno")

        # env remove
        env_remove = env_subparsers.add_parser("remove", help="Eliminar entorno")
        env_remove.add_argument("name", help="Nombre del entorno")

        # Comando lock
        lock_parser = subparsers.add_parser("lock", help="Generar archivos de bloqueo")
        lock_parser.add_argument(
            "--format",
            choices=["poetry", "pip-tools", "requirements"],
            default="poetry",
            help="Formato de lock",
        )
        lock_parser.add_argument("--output", "-o", help="Archivo de salida")
        lock_parser.add_argument(
            "--strict", action="store_true", help="Bloqueo estricto sin ranges"
        )

        # Comando security
        security_parser = subparsers.add_parser(
            "security-scan", help="Escanear vulnerabilidades"
        )
        security_parser.add_argument(
            "--format", choices=["json", "table", "html"], default="table"
        )
        security_parser.add_argument("--output", "-o", help="Archivo de salida")
        security_parser.add_argument(
            "--severity",
            choices=["low", "medium", "high", "critical"],
            default="medium",
            help="Severidad mínima a reportar",
        )
        security_parser.add_argument(
            "--fix", action="store_true", help="Intentar correcciones automáticas"
        )

        # Comando update
        update_parser = subparsers.add_parser("update", help="Actualizar dependencias")
        update_parser.add_argument(
            "--safe", action="store_true", help="Solo actualizaciones seguras"
        )
        update_parser.add_argument(
            "--target",
            choices=["core", "ai", "frontend", "dev"],
            help="Componente específico a actualizar",
        )
        update_parser.add_argument(
            "--interactive", action="store_true", help="Modo interactivo"
        )

        # Comando validate
        validate_parser = subparsers.add_parser("validate", help="Validar instalación")
        validate_parser.add_argument(
            "--quick", action="store_true", help="Validación rápida"
        )
        validate_parser.add_argument(
            "--full", action="store_true", help="Validación completa"
        )
        validate_parser.add_argument(
            "--repair", action="store_true", help="Intentar reparaciones automáticas"
        )

        # Comando optimize
        optimize_parser = subparsers.add_parser(
            "optimize", help="Optimizar dependencias"
        )
        optimize_parser.add_argument(
            "--action",
            choices=["dedupe", "compact", "cleanup"],
            help="Tipo de optimización",
        )
        optimize_parser.add_argument("--dry-run", action="store_true")

        return parser

    async def run_command(self, args):
        """Ejecutar comando basado en argumentos"""

        if args.command == "analyze":
            await self.cmd_analyze(args)
        elif args.command == "install":
            await self.cmd_install(args)
        elif args.command == "env":
            await self.cmd_env(args)
        elif args.command == "lock":
            await self.cmd_lock(args)
        elif args.command == "security-scan":
            await self.cmd_security_scan(args)
        elif args.command == "update":
            await self.cmd_update(args)
        elif args.command == "validate":
            await self.cmd_validate(args)
        elif args.command == "optimize":
            await self.cmd_optimize(args)
        else:
            logger.error(f"Comando desconocido: {args.command}")

    async def cmd_analyze(self, args):
        """Comando de análisis de dependencias"""
        print("🔍 Analizando dependencias del proyecto Sheily MCP...")

        try:
            results = await self.analyzer.analyze_project(args.deep)

            if args.format == "json":
                output = json.dumps(results, indent=2)
            elif args.format == "table":
                output = self._format_analysis_table(results)
            elif args.format == "html":
                output = self._format_analysis_html(results)

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(output)
                print(f"✅ Análisis guardado en: {args.output}")
            else:
                print(output)

        except Exception as e:
            logger.error(f"Error en análisis: {e}")
            sys.exit(1)

    async def cmd_install(self, args):
        """Comando de instalación de dependencias"""
        print("📦 Instalando dependencias...")

        try:
            if args.dry_run:
                print("🔍 Modo simulación - No se instalará nada")

            results = await self.installer.install_dependencies(
                target=args.target,
                dry_run=args.dry_run,
                force=args.force,
                parallel=args.parallel,
            )

            print("✅ Instalación completada:")
            print(f"  - Python packages: {results.get('python_packages', 0)}")
            print(f"  - Frontend packages: {results.get('frontend_packages', 0)}")
            print(f"  - Tiempo total: {results.get('total_time', 0):.1f}s")

        except Exception as e:
            logger.error(f"Error en instalación: {e}")
            sys.exit(1)

    async def cmd_env(self, args):
        """Comando de gestión de entornos"""
        if args.env_command == "create":
            await self.cmd_env_create(args)
        elif args.env_command == "list":
            await self.cmd_env_list(args)
        elif args.env_command == "activate":
            await self.cmd_env_activate(args)
        elif args.env_command == "remove":
            await self.cmd_env_remove(args)

    async def cmd_env_create(self, args):
        """Crear entorno virtual"""
        print(f"🏗️ Creando entorno virtual: {args.name}")

        try:
            result = await self.env_manager.create_environment(args.name, args.python)

            if result["success"]:
                print(f"✅ Entorno '{args.name}' creado exitosamente")
                print(f"   📍 Ubicación: {result['path']}")
                print(f"   🐍 Python: {result.get('python_version', args.python)}")
            else:
                print(f"❌ Error creando entorno: {result.get('error', 'Desconocido')}")

        except Exception as e:
            logger.error(f"Error creando entorno: {e}")
            sys.exit(1)

    async def cmd_env_list(self, args):
        """Listar entornos virtuales"""
        print("📋 Entornos virtuales disponibles:")

        try:
            envs = await self.env_manager.list_environments()

            if not envs:
                print("   (ninguno)")
                return

            for env_name, env_info in envs.items():
                status = "✅" if env_info.get("active") else "  "
                python_ver = env_info.get("python_version", "unknown")
                path = env_info.get("path", "unknown")
                print(f"   {status} {env_name} (Python {python_ver})")
                print(f"      📍 {path}")

        except Exception as e:
            logger.error(f"Error listando entornos: {e}")

    async def cmd_env_activate(self, args):
        """Activar entorno virtual"""
        print(f"🔄 Activando entorno: {args.name}")

        try:
            result = await self.env_manager.activate_environment(args.name)

            if result["success"]:
                print(f"✅ Entorno '{args.name}' activado")
                print("💡 Para activar en tu shell actual, ejecuta:")
                print(f"   source {result.get('activate_script', 'activate')}")
            else:
                print(
                    f"❌ Error activando entorno: {result.get('error', 'Desconocido')}"
                )

        except Exception as e:
            logger.error(f"Error activando entorno: {e}")

    async def cmd_env_remove(self, args):
        """Eliminar entorno virtual"""
        print(f"🗑️ Eliminando entorno: {args.name}")

        try:
            confirm = input(
                f"¿Estás seguro de eliminar el entorno '{args.name}'? (y/N): "
            )
            if confirm.lower() not in ["y", "yes"]:
                print("Operación cancelada")
                return

            result = await self.env_manager.remove_environment(args.name)

            if result["success"]:
                print(f"✅ Entorno '{args.name}' eliminado")
            else:
                print(
                    f"❌ Error eliminando entorno: {result.get('error', 'Desconocido')}"
                )

        except Exception as e:
            logger.error(f"Error eliminando entorno: {e}")

    async def cmd_lock(self, args):
        """Comando de bloqueo de versiones"""
        print("🔒 Generando archivos de bloqueo de versiones...")

        try:
            results = await self.locker.generate_lock_file(
                format_type=args.format, strict=args.strict
            )

            # Determine output path
            if args.output:
                output_path = Path(args.output)
            else:
                if args.format == "requirements":
                    # Align with Dockerfile.backend which prefers `requirements.lock`
                    output_path = self.root_dir / "requirements.lock"
                else:
                    output_path = self.root_dir / f"requirements.{args.format}.lock"

            # Save lock file
            if args.format == "requirements":
                # Write plain pinned requirements content for install tooling
                content = results.get("content", "")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                # For other formats keep JSON metadata
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

            print(f"✅ Archivo de bloqueo generado: {output_path}")

            # Summary output
            if args.format == "poetry":
                package_count = len(results.get("package", {}))
                hash_enabled = args.strict
            elif args.format == "pip-tools":
                package_count = len(results.get("dependencies", {}))
                hash_enabled = args.strict
            else:  # requirements
                package_count = len(results.get("dependencies", {}))
                hash_enabled = False

            print(f"   📦 Paquetes bloqueados: {package_count}")
            print(f"   🔒 Hash verification: {hash_enabled}")

        except Exception as e:
            logger.error(f"Error generando bloqueo: {e}")
            sys.exit(1)

    async def cmd_security_scan(self, args):
        """Comando de escaneo de seguridad"""
        print("🛡️ Escaneando vulnerabilidades de seguridad...")

        try:
            results = await self.security.scan_vulnerabilities()

            # Filtrar por severidad
            filtered_results = self._filter_by_severity(results, args.severity)

            if args.format == "json":
                output = json.dumps(filtered_results, indent=2)
            elif args.format == "table":
                output = self._format_security_table(filtered_results)
            elif args.format == "html":
                output = self._format_security_html(filtered_results)

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(output)
                print(f"✅ Reporte de seguridad guardado en: {args.output}")

            # Mostrar resumen
            self._print_security_summary(filtered_results)

        except Exception as e:
            logger.error(f"Error en escaneo de seguridad: {e}")
            sys.exit(1)

    async def cmd_update(self, args):
        """Comando de actualización de dependencias"""
        print("⬆️ Actualizando dependencias...")

        try:
            if args.safe:
                print("🛡️ Modo seguro: Solo actualizaciones no breaking")

            results = await self.updater.update_dependencies(
                safe_mode=args.safe, target=args.target, interactive=args.interactive
            )

            print("✅ Actualización completada:")
            print(f"   📦 Paquetes actualizados: {results.get('updated_packages', 0)}")
            print(f"   ⚠️ Advertencias: {results.get('warnings', 0)}")
            print(f"   🚫 Errores: {results.get('errors', 0)}")

            if results.get("breaking_changes"):
                print("⚠️ Breaking changes detectados:")
                for change in results["breaking_changes"][:5]:
                    print(f"   - {change}")

        except Exception as e:
            logger.error(f"Error en actualización: {e}")
            sys.exit(1)

    async def cmd_validate(self, args):
        """Comando de validación de instalación"""
        print("✓ Validando instalación de dependencias...")

        try:
            if args.quick:
                results = await self.validator.quick_validate()
            elif args.full:
                results = await self.validator.full_validate()
            else:
                results = await self.validator.standard_validate()

            print(f"✅ Validación {'rápida' if args.quick else 'completa'}:")

            # Handle different result structures
            if isinstance(results.get("python_packages"), int):
                # New validation engine format
                print(f"   📦 Python packages: {results.get('python_packages', 0)}")
                print(f"   ✅ Paquetes válidos: {results.get('valid_packages', 0)}")
                print(f"   ❌ Problemas encontrados: {results.get('issues_found', 0)}")

                if results.get("pip_check_ok") is not None:
                    pip_status = "✅" if results.get("pip_check_ok") else "⚠️"
                    print(
                        f"   {pip_status} Pip check: {'OK' if results.get('pip_check_ok') else 'Issues found'}"
                    )

                if results.get("requirements_found"):
                    aligned_status = (
                        "✅" if results.get("requirements_aligned", True) else "⚠️"
                    )
                    print(
                        f"   {aligned_status} Requirements alignment: {'OK' if results.get('requirements_aligned', True) else 'Issues found'}"
                    )
            else:
                # Legacy format (if any)
                python_pkgs = results.get("python_packages", [])
                frontend_pkgs = results.get("frontend_packages", [])
                print(
                    f"   📦 Python packages: {len(python_pkgs) if isinstance(python_pkgs, list) else python_pkgs}"
                )
                print(
                    f"   🎨 Frontend packages: {len(frontend_pkgs) if isinstance(frontend_pkgs, list) else frontend_pkgs}"
                )
                print(f"   ✅ Paquetes válidos: {results.get('valid_packages', 0)}")
                print(f"   ❌ Problemas encontrados: {results.get('issues_found', 0)}")

            if results.get("issues_found", 0) > 0 and args.repair:
                print("🔧 Intentando reparaciones automáticas...")
                repair_results = await self.validator.auto_repair(results)
                print(
                    f"   ✅ Reparaciones realizadas: {repair_results.get('repairs_done', 0)}"
                )

        except Exception as e:
            logger.error(f"Error en validación: {e}")
            sys.exit(1)

    async def cmd_optimize(self, args):
        """Comando de optimización de dependencias"""
        print("⚡ Optimizando dependencias...")

        try:
            if args.dry_run:
                print("🔍 Modo simulación - No se harán cambios")

            results = await self.optimizer.optimize_dependencies(
                action=args.action, dry_run=args.dry_run
            )

            print("✅ Optimización completada:")
            print(f"   🧹 Paquetes eliminados: {results.get('removed_packages', 0)}")
            print(f"   📦 Duplicados eliminados: {results.get('deduped_packages', 0)}")
            print(f"   💾 Espacio ahorrado: {results.get('space_saved_mb', 0):.1f} MB")

        except Exception as e:
            logger.error(f"Error en optimización: {e}")
            sys.exit(1)

    def _format_analysis_table(self, results: Dict[str, Any]) -> str:
        """Formatear resultados de análisis como tabla"""
        lines = []
        lines.append("📊 ANÁLISIS DE DEPENDENCIAS - SHEILY MCP ENTERPRISE")
        lines.append("=" * 60)

        # Python packages with categorization
        python_deps = results.get("python_dependencies", {})

        # Count AI/ML packages across all categories
        ai_ml_count = 0
        for category_deps in [
            python_deps.get("core", []),
            python_deps.get("dev", []),
            python_deps.get("ci", []),
            python_deps.get("rag", []),
        ]:
            for dep in category_deps:
                if dep.get("category") == "ai_ml":
                    ai_ml_count += 1

        lines.append("\n🐍 PYTHON DEPENDENCIES:")
        lines.append(f"   Core packages: {len(python_deps.get('core', []))} ⚙️")
        lines.append(f"   AI/ML packages: {ai_ml_count} 🤖")
        lines.append(f"   Development: {len(python_deps.get('dev', []))} 🔧")
        lines.append(f"   CI/CD: {len(python_deps.get('ci', []))} 🚀")
        lines.append(f"   RAG Advanced: {len(python_deps.get('rag', []))} 📚")
        lines.append(f"   Total packages: {python_deps.get('total_packages', 0)} 📦")

        # Frontend packages
        frontend_deps = results.get("frontend_dependencies", {})
        lines.append("\n🎨 FRONTEND DEPENDENCIES:")
        lines.append(f"   Dependencies: {frontend_deps.get('total_packages', 0)} 🌐")
        lines.append(
            f"   Dev dependencies: {frontend_deps.get('total_dev_packages', 0)} 🛠️"
        )

        # Conflicts
        conflicts = results.get("conflicts", [])
        lines.append("\n⚠️ POTENTIAL CONFLICTS:")
        lines.append(f"   Detected: {len(conflicts)} ⚠️")

        if conflicts:
            for conflict in conflicts[:5]:  # Top 5
                lines.append(f"   - {conflict.get('package', 'unknown')} 🔄")

        # Security status
        security_issues = results.get("security_issues", [])
        lines.append("\n🛡️ SECURITY STATUS:")
        if not security_issues:
            lines.append("   ✅ No security issues detected 🛡️")
        else:
            lines.append(f"   ⚠️ Security issues found: {len(security_issues)} 🚨")

        return "\n".join(lines)

    def _format_analysis_html(self, results: Dict[str, Any]) -> str:
        """Formatear resultados de análisis como HTML"""
        html_lines = []
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html>")
        html_lines.append("<head>")
        html_lines.append(
            "    <title>Análisis de Dependencias - Sheily MCP Enterprise</title>"
        )
        html_lines.append("    <style>")
        html_lines.append(
            "        body { font-family: Arial, sans-serif; margin: 40px; }"
        )
        html_lines.append("        h1 { color: #333; }")
        html_lines.append(
            "        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }"
        )
        html_lines.append("        .metric { font-weight: bold; }")
        html_lines.append("    </style>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        html_lines.append(
            "    <h1>📊 Análisis de Dependencias - Sheily MCP Enterprise</h1>"
        )

        # Python dependencies section
        python_deps = results.get("python_dependencies", {})
        html_lines.append("    <div class='section'>")
        html_lines.append("        <h2>🐍 Python Dependencies</h2>")
        html_lines.append(
            f"        <p><span class='metric'>Core packages:</span> {len(python_deps.get('core', []))}</p>"
        )
        html_lines.append(
            f"        <p><span class='metric'>AI/ML packages:</span> {len(python_deps.get('ai', []))}</p>"
        )
        html_lines.append(
            f"        <p><span class='metric'>Dev packages:</span> {len(python_deps.get('dev', []))}</p>"
        )
        html_lines.append("    </div>")

        # Frontend dependencies section
        frontend_deps = results.get("frontend_dependencies", {})
        html_lines.append("    <div class='section'>")
        html_lines.append("        <h2>🎨 Frontend Dependencies</h2>")
        html_lines.append(
            f"        <p><span class='metric'>Dependencies:</span> {len(frontend_deps.get('dependencies', {}))}</p>"
        )
        html_lines.append(
            f"        <p><span class='metric'>Dev dependencies:</span> {len(frontend_deps.get('devDependencies', {}))}</p>"
        )
        html_lines.append("    </div>")

        # Conflicts section
        conflicts = results.get("conflicts", [])
        html_lines.append("    <div class='section'>")
        html_lines.append("        <h2>⚠️ Potential Conflicts</h2>")
        html_lines.append(
            f"        <p><span class='metric'>Detected:</span> {len(conflicts)}</p>"
        )
        if conflicts:
            html_lines.append("        <ul>")
            for conflict in conflicts[:5]:
                html_lines.append(
                    f"            <li>{conflict.get('package', 'unknown')}</li>"
                )
            html_lines.append("        </ul>")

        html_lines.append("    </div>")
        html_lines.append("</body>")
        html_lines.append("</html>")

        return "\n".join(html_lines)

    def _format_security_table(self, results: Dict[str, Any]) -> str:
        """Formatear resultados de seguridad como tabla"""
        lines = []
        lines.append("🛡️ SECURITY SCAN RESULTS")
        lines.append("=" * 40)

        vulnerabilities = results.get("vulnerabilities", [])

        if not vulnerabilities:
            lines.append("✅ No se encontraron vulnerabilidades")
            return "\n".join(lines)

        lines.append(f"❌ Vulnerabilidades encontradas: {len(vulnerabilities)}")

        for vuln in vulnerabilities[:10]:  # Top 10
            severity = vuln.get("severity", "unknown").upper()
            package = vuln.get("package", "unknown")
            description = vuln.get("description", "")[:50]

            lines.append(f"   {severity}: {package} - {description}...")

        return "\n".join(lines)

    def _format_security_html(self, results: Dict[str, Any]) -> str:
        """Formatear resultados de seguridad como HTML"""
        html_lines = []
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html>")
        html_lines.append("<head>")
        html_lines.append(
            "    <title>Reporte de Seguridad - Sheily MCP Enterprise</title>"
        )
        html_lines.append("    <style>")
        html_lines.append(
            "        body { font-family: Arial, sans-serif; margin: 40px; }"
        )
        html_lines.append("        h1 { color: #d32f2f; }")
        html_lines.append(
            "        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }"
        )
        html_lines.append(
            "        .vulnerability { margin: 10px 0; padding: 10px; border-left: 4px solid; }"
        )
        html_lines.append(
            "        .critical { border-color: #d32f2f; background-color: #ffebee; }"
        )
        html_lines.append(
            "        .high { border-color: #f44336; background-color: #fce4ec; }"
        )
        html_lines.append(
            "        .medium { border-color: #ff9800; background-color: #fff3e0; }"
        )
        html_lines.append(
            "        .low { border-color: #4caf50; background-color: #e8f5e8; }"
        )
        html_lines.append("        .metric { font-weight: bold; }")
        html_lines.append("    </style>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        html_lines.append("    <h1>🛡️ Security Scan Results</h1>")

        vulnerabilities = results.get("vulnerabilities", [])

        if not vulnerabilities:
            html_lines.append("    <div class='section'>")
            html_lines.append("        <h2>✅ No vulnerabilities found</h2>")
            html_lines.append("    </div>")
        else:
            html_lines.append("    <div class='section'>")
            html_lines.append(
                f"        <h2>❌ Vulnerabilities Found: {len(vulnerabilities)}</h2>"
            )

            for vuln in vulnerabilities[:10]:  # Top 10
                severity = vuln.get("severity", "unknown").lower()
                css_class = "vulnerability " + severity
                severity_display = severity.upper()
                package = vuln.get("package", "unknown")
                description = vuln.get("description", "No description")[:100]

                html_lines.append(f"        <div class='{css_class}'>")
                html_lines.append(
                    f"            <strong>{severity_display}:</strong> {package}"
                )
                html_lines.append(f"            <p>{description}...</p>")
                html_lines.append("        </div>")

            html_lines.append("    </div>")

        # Summary section
        critical = len([v for v in vulnerabilities if v.get("severity") == "critical"])
        high = len([v for v in vulnerabilities if v.get("severity") == "high"])
        medium = len([v for v in vulnerabilities if v.get("severity") == "medium"])
        low = len([v for v in vulnerabilities if v.get("severity") == "low"])

        html_lines.append("    <div class='section'>")
        html_lines.append("        <h2>📊 Summary</h2>")
        html_lines.append(
            f"        <p><span class='metric'>Critical:</span> {critical}</p>"
        )
        html_lines.append(f"        <p><span class='metric'>High:</span> {high}</p>")
        html_lines.append(
            f"        <p><span class='metric'>Medium:</span> {medium}</p>"
        )
        html_lines.append(f"        <p><span class='metric'>Low:</span> {low}</p>")
        html_lines.append("    </div>")

        html_lines.append("</body>")
        html_lines.append("</html>")

        return "\n".join(html_lines)

    def _print_security_summary(self, results: Dict[str, Any]):
        """Imprimir resumen de seguridad"""
        vulnerabilities = results.get("vulnerabilities", [])

        critical = len([v for v in vulnerabilities if v.get("severity") == "critical"])
        high = len([v for v in vulnerabilities if v.get("severity") == "high"])
        medium = len([v for v in vulnerabilities if v.get("severity") == "medium"])
        low = len([v for v in vulnerabilities if v.get("severity") == "low"])

        print(f"\n📊 RESUMEN DE SEGURIDAD:")
        print(f"   🔴 Critical: {critical}")
        print(f"   🟠 High: {high}")
        print(f"   🟡 Medium: {medium}")
        print(f"   🔵 Low: {low}")

        if vulnerabilities:
            print(f"\n💡 ACCIONES RECOMENDADAS:")
            print(f"   1. Revisar vulnerabilidades críticas inmediatamente")
            print(f"   2. Ejecutar: cline deps security-scan --fix")
            print(f"   3. Monitorear dependencias regularmente")

    def _filter_by_severity(
        self, results: Dict[str, Any], min_severity: str
    ) -> Dict[str, Any]:
        """Filtrar vulnerabilidades por severidad mínima"""
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        min_level = severity_levels.get(min_severity, 2)

        filtered_vulns = []
        for vuln in results.get("vulnerabilities", []):
            vuln_level = severity_levels.get(vuln.get("severity", "unknown").lower(), 0)
            if vuln_level >= min_level:
                filtered_vulns.append(vuln)

        filtered_results = results.copy()
        filtered_results["vulnerabilities"] = filtered_vulns

        return filtered_results


async def main():
    """Función principal del CLI"""
    cli = DependencyCLI()
    parser = cli.create_parser()

    # Si no hay argumentos, mostrar ayuda
    if len(sys.argv) == 1:
        parser.print_help()
        return

    try:
        args = parser.parse_args()
        await cli.run_command(args)
    except KeyboardInterrupt:
        print("\n⏹️ Operación cancelada por usuario")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
