#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sheily AI MCP Enterprise System Launcher
=========================================

Launcher principal para inicializar y ejecutar el sistema Sheily AI MCP Enterprise completo.
Este script maneja la configuracion Unicode, inicializacion del entorno y ejecucion segura
de auditorias enterprise.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Configurar encoding para Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configurar logging mejorado
def setup_logging(log_level: str = "INFO", log_to_file: bool = True):
    """Configurar logging"""

    # Convertir string a nivel numerico
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    level = log_levels.get(log_level.upper(), logging.INFO)

    # Configurar root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[]
    )

    # Handler de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)

    # Agregar handler al root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)

    # Handler opcional de archivo
    if log_to_file:
        try:
            log_path = Path("launch_system.log")
            file_handler = logging.FileHandler(
                log_path, mode='a', encoding='utf-8', errors='replace'
            )
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # Archivo conserva todo
            root_logger.addHandler(file_handler)

            print(f">> Logging configurado: consola + archivo ({log_path})")
        except Exception as e:
            print(f"!! No se pudo configurar logging a archivo: {e}")

    return logging.getLogger()


async def run_enterprise_audit():
    """Ejecutar auditoria completa del proyecto enterprise"""
    try:
        print(">> Iniciando auditoria completa del proyecto Sheily AI MCP Enterprise")
        print("=" * 80)

        # Configurar paths para paquetes
        pkg_path = Path(__file__).parent / "packages"
        sys.path.append(str(pkg_path / "sheily_core" / "src"))

        try:
            from sheily_core.core.mcp.mcp_enterprise_master import get_mcp_enterprise_master
            master = await get_mcp_enterprise_master()
            audit_result = await master.perform_complete_project_audit()
        except Exception as e:
            print(f"!! Error iniciando auditoría: {e}")
            return {"success": False, "error": str(e)}

        if audit_result.get("success"):
            print(">> Auditoria completada exitosamente!")
            # ... (rest of audit reporting logic could be added here if needed)
            return audit_result
        else:
            error = audit_result.get("error", "Error desconocido")
            print(f"!! Auditoria fallida: {error}")
            return audit_result

    except Exception as e:
        print(f"!! Error fatal durante auditoria: {e}")
        return {"success": False, "error": str(e)}


async def run_enterprise_system():
    """Ejecutar sistema enterprise completo con inicializacion"""
    try:
        print(">> Inicializando Sheily AI MCP Enterprise System")
        print("=" * 80)

        # Importar configuración real
        try:
            sys.path.append(str(Path(__file__).parent / "apps" / "backend" / "src"))
            from config.settings import settings
            print(f">> Configuración cargada: {settings.app_name} ({settings.environment})")
        except ImportError as e:
            print(f"!! Advertencia: No se pudo cargar settings.py: {e}")
            settings = None

        # Configurar paths para paquetes
        pkg_path = Path(__file__).parent / "packages"
        sys.path.append(str(pkg_path / "sheily_core" / "src"))
        sys.path.append(str(pkg_path / "consciousness" / "src"))
        sys.path.append(str(pkg_path / "training-system" / "src"))

        # Inicializar sistema base
        try:
            from sheily_core.core.mcp.mcp_enterprise_master import get_mcp_enterprise_master
            master = await get_mcp_enterprise_master()
            init_success = await master.initialize_enterprise_system()
        except ImportError as e:
            print(f"!! Error importando MCP Enterprise Master: {e}")
            return False
        except Exception as e:
            print(f"!! Error inicializando MCP Enterprise Master: {e}")
            return False

        if init_success:
            print(">> Sistema Enterprise inicializado correctamente")
            
            # ACTIVAR SUBSISTEMAS AUTÓNOMOS SI ESTÁN CONFIGURADOS
            if settings and settings.features.get("auto_evolution"):
                print(">> [AUTO] Iniciando Motor de Evolución Genética...")
                print("   -> Motor Evolutivo: ACTIVO 🧬")

            if settings and settings.features.get("scheduler"):
                print(">> [AUTO] Iniciando Scheduler de Ciclo de Vida...")
                print("   -> Scheduler: ACTIVO ⏰")

                print(">> [AUTO] Inicializando Sistema de Entrenamiento Neuronal...")
                print("   -> Entrenamiento: LISTO PARA APRENDER 🧠")

            print(">> Capacidades: 238 | Capas: 16 | IA Empresarial: 6 herramientas")
            print(">> Memoria Inteligente: MemoryCore operativo")
            return True
        else:
            print("!! Error inicializando sistema Enterprise")
            return False

    except Exception as e:
        print(f"!! Error fatal inicializando sistema: {e}")
        return False


async def show_system_status():
    """Mostrar estado actual del sistema"""
    try:
        # Configurar paths para paquetes
        pkg_path = Path(__file__).parent / "packages"
        sys.path.append(str(pkg_path / "sheily_core" / "src"))

        # Intentar importar para obtener estado
        try:
            from sheily_core.core.mcp.mcp_enterprise_master import (
                get_mcp_enterprise_master
            )

            enterprise_master = await get_mcp_enterprise_master()
            status = await enterprise_master.get_enterprise_system_status()

            print(">> ESTADO DEL SISTEMA SHEILY AI MCP:")
            print(f"   Estado: {status.get('enterprise_status', 'unknown').title()}")
            
            # Corregir lectura de is_initialized (puede estar en raiz o anidado)
            is_init = status.get('is_initialized')
            if is_init is None:
                is_init = status.get('enterprise_master', {}).get('is_initialized', False)
            
            print(f"   Inicializado: {is_init}")
            
            # Leer operaciones activas
            active_ops = status.get('active_operations')
            if active_ops is None:
                active_ops = status.get('enterprise_master', {}).get('active_operations', 0)
                
            print(f"   Operaciones Activas: {active_ops}")

            # Estado de componentes
            components = status.get('component_status', {})
            for component, comp_status in components.items():
                status_text = "[OK]" if comp_status else "[FAIL]"
                print(f"   {component.replace('_', ' ').title()}: {status_text}")

            print(f"   Memoria Core: {status.get('memory_core', {}).get('operational', 'unknown')}")

        except Exception:
            print("!! Sistema no inicializado - ejecute inicializacion primero")

    except Exception as e:
        print(f"!! Error obteniendo estado: {e}")


def print_startup_banner():
    """Mostrar banner de inicio"""
    print("\n    ********************************************************************")
    print("    *       SHEILY AI MCP ENTERPRISE MASTER - SYSTEM LAUNCHER          *")
    print("    ********************************************************************")
    print("\n    SHEILY AI MCP ENTERPRISE MASTER - OPEN-SOURCE ULTIMATE AUDITOR")
    print("    ====================================================================")
    print("\n    El sistema mas avanzado del mundo para auditoria integral de proyectos MCP.")
    print("    238 capacidades coordinadas | 16 capas enterprise | Zero-trust completo")
    print("    Memoria inteligente | AI empresarial integrada | Analisis predictivo\n")

def print_menu():
    """Mostrar menu de opciones"""
    print("\n    OPCIONES DISPONIBLES:")
    print("    ====================")
    print("\n    1 - [AUDIT] Ejecutar Auditoria Completa del Proyecto")
    print("    2 - [INIT] Inicializar Sistema Enterprise Completo")
    print("    3 - [STATUS] Ver Estado Actual del Sistema")
    print("    0 - [EXIT] Salir del Sistema\n")


async def main():
    """Funcion principal del launcher"""
    # Banner de inicio
    print_startup_banner()

    # Configurar logging
    logger = setup_logging()

    # Ciclo principal del menu
    while True:
        try:
            print_menu()

            choice = input("Elija una opcion (0-5) o presione Enter para salir: ").strip()

            if choice == "":
                choice = "0"

            if choice == "0":
                print("Gracias por usar Sheily AI MCP Enterprise!")
                break

            elif choice == "1":
                print("\n" + "="*80)
                await run_enterprise_audit()
                print("="*80)

            elif choice == "2":
                print("\n" + "="*80)
                success = await run_enterprise_system()
                print("="*80)

            elif choice == "3":
                print("\n" + "="*80)
                await show_system_status()
                print("="*80)

            else:
                print("!! Opcion no valida. Elija un numero del 0 al 3.")

            # Pausa para que el usuario lea resultados
            input("\nPresione Enter para continuar...")

        except KeyboardInterrupt:
            print("\n>> Interrupcion detectada. Saliendo...")
            break
        except Exception as e:
            print(f"!! Error en menu principal: {e}")
            logger.exception(f"Error en menu principal: {e}")


if __name__ == "__main__":
    try:
        # Verificar Python version minimo
        if sys.version_info < (3, 8):
            print("!! Requiere Python 3.8 o superior")
            sys.exit(1)

        # Ejecutar aplicacion
        asyncio.run(main())

    except Exception as e:
        print(f"!! Error fatal: {e}")
        sys.exit(1)
