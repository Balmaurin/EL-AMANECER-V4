#!/usr/bin/env python3
"""
Script de Backup y Recovery - Sheily AI
=======================================

Herramientas para gestión de backups del sistema Sheily AI:
- Crear backups manuales
- Listar backups disponibles
- Restaurar desde backups
- Verificar integridad
- Limpieza automática
"""

import asyncio
import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from sheily_core.backup.backup_manager import BackupConfig, BackupManager


class BackupCLI:
    """Interfaz de línea de comandos para gestión de backups"""

    def __init__(self):
        self.manager = BackupManager()

    async def create_backup(self, backup_type: str = "full", components: str = None):
        """Crear un nuevo backup"""
        print(f"🚀 Creando backup {backup_type}...")

        comp_list = components.split(",") if components else None

        result = await self.manager.create_backup(backup_type, comp_list)

        if result:
            print("✅ Backup creado exitosamente")
            print(f"   ID: {result.id}")
            print(f"   Tipo: {result.type}")
            print(f"   Tamaño: {result.size_bytes / (1024*1024):.2f} MB")
            print(f"   Componentes: {', '.join(result.components)}")
        else:
            print("❌ Error creando backup")
            return 1

        return 0

    async def list_backups(self, backup_type: str = None):
        """Listar backups disponibles"""
        backups = self.manager.list_backups(backup_type)

        if not backups:
            print("📭 No hay backups disponibles")
            return 0

        print("📋 BACKUPS DISPONIBLES")
        print("=" * 80)

        for backup in backups[:20]:  # Mostrar máximo 20
            status_icon = (
                "✅"
                if backup.status == "completed"
                else "❌" if backup.status == "failed" else "⏳"
            )
            print(f"{status_icon} {backup.id}")
            print(f"   Tipo: {backup.type}")
            print(f"   Fecha: {backup.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Tamaño: {backup.size_bytes / (1024*1024):.2f} MB")
            print(f"   Estado: {backup.status}")
            if backup.components:
                print(f"   Componentes: {', '.join(backup.components)}")
            print()

        if len(backups) > 20:
            print(f"... y {len(backups) - 20} backups más")

        return 0

    async def restore_backup(self, backup_id: str, components: str = None):
        """Restaurar desde un backup"""
        print(f"🔄 Restaurando desde backup: {backup_id}")

        comp_list = components.split(",") if components else None

        success = await self.manager.restore_backup(backup_id, comp_list)

        if success:
            print("✅ Restauración completada exitosamente")
            if comp_list:
                print(f"   Componentes restaurados: {', '.join(comp_list)}")
        else:
            print("❌ Error durante la restauración")
            return 1

        return 0

    async def verify_backup(self, backup_id: str):
        """Verificar integridad de un backup"""
        print(f"🔍 Verificando integridad de backup: {backup_id}")

        is_valid = await self.manager.verify_backup_integrity(backup_id)

        if is_valid:
            print("✅ Backup verificado correctamente - Integridad OK")
        else:
            print("❌ Backup corrupto o inválido")
            return 1

        return 0

    async def cleanup_backups(self):
        """Limpiar backups antiguos"""
        print("🧹 Limpiando backups antiguos...")

        removed = await self.manager.cleanup_old_backups()

        if removed > 0:
            print(f"✅ Eliminados {removed} backups antiguos")
        else:
            print("📭 No hay backups antiguos para eliminar")

        return 0

    async def show_stats(self):
        """Mostrar estadísticas del sistema de backup"""
        print("📊 ESTADÍSTICAS DEL SISTEMA DE BACKUP")
        print("=" * 50)

        stats = await self.manager.get_backup_stats()

        print(f"Total de backups: {stats['total_backups']}")
        print(f"Backups exitosos: {stats['successful_backups']}")
        print(f"Backups fallidos: {stats['failed_backups']}")
        print(f"Espacio total usado: {stats.get('total_size_mb', 0):.1f} MB")
        print(
            f"Espacio promedio por backup: {stats.get('avg_backup_size_mb', 0):.2f} MB"
        )
        print(f"Tasa de éxito: {stats.get('success_rate', 0):.1f}%")
        print(f"Backups esta semana: {stats.get('backups_this_week', 0)}")
        print(f"Último backup: {stats['last_backup'] or 'Nunca'}")
        print(f"Retención: {stats['retention_days']} días")

        return 0

    async def run_scheduler(self):
        """Ejecutar el programador automático (modo continuo)"""
        print("⏰ Iniciando programador automático de backups...")
        print("Presiona Ctrl+C para detener")

        try:
            await self.manager.auto_backup_scheduler()
        except KeyboardInterrupt:
            print("\n👋 Programador detenido")
            return 0


def print_help():
    """Mostrar ayuda del comando"""
    print("🔧 SHEILY AI - HERRAMIENTAS DE BACKUP")
    print("=" * 50)
    print()
    print("USO: python backup_tools.py <comando> [opciones]")
    print()
    print("COMANDOS DISPONIBLES:")
    print("  create [tipo] [componentes]    Crear nuevo backup")
    print("                                 tipo: full, incremental, config, model")
    print(
        "                                 componentes: config,data,models,logs (separados por coma)"
    )
    print("  list [tipo]                   Listar backups disponibles")
    print("  restore <id> [componentes]    Restaurar desde backup específico")
    print("  verify <id>                   Verificar integridad de backup")
    print("  cleanup                       Limpiar backups antiguos")
    print("  stats                         Mostrar estadísticas")
    print("  scheduler                     Ejecutar programador automático")
    print("  help                          Mostrar esta ayuda")
    print()
    print("EJEMPLOS:")
    print("  python backup_tools.py create full")
    print("  python backup_tools.py create config config")
    print("  python backup_tools.py list")
    print("  python backup_tools.py restore full_20241107_042100")
    print("  python backup_tools.py verify full_20241107_042100")


async def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print_help()
        return 1

    command = sys.argv[1].lower()
    cli = BackupCLI()

    try:
        if command == "create":
            backup_type = sys.argv[2] if len(sys.argv) > 2 else "full"
            components = sys.argv[3] if len(sys.argv) > 3 else None
            return await cli.create_backup(backup_type, components)

        elif command == "list":
            backup_type = sys.argv[2] if len(sys.argv) > 2 else None
            return await cli.list_backups(backup_type)

        elif command == "restore":
            if len(sys.argv) < 3:
                print("❌ Debes especificar el ID del backup")
                return 1
            components = sys.argv[3] if len(sys.argv) > 3 else None
            return await cli.restore_backup(sys.argv[2], components)

        elif command == "verify":
            if len(sys.argv) < 3:
                print("❌ Debes especificar el ID del backup")
                return 1
            return await cli.verify_backup(sys.argv[2])

        elif command == "cleanup":
            return await cli.cleanup_backups()

        elif command == "stats":
            return await cli.show_stats()

        elif command == "scheduler":
            return await cli.run_scheduler()

        elif command == "help":
            print_help()
            return 0

        else:
            print(f"❌ Comando desconocido: {command}")
            print_help()
            return 1

    except Exception as e:
        print(f"❌ Error ejecutando comando: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
