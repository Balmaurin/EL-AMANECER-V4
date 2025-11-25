#!/usr/bin/env python3
"""
LIMPIEZA DE SISTEMAS EMOCIONALES REDUNDANTES
=============================================

Este script elimina de forma segura los sistemas emocionales NO utilizados:
- emotional_neuro_system.py (NO usado)
- authentic_emotional_system.py (NO usado)

MANTIENE:
- human_emotions_system.py (✅ ACTIVO en ConsciousPromptGenerator)

Análisis realizado: 2025-11-25
"""

import os
from pathlib import Path

# Rutas
CONSCIOUSNESS_DIR = Path(__file__).parent.parent / "src" / "conciencia" / "modulos"

FILES_TO_DELETE = [
    "emotional_neuro_system.py",
    "authentic_emotional_system.py"
]

FILES_TO_KEEP = [
    "human_emotions_system.py"  # ✅ ACTIVO - NO TOCAR
]

def verify_safety():
    """Verifica que es seguro eliminar los archivos"""
    print("🔍 Verificando seguridad de eliminación...")
    print("-" * 70)
    
    # Verificar que el archivo activo existe
    active_file = CONSCIOUSNESS_DIR / FILES_TO_KEEP[0]
    if not active_file.exists():
        print(f"❌ ERROR: {FILES_TO_KEEP[0]} NO EXISTE!")
        print("   Este archivo es CRÍTICO - no se puede continuar.")
        return False
    
    print(f"✅ {FILES_TO_KEEP[0]} existe y está protegido")
    
    # Verificar archivos a eliminar
    for filename in FILES_TO_DELETE:
        file_path = CONSCIOUSNESS_DIR / filename
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"📄 {filename} ({size_kb:.1f} KB) - Listo para eliminar")
        else:
            print(f"⚠️  {filename} - Ya no existe")
    
    return True


def create_backup():
    """Crea backup de los archivos antes de eliminar"""
    print("\n💾 Creando backup...")
    print("-" * 70)
    
    backup_dir = CONSCIOUSNESS_DIR / "_backup_emotional_systems"
    backup_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for filename in FILES_TO_DELETE:
        source = CONSCIOUSNESS_DIR / filename
        if source.exists():
            dest = backup_dir / f"{timestamp}_{filename}"
            import shutil
            shutil.copy2(source, dest)
            print(f"✅ Backup: {filename} → {dest.name}")
    
    print(f"\n📁 Backups en: {backup_dir}")
    return backup_dir


def delete_files():
    """Elimina los archivos NO utilizados"""
    print("\n🗑️  Eliminando sistemas emocionales redundantes...")
    print("-" * 70)
    
    deleted = []
    errors = []
    
    for filename in FILES_TO_DELETE:
        file_path = CONSCIOUSNESS_DIR / filename
        
        if file_path.exists():
            try:
                file_path.unlink()
                deleted.append(filename)
                print(f"✅ Eliminado: {filename}")
            except Exception as e:
                errors.append((filename, str(e)))
                print(f"❌ Error eliminando {filename}: {e}")
        else:
            print(f"⚠️  {filename} - Ya no existe (saltar)")
    
    # También eliminar .pyc si existen
    pycache_dir = CONSCIOUSNESS_DIR / "__pycache__"
    if pycache_dir.exists():
        for pyc_file in pycache_dir.glob("emotional_neuro_system*.pyc"):
            try:
                pyc_file.unlink()
                print(f"🧹 Limpiado: {pyc_file.name}")
            except:
                pass
        
        for pyc_file in pycache_dir.glob("authentic_emotional_system*.pyc"):
            try:
                pyc_file.unlink()
                print(f"🧹 Limpiado: {pyc_file.name}")
            except:
                pass
    
    return deleted, errors


def verify_integrity():
    """Verifica integridad del sistema después de eliminación"""
    print("\n🔬 Verificando integridad del sistema...")
    print("-" * 70)
    
    # Verificar que el archivo crítico sigue ahí
    active_file = CONSCIOUSNESS_DIR / FILES_TO_KEEP[0]
    if not active_file.exists():
        print(f"❌ CRÍTICO: {FILES_TO_KEEP[0]} FUE ELIMINADO!")
        return False
    
    print(f"✅ {FILES_TO_KEEP[0]} intacto")
    
    # Verificar que los redundantes fueron eliminados
    all_deleted = True
    for filename in FILES_TO_DELETE:
        file_path = CONSCIOUSNESS_DIR / filename
        if file_path.exists():
            print(f"⚠️  {filename} aún existe")
            all_deleted = False
        else:
            print(f"✅ {filename} eliminado correctamente")
    
    return all_deleted


def show_summary():
    """Muestra resumen de lo que quedó"""
    print("\n📊 RESUMEN DEL SISTEMA EMOCIONAL")
    print("=" * 70)
    
    print("\n✅ SISTEMA ACTIVO:")
    print(f"   • human_emotions_system.py (35 emociones, neuroquímico)")
    print(f"     └─ Integrado con ConsciousPromptGenerator")
    
    print("\n🗑️  SISTEMAS ELIMINADOS:")
    for filename in FILES_TO_DELETE:
        print(f"   • {filename}")
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("   1. ✅ Sistema simplificado - solo un sistema emocional")
    print("   2. ✅ No hay código duplicado")
    print("   3. ✅ Backups disponibles en _backup_emotional_systems/")
    print("   4. 🔄 Reiniciar sistema si está corriendo")


def main():
    print("=" * 70)
    print("LIMPIEZA DE SISTEMAS EMOCIONALES REDUNDANTES")
    print("=" * 70)
    
    # Paso 1: Verificar seguridad
    if not verify_safety():
        print("\n❌ Verificación de seguridad FALLÓ - abortando.")
        return
    
    # Paso 2: Confirmar con usuario
    print("\n⚠️  CONFIRMACIÓN REQUERIDA")
    print("-" * 70)
    print("Se eliminarán los siguientes archivos:")
    for filename in FILES_TO_DELETE:
        print(f"   • {filename}")
    print("\nSe MANTENDRÁ:")
    for filename in FILES_TO_KEEP:
        print(f"   • {filename} ✅")
    
    response = input("\n¿Continuar? (si/no): ").strip().lower()
    
    if response not in ['si', 'sí', 's', 'yes', 'y']:
        print("\n❌ Operación cancelada por el usuario.")
        return
    
    # Paso 3: Crear backup
    backup_dir = create_backup()
    
    # Paso 4: Eliminar archivos
    deleted, errors = delete_files()
    
    # Paso 5: Verificar integridad
    if not verify_integrity():
        print("\n❌ Verificación de integridad FALLÓ!")
        print(f"💡 Puedes restaurar desde: {backup_dir}")
        return
    
    # Paso 6: Mostrar resumen
    show_summary()
    
    print("\n" + "=" * 70)
    if errors:
        print("⚠️  LIMPIEZA COMPLETADA CON ERRORES")
        for filename, error in errors:
            print(f"   • {filename}: {error}")
    else:
        print("✅ LIMPIEZA COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    
    print(f"\n📁 Backups guardados en:")
    print(f"   {backup_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operación cancelada por el usuario (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
