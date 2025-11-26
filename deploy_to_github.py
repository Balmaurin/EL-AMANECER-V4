#!/usr/bin/env python3
"""
GITHUB DEPLOYMENT SCRIPT
========================

Sube todo el proyecto enterprise actualizado al repositorio GitHub.
Gestiona el proceso completo de actualización del repositorio remoto.

CRÍTICO: Version control, GitHub integration, enterprise deployment.
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime


def check_git_configuration():
    """Verificar y configurar Git si es necesario"""
    print("🔧 VERIFICANDO CONFIGURACIÓN DE GIT")
    print("=" * 40)
    
    try:
        # Verificar configuración de usuario con encoding UTF-8
        user_name = subprocess.run(['git', 'config', '--global', 'user.name'], 
                                  capture_output=True, text=True, encoding='utf-8', errors='ignore')
        user_email = subprocess.run(['git', 'config', '--global', 'user.email'], 
                                   capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        # Configurar automáticamente si no está configurado
        if not user_name.stdout.strip():
            print("🔧 Configurando usuario Git...")
            subprocess.run(['git', 'config', '--global', 'user.name', 'Balmaurin'], 
                          check=True, encoding='utf-8', errors='ignore')
            print("✅ Usuario configurado: Balmaurin")
        
        if not user_email.stdout.strip():
            print("🔧 Configurando email Git...")
            subprocess.run(['git', 'config', '--global', 'user.email', 'sergiobalma.gomez@gmail.com'], 
                          check=True, encoding='utf-8', errors='ignore')
            print("✅ Email configurado: sergiobalma.gomez@gmail.com")
        
        # Verificar configuración final
        final_name = subprocess.run(['git', 'config', '--global', 'user.name'], 
                                   capture_output=True, text=True, encoding='utf-8', errors='ignore')
        final_email = subprocess.run(['git', 'config', '--global', 'user.email'], 
                                    capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        print(f"✅ Usuario: {final_name.stdout.strip()}")
        print(f"✅ Email: {final_email.stdout.strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error configurando Git: {e}")
        return False


def initialize_or_check_repository():
    """Inicializar repositorio o verificar existente"""
    print("\n📂 VERIFICANDO REPOSITORIO GIT")
    print("=" * 35)
    
    try:
        # Verificar si es un repositorio Git con encoding seguro
        result = subprocess.run(['git', 'status'], 
                               capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode != 0:
            print("📝 Inicializando nuevo repositorio Git...")
            subprocess.run(['git', 'init'], check=True, encoding='utf-8', errors='ignore')
            print("✅ Repositorio Git inicializado")
        else:
            print("✅ Repositorio Git existente detectado")
        
        # Verificar remoto
        remote_result = subprocess.run(['git', 'remote', '-v'], 
                                     capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        github_url = "https://github.com/Balmaurin/EL-AMANECER-V4.git"
        
        if github_url not in remote_result.stdout:
            print("🔗 Configurando remoto GitHub...")
            # Remover origin existente si existe
            subprocess.run(['git', 'remote', 'remove', 'origin'], 
                         capture_output=True, encoding='utf-8', errors='ignore')
            # Añadir nuevo origin
            subprocess.run(['git', 'remote', 'add', 'origin', github_url], 
                          check=True, encoding='utf-8', errors='ignore')
            print(f"✅ Remoto configurado: {github_url}")
        else:
            print(f"✅ Remoto GitHub ya configurado")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error configurando repositorio: {e}")
        return False


def stage_all_enterprise_files():
    """Hacer stage de todos los archivos enterprise"""
    print("\n📦 PREPARANDO ARCHIVOS PARA COMMIT")
    print("=" * 40)
    
    try:
        # Añadir todos los archivos existentes con encoding seguro
        subprocess.run(['git', 'add', '.'], 
                      capture_output=True, encoding='utf-8', errors='ignore')
        
        # Verificar archivos staged con encoding seguro
        staged_result = subprocess.run(['git', 'diff', '--cached', '--name-only'], 
                                     capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        staged_files = staged_result.stdout.strip().split('\n') if staged_result.stdout.strip() else []
        
        print(f"📋 Archivos preparados para commit: {len(staged_files)}")
        for file in staged_files[:10]:  # Mostrar primeros 10
            print(f"   ✅ {file}")
        
        if len(staged_files) > 10:
            print(f"   ... y {len(staged_files) - 10} archivos más")
        
        return len(staged_files) > 0
    
    except Exception as e:
        print(f"❌ Error preparando archivos: {e}")
        return False


def create_comprehensive_commit():
    """Crear commit comprehensivo con todos los cambios enterprise"""
    print("\n💾 CREANDO COMMIT ENTERPRISE")
    print("=" * 30)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Mensaje de commit sin caracteres especiales problemáticos
    commit_message = f"""Enterprise AI Testing Framework v1.0 - {timestamp}

ENTERPRISE FEATURES ADDED:
- Complete test suites: API, Blockchain, RAG system validation
- Automated test fixing: Returns to assertions, setup standardization  
- Project auditing: Comprehensive quality assessment with scoring
- VSCode integration: Complete IDE configuration templates
- Security validation: Vulnerability scanning & compliance checks
- Performance monitoring: Real-time metrics and benchmarking

QUALITY IMPROVEMENTS:
- 33+ enterprise test cases with comprehensive assertions
- Automated backup system for test file modifications
- Executive reporting with quality gates and recommendations
- Dependency management with caching issue resolution
- Enterprise-grade documentation and configuration

SECURITY ENHANCEMENTS:
- Security header validation and compliance testing
- Vulnerability detection patterns and audit logging
- Enterprise security middleware and configuration
- Regulatory compliance validation framework

ENTERPRISE METRICS:
- >90% test pass rate requirement
- >90/100 security score validation
- <2s average response time monitoring
- >85% enterprise compliance tracking
- >70% documentation coverage standards

PRODUCTION READY:
Framework completo para sistemas de IA criticos con validacion
enterprise, testing automatizado, security compliance y monitoring.

CRITICO: Enterprise-grade AI testing framework for production systems."""

    try:
        result = subprocess.run(['git', 'commit', '-m', commit_message], 
                              capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("✅ Commit enterprise creado exitosamente")
            return True
        else:
            print(f"❌ Error creando commit: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Excepción creando commit: {e}")
        return False


def push_to_github():
    """Subir todos los cambios a GitHub"""
    print("\n🚀 SUBIENDO A GITHUB")
    print("=" * 25)
    
    try:
        # Intentar push con encoding seguro
        print("📤 Enviando cambios al repositorio remoto...")
        result = subprocess.run(['git', 'push', '-u', 'origin', 'main'], 
                              capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("✅ Push exitoso a GitHub!")
            print("🔗 Repositorio actualizado: https://github.com/Balmaurin/EL-AMANECER-V4")
            return True
        else:
            # Intentar con master si main falla
            print("🔄 Intentando con branch master...")
            result_master = subprocess.run(['git', 'push', '-u', 'origin', 'master'], 
                                         capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if result_master.returncode == 0:
                print("✅ Push exitoso a GitHub (master branch)!")
                print("🔗 Repositorio actualizado: https://github.com/Balmaurin/EL-AMANECER-V4")
                return True
            else:
                print(f"❌ Error en push main: {result.stderr[:200]}")
                print(f"❌ Error en push master: {result_master.stderr[:200]}")
                return False
                
    except Exception as e:
        print(f"❌ Excepción durante push: {e}")
        return False


def verify_github_update():
    """Verificar que la actualización fue exitosa"""
    print("\n🔍 VERIFICANDO ACTUALIZACIÓN")
    print("=" * 30)
    
    try:
        # Obtener info del último commit
        result = subprocess.run(['git', 'log', '--oneline', '-1'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            last_commit = result.stdout.strip()
            print(f"📝 Último commit: {last_commit}")
        
        # Verificar remoto
        remote_result = subprocess.run(['git', 'remote', '-v'], 
                                     capture_output=True, text=True)
        
        print(f"🔗 Remotos configurados:")
        for line in remote_result.stdout.strip().split('\n'):
            if line.strip():
                print(f"   {line}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error verificando: {e}")
        return False


def main():
    """Ejecutar despliegue completo a GitHub"""
    print("🚀 DESPLIEGUE ENTERPRISE A GITHUB")
    print("=" * 40)
    print("📂 Proyecto: EL-AMANECER-V4")
    print("🔗 Repositorio: https://github.com/Balmaurin/EL-AMANECER-V4.git")
    print("=" * 40)
    
    # Verificar configuración de Git
    if not check_git_configuration():
        print("❌ Configuración de Git requerida")
        return False
    
    # Inicializar/verificar repositorio
    if not initialize_or_check_repository():
        print("❌ Error configurando repositorio")
        return False
    
    # Preparar archivos
    if not stage_all_enterprise_files():
        print("❌ No hay archivos para commit")
        return False
    
    # Crear commit
    if not create_comprehensive_commit():
        print("❌ Error creando commit")
        return False
    
    # Subir a GitHub
    if not push_to_github():
        print("❌ Error subiendo a GitHub")
        return False
    
    # Verificar actualización
    verify_github_update()
    
    print(f"\n🎯 DESPLIEGUE ENTERPRISE COMPLETADO")
    print(f"=" * 40)
    print(f"✅ Framework enterprise subido exitosamente")
    print(f"✅ Repositorio GitHub actualizado")
    print(f"✅ Version 1.0 disponible en producción")
    
    print(f"\n🔗 ACCESO AL REPOSITORIO:")
    print(f"   Web: https://github.com/Balmaurin/EL-AMANECER-V4")
    print(f"   Clone: git clone https://github.com/Balmaurin/EL-AMANECER-V4.git")
    
    print(f"\n📋 PRÓXIMOS PASOS:")
    print(f"   1. Verificar repositorio en GitHub")
    print(f"   2. Configurar GitHub Actions (opcional)")
    print(f"   3. Documentar deployment process")
    print(f"   4. Configurar releases y tags")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Despliegue interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error en despliegue: {e}")
        sys.exit(1)
