#!/usr/bin/env python3
"""
GIT CONFIGURATION SETUP
=======================

Configura automáticamente Git con las credenciales enterprise
para el repositorio EL-AMANECER-V4.

CRÍTICO: Git configuration, user credentials, repository setup.
"""

import subprocess
import sys


def configure_git_credentials():
    """Configurar credenciales de Git para el proyecto"""
    print("🔧 CONFIGURANDO CREDENCIALES GIT")
    print("=" * 35)
    
    try:
        # Configurar usuario
        subprocess.run([
            'git', 'config', '--global', 'user.name', 'Balmaurin'
        ], check=True)
        print("✅ Usuario configurado: Balmaurin")
        
        # Configurar email
        subprocess.run([
            'git', 'config', '--global', 'user.email', 'sergiobalma.gomez@gmail.com'
        ], check=True)
        print("✅ Email configurado: sergiobalma.gomez@gmail.com")
        
        # Configurar editor por defecto
        subprocess.run([
            'git', 'config', '--global', 'core.editor', 'code --wait'
        ], capture_output=True)
        print("✅ Editor configurado: VS Code")
        
        # Configurar credencial helper para Windows
        subprocess.run([
            'git', 'config', '--global', 'credential.helper', 'manager-core'
        ], capture_output=True)
        print("✅ Credential helper configurado")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error configurando Git: {e}")
        return False


def verify_git_configuration():
    """Verificar que Git está correctamente configurado"""
    print("\n🔍 VERIFICANDO CONFIGURACIÓN")
    print("=" * 30)
    
    try:
        # Verificar usuario
        user_result = subprocess.run([
            'git', 'config', '--global', 'user.name'
        ], capture_output=True, text=True)
        
        # Verificar email
        email_result = subprocess.run([
            'git', 'config', '--global', 'user.email'
        ], capture_output=True, text=True)
        
        print(f"👤 Usuario: {user_result.stdout.strip()}")
        print(f"📧 Email: {email_result.stdout.strip()}")
        
        # Verificar que coinciden con los valores esperados
        expected_user = "Balmaurin"
        expected_email = "sergiobalma.gomez@gmail.com"
        
        if user_result.stdout.strip() == expected_user:
            print("✅ Usuario correcto")
        else:
            print(f"⚠️ Usuario incorrecto: esperado {expected_user}")
        
        if email_result.stdout.strip() == expected_email:
            print("✅ Email correcto")
        else:
            print(f"⚠️ Email incorrecto: esperado {expected_email}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando configuración: {e}")
        return False


def main():
    """Ejecutar configuración completa de Git"""
    print("🚀 CONFIGURACIÓN GIT ENTERPRISE")
    print("=" * 35)
    
    # Configurar credenciales
    if not configure_git_credentials():
        print("❌ Error en configuración")
        return False
    
    # Verificar configuración
    if not verify_git_configuration():
        print("❌ Error en verificación")
        return False
    
    print(f"\n🎯 CONFIGURACIÓN GIT COMPLETA")
    print(f"✅ Listo para deploy a GitHub")
    print(f"📂 Repositorio: EL-AMANECER-V4")
    print(f"🔗 URL: https://github.com/Balmaurin/EL-AMANECER-V4.git")
    
    print(f"\n📋 Próximo paso:")
    print(f"   python deploy_to_github.py")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Error en configuración: {e}")
        sys.exit(1)
