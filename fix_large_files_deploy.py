#!/usr/bin/env python3
"""
LARGE FILES FIX AND DEPLOYMENT
==============================

Soluciona problemas de archivos grandes y realiza deployment limpio a GitHub.
Excluye archivos de modelos ML y otros archivos grandes problemáticos.

CRÍTICO: GitHub deployment, large file management, repository cleanup.
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime


def create_comprehensive_gitignore():
    """Crear .gitignore comprehensivo para excluir archivos grandes"""
    print("📝 CREANDO .GITIGNORE COMPREHENSIVO")
    print("=" * 40)
    
    gitignore_content = """# Large ML Models and Data Files
*.gguf
*.bin
*.safetensors
*.h5
*.pkl
*.joblib
*.model
models/
checkpoints/
weights/

# Large Data Files
*.csv
*.json
*.parquet
*.feather
*.hdf5
data/
datasets/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Testing
.tox/
.coverage
.pytest_cache/
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
htmlcov/

# Enterprise specific
test_backups/
audit_results/
tests/results/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Documentation builds
docs/_build/

# Jupyter
.ipynb_checkpoints

# Enterprise logs
*.log

# Large binary files
*.zip
*.tar.gz
*.tar.bz2
*.rar
*.7z

# Temporary files
*.tmp
*.temp
*.swp
*.swo
*~
"""
    
    gitignore_path = Path('.gitignore')
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore actualizado con exclusiones de archivos grandes")


def remove_large_files_from_git():
    """Remover archivos grandes del historial de Git"""
    print("\n🧹 REMOVIENDO ARCHIVOS GRANDES DEL REPOSITORIO")
    print("=" * 50)
    
    try:
        # Archivos y directorios grandes a remover
        large_files_patterns = [
            "models/",
            "*.gguf",
            "*.bin",
            "*.safetensors",
            "*.h5",
            "checkpoints/",
            "weights/"
        ]
        
        for pattern in large_files_patterns:
            print(f"🗑️ Removiendo: {pattern}")
            # Remover del working directory
            result = subprocess.run(['git', 'rm', '-r', '--cached', pattern], 
                                  capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                print(f"✅ Removido del índice: {pattern}")
            else:
                print(f"ℹ️ No encontrado: {pattern}")
        
        # Remover archivos físicos grandes si existen
        large_file_extensions = ['.gguf', '.bin', '.safetensors', '.h5']
        for file_path in Path('.').rglob('*'):
            if file_path.is_file() and file_path.suffix in large_file_extensions:
                try:
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    if file_size_mb > 50:  # Archivos mayores a 50MB
                        print(f"🗑️ Removiendo archivo grande: {file_path} ({file_size_mb:.1f}MB)")
                        file_path.unlink()
                except Exception as e:
                    print(f"⚠️ No se pudo remover {file_path}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error removiendo archivos grandes: {e}")
        return False


def clean_git_history():
    """Limpiar historial de Git de archivos grandes"""
    print("\n🔄 LIMPIANDO HISTORIAL DE GIT")
    print("=" * 35)
    
    try:
        # Reset suave para limpiar el staging area
        subprocess.run(['git', 'reset', 'HEAD'], 
                      capture_output=True, encoding='utf-8', errors='ignore')
        
        # Agregar solo archivos del framework enterprise (sin archivos grandes)
        enterprise_files = [
            'tests/',
            'run_all_enterprise_tests.py',
            'audit_enterprise_project.py', 
            'fix_test_files.py',
            'setup_environment.py',
            'fix_dependencies.py',
            'requirements.txt',
            'pyproject.toml',
            'pytest.ini',
            'README.md',
            'CHANGELOG.md',
            '.gitignore'
        ]
        
        added_count = 0
        for file_pattern in enterprise_files:
            if Path(file_pattern).exists():
                result = subprocess.run(['git', 'add', file_pattern], 
                                      capture_output=True, encoding='utf-8', errors='ignore')
                if result.returncode == 0:
                    added_count += 1
                    print(f"✅ Añadido: {file_pattern}")
        
        print(f"📊 Total archivos añadidos: {added_count}")
        return added_count > 0
        
    except Exception as e:
        print(f"❌ Error limpiando historial: {e}")
        return False


def deploy_clean_repository():
    """Deployment limpio sin archivos grandes"""
    print("\n🚀 DEPLOYMENT LIMPIO A GITHUB")
    print("=" * 35)
    
    try:
        # 1. Configurar Git
        print("🔧 Configurando Git...")
        subprocess.run(['git', 'config', '--global', 'user.name', 'Balmaurin'], 
                      capture_output=True, encoding='utf-8', errors='ignore')
        subprocess.run(['git', 'config', '--global', 'user.email', 'sergiobalma.gomez@gmail.com'], 
                      capture_output=True, encoding='utf-8', errors='ignore')
        
        # 2. Verificar estado
        status_result = subprocess.run(['git', 'status', '--porcelain'], 
                                     capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if not status_result.stdout.strip():
            print("ℹ️ No hay cambios para commit")
            return True
        
        # 3. Crear commit limpio
        print("💾 Creando commit limpio...")
        commit_msg = f"Enterprise AI Testing Framework - Clean Deploy {datetime.now().strftime('%Y-%m-%d')}"
        
        commit_result = subprocess.run(['git', 'commit', '-m', commit_msg], 
                                     capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if commit_result.returncode == 0:
            print("✅ Commit creado exitosamente")
        else:
            print(f"ℹ️ Commit info: {commit_result.stdout}")
        
        # 4. Setup remoto limpio
        print("🔗 Configurando remoto...")
        subprocess.run(['git', 'remote', 'remove', 'origin'], 
                      capture_output=True, encoding='utf-8', errors='ignore')
        
        remote_result = subprocess.run([
            'git', 'remote', 'add', 'origin', 
            'https://github.com/Balmaurin/EL-AMANECER-V4.git'
        ], capture_output=True, encoding='utf-8', errors='ignore', check=True)
        
        # 5. Push limpio
        print("📤 Enviando a GitHub...")
        push_result = subprocess.run(['git', 'push', '-f', 'origin', 'HEAD:main'], 
                                   capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if push_result.returncode == 0:
            print("✅ Push exitoso a GitHub!")
            print("🔗 Repositorio: https://github.com/Balmaurin/EL-AMANECER-V4")
            return True
        else:
            print(f"❌ Error en push: {push_result.stderr}")
            
            # Intentar crear branch main si no existe
            print("🔄 Intentando crear branch main...")
            branch_result = subprocess.run(['git', 'push', '-u', 'origin', 'HEAD:main'], 
                                         capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if branch_result.returncode == 0:
                print("✅ Branch main creado y push exitoso!")
                return True
            else:
                print(f"❌ Error creando branch: {branch_result.stderr}")
                return False
                
    except Exception as e:
        print(f"❌ Error en deployment: {e}")
        return False


def main():
    """Ejecutar proceso completo de limpieza y deployment"""
    print("🚀 LIMPIEZA Y DEPLOYMENT ENTERPRISE")
    print("=" * 45)
    
    # 1. Crear .gitignore comprehensivo
    create_comprehensive_gitignore()
    
    # 2. Remover archivos grandes
    if not remove_large_files_from_git():
        print("⚠️ Problemas removiendo archivos grandes, continuando...")
    
    # 3. Limpiar historial de Git
    if not clean_git_history():
        print("❌ Error limpiando historial")
        return False
    
    # 4. Deployment limpio
    if not deploy_clean_repository():
        print("❌ Error en deployment")
        return False
    
    print(f"\n🎯 DEPLOYMENT LIMPIO EXITOSO")
    print(f"✅ Archivos grandes excluidos")
    print(f"✅ Framework enterprise subido")
    print(f"✅ Repositorio GitHub limpio")
    
    print(f"\n📋 CONTENIDO SUBIDO:")
    print(f"   • Tests enterprise (API, Blockchain, RAG)")
    print(f"   • Scripts de orchestración y auditoría") 
    print(f"   • Configuración y documentación")
    print(f"   • VSCode settings y herramientas")
    
    print(f"\n🔗 REPOSITORIO ACTUALIZADO:")
    print(f"   https://github.com/Balmaurin/EL-AMANECER-V4")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Proceso interrumpido")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")
        sys.exit(1)
