#!/usr/bin/env python3
"""
FORCE CLEAN DEPLOYMENT
=====================

Solución definitiva para archivos grandes. Crea un repositorio completamente
limpio con solo el framework enterprise, sin historial problemático.

CRÍTICO: Clean deployment, large file removal, fresh repository.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime


def delete_git_history():
    """Eliminar completamente el historial de Git problemático"""
    print("🗑️ ELIMINANDO HISTORIAL GIT PROBLEMÁTICO")
    print("=" * 45)
    
    try:
        # Eliminar directorio .git completamente
        git_dir = Path('.git')
        if git_dir.exists():
            print("🧹 Removiendo directorio .git...")
            shutil.rmtree(git_dir, ignore_errors=True)
            print("✅ Historial Git eliminado")
        
        return True
    except Exception as e:
        print(f"❌ Error eliminando .git: {e}")
        return False


def remove_large_files_physically():
    """Remover físicamente todos los archivos grandes"""
    print("\n🗑️ REMOVIENDO ARCHIVOS GRANDES FÍSICAMENTE")
    print("=" * 45)
    
    try:
        # Directorios y archivos a eliminar
        large_items = [
            'models/',
            'checkpoints/',
            'weights/',
            'data/',
            'datasets/',
            '__pycache__/',
            '.pytest_cache/',
            'test_backups/',
            'audit_results/'
        ]
        
        # Extensiones de archivos grandes a eliminar
        large_extensions = ['.gguf', '.bin', '.safetensors', '.h5', '.pkl', 
                          '.joblib', '.model', '.csv', '.json', '.log']
        
        removed_count = 0
        
        # Remover directorios específicos
        for item in large_items:
            item_path = Path(item)
            if item_path.exists():
                try:
                    if item_path.is_dir():
                        shutil.rmtree(item_path, ignore_errors=True)
                        print(f"✅ Directorio removido: {item}")
                    else:
                        item_path.unlink()
                        print(f"✅ Archivo removido: {item}")
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ No se pudo remover {item}: {e}")
        
        # Buscar y remover archivos por extensión
        for ext in large_extensions:
            for file_path in Path('.').rglob(f'*{ext}'):
                try:
                    if file_path.is_file():
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        if file_size_mb > 1:  # Archivos mayores a 1MB
                            file_path.unlink()
                            print(f"✅ Archivo grande removido: {file_path} ({file_size_mb:.1f}MB)")
                            removed_count += 1
                except Exception as e:
                    print(f"⚠️ Error removiendo {file_path}: {e}")
        
        print(f"📊 Total items removidos: {removed_count}")
        return True
        
    except Exception as e:
        print(f"❌ Error removiendo archivos: {e}")
        return False


def create_enterprise_only_structure():
    """Crear estructura con solo archivos del framework enterprise"""
    print("\n📂 CREANDO ESTRUCTURA ENTERPRISE LIMPIA")
    print("=" * 45)
    
    # Asegurar que solo existen los directorios necesarios
    essential_dirs = ['tests/enterprise', '.vscode']
    for dir_path in essential_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Estructura enterprise preparada")


def create_comprehensive_gitignore():
    """Crear .gitignore ultra-comprehensivo"""
    print("\n📝 CREANDO .GITIGNORE ULTRA-COMPREHENSIVO")
    print("=" * 50)
    
    gitignore_content = """# LARGE FILES - NEVER COMMIT
*.gguf
*.bin
*.safetensors
*.h5
*.pkl
*.joblib
*.model
*.weights
*.checkpoint

# DIRECTORIES TO IGNORE
models/
checkpoints/
weights/
data/
datasets/
logs/
cache/
temp/

# PYTHON
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
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# TESTING
.tox/
.coverage
.pytest_cache/
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
htmlcov/

# ENTERPRISE GENERATED
test_backups/
audit_results/
tests/results/

# ENVIRONMENTS
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# LARGE DATA FILES
*.csv
*.json
*.parquet
*.feather
*.hdf5

# LOGS
*.log
*.out
*.err

# COMPRESSED
*.zip
*.tar.gz
*.tar.bz2
*.rar
*.7z

# TEMPORARY
*.tmp
*.temp
*.swp
*.swo

# JUPYTER
.ipynb_checkpoints

# NODE (if any)
node_modules/
npm-debug.log*

# ANY FILE > 50MB
**/large_*
**/big_*
**/huge_*
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore ultra-comprehensivo creado")


def initialize_fresh_repository():
    """Inicializar repositorio completamente nuevo"""
    print("\n🆕 INICIALIZANDO REPOSITORIO FRESCO")
    print("=" * 40)
    
    try:
        # 1. Inicializar Git nuevo
        subprocess.run(['git', 'init'], check=True, 
                      capture_output=True, encoding='utf-8', errors='ignore')
        print("✅ Repositorio Git inicializado")
        
        # 2. Configurar usuario
        subprocess.run(['git', 'config', 'user.name', 'Balmaurin'], check=True,
                      capture_output=True, encoding='utf-8', errors='ignore')
        subprocess.run(['git', 'config', 'user.email', 'sergiobalma.gomez@gmail.com'], check=True,
                      capture_output=True, encoding='utf-8', errors='ignore')
        print("✅ Usuario Git configurado")
        
        # 3. Añadir solo archivos del framework
        enterprise_files = [
            'tests/enterprise/test_blockchain_enterprise.py',
            'tests/enterprise/test_api_enterprise_suites.py',
            'tests/enterprise/test_rag_system_enterprise.py',
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
        
        added_files = []
        for file_path in enterprise_files:
            if Path(file_path).exists():
                subprocess.run(['git', 'add', file_path], 
                             capture_output=True, encoding='utf-8', errors='ignore')
                added_files.append(file_path)
        
        print(f"📦 Archivos añadidos: {len(added_files)}")
        
        # 4. Commit inicial
        commit_msg = f"Enterprise AI Testing Framework - Clean Deploy {datetime.now().strftime('%Y-%m-%d')}"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True,
                      capture_output=True, encoding='utf-8', errors='ignore')
        print("✅ Commit inicial creado")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error inicializando repositorio: {e}")
        return False


def force_push_to_github():
    """Push forzado al repositorio GitHub"""
    print("\n🚀 FORCE PUSH A GITHUB")
    print("=" * 30)
    
    try:
        # 1. Añadir remoto
        subprocess.run(['git', 'remote', 'add', 'origin', 
                       'https://github.com/Balmaurin/EL-AMANECER-V4.git'], 
                      capture_output=True, encoding='utf-8', errors='ignore')
        print("✅ Remoto configurado")
        
        # 2. Force push para sobrescribir historial problemático
        push_result = subprocess.run(['git', 'push', '--force', '--set-upstream', 'origin', 'main'], 
                                   capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if push_result.returncode == 0:
            print("✅ Force push exitoso!")
            print("🔗 Repositorio limpio: https://github.com/Balmaurin/EL-AMANECER-V4")
            return True
        else:
            # Si main no funciona, intentar con master
            push_result = subprocess.run(['git', 'push', '--force', '--set-upstream', 'origin', 'master'], 
                                       capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if push_result.returncode == 0:
                print("✅ Force push exitoso (master)!")
                return True
            else:
                print(f"❌ Error en force push: {push_result.stderr}")
                return False
        
    except Exception as e:
        print(f"❌ Error en push: {e}")
        return False


def main():
    """Ejecutar limpieza completa y deployment forzado"""
    print("🚀 FORCE CLEAN DEPLOYMENT TO GITHUB")
    print("=" * 45)
    print("⚠️ ADVERTENCIA: Esto eliminará TODO el historial Git")
    print("=" * 45)
    
    # 1. Eliminar historial Git problemático
    if not delete_git_history():
        return False
    
    # 2. Remover archivos grandes físicamente
    if not remove_large_files_physically():
        return False
    
    # 3. Crear estructura enterprise limpia
    create_enterprise_only_structure()
    
    # 4. Crear .gitignore comprehensivo
    create_comprehensive_gitignore()
    
    # 5. Inicializar repositorio fresco
    if not initialize_fresh_repository():
        return False
    
    # 6. Force push a GitHub
    if not force_push_to_github():
        return False
    
    print(f"\n🎯 DEPLOYMENT LIMPIO EXITOSO")
    print(f"=" * 35)
    print(f"✅ Historial problemático eliminado")
    print(f"✅ Archivos grandes removidos")
    print(f"✅ Repositorio GitHub limpio")
    print(f"✅ Solo framework enterprise subido")
    
    print(f"\n📋 CONTENIDO FINAL:")
    print(f"   • Tests enterprise (API, Blockchain, RAG)")
    print(f"   • Scripts de orchestración")
    print(f"   • Documentación y configuración")
    print(f"   • Sin archivos grandes ni historial problemático")
    
    print(f"\n🔗 REPOSITORIO LIMPIO:")
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
