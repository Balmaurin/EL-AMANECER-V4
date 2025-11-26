#!/usr/bin/env python
"""
Diagnóstico básico del framework de testing
==========================================

Script simple para identificar problemas de configuración.
"""

def main():
    print("🔍 Diagnóstico de Testing EL-AMANECER")
    print("=" * 50)

    # 1. Python version
    import sys
    print(f"Python: {sys.version}")

    # 2. Check dependencies
    deps = {
        'pytest': 'pytest',
        'psutil': 'psutil',
        'numpy': 'numpy',
        'requests': 'requests'
    }

    all_ok = True
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"✓ {name}: OK")
        except ImportError:
            print(f"✗ {name}: MISSING")
            all_ok = False

    # 3. Check test structure
    import os
    if os.path.exists('tests/__init__.py'):
        print("✓ tests/__init__.py: exists")
    else:
        print("✗ tests/__init__.py: missing")
        all_ok = False

    # 4. Config files
    if os.path.exists('pytest.ini'):
        print("✓ pytest.ini: exists")
    else:
        print("✗ pytest.ini: missing")
        all_ok = False

    if os.path.exists('pyproject.toml'):
        print("✓ pyproject.toml: exists")
    else:
        print("✗ pyproject.toml: missing")
        all_ok = False

    # 5. Test discovery
    test_files = []
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))

    print(f"✓ Test files found: {len(test_files)}")

    # Final verdict
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ Sistema de testing: CONFIGURADO CORRECTAMENTE")
        print("\nPróximos pasos:")
        print("  1. Ejecutar: python -m pytest tests/ -v")
        print("  2. Ejecutar: python -m pytest tests/enterprise/ -v")
        print("  3. Revisar resultados de coverage")
    else:
        print("❌ Sistema de testing: PROBLEMAS DETECTADOS")
        print("\nProblemas a solucionar:")
        if not all_ok:
            print("  - Instalar dependencias faltantes")
            print("  - Verificar archivos de configuración")
            print("  - Crear archivos de test faltantes")

if __name__ == "__main__":
    main()
