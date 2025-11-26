"""
EL-AMANECER-V4 - PYTEST CONFIGURATION
======================================

Configuración global de pytest para todos los tests.
Arregla problemas de importación y configura fixtures compartidas.
"""

import sys
import os
from pathlib import Path
import pytest
from collections import defaultdict

# ===========================
# PATH CONFIGURATION
# ===========================

# Agregar el directorio raíz al PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "apps"))
sys.path.insert(0, str(ROOT_DIR / "packages"))
sys.path.insert(0, str(ROOT_DIR / "tools"))

# Configurar PYTHONPATH en el entorno
os.environ["PYTHONPATH"] = f"{ROOT_DIR};{ROOT_DIR / 'apps'};{ROOT_DIR / 'packages'};{ROOT_DIR / 'tools'}"

print(f"✅ PYTHONPATH configurado: {ROOT_DIR}")


# ===========================
# PYTEST CONFIGURATION HOOKS
# ===========================

def pytest_configure(config):
    """Configuración inicial de pytest"""
    config.addinivalue_line(
        "markers", "enterprise: Enterprise-level tests"
    )
    config.addinivalue_line(
        "markers", "chaos: Chaos engineering tests"
    )
    config.addinivalue_line(
        "markers", "slow: Mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "security: Security and penetration tests"
    )
    config.addinivalue_line(
        "markers", "ui: UI/UX validation tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modificar items de test durante la colección"""
    for item in items:
        # Agregar marker 'enterprise' a todos los tests en tests/enterprise/
        if "enterprise" in str(item.fspath):
            item.add_marker(pytest.mark.enterprise)
        
        # Agregar markers específicos basados en el nombre del archivo
        if "chaos" in str(item.fspath):
            item.add_marker(pytest.mark.chaos)
            item.add_marker(pytest.mark.slow)
        
        if "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        
        if "ui" in str(item.fspath) or "frontend" in str(item.fspath):
            item.add_marker(pytest.mark.ui)
        
        if "performance" in str(item.fspath) or "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


# ===========================
# SHARED FIXTURES
# ===========================

@pytest.fixture(scope="session")
def test_config():
    """Configuración de test compartida"""
    return {
        "base_url": "http://localhost:8000",
        "frontend_url": "http://localhost:3000",
        "timeout": 30,
        "retry_attempts": 3
    }


@pytest.fixture(scope="session")
def mock_dependencies():
    """Mock de dependencias externas para tests que no requieren servicios reales"""
    # Importar numpy solo si está disponible
    try:
        import numpy as np
        numpy_available = True
    except ImportError:
        numpy_available = False
        # Crear mock de numpy para tests que lo necesiten
        class MockNumpy:
            @staticmethod
            def mean(data):
                return sum(data) / len(data) if data else 0
            
            @staticmethod
            def std(data):
                if not data:
                    return 0
                mean_val = sum(data) / len(data)
                variance = sum((x - mean_val) ** 2 for x in data) / len(data)
                return variance ** 0.5
        
        np = MockNumpy()
    
    # Agregar numpy al namespace global para tests
    import builtins
    if not numpy_available:
        builtins.np = np
    
    return {
        "numpy_available": numpy_available,
        "np": np
    }


@pytest.fixture(scope="function")
def temp_test_dir(tmp_path):
    """Directorio temporal para tests"""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(scope="function")
def mock_defaultdict():
    """Fixture para defaultdict (arregla importación faltante)"""
    return defaultdict


# ===========================
# ENTERPRISE TEST FIXTURES
# ===========================

@pytest.fixture(scope="session")
def enterprise_test_environment():
    """Configurar entorno de testing empresarial"""
    # Crear directorios necesarios
    results_dir = ROOT_DIR / "tests" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    (results_dir / "performance").mkdir(exist_ok=True)
    (results_dir / "security").mkdir(exist_ok=True)
    (results_dir / "chaos").mkdir(exist_ok=True)
    
    return {
        "results_dir": results_dir,
        "root_dir": ROOT_DIR
    }


@pytest.fixture(scope="function")
def skip_if_no_browser():
    """Skip test si no hay navegador disponible"""
    try:
        from selenium import webdriver
        return True
    except ImportError:
        pytest.skip("Selenium no disponible - test de UI omitido")


@pytest.fixture(scope="function")
def skip_if_no_service(test_config):
    """Skip test si el servicio no está disponible"""
    import requests
    try:
        response = requests.get(test_config["base_url"] + "/health", timeout=2)
        if response.status_code != 200:
            pytest.skip(f"Servicio no disponible en {test_config['base_url']}")
    except requests.exceptions.RequestException:
        pytest.skip(f"Servicio no disponible en {test_config['base_url']}")


# ===========================
# TEST REPORTING
# ===========================

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook para generar reportes personalizados"""
    outcome = yield
    report = outcome.get_result()
    
    # Agregar información adicional a los reportes
    if report.when == "call":
        if hasattr(item, "funcargs"):
            report.test_metadata = {
                "test_name": item.name,
                "test_file": str(item.fspath),
                "duration": report.duration,
                "outcome": report.outcome
            }


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Resumen personalizado al final de la ejecución de tests"""
    print("\n" + "="*60)
    print(">> EL-AMANECER-V4 TEST EXECUTION SUMMARY")
    print("="*60)
    
    # Estadísticas
    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    skipped = len(terminalreporter.stats.get('skipped', []))
    errors = len(terminalreporter.stats.get('error', []))
    
    total = passed + failed + skipped + errors
    
    if total > 0:
        success_rate = (passed / total) * 100
        print(f"[+] Passed:  {passed}/{total} ({success_rate:.1f}%)")
        print(f"[-] Failed:  {failed}/{total}")
        print(f"[>] Skipped: {skipped}/{total}")
        print(f"[!] Errors:  {errors}/{total}")
        
        if success_rate >= 95:
            print("\n[+] ENTERPRISE QUALITY GATE: PASSED")
        elif success_rate >= 80:
            print("\n[!] ENTERPRISE QUALITY GATE: WARNING")
        else:
            print("\n[-] ENTERPRISE QUALITY GATE: FAILED")
    
    print("="*60 + "\n")


# ===========================
# CLEANUP
# ===========================

@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests():
    """Limpieza automática después de todos los tests"""
    yield
    # Aquí se puede agregar lógica de limpieza si es necesaria
    print("\n[+] Limpieza de tests completada")
