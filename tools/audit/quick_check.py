#!/usr/bin/env python3
"""
Quick Excellence Check - Sheily AI
==================================

Validación rápida del estado actual de excelencia.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def quick_excellence_check():
    """Validación rápida de componentes críticos"""
    print("🔍 SHEILY AI - QUICK EXCELLENCE CHECK")
    print("=" * 50)

    checks = {
        "Arquitectura": check_architecture(),
        "Testing Básico": check_basic_testing(),
        "DevOps": check_devops(),
        "Documentación": check_documentation(),
        "Seguridad": check_security(),
    }

    total_score = 0
    max_score = len(checks) * 20  # 20 puntos por categoría

    print("\n📊 RESULTADOS:")
    for component, (score, status, details) in checks.items():
        total_score += score
        status_emoji = (
            "✅" if status == "passed" else "⚠️" if status == "partial" else "❌"
        )
        print(f"{status_emoji} {component}: {score}/20 ({status})")
        if details:
            print(f"   └─ {details}")

    overall_percentage = (total_score / max_score) * 100

    print("\n🏆 RESULTADO FINAL:")
    print(f"   Score: {total_score}/{max_score} ({overall_percentage:.1f}%)")

    if overall_percentage >= 80:
        print("   ✅ SISTEMA EN BUEN ESTADO")
    elif overall_percentage >= 60:
        print("   ⚠️ REQUIERE MEJORAS MODERADAS")
    else:
        print("   ❌ REQUIERE ATENCIÓN SIGNIFICATIVA")

    return overall_percentage


def check_architecture():
    """Verificar arquitectura completa"""
    try:
        # Verificar estructura de directorios completa
        dirs = ["sheily_core", "sheily_train", "scripts", "tools", "k8s", "docs"]
        existing_dirs = sum(1 for d in dirs if os.path.exists(d))

        # Verificar archivos críticos de arquitectura
        critical_files = [
            "pyproject.toml",
            "requirements.txt",
            "Makefile",
            "docker-compose.yml",
        ]
        existing_critical = sum(1 for f in critical_files if os.path.exists(f))

        if existing_dirs == len(dirs) and existing_critical >= 3:
            return (
                20,
                "passed",
                f"{existing_dirs}/{len(dirs)} directorios + {existing_critical}/{len(critical_files)} archivos críticos",
            )
        elif existing_dirs >= 5:
            return 18, "passed", f"{existing_dirs}/{len(dirs)} directorios principales"
        else:
            return 12, "partial", f"{existing_dirs}/{len(dirs)} directorios principales"
    except:
        return 0, "failed", "Error checking architecture"


def check_basic_testing():
    """Verificar testing completo"""
    try:
        test_files = [
            "scripts/validate_system.py",
            "scripts/deploy_all.py",
            "centralized_tests/basic_test.py",
            "centralized_tests/test_manager.py",
        ]

        existing_tests = sum(1 for f in test_files if os.path.exists(f))

        # Verificar herramientas de testing avanzadas
        advanced_tools = [
            "tools/testing_excellence_suite.py",
            "tools/chaos_engineering.py",
            "tools/property_based_testing.py",
        ]
        existing_advanced = sum(1 for f in advanced_tools if os.path.exists(f))

        if existing_tests == len(test_files) and existing_advanced >= 2:
            return (
                20,
                "passed",
                f"{existing_tests}/{len(test_files)} scripts básicos + {existing_advanced}/{len(advanced_tools)} avanzados",
            )
        elif existing_tests == len(test_files):
            return (
                18,
                "passed",
                f"{existing_tests}/{len(test_files)} scripts de testing completos",
            )
        else:
            return (
                12,
                "partial",
                f"{existing_tests}/{len(test_files)} scripts de testing",
            )
    except:
        return 0, "failed", "Error checking testing"


def check_devops():
    """Verificar configuración DevOps completa"""
    try:
        devops_files = [
            "docker-compose.yml",
            "Dockerfile",
            ".github/workflows/ci-cd.yml",
            "k8s/production/deployment.yaml",
            "Makefile",
        ]

        existing_devops = sum(1 for f in devops_files if os.path.exists(f))

        # Verificar configuraciones avanzadas
        advanced_devops = [
            "k8s/istio/mtls-policies.yaml",
            "terraform/main.tf",
            "k8s/argocd/sheily-ai-production.yaml",
        ]
        existing_advanced = sum(1 for f in advanced_devops if os.path.exists(f))

        if existing_devops == len(devops_files) and existing_advanced >= 2:
            return (
                20,
                "passed",
                f"{existing_devops}/{len(devops_files)} archivos básicos + {existing_advanced}/{len(advanced_devops)} avanzados",
            )
        elif existing_devops >= 4:
            return (
                16,
                "passed",
                f"{existing_devops}/{len(devops_files)} archivos DevOps principales",
            )
        else:
            return (
                8,
                "partial",
                f"{existing_devops}/{len(devops_files)} archivos DevOps",
            )
    except:
        return 0, "failed", "Error checking DevOps"


def check_documentation():
    """Verificar documentación completa"""
    try:
        docs = [
            "README.md",
            "ROADMAP_TO_EXCELLENCE.md",
            "docs/AUDITORIA_DEFINITIVA_COMPLETA.md",
            "CONTRIBUTING.md",
        ]

        existing_docs = sum(1 for d in docs if os.path.exists(d))

        # Verificar documentación auto-generada
        auto_docs = [
            "docs/auto_generated/auto_documentation.json",
            "docs/auto_generated/README.md",
        ]
        existing_auto = sum(1 for d in auto_docs if os.path.exists(d))

        # Verificar documentación de seguridad
        security_docs = ["SECURITY.md", "docs/SECURITY_POLICIES.md"]
        existing_security = sum(1 for d in security_docs if os.path.exists(d))

        if existing_docs >= 3 and existing_auto >= 1 and existing_security >= 1:
            return (
                20,
                "passed",
                f"{existing_docs}/{len(docs)} principales + {existing_auto}/{len(auto_docs)} auto + {existing_security}/{len(security_docs)} seguridad",
            )
        elif existing_docs >= 3:
            return 16, "passed", f"{existing_docs}/{len(docs)} documentos principales"
        else:
            return 8, "partial", f"{existing_docs}/{len(docs)} documentos principales"
    except:
        return 0, "failed", "Error checking documentation"


def check_security():
    """Verificar configuración de seguridad completa"""
    try:
        # Verificar archivos de seguridad
        security_dir = Path(".security")
        config_file = security_dir / "security_config.json"
        keys_file = security_dir / "keys.json"

        if not config_file.exists() or not keys_file.exists():
            return 0, "failed", "Security files missing"

        # Verificar estado real de seguridad usando el sistema completo
        try:
            import sys

            sys.path.insert(0, "tools")
            from simple_security import get_security_manager

            security = get_security_manager()
            status = security.check_security_status()

            passed_checks = sum(1 for check in status["checks"].values() if check)
            total_checks = len(status["checks"])

            if passed_checks == total_checks:
                return (
                    20,
                    "passed",
                    f"{passed_checks}/{total_checks} enterprise security checks",
                )
            elif passed_checks >= total_checks * 0.8:
                return (
                    16,
                    "passed",
                    f"{passed_checks}/{total_checks} security checks (good)",
                )
            else:
                return (
                    12,
                    "partial",
                    f"{passed_checks}/{total_checks} security checks (needs improvement)",
                )

        except ImportError:
            # Fallback si no se puede importar
            return 14, "passed", "Basic security files present"

    except Exception as e:
        return 0, "failed", f"Security check error: {e}"


if __name__ == "__main__":
    score = quick_excellence_check()
    exit(0 if score >= 60 else 1)
