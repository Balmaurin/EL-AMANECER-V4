#!/usr/bin/env python3
"""
MASTER RUNNER PARA SUITES DE TESTING ENTERPRISE
=============================================

Ejecuta todas las suites de testing enterprise de EL-AMANECER V3
con calidad máxima y reporting completo.

Uso: python run_all_enterprise_tests.py
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import json


class EnterpriseTestRunner:
    """Master test runner for all enterprise test suites"""

    def __init__(self):
        self.start_time = time.time()
        self.results = {
            'total_suites': 0,
            'total_tests': 0,
            'total_passed': 0,
            'total_failed': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'suite_results': [],
            'performance_metrics': {},
            'coverage_summary': {}
        }

    def run_all_suites(self):
        """Execute all enterprise test suites"""
        print("🏆 EL-AMANECER V3 - ENTERPRISE TESTING SUITES")
        print("=" * 70)
        print("🔬 Ejecutando suites de máxima calidad enterprise")
        print(f"⏰ Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Test suites to run
        test_suites = [
            {
                'name': 'Core Consciousness Enterprise',
                'file': 'tests/enterprise/test_core_consciousness_suites.py',
                'description': 'Φ calculations, theory integration, edge cases, performance'
            },
            {
                'name': 'API Enterprise',
                'file': 'tests/enterprise/test_api_enterprise_suites.py',
                'description': 'REST API testing, authentication, security, load'
            },
            {
                'name': 'Consciousness Integration (Enhanced)',
                'file': 'tests/enterprise/test_consciousness_integration.py',
                'description': 'IIT+GWT, metrics, scientific validation'
            },
            {
                'name': 'Performance Benchmarks',
                'file': 'tests/enterprise/test_performance_benchmarks_enterprise.py',
                'description': 'Latency, throughput, memory analysis'
            },
            {
                'name': 'RAG Enterprise System',
                'file': 'tests/enterprise/test_rag_enterprise_system.py',
                'description': 'Retrieval augmented generation, knowledge base'
            },
            {
                'name': 'Security Enterprise',
                'file': 'tests/enterprise/test_security_enterprise.py',
                'description': 'Authentication, encryption, vulnerability testing'
            }
        ]

        for suite in test_suites:
            self.run_single_suite(suite)

        self.print_final_report()

        # Return success based on enterprise quality standards
        return self.assess_enterprise_quality()

    def run_single_suite(self, suite_config: dict):
        """Run a single test suite"""
        suite_start = time.time()

        print(f"🔬 EJECUTANDO SUITE: {suite_config['name']}")
        print(f"📄 Archivo: {suite_config['file']}")
        print(f"📋 Descripción: {suite_config['description']}")
        print("-" * 60)

        suite_result = {
            'name': suite_config['name'],
            'file': suite_config['file'],
            'start_time': suite_start,
            'status': 'running'
        }

        if not Path(suite_config['file']).exists():
            print(f"❌ SUITE NO ENCONTRADO: {suite_config['file']}")
            suite_result.update({
                'status': 'not_found',
                'duration': time.time() - suite_start,
                'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0
            })
            self.results['suite_results'].append(suite_result)
            return

        try:
            # Execute with enterprise quality parameters
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                suite_config['file'],
                "-v",  # Verbose output
                "--tb=short",  # Short traceback
                "--durations=5",  # Show 5 slowest tests
                "--maxfail=5",  # Stop after 5 failures
                "--strict-markers",  # Strict marker validation
                "--disable-warnings",  # Clean output
                "--color=yes",  # Colored output
                "--cov-report=",  # No coverage for speed
                "--capture=no",  # Show print statements
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout

            # Parse pytest output
            suite_metrics = self.parse_pytest_output(result.stdout, result.stderr)

            suite_result.update({
                'status': 'completed',
                'duration': time.time() - suite_start,
                'returncode': result.returncode,
                **suite_metrics
            })

            # Update global metrics
            self.results['total_passed'] += suite_metrics['passed']
            self.results['total_failed'] += suite_metrics['failed']
            self.results['total_skipped'] += suite_metrics['skipped']
            self.results['total_errors'] += suite_metrics['errors']
            self.results['total_tests'] += suite_metrics['total']
            self.results['total_suites'] += 1

            # Print suite summary
            self.print_suite_summary(suite_result)

        except subprocess.TimeoutExpired:
            print("⏰ SUITE TIMEOUT: Excedió límite de 10 minutos")
            suite_result.update({
                'status': 'timeout',
                'duration': time.time() - suite_start,
                'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0
            })

        except Exception as e:
            print(f"❌ ERROR EJECUTANDO SUITE: {e}")
            suite_result.update({
                'status': 'error',
                'duration': time.time() - suite_start,
                'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 1
            })

        self.results['suite_results'].append(suite_result)
        print()

    def parse_pytest_output(self, stdout: str, stderr: str) -> dict:
        """Parse pytest output to extract metrics"""
        metrics = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'warnings': 0
        }

        lines = stdout.split('\n')

        # Find summary line
        for line in lines:
            line = line.strip()
            if '===' in line and ('failed' in line or 'passed' in line):
                continue  # Skip separator lines

            # Look for final summary like: "10 passed, 2 failed, 1 skipped"
            if ('passed' in line or 'failed' in line or 'skipped' in line or 'errors' in line):
                parts = line.replace(',', '').split()
                i = 0
                while i < len(parts):
                    try:
                        count = int(parts[i])
                        if i + 1 < len(parts):
                            category = parts[i + 1].lower()
                            if category.startswith('passed'):
                                metrics['passed'] = count
                            elif category.startswith('failed'):
                                metrics['failed'] = count
                            elif category.startswith('skipped'):
                                metrics['skipped'] = count
                            elif category.startswith('error'):
                                metrics['errors'] = count
                        i += 2
                    except ValueError:
                        i += 1

                metrics['total'] = metrics['passed'] + metrics['failed'] + metrics['skipped'] + metrics['errors']
                break

        # Check for errors in stderr
        if stderr and len(stderr.strip()) > 0:
            error_lines = stderr.strip().split('\n')
            metrics['errors'] += len([line for line in error_lines if 'ERROR' in line or 'FAILED' in line])

        return metrics

    def print_suite_summary(self, suite_result: dict):
        """Print formatted suite summary"""
        duration = suite_result.get('duration', 0)
        passed = suite_result.get('passed', 0)
        failed = suite_result.get('failed', 0)
        skipped = suite_result.get('skipped', 0)
        errors = suite_result.get('errors', 0)
        status = suite_result.get('status', 'unknown')

        total_tests = passed + failed + skipped + errors

        # Status indicator
        if status == 'completed':
            status_icon = "✅" if failed == 0 else "⚠️" if failed < 3 else "❌"
        elif status == 'not_found':
            status_icon = "📁"
        elif status == 'timeout':
            status_icon = "⏰"
        else:
            status_icon = "❓"

        print(f"{status_icon} Status: {status.title()}")
        print(f"⏱️  Duración: {duration:.2f}s")

        if total_tests > 0:
            success_rate = (passed / total_tests) * 100
            print(f"📊 Tests: {total_tests} total | {passed} ✅ | {failed} ❌ | {skipped} ⏭️ | {errors} 🚨")
            print(f"🎯 Tasa de éxito: {success_rate:.1f}%")
        else:
            print("📊 Tests: No tests found")

    def print_final_report(self):
        """Print comprehensive final report"""
        total_duration = time.time() - self.start_time

        print("=" * 70)
        print("🏆 REPORTE FINAL - SUITES DE TESTING ENTERPRISE")
        print("=" * 70)
        print(f"🔬 Suites ejecutadas: {self.results['total_suites']}")
        print(f"📊 Tests totales: {self.results['total_tests']}")
        print(f"✅ Tests exitosos: {self.results['total_passed']}")
        print(f"❌ Tests fallidos: {self.results['total_failed']}")
        print(f"⏭️  Tests omitidos: {self.results['total_skipped']}")
        print(f"🚨 Errores: {self.results['total_errors']}")
        print(f"⏱️  Duración total: {total_duration:.2f}s")
        print(f"📅 Fin: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if self.results['total_tests'] > 0:
            overall_success_rate = (self.results['total_passed'] / self.results['total_tests']) * 100
            print(f"🎯 TASA DE ÉXITO GENERAL: {overall_success_rate:.1f}%")
        else:
            print("🎯 TASA DE ÉXITO GENERAL: N/A (no tests found)")

        # Suite breakdown
        print("\n📋 DESGLOSE POR SUITE:")
        for suite in self.results['suite_results']:
            name = suite['name']
            total = suite.get('total', 0)
            passed = suite.get('passed', 0)
            failed = suite.get('failed', 0)
            status = suite.get('status', 'unknown')

            if total > 0:
                rate = (passed / total) * 100
                print(f"  ↳ {name}: {total} tests | {failed} failed | Status: {status} | Rate: {rate:.1f}%")
            else:

        # Assessment
        assessment = self.get_quality_assessment()
        print(f"\n🏅 VALORACIÓN DE CALIDAD: {assessment['level']}")
        print(f"💡 {assessment['recommendation']}")

    def get_quality_assessment(self) -> dict:
        """Assess overall enterprise quality"""
        if self.results['total_tests'] == 0:
            return {
                'level': 'UNKNOWN',
                'recommendation': 'No tests executed. Critical issue.'
            }

        success_rate = (self.results['total_passed'] / self.results['total_tests']) * 100

        # Failed suites penalty
        failed_suites = sum(1 for s in self.results['suite_results']
                          if s.get('failed', 0) > 0)

        # Enterprise quality thresholds
        if success_rate >= 95 and failed_suites == 0:
            return {
                'level': 'EXCELENTE (ENTERPRISE GOLD)',
                'recommendation': 'Production-ready. All systems operational.'
            }
        elif success_rate >= 85 and failed_suites <= 1:
            return {
                'level': 'MUY BUENA (ENTERPRISE SILVER)',
                'recommendation': 'Production-ready with minor monitoring.'
            }
        elif success_rate >= 70 and failed_suites <= 2:
            return {
                'level': 'ACEPTABLE (ENTERPRISE BRONZE)',
                'recommendation': 'Production with close monitoring required.'
            }
        elif success_rate >= 50:
            return {
                'level': 'PREOCUPANTE',
                'recommendation': 'Critical fixes needed before production.'
            }
        else:
            return {
                'level': 'CRÍTICA',
                'recommendation': 'DO NOT DEPLOY. Immediate fixes required.'
            }

    def assess_enterprise_quality(self) -> bool:
        """Return True if meets minimum enterprise standards"""
        if self.results['total_tests'] == 0:
            return False

        success_rate = (self.results['total_passed'] / self.results['total_tests']) * 100
        failed_suites = sum(1 for s in self.results['suite_results']
                          if s.get('failed', 0) > 0)

        # Minimum standards: 70% success, max 2 failed suites
        return success_rate >= 70 and failed_suites <= 2


def main():
    """Main entry point"""
    runner = EnterpriseTestRunner()
    success = runner.run_all_suites()

    print("\n" + "=" * 70)
    if success:
        print("🎉 RESULTADO: ENTERPRISE QUALITY ACHIEVED")
        return 0
    else:
        print("⚠️  RESULTADO: QUALITY CONCERNS - REVIEW REQUIRED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
