#!/usr/bin/env python3
"""Check security status"""
import sys

sys.path.insert(0, "tools")
from simple_security import get_security_manager

security = get_security_manager()
status = security.check_security_status()

print(f'Checks totales: {len(status["checks"])}')
passed = sum(1 for check in status["checks"].values() if check)
print(f"Checks pasados: {passed}")
print(f'Status: {status["overall_status"]}')
print(f'Security Score: {status["security_score"]}%')

if passed == len(status["checks"]):
    print("🎉 ¡SEGURIDAD 20/20 ALCANZADA!")
else:
    print(f'Faltan {len(status["checks"]) - passed} checks')
