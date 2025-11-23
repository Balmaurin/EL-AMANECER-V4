#!/usr/bin/env python3
"""Script para diagnosticar y arreglar problemas de sintaxis"""

with open("dashboard_backend.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Buscando líneas con comillas triples:")
for i, line in enumerate(lines):
    if '"""' in line:
        print(f"Línea {i+1}: {repr(line.strip())}")

# Contar comillas triples
total_quotes = sum(line.count('"""') for line in lines)
print(f"\nTotal de comillas triples: {total_quotes}")

if total_quotes % 2 != 0:
    print("ERROR: Número impar de comillas triples")
else:
    print("Comillas triples balanceadas")
