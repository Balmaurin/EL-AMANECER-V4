#!/usr/bin/env python
"""Script para reemplazar caracteres Unicode con ASCII en archivos Python."""

import os
import re
from pathlib import Path

# Mapeo de reemplazos Unicode -> ASCII
REPLACEMENTS = {
    "✓": "[OK]",
    "✅": "[+]",
    "🔄": "[~]",
    "⚠": "[!]",
    "⚠️": "[!]",
    "❌": "[X]",
    "▶": "[>]",
    "★": "[*]",
    "☐": "[ ]",
    "☑": "[x]",
    "🚀": "[>>]",
    "📝": "[doc]",
    "📊": "[data]",
    "🔍": "[?]",
    "💾": "[save]",
}


def fix_unicode_in_file(filepath):
    """Reemplaza Unicode en un archivo."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original = content
        for unicode_char, ascii_replacement in REPLACEMENTS.items():
            content = content.replace(unicode_char, ascii_replacement)

        if content != original:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[+] Fixed: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"[X] Error in {filepath}: {e}")
        return False


def main():
    """Procesa todos los archivos Python en tools/."""
    tools_dir = Path(__file__).parent / "tools"
    fixed_count = 0

    for py_file in tools_dir.rglob("*.py"):
        if fix_unicode_in_file(py_file):
            fixed_count += 1

    print(f"\n[OK] Procesados {fixed_count} archivos con caracteres Unicode")


if __name__ == "__main__":
    main()
