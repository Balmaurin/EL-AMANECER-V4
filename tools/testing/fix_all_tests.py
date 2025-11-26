"""
Script para corregir todos los tests del proyecto
Convierte returns en asserts y elimina setup_module problemáticos
"""
import os
import re
from pathlib import Path

def fix_test_file(filepath: Path):
    """Corrige un archivo de test"""
    try:
        content = filepath.read_text(encoding='utf-8')
        original = content
        
        # Eliminar setup_module si existe
        content = re.sub(r'def setup_module\([^)]*\):.*?(?=\ndef|\Z)', '', content, flags=re.DOTALL)
        
        # Convertir returns booleanos en asserts dentro de funciones test_
        def replace_test_returns(match):
            func_content = match.group(0)
            # Reemplazar return True/False con asserts
            func_content = re.sub(r'return True\s*$', 'assert True', func_content, flags=re.MULTILINE)
            func_content = re.sub(r'return False\s*$', 'assert False, "Test failed"', func_content, flags=re.MULTILINE)
            # Eliminar try-except que solo retornan
            func_content = re.sub(
                r'try:\s*\n(.*?)except.*?:\s*\n.*?return False',
                r'\1',
                func_content,
                flags=re.DOTALL
            )
            return func_content
        
        content = re.sub(
            r'def test_[^(]+\([^)]*\):.*?(?=\ndef|\Z)',
            replace_test_returns,
            content,
            flags=re.DOTALL
        )
        
        if content != original:
            filepath.write_text(content, encoding='utf-8')
            print(f"✅ Fixed: {filepath.name}")
            return True
        else:
            print(f"ℹ️  No changes: {filepath.name}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def main():
    tests_dir = Path("tests")
    fixed_count = 0
    
    for test_file in tests_dir.glob("test_*.py"):
        if fix_test_file(test_file):
            fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} test files")

if __name__ == "__main__":
    main()
