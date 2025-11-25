import os

file_path = r"c:\Users\YO\Desktop\EL-AMANECERV3-main\packages\consciousness\src\conciencia\modulos\human_emotions_system.py"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add coding declaration if missing
    if "# -*- coding: utf-8 -*-" not in content:
        content = "# -*- coding: utf-8 -*-\n" + content
        
    # Replace non-ascii identifiers
    content = content.replace("ÉXTASIS =", "EXTASIS =")
    content = content.replace("PÁNICO =", "PANICO =")
    content = content.replace("VERGÜENZA =", "VERGUENZA =")
    content = content.replace("COMPASION =", "COMPASION =") # Already ascii? No, check others
    
    # Check for other potential issues
    # MELANCOLIA is ascii.
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("File sanitized and saved as UTF-8")

except Exception as e:
    print(f"Error: {e}")
