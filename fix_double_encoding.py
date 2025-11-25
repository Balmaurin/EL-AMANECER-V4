import os

file_path = r"c:\Users\YO\Desktop\EL-AMANECERV3-main\packages\consciousness\src\conciencia\modulos\human_emotions_system.py"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to fix double encoding
    # Common pattern: UTF-8 bytes interpreted as Latin-1, then saved as UTF-8
    try:
        fixed_content = content.encode('latin1').decode('utf-8')
        print("Successfully reversed double encoding.")
        content = fixed_content
    except Exception as e:
        print(f"Could not reverse double encoding: {e}")
        # Fallback: manual replacements for common mojibake
        replacements = {
            "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
            "Ã±": "ñ", "Ã": "Á", "Ã‰": "É", "Ã": "Í", "Ã“": "Ó", "Ãš": "Ú",
            "Ã‘": "Ñ", "Â": "", "Ã¼": "ü"
        }
        for bad, good in replacements.items():
            content = content.replace(bad, good)

    # Also ensure the identifiers are ASCII
    content = content.replace("ÉXTASIS =", "EXTASIS =")
    content = content.replace("PÁNICO =", "PANICO =")
    content = content.replace("VERGÜENZA =", "VERGUENZA =")
    content = content.replace("Ã‰XTASIS =", "EXTASIS =") # Handle mojibake version if exists
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("File fixed and saved as UTF-8")

except Exception as e:
    print(f"Error: {e}")
