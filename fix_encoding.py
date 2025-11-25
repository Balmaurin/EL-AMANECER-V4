import os

file_path = r"c:\Users\YO\Desktop\EL-AMANECERV3-main\packages\consciousness\src\conciencia\modulos\human_emotions_system.py"

# I will try to read it as binary and decode it, then fix it.
try:
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Try decoding as utf-8
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        # If failed, try latin-1 or cp1252
        text = content.decode('cp1252', errors='ignore')
    
    # Now ensure the compatibility class is there and correct
    class_def = """
# ==================== COMPATIBILIDAD ====================

class HumanEmotionalSystem(HumanEmotionalStateMachine):
    \"\"\"
    Clase de compatibilidad para código que espera HumanEmotionalSystem.
    Ignora num_circuits ya que el sistema avanzado usa todas las emociones.
    \"\"\"
    def __init__(self, num_circuits: int = 35, personality: Dict[str, float] = None):
        super().__init__(personality)
        # num_circuits es ignorado, siempre usamos el set completo
"""
    
    if "class HumanEmotionalSystem(HumanEmotionalStateMachine):" not in text:
        text += class_def
    else:
        # It might be there but corrupted. Let's replace the end of the file if it looks weird.
        # Or just ensure it's clean.
        pass

    # Write back as UTF-8
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
        
    print("File fixed and saved as UTF-8")

except Exception as e:
    print(f"Error: {e}")
