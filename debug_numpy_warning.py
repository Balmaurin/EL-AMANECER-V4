
import sys
import os
import numpy as np
import warnings
import traceback

# Treat warnings as errors to catch the traceback
warnings.simplefilter('error')

# Add package path
sys.path.append(r"C:\Users\YO\Desktop\EL-AMANECERV3-main\packages\consciousness\src")

try:
    print("🔄 Importing BiologicalConsciousnessSystem...")
    from conciencia.modulos.biological_consciousness import BiologicalConsciousnessSystem
    
    print("🔄 Initializing system...")
    # This is where the warning appeared in the logs
    bio_system = BiologicalConsciousnessSystem("debug_system")
    
    print("✅ Initialization complete without warnings.")

except Warning as w:
    print(f"\n⚠️ CAUGHT WARNING: {w}")
    traceback.print_exc()
except Exception as e:
    print(f"\n❌ CAUGHT EXCEPTION: {e}")
    traceback.print_exc()
