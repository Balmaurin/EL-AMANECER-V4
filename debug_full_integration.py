
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
    print("🔄 Importing DigitalHumanConsciousness...")
    from conciencia.modulos.digital_human_consciousness import DigitalHumanConsciousness, ConsciousnessConfig
    
    print("🔄 Initializing full system...")
    config = ConsciousnessConfig(system_name="DebugSystem")
    
    # This triggers initialization of all subsystems including biological
    dhc = DigitalHumanConsciousness(config)
    
    print("✅ Full initialization complete without warnings.")

except Warning as w:
    print(f"\n⚠️ CAUGHT WARNING: {w}")
    traceback.print_exc()
except Exception as e:
    print(f"\n❌ CAUGHT EXCEPTION: {e}")
    traceback.print_exc()
