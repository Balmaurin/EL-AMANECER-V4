"""
Config bridge for sheily_core
Connects to the unified settings from apps/backend
"""
import os
import sys
from pathlib import Path# Añadir ruta al backend si no está
_backend_config_path = Path(__file__).resolve().parents[4] / "apps" / "backend" / "src"
if str(_backend_config_path) not in sys.path:
    sys.path.insert(0, str(_backend_config_path))

try:
    import importlib.util
    
    # Import settings usando ruta absoluta para evitar conflictos
    _settings_file = _backend_config_path / "config" / "settings.py"
    
    spec = importlib.util.spec_from_file_location("backend_settings", _settings_file)
    backend_settings_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend_settings_module)
    
    unified_settings = backend_settings_module.settings
    
    # Usar el settings unificado como Config
    Config = type(unified_settings)
    
    def get_config():
        """Get global unified config instance"""
        return unified_settings
        
except Exception as e:
    # Fallback si no se puede importar
    print(f"⚠️ Warning: Could not import unified settings: {e}")
    print(f"   Using fallback Config. Backend features may not work.")
    
    class Config:
        """Fallback configuration class"""
        def __init__(self):
            self.log_level = "INFO"
            base_dir = Path(r"c:\Users\YO\Desktop\EL-AMANECERV3-main")
            self.model_path = str(base_dir / "modelsLLM" / "llama-3.2-3b" / "Llama-3.2-3B-Instruct-f16.gguf")
            self.llama_binary_path = str(base_dir / "llama_cpp_install" / "bin" / "llama-cli.exe")
            self.corpus_root = "data"
            self.branches_config_path = "config/branches.json"
            self.model_max_tokens = 512
            self.model_temperature = 0.7
            self.model_threads = 4
            self.model_timeout = 30
    
    def get_config():
        """Get fallback config instance"""
        return Config()
