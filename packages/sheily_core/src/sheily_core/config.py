"""
Simple config module for sheily_core
"""
import os
from pathlib import Path

class Config:
    """Configuration class"""
    def __init__(self):
        self.log_level = "INFO"
        
        # Rutas absolutas detectadas
        base_dir = Path(r"c:\Users\YO\Desktop\EL-AMANECERV3-main")
        
def get_config():
    """Get global config instance"""
    return Config()
