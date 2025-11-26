#!/usr/bin/env python3
"""
LOCAL LLM LOADER
===============

Carga y gestiona modelos LLM locales sin subirlos a GitHub.
Manejo de modelos GGUF para inferencia local enterprise.

CRÍTICO: Local model management, enterprise LLM loading.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any


class LocalLLMManager:
    """Gestor de modelos LLM locales para enterprise"""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize LLM manager with models directory
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.config_file = self.models_dir / "model_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "models": {},
                "default_model": None,
                "model_path": str(self.models_dir)
            }
    
    def check_model_exists(self, model_name: Optional[str] = None) -> bool:
        """Check if model exists locally
        
        Args:
            model_name: Name of model to check, uses default if None
            
        Returns:
            True if model file exists
        """
        if model_name is None:
            model_name = self.config.get('default_model')
        
        if not model_name or model_name not in self.config['models']:
            return False
        
        model_info = self.config['models'][model_name]
        model_path = self.models_dir / model_info['filename']
        
        return model_path.exists()
    
    def get_model_path(self, model_name: Optional[str] = None) -> Optional[Path]:
        """Get path to model file
        
        Args:
            model_name: Name of model, uses default if None
            
        Returns:
            Path to model file if exists, None otherwise
        """
        if model_name is None:
            model_name = self.config.get('default_model')
        
        if not model_name or model_name not in self.config['models']:
            return None
        
        model_info = self.config['models'][model_name]
        model_path = self.models_dir / model_info['filename']
        
        return model_path if model_path.exists() else None
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all available models and their status"""
        available_models = {}
        
        for model_name, model_info in self.config['models'].items():
            model_path = self.models_dir / model_info['filename']
            available_models[model_name] = {
                'info': model_info,
                'exists': model_path.exists(),
                'path': str(model_path),
                'size_mb': model_path.stat().st_size / (1024*1024) if model_path.exists() else 0
            }
        
        return available_models
    
    def download_instructions(self, model_name: Optional[str] = None) -> str:
        """Get download instructions for missing model
        
        Args:
            model_name: Name of model, uses default if None
            
        Returns:
            Download instructions string
        """
        if model_name is None:
            model_name = self.config.get('default_model', 'gemma-2-2b-it-q4_k_m')
        
        instructions = f"""
📥 DOWNLOAD INSTRUCTIONS FOR {model_name.upper()}

1. Manual Download:
   Visit: https://huggingface.co/google/gemma-2-2b-it-gguf
   Download: gemma-2-2b-it-q4_k_m.gguf
   Place in: {self.models_dir}/

2. Using HuggingFace CLI:
   pip install huggingface_hub
   huggingface-cli download google/gemma-2-2b-it-gguf gemma-2-2b-it-q4_k_m.gguf --local-dir {self.models_dir}/

3. Using wget (Linux/Mac):
   wget https://huggingface.co/google/gemma-2-2b-it-gguf/resolve/main/gemma-2-2b-it-q4_k_m.gguf -P {self.models_dir}/

⚠️ Note: Model file is ~1.6GB
"""
        return instructions


def main():
    """Check local LLM status and provide instructions"""
    print("🤖 LOCAL LLM MANAGER")
    print("=" * 25)
    
    llm_manager = LocalLLMManager()
    
    # Check if default model exists
    default_model = llm_manager.config.get('default_model', 'gemma-2-2b-it-q4_k_m')
    model_exists = llm_manager.check_model_exists(default_model)
    
    print(f"📂 Models directory: {llm_manager.models_dir}")
    print(f"🎯 Default model: {default_model}")
    
    if model_exists:
        model_path = llm_manager.get_model_path(default_model)
        model_size = model_path.stat().st_size / (1024*1024*1024)  # GB
        print(f"✅ Model found: {model_path}")
        print(f"📊 Model size: {model_size:.2f} GB")
        
        print(f"\n🚀 Model ready for use!")
        print(f"   Path: {model_path}")
        
    else:
        print(f"❌ Model not found: {default_model}")
        print(f"\n{llm_manager.download_instructions()}")
    
    # List all configured models
    available_models = llm_manager.list_available_models()
    if available_models:
        print(f"\n📋 ALL CONFIGURED MODELS:")
        for name, info in available_models.items():
            status = "✅ Available" if info['exists'] else "❌ Missing"
            size = f"{info['size_mb']:.1f}MB" if info['exists'] else "N/A"
            print(f"   {name}: {status} ({size})")
    
    print(f"\n💡 Remember: Models are excluded from Git due to size")
    print(f"   Check .gitignore for: *.gguf, models/, *.bin")


if __name__ == "__main__":
    main()
