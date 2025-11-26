#!/usr/bin/env python3
"""
MODEL DOWNLOADER
===============

Script para descargar el modelo LLM automáticamente.
Gestiona la descarga segura sin afectar el repositorio Git.

CRÍTICO: Model acquisition, automated download.
"""

import os
import sys
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm


def download_with_progress(url: str, filepath: Path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))


def download_gemma_model():
    """Download Gemma 2 2B model"""
    print("📥 DOWNLOADING GEMMA 2 MODEL")
    print("=" * 35)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_filename = "gemma-2-2b-it-q4_k_m.gguf"
    model_path = models_dir / model_filename
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return True
    
    # Try HuggingFace CLI first
    try:
        print("🔧 Attempting download via HuggingFace CLI...")
        result = subprocess.run([
            'huggingface-cli', 'download',
            'google/gemma-2-2b-it-gguf',
            model_filename,
            '--local-dir', str(models_dir)
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and model_path.exists():
            print(f"✅ Download successful via HuggingFace CLI")
            return True
        
    except FileNotFoundError:
        print("⚠️ HuggingFace CLI not found")
    
    # Manual download instructions
    print(f"\n📋 MANUAL DOWNLOAD REQUIRED")
    print(f"=" * 30)
    print(f"1. Visit: https://huggingface.co/google/gemma-2-2b-it-gguf")
    print(f"2. Download: {model_filename}")
    print(f"3. Place in: {models_dir.absolute()}")
    print(f"4. File size: ~1.6GB")
    
    print(f"\n💡 Alternative methods:")
    print(f"   pip install huggingface_hub")
    print(f"   huggingface-cli download google/gemma-2-2b-it-gguf {model_filename} --local-dir {models_dir}")
    
    return False


def main():
    """Execute model download"""
    download_gemma_model()
    
    # Verify download
    from load_local_llm import LocalLLMManager
    llm_manager = LocalLLMManager()
    
    if llm_manager.check_model_exists():
        print(f"\n🎯 MODEL READY!")
        print(f"✅ Gemma 2 model available for local inference")
    else:
        print(f"\n⚠️ Model download incomplete")
        print(f"Please follow manual instructions above")


if __name__ == "__main__":
    main()
