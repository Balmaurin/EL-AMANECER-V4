from llama_cpp import Llama
import os

model_path = "models/gemma-2-2b-it-q4_k_m.gguf"
abs_path = os.path.abspath(model_path)
print(f"Testing model load from: {abs_path}")

if not os.path.exists(model_path):
    print("❌ Model file not found!")
    exit(1)

try:
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        verbose=True
    )
    print("✅ Model loaded successfully!")
    
    output = llm(
        "Hello, how are you?",
        max_tokens=32,
        stop=["<end_of_turn>"],
        echo=False
    )
    print(f"✅ Inference successful: {output['choices'][0]['text']}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
