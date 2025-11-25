import requests
import json
import time

def test_chat(message):
    url = "http://localhost:8000/api/chat/send"
    payload = {
        "message": message,
        "conversation_id": "test-verification-1"
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Sending message: '{message}'")
    try:
        start = time.time()
        response = requests.post(url, json=payload, headers=headers)
        duration = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response received in {duration:.2f}s")
            print(f"Content: {data['message']['content']}")
            
            if 'metadata' in data['message'] and data['message']['metadata']:
                print("\nMetadata found:")
                print(json.dumps(data['message']['metadata'], indent=2, ensure_ascii=False))
            else:
                print("\nNo metadata found in response.")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    print("--- Test 1: Neutral/Technical ---")
    test_chat("Hola, ¿cómo funciona tu sistema de emociones?")
    
    print("\n--- Test 2: Emotional (Sad) ---")
    test_chat("Me siento muy triste hoy porque perdí mi trabajo.")
    
    print("\n--- Test 3: Emotional (Happy) ---")
    test_chat("¡Estoy muy feliz! ¡Conseguí el trabajo de mis sueños!")
