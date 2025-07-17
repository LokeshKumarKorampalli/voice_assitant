import requests
import json

def stream_ollama(prompt: str, model: str = "mistral"):
    url = "http://localhost:11434/api/generate"
    print(f"[🤖] Sending to Ollama (streaming): {prompt}")

    try:
        response = requests.post(url, json={
            "model": model,
            "prompt": prompt,
            "stream": True
        }, stream=True)

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        token = data.get("response", "")
                        if token:
                            print(token, end="", flush=True)  # For debug
                            yield token
                    except Exception as e:
                        print(f"[⚠️ Parse error]: {e}")
    except Exception as e:
        print(f"[❌ Ollama stream error]: {e}")


stream_ollama("Hello, how are you?")  # Example usage