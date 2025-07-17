import re
from llm.test import stream_ollama
from audio.TTS.test import speak

def speak_streamed_tokens(prompt: str, max_buffer_len: int = 100):
    buffer = ""
    print("[📣] Generating and speaking...")

    for token in stream_ollama(prompt):
        print(token, end="", flush=True)
        buffer += token

        # Speak if we detect a sentence end
        if re.search(r'[.!?]["\')\]]?\s$', buffer) or len(buffer) > max_buffer_len:
            clean = buffer.strip()
            if clean:
                speak(clean, speed=1.3)
            buffer = ""

    # Speak any remaining tail
    if buffer.strip():
        speak(buffer.strip(), speed=1.3)

if __name__ == "__main__":
    prompt = "Tell me why artificial intelligence is useful for students."
    speak_streamed_tokens(prompt)
