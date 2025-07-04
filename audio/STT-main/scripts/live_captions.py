import whisper
import pyaudio
import numpy as np
import time
from collections import deque

# === Model paths ===
model_paths = {
    'tiny': '../models/tiny.pt',
    'base': '../models/base.pt',
    'small': '../models/small.pt',
    'medium': '../models/medium.pt',
    'large': '../models/large-v3.pt'
}

# === Set model size here ===
model_size = 'base'  # Change to 'tiny', 'small', etc., as needed
model_path = model_paths[model_size]

# === Load model ===
print(f"🔍 Loading Whisper model from {model_path}...")
model = whisper.load_model(model_path)
print(f"✅ Whisper '{model_size}' model loaded.")

# === Audio config ===
RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 3

# === PyAudio setup ===
p = pyaudio.PyAudio()
buffer = deque(maxlen=RATE * 10)  # 10-second ring buffer

def audio_callback(in_data, frame_count, time_info, status):
    data_np = np.frombuffer(in_data, dtype=np.int16)
    buffer.extend(data_np)
    return (in_data, pyaudio.paContinue)

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    stream_callback=audio_callback
)
stream.start_stream()

print("\n🎙️ Live Captioning Started")
print("Press Ctrl+C to stop\n")

try:
    while True:
        if len(buffer) < RATE * RECORD_SECONDS:
            time.sleep(0.1)
            continue

        # Get last N seconds of audio
        chunk_size = RATE * RECORD_SECONDS
        audio_chunk = np.array(list(buffer)[-chunk_size:], dtype=np.float32) / 32768.0

        # Skip silence
        if np.max(np.abs(audio_chunk)) > 0.01:
            result = model.transcribe(audio_chunk, language="en", task="transcribe", fp16=False, verbose=False)
            text = result.get("text", "").strip()
            if text:
                print(f"[{time.strftime('%H:%M:%S')}] {text}")

        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Stopping transcription...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("✅ Audio stream closed.")
