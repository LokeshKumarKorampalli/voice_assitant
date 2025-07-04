import whisper
import os
from pathlib import Path

# Add FFmpeg to PATH
ffmpeg_path = r"D:\whisper_stt_project\ffmpeg-7.1.1-essentials_build\bin"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# Load model once
model = whisper.load_model("../models/base.pt")

# Get all wav files
audio_dir = Path("../test_audio")
wav_files = list(audio_dir.glob("*.wav"))

print(f"Found {len(wav_files)} WAV files")
print("="*50)

# Process each file
for i, audio_file in enumerate(wav_files):
    print(f"[{i+1}/{len(wav_files)}] Processing: {audio_file.name}")
    
    try:
        result = model.transcribe(str(audio_file))
        print(f"Text: {result['text']}")
        print(f"Language: {result['language']}")
        print("-" * 30)
    except Exception as e:
        print(f"Error processing {audio_file.name}: {e}")
        print("-" * 30)