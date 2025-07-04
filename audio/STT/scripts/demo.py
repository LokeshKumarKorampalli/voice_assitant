import whisper
import os

# Add FFmpeg to PATH for this session
ffmpeg_path = r"D:\whisper_stt_project\ffmpeg-7.1.1-essentials_build\bin"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# Load model from local file
model_path = "../models/base.pt"  # Path to your local model
model = whisper.load_model(model_path)
result = model.transcribe("../test_audio/000000.wav")
print("Transcription:")
print(result["text"])