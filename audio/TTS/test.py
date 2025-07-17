import os
import sounddevice as sd
from TTS.api import TTS

# Local directory to cache model
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "local_tts_model")
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["TORCH_HOME"] = MODEL_DIR

# Load TTS model once
_tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

def speak(text: str, speed: float = 1.3):
    print(f"[🔊] Speaking: {text}")
    wav = _tts.tts(text=text, speed=speed)
    sd.play(wav, samplerate=22050)
    sd.wait()
