import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
AUDIO_DIR = PROJECT_ROOT / "test_audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Model settings
WHISPER_MODELS = {
    "tiny": {"size": "39M", "vram": "~1GB", "langs": "99"},
    "base": {"size": "74M", "vram": "~1GB", "langs": "99"},
    "small": {"size": "244M", "vram": "~2GB", "langs": "99"},
    "medium": {"size": "769M", "vram": "~5GB", "langs": "99"},
    "large": {"size": "1550M", "vram": "~10GB", "langs": "99"},
    "large-v2": {"size": "1550M", "vram": "~10GB", "langs": "99"},
    "large-v3": {"size": "1550M", "vram": "~10GB", "langs": "99"}
}

DEFAULT_MODEL = "base"  # Change to "large-v3" for best accuracy

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']

# Transcription settings
DEFAULT_LANGUAGE = "en"  # None for auto-detection
TEMPERATURE = 0.0  # 0.0 for most deterministic
BEAM_SIZE = 5