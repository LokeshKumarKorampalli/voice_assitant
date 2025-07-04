import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pathlib import Path
import logging

class AudioProcessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.logger = logging.getLogger(__name__)
    
    def load_audio(self, file_path):
        """Load and preprocess audio file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Try multiple loading methods
        try:
            # Method 1: librosa
            audio, sr = librosa.load(str(file_path), sr=self.target_sr, mono=True)
            return audio, sr
        except:
            try:
                # Method 2: pydub + librosa
                audio_segment = AudioSegment.from_file(str(file_path))
                audio_segment = audio_segment.set_channels(1).set_frame_rate(self.target_sr)
                
                # Convert to numpy array
                audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio = audio / (2**15)  # Normalize to [-1, 1]
                
                return audio, self.target_sr
            except Exception as e:
                raise Exception(f"Failed to load audio: {e}")
    
    def preprocess(self, audio):
        """Preprocess audio for better transcription"""
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Remove silence (optional)
        # audio = self.remove_silence(audio)
        
        return audio
    
    def remove_silence(self, audio, top_db=20):
        """Remove silence from audio"""
        intervals = librosa.effects.split(audio, top_db=top_db)
        if len(intervals) > 0:
            audio_trimmed = np.concatenate([audio[start:end] for start, end in intervals])
            return audio_trimmed
        return audio