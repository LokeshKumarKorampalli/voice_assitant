import os
from pathlib import Path
import json
import logging
from datetime import datetime
from ..models.whisper_model import WhisperModel
from ..audio.processor import AudioProcessor
from ..config.settings import SUPPORTED_FORMATS, OUTPUT_DIR

class BatchTranscriber:
    def __init__(self, model_size="base"):
        self.model = WhisperModel(model_size)
        self.audio_processor = AudioProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(exist_ok=True)
    
    def transcribe_file(self, audio_path, output_format="txt", **options):
        """Transcribe a single audio file"""
        audio_path = Path(audio_path)
        
        if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {audio_path.suffix}")
        
        self.logger.info(f"Transcribing: {audio_path}")
        
        # Transcribe
        result = self.model.transcribe(audio_path, **options)
        
        if result:
            # Save result
            output_file = self._save_result(audio_path, result, output_format)
            self.logger.info(f"Transcription saved: {output_file}")
            return result, output_file
        
        return None, None
    
    def transcribe_directory(self, directory_path, **options):
        """Transcribe all audio files in a directory"""
        directory_path = Path(directory_path)
        audio_files = []
        
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(directory_path.glob(f"*{ext}"))
        
        results = []
        for audio_file in audio_files:
            try:
                result, output_file = self.transcribe_file(audio_file, **options)
                if result:
                    results.append({
                        'input_file': str(audio_file),
                        'output_file': str(output_file),
                        'text': result['text'],
                        'language': result['language'],
                        'processing_time': result['processing_time']
                    })
            except Exception as e:
                self.logger.error(f"Failed to transcribe {audio_file}: {e}")
        
        return results
    
    def _save_result(self, audio_path, result, output_format):
        """Save transcription result"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{audio_path.stem}_{timestamp}"
        
        if output_format == "txt":
            output_file = OUTPUT_DIR / f"{base_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"File: {audio_path}\n")
                f.write(f"Language: {result['language']}\n")
                f.write(f"Processing time: {result['processing_time']:.2f}s\n")
                f.write(f"Model: {result['model_info']['model_size']}\n")
                f.write("=" * 50 + "\n")
                f.write(result['text'])
        
        elif output_format == "json":
            output_file = OUTPUT_DIR / f"{base_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        elif output_format == "srt":
            output_file = OUTPUT_DIR / f"{base_name}.srt"
            self._save_as_srt(output_file, result)
        
        return output_file
    
    def _save_as_srt(self, output_file, result):
        """Save as SRT subtitle format"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                start_time = self._format_timestamp(segment['start'])
                end_time = self._format_timestamp(segment['end'])
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text'].strip()}\n\n")
    
    def _format_timestamp(self, seconds):
        """Format timestamp for SRT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')