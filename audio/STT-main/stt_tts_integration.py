import whisper
import pyaudio
import numpy as np
import time
import os
import subprocess
import json
from collections import deque
from pathlib import Path

class STTTTSSystem:
    def __init__(self):
        # Load configuration
        self.load_config()
        
        # Initialize Whisper
        print(f"Loading Whisper {self.config['whisper_model']} model...")
        self.model = whisper.load_model(self.config['whisper_model'])
        print("Whisper model loaded successfully!")
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 3
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(self.RATE * 10))
        self.is_recording = False
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        
        # TTS setup
        self.piper_path = Path(self.config['piper_tts_path'])
        self.output_dir = self.piper_path / "output"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open("stt_tts_config.json", "r") as f:
                self.config = json.load(f)
            print("Configuration loaded successfully")
        except FileNotFoundError:
            print("Configuration file not found. Please run stt_tts_config.py first.")
            exit(1)
    
    def start_audio_stream(self):
        """Start audio input stream"""
        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )
            print("Audio stream started successfully!")
            return True
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            return False
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.audio_buffer.extend(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def get_audio_chunk(self):
        """Get audio chunk for processing"""
        if len(self.audio_buffer) < self.RATE * self.RECORD_SECONDS:
            return None
            
        chunk_size = int(self.RATE * self.RECORD_SECONDS)
        audio_chunk = list(self.audio_buffer)[-chunk_size:]
        
        audio_array = np.array(audio_chunk, dtype=np.float32)
        audio_array = audio_array / 32768.0
        
        return audio_array
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
        try:
            result = self.model.transcribe(
                audio_data,
                language="en",
                task="transcribe",
                fp16=False,
                verbose=False
            )
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def generate_response(self, transcribed_text):
        """Generate response text"""
        if not transcribed_text:
            return None
        
        text_lower = transcribed_text.lower()
        return text_lower
        
    

    def text_to_speech(self, text):
        """Convert text to speech using Piper TTS"""
        if not text:
            return False
        
        try:
            # Create output filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"response_{timestamp}.wav"
            
            # Use the PowerShell wrapper script
            wrapper_script = self.piper_path / "stt_tts_wrapper.ps1"
            
            if wrapper_script.exists():
                # Use the wrapper script - FIXED PARAMETER NAME
                cmd = [
                    "powershell", "-File", str(wrapper_script),
                    "-Text", text,  # Changed from "-InputText" to "-Text"
                    "-OutputFile", str(output_file)
                ]
            else:
                # Fallback: direct piper call (adjust as needed)
                cmd = [
                    "echo", text, "|", "piper.exe", 
                    "--model", "piper_models/en_US-lessac-medium.onnx",
                    "--output_file", str(output_file)
                ]
            
            # Execute TTS command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.piper_path),
                timeout=30
            )
            
            if result.returncode == 0 and output_file.exists():
                print(f"TTS audio generated: {output_file.name}")
                # Play the audio
                self.play_audio(output_file)
                return True
            else:
                print(f"TTS error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"TTS subprocess error: {e}")
            return False
    
    def play_audio(self, audio_file):
        """Play audio file"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(str(audio_file))
            else:  # Linux/Mac
                subprocess.run(['xdg-open', str(audio_file)])
        except Exception as e:
            print(f"Could not play audio: {e}")
    
    def interactive_mode(self):
        """Run interactive STT-TTS mode"""
        print("\n" + "="*50)
        print("STT-TTS INTERACTIVE MODE")
        print("="*50)
        print("Speak into your microphone...")
        print("Say 'stop', 'exit', or 'quit' to end")
        print("Press Ctrl+C to force stop")
        print("="*50 + "\n")
        
        if not self.start_audio_stream():
            return
        
        self.is_recording = True
        self.stream.start_stream()
        
        try:
            while self.is_recording:
                # Get audio chunk
                audio_chunk = self.get_audio_chunk()
                
                if audio_chunk is not None:
                    # Check for actual audio (not silence)
                    if np.max(np.abs(audio_chunk)) > 0.01:
                        # Transcribe
                        transcribed_text = self.transcribe_audio(audio_chunk)
                        
                        if transcribed_text and len(transcribed_text.strip()) > 0:
                            timestamp = time.strftime("%H:%M:%S")
                            print(f"[{timestamp}] You said: {transcribed_text}")
                            
                            # Generate response
                            response_text = self.generate_response(transcribed_text)
                            
                            if response_text:
                                print(f"[{timestamp}] Response: {response_text}")
                                
                                # Convert to speech
                                success = self.text_to_speech(response_text)
                                if not success:
                                    print("TTS failed")
                                
                                # Check for exit commands
                                if any(word in transcribed_text.lower() for word in ["stop", "exit", "quit"]):
                                    print("Exit command detected. Stopping...")
                                    break
                
                # Small delay
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop_system()
    
    def batch_mode(self):
        """Process audio files in batch"""
        audio_dir = input("Enter directory path containing audio files: ").strip()
        if not audio_dir or not os.path.exists(audio_dir):
            print("Invalid directory path")
            return
        
        audio_dir = Path(audio_dir)
        audio_files = []
        
        # Find audio files
        for ext in ['.wav', '.mp3', '.m4a', '.flac']:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
        
        if not audio_files:
            print("No audio files found")
            return
        
        print(f"Found {len(audio_files)} audio files")
        
        for audio_file in audio_files:
            print(f"\nProcessing: {audio_file.name}")
            
            try:
                # Transcribe
                result = self.model.transcribe(str(audio_file))
                transcribed_text = result["text"].strip()
                
                if transcribed_text:
                    print(f"Transcription: {transcribed_text}")
                    
                    # Generate response
                    response_text = self.generate_response(transcribed_text)
                    
                    if response_text:
                        print(f"Response: {response_text}")
                        
                        # Generate TTS
                        success = self.text_to_speech(response_text)
                        if success:
                            print("TTS audio generated")
                        else:
                            print("TTS failed")
                else:
                    print("No speech detected")
                    
            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")
    
    def stop_system(self):
        """Stop the system"""
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        print("System stopped.")

def main():
    print("STT-TTS Integration System")
    print("==========================")
    
    # Initialize system
    try:
        system = STTTTSSystem()
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        return
    
    # Choose mode
    print("\nChoose mode:")
    print("1. Interactive mode (live STT-TTS)")
    print("2. Batch process audio files")
    
    choice = input("\nEnter choice (1-2) [default: 1]: ").strip()
    
    if choice == '2':
        system.batch_mode()
    else:
        system.interactive_mode()

if __name__ == "__main__":
    main()