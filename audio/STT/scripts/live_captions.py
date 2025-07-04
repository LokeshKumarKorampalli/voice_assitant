import whisper
import pyaudio
import numpy as np
import threading
import time
from datetime import datetime
import queue
import signal
import sys

class LiveCaptioning:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.chunk_duration = 5  # seconds
        self.sample_rate = 16000
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        print(f"Loading Whisper {model_size} model...")
        # Fix: Use model name directly, not file path
        self.model = whisper.load_model(model_size)
        print(f"Whisper {model_size} model loaded successfully!")
        
        self.audio_queue = queue.Queue()
        self.is_running = False
        
        # Audio recording parameters
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        self.frames_per_buffer = 1024
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def process_audio_queue(self):
        """Process audio chunks from the queue"""
        audio_buffer = np.array([], dtype=np.float32)
        
        while self.is_running:
            try:
                # Get audio data from queue (with timeout to check is_running)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                
                # If we have enough audio data, transcribe it
                if len(audio_buffer) >= self.chunk_size:
                    # Take the first chunk_size samples
                    chunk_to_process = audio_buffer[:self.chunk_size]
                    # Keep the rest for next iteration
                    audio_buffer = audio_buffer[self.chunk_size:]
                    
                    # Transcribe in a separate thread to avoid blocking
                    threading.Thread(target=self.transcribe_chunk, args=(chunk_to_process,), daemon=True).start()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def transcribe_chunk(self, audio_chunk):
        """Transcribe a chunk of audio"""
        try:
            # Whisper expects audio to be between -1 and 1
            # Ensure the audio is in the right range
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            
            # Transcribe the audio
            result = self.model.transcribe(audio_chunk, language="en", fp16=False)
            text = result["text"].strip()
            
            if text:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {text}")
                
        except Exception as e:
            print(f"Transcription error: {e}")
    
    def start_live_captioning(self):
        """Start the live captioning system"""
        self.is_running = True
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        try:
            # Start audio stream
            stream = audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self.audio_callback
            )
            
            print("Audio stream started successfully!")
            
            # Start audio processing thread
            processing_thread = threading.Thread(target=self.process_audio_queue, daemon=True)
            processing_thread.start()
            
            # Start the stream
            stream.start_stream()
            
            print("==================================================")
            print("LIVE CAPTIONING STARTED")
            print("==================================================")
            print("Speak into your microphone...")
            print("Press Ctrl+C to stop")
            print("==================================================")
            print()
            
            # Keep the main thread alive
            while self.is_running and stream.is_active():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error starting audio stream: {e}")
        finally:
            # Cleanup
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            audio.terminate()
    
    def stop(self):
        """Stop the live captioning system"""
        print("\nStopping live captioning...")
        self.is_running = False
        print("Live captioning stopped.")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global captioner
    if 'captioner' in globals():
        captioner.stop()
    sys.exit(0)

def main():
    global captioner
    
    print("Live Captioning System")
    print("======================")
    print()
    
    # Model selection
    models = {
        "1": ("tiny", "Tiny (fastest)"),
        "2": ("base", "Base (recommended for CPU)"), 
        "3": ("small", "Small (better accuracy)"),
        "4": ("medium", "Medium (high accuracy, slower)"),
        "5": ("large", "Large (best accuracy, very slow)")
    }
    
    print("Choose Whisper model size:")
    for key, (model_name, description) in models.items():
        print(f"{key}. {description}")
    print()
    
    choice = input("Enter choice (1-5) [default: 2]: ").strip()
    if not choice:
        choice = "2"
    
    if choice not in models:
        print("Invalid choice. Using base model.")
        choice = "2"
    
    model_size, model_desc = models[choice]
    print(f"\nUsing {model_size} model...")
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create and start the live captioning system
        captioner = LiveCaptioning(model_size=model_size)
        captioner.start_live_captioning()
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
        if 'captioner' in locals():
            captioner.stop()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()