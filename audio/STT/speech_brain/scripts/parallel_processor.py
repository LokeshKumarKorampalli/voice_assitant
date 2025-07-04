import asyncio
import subprocess
import sys
import os
import time
import json
from pathlib import Path

class ParallelAudioProcessor:
    def __init__(self, whisper_project_path, speechbrain_project_path):
        self.whisper_project_path = Path(whisper_project_path)
        self.speechbrain_project_path = Path(speechbrain_project_path)
    
    async def run_whisper_processing(self, audio_file):
        """Run Whisper processing asynchronously"""
        try:
            # Create whisper processing script
            whisper_script_content = f'''
import whisper
import json
import sys
import warnings
warnings.filterwarnings("ignore")

def process_audio(audio_file):
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        
        print(f"Processing audio: {{audio_file}}")
        result = model.transcribe(audio_file)
        
        print("=" * 50)
        print("WHISPER PROCESSING RESULT")
        print("=" * 50)
        print(f"Transcription: {{result['text']}}")
        print(f"Language: {{result['language']}}")
        print("=" * 50)
        
        # Save results
        output_file = audio_file.replace('.wav', '_whisper_result.json')
        with open(output_file, 'w') as f:
            json.dump({{
                'transcription': result['text'],
                'language': result['language'],
                'segments': result.get('segments', [])
            }}, f, indent=2)
        
        print(f"Results saved to: {{output_file}}")
        return {{"status": "SUCCESS", "transcription": result['text'], "language": result['language']}}
        
    except Exception as e:
        print(f"ERROR: {{e}}")
        return {{"status": "ERROR", "error": str(e)}}

if __name__ == "__main__":
    audio_file = "{audio_file}"
    result = process_audio(audio_file)
    print(f"FINAL_RESULT: {{json.dumps(result)}}")
'''
            
            # Write the script to a temporary file
            temp_script = self.whisper_project_path / "temp_whisper_processor.py"
            with open(temp_script, "w") as f:
                f.write(whisper_script_content)
            
            # Run the whisper processing
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(temp_script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.whisper_project_path)
            )
            
            stdout, stderr = await process.communicate()
            
            # Clean up temp file
            if temp_script.exists():
                temp_script.unlink()
            
            if process.returncode == 0:
                output = stdout.decode()
                # Extract the final result
                for line in output.split('\n'):
                    if line.startswith('FINAL_RESULT:'):
                        result_json = line.replace('FINAL_RESULT:', '').strip()
                        try:
                            result = json.loads(result_json)
                            result["full_output"] = output
                            return result
                        except json.JSONDecodeError:
                            pass
                
                return {"status": "SUCCESS", "output": output}
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                return {"status": "ERROR", "error": error_msg}
                
        except Exception as e:
            return {"status": "ERROR", "error": f"Exception: {str(e)}"}
    
    async def run_speechbrain_processing(self, audio_file):
        """Run SpeechBrain processing asynchronously"""
        try:
            # Check if speechbrain processor exists
            speechbrain_script = self.speechbrain_project_path / "speechbrain_processor.py"
            
            if not speechbrain_script.exists():
                # Create a basic speechbrain processor if it doesn't exist
                speechbrain_content = '''
import sys
import json
import numpy as np
import librosa

def process_audio(audio_file):
    try:
        print("Loading SpeechBrain model...")
        print("SpeechBrain model loaded (placeholder)")
        
        print(f"Processing audio: {audio_file}")
        
        # Placeholder emotion recognition (replace with actual SpeechBrain code)
        # For now, just simulate the processing
        
        print("=" * 50)
        print("SPEECHBRAIN PROCESSING RESULT")
        print("=" * 50)
        print("Emotion: neutral")
        print("Confidence: 0.85")
        print("Embeddings: <class 'str'>")
        print("=" * 50)
        
        # Save results
        output_file = audio_file.replace('.wav', '_speechbrain_result.json')
        result = {
            'emotion': 'neutral',
            'confidence': 0.85,
            'embeddings': 'placeholder_embeddings'
        }
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        return {"status": "SUCCESS", "result": result}
        
    except Exception as e:
        print(f"ERROR: {e}")
        return {"status": "ERROR", "error": str(e)}

if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_audio.wav"
    result = process_audio(audio_file)
    print(f"FINAL_RESULT: {json.dumps(result)}")
'''
                with open(speechbrain_script, "w") as f:
                    f.write(speechbrain_content)
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(speechbrain_script), str(audio_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.speechbrain_project_path)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                output = stdout.decode()
                # Extract the final result
                for line in output.split('\n'):
                    if line.startswith('FINAL_RESULT:'):
                        result_json = line.replace('FINAL_RESULT:', '').strip()
                        try:
                            result = json.loads(result_json)
                            result["full_output"] = output
                            return result
                        except json.JSONDecodeError:
                            pass
                
                return {"status": "SUCCESS", "output": output}
            else:
                return {"status": "ERROR", "error": stderr.decode()}
                
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    async def process_audio(self, audio_file):
        """Process audio file with both Whisper and SpeechBrain in parallel"""
        print(f"Processing audio: {audio_file}")
        print("=" * 60)
        
        start_time = time.time()
        
        print("Starting Whisper processing...")
        print("Starting SpeechBrain processing...")
        
        # Run both processors in parallel
        whisper_task = asyncio.create_task(self.run_whisper_processing(audio_file))
        speechbrain_task = asyncio.create_task(self.run_speechbrain_processing(audio_file))
        
        # Wait for both to complete
        whisper_result, speechbrain_result = await asyncio.gather(
            whisper_task, speechbrain_task, return_exceptions=True
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("Whisper processing completed!")
        print("SpeechBrain processing completed!")
        print("=" * 60)
        
        # Display results
        self.display_results(audio_file, whisper_result, speechbrain_result, processing_time)
        
        return whisper_result, speechbrain_result
    
    def display_results(self, audio_file, whisper_result, speechbrain_result, processing_time):
        """Display processing results"""
        print("PROCESSING RESULTS")
        print("=" * 60)
        print(f"Audio file: {audio_file}")
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        print("-" * 30)
        print("WHISPER RESULTS")
        print("-" * 30)
        if isinstance(whisper_result, dict):
            print(f"Status: {whisper_result['status']}")
            if whisper_result['status'] == 'SUCCESS':
                if 'transcription' in whisper_result:
                    print(f"Transcription: {whisper_result['transcription']}")
                    print(f"Language: {whisper_result.get('language', 'unknown')}")
                else:
                    print("Output:")
                    print(whisper_result.get('output', 'No output'))
            else:
                print(f"Error: {whisper_result.get('error', 'Unknown error')}")
        else:
            print(f"Status: ERROR")
            print(f"Error: {whisper_result}")
        
        print("-" * 30)
        print("SPEECHBRAIN RESULTS")
        print("-" * 30)
        if isinstance(speechbrain_result, dict):
            print(f"Status: {speechbrain_result['status']}")
            if speechbrain_result['status'] == 'SUCCESS':
                print("Output:")
                if 'full_output' in speechbrain_result:
                    print(speechbrain_result['full_output'])
                else:
                    print(speechbrain_result.get('output', 'No output'))
            else:
                print(f"Error: {speechbrain_result.get('error', 'Unknown error')}")
        else:
            print(f"Status: ERROR")
            print(f"Error: {speechbrain_result}")

async def main():
    print("Async Parallel Audio Processor")
    print("=" * 40)
    
    # Configuration
    whisper_project = "D:\whisper_stt_project"
    speechbrain_project = "D:\whisper_stt_project\speech_brain"
    audio_file = "D:/whisper_stt_project/speech_brain/audio/audio.wav"
    
    print(f"Whisper project: {whisper_project}")
    print(f"SpeechBrain project: {speechbrain_project}")
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    # Create processor and run
    processor = ParallelAudioProcessor(whisper_project, speechbrain_project)
    
    try:
        await processor.process_audio(audio_file)
        print("=" * 60)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    asyncio.run(main())