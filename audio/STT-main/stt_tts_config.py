import os
import json
from pathlib import Path

def create_config():
    """Create simple configuration file"""
    print("STT-TTS Configuration Setup")
    print("===========================")
    
    config = {
        "whisper_model": "base",
        "piper_tts_path": r"E:\Voice_assitant\SST_TTS-main",
        "whisper_stt_path": r"E:\Voice_assitant\STT-main",
        "ffmpeg_path": r"E:\Voice_assitant\STT-main\ffmpeg-7.1.1-essentials_build\bin"
    }
    
    # Whisper model selection
    print("\nWhisper Model Selection:")
    print("1. tiny")
    print("2. base") 
    print("3. small")
    print("4. medium")
    print("5. large")
    
    choice = input("Choose model [default: base]: ").strip()
    models = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large"}
    config["whisper_model"] = models.get(choice, "base")
    
    # Path configurations
    print("\nPath Configuration:")
    
    piper_path = input(f"Piper TTS path [{config['piper_tts_path']}]: ").strip()
    if piper_path:
        config["piper_tts_path"] = piper_path
    
    stt_path = input(f"Whisper STT path [{config['whisper_stt_path']}]: ").strip()
    if stt_path:
        config["whisper_stt_path"] = stt_path
    
    ffmpeg_path = input(f"FFmpeg path [{config['ffmpeg_path']}]: ").strip()
    if ffmpeg_path:
        config["ffmpeg_path"] = ffmpeg_path
    
    # Verify paths
    print("\nVerifying paths...")
    paths_to_check = [
        ("Piper TTS", config["piper_tts_path"]),
        ("Whisper STT", config["whisper_stt_path"]),
        ("FFmpeg", config["ffmpeg_path"])
    ]
    
    all_valid = True
    for name, path in paths_to_check:
        if os.path.exists(path):
            print(f"Valid: {name} path exists: {path}")
        else:
            print(f"Error: {name} path not found: {path}")
            all_valid = False
    
    if not all_valid:
        print("\nSome paths are invalid. Please check and run again.")
        return None
    
    # Save configuration
    with open("stt_tts_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"\nConfiguration saved to stt_tts_config.json")
    return config

def load_config():
    """Load existing configuration"""
    if os.path.exists("stt_tts_config.json"):
        with open("stt_tts_config.json", "r") as f:
            return json.load(f)
    return None

def main():
    # Check if config exists
    existing_config = load_config()
    
    if existing_config:
        print("Found existing configuration:")
        print(json.dumps(existing_config, indent=2))
        
        use_existing = input("\nUse existing configuration? (y/n) [y]: ").strip().lower()
        if use_existing in ['', 'y', 'yes']:
            config = existing_config
        else:
            config = create_config()
            if config is None:
                return
    else:
        config = create_config()
        if config is None:
            return
    
    # Set up environment
    ffmpeg_path = config["ffmpeg_path"]
    if ffmpeg_path not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
        print(f"Added FFmpeg to PATH: {ffmpeg_path}")
    
    print("\nSetup complete!")
    print("Run: python stt_tts_integration.py")

if __name__ == "__main__":
    main()