# Speech-to-Text and Text-to-Speech Integration System

A real-time speech recognition and synthesis system using OpenAI's Whisper for STT and Piper TTS for speech generation.

## Features

- Real-time Speech Recognition: Live audio capture and transcription using Whisper
- Text-to-Speech Generation: High-quality voice synthesis using Piper TTS
- Interactive Mode: Continuous conversation with voice responses
- Batch Processing: Process multiple audio files at once
- Configurable Models: Support for different Whisper model sizes
- Cross-Platform: Works on Windows, Linux, and macOS

## Project Structure

```
whisper_stt_project/
├── demo.py                     # Basic Whisper demo
├── live_captions.py           # Live captioning system
├── stt_tts_integration.py     # Main integration system
├── stt_tts_config.py          # Configuration generator
├── stt_tts_config.json        # Configuration file (auto-generated)
├── test_audio/                # Sample audio files
│   └── audio.wav
├── requirements.txt           # Python dependencies
└── README.md                  # This file

piper_tts/                     # Piper TTS installation (separate)
├── piper.exe                  # Piper executable
├── piper_models/              # Voice models
│   ├── piper/
│   │   └── piper.exe
│   ├── en_US-lessac-medium.onnx
│   └── en_US-lessac-medium.onnx.json
├── stt_tts_wrapper.ps1        # PowerShell TTS wrapper
├── temp/                      # Temporary files
└── output/                    # Generated audio files
```

## Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- PowerShell (Windows) or equivalent shell
- Microphone for real-time input
- Speakers/headphones for audio output

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd whisper_stt_project
```

### 2. Create Virtual Environment

```bash
python -m venv whisper_env
```

**Activate the environment:**

Windows:
```bash
whisper_env\Scripts\activate
```

Linux/macOS:
```bash
source whisper_env/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

**Windows:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `D:\whisper_stt_project\ffmpeg-7.1.1-essentials_build\`)
3. The demo.py file already includes FFmpeg path setup

**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 5. Set up Piper TTS

#### Download Piper TTS

1. Go to https://github.com/rhasspy/piper/releases
2. Download the appropriate release for your platform:
   - Windows: `piper_windows_amd64.zip`
   - Linux: `piper_linux_x86_64.tar.gz`
   - macOS: `piper_macos_x64.tar.gz`

#### Install Piper TTS

1. Create the Piper directory structure:
```
D:\piper_tts\                  # Or your preferred location
├── piper_models\
│   └── piper\
├── temp\
└── output\
```

2. Extract the downloaded Piper archive
3. Copy `piper.exe` (or `piper` on Linux/macOS) to `D:\piper_tts\piper_models\piper\`

#### Download Voice Models

1. Go to https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium
2. Download both files:
   - `en_US-lessac-medium.onnx`
   - `en_US-lessac-medium.onnx.json`
3. Place them in `D:\piper_tts\piper_models\`

#### Create PowerShell Wrapper (Windows)

Create `D:\piper_tts\stt_tts_wrapper.ps1`:

```powershell
param(
    [Parameter(Mandatory=$true)]
    [string]$Text,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputFile = "D:\piper_tts\output\speech.wav"
)

# PATHS - Update these to match your setup
$piperPath = "D:\piper_tts\piper_models\piper\piper.exe"
$modelPath = "D:\piper_tts\piper_models\en_US-lessac-medium.onnx"
$tempDir = "D:\piper_tts\temp"
$outputDir = Split-Path $OutputFile -Parent

# Create directories if they don't exist
if (-not (Test-Path $tempDir)) { New-Item -ItemType Directory -Force -Path $tempDir | Out-Null }
if (-not (Test-Path $outputDir)) { New-Item -ItemType Directory -Force -Path $outputDir | Out-Null }

# Validate paths
if (-not (Test-Path $piperPath)) {
    Write-Error "Piper executable not found at: $piperPath"
    exit 1
}

if (-not (Test-Path $modelPath)) {
    Write-Error "Model file not found at: $modelPath"
    exit 1
}

# Create temporary text file
$tempFile = Join-Path $tempDir "temp_$(Get-Random).txt"
$Text | Out-File -FilePath $tempFile -Encoding UTF8 -NoNewline

try {
    # Use simpler approach with .NET Process class
    $arguments = "--model `"$modelPath`" --output_file `"$OutputFile`""
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $piperPath
    $processInfo.Arguments = $arguments
    $processInfo.RedirectStandardInput = $true
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo
    $process.Start()
    
    # Send text to stdin
    $process.StandardInput.WriteLine($Text)
    $process.StandardInput.Close()
    
    # Wait for completion
    $process.WaitForExit()
    
    if ($process.ExitCode -eq 0 -and (Test-Path $OutputFile)) {
        Write-Host "TTS Success: Audio generated at $OutputFile"
        Start-Process -FilePath $OutputFile -WindowStyle Hidden
        exit 0
    } else {
        Write-Error "TTS generation failed with exit code: $($process.ExitCode)"
        exit 1
    }
    
} catch {
    Write-Error "TTS Error: $($_.Exception.Message)"
    exit 1
} finally {
    # Clean up temp file
    if (Test-Path $tempFile) {
        Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
    }
}
```

### 6. Generate Configuration

Run the configuration generator:

```bash
python stt_tts_config.py
```

This will create `stt_tts_config.json` with your system paths.

## Usage

### Basic Demo

Test Whisper installation:

```bash
python demo.py
```

### Live Captioning

Real-time speech-to-text without TTS:

```bash
python live_captions.py
```

### Full STT-TTS Integration

Complete speech recognition and synthesis:

```bash
python stt_tts_integration.py
```

Choose from:
1. Interactive mode (live conversation)
2. Batch processing (process audio files)

## Configuration

### Whisper Models

Available models (size vs accuracy trade-off):
- `tiny`: Fastest, least accurate
- `base`: Good balance (recommended for CPU)
- `small`: Better accuracy, slower
- `medium`: High accuracy, much slower
- `large`: Best accuracy, very slow

Edit `stt_tts_config.json` to change the model:

```json
{
    "whisper_model": "base",
    "piper_tts_path": "D:\\piper_tts"
}
```

### Audio Settings

Default settings in the code:
- Sample Rate: 16kHz (optimal for Whisper)
- Channels: 1 (mono)
- Chunk Size: 1024 samples
- Processing Interval: 3 seconds

## Troubleshooting

### Common Issues

**1. FFmpeg not found**
- Ensure FFmpeg is in your PATH or update the path in demo.py

**2. Piper TTS errors**
- Verify piper.exe exists at the specified path
- Check that model files (.onnx and .onnx.json) are present
- Test PowerShell script manually: `.\stt_tts_wrapper.ps1 -Text "test"`

**3. Audio input issues**
- Check microphone permissions
- Verify audio device is working
- Try different audio input devices

**4. PowerShell execution policy**
- Run PowerShell as administrator
- Execute: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**5. Python import errors**
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

### Performance Optimization

**For better real-time performance:**
- Use `tiny` or `base` Whisper models
- Reduce processing interval (RECORD_SECONDS)
- Use dedicated GPU if available (requires CUDA setup)

**For better accuracy:**
- Use `small` or `medium` Whisper models
- Increase processing interval
- Ensure good audio quality (quiet environment, good microphone)

## Dependencies

### Python Packages

```
openai-whisper>=20231117
torch>=2.0.0
torchaudio>=2.0.0
pyaudio>=0.2.11
numpy>=1.21.0
```

### System Requirements

- **CPU**: Modern multi-core processor (Intel/AMD)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for models
- **Audio**: Working microphone and speakers/headphones

## Model Downloads

Note: Whisper models are downloaded automatically on first use:
- `tiny`: ~39 MB
- `base`: ~74 MB  
- `small`: ~244 MB
- `medium`: ~769 MB
- `large`: ~1550 MB

## License

This project uses:
- OpenAI Whisper: MIT License
- Piper TTS: MIT License
- PyAudio: MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review system requirements
3. Test individual components (demo.py, live_captions.py)
4. Create an issue with detailed error messages and system information
