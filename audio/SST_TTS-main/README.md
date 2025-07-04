# Piper Text-to-Speech Project

A text-to-speech application using Piper for offline voice synthesis.

## Important Notes

**Missing Files**: Some important files are not included in this repository:

- **Environment file** (`.env`) - Contains configuration settings
- **Piper voice models** - Large voice model files are not uploaded due to size

## Setup Instructions

### 1. Install Piper
Download Piper from the official repository:
```bash
# For Linux/Mac
wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz
tar -xzf piper_linux_x86_64.tar.gz

# For Windows
Download piper_windows_amd64.zip from GitHub releases
```

### 2. Download Voice Models
- Go to https://github.com/rhasspy/piper/releases/tag/v0.0.2
- Download your preferred voice model (e.g., `en_US-lessac-medium.onnx`)
- Download the corresponding JSON file (e.g., `en_US-lessac-medium.onnx.json`)
- Place both files in a `models/` folder

### 3. Create Environment File
Create a new file called `.env` in the main folder:

```
PIPER_PATH=./piper/piper
MODEL_PATH=./models/en_US-lessac-medium.onnx
OUTPUT_PATH=./output/
```

*Update the paths based on where you placed Piper and your voice models*

### 4. Install Python Requirements
```bash
pip install -r requirements.txt
```

### 5. Run the Project
```bash
python main.py
```

## What You Need

1. **Piper Binary** - Download the Piper executable for your operating system
2. **Voice Models** - Download .onnx voice model files and their .json config files
3. **Environment Setup** - Create the `.env` file with correct paths to Piper and models

## Folder Structure
```
project/
├── main.py
├── requirements.txt
├── .env (you need to create this)
├── piper/ (extract Piper here)
│   └── piper (executable)
├── models/ (place voice models here)
│   ├── en_US-lessac-medium.onnx
│   └── en_US-lessac-medium.onnx.json
├── output/ (generated audio files)
└── README.md
```

## Available Voice Models

Popular English voices:
- `en_US-lessac-medium` - Clear American English
- `en_US-amy-medium` - Female American English
- `en_GB-alan-medium` - British English
