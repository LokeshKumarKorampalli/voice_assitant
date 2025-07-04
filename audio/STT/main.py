#!/usr/bin/env python3
"""
Whisper Speech-to-Text Main Application
"""

import argparse
import logging
from pathlib import Path
from transcription.batch_transcription import BatchTranscriber
from config.settings import DEFAULT_MODEL, WHISPER_MODELS

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/logs/whisper_stt.log'),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Whisper Speech-to-Text System")
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, 
                       choices=list(WHISPER_MODELS.keys()),
                       help="Whisper model size")
    parser.add_argument("-l", "--language", default="en",
                       help="Audio language (en, es, fr, etc.)")
    parser.add_argument("-f", "--format", default="txt",
                       choices=["txt", "json", "srt"],
                       help="Output format")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--beam-size", type=int, default=5,
                       help="Beam size for decoding")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize transcriber
    logger.info(f"Initializing Whisper {args.model} model...")
    transcriber = BatchTranscriber(model_size=args.model)
    
    # Transcription options
    options = {
        "language": args.language if args.language != "auto" else None,
        "temperature": args.temperature,
        "beam_size": args.beam_size
    }
    
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            # Single file transcription
            result, output_file = transcriber.transcribe_file(
                input_path, 
                output_format=args.format,
                **options
            )
            if result:
                logger.info(f"✓ Transcription completed: {output_file}")
                print(f"Text: {result['text']}")
            else:
                logger.error("Transcription failed")
        
        elif input_path.is_dir():
            # Directory transcription
            results = transcriber.transcribe_directory(input_path, **options)
            logger.info(f"✓ Processed {len(results)} files")
            for result in results:
                print(f"File: {result['input_file']}")
                print(f"Text: {result['text'][:100]}...")
                print("-" * 50)
        
        else:
            logger.error(f"Input path not found: {input_path}")
    
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()