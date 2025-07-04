import subprocess
import os
import csv
import json
from pathlib import Path
from datetime import datetime
import argparse

class BatchPiperTTS:
    def __init__(self, piper_dir="D:/piper_tts"):
        self.piper_dir = Path(piper_dir)
        self.piper_exe = self.piper_dir / "piper_models" / "piper" / "piper.exe"
        self.model_path = self.piper_dir / "piper_models" / "en_US-lessac-medium.onnx"
        self.output_dir = self.piper_dir / "output"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
    
    def process_text_list(self, texts, prefix="audio", start_index=1):
        """Process a list of texts"""
        results = []
        total = len(texts)
        
        print(f"Processing {total} texts...")
        
        for i, text in enumerate(texts):
            if not text.strip():
                continue
                
            filename = f"{prefix}_{i + start_index:03d}.wav"
            success = self._convert_single(text, filename, i + 1, total)
            
            results.append({
                'index': i + start_index,
                'filename': filename,
                'text': text,
                'success': success
            })
        
        return results
    
    def process_from_file(self, file_path, file_type='auto'):
        """Process texts from various file formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Error: File {file_path} not found")
            return []
        
        # Auto-detect file type
        if file_type == 'auto':
            ext = file_path.suffix.lower()
            if ext == '.csv':
                file_type = 'csv'
            elif ext == '.json':
                file_type = 'json'
            else:
                file_type = 'txt'
        
        # Read texts based on file type
        if file_type == 'txt':
            texts = self._read_txt_file(file_path)
        elif file_type == 'csv':
            texts = self._read_csv_file(file_path)
        elif file_type == 'json':
            texts = self._read_json_file(file_path)
        else:
            print(f"Unsupported file type: {file_type}")
            return []
        
        if not texts:
            print("No texts found in file")
            return []
        
        return self.process_text_list(texts, prefix=file_path.stem)
    
    def _read_txt_file(self, file_path):
        """Read texts from plain text file (one per line)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _read_csv_file(self, file_path):
        """Read texts from CSV file (assumes first column contains text)"""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip():
                    texts.append(row[0].strip())
        return texts
    
    def _read_json_file(self, file_path):
        """Read texts from JSON file (assumes array of strings or objects with 'text' field)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and 'text' in item:
                    texts.append(item['text'])
        
        return texts
    
    def _convert_single(self, text, filename, current, total):
        """Convert single text to audio"""
        output_path = self.output_dir / filename
        
        print(f"[{current}/{total}] {text[:50]}... -> {filename}")
        
        try:
            process = subprocess.Popen(
                [str(self.piper_exe), "--model", str(self.model_path), "--output_file", str(output_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.piper_exe.parent)
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode == 0 and output_path.exists():
                print(f"  ✓ Success: {filename}")
                return True
            else:
                print(f"  ✗ Failed: {stderr}")
                return False
                
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            return False
    
    def generate_report(self, results, output_file="batch_report.txt"):
        """Generate processing report"""
        report_path = self.output_dir / output_file
        
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Batch TTS Processing Report\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Success Rate: {successful}/{total} ({successful/total*100:.1f}%)\n\n")
            
            for result in results:
                status = "✓" if result['success'] else "✗"
                f.write(f"{status} {result['filename']}: {result['text'][:100]}\n")
        
        print(f"Report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Batch Text-to-Speech using Piper')
    parser.add_argument('--file', '-f', help='Input file path')
    parser.add_argument('--type', '-t', choices=['txt', 'csv', 'json', 'auto'], default='auto', help='File type')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    tts = BatchPiperTTS()
    
    if args.file:
        # Process from file
        results = tts.process_from_file(args.file, args.type)
        if results:
            tts.generate_report(results)
    elif args.interactive:
        # Interactive mode
        print("=== Interactive Batch TTS ===")
        print("Enter texts (one per line). Type 'END' to finish:")
        
        texts = []
        while True:
            text = input("Text: ").strip()
            if text.upper() == 'END':
                break
            if text:
                texts.append(text)
        
        if texts:
            results = tts.process_text_list(texts)
            tts.generate_report(results)
    else:
        # Demo mode with sample texts
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Speech synthesis technology has advanced significantly in recent years.",
            "This is a demonstration of batch text-to-speech processing.",
            "Machine learning models can generate human-like speech patterns.",
            "Voice cloning and synthesis raise important ethical considerations."
        ]
        
        print("Demo mode: Processing sample texts...")
        results = tts.process_text_list(sample_texts, prefix="demo")
        tts.generate_report(results, "demo_report.txt")

if __name__ == "__main__":
    main()