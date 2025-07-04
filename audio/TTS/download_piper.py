import requests
import os
import tarfile
import zipfile
from pathlib import Path

def download_piper():
    # Try different possible URLs
    urls = [
        "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_amd64.tar.gz",
        "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz",
        "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
    ]
    
    for url in urls:
        try:
            print(f"Trying: {url}")
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                filename = url.split('/')[-1]
                print(f"Downloading {filename}...")
                
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Downloaded: {filename}")
                
                # Extract the file
                if filename.endswith('.tar.gz'):
                    with tarfile.open(filename, 'r:gz') as tar:
                        tar.extractall('.')
                elif filename.endswith('.zip'):
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall('.')
                
                print("Extracted successfully!")
                os.remove(filename)  # Clean up
                return True
                
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
    print("All download attempts failed.")
    return False

if __name__ == "__main__":
    os.chdir("D:/piper_tts")
    download_piper()
    
    # List files to see what was downloaded
    print("\nFiles in directory:")
    for item in os.listdir('.'):
        print(f"  {item}")