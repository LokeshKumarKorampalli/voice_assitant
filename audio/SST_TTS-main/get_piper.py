import requests
import zipfile
import tarfile
import os

def download_piper():
    print("Attempting to download Piper...")
    
    # Possible URLs (you may need to check the actual release page)
    urls = [
        "https://github.com/rhasspy/piper/releases/latest/download/piper_windows_amd64.zip",
        "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_amd64.tar.gz"
    ]
    
    for url in urls:
        try:
            print(f"Trying: {url}")
            response = requests.get(url)
            if response.status_code == 200:
                filename = url.split('/')[-1]
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {filename}")
                
                # Extract
                if filename.endswith('.zip'):
                    with zipfile.ZipFile(filename) as z:
                        z.extractall('.')
                elif filename.endswith('.tar.gz'):
                    with tarfile.open(filename) as t:
                        t.extractall('.')
                
                os.remove(filename)
                print("Extracted successfully!")
                return True
        except Exception as e:
            print(f"Failed: {e}")
    
    return False

if __name__ == "__main__":
    os.chdir("D:/piper_tts")
    if download_piper():
        print("Check for piper.exe in the directory")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if 'piper' in file.lower():
                    print(f"Found: {os.path.join(root, file)}")
    else:
        print("Download failed. Please download manually from GitHub.")