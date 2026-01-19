import os
import requests
from zipfile import ZipFile
import tarfile
import shutil

# Define dataset URLs and target directories
datasets = {
    'CREMA-D': {
        'url': 'https://github.com/CheyneyComputerScience/CREMA-D/archive/master.zip',  # Note: Actual download may require manual access or alternative source
        'extract_type': 'zip',
        'target_dir': 'data/CREMA-D'
    },
    'SAVEE': {
        'url': 'http://personal.ee.surrey.ac.uk/Personal/P.Jackson/SAVEE/Database.zip',  # Note: May require manual download due to licensing
        'extract_type': 'zip',
        'target_dir': 'data/SAVEE'
    },
    'Emo-DB': {
        'url': 'http://www.emodb.bilderbar.info/download/download.zip',  # Note: May require manual download
        'extract_type': 'zip',
        'target_dir': 'data/Emo-DB'
    }
}

def download_file(url, filename):
    """Download a file from URL to filename."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def extract_archive(filename, extract_type, target_dir):
    """Extract zip or tar archive to target directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    if extract_type == 'zip':
        with ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    elif extract_type == 'tar':
        with tarfile.open(filename, 'r:gz') as tar_ref:
            tar_ref.extractall(target_dir)
    
    # Clean up archive file
    os.remove(filename)

def main():
    for name, info in datasets.items():
        print(f"Downloading {name}...")
        try:
            zip_filename = f"{name}.zip"
            download_file(info['url'], zip_filename)
            print(f"Extracting {name}...")
            extract_archive(zip_filename, info['extract_type'], info['target_dir'])
            print(f"{name} downloaded and extracted successfully.")
        except Exception as e:
            print(f"Error downloading/extracting {name}: {e}")
            print(f"Please download {name} manually from {info['url']} and extract to {info['target_dir']}")

if __name__ == "__main__":
    main()