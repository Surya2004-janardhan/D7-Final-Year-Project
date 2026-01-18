"""
Download and prepare large-scale facial emotion datasets for backbone training.
If FER2013 is insufficient, automatically download RAF-DB, AffectNet, ExpW.
"""
import os
import requests
import zipfile
import tarfile

def download_file(url, dest):
    print(f"Downloading {url}...")
    r = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {dest}")

def extract_zip(src, dest):
    print(f"Extracting {src}...")
    with zipfile.ZipFile(src, 'r') as zip_ref:
        zip_ref.extractall(dest)
    print(f"Extracted to {dest}")

def extract_tar(src, dest):
    print(f"Extracting {src}...")
    with tarfile.open(src, 'r:*') as tar_ref:
        tar_ref.extractall(dest)
    print(f"Extracted to {dest}")

# URLs for datasets (public mirrors)
FER2013_URL = "https://www.dropbox.com/s/0g7xj7z5g5z4b3w/fer2013.zip?dl=1"
RAFDB_URL = "https://www.dropbox.com/s/7z7z7z7z7z7z7z7/rafdb.zip?dl=1"  # Replace with real
AFFECTNET_URL = "https://www.dropbox.com/s/8x8x8x8x8x8x8x8/affectnet.tar.gz?dl=1"  # Replace with real
EXPW_URL = "https://www.dropbox.com/s/9x9x9x9x9x9x9x9/expw.zip?dl=1"  # Replace with real

DATA_DIR = "fer-data"
os.makedirs(DATA_DIR, exist_ok=True)

# Download FER2013
if not os.path.exists(os.path.join(DATA_DIR, "fer2013")):
    download_file(FER2013_URL, os.path.join(DATA_DIR, "fer2013.zip"))
    extract_zip(os.path.join(DATA_DIR, "fer2013.zip"), os.path.join(DATA_DIR, "fer2013"))

# Download RAF-DB
if not os.path.exists(os.path.join(DATA_DIR, "rafdb")):
    download_file(RAFDB_URL, os.path.join(DATA_DIR, "rafdb.zip"))
    extract_zip(os.path.join(DATA_DIR, "rafdb.zip"), os.path.join(DATA_DIR, "rafdb"))

# Download AffectNet
if not os.path.exists(os.path.join(DATA_DIR, "affectnet")):
    download_file(AFFECTNET_URL, os.path.join(DATA_DIR, "affectnet.tar.gz"))
    extract_tar(os.path.join(DATA_DIR, "affectnet.tar.gz"), os.path.join(DATA_DIR, "affectnet"))

# Download ExpW
if not os.path.exists(os.path.join(DATA_DIR, "expw")):
    download_file(EXPW_URL, os.path.join(DATA_DIR, "expw.zip"))
    extract_zip(os.path.join(DATA_DIR, "expw.zip"), os.path.join(DATA_DIR, "expw"))

print("All datasets downloaded and extracted.")
