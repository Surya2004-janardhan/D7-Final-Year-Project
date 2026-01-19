import os
import numpy as np

FEATURE_DIR = 'data/audio_features'

files = os.listdir(FEATURE_DIR)
print(f"Found {len(files)} files in {FEATURE_DIR}")

if files:
    sample_file = os.path.join(FEATURE_DIR, files[0])
    mfcc = np.load(sample_file)
    print(f"Sample file {files[0]} shape: {mfcc.shape}")
else:
    print("No files found")