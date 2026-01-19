import os
import librosa
import numpy as np
from tqdm import tqdm

# RAVDESS dataset path (adjust if needed)
RAVDESS_PATH = 'data'
OUTPUT_DIR = 'data/audio_features'

# MFCC parameters
SR = 22050
N_MFCC = 13
HOP_LENGTH = 512
N_FRAMES = 300

def extract_mfcc(audio_path):
    """Extract MFCC features from audio file."""
    y, _ = librosa.load(audio_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc = mfcc.T  # (time, mfcc)

    # Pad or truncate to N_FRAMES
    if mfcc.shape[0] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, N_FRAMES - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:N_FRAMES]

    return mfcc[..., np.newaxis]  # (300, 13, 1)

def process_ravdess():
    """Process all RAVDESS audio files and save MFCC features."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Output dir created")

    actor_dirs = [d for d in os.listdir(RAVDESS_PATH) if d.startswith('Actor_') and os.path.isdir(os.path.join(RAVDESS_PATH, d))]
    print(f"Found {len(actor_dirs)} actor directories: {actor_dirs[:5]}...")
    
    for actor_dir in actor_dirs:
        actor_path = os.path.join(RAVDESS_PATH, actor_dir)
        files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
        print(f"Processing {actor_dir} with {len(files)} files...")
        
        for file in tqdm(files):
            audio_path = os.path.join(actor_path, file)
            try:
                mfcc = extract_mfcc(audio_path)
                output_path = os.path.join(OUTPUT_DIR, file.replace('.wav', '.npy'))
                np.save(output_path, mfcc)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        print(f"Finished {actor_dir}")

if __name__ == "__main__":
    process_ravdess()