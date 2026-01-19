import os
import cv2
import numpy as np
from tqdm import tqdm

# RAVDESS video path (assuming videos are in the same folder or adjust)
RAVDESS_VIDEO_PATH = 'data'
OUTPUT_DIR = 'data/video_frames'

NUM_FRAMES = 16
TARGET_SIZE = (112, 112)

def sample_frames(video_path):
    """Sample NUM_FRAMES from video uniformly."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return None

    step = max(1, total_frames // NUM_FRAMES)
    frames = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, TARGET_SIZE) / 255.0
            frames.append(frame)
        if len(frames) == NUM_FRAMES:
            break

    cap.release()
    return np.array(frames) if len(frames) == NUM_FRAMES else None

def process_ravdess_videos():
    """Process all RAVDESS video files and save sampled frames."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for root, dirs, files in os.walk(RAVDESS_VIDEO_PATH):
        for file in tqdm(files):
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                try:
                    frames = sample_frames(video_path)
                    if frames is not None:
                        output_path = os.path.join(OUTPUT_DIR, file.replace('.mp4', '.npy'))
                        np.save(output_path, frames)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    process_ravdess_videos()