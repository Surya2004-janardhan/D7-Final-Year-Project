"""
RAVDESS Data Loader and Preprocessor for CNN/LSTM Training
Handles audio feature extraction and video frame extraction
"""

import os
import cv2
import librosa
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pickle

class RAVDESSDataLoader:
    """Load and preprocess RAVDESS audio and video data"""
    
    EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
    
    def __init__(self, data_path='data/organized'):
        self.data_path = Path(data_path)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.EMOTIONS)
        
    def extract_audio_features(self, audio_path, sr=22050, n_mfcc=13):
        """
        Extract MFCC features from audio file
        Args:
            audio_path: Path to audio file
            sr: Sampling rate
            n_mfcc: Number of MFCC coefficients
        Returns:
            numpy array of shape (n_mfcc, time_steps)
        """
        try:
            y, sr = librosa.load(audio_path, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            return mfcc
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def extract_video_frames(self, video_path, n_frames=8):
        """
        Extract frames from video
        Args:
            video_path: Path to video file
            n_frames: Number of frames to extract (8 frames for RTX 2050)
        Returns:
            numpy array of shape (n_frames, height, width, 3)
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                return None
            
            # Sample frames evenly across video
            frame_indices = np.linspace(0, frame_count - 1, n_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Resize to 160x160 (smaller for memory efficiency)
                    frame = cv2.resize(frame, (160, 160))
                    frames.append(frame)
            
            cap.release()
            
            if len(frames) == n_frames:
                return np.array(frames, dtype=np.float32) / 255.0
            return None
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
    
    def load_audio_dataset(self, modality='speech'):
        """
        Load audio dataset
        Args:
            modality: 'speech' or 'song'
        Returns:
            X: list of MFCC features
            y: list of emotion labels
        """
        audio_path = self.data_path / 'audio' / modality
        X, y = [], []
        
        for emotion_dir in audio_path.iterdir():
            if emotion_dir.is_dir():
                emotion = emotion_dir.name
                emotion_label = self.label_encoder.transform([emotion])[0]
                
                for audio_file in emotion_dir.glob('*.wav'):
                    features = self.extract_audio_features(audio_file)
                    if features is not None:
                        X.append(features)
                        y.append(emotion_label)
                        print(f"Loaded: {audio_file.name}")
        
        return X, y
    
    def load_video_dataset(self, modality='speech', n_frames=8):
        """
        Load video dataset
        Args:
            modality: 'speech' or 'song'
            n_frames: Number of frames to extract per video (8 for RTX 2050)
        Returns:
            X: list of video frames
            y: list of emotion labels
        """
        video_path = self.data_path / 'video' / modality
        X, y = [], []
        
        for emotion_dir in video_path.iterdir():
            if emotion_dir.is_dir():
                emotion = emotion_dir.name
                emotion_label = self.label_encoder.transform([emotion])[0]
                
                for video_file in emotion_dir.glob('*.mp4'):
                    frames = self.extract_video_frames(video_file, n_frames=n_frames)
                    if frames is not None:
                        X.append(frames)
                        y.append(emotion_label)
                        print(f"Loaded: {video_file.name}")
        
        return X, y
    
    def save_features(self, X, y, filepath):
        """Save extracted features to disk"""
        data = {'X': X, 'y': y, 'label_encoder': self.label_encoder}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved features to {filepath}")
    
    def load_features(self, filepath):
        """Load pre-extracted features"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y']


# Example usage
if __name__ == '__main__':
    loader = RAVDESSDataLoader()
    
    print("Loading audio features (speech)...")
    audio_X, audio_y = loader.load_audio_dataset('speech')
    print(f"Audio data shape: {len(audio_X)} samples")
    
    print("\nLoading video frames (speech)...")
    video_X, video_y = loader.load_video_dataset('speech', n_frames=30)
    print(f"Video data shape: {len(video_X)} samples")
    
    # Save features for later use
    if audio_X:
        loader.save_features(audio_X, audio_y, 'features_audio_speech.pkl')
    if video_X:
        loader.save_features(video_X, video_y, 'features_video_speech.pkl')
