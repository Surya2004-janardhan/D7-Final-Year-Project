"""
RAVDESS Data Loader and Preprocessor for CNN/LSTM Training
Handles audio feature extraction and video frame extraction with face detection
"""

import os
import cv2
try:
    import librosa
except ImportError:
    librosa = None
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pickle
from face_detector import FaceDetector

class RAVDESSDataLoader:
    """Load and preprocess RAVDESS audio and video data with face detection"""
    
    EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
    
    def __init__(self, data_path='data/organized'):
        self.data_path = Path(data_path)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.EMOTIONS)
        self.face_detector = FaceDetector()  # Initialize face detector
        
        # Load face cascade detector
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print("WARNING: Face cascade not found, using full frames")
            self.face_cascade = None
        
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
    
    def extract_video_frames(self, video_path, n_frames=16):
        """
        Extract frames from video with face detection
        Args:
            video_path: Path to video file
            n_frames: Number of frames to extract (16 for better temporal info)
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
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    if self.face_cascade is not None:
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                        if len(faces) > 0:
                            # Use largest face
                            (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
                            # Add padding
                            x = max(0, x - int(w*0.2))
                            y = max(0, y - int(h*0.2))
                            w = min(frame.shape[1] - x, int(w*1.4))
                            h = min(frame.shape[0] - y, int(h*1.4))
                            
                            face = frame[y:y+h, x:x+w]
                        else:
                            face = frame  # Fallback to full frame
                    else:
                        face = frame
                    
                    # Resize to 128x128 (smaller, more efficient)
                    face = cv2.resize(face, (128, 128))
                    frames.append(face)
            
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
    
    def load_video_dataset(self, modality='speech', n_frames=16):
        """
        Load video dataset with face detection
        Args:
            modality: 'speech' or 'song'
            n_frames: Number of frames to extract per video (16 for face-only)
        Returns:
            X: list of face frames (128x128)
            y: list of emotion labels
        """
        video_path = self.data_path / 'video' / modality
        X, y = [], []
        
        for emotion_dir in video_path.iterdir():
            if emotion_dir.is_dir():
                emotion = emotion_dir.name
                emotion_label = self.label_encoder.transform([emotion])[0]
                
                for video_file in emotion_dir.glob('*.mp4'):
                    try:
                        # Extract faces from video
                        faces = self.face_detector.extract_faces_from_video(video_file, n_frames=n_frames)
                        if faces is not None:
                            X.append(faces)
                            y.append(emotion_label)
                            print(f"Loaded: {video_file.name}")
                        else:
                            print(f"Skipped (insufficient frames): {video_file.name}")
                    except Exception as e:
                        print(f"Error loading {video_file.name}: {str(e)[:50]}")
        
        return X, y
    
    def augment_video_data(self, X):
        """
        Apply data augmentation to video frames
        - Random brightness adjustment
        - Random horizontal flip
        - Random noise addition
        Helps prevent overfitting on small dataset
        """
        augmented = []
        
        for video_frames in X:  # video_frames shape: (n_frames, 128, 128, 3)
            augmented_frames = []
            
            # Randomly choose augmentation for this video
            do_flip = np.random.rand() > 0.5
            do_brightness = np.random.rand() > 0.5
            do_noise = np.random.rand() > 0.7
            brightness_factor = np.random.uniform(0.8, 1.2)
            
            for frame in video_frames:
                aug_frame = frame.copy()
                
                # Random horizontal flip
                if do_flip:
                    aug_frame = np.fliplr(aug_frame)
                
                # Random brightness
                if do_brightness:
                    aug_frame = np.clip(aug_frame * brightness_factor, 0, 1)
                
                # Random noise (small Gaussian)
                if do_noise:
                    noise = np.random.normal(0, 0.02, aug_frame.shape)
                    aug_frame = np.clip(aug_frame + noise, 0, 1)
                
                augmented_frames.append(aug_frame)
            
            augmented.append(np.array(augmented_frames))
        
        return np.array(augmented)
    
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
