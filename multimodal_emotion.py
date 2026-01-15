"""
Multimodal Emotion Recognition System
Combines:
1. FER-trained base model for face emotion features
2. Temporal video model (2D CNN + LSTM) for video emotion
3. Pre-trained audio models for speech emotion

Final fusion of audio and video predictions for robust emotion detection
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from pathlib import Path
import librosa
from tqdm import tqdm

# Emotion labels
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
NUM_EMOTIONS = 7


class FaceDetector:
    """Haar Cascade face detector"""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            local_path = 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(local_path)
    
    def detect(self, frame, target_size=(48, 48)):
        """Detect face and return grayscale resized"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Padding
        pad = int(w * 0.1)
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)
        
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        
        return face


class MultimodalEmotionRecognizer:
    """
    Combined audio + video emotion recognition system
    """
    
    def __init__(self, n_frames=16):
        self.n_frames = n_frames
        self.face_detector = FaceDetector()
        
        # Models (load later)
        self.fer_model = None
        self.temporal_video_model = None
        self.audio_model = None
        self.fusion_model = None
    
    def load_models(self, 
                    fer_path='models/fer_base_model.h5',
                    temporal_video_path='models/temporal_video_emotion.h5',
                    audio_speech_path='models/audio_emotion_speech.h5'):
        """Load all pre-trained models"""
        
        if os.path.exists(fer_path):
            self.fer_model = keras.models.load_model(fer_path)
            print(f"Loaded FER model from {fer_path}")
        
        if os.path.exists(temporal_video_path):
            self.temporal_video_model = keras.models.load_model(temporal_video_path)
            print(f"Loaded Temporal Video model from {temporal_video_path}")
        
        if os.path.exists(audio_speech_path):
            self.audio_model = keras.models.load_model(audio_speech_path)
            print(f"Loaded Audio model from {audio_speech_path}")
    
    def extract_video_features(self, video_path):
        """
        Extract face sequence from video
        Returns: (n_frames, 48, 48, 1) array
        """
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            cap.release()
            return None
        
        indices = np.linspace(0, frame_count - 1, self.n_frames, dtype=int)
        
        faces = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            face = self.face_detector.detect(frame, (48, 48))
            if face is not None:
                faces.append(face)
            elif len(faces) > 0:
                faces.append(faces[-1])
        
        cap.release()
        
        if len(faces) < self.n_frames // 2:
            return None
        
        while len(faces) < self.n_frames:
            faces.append(faces[-1] if faces else np.zeros((48, 48)))
        
        faces = np.array(faces[:self.n_frames], dtype=np.float32) / 255.0
        faces = np.expand_dims(faces, axis=-1)
        
        return faces
    
    def extract_audio_features(self, audio_path, sr=22050, n_mfcc=40, max_len=216):
        """
        Extract MFCC features from audio
        Returns: (n_mfcc, max_len) array
        """
        try:
            y, sr = librosa.load(str(audio_path), sr=sr, duration=3.0)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
            # Pad or truncate
            if mfcc.shape[1] < max_len:
                mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
            else:
                mfcc = mfcc[:, :max_len]
            
            return mfcc
        
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def predict_video(self, video_path):
        """Predict emotion from video only"""
        if self.temporal_video_model is None:
            print("Temporal video model not loaded!")
            return None
        
        faces = self.extract_video_features(video_path)
        if faces is None:
            return None
        
        faces = np.expand_dims(faces, axis=0)  # Add batch dim
        pred = self.temporal_video_model.predict(faces, verbose=0)
        
        return pred[0]
    
    def predict_audio(self, audio_path):
        """Predict emotion from audio only"""
        if self.audio_model is None:
            print("Audio model not loaded!")
            return None
        
        mfcc = self.extract_audio_features(audio_path)
        if mfcc is None:
            return None
        
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dim
        pred = self.audio_model.predict(mfcc, verbose=0)
        
        return pred[0]
    
    def predict_multimodal(self, video_path, audio_path=None, fusion='average'):
        """
        Predict emotion using both video and audio
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file (if None, extracts from video)
            fusion: 'average', 'weighted', or 'max'
        
        Returns:
            emotion_label, confidence, predictions_dict
        """
        predictions = {}
        
        # Video prediction
        video_pred = self.predict_video(video_path)
        if video_pred is not None:
            predictions['video'] = video_pred
        
        # Audio prediction
        if audio_path is None:
            # Try to extract audio from video
            audio_path = video_path.replace('.mp4', '.wav')
        
        if audio_path and os.path.exists(audio_path):
            audio_pred = self.predict_audio(audio_path)
            if audio_pred is not None:
                predictions['audio'] = audio_pred
        
        if len(predictions) == 0:
            return None, 0, {}
        
        # Fusion
        if fusion == 'average':
            combined = np.mean(list(predictions.values()), axis=0)
        elif fusion == 'weighted':
            # Weight video more (0.6 video, 0.4 audio)
            if 'video' in predictions and 'audio' in predictions:
                combined = 0.6 * predictions['video'] + 0.4 * predictions['audio']
            else:
                combined = list(predictions.values())[0]
        elif fusion == 'max':
            # Take max confidence per class
            combined = np.max(list(predictions.values()), axis=0)
        else:
            combined = np.mean(list(predictions.values()), axis=0)
        
        # Get final prediction
        emotion_idx = np.argmax(combined)
        confidence = combined[emotion_idx]
        emotion_label = EMOTIONS[emotion_idx] if emotion_idx < len(EMOTIONS) else 'unknown'
        
        return emotion_label, confidence, predictions
    
    def predict_frame(self, frame):
        """
        Predict emotion from a single frame using FER model
        For real-time applications
        """
        if self.fer_model is None:
            print("FER model not loaded!")
            return None
        
        face = self.face_detector.detect(frame, (48, 48))
        if face is None:
            return None
        
        face = face.astype(np.float32) / 255.0
        face = np.expand_dims(face, axis=(0, -1))  # (1, 48, 48, 1)
        
        pred = self.fer_model.predict(face, verbose=0)
        
        return pred[0]


class LateFusionModel:
    """
    Late fusion model that combines video and audio predictions
    Learns optimal fusion weights
    """
    
    def __init__(self, num_emotions=7):
        self.num_emotions = num_emotions
        self.model = None
    
    def build_model(self, video_dim=7, audio_dim=8):
        """
        Build fusion network
        Takes prediction vectors from video and audio models
        """
        video_input = layers.Input(shape=(video_dim,), name='video_pred')
        audio_input = layers.Input(shape=(audio_dim,), name='audio_pred')
        
        # Concatenate predictions
        concat = layers.Concatenate()([video_input, audio_input])
        
        # Learnable fusion
        x = layers.Dense(64, activation='relu')(concat)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        outputs = layers.Dense(self.num_emotions, activation='softmax')(x)
        
        self.model = models.Model(
            inputs=[video_input, audio_input],
            outputs=outputs
        )
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, video_preds, audio_preds, y, epochs=50, batch_size=32):
        """Train fusion model"""
        from sklearn.model_selection import train_test_split
        
        X_video_train, X_video_val, X_audio_train, X_audio_val, y_train, y_val = \
            train_test_split(video_preds, audio_preds, y, test_size=0.2, random_state=42)
        
        history = self.model.fit(
            [X_video_train, X_audio_train], y_train,
            validation_data=([X_video_val, X_audio_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def save(self, path='models/fusion_model.h5'):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
    
    def load(self, path='models/fusion_model.h5'):
        """Load model"""
        self.model = keras.models.load_model(path)


def main():
    """Demo: Test the multimodal system"""
    
    print("=" * 60)
    print("Multimodal Emotion Recognition System")
    print("=" * 60)
    
    # Initialize recognizer
    recognizer = MultimodalEmotionRecognizer(n_frames=16)
    
    # Load models
    print("\nLoading models...")
    recognizer.load_models()
    
    # Test on sample data
    data_dir = Path('data')
    actor_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('Actor')]
    
    if actor_dirs:
        # Get a sample video
        sample_dir = actor_dirs[0]
        videos = list(sample_dir.glob('*.mp4'))
        
        if videos:
            sample_video = videos[0]
            print(f"\nTesting on: {sample_video}")
            
            # Predict
            emotion, confidence, preds = recognizer.predict_multimodal(
                str(sample_video),
                fusion='average'
            )
            
            if emotion:
                print(f"\nPredicted Emotion: {emotion}")
                print(f"Confidence: {confidence:.2%}")
                
                if 'video' in preds:
                    video_emotion = EMOTIONS[np.argmax(preds['video'])]
                    print(f"Video Prediction: {video_emotion} ({np.max(preds['video']):.2%})")
                
                if 'audio' in preds:
                    print(f"Audio Prediction: {np.argmax(preds['audio'])} ({np.max(preds['audio']):.2%})")
            else:
                print("Could not make prediction")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
