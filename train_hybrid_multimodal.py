"""
Hybrid Multimodal Emotion Recognition System
==============================================
Video Stream: Face extraction → Pretrained CNN (MobileNet) → Visual emotion features
Audio Stream: MFCC features → LSTM → Audio emotion features
Fusion: Late fusion of both predictions

This approach is more effective than temporal video-only models.
"""

import os
import numpy as np
import cv2
import librosa
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import gc

# Force CPU memory management
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.config.set_visible_devices([], 'GPU')  # Force CPU only
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# RAVDESS emotion mapping
RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Simplified to 5 main emotions for better accuracy
EMOTION_TO_IDX = {
    'neutral': 0, 'calm': 0,  # Merge calm with neutral
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fearful': 4,
    'disgust': 4,  # Merge disgust with fearful (both negative)
    'surprised': 1  # Merge surprised with happy (both positive)
}

EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful']
NUM_EMOTIONS = 5


class FaceExtractor:
    """Extract faces from video frames"""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def extract_face(self, frame, target_size=(96, 96)):
        """Extract and resize face from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding
        pad = int(w * 0.2)
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)
        
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        return face
    
    def extract_best_face_from_video(self, video_path, target_size=(96, 96)):
        """Extract the best (middle) face from video"""
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            cap.release()
            return None
        
        # Try multiple frames to find a good face
        frame_indices = [frame_count // 2, frame_count // 3, 2 * frame_count // 3]
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                face = self.extract_face(frame, target_size)
                if face is not None:
                    cap.release()
                    return face
        
        cap.release()
        return None


class AudioFeatureExtractor:
    """Extract MFCC features from audio"""
    
    def __init__(self, sr=22050, n_mfcc=40, max_len=100):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len
    
    def extract_mfcc(self, audio_path):
        """Extract MFCC features from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.sr, duration=3.0)
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # Normalize
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            
            # Pad or truncate to fixed length
            if mfcc.shape[1] < self.max_len:
                mfcc = np.pad(mfcc, ((0, 0), (0, self.max_len - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :self.max_len]
            
            # Transpose to (time, features) for LSTM
            return mfcc.T  # Shape: (max_len, n_mfcc)
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None
    
    def extract_from_video(self, video_path):
        """Extract audio from video and get MFCC"""
        # For RAVDESS, audio files have same name but .wav extension
        audio_path = str(video_path).replace('.mp4', '.wav')
        
        if os.path.exists(audio_path):
            return self.extract_mfcc(audio_path)
        
        # Try to extract audio from video using moviepy/ffmpeg
        try:
            import subprocess
            temp_audio = 'temp_audio.wav'
            subprocess.run([
                'ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
                '-ar', '22050', '-ac', '1', temp_audio, '-y'
            ], capture_output=True, check=True)
            
            mfcc = self.extract_mfcc(temp_audio)
            os.remove(temp_audio)
            return mfcc
        except:
            return None


def build_video_model(input_shape=(96, 96, 3), num_classes=5):
    """
    Build EfficientNetB0 video model for transfer learning
    """
    from tensorflow.keras.applications import EfficientNetB0
    inputs = layers.Input(shape=input_shape)
    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    base.trainable = False  # Freeze backbone for fusion
    x = base(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base


def build_audio_model(input_shape=(100, 40), num_classes=5):
    """
    Build audio emotion model using LSTM
    Input: MFCC features (time_steps, n_mfcc)
    """
    inputs = layers.Input(shape=input_shape)
    
    # LSTM layers for temporal modeling
    x = layers.LSTM(128, return_sequences=True, dropout=0.3)(inputs)
    x = layers.LSTM(64, dropout=0.3)(x)
    
    # Classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_fusion_model(video_features=5, audio_features=5, num_classes=5):
    """
    Late fusion model combining video and audio predictions
    """
    video_input = layers.Input(shape=(video_features,), name='video_pred')
    audio_input = layers.Input(shape=(audio_features,), name='audio_pred')
    
    # Concatenate predictions
    concat = layers.Concatenate()([video_input, audio_input])
    
    # Learnable fusion
    x = layers.Dense(32, activation='relu')(concat)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=[video_input, audio_input], outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_ravdess_data(data_dir='data'):
    """
    Load RAVDESS data for both video and audio streams
    RAVDESS naming: XX-XX-EE-XX-XX-XX-XX where EE is emotion
    - 01, 02: Video (01=speech, 02=song)
    - 03: Audio only
    
    Returns: video_data, audio_data, labels
    """
    face_extractor = FaceExtractor()
    audio_extractor = AudioFeatureExtractor()
    
    video_data = []
    audio_data = []
    labels = []
    
    data_path = Path(data_dir)
    actor_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('Actor')])
    
    print(f"Loading data from {len(actor_dirs)} actors...")
    
    for actor_dir in tqdm(actor_dirs):
        # Get video files (modality 01 or 02)
        video_files = list(actor_dir.glob('01-*.mp4')) + list(actor_dir.glob('02-*.mp4'))
        
        for video_file in video_files:
            parts = video_file.stem.split('-')
            if len(parts) >= 7:
                emotion_code = parts[2]
                
                if emotion_code in RAVDESS_EMOTIONS:
                    emotion_name = RAVDESS_EMOTIONS[emotion_code]
                    
                    if emotion_name in EMOTION_TO_IDX:
                        # Extract face from video
                        face = face_extractor.extract_best_face_from_video(video_file)
                        
                        if face is None:
                            continue
                        
                        # Find matching audio file (03-XX-EE-XX-XX-XX-XX.wav)
                        # Audio has same emotion and other codes, just modality=03
                        audio_pattern = f"03-*-{parts[2]}-{parts[3]}-{parts[4]}-{parts[5]}-{parts[6]}.wav"
                        audio_files = list(actor_dir.glob(audio_pattern))
                        
                        if not audio_files:
                            # Try simpler pattern - just match emotion
                            audio_files = list(actor_dir.glob(f"03-*-{parts[2]}-*.wav"))
                        
                        if audio_files:
                            mfcc = audio_extractor.extract_mfcc(audio_files[0])
                            
                            if mfcc is not None:
                                video_data.append(face.astype(np.float32) / 255.0)
                                audio_data.append(mfcc)
                                labels.append(EMOTION_TO_IDX[emotion_name])
    
    # If no paired data, load separately
    if len(video_data) == 0:
        print("\nNo paired video-audio data found. Loading separately...")
        
        for actor_dir in tqdm(actor_dirs):
            # Load audio files directly (03-XX files)
            audio_files = list(actor_dir.glob('03-*.wav'))
            
            for audio_file in audio_files:
                parts = audio_file.stem.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    
                    if emotion_code in RAVDESS_EMOTIONS:
                        emotion_name = RAVDESS_EMOTIONS[emotion_code]
                        
                        if emotion_name in EMOTION_TO_IDX:
                            mfcc = audio_extractor.extract_mfcc(audio_file)
                            
                            if mfcc is not None:
                                audio_data.append(mfcc)
                                labels.append(EMOTION_TO_IDX[emotion_name])
            
            # Load video files
            video_files = list(actor_dir.glob('*.mp4'))
            
            for video_file in video_files:
                parts = video_file.stem.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    
                    if emotion_code in RAVDESS_EMOTIONS:
                        emotion_name = RAVDESS_EMOTIONS[emotion_code]
                        
                        if emotion_name in EMOTION_TO_IDX:
                            face = face_extractor.extract_best_face_from_video(video_file)
                            
                            if face is not None:
                                video_data.append(face.astype(np.float32) / 255.0)
        
        # Make arrays same length (use minimum)
        min_len = min(len(video_data), len(audio_data))
        video_data = video_data[:min_len]
        audio_data = audio_data[:min_len]
        labels = labels[:min_len]
    
    return np.array(video_data), np.array(audio_data), np.array(labels)


def train_hybrid_system():
    """Train the complete hybrid emotion recognition system"""
    
    print("=" * 70)
    print("HYBRID MULTIMODAL EMOTION RECOGNITION SYSTEM")
    print("Video: Pretrained CNN (MobileNetV2) | Audio: LSTM on MFCC")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading RAVDESS data...")
    video_data, audio_data, labels = load_ravdess_data()
    
    print(f"\nDataset loaded:")
    print(f"  Video data shape: {video_data.shape}")
    print(f"  Audio data shape: {audio_data.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"  {EMOTIONS[u]}: {c}")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"\nClass weights: {class_weight_dict}")
    
    # Split data
    X_video_train, X_video_val, X_audio_train, X_audio_val, y_train, y_val = train_test_split(
        video_data, audio_data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # One-hot encode labels
    y_train_onehot = keras.utils.to_categorical(y_train, NUM_EMOTIONS)
    y_val_onehot = keras.utils.to_categorical(y_val, NUM_EMOTIONS)
    
    print(f"\nTraining: {len(y_train)}, Validation: {len(y_val)}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    # ==================== TRAIN VIDEO MODEL ====================
    print("\n" + "=" * 50)
    print("[2/5] Training VIDEO model (Simple CNN)...")
    print("=" * 50)
    
    gc.collect()  # Clear memory before training
    
    video_model, _ = build_video_model(input_shape=(96, 96, 3), num_classes=NUM_EMOTIONS)
    print(f"Video model parameters: {video_model.count_params():,}")
    
    # Train CNN
    print("\nTraining video CNN...")
    video_model.fit(
        X_video_train, y_train_onehot,
        validation_data=(X_video_val, y_val_onehot),
        epochs=30,
        batch_size=8,  # Smaller batch size for memory
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    video_acc = video_model.evaluate(X_video_val, y_val_onehot, verbose=0)[1]
    print(f"\n>>> Video Model Accuracy: {video_acc*100:.2f}%")
    
    video_model.save('models/video_emotion_mobilenet.h5')
    print("Saved: models/video_emotion_mobilenet.h5")
    
    # ==================== TRAIN AUDIO MODEL ====================
    print("\n" + "=" * 50)
    print("[3/5] Training AUDIO model (LSTM)...")
    print("=" * 50)
    
    audio_model = build_audio_model(input_shape=(100, 40), num_classes=NUM_EMOTIONS)
    print(f"Audio model parameters: {audio_model.count_params():,}")
    
    audio_model.fit(
        X_audio_train, y_train_onehot,
        validation_data=(X_audio_val, y_val_onehot),
        epochs=50,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    audio_acc = audio_model.evaluate(X_audio_val, y_val_onehot, verbose=0)[1]
    print(f"\n>>> Audio Model Accuracy: {audio_acc*100:.2f}%")
    
    audio_model.save('models/audio_emotion_lstm.h5')
    print("Saved: models/audio_emotion_lstm.h5")
    
    # ==================== FUSION ====================
    print("\n" + "=" * 50)
    print("[4/5] Training FUSION model...")
    print("=" * 50)
    
    # Get predictions from both models
    video_preds_train = video_model.predict(X_video_train, verbose=0)
    video_preds_val = video_model.predict(X_video_val, verbose=0)
    
    audio_preds_train = audio_model.predict(X_audio_train, verbose=0)
    audio_preds_val = audio_model.predict(X_audio_val, verbose=0)
    
    # Train fusion model
    fusion_model = build_fusion_model(num_classes=NUM_EMOTIONS)
    
    fusion_model.fit(
        [video_preds_train, audio_preds_train], y_train_onehot,
        validation_data=([video_preds_val, audio_preds_val], y_val_onehot),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    fusion_acc = fusion_model.evaluate([video_preds_val, audio_preds_val], y_val_onehot, verbose=0)[1]
    print(f"\n>>> Fusion Model Accuracy: {fusion_acc*100:.2f}%")
    
    fusion_model.save('models/fusion_emotion.h5')
    print("Saved: models/fusion_emotion.h5")
    
    # ==================== SIMPLE AVERAGE FUSION ====================
    print("\n" + "=" * 50)
    print("[5/5] Testing simple average fusion...")
    print("=" * 50)
    
    # Average fusion
    avg_preds = (video_preds_val + audio_preds_val) / 2
    avg_pred_labels = np.argmax(avg_preds, axis=1)
    avg_acc = np.mean(avg_pred_labels == y_val)
    print(f"\n>>> Average Fusion Accuracy: {avg_acc*100:.2f}%")
    
    # Weighted fusion (video 0.4, audio 0.6 - audio is usually more reliable)
    weighted_preds = 0.4 * video_preds_val + 0.6 * audio_preds_val
    weighted_pred_labels = np.argmax(weighted_preds, axis=1)
    weighted_acc = np.mean(weighted_pred_labels == y_val)
    print(f">>> Weighted Fusion (0.4v + 0.6a) Accuracy: {weighted_acc*100:.2f}%")
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Accuracy':>10}")
    print("-" * 42)
    print(f"{'Video (MobileNetV2)':<30} {video_acc*100:>9.2f}%")
    print(f"{'Audio (LSTM)':<30} {audio_acc*100:>9.2f}%")
    print(f"{'Learned Fusion':<30} {fusion_acc*100:>9.2f}%")
    print(f"{'Average Fusion':<30} {avg_acc*100:>9.2f}%")
    print(f"{'Weighted Fusion (0.4v+0.6a)':<30} {weighted_acc*100:>9.2f}%")
    print("-" * 42)
    
    best_acc = max(video_acc, audio_acc, fusion_acc, avg_acc, weighted_acc)
    print(f"\n>>> BEST ACCURACY: {best_acc*100:.2f}%")
    
    # Per-class analysis
    print("\nPer-class accuracy (best fusion):")
    best_preds = weighted_pred_labels if weighted_acc >= avg_acc else avg_pred_labels
    for i in range(NUM_EMOTIONS):
        mask = y_val == i
        if np.sum(mask) > 0:
            class_acc = np.mean(best_preds[mask] == y_val[mask])
            print(f"  {EMOTIONS[i]}: {class_acc*100:.1f}% ({np.sum(mask)} samples)")
    
    return video_model, audio_model, fusion_model


if __name__ == '__main__':
    video_model, audio_model, fusion_model = train_hybrid_system()
