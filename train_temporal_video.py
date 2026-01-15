"""
Temporal Video Emotion Detection Model
Uses pre-trained FER model as feature extractor + LSTM for temporal modeling
Processes RAVDESS video frames with face detection
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# RAVDESS emotion mapping (3rd position in filename)
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm', 
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Map to FER-compatible emotions (8 classes)
EMOTION_TO_IDX = {
    'neutral': 0,
    'calm': 0,       # Map calm to neutral
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fearful': 4,
    'disgust': 5,
    'surprised': 6
}

NUM_EMOTIONS = 7  # 7 distinct emotions after mapping


class FaceDetector:
    """Face detection using Haar Cascade"""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            # Try local path
            local_path = 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(local_path)
    
    def detect_face(self, frame, target_size=(48, 48)):
        """
        Detect and extract face, resize to target size
        Returns grayscale face image
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest
        
        # Add padding
        padding = int(w * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Extract and resize face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        
        return face


class VideoFaceExtractor:
    """Extract face sequences from RAVDESS videos"""
    
    def __init__(self, n_frames=16, target_size=(48, 48)):
        self.face_detector = FaceDetector()
        self.n_frames = n_frames
        self.target_size = target_size
    
    def extract_from_video(self, video_path):
        """
        Extract sequence of faces from video
        Returns: numpy array (n_frames, height, width, 1) or None
        """
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            cap.release()
            return None
        
        # Sample frames evenly
        frame_indices = np.linspace(0, frame_count - 1, self.n_frames, dtype=int)
        
        faces = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            face = self.face_detector.detect_face(frame, self.target_size)
            
            if face is not None:
                faces.append(face)
            else:
                # Use previous face or skip
                if len(faces) > 0:
                    faces.append(faces[-1])
        
        cap.release()
        
        if len(faces) < self.n_frames // 2:
            return None
        
        # Pad if needed
        while len(faces) < self.n_frames:
            faces.append(faces[-1] if faces else np.zeros(self.target_size))
        
        faces = np.array(faces[:self.n_frames], dtype=np.float32)
        faces = faces / 255.0  # Normalize
        faces = np.expand_dims(faces, axis=-1)  # Add channel dim
        
        return faces


class TemporalEmotionModel:
    """
    2D CNN + LSTM for temporal emotion detection from video
    Uses FER-trained CNN as base, adds LSTM for temporal modeling
    """
    
    def __init__(self, num_emotions=7, n_frames=16, input_shape=(48, 48, 1)):
        self.num_emotions = num_emotions
        self.n_frames = n_frames
        self.input_shape = input_shape
        self.model = None
    
    def build_cnn_lstm_model(self, pretrained_fer_path=None):
        """
        Build 2D CNN + LSTM model
        Option to load pre-trained FER weights for CNN backbone
        """
        # Input: sequence of face images
        inputs = layers.Input(shape=(self.n_frames,) + self.input_shape)
        
        # TimeDistributed CNN blocks (applied to each frame)
        # Block 1
        x = layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        )(inputs)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Dropout(0.25))(x)
        
        # Block 2
        x = layers.TimeDistributed(
            layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Dropout(0.25))(x)
        
        # Block 3
        x = layers.TimeDistributed(
            layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Dropout(0.25))(x)
        
        # Global pooling per frame
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        # Shape: (batch, n_frames, 128)
        
        # LSTM for temporal modeling
        x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
        x = layers.LSTM(64, dropout=0.3)(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output
        outputs = layers.Dense(self.num_emotions, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_with_fer_backbone(self, fer_model_path='models/fer_base_model.h5'):
        """
        Build model using pre-trained FER CNN as feature extractor
        Then add LSTM layers on top
        """
        # Load FER model
        if not os.path.exists(fer_model_path):
            print(f"FER model not found at {fer_model_path}, using fresh CNN")
            return self.build_cnn_lstm_model()
        
        fer_model = keras.models.load_model(fer_model_path)
        
        # Create feature extractor (remove classification head)
        # Get features before the final dense layers
        fer_features = models.Model(
            inputs=fer_model.input,
            outputs=fer_model.layers[-5].output  # 512-dim after GlobalAveragePooling
        )
        
        # Freeze FER backbone initially
        for layer in fer_features.layers:
            layer.trainable = False
        
        # Build temporal model
        inputs = layers.Input(shape=(self.n_frames,) + self.input_shape)
        
        # Apply FER feature extractor to each frame
        x = layers.TimeDistributed(fer_features)(inputs)
        # Shape: (batch, n_frames, 512)
        
        # LSTM layers
        x = layers.LSTM(256, return_sequences=True, dropout=0.3)(x)
        x = layers.LSTM(128, dropout=0.3)(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(self.num_emotions, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        """Train the model"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'models/temporal_video_emotion.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save(self, path='models/temporal_video_emotion.h5'):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path='models/temporal_video_emotion.h5'):
        """Load model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")


def load_ravdess_videos(data_dir='data', n_frames=16):
    """
    Load RAVDESS video data with face extraction
    Returns: X (face sequences), y (emotion labels)
    """
    extractor = VideoFaceExtractor(n_frames=n_frames)
    
    X = []
    y = []
    
    # Get all Actor directories
    data_path = Path(data_dir)
    actor_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('Actor')])
    
    print(f"Found {len(actor_dirs)} actor directories")
    
    for actor_dir in tqdm(actor_dirs, desc="Processing actors"):
        # Get video files (mp4)
        video_files = list(actor_dir.glob('*.mp4'))
        
        for video_file in video_files:
            # Parse filename to get emotion
            # Format: XX-XX-EE-XX-XX-XX-XX.mp4 (EE = emotion)
            parts = video_file.stem.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                
                if emotion_code in RAVDESS_EMOTIONS:
                    emotion_name = RAVDESS_EMOTIONS[emotion_code]
                    
                    if emotion_name in EMOTION_TO_IDX:
                        # Extract faces
                        faces = extractor.extract_from_video(video_file)
                        
                        if faces is not None:
                            X.append(faces)
                            y.append(EMOTION_TO_IDX[emotion_name])
    
    X = np.array(X)
    y = np.array(y)
    
    # One-hot encode labels
    y_onehot = keras.utils.to_categorical(y, num_classes=NUM_EMOTIONS)
    
    print(f"Loaded {len(X)} video sequences")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y_onehot.shape}")
    
    return X, y_onehot


def plot_training_history(history, save_path='plots/temporal_video_training.png'):
    """Plot training history"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Temporal Video Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Temporal Video Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("Training Temporal Video Emotion Detection Model")
    print("=" * 60)
    
    # Configuration
    N_FRAMES = 16
    BATCH_SIZE = 8
    EPOCHS = 50
    
    # Load data
    print("\nLoading RAVDESS video data...")
    X, y = load_ravdess_videos(data_dir='data', n_frames=N_FRAMES)
    
    if len(X) == 0:
        print("No data loaded! Check your data directory.")
        exit(1)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Build model
    print("\nBuilding Temporal Emotion Model...")
    model = TemporalEmotionModel(
        num_emotions=NUM_EMOTIONS,
        n_frames=N_FRAMES,
        input_shape=(48, 48, 1)
    )
    
    # Build fresh CNN-LSTM model (FER backbone loading has issues with corrupted saves)
    print("Building fresh CNN-LSTM model for temporal emotion detection")
    model.build_cnn_lstm_model()
    
    print("\nModel Architecture:")
    model.model.summary()
    
    # Train
    print("\nStarting training...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Plot results
    plot_training_history(history)
    
    # Final evaluation
    loss, accuracy = model.model.evaluate(X_val, y_val)
    print(f"\nFinal Validation Accuracy: {accuracy*100:.2f}%")
    print(f"Final Validation Loss: {loss:.4f}")
    
    # Save model
    model.save('models/temporal_video_emotion.h5')
    print("\nTemporal Video Emotion Model training complete!")
