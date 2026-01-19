"""
Improved Temporal Video Model Training
Addresses class imbalance and model bias issues
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Emotion mappings
RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

EMOTION_TO_IDX = {
    'neutral': 0, 'calm': 0, 'happy': 1, 'sad': 2,
    'angry': 3, 'fearful': 4, 'disgust': 5, 'surprised': 6
}

EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
NUM_EMOTIONS = 7


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect(self, frame, target_size=(48, 48)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
        
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(w * 0.1)
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)
        
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        return face


def extract_faces_from_video(video_path, n_frames=16):
    """Extract face sequence from video"""
    detector = FaceDetector()
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        cap.release()
        return None
    
    indices = np.linspace(0, frame_count - 1, n_frames, dtype=int)
    faces = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        face = detector.detect(frame, (48, 48))
        if face is not None:
            faces.append(face)
        elif len(faces) > 0:
            faces.append(faces[-1])
    
    cap.release()
    
    if len(faces) < n_frames // 2:
        return None
    
    while len(faces) < n_frames:
        faces.append(faces[-1] if faces else np.zeros((48, 48)))
    
    faces = np.array(faces[:n_frames], dtype=np.float32) / 255.0
    faces = np.expand_dims(faces, axis=-1)
    return faces


def load_ravdess_data(data_dir='data', n_frames=16):
    """Load all RAVDESS video data"""
    X, y = [], []
    
    data_path = Path(data_dir)
    actor_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('Actor')])
    
    print(f"Loading from {len(actor_dirs)} actors...")
    
    for actor_dir in tqdm(actor_dirs):
        for video_file in actor_dir.glob('*.mp4'):
            parts = video_file.stem.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in RAVDESS_EMOTIONS:
                    emotion_name = RAVDESS_EMOTIONS[emotion_code]
                    if emotion_name in EMOTION_TO_IDX:
                        faces = extract_faces_from_video(video_file, n_frames)
                        if faces is not None:
                            X.append(faces)
                            y.append(EMOTION_TO_IDX[emotion_name])
    
    return np.array(X), np.array(y)


def build_improved_model(n_frames=16, input_shape=(48, 48, 1), num_emotions=7):
    """
    Simpler architecture to avoid memory issues
    """
    inputs = layers.Input(shape=(n_frames,) + input_shape)
    
    # Simple CNN per frame
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(0.25))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(0.25))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    # Shape: (batch, n_frames, 64)
    
    # Simple LSTM
    x = layers.LSTM(64, return_sequences=True, dropout=0.3)(x)
    x = layers.LSTM(32, dropout=0.3)(x)
    
    # Classification head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_emotions, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def train_with_class_weights():
    """Train with class weights to handle imbalance"""
    
    print("=" * 70)
    print("IMPROVED TEMPORAL VIDEO MODEL TRAINING")
    print("=" * 70)
    
    N_FRAMES = 16
    BATCH_SIZE = 8
    EPOCHS = 80
    
    # Load data
    print("\nLoading data...")
    X, y = load_ravdess_data(n_frames=N_FRAMES)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"X shape: {X.shape}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"  {EMOTIONS[u]}: {c}")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"\nClass weights: {class_weight_dict}")
    
    # Split data with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # One-hot encode
    y_train_onehot = keras.utils.to_categorical(y_train, NUM_EMOTIONS)
    y_val_onehot = keras.utils.to_categorical(y_val, NUM_EMOTIONS)
    
    print(f"\nTraining: {len(X_train)}, Validation: {len(X_val)}")
    
    # Build model
    print("\nBuilding improved model...")
    model = build_improved_model(n_frames=N_FRAMES)
    
    # Compile with label smoothing
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
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
            'models/temporal_video_emotion_v2.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train with class weights
    print("\nTraining with class weights...")
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating...")
    loss, acc = model.evaluate(X_val, y_val_onehot)
    print(f"\nFinal Validation Accuracy: {acc*100:.2f}%")
    
    # Per-class accuracy
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\nPer-class accuracy:")
    for i in range(NUM_EMOTIONS):
        mask = y_val == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_val[mask])
            print(f"  {EMOTIONS[i]}: {class_acc*100:.1f}%")
    
    # Prediction distribution
    print("\nPrediction distribution:")
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    for u, c in zip(unique_pred, counts_pred):
        print(f"  {EMOTIONS[u]}: {c} ({c/len(y_pred)*100:.1f}%)")
    
    # Save final model
    model.save('models/temporal_video_emotion.h5')
    print("\nModel saved to models/temporal_video_emotion.h5")
    
    return model, history


if __name__ == '__main__':
    model, history = train_with_class_weights()
