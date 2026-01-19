import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

class TrainingLogger(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch in [0, 9, 24, 49]:
            print(f"Starting epoch {epoch+1}")
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch in [9, 24, 49]:
            print(f"Epoch {epoch+1} complete - loss: {logs['loss']:.4f}, acc: {logs['accuracy']:.4f}, val_acc: {logs['val_accuracy']:.4f}")

# Paths
FEATURE_DIR = os.path.join(os.path.dirname(__file__), '../data/audio_features')
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']  # RAVDESS classes

def load_audio_data():
    """Load MFCC features and labels."""
    print("FEATURE_DIR:", FEATURE_DIR, flush=True)
    features = []
    labels = []

    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('.npy')]
    print(f"Loading {len(files)} audio features...", flush=True)
    for i, file in enumerate(files):
        if i % 500 == 0:
            print(f"Loaded {i}/{len(files)} files...", flush=True)
        mfcc = np.load(os.path.join(FEATURE_DIR, file))
        features.append(mfcc)

        # Extract emotion from filename (RAVDESS format: 03-01-01-01-01-01-01.wav -> emotion is 3rd part)
        parts = file.split('-')
        emotion_code = parts[2]
        emotion_map = {'01': 0, '02': 0, '03': 1, '04': 2, '05': 3, '06': 4, '07': 5, '08': 6}
        emotion_idx = emotion_map.get(emotion_code, 0)  # default to 0 if unknown
        labels.append(emotion_idx)

    return np.array(features), np.array(labels)

def build_audio_model():
    """Build 2D CNN for audio."""
    model = keras.Sequential([
        layers.Input(shape=(300, 13, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_audio_model():
    """Train the audio model."""
    print("Starting audio model training...", flush=True)
    X, y = load_audio_data()
    print(f"Data loaded: X.shape={X.shape}, y.shape={y.shape}", flush=True)
    y_cat = keras.utils.to_categorical(y, num_classes=7)

    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

    model = build_audio_model()

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights))

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        TrainingLogger()
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print(f"Audio training complete. Final train acc: {history.history['accuracy'][-1]:.4f}, val acc: {history.history['val_accuracy'][-1]:.4f}", flush=True)
    model.save(os.path.join(os.path.dirname(__file__), '../models/audio_emotion_model.h5'))
    print("Audio model trained and saved.", flush=True)

if __name__ == "__main__":
    train_audio_model()