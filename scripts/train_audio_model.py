import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Paths
FEATURE_DIR = 'data/audio_features'
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']  # RAVDESS classes

def load_audio_data():
    """Load MFCC features and labels."""
    features = []
    labels = []

    for file in os.listdir(FEATURE_DIR):
        if file.endswith('.npy'):
            mfcc = np.load(os.path.join(FEATURE_DIR, file))
            features.append(mfcc)

            # Extract emotion from filename (RAVDESS format: 03-01-01-01-01-01-01.wav -> emotion is 3rd part)
            parts = file.split('-')
            emotion_idx = int(parts[2]) - 1  # 01=neutral, 02=calm->neutral, etc.
            if emotion_idx == 1:  # calm -> neutral
                emotion_idx = 0
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
    X, y = load_audio_data()
    y_cat = keras.utils.to_categorical(y, num_classes=7)

    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

    model = build_audio_model()

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights))

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save('models/audio_emotion_model.h5')
    print("Audio model trained and saved.")

if __name__ == "__main__":
    train_audio_model()