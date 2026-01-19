import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Paths
FRAME_DIR = 'data/video_frames'
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

def load_video_data():
    """Load frame features and labels."""
    features = []
    labels = []

    # Pretrained extractor
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
    base_model.trainable = False

    for file in os.listdir(FRAME_DIR):
        if file.endswith('.npy'):
            frames = np.load(os.path.join(FRAME_DIR, file))  # (16, 112, 112, 3)

            # Extract features for each frame
            frame_features = []
            for frame in frames:
                frame = np.expand_dims(frame, axis=0)
                feat = base_model(frame)
                feat = keras.layers.GlobalAveragePooling2D()(feat)
                frame_features.append(feat.numpy().flatten())

            # Average temporal features
            video_feat = np.mean(frame_features, axis=0)  # (1280,)
            features.append(video_feat)

            # Extract emotion from filename
            parts = file.split('-')
            emotion_idx = int(parts[2]) - 1
            if emotion_idx == 1:  # calm -> neutral
                emotion_idx = 0
            labels.append(emotion_idx)

    return np.array(features), np.array(labels)

def build_video_model():
    """Build video model with frozen MobileNetV2."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
    base_model.trainable = False

    model = keras.Sequential([
        layers.Input(shape=(1280,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_video_model():
    """Train the video model."""
    X, y = load_video_data()
    y_cat = keras.utils.to_categorical(y, num_classes=7)

    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

    model = build_video_model()

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights))

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save('models/video_emotion_model.h5')
    print("Video model trained and saved.")

if __name__ == "__main__":
    train_video_model()