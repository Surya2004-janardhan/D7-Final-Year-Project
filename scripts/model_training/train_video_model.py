import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

class TrainingLogger(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch in [0, 9, 19, 29]:
            print(f"Starting epoch {epoch+1}")
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch in [9, 19, 29]:
            print(f"Epoch {epoch+1} complete - loss: {logs['loss']:.4f}, acc: {logs['accuracy']:.4f}, val_acc: {logs['val_accuracy']:.4f}")

# Paths
FRAME_DIR = os.path.join(os.path.dirname(__file__), '../data/video_frames')
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

def load_video_data():
    """Load frame features and labels."""
    features = []
    labels = []

    # Pretrained extractor
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
    base_model.trainable = False

    files = [f for f in os.listdir(FRAME_DIR) if f.endswith('.npy')]
    print(f"Loading {len(files)} video features...")
    for file in tqdm(files):
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
        emotion_code = parts[2]
        emotion_map = {'01': 0, '02': 0, '03': 1, '04': 2, '05': 3, '06': 4, '07': 5, '08': 6}
        emotion_idx = emotion_map.get(emotion_code, 0)
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
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        TrainingLogger()
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print(f"Video training complete. Final train acc: {history.history['accuracy'][-1]:.4f}, val acc: {history.history['val_accuracy'][-1]:.4f}")
    model.save(os.path.join(os.path.dirname(__file__), '../models/video_emotion_model.h5'))
    print("Video model trained and saved.")

if __name__ == "__main__":
    train_video_model()