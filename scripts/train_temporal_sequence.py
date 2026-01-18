"""
Temporal Sequence Video Emotion Recognition
Uses sequences of frames with CNN + LSTM for better video accuracy
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import gc

# Disable GPU to avoid memory issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ===================== CONFIGURATION =====================
DATA_DIR = "data"
MODELS_DIR = "models"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Sequence parameters
SEQUENCE_LENGTH = 10  # Number of frames per sequence
FRAME_SIZE = 64       # Smaller frames for sequences
BATCH_SIZE = 8

# RAVDESS emotion mapping (simplified to 5 emotions)
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'neutral',   # calm -> neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'fearful',   # disgust -> fearful (similar arousal)
    '08': 'happy',     # surprised -> happy (positive valence)
}

EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful']
NUM_EMOTIONS = len(EMOTIONS)

# ===================== DATA LOADING =====================

def extract_face_sequence(video_path, face_cascade, seq_length=10, frame_size=64):
    """
    Extract sequence of face frames from video
    Returns array of shape (seq_length, frame_size, frame_size, 1)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < seq_length:
        cap.release()
        return None
    
    # Sample frames evenly across video
    frame_indices = np.linspace(0, total_frames - 1, seq_length, dtype=int)
    
    faces = []
    last_valid_face = None
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            if last_valid_face is not None:
                faces.append(last_valid_face)
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        detected = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(detected) > 0:
            # Get largest face
            x, y, w, h = max(detected, key=lambda f: f[2] * f[3])
            
            # Add padding
            pad = int(0.1 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)
            
            face = gray[y1:y2, x1:x2]
            face = cv2.resize(face, (frame_size, frame_size))
            face = face.astype(np.float32) / 255.0
            
            last_valid_face = face
            faces.append(face)
        elif last_valid_face is not None:
            # Use last valid face if detection fails
            faces.append(last_valid_face)
    
    cap.release()
    
    # Need at least half the sequence
    if len(faces) < seq_length // 2:
        return None
    
    # Pad or truncate to exact sequence length
    while len(faces) < seq_length:
        faces.append(faces[-1] if faces else np.zeros((frame_size, frame_size)))
    
    faces = faces[:seq_length]
    
    # Stack and add channel dimension
    sequence = np.stack(faces, axis=0)
    sequence = np.expand_dims(sequence, axis=-1)  # (seq_len, h, w, 1)
    
    return sequence


def load_ravdess_sequences(data_dir, seq_length=10, frame_size=64):
    """
    Load RAVDESS videos as sequences of face frames
    """
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    sequences = []
    labels = []
    skipped = 0
    
    actor_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('Actor_')])
    
    print(f"Loading sequences from {len(actor_dirs)} actors...")
    
    for actor_dir in tqdm(actor_dirs):
        actor_path = os.path.join(data_dir, actor_dir)
        
        video_files = [f for f in os.listdir(actor_path) if f.endswith('.mp4')]
        
        for video_file in video_files:
            # Parse RAVDESS filename: XX-XX-EMOTION-XX-XX-XX-XX.mp4
            parts = video_file.split('-')
            if len(parts) < 3:
                continue
            
            emotion_code = parts[2]
            if emotion_code not in EMOTION_MAP:
                continue
            
            emotion = EMOTION_MAP[emotion_code]
            label = EMOTIONS.index(emotion)
            
            video_path = os.path.join(actor_path, video_file)
            sequence = extract_face_sequence(
                video_path, face_cascade, seq_length, frame_size
            )
            
            if sequence is not None:
                sequences.append(sequence)
                labels.append(label)
            else:
                skipped += 1
    
    print(f"Loaded {len(sequences)} sequences, skipped {skipped}")
    
    return np.array(sequences), np.array(labels)


def augment_sequence(sequence):
    """
    Data augmentation for sequence
    """
    augmented = sequence.copy()
    
    # Random horizontal flip (entire sequence)
    if np.random.random() > 0.5:
        augmented = augmented[:, :, ::-1, :]
    
    # Random brightness adjustment
    brightness = np.random.uniform(0.8, 1.2)
    augmented = np.clip(augmented * brightness, 0, 1)
    
    # Random contrast adjustment
    contrast = np.random.uniform(0.8, 1.2)
    mean = np.mean(augmented)
    augmented = np.clip((augmented - mean) * contrast + mean, 0, 1)
    
    return augmented


class SequenceDataGenerator(keras.utils.Sequence):
    """
    Data generator with augmentation for sequences
    """
    def __init__(self, sequences, labels, batch_size=8, augment=True, shuffle=True):
        self.sequences = sequences
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(sequences))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.sequences) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_sequences = []
        batch_labels = []
        
        for i in batch_indices:
            seq = self.sequences[i]
            if self.augment:
                seq = augment_sequence(seq)
            batch_sequences.append(seq)
            batch_labels.append(self.labels[i])
        
        X = np.array(batch_sequences)
        y = keras.utils.to_categorical(batch_labels, num_classes=NUM_EMOTIONS)
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ===================== MODEL ARCHITECTURE =====================

def build_temporal_cnn_lstm(seq_length=10, frame_size=64, num_classes=5):
    """
    CNN + LSTM model for temporal video emotion recognition
    
    Architecture:
    1. TimeDistributed CNN extracts features from each frame
    2. LSTM processes the sequence of features
    3. Dense layers for classification
    """
    inputs = layers.Input(shape=(seq_length, frame_size, frame_size, 1))
    
    # TimeDistributed CNN for per-frame feature extraction
    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), activation='relu', padding='same')
    )(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(0.25))(x)
    
    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), activation='relu', padding='same')
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(0.25))(x)
    
    x = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), activation='relu', padding='same')
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(0.25))(x)
    
    # Flatten spatial dimensions for each timestep
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)
    
    # Bidirectional LSTM for temporal modeling
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    
    # Classification head
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model


def build_conv3d_model(seq_length=10, frame_size=64, num_classes=5):
    """
    3D CNN model that processes spatiotemporal features together
    """
    inputs = layers.Input(shape=(seq_length, frame_size, frame_size, 1))
    
    # 3D Convolutions for spatiotemporal features
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((1, 2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model


# ===================== TRAINING =====================

def train_model(model, train_gen, val_gen, model_name, epochs=50):
    """
    Train model with callbacks
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'{MODELS_DIR}/{model_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def main():
    print("=" * 60)
    print("TEMPORAL SEQUENCE VIDEO EMOTION RECOGNITION")
    print("Using sequences of frames with CNN + LSTM")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading video sequences...")
    X, y = load_ravdess_sequences(DATA_DIR, SEQUENCE_LENGTH, FRAME_SIZE)
    
    if len(X) == 0:
        print("ERROR: No data loaded!")
        return
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Class distribution
    print("\nClass distribution:")
    for i, emotion in enumerate(EMOTIONS):
        count = np.sum(y == i)
        print(f"  {emotion}: {count}")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTraining: {len(X_train)}, Validation: {len(X_val)}")
    
    # Create data generators
    train_gen = SequenceDataGenerator(
        X_train, y_train, batch_size=BATCH_SIZE, augment=True
    )
    val_gen = SequenceDataGenerator(
        X_val, y_val, batch_size=BATCH_SIZE, augment=False, shuffle=False
    )
    
    # ==================== MODEL 1: CNN + LSTM ====================
    print("\n" + "=" * 60)
    print("[2/4] Training CNN + Bidirectional LSTM model...")
    print("=" * 60)
    
    gc.collect()
    
    cnn_lstm_model = build_temporal_cnn_lstm(
        seq_length=SEQUENCE_LENGTH,
        frame_size=FRAME_SIZE,
        num_classes=NUM_EMOTIONS
    )
    print(f"CNN-LSTM parameters: {cnn_lstm_model.count_params():,}")
    cnn_lstm_model.summary()
    
    # Override fit to use class weights
    cnn_lstm_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Manual training loop with class weights
    cnn_lstm_history = cnn_lstm_model.fit(
        X_train, keras.utils.to_categorical(y_train, NUM_EMOTIONS),
        validation_data=(X_val, keras.utils.to_categorical(y_val, NUM_EMOTIONS)),
        epochs=50,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    cnn_lstm_acc = cnn_lstm_model.evaluate(
        X_val, keras.utils.to_categorical(y_val, NUM_EMOTIONS), verbose=0
    )[1]
    print(f"\n>>> CNN-LSTM Accuracy: {cnn_lstm_acc * 100:.2f}%")
    
    cnn_lstm_model.save(f'{MODELS_DIR}/temporal_cnn_lstm.h5')
    print(f"Saved: {MODELS_DIR}/temporal_cnn_lstm.h5")
    
    # ==================== MODEL 2: 3D CNN ====================
    print("\n" + "=" * 60)
    print("[3/4] Training 3D CNN model...")
    print("=" * 60)
    
    gc.collect()
    
    conv3d_model = build_conv3d_model(
        seq_length=SEQUENCE_LENGTH,
        frame_size=FRAME_SIZE,
        num_classes=NUM_EMOTIONS
    )
    print(f"3D CNN parameters: {conv3d_model.count_params():,}")
    conv3d_model.summary()
    
    conv3d_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    conv3d_history = conv3d_model.fit(
        X_train, keras.utils.to_categorical(y_train, NUM_EMOTIONS),
        validation_data=(X_val, keras.utils.to_categorical(y_val, NUM_EMOTIONS)),
        epochs=50,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    conv3d_acc = conv3d_model.evaluate(
        X_val, keras.utils.to_categorical(y_val, NUM_EMOTIONS), verbose=0
    )[1]
    print(f"\n>>> 3D CNN Accuracy: {conv3d_acc * 100:.2f}%")
    
    conv3d_model.save(f'{MODELS_DIR}/temporal_conv3d.h5')
    print(f"Saved: {MODELS_DIR}/temporal_conv3d.h5")
    
    # ==================== ENSEMBLE ====================
    print("\n" + "=" * 60)
    print("[4/4] Ensemble evaluation...")
    print("=" * 60)
    
    # Get predictions from both models
    pred_cnn_lstm = cnn_lstm_model.predict(X_val, verbose=0)
    pred_conv3d = conv3d_model.predict(X_val, verbose=0)
    
    # Average ensemble
    pred_ensemble = (pred_cnn_lstm + pred_conv3d) / 2
    ensemble_pred = np.argmax(pred_ensemble, axis=1)
    ensemble_acc = np.mean(ensemble_pred == y_val)
    
    print(f">>> Ensemble Accuracy: {ensemble_acc * 100:.2f}%")
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    
    results = [
        ("CNN + BiLSTM", cnn_lstm_acc),
        ("3D CNN", conv3d_acc),
        ("Ensemble", ensemble_acc)
    ]
    
    print(f"\n{'Model':<25} {'Accuracy':>10}")
    print("-" * 35)
    for name, acc in results:
        print(f"{name:<25} {acc * 100:>9.2f}%")
    
    best_model = max(results, key=lambda x: x[1])
    print("-" * 35)
    print(f"\n>>> BEST MODEL: {best_model[0]} with {best_model[1] * 100:.2f}%")
    
    # Per-class accuracy for best model
    if best_model[0] == "CNN + BiLSTM":
        best_pred = np.argmax(pred_cnn_lstm, axis=1)
    elif best_model[0] == "3D CNN":
        best_pred = np.argmax(pred_conv3d, axis=1)
    else:
        best_pred = ensemble_pred
    
    print("\nPer-class accuracy (best model):")
    for i, emotion in enumerate(EMOTIONS):
        mask = y_val == i
        if np.sum(mask) > 0:
            class_acc = np.mean(best_pred[mask] == y_val[mask])
            print(f"  {emotion}: {class_acc * 100:.1f}% ({np.sum(mask)} samples)")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, best_pred)
    print(f"{'':>10}", end='')
    for e in EMOTIONS:
        print(f"{e[:4]:>6}", end='')
    print()
    for i, emotion in enumerate(EMOTIONS):
        print(f"{emotion[:8]:>10}", end='')
        for j in range(NUM_EMOTIONS):
            print(f"{cm[i, j]:>6}", end='')
        print()


if __name__ == "__main__":
    main()
