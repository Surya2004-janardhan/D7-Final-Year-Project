"""
==========================================================================
COMPLETE MULTIMODAL EMOTION RECOGNITION PIPELINE
==========================================================================

Phase 1: Train CNN backbone on FER2013 (large-scale static facial dataset)
Phase 2: Fine-tune + add temporal LSTM using RAVDESS video frames
Phase 3: Train audio BiLSTM + Multimodal fusion at embedding level

This approach:
- Prevents overfitting by pretraining on large dataset
- Captures temporal dynamics via TimeDistributed CNN + BiLSTM
- Leverages complementary audio information
- Fuses modalities at embedding level for best accuracy
==========================================================================
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import librosa
import gc
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================
# Paths
FER_DATA_DIR = "fer-data"
RAVDESS_DATA_DIR = "data"
MODELS_DIR = "models"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Image parameters
IMG_SIZE = 48  # FER2013 standard size
RAVDESS_FACE_SIZE = 48  # Same size for transfer

# Sequence parameters for temporal model
SEQUENCE_LENGTH = 15  # Frames per video
BATCH_SIZE = 16

# Audio parameters
AUDIO_SR = 22050
N_MFCC = 40
AUDIO_MAX_LEN = 100  # Time steps

# Emotion mappings
FER_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_FER_EMOTIONS = 7

# RAVDESS to FER emotion mapping (align with FER emotions)
RAVDESS_TO_FER = {
    '01': 'neutral',   # neutral
    '02': 'neutral',   # calm -> neutral
    '03': 'happy',     # happy
    '04': 'sad',       # sad
    '05': 'angry',     # angry
    '06': 'fear',      # fearful -> fear
    '07': 'disgust',   # disgust
    '08': 'surprise',  # surprised -> surprise
}

os.makedirs(MODELS_DIR, exist_ok=True)


# ==========================================================================
# PHASE 1: CNN BACKBONE TRAINING ON FER2013
# ==========================================================================

def build_fer_cnn_backbone(input_shape=(48, 48, 1), num_classes=7):
    """
    Build a robust CNN for facial emotion recognition.
    This becomes the pretrained backbone for video temporal model.
    
    Architecture inspired by VGGNet with BatchNorm and aggressive regularization.
    """
    inputs = layers.Input(shape=input_shape, name='image_input')
    
    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 4
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Feature embedding layer (this will be used for fusion later)
    embedding = layers.Dense(256, activation='relu', name='fer_embedding')(x)
    embedding = layers.Dropout(0.5)(embedding)
    
    # Classification head
    outputs = layers.Dense(num_classes, activation='softmax', name='fer_output')(embedding)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='FER_CNN_Backbone')
    
    return model


def create_fer_feature_extractor(trained_model):
    """
    Extract the feature embedding part from trained FER model.
    This will be wrapped in TimeDistributed for temporal modeling.
    """
    # Get the embedding layer output
    embedding_layer = trained_model.get_layer('fer_embedding')
    
    # Create feature extractor model
    feature_extractor = models.Model(
        inputs=trained_model.input,
        outputs=embedding_layer.output,
        name='FER_Feature_Extractor'
    )
    
    return feature_extractor


def load_fer_data(data_dir):
    """
    Load FER2013-style data from directory structure.
    Expects: data_dir/train/emotion_name/*.jpg
             data_dir/test/emotion_name/*.jpg
    """
    print("Loading FER2013 data...")
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Check available directories
    train_dir = os.path.join(data_dir, 'train_split')
    val_dir = os.path.join(data_dir, 'val_split')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir):
        train_dir = os.path.join(data_dir, 'train')
    
    def load_from_dir(directory, desc="Loading"):
        images, labels = [], []
        for emotion in FER_EMOTIONS:
            emotion_dir = os.path.join(directory, emotion)
            # Handle spelling variation
            if not os.path.exists(emotion_dir) and emotion == 'surprise':
                emotion_dir = os.path.join(directory, 'suprise')
            
            if not os.path.exists(emotion_dir):
                print(f"  Warning: {emotion} directory not found in {directory}")
                continue
            
            label = FER_EMOTIONS.index(emotion)
            files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_file in files:
                img_path = os.path.join(emotion_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img)
                    labels.append(label)
        
        return np.array(images), np.array(labels)
    
    # Load training data
    if os.path.exists(train_dir):
        X_train, y_train = load_from_dir(train_dir, "Training")
        print(f"  Training samples: {len(X_train)}")
    
    # Load validation data
    if os.path.exists(val_dir):
        X_val, y_val = load_from_dir(val_dir, "Validation")
        print(f"  Validation samples: {len(X_val)}")
    else:
        X_val, y_val = np.array([]), np.array([])
    
    # Load test data
    if os.path.exists(test_dir):
        X_test, y_test = load_from_dir(test_dir, "Test")
        print(f"  Test samples: {len(X_test)}")
    
    # Combine train and validation if validation exists
    if len(X_val) > 0 and len(X_train) > 0:
        X_train = np.concatenate([X_train, X_val])
        y_train = np.concatenate([y_train, y_val])
    elif len(X_train) == 0 and len(X_test) > 0:
        # If no train data, split test data
        X_train, X_test, y_train, y_test = train_test_split(
            X_test, y_test, test_size=0.2, stratify=y_test, random_state=42
        )
    
    # Normalize and reshape
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    return X_train, y_train, X_test, y_test


def train_phase1_fer_backbone():
    """
    PHASE 1: Train CNN backbone on FER2013 dataset.
    This learns robust facial features from a large, diverse dataset.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING CNN BACKBONE ON FER2013")
    print("Learning robust facial features from large-scale static dataset")
    print("=" * 70)
    
    # Load FER data
    X_train, y_train, X_test, y_test = load_fer_data(FER_DATA_DIR)
    
    if len(X_train) == 0:
        print("ERROR: No FER training data found!")
        return None
    
    print(f"\nTotal training samples: {len(X_train)}")
    print(f"Total test samples: {len(X_test)}")
    
    # Class distribution
    print("\nClass distribution (training):")
    for i, emotion in enumerate(FER_EMOTIONS):
        count = np.sum(y_train == i)
        print(f"  {emotion}: {count}")
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Convert labels to one-hot
    y_train_onehot = keras.utils.to_categorical(y_train, NUM_FER_EMOTIONS)
    y_test_onehot = keras.utils.to_categorical(y_test, NUM_FER_EMOTIONS)
    
    # Split training data for validation
    X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
        X_train, y_train_onehot, test_size=0.15, random_state=42
    )
    
    print(f"\nAfter split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    
    # Build model
    model = build_fer_cnn_backbone(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_FER_EMOTIONS)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel parameters: {model.count_params():,}")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
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
            f'{MODELS_DIR}/fer_backbone_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train with data augmentation
    print("\nTraining FER CNN backbone...")
    history = model.fit(
        datagen.flow(X_train, y_train_onehot, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val_onehot),
        epochs=50,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"\n>>> FER Backbone Test Accuracy: {test_acc * 100:.2f}%")
    
    # Save final model
    model.save(f'{MODELS_DIR}/fer_backbone_final.h5')
    print(f"Saved: {MODELS_DIR}/fer_backbone_final.h5")
    
    # Per-class accuracy
    print("\nPer-class test accuracy:")
    predictions = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test_onehot, axis=1)
    
    for i, emotion in enumerate(FER_EMOTIONS):
        mask = true_labels == i
        if np.sum(mask) > 0:
            acc = np.mean(pred_labels[mask] == true_labels[mask])
            print(f"  {emotion}: {acc * 100:.1f}% ({np.sum(mask)} samples)")
    
    return model


# ==========================================================================
# PHASE 2: TEMPORAL VIDEO MODEL USING RAVDESS
# ==========================================================================

def extract_video_sequences(data_dir, face_cascade, seq_length=15, img_size=48):
    """
    Extract face sequences from RAVDESS videos.
    Returns sequences of grayscale face crops.
    """
    sequences = []
    labels = []
    skipped = 0
    
    actor_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('Actor_')])
    
    print(f"Extracting sequences from {len(actor_dirs)} actors...")
    
    for actor_dir in tqdm(actor_dirs):
        actor_path = os.path.join(data_dir, actor_dir)
        
        video_files = [f for f in os.listdir(actor_path) if f.endswith('.mp4')]
        
        for video_file in video_files:
            parts = video_file.split('-')
            if len(parts) < 3:
                continue
            
            emotion_code = parts[2]
            if emotion_code not in RAVDESS_TO_FER:
                continue
            
            emotion_name = RAVDESS_TO_FER[emotion_code]
            if emotion_name not in FER_EMOTIONS:
                continue
            
            label = FER_EMOTIONS.index(emotion_name)
            
            video_path = os.path.join(actor_path, video_file)
            sequence = extract_face_sequence_from_video(
                video_path, face_cascade, seq_length, img_size
            )
            
            if sequence is not None:
                sequences.append(sequence)
                labels.append(label)
            else:
                skipped += 1
            
            # Clear memory periodically
            gc.collect()
        
        # Force garbage collection after each actor
        gc.collect()
    
    print(f"Extracted {len(sequences)} sequences, skipped {skipped}")
    
    return np.array(sequences, dtype=np.float32), np.array(labels)


def extract_face_sequence_from_video(video_path, face_cascade, seq_length, img_size):
    """
    Extract evenly-spaced face frames from a video.
    Returns shape: (seq_length, img_size, img_size, 1)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < seq_length:
        cap.release()
        return None
    
    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, seq_length, dtype=int)
    
    faces = []
    last_valid_face = None
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            if last_valid_face is not None:
                faces.append(last_valid_face.copy())
            continue
        
        # Convert to grayscale immediately to save memory
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        del frame  # Free memory
        
        detected = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(detected) > 0:
            x, y, w, h = max(detected, key=lambda f: f[2] * f[3])
            
            # Padding
            pad = int(0.15 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)
            
            face = gray[y1:y2, x1:x2]
            face = cv2.resize(face, (img_size, img_size))
            face = face.astype(np.float32) / 255.0
            
            last_valid_face = face
            faces.append(face)
        elif last_valid_face is not None:
            faces.append(last_valid_face.copy())
        
        del gray  # Free memory
    
    cap.release()
    
    if len(faces) < seq_length // 2:
        return None
    
    # Pad to exact length
    while len(faces) < seq_length:
        faces.append(faces[-1].copy() if faces else np.zeros((img_size, img_size), dtype=np.float32))
    
    faces = faces[:seq_length]
    sequence = np.stack(faces, axis=0)
    sequence = np.expand_dims(sequence, axis=-1)  # (seq, h, w, 1)
    
    return sequence.astype(np.float32)


def build_temporal_video_model(fer_feature_extractor, seq_length=15, num_classes=7):
    """
    Build temporal video model using pretrained FER backbone.
    
    Architecture:
    1. TimeDistributed pretrained CNN (frozen) for spatial features
    2. Bidirectional LSTM for temporal dynamics
    3. Dense layers for classification
    
    Returns model and the embedding output for fusion.
    """
    # Input: sequence of frames
    inputs = layers.Input(shape=(seq_length, IMG_SIZE, IMG_SIZE, 1), name='video_sequence_input')
    
    # TimeDistributed feature extraction using pretrained FER backbone
    # Freeze the CNN weights initially
    for layer in fer_feature_extractor.layers:
        layer.trainable = False
    
    x = layers.TimeDistributed(fer_feature_extractor, name='td_fer_features')(inputs)
    
    # Temporal modeling with BiLSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.3))(x)
    
    # Embedding layer (for fusion)
    video_embedding = layers.Dense(128, activation='relu', name='video_embedding')(x)
    video_embedding = layers.Dropout(0.4)(video_embedding)
    
    # Classification head
    outputs = layers.Dense(num_classes, activation='softmax', name='video_output')(video_embedding)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Temporal_Video_Model')
    
    return model


def train_phase2_temporal_video(fer_model):
    """
    PHASE 2: Fine-tune temporal video model on RAVDESS.
    Uses pretrained FER backbone wrapped in TimeDistributed + BiLSTM.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING TEMPORAL VIDEO MODEL ON RAVDESS")
    print("Using pretrained FER backbone + BiLSTM for temporal dynamics")
    print("=" * 70)
    
    gc.collect()
    
    # Create feature extractor from trained FER model
    fer_feature_extractor = create_fer_feature_extractor(fer_model)
    print(f"FER feature extractor output shape: {fer_feature_extractor.output_shape}")
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    # Extract video sequences
    print("\nExtracting face sequences from RAVDESS videos...")
    X_video, y_video = extract_video_sequences(
        RAVDESS_DATA_DIR, face_cascade, SEQUENCE_LENGTH, IMG_SIZE
    )
    
    if len(X_video) == 0:
        print("ERROR: No video sequences extracted!")
        return None
    
    print(f"\nVideo data shape: {X_video.shape}")
    print(f"Labels shape: {y_video.shape}")
    
    # Class distribution
    print("\nClass distribution:")
    for i, emotion in enumerate(FER_EMOTIONS):
        count = np.sum(y_video == i)
        if count > 0:
            print(f"  {emotion}: {count}")
    
    # Filter out classes with no samples
    valid_classes = [i for i in range(NUM_FER_EMOTIONS) if np.sum(y_video == i) > 0]
    print(f"\nValid classes: {[FER_EMOTIONS[i] for i in valid_classes]}")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_video), y=y_video)
    class_weight_dict = dict(zip(np.unique(y_video), class_weights))
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_video, y_video, test_size=0.2, stratify=y_video, random_state=42
    )
    
    print(f"\nTraining: {len(X_train)}, Validation: {len(X_val)}")
    
    # Convert to one-hot
    y_train_onehot = keras.utils.to_categorical(y_train, NUM_FER_EMOTIONS)
    y_val_onehot = keras.utils.to_categorical(y_val, NUM_FER_EMOTIONS)
    
    # Build temporal model
    temporal_model = build_temporal_video_model(fer_feature_extractor, SEQUENCE_LENGTH, NUM_FER_EMOTIONS)
    
    print(f"\nTemporal model parameters: {temporal_model.count_params():,}")
    temporal_model.summary()
    
    # Phase 2a: Train with frozen CNN
    print("\n--- Phase 2a: Training with frozen FER backbone ---")
    temporal_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    temporal_model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=30,
        batch_size=8,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2b: Fine-tune top CNN layers
    print("\n--- Phase 2b: Fine-tuning top FER backbone layers ---")
    
    # Unfreeze last few layers of feature extractor
    fer_feature_extractor.trainable = True
    for layer in fer_feature_extractor.layers[:-8]:  # Freeze all but last 8 layers
        layer.trainable = False
    
    temporal_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    temporal_model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=20,
        batch_size=8,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = temporal_model.evaluate(X_val, y_val_onehot, verbose=0)
    print(f"\n>>> Temporal Video Model Accuracy: {val_acc * 100:.2f}%")
    
    # Save model
    temporal_model.save(f'{MODELS_DIR}/temporal_video_ravdess.h5')
    print(f"Saved: {MODELS_DIR}/temporal_video_ravdess.h5")
    
    # Per-class accuracy
    print("\nPer-class validation accuracy:")
    predictions = temporal_model.predict(X_val, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    for i, emotion in enumerate(FER_EMOTIONS):
        mask = y_val == i
        if np.sum(mask) > 0:
            acc = np.mean(pred_labels[mask] == y_val[mask])
            print(f"  {emotion}: {acc * 100:.1f}% ({np.sum(mask)} samples)")
    
    return temporal_model, X_video, y_video


# ==========================================================================
# PHASE 3: AUDIO MODEL + MULTIMODAL FUSION
# ==========================================================================

def extract_audio_features(data_dir, max_len=100, n_mfcc=40):
    """
    Extract MFCC features from RAVDESS audio files.
    """
    features = []
    labels = []
    video_ids = []  # To match with video
    
    actor_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('Actor_')])
    
    print(f"Extracting audio from {len(actor_dirs)} actors...")
    
    for actor_dir in tqdm(actor_dirs):
        actor_path = os.path.join(data_dir, actor_dir)
        
        # Find audio files (wav) or extract from video (mp4)
        audio_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
        video_files = [f for f in os.listdir(actor_path) if f.endswith('.mp4')]
        
        # Use video files as audio source if no wav files
        files_to_process = audio_files if audio_files else video_files
        
        for file in files_to_process:
            parts = file.split('-')
            if len(parts) < 3:
                continue
            
            emotion_code = parts[2]
            if emotion_code not in RAVDESS_TO_FER:
                continue
            
            emotion_name = RAVDESS_TO_FER[emotion_code]
            if emotion_name not in FER_EMOTIONS:
                continue
            
            label = FER_EMOTIONS.index(emotion_name)
            
            file_path = os.path.join(actor_path, file)
            
            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=AUDIO_SR, duration=3.0)
                
                # Extract MFCC
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                mfcc = mfcc.T  # (time, n_mfcc)
                
                # Pad or truncate
                if len(mfcc) < max_len:
                    pad_width = max_len - len(mfcc)
                    mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
                else:
                    mfcc = mfcc[:max_len]
                
                # Normalize
                mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
                
                features.append(mfcc)
                labels.append(label)
                video_ids.append(file.replace('.wav', '').replace('.mp4', ''))
                
            except Exception as e:
                continue
    
    print(f"Extracted {len(features)} audio samples")
    
    return np.array(features), np.array(labels), video_ids


def build_audio_bilstm_model(input_shape=(100, 40), num_classes=7):
    """
    Build BiLSTM model for audio emotion recognition.
    
    Architecture:
    1. Bidirectional LSTM layers for temporal modeling
    2. Dense layers for classification
    """
    inputs = layers.Input(shape=input_shape, name='audio_input')
    
    # BiLSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.3))(x)
    
    # Embedding layer (for fusion)
    audio_embedding = layers.Dense(128, activation='relu', name='audio_embedding')(x)
    audio_embedding = layers.Dropout(0.4)(audio_embedding)
    
    # Classification head
    outputs = layers.Dense(num_classes, activation='softmax', name='audio_output')(audio_embedding)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Audio_BiLSTM_Model')
    
    return model


def train_phase3_audio_model():
    """
    Train audio BiLSTM model on RAVDESS.
    """
    print("\n" + "=" * 70)
    print("PHASE 3a: TRAINING AUDIO BiLSTM MODEL ON RAVDESS")
    print("Learning emotional tone from speech prosody and acoustics")
    print("=" * 70)
    
    gc.collect()
    
    # Extract audio features
    X_audio, y_audio, audio_ids = extract_audio_features(
        RAVDESS_DATA_DIR, AUDIO_MAX_LEN, N_MFCC
    )
    
    if len(X_audio) == 0:
        print("ERROR: No audio features extracted!")
        return None, None, None, None
    
    print(f"\nAudio data shape: {X_audio.shape}")
    
    # Class distribution
    print("\nClass distribution:")
    for i, emotion in enumerate(FER_EMOTIONS):
        count = np.sum(y_audio == i)
        if count > 0:
            print(f"  {emotion}: {count}")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_audio), y=y_audio)
    class_weight_dict = dict(zip(np.unique(y_audio), class_weights))
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_audio, y_audio, test_size=0.2, stratify=y_audio, random_state=42
    )
    
    print(f"\nTraining: {len(X_train)}, Validation: {len(X_val)}")
    
    # Convert to one-hot
    y_train_onehot = keras.utils.to_categorical(y_train, NUM_FER_EMOTIONS)
    y_val_onehot = keras.utils.to_categorical(y_val, NUM_FER_EMOTIONS)
    
    # Build model
    audio_model = build_audio_bilstm_model(
        input_shape=(AUDIO_MAX_LEN, N_MFCC),
        num_classes=NUM_FER_EMOTIONS
    )
    
    print(f"\nAudio model parameters: {audio_model.count_params():,}")
    audio_model.summary()
    
    audio_model.compile(
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
    
    audio_model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=50,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = audio_model.evaluate(X_val, y_val_onehot, verbose=0)
    print(f"\n>>> Audio BiLSTM Model Accuracy: {val_acc * 100:.2f}%")
    
    # Save model
    audio_model.save(f'{MODELS_DIR}/audio_bilstm_ravdess.h5')
    print(f"Saved: {MODELS_DIR}/audio_bilstm_ravdess.h5")
    
    # Per-class accuracy
    print("\nPer-class validation accuracy:")
    predictions = audio_model.predict(X_val, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    for i, emotion in enumerate(FER_EMOTIONS):
        mask = y_val == i
        if np.sum(mask) > 0:
            acc = np.mean(pred_labels[mask] == y_val[mask])
            print(f"  {emotion}: {acc * 100:.1f}% ({np.sum(mask)} samples)")
    
    return audio_model, X_audio, y_audio, audio_ids


def build_multimodal_fusion_model(video_model, audio_model, num_classes=7):
    """
    Build multimodal fusion model combining video and audio at embedding level.
    
    Architecture:
    1. Video embedding from temporal model
    2. Audio embedding from BiLSTM model
    3. Concatenate embeddings
    4. Dense layers for joint classification
    """
    # Get embedding layers
    video_embedding_layer = video_model.get_layer('video_embedding')
    audio_embedding_layer = audio_model.get_layer('audio_embedding')
    
    # Create embedding extractors
    video_embedding_model = models.Model(
        inputs=video_model.input,
        outputs=video_embedding_layer.output,
        name='video_embedding_extractor'
    )
    
    audio_embedding_model = models.Model(
        inputs=audio_model.input,
        outputs=audio_embedding_layer.output,
        name='audio_embedding_extractor'
    )
    
    # Freeze embedding models
    video_embedding_model.trainable = False
    audio_embedding_model.trainable = False
    
    # Fusion model inputs
    video_input = layers.Input(shape=video_model.input_shape[1:], name='fusion_video_input')
    audio_input = layers.Input(shape=audio_model.input_shape[1:], name='fusion_audio_input')
    
    # Get embeddings
    video_emb = video_embedding_model(video_input)
    audio_emb = audio_embedding_model(audio_input)
    
    # Concatenate embeddings
    fused = layers.Concatenate(name='embedding_fusion')([video_emb, audio_emb])
    
    # Dense layers for classification
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(fused)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='fusion_output')(x)
    
    fusion_model = models.Model(
        inputs=[video_input, audio_input],
        outputs=outputs,
        name='Multimodal_Fusion_Model'
    )
    
    return fusion_model, video_embedding_model, audio_embedding_model


def train_multimodal_fusion(video_model, audio_model, X_video, y_video, X_audio, y_audio):
    """
    PHASE 3b: Train multimodal fusion model.
    Combines video and audio embeddings for final classification.
    """
    print("\n" + "=" * 70)
    print("PHASE 3b: TRAINING MULTIMODAL FUSION MODEL")
    print("Combining video temporal features + audio prosody features")
    print("=" * 70)
    
    gc.collect()
    
    # Ensure matched samples (same number of samples for video and audio)
    n_samples = min(len(X_video), len(X_audio))
    
    # Check label alignment
    print(f"\nVideo samples: {len(X_video)}, Audio samples: {len(X_audio)}")
    print(f"Using {n_samples} matched samples")
    
    X_video = X_video[:n_samples]
    y_video = y_video[:n_samples]
    X_audio = X_audio[:n_samples]
    y_audio = y_audio[:n_samples]
    
    # Verify label alignment
    if not np.array_equal(y_video, y_audio):
        print("WARNING: Labels don't match! Using video labels.")
        y = y_video
    else:
        y = y_video
    
    # Split data
    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    
    X_video_train, X_video_val = X_video[train_idx], X_video[val_idx]
    X_audio_train, X_audio_val = X_audio[train_idx], X_audio[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Training: {len(train_idx)}, Validation: {len(val_idx)}")
    
    # Convert to one-hot
    y_train_onehot = keras.utils.to_categorical(y_train, NUM_FER_EMOTIONS)
    y_val_onehot = keras.utils.to_categorical(y_val, NUM_FER_EMOTIONS)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Build fusion model
    fusion_model, video_emb_model, audio_emb_model = build_multimodal_fusion_model(
        video_model, audio_model, NUM_FER_EMOTIONS
    )
    
    print(f"\nFusion model parameters: {fusion_model.count_params():,}")
    fusion_model.summary()
    
    fusion_model.compile(
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
    
    # Train fusion model
    fusion_model.fit(
        [X_video_train, X_audio_train], y_train_onehot,
        validation_data=([X_video_val, X_audio_val], y_val_onehot),
        epochs=50,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = fusion_model.evaluate(
        [X_video_val, X_audio_val], y_val_onehot, verbose=0
    )
    print(f"\n>>> Multimodal Fusion Model Accuracy: {val_acc * 100:.2f}%")
    
    # Save model
    fusion_model.save(f'{MODELS_DIR}/multimodal_fusion_ravdess.h5')
    print(f"Saved: {MODELS_DIR}/multimodal_fusion_ravdess.h5")
    
    # Compare with single modalities
    print("\n" + "=" * 50)
    print("COMPARISON: Single vs Multimodal")
    print("=" * 50)
    
    # Video only
    video_pred = video_model.predict(X_video_val, verbose=0)
    video_acc = np.mean(np.argmax(video_pred, axis=1) == y_val)
    
    # Audio only
    audio_pred = audio_model.predict(X_audio_val, verbose=0)
    audio_acc = np.mean(np.argmax(audio_pred, axis=1) == y_val)
    
    # Late fusion (average probabilities)
    late_fusion_pred = (video_pred + audio_pred) / 2
    late_fusion_acc = np.mean(np.argmax(late_fusion_pred, axis=1) == y_val)
    
    # Learned fusion
    fusion_pred = fusion_model.predict([X_video_val, X_audio_val], verbose=0)
    
    print(f"\n{'Model':<30} {'Accuracy':>10}")
    print("-" * 42)
    print(f"{'Video (Temporal CNN+LSTM)':<30} {video_acc * 100:>9.2f}%")
    print(f"{'Audio (BiLSTM)':<30} {audio_acc * 100:>9.2f}%")
    print(f"{'Late Fusion (Average)':<30} {late_fusion_acc * 100:>9.2f}%")
    print(f"{'Learned Fusion (Embedding)':<30} {val_acc * 100:>9.2f}%")
    print("-" * 42)
    
    # Per-class for best model
    best_acc = max(video_acc, audio_acc, late_fusion_acc, val_acc)
    if val_acc == best_acc:
        best_pred = np.argmax(fusion_pred, axis=1)
        best_name = "Learned Fusion"
    elif late_fusion_acc == best_acc:
        best_pred = np.argmax(late_fusion_pred, axis=1)
        best_name = "Late Fusion"
    elif audio_acc == best_acc:
        best_pred = np.argmax(audio_pred, axis=1)
        best_name = "Audio"
    else:
        best_pred = np.argmax(video_pred, axis=1)
        best_name = "Video"
    
    print(f"\n>>> BEST MODEL: {best_name} with {best_acc * 100:.2f}%")
    
    print(f"\nPer-class accuracy ({best_name}):")
    for i, emotion in enumerate(FER_EMOTIONS):
        mask = y_val == i
        if np.sum(mask) > 0:
            acc = np.mean(best_pred[mask] == y_val[mask])
            print(f"  {emotion}: {acc * 100:.1f}% ({np.sum(mask)} samples)")
    
    return fusion_model


# ==========================================================================
# MAIN PIPELINE
# ==========================================================================

def main():
    print("\n" + "=" * 70)
    print("COMPLETE MULTIMODAL EMOTION RECOGNITION PIPELINE")
    print("=" * 70)
    print("""
    Phase 1: Train CNN backbone on FER2013 (large-scale static dataset)
    Phase 2: Fine-tune temporal model on RAVDESS videos
    Phase 3: Train audio BiLSTM + Multimodal embedding fusion
    """)
    
    # ===== PHASE 1: FER Backbone =====
    fer_model_path = f'{MODELS_DIR}/fer_backbone_final.h5'
    
    if os.path.exists(fer_model_path):
        print(f"\nLoading pretrained FER backbone from {fer_model_path}")
        fer_model = keras.models.load_model(fer_model_path)
    else:
        fer_model = train_phase1_fer_backbone()
        if fer_model is None:
            print("ERROR: Phase 1 failed!")
            return
    
    # ===== PHASE 2: Temporal Video =====
    result = train_phase2_temporal_video(fer_model)
    if result is None:
        print("ERROR: Phase 2 failed!")
        return
    video_model, X_video, y_video = result
    
    # ===== PHASE 3: Audio + Fusion =====
    result = train_phase3_audio_model()
    if result[0] is None:
        print("ERROR: Phase 3a failed!")
        return
    audio_model, X_audio, y_audio, _ = result
    
    # ===== PHASE 3b: Multimodal Fusion =====
    fusion_model = train_multimodal_fusion(
        video_model, audio_model,
        X_video, y_video,
        X_audio, y_audio
    )
    
    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - ALL MODELS SAVED")
    print("=" * 70)
    print(f"""
    Models saved in {MODELS_DIR}/:
    1. fer_backbone_final.h5       - Pretrained CNN on FER2013
    2. temporal_video_ravdess.h5   - Temporal video model (CNN+LSTM)
    3. audio_bilstm_ravdess.h5     - Audio BiLSTM model
    4. multimodal_fusion_ravdess.h5 - Multimodal fusion model
    
    To use for inference, load the fusion model and provide:
    - Video: sequence of {SEQUENCE_LENGTH} face frames ({IMG_SIZE}x{IMG_SIZE} grayscale)
    - Audio: MFCC features ({AUDIO_MAX_LEN}x{N_MFCC})
    """)


if __name__ == "__main__":
    main()
