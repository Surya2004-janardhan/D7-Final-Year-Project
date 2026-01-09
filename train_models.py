"""
Complete training pipeline for emotion detection models
Loads data, trains models, and evaluates performance
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from data_loader import RAVDESSDataLoader
from emotion_models import AudioEmotionCNNLSTM, VideoEmotionCNNLSTM, MultimodalEmotionCNNLSTM
import pickle
import os

def prepare_data(X, y):
    """Prepare data for training"""
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # One-hot encode labels
    y_categorical = to_categorical(y, num_classes=8)
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, y_train, y_val

def pad_audio_features(X, target_length=216):
    """Pad audio features to target length"""
    padded = []
    for features in X:
        if features.shape[1] < target_length:
            padding = np.zeros((features.shape[0], target_length - features.shape[1]))
            padded.append(np.hstack([features, padding]))
        else:
            padded.append(features[:, :target_length])
    return np.array(padded)

def train_audio_model(modality='speech'):
    """Train audio emotion detection model"""
    print(f"\n{'='*60}")
    print(f"Training Audio Model - {modality.upper()}")
    print(f"{'='*60}\n")
    
    # Check if model already exists
    model_path = f'models/audio_emotion_{modality}.h5'
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        print(f"Skipping training...\n")
        return None, None
    
    loader = RAVDESSDataLoader()
    
    # Load data
    print(f"Loading audio data ({modality})...")
    X, y = loader.load_audio_dataset(modality)
    
    if not X:
        print(f"No audio data found for {modality}!")
        return None
    
    # Pad features
    print("Padding audio features...")
    X = pad_audio_features(X)
    
    # Prepare data
    print("Preparing data...")
    X_train, X_val, y_train, y_val = prepare_data(X, y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Build and train model
    print("Building model...")
    model = AudioEmotionCNNLSTM(num_emotions=8)
    model.build_model(X_train.shape[1:])
    
    print("Training...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=16)
    
    # Save model
    model_path = f'models/audio_emotion_{modality}.h5'
    os.makedirs('models', exist_ok=True)
    model.model.save(model_path)
    print(f"Model saved: {model_path}")
    
    return model, history

def train_video_model(modality='speech'):
    """Train video emotion detection model"""
    print(f"\n{'='*60}")
    print(f"Training Video Model - {modality.upper()}")
    print(f"Optimized 3D CNN with Residual Blocks")
    print(f"{'='*60}\n")
    
    # Check if model already exists
    model_path = f'models/video_emotion_{modality}.h5'
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        print(f"Skipping training...\n")
        return None, None
    
    loader = RAVDESSDataLoader()
    
    # Load data
    print(f"Loading video data ({modality})...")
    X, y = loader.load_video_dataset(modality, n_frames=16)
    
    if not X:
        print(f"No video data found for {modality}!")
        return None
    
    # Prepare data
    print("Preparing data...")
    X_train, X_val, y_train, y_val = prepare_data(X, y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Build and train model
    print("Building model...")
    model = VideoEmotionCNNLSTM(num_emotions=8)
    model.build_model(X_train.shape[1:])
    
    print("Training...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=4)
    
    # Save model
    model_path = f'models/video_emotion_{modality}.h5'
    os.makedirs('models', exist_ok=True)
    model.model.save(model_path)
    print(f"Model saved: {model_path}")
    
    return model, history

def plot_training_history(history, title):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/{title.replace(" ", "_").lower()}.png', dpi=100)
    print(f"Plot saved: plots/{title.replace(' ', '_').lower()}.png")

def main():
    """Main training pipeline"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("\n" + "="*60)
    print("EMOTION DETECTION MODEL TRAINING PIPELINE")
    print("OPTIMIZED 3D CNN: Residual Blocks + LayerNorm")
    print("Hardware: 16GB RAM + RTX 2050")
    print("Frames: 16 @ 160×160 | Batch: 4 | LR Schedule: ReduceLROnPlateau")
    print("="*60)
    
    # Train audio models
    print("\nStarting Audio Model Training...")
    audio_speech_result = train_audio_model('speech')
    if audio_speech_result[0] is not None:
        audio_speech_model, audio_speech_history = audio_speech_result
        plot_training_history(audio_speech_history, "Audio Speech Model")
    
    audio_song_result = train_audio_model('song')
    if audio_song_result[0] is not None:
        audio_song_model, audio_song_history = audio_song_result
        plot_training_history(audio_song_history, "Audio Song Model")
    
    # Train video models
    print("\nStarting Video Model Training...")
    print("Note: RAVDESS only has video of singing, not speech")
    
    # Skip video speech (doesn't exist in RAVDESS)
    print("\nSkipping Video Speech (not available in dataset)")
    
    video_song_result = train_video_model('song')
    if video_song_result[0] is not None:
        video_song_model, video_song_history = video_song_result
        plot_training_history(video_song_history, "Video Song Model")
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    print("\nModels saved in: models/")
    print("Plots saved in: plots/")

if __name__ == '__main__':
    main()
