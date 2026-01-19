"""
Test Audio, Video, and Fusion Emotion Recognition Models
Generate plots and evaluation metrics
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import cv2

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Emotion labels (7 emotions)
EMOTIONS_7 = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# For 5 emotions (merged)
EMOTIONS_5 = ['neutral', 'happy', 'sad', 'angry', 'fearful']

# RAVDESS mapping
RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

EMOTION_TO_IDX_7 = {
    'neutral': 0, 'calm': 0, 'happy': 1, 'sad': 2,
    'angry': 3, 'fearful': 4, 'disgust': 5, 'surprised': 6
}

EMOTION_TO_IDX_5 = {
    'neutral': 0, 'calm': 0,  # Merge calm with neutral
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fearful': 4,
    'disgust': 4,  # Merge disgust with fearful
    'surprised': 1  # Merge surprised with happy
}

def load_audio_test_data(max_samples=1000, num_emotions=7):
    """Load audio features for testing"""
    emotion_to_idx = EMOTION_TO_IDX_7 if num_emotions == 7 else EMOTION_TO_IDX_5
    audio_dir = Path('data/audio_features')
    X_audio, y_audio = [], []

    files = list(audio_dir.glob('*.npy'))
    print(f"Found {len(files)} audio feature files")

    for file in tqdm(files[:max_samples], desc="Loading audio features"):
        parts = file.stem.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in RAVDESS_EMOTIONS:
                emotion_name = RAVDESS_EMOTIONS[emotion_code]
                emotion_idx = emotion_to_idx[emotion_name]

                try:
                    feat = np.load(file)
                    X_audio.append(feat)
                    y_audio.append(emotion_idx)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    return np.array(X_audio), np.array(y_audio)

def load_video_test_data(max_samples=1000, num_emotions=7):
    """Load video frames for testing"""
    emotion_to_idx = EMOTION_TO_IDX_7 if num_emotions == 7 else EMOTION_TO_IDX_5
    video_dir = Path('data/video_frames')
    X_video, y_video = [], []

    files = list(video_dir.glob('*.npy'))
    print(f"Found {len(files)} video frame files")

    # Load base model for feature extraction
    base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
    base_model.trainable = False

    for file in tqdm(files[:max_samples], desc="Loading video features"):
        parts = file.stem.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in RAVDESS_EMOTIONS:
                emotion_name = RAVDESS_EMOTIONS[emotion_code]
                emotion_idx = emotion_to_idx[emotion_name]

                try:
                    frames = np.load(file)
                    # Extract features as in training
                    frame_features = []
                    for frame in frames:
                        frame = np.expand_dims(frame, axis=0)
                        feat = base_model(frame)
                        feat = keras.layers.GlobalAveragePooling2D()(feat)
                        frame_features.append(feat.numpy().flatten())
                    video_feat_avg = np.mean(frame_features, axis=0)
                    X_video.append(video_feat_avg)
                    y_video.append(emotion_idx)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    return np.array(X_video), np.array(y_video)

def load_matching_test_data(max_samples=500, num_emotions=7):
    """Load matching audio and video data for fusion testing"""
    emotion_to_idx = EMOTION_TO_IDX_7 if num_emotions == 7 else EMOTION_TO_IDX_5
    audio_dir = Path('data/audio_features')
    video_dir = Path('data/video_frames')

    X_audio, X_video, y = [], [], []

    # Load base model for video features
    base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
    base_model.trainable = False

    audio_files = list(audio_dir.glob('*.npy'))
    print(f"Found {len(audio_files)} audio files")

    count = 0
    for audio_file in tqdm(audio_files, desc="Loading matching data"):
        if count >= max_samples:
            break

        # Find corresponding video file (replace 03- with 01-)
        video_filename = audio_file.name.replace('03-', '01-', 1)
        video_file = video_dir / video_filename

        if video_file.exists():
            parts = audio_file.stem.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in RAVDESS_EMOTIONS:
                    emotion_name = RAVDESS_EMOTIONS[emotion_code]
                    emotion_idx = emotion_to_idx[emotion_name]

                    try:
                        # Load audio
                        audio_feat = np.load(audio_file)
                        X_audio.append(audio_feat)

                        # Load video and extract features
                        frames = np.load(video_file)
                        frame_features = []
                        for frame in frames:
                            frame = np.expand_dims(frame, axis=0)
                            feat = base_model(frame)
                            feat = keras.layers.GlobalAveragePooling2D()(feat)
                            frame_features.append(feat.numpy().flatten())
                        video_feat_avg = np.mean(frame_features, axis=0)
                        X_video.append(video_feat_avg)

                        y.append(emotion_idx)
                        count += 1
                    except Exception as e:
                        print(f"Error loading pair {audio_file.name}: {e}")

    print(f"Loaded {len(X_audio)} matching audio-video pairs")
    return np.array(X_audio), np.array(X_video), np.array(y)

def test_model(model_path, X, y, model_name, save_plot=True):
    """Test a single model and generate plots"""
    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print(f"{'='*50}")

    # Load model
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Get number of classes from model
    num_classes = model.output_shape[-1]
    emotions = EMOTIONS_7[:num_classes] if num_classes <= len(EMOTIONS_7) else EMOTIONS_7

    # Make predictions
    print("Making predictions...")
    y_pred_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=emotions))

    # Confusion matrix
    if save_plot:
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emotions, yticklabels=emotions)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plot_path = f'plots/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Confusion matrix saved to {plot_path}")

    return accuracy, y_pred

def test_fusion(audio_model_path, video_model_path, X_audio, X_video, y, save_plot=True):
    """Test fusion model"""
    print(f"\n{'='*50}")
    print("Testing Fusion Model")
    print(f"{'='*50}")

    # Load models
    audio_model = keras.models.load_model(audio_model_path)
    video_model = keras.models.load_model(video_model_path)
    print("Models loaded")
    
    # Get number of classes from audio model (assuming both have same)
    num_classes = audio_model.output_shape[-1]
    emotions = EMOTIONS_7[:num_classes] if num_classes <= len(EMOTIONS_7) else EMOTIONS_7

    # Make predictions
    print("Making predictions...")
    audio_preds = audio_model.predict(X_audio, verbose=0)
    video_preds = video_model.predict(X_video, verbose=0)

    # Fuse predictions (weighted average)
    weight_audio = 0.65
    weight_video = 0.35
    fused_preds = weight_audio * audio_preds + weight_video * video_preds
    y_pred = np.argmax(fused_preds, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Fusion Accuracy: {accuracy*100:.2f}% (weights: audio {weight_audio}, video {weight_video})")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=emotions))

    # Confusion matrix
    if save_plot:
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emotions, yticklabels=emotions)
        plt.title('Confusion Matrix - Fusion Model')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix_fusion.png')
        plt.close()
        print("Confusion matrix saved to plots/confusion_matrix_fusion.png")

    return accuracy, y_pred

def main():
    print("Testing Emotion Recognition Models")
    print("="*50)

    # Load audio model to check num classes
    audio_model = keras.models.load_model('models/audio_emotion_model.h5')
    audio_num_classes = audio_model.output_shape[-1]
    print(f"Audio model has {audio_num_classes} classes")

    # Load video model to check num classes
    video_model = keras.models.load_model('models/video_emotion_model.h5')
    video_num_classes = video_model.output_shape[-1]
    print(f"Video model has {video_num_classes} classes")

    # Test Audio Model
    print("\nLoading audio test data...")
    X_audio, y_audio = load_audio_test_data(num_emotions=audio_num_classes)
    if len(X_audio) > 0:
        audio_acc, _ = test_model('models/audio_emotion_model.h5', X_audio, y_audio, "Audio Model")
    else:
        print("No audio data found")
        audio_acc = 0

    # Test Video Model
    print("\nLoading video test data...")
    X_video, y_video = load_video_test_data(num_emotions=video_num_classes)
    if len(X_video) > 0:
        video_acc, _ = test_model('models/video_emotion_model.h5', X_video, y_video, "Video Model")
    else:
        print("No video data found")
        video_acc = 0

    # Test Fusion Model
    fusion_num_classes = min(audio_num_classes, video_num_classes)  # Use the smaller one
    print(f"\nUsing {fusion_num_classes} classes for fusion")
    print("\nLoading matching audio-video test data...")
    X_audio_match, X_video_match, y_match = load_matching_test_data(num_emotions=fusion_num_classes)
    if len(X_audio_match) > 0:
        fusion_acc, _ = test_fusion('models/audio_emotion_model.h5', 'models/video_emotion_model.h5',
                                   X_audio_match, X_video_match, y_match)
    else:
        print("No matching data found")
        fusion_acc = 0

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Audio Model Accuracy: {audio_acc*100:.2f}%")
    print(f"Video Model Accuracy: {video_acc*100:.2f}%")
    print(f"Fusion Model Accuracy: {fusion_acc*100:.2f}%")

    # Save summary plot
    models = ['Audio', 'Video', 'Fusion']
    accuracies = [audio_acc, video_acc, fusion_acc]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'red'])
    plt.title('Model Accuracies Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc*100:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('plots/model_accuracies_comparison.png')
    plt.close()
    print("Accuracy comparison plot saved to plots/model_accuracies_comparison.png")

if __name__ == '__main__':
    main()