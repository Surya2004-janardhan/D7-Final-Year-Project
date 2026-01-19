import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Paths
AUDIO_MODEL_PATH = 'models/audio_emotion_model.h5'
VIDEO_MODEL_PATH = 'models/video_emotion_model.h5'
TEST_FEATURE_DIR = 'data/audio_features'  # Assuming same for video
TEST_FRAME_DIR = 'data/video_frames'

EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

def load_models():
    """Load trained audio and video models."""
    audio_model = keras.models.load_model(AUDIO_MODEL_PATH)
    video_model = keras.models.load_model(VIDEO_MODEL_PATH)
    return audio_model, video_model

def get_predictions(model, features, is_audio=True):
    """Get predictions from model."""
    if is_audio:
        # For audio, features are MFCCs
        return model.predict(features)
    else:
        # For video, features are averaged frame features
        return model.predict(features)

def fuse_predictions(audio_pred, video_pred, weight_audio=0.65, weight_video=0.35):
    """Fuse predictions using weighted averaging."""
    return weight_audio * audio_pred + weight_video * video_pred

def evaluate_fusion():
    """Evaluate fused model on test set."""
    audio_model, video_model = load_models()

    # Load test data (assuming a test split; for simplicity, use all and split)
    # In practice, load only test features
    audio_features = []
    video_features = []
    labels = []

    for file in os.listdir(TEST_FEATURE_DIR):
        if file.endswith('.npy'):
            audio_feat = np.load(os.path.join(TEST_FEATURE_DIR, file))
            audio_features.append(audio_feat)

            # Corresponding video
            video_file = file.replace('.npy', '.npy')  # Assuming same naming
            if os.path.exists(os.path.join(TEST_FRAME_DIR, video_file)):
                video_feat = np.load(os.path.join(TEST_FRAME_DIR, video_file))
                # Extract features as in training
                base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
                base_model.trainable = False
                frame_features = []
                for frame in video_feat:
                    frame = np.expand_dims(frame, axis=0)
                    feat = base_model(frame)
                    feat = keras.layers.GlobalAveragePooling2D()(feat)
                    frame_features.append(feat.numpy().flatten())
                video_feat_avg = np.mean(frame_features, axis=0)
                video_features.append(video_feat_avg)

                # Label
                parts = file.split('-')
                emotion_idx = int(parts[2]) - 1
                if emotion_idx == 1:
                    emotion_idx = 0
                labels.append(emotion_idx)

    audio_features = np.array(audio_features)
    video_features = np.array(video_features)
    labels = np.array(labels)

    # Get predictions
    audio_preds = get_predictions(audio_model, audio_features, is_audio=True)
    video_preds = get_predictions(video_model, video_features, is_audio=False)

    # Fuse
    fused_preds = fuse_predictions(audio_preds, video_preds)

    # Accuracy
    pred_classes = np.argmax(fused_preds, axis=1)
    accuracy = np.mean(pred_classes == labels)
    print(f"Fused Model Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    evaluate_fusion()