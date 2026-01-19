import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Paths
AUDIO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/audio_emotion_model.h5')
VIDEO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/video_emotion_model.h5')
TEST_FEATURE_DIR = os.path.join(os.path.dirname(__file__), '../data/audio_features')
TEST_FRAME_DIR = os.path.join(os.path.dirname(__file__), '../data/video_frames')

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
    print("TEST_FEATURE_DIR:", TEST_FEATURE_DIR)
    print("TEST_FRAME_DIR:", TEST_FRAME_DIR)
    print("Video dir exists:", os.path.exists(TEST_FRAME_DIR))
    if os.path.exists(TEST_FRAME_DIR):
        video_files = os.listdir(TEST_FRAME_DIR)
        print(f"Video files: {len(video_files)}, first 5: {video_files[:5]}")
    audio_features = []
    video_features = []
    labels = []

    files = os.listdir(TEST_FEATURE_DIR)
    print(f"Files in audio dir: {len(files)}, first 5: {files[:5]}")
    for i, file in enumerate(files):
        if file.endswith('.npy'):
            # Corresponding video: replace '03' with '01' in filename
            video_file = file.replace('03-', '01-', 1)
            video_path = os.path.join(TEST_FRAME_DIR, video_file)
            if os.path.exists(video_path):
                # load etc
                audio_feat = np.load(os.path.join(TEST_FEATURE_DIR, file))
                audio_features.append(audio_feat)

                video_feat = np.load(video_path)
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
                emotion_code = parts[2]
                emotion_map = {'01': 0, '02': 0, '03': 1, '04': 2, '05': 3, '06': 4, '07': 5, '08': 6}
                emotion_idx = emotion_map.get(emotion_code, 0)
                labels.append(emotion_idx)

    print(f"Loaded {len(audio_features)} audio, {len(video_features)} video, {len(labels)} labels")
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