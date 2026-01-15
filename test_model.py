"""
Test and Analyze Temporal Video Emotion Model
Diagnose low accuracy issues
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import cv2

# Emotion labels
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# RAVDESS mapping
RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

EMOTION_TO_IDX = {
    'neutral': 0, 'calm': 0, 'happy': 1, 'sad': 2,
    'angry': 3, 'fearful': 4, 'disgust': 5, 'surprised': 6
}


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect(self, frame, target_size=(48, 48)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
        
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(w * 0.1)
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)
        
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        return face


def extract_faces_from_video(video_path, n_frames=16):
    """Extract face sequence from video"""
    detector = FaceDetector()
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        cap.release()
        return None, 0
    
    indices = np.linspace(0, frame_count - 1, n_frames, dtype=int)
    
    faces = []
    faces_detected = 0
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        face = detector.detect(frame, (48, 48))
        if face is not None:
            faces.append(face)
            faces_detected += 1
        elif len(faces) > 0:
            faces.append(faces[-1])
    
    cap.release()
    
    if len(faces) < n_frames // 2:
        return None, faces_detected
    
    while len(faces) < n_frames:
        faces.append(faces[-1] if faces else np.zeros((48, 48)))
    
    faces = np.array(faces[:n_frames], dtype=np.float32) / 255.0
    faces = np.expand_dims(faces, axis=-1)
    
    return faces, faces_detected


def load_test_data(data_dir='data', n_frames=16, max_per_class=50):
    """Load test data with detailed info"""
    X, y, video_info = [], [], []
    
    data_path = Path(data_dir)
    actor_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('Actor')])
    
    class_counts = {i: 0 for i in range(7)}
    
    print(f"Loading data from {len(actor_dirs)} actors...")
    
    for actor_dir in tqdm(actor_dirs):
        video_files = list(actor_dir.glob('*.mp4'))
        
        for video_file in video_files:
            parts = video_file.stem.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                
                if emotion_code in RAVDESS_EMOTIONS:
                    emotion_name = RAVDESS_EMOTIONS[emotion_code]
                    
                    if emotion_name in EMOTION_TO_IDX:
                        emotion_idx = EMOTION_TO_IDX[emotion_name]
                        
                        if class_counts[emotion_idx] >= max_per_class:
                            continue
                        
                        faces, n_detected = extract_faces_from_video(video_file, n_frames)
                        
                        if faces is not None:
                            X.append(faces)
                            y.append(emotion_idx)
                            video_info.append({
                                'path': str(video_file),
                                'emotion': emotion_name,
                                'faces_detected': n_detected,
                                'actor': actor_dir.name
                            })
                            class_counts[emotion_idx] += 1
    
    return np.array(X), np.array(y), video_info, class_counts


def analyze_model(model_path='models/temporal_video_emotion.h5'):
    """Comprehensive model analysis"""
    
    print("=" * 70)
    print("TEMPORAL VIDEO EMOTION MODEL ANALYSIS")
    print("=" * 70)
    
    # Load model
    print("\n1. Loading model...")
    model = keras.models.load_model(model_path)
    print(f"   Model loaded from {model_path}")
    
    # Load test data
    print("\n2. Loading test data...")
    X, y, video_info, class_counts = load_test_data(max_per_class=30)
    
    print(f"\n   Loaded {len(X)} samples")
    print(f"   Class distribution:")
    for idx, count in class_counts.items():
        if count > 0:
            print(f"      {EMOTIONS[idx]}: {count}")
    
    # Make predictions
    print("\n3. Making predictions...")
    y_pred_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y)
    print(f"\n4. Overall Accuracy: {accuracy*100:.2f}%")
    
    # Per-class accuracy
    print("\n5. Per-class accuracy:")
    for i in range(7):
        mask = y == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y[mask])
            print(f"   {EMOTIONS[i]}: {class_acc*100:.2f}% ({np.sum(mask)} samples)")
    
    # Classification report
    print("\n6. Classification Report:")
    present_classes = sorted(list(set(y)))
    target_names = [EMOTIONS[i] for i in present_classes]
    print(classification_report(y, y_pred, target_names=target_names, labels=present_classes))
    
    # Confusion matrix
    print("\n7. Generating confusion matrix...")
    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[EMOTIONS[i] for i in present_classes],
                yticklabels=[EMOTIONS[i] for i in present_classes])
    plt.title('Confusion Matrix - Temporal Video Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    print("   Saved to plots/confusion_matrix.png")
    
    # Analyze misclassifications
    print("\n8. Misclassification Analysis:")
    misclassified = []
    for i, (true, pred) in enumerate(zip(y, y_pred)):
        if true != pred:
            misclassified.append({
                'video': video_info[i]['path'],
                'actor': video_info[i]['actor'],
                'true': EMOTIONS[true],
                'pred': EMOTIONS[pred],
                'confidence': y_pred_probs[i][pred],
                'faces_detected': video_info[i]['faces_detected']
            })
    
    print(f"   Total misclassified: {len(misclassified)} / {len(y)} ({len(misclassified)/len(y)*100:.1f}%)")
    
    # Common confusion pairs
    confusion_pairs = {}
    for m in misclassified:
        pair = f"{m['true']} -> {m['pred']}"
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    print("\n   Most common confusions:")
    for pair, count in sorted(confusion_pairs.items(), key=lambda x: -x[1])[:10]:
        print(f"      {pair}: {count}")
    
    # Analyze face detection quality
    print("\n9. Face Detection Quality:")
    face_counts = [v['faces_detected'] for v in video_info]
    print(f"   Average faces detected per video: {np.mean(face_counts):.1f} / 16")
    print(f"   Min faces detected: {np.min(face_counts)}")
    print(f"   Videos with < 8 faces: {sum(1 for f in face_counts if f < 8)}")
    
    # Low confidence predictions
    print("\n10. Confidence Analysis:")
    max_probs = np.max(y_pred_probs, axis=1)
    print(f"    Average confidence: {np.mean(max_probs)*100:.1f}%")
    print(f"    Min confidence: {np.min(max_probs)*100:.1f}%")
    print(f"    Max confidence: {np.max(max_probs)*100:.1f}%")
    
    low_conf = max_probs < 0.3
    print(f"    Low confidence (<30%): {np.sum(low_conf)} samples")
    if np.sum(low_conf) > 0:
        print(f"    Accuracy on low confidence: {np.mean(y_pred[low_conf] == y[low_conf])*100:.1f}%")
    
    # Prediction distribution
    print("\n11. Prediction Distribution (what model tends to predict):")
    unique, counts = np.unique(y_pred, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {EMOTIONS[u]}: {c} ({c/len(y_pred)*100:.1f}%)")
    
    return accuracy, cm


def test_single_video(video_path, model_path='models/temporal_video_emotion.h5'):
    """Test model on a single video"""
    print(f"\nTesting on: {video_path}")
    
    model = keras.models.load_model(model_path)
    faces, n_detected = extract_faces_from_video(video_path)
    
    if faces is None:
        print("Could not extract faces from video")
        return
    
    print(f"Faces detected: {n_detected}/16")
    
    # Predict
    faces = np.expand_dims(faces, axis=0)
    pred = model.predict(faces, verbose=0)[0]
    
    print("\nPrediction probabilities:")
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, pred)):
        bar = '█' * int(prob * 30)
        print(f"  {emotion:12s}: {prob*100:5.1f}% {bar}")
    
    predicted_emotion = EMOTIONS[np.argmax(pred)]
    confidence = np.max(pred) * 100
    print(f"\nPredicted: {predicted_emotion} ({confidence:.1f}% confidence)")
    
    return predicted_emotion, pred


if __name__ == '__main__':
    # Run analysis
    accuracy, cm = analyze_model()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nFinal accuracy: {accuracy*100:.2f}%")
    print("\nPossible reasons for low accuracy:")
    print("  1. Limited training data (529 videos)")
    print("  2. Class imbalance in RAVDESS dataset")
    print("  3. Face detection missing frames")
    print("  4. Subtle emotion differences (e.g., sad vs fearful)")
    print("  5. Model needs more epochs or different architecture")
