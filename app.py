from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import librosa
from tensorflow import keras

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit for uploads

# Emotion labels
EMOTIONS_7 = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# MFCC parameters
SR = 22050
N_MFCC = 13
HOP_LENGTH = 512
N_FRAMES = 300

# Video parameters
NUM_FRAMES = 16
TARGET_SIZE = (112, 112)

# Load models once at startup
print("Loading models...")
try:
    audio_model = keras.models.load_model('models/audio_emotion_model.h5')
    video_model = keras.models.load_model('models/video_emotion_model.h5')
    base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
    base_model.trainable = False
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    audio_model = None
    video_model = None
    base_model = None

def extract_mfcc(audio_path):
    """Extract MFCC features from audio file."""
    try:
        y, _ = librosa.load(audio_path, sr=SR)
        if len(y) == 0:
            raise ValueError("Empty audio")
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        mfcc = mfcc.T

        if mfcc.shape[0] < N_FRAMES:
            mfcc = np.pad(mfcc, ((0, N_FRAMES - mfcc.shape[0]), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:N_FRAMES]

        return mfcc[..., np.newaxis]
    except Exception as e:
        print(f"Failed to extract MFCC: {e}")
        return None

def sample_frames(video_path):
    """Sample NUM_FRAMES from video uniformly."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return None

    step = max(1, total_frames // NUM_FRAMES)
    frames = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, TARGET_SIZE) / 255.0
            frames.append(frame)
        if len(frames) == NUM_FRAMES:
            break

    cap.release()
    return np.array(frames) if len(frames) == NUM_FRAMES else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    print("Processing request...")
    if audio_model is None or video_model is None or base_model is None:
        print("Models not loaded")
        return jsonify({'error': 'Models not loaded'})

    try:
        # Save uploaded file (video with audio)
        video_file = request.files['video']

        video_path = 'temp_video.webm'
        print(f"Saving file to {video_path}")

        video_file.save(video_path)

        # Process audio from video
        print("Extracting audio features...")
        mfcc = extract_mfcc(video_path)
        if mfcc is None:
            os.remove(video_path)
            return jsonify({'error': 'Could not extract audio features'})
        print(f"MFCC shape: {mfcc.shape}")

        # Process video
        print("Extracting video features...")
        frames = sample_frames(video_path)
        if frames is None:
            os.remove(video_path)
            return jsonify({'error': 'Could not extract frames from video'})

        print(f"Frames shape: {frames.shape}")

        # Predict audio
        print("Predicting audio...")
        audio_pred = audio_model.predict(np.expand_dims(mfcc, axis=0), verbose=0)
        audio_emotion_idx = np.argmax(audio_pred)
        audio_emotion = EMOTIONS_7[audio_emotion_idx]
        print(f"Audio emotion: {audio_emotion}")

        # Predict video
        print("Predicting video...")
        frame_features = []
        for frame in frames:
            frame_exp = np.expand_dims(frame, axis=0)
            feat = base_model(frame_exp)
            feat = keras.layers.GlobalAveragePooling2D()(feat)
            frame_features.append(feat.numpy().flatten())
        video_feat = np.mean(frame_features, axis=0)

        video_pred = video_model.predict(np.expand_dims(video_feat, axis=0), verbose=0)
        video_emotion_idx = np.argmax(video_pred)
        video_emotion = EMOTIONS_7[video_emotion_idx]
        print(f"Video emotion: {video_emotion}")

        # Fuse predictions
        weight_audio = 0.35
        weight_video = 0.65
        fused_pred = weight_audio * audio_pred + weight_video * video_pred
        fused_emotion_idx = np.argmax(fused_pred)
        fused_emotion = EMOTIONS_7[fused_emotion_idx]
        print(f"Fused emotion: {fused_emotion}")

        # Clean up
        os.remove(video_path)
        print("Processing complete")

        return jsonify({
            'audio_emotion': audio_emotion,
            'video_emotion': video_emotion,
            'fused_emotion': fused_emotion
        })

    except Exception as e:
        print(f"Error during processing: {e}")
        # Clean up on error
        if os.path.exists('temp_video.webm'):
            os.remove('temp_video.webm')
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)