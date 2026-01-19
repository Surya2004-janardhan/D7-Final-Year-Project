"""
Real-time Emotion Recognition Demo
Records 30 seconds of video and audio, processes, and predicts emotion using trained models.
"""

import os
import cv2
import numpy as np
import sounddevice as sd
import wave
import time
import librosa
from tensorflow import keras

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

def extract_mfcc(audio_path):
    """Extract MFCC features from audio file."""
    y, _ = librosa.load(audio_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc = mfcc.T  # (time, mfcc)

    # Pad or truncate to N_FRAMES
    if mfcc.shape[0] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, N_FRAMES - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:N_FRAMES]

    return mfcc[..., np.newaxis]  # (300, 13, 1)

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

def main():
    print("Starting real-time emotion recognition demo...")
    print("Make sure your camera and microphone are enabled.")
    print("Recording will start in 3 seconds...")

    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    duration = 30
    fs = 22050  # Audio sample rate

    # Record audio
    print("Recording audio for 30 seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    # Save audio to temp.wav
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open('temp.wav', 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_int16.tobytes())

    # Record video
    print("Recording video for 30 seconds...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp.mp4', fourcc, 20.0, (640, 480))

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            display_frame = cv2.flip(frame, 1)  # Mirror for display
            cv2.imshow('Recording', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Recording complete. Processing...")

    # Process audio
    try:
        mfcc = extract_mfcc('temp.wav')
    except Exception as e:
        print(f"Error processing audio: {e}")
        return

    # Process video
    try:
        frames = sample_frames('temp.mp4')
        if frames is None:
            print("Error: Could not extract frames from video.")
            return
    except Exception as e:
        print(f"Error processing video: {e}")
        return

    # Load models
    try:
        audio_model = keras.models.load_model('../models/audio_emotion_model.h5')
        video_model = keras.models.load_model('../models/video_emotion_model.h5')
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Predict audio
    audio_pred = audio_model.predict(np.expand_dims(mfcc, axis=0), verbose=0)
    audio_emotion_idx = np.argmax(audio_pred)
    audio_emotion = EMOTIONS_7[audio_emotion_idx]

    # Predict video
    base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
    base_model.trainable = False

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

    # Fuse predictions
    weight_audio = 0.35
    weight_video = 0.65
    fused_pred = weight_audio * audio_pred + weight_video * video_pred
    fused_emotion_idx = np.argmax(fused_pred)
    fused_emotion = EMOTIONS_7[fused_emotion_idx]

    # Display results
    print("\n" + "="*50)
    print("EMOTION RECOGNITION RESULTS")
    print("="*50)
    print(f"Audio Emotion: {audio_emotion}")
    print(f"Video Emotion: {video_emotion}")
    print(f"Fused Emotion: {fused_emotion}")
    print("="*50)

    # Clean up temp files
    try:
        os.remove('temp.wav')
        os.remove('temp.mp4')
    except:
        pass

if __name__ == '__main__':
    main()