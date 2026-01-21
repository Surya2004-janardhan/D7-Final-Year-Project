from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import librosa
from tensorflow import keras
import requests
from pydub import AudioSegment
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit for uploads
# Emotion labels
EMOTIONS_7 = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

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

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Audio processing parameters
SR = 16000
WINDOW_SIZE = 0.025
HOP_SIZE = 0.01
N_MFCC = 13
HOP_LENGTH = 512

# Video processing parameters
VIDEO_WINDOW_SIZE = 1  # seconds
TARGET_SIZE = (112, 112)
NUM_FRAMES = 10

def extract_mfcc(audio_path):
    """Extract MFCC features from audio file in windows."""
    try:
        # Load audio using pydub (handles WebM better than librosa)
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono and get raw samples
        audio = audio.set_channels(1)  # Convert to mono
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize to [-1, 1] range
        samples = samples / (2**(audio.sample_width*8-1))
        
        # Resample if needed
        if audio.frame_rate != SR:
            samples = librosa.resample(samples, orig_sr=audio.frame_rate, target_sr=SR)
        
        y = samples
        sr = SR
        
        if len(y) == 0:
            raise ValueError("Empty audio")
        
        # Calculate window parameters
        window_samples = int(WINDOW_SIZE * SR)
        hop_samples = int(HOP_SIZE * SR)
        
        mfcc_windows = []
        for start in range(0, len(y) - window_samples + 1, hop_samples):
            end = start + window_samples
            y_window = y[start:end]
            mfcc = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
            mfcc = mfcc.T
            
            if mfcc.shape[0] < N_FRAMES:
                mfcc = np.pad(mfcc, ((0, N_FRAMES - mfcc.shape[0]), (0, 0)), mode='constant')
            else:
                mfcc = mfcc[:N_FRAMES]
            
            mfcc_windows.append(mfcc[..., np.newaxis])
        
        return np.array(mfcc_windows) if mfcc_windows else None
    except Exception as e:
        print(f"Failed to extract MFCC: {e}")
        return None

def cognitive_reasoning(audio_emotion, video_emotion, fused_emotion, audio_preds, video_preds):
    """Add cognitive reasoning to the emotion predictions."""
    reasoning = []
    
    # Check agreement
    if audio_emotion == video_emotion:
        reasoning.append(f"Audio and video modalities agree on {audio_emotion}.")
    else:
        reasoning.append(f"Audio suggests {audio_emotion}, while video suggests {video_emotion}. Fusion resulted in {fused_emotion}.")
    
    # Check confidence
    audio_conf = np.max(np.mean(audio_preds, axis=0))
    video_conf = np.max(np.mean(video_preds, axis=0))
    reasoning.append(f"Audio confidence: {audio_conf:.2f}, Video confidence: {video_conf:.2f}.")
    
    # Temporal consistency
    audio_consistency = len(set([EMOTIONS_7[np.argmax(p)] for p in audio_preds])) / len(audio_preds)
    video_consistency = len(set([EMOTIONS_7[np.argmax(p)] for p in video_preds])) / len(video_preds)
    reasoning.append(f"Temporal consistency - Audio: {audio_consistency:.2f}, Video: {video_consistency:.2f}.")
    
    # Human-like reasoning
    if fused_emotion in ['angry', 'fearful', 'sad']:
        reasoning.append("Detected negative emotion. Consider context: is this appropriate for the situation?")
    elif fused_emotion in ['happy', 'surprised']:
        reasoning.append("Positive emotion detected. This might indicate engagement or excitement.")
    elif fused_emotion == 'neutral':
        reasoning.append("Neutral expression. Could indicate calmness or lack of strong emotion.")
    
    # Modality reliability
    if audio_conf > video_conf:
        reasoning.append("Audio seems more reliable than video in this case.")
    elif video_conf > audio_conf:
        reasoning.append("Video seems more reliable than audio in this case.")
    
    return " ".join(reasoning)

def generate_llm_content(fused_emotion, reasoning, audio_temporal, video_temporal):
    """Generate story, quote, video, and songs using Groq LLM."""
    prompt = f"""
Based on the detected emotion: {fused_emotion}
Cognitive reasoning: {reasoning}
Temporal audio emotions: {', '.join(audio_temporal)}
Temporal video emotions: {', '.join(video_temporal)}

Generate:
1. A short story (2-3 sentences) related to this emotion.
2. An inspirational quote about this emotion.
3. A YouTube video suggestion (title and why it fits).
4. 2-3 song recommendations strongly relevant to this emotion, with artist names.

Format the response as JSON with keys: story, quote, video, songs
"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        response = requests.post(GROQ_URL, headers=headers, json=data)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            import json
            return json.loads(content)
        else:
            print(f"Groq API error: {response.status_code} - {response.text}")
            return {
                "story": "Unable to generate story.",
                "quote": "Unable to generate quote.",
                "video": "Unable to generate video suggestion.",
                "songs": ["Unable to generate songs."]
            }
    except Exception as e:
        print(f"LLM error: {e}")
        return {
            "story": "Unable to generate story.",
            "quote": "Unable to generate quote.",
            "video": "Unable to generate video suggestion.",
            "songs": ["Unable to generate songs."]
        }

def sample_frames(video_path):
    """Sample frame sequences from video over time."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    if total_frames == 0 or duration == 0:
        cap.release()
        return None

    # Sample sequences every VIDEO_WINDOW_SIZE seconds
    sequences = []
    window_frames = int(VIDEO_WINDOW_SIZE * fps)
    hop_frames = int(VIDEO_WINDOW_SIZE * fps / 2)  # overlap
    
    for start_frame in range(0, total_frames - window_frames + 1, hop_frames):
        end_frame = start_frame + window_frames
        frames = []
        
        for frame_idx in range(start_frame, end_frame, max(1, window_frames // NUM_FRAMES)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, TARGET_SIZE) / 255.0
                frames.append(frame)
            if len(frames) == NUM_FRAMES:
                break
        
        if len(frames) == NUM_FRAMES:
            sequences.append(np.array(frames))
    
    cap.release()
    return np.array(sequences) if sequences else None

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
        # Save uploaded files
        video_file = request.files['video']
        audio_file = request.files['audio']

        video_path = 'temp_video.webm'
        audio_path = 'temp_audio.webm'
        print(f"Saving files to {video_path} and {audio_path}")

        video_file.save(video_path)
        audio_file.save(audio_path)

        # Process audio
        print("Extracting audio features...")
        # Convert .webm audio to .wav for better compatibility
        try:
            audio_seg = AudioSegment.from_file(audio_path)
            wav_path = 'temp_audio.wav'
            audio_seg.export(wav_path, format='wav')
            print(f"Converted audio to {wav_path}")
        except Exception as e:
            print(f"Failed to convert audio to wav: {e}")
            print("Processing request...")
            if video_model is None or base_model is None:
                print("Models not loaded")
                return jsonify({'error': 'Models not loaded'})

            try:
                # Save uploaded video file only
                video_file = request.files['video']
                video_path = 'temp_video.webm'
                print(f"Saving file to {video_path}")
                video_file.save(video_path)

                # Process video
                print("Extracting video features...")
                frame_sequences = sample_frames(video_path)
                if frame_sequences is None or len(frame_sequences) == 0:
                    if os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                        except Exception as e:
                            print(f"Failed to remove {video_path}: {e}")
                    return jsonify({'error': 'Could not extract frames from video'})
                print(f"Frame sequences shape: {frame_sequences.shape}")

                # Predict video temporal
                print("Predicting video temporal...")
                video_preds = []
                for seq in frame_sequences:
                    frame_features = []
                    for frame in seq:
                        frame_exp = np.expand_dims(frame, axis=0)
                        feat = base_model(frame_exp)
                        feat = keras.layers.GlobalAveragePooling2D()(feat)
                        frame_features.append(feat.numpy().flatten())
                    video_feat = np.mean(frame_features, axis=0)
                    pred = video_model.predict(np.expand_dims(video_feat, axis=0), verbose=0)[0]
                    video_preds.append(pred)
                video_preds = np.array(video_preds)
                video_emotions_temporal = [EMOTIONS_7[np.argmax(pred)] for pred in video_preds]

                # Overall predictions (average)
                video_pred_avg = np.mean(video_preds, axis=0)
                video_emotion = EMOTIONS_7[np.argmax(video_pred_avg)]

                # Clean up
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except Exception as e:
                        print(f"Failed to remove {video_path}: {e}")
                print("Processing complete")

                return jsonify({
                    'audio_emotion': None,
                    'video_emotion': video_emotion,
                    'fused_emotion': video_emotion,
                    'reasoning': f'Only video processed. Detected emotion: {video_emotion}',
                    'story': '',
                    'quote': '',
                    'video': '',
                    'songs': [],
                    'audio_temporal': [],
                    'video_temporal': video_emotions_temporal,
                    'audio_probs_temporal': [],
                    'video_probs_temporal': video_preds.tolist(),
                    'time_points': list(range(len(video_emotions_temporal)))
                })

        # Overall predictions (average)
        audio_pred_avg = np.mean(audio_preds, axis=0)
        video_pred_avg = np.mean(video_preds, axis=0)
        audio_emotion = EMOTIONS_7[np.argmax(audio_pred_avg)]
        video_emotion = EMOTIONS_7[np.argmax(video_pred_avg)]

        # Fuse predictions
        weight_audio = 0.35
        weight_video = 0.65
        fused_pred = weight_audio * audio_pred_avg + weight_video * video_pred_avg
        fused_emotion = EMOTIONS_7[np.argmax(fused_pred)]

        # Cognitive layer: Add reasoning
        reasoning = cognitive_reasoning(audio_emotion, video_emotion, fused_emotion, audio_preds, video_preds)

        # LLM layer: Generate content
        llm_content = generate_llm_content(fused_emotion, reasoning, audio_emotions_temporal, video_emotions_temporal)

        print(f"Audio emotion: {audio_emotion}")
        print(f"Video emotion: {video_emotion}")
        print(f"Fused emotion: {fused_emotion}")
        print(f"Reasoning: {reasoning}")
        print(f"LLM content: {llm_content}")


        # Clean up
        for path in [video_path, audio_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)
        print("Processing complete")

        return jsonify({
            'audio_emotion': audio_emotion,
            'video_emotion': video_emotion,
            'fused_emotion': fused_emotion,
            'reasoning': reasoning,
            'story': llm_content.get('story', ''),
            'quote': llm_content.get('quote', ''),
            'video': llm_content.get('video', ''),
            'songs': llm_content.get('songs', []),
            'audio_temporal': audio_emotions_temporal,
            'video_temporal': video_emotions_temporal,
            'audio_probs_temporal': audio_preds.tolist(),
            'video_probs_temporal': video_preds.tolist(),
            'time_points': list(range(len(audio_emotions_temporal)))
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Error during processing: {e}\n{tb}")
        # Clean up on error
        for path in ['temp_video.webm']:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as ex:
                    print(f"Failed to remove {path}: {ex}")
        return jsonify({'error': f'{e}\n{tb}'})

if __name__ == '__main__':
    app.run(debug=True)