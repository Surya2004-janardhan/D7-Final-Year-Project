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
# below trying to set env of groq
# GROQ_API_KEY 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("Groq client initialized", os.getenv("GROQ_API_KEY"))
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
    """Enhanced cognitive reasoning with more detailed analysis."""
    reasoning = []

    # Basic agreement analysis
    if audio_emotion == video_emotion:
        reasoning.append(f"Both audio and video modalities strongly agree on {audio_emotion}.")
    else:
        reasoning.append(f"Modalities show disagreement: audio suggests {audio_emotion} while video indicates {video_emotion}. Fusion resulted in {fused_emotion} as the most balanced interpretation.")

    # Confidence analysis with scores
    audio_conf = np.max(np.mean(audio_preds, axis=0))
    video_conf = np.max(np.mean(video_preds, axis=0))
    reasoning.append(f"Confidence levels: Audio {audio_conf:.2f}, Video {video_conf:.2f}. Higher confidence indicates more reliable detection.")

    # Temporal consistency analysis
    audio_consistency = len(set([EMOTIONS_7[np.argmax(p)] for p in audio_preds])) / len(audio_preds)
    video_consistency = len(set([EMOTIONS_7[np.argmax(p)] for p in video_preds])) / len(video_preds)
    reasoning.append(f"Temporal stability: Audio consistency {audio_consistency:.2f}, Video consistency {video_consistency:.2f}. Lower values indicate more emotional fluctuation.")

    # Emotion intensity analysis
    audio_intensity = np.mean([np.max(p) for p in audio_preds])
    video_intensity = np.mean([np.max(p) for p in video_preds])
    reasoning.append(f"Emotional intensity: Audio {audio_intensity:.2f}, Video {video_intensity:.2f}. Higher values suggest stronger emotional expression.")

    # Contextual interpretation
    if fused_emotion in ['angry', 'fearful', 'sad']:
        reasoning.append("Detected negative emotion cluster. This may indicate stress, concern, or dissatisfaction. Consider environmental factors and personal context.")
    elif fused_emotion in ['happy', 'surprised']:
        reasoning.append("Positive emotional state detected. This suggests engagement, satisfaction, or pleasant surprise. The person appears to be in a favorable emotional state.")
    elif fused_emotion == 'neutral':
        reasoning.append("Neutral emotional state observed. This could indicate calmness, concentration, or emotional restraint. May also suggest controlled or professional demeanor.")
    elif fused_emotion == 'disgust':
        reasoning.append("Disgust detected. This emotion often relates to aversion or strong disapproval. Consider recent experiences or environmental factors.")

    # Modality reliability assessment
    if abs(audio_conf - video_conf) > 0.3:
        if audio_conf > video_conf:
            reasoning.append("Audio modality appears more reliable. This could be due to clear vocal expression or poor video quality.")
        else:
            reasoning.append("Video modality appears more reliable. This might indicate clear facial expressions or audio recording issues.")
    else:
        reasoning.append("Both modalities show balanced reliability, suggesting consistent emotional expression across channels.")

    # Temporal pattern analysis
    audio_changes = sum(1 for i in range(1, len(audio_preds)) if np.argmax(audio_preds[i]) != np.argmax(audio_preds[i-1]))
    video_changes = sum(1 for i in range(1, len(video_preds)) if np.argmax(video_preds[i]) != np.argmax(video_preds[i-1]))
    reasoning.append(f"Emotional transitions: Audio {audio_changes}, Video {video_changes}. Frequent changes may indicate emotional volatility or complex feelings.")

    return " ".join(reasoning)

def generate_llm_content(fused_emotion, reasoning, audio_temporal, video_temporal):
    """Generate personalized story, quote, video, and songs using Groq LLM."""
    prompt = f"""
Based on the emotion analysis results:

Primary Emotion Detected: {fused_emotion}
Cognitive Analysis: {reasoning}
Audio Emotional Timeline: {', '.join(audio_temporal)}
Video Emotional Timeline: {', '.join(video_temporal)}

Please generate highly personalized content that directly relates to this specific emotional state and analysis:

1. A short, personalized story (3-4 sentences) that captures the emotional journey shown in the timeline and explains the fusion result.

2. An inspirational quote specifically tailored to someone experiencing this emotion, considering the cognitive analysis insights.

3. A YouTube video recommendation with specific title, creator/channel, and detailed explanation of why it would help someone in this emotional state.

4. 3-4 song recommendations that are currently popular/relevant (2024-2025 era), with specific artist names, song titles, and brief explanations of why each song matches this emotional profile.

Format the response as valid JSON with keys: story, quote, video, songs (as array of strings with artist and title).
Ensure the content is empathetic, supportive, and directly addresses the detected emotional state and cognitive insights.
"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": 1000
        }
        response = requests.post(GROQ_URL, headers=headers, json=data)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            # Clean up JSON response
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            return json.loads(content)
        else:
            print(f"Groq API error: {response.status_code} - {response.text}")
            return generate_fallback_content(fused_emotion)
    except Exception as e:
        print(f"LLM error: {e}")
        return generate_fallback_content(fused_emotion)

def generate_fallback_content(fused_emotion):
    """Generate fallback content when LLM fails."""
    fallbacks = {
        'happy': {
            'story': 'A person\'s face lit up with genuine joy as they shared exciting news, their laughter echoing with pure delight. The moment captured both the sparkle in their eyes and the warmth in their voice.',
            'quote': '"Joy is the simplest form of gratitude." - Karl Barth',
            'video': '"The Science of Happiness" by Yale University - A fascinating exploration of what truly makes people happy, perfect for understanding and amplifying positive emotions.',
            'songs': ['"Happy" by Pharrell Williams - An upbeat anthem celebrating joy', '"Can\'t Stop the Feeling!" by Justin Timberlake - Infectious positivity', '"Good as Hell" by Lizzo - Self-love and confidence booster']
        },
        'sad': {
            'story': 'In a quiet moment, tears welled up as memories flooded back. The weight of unspoken emotions showed in both the downward gaze and the soft, trembling voice.',
            'quote': '"The emotion that can break your heart is sometimes the very one that heals it." - Nicholas Sparks',
            'video': '"The Power of Vulnerability" by Brené Brown - Understanding sadness and emotional healing',
            'songs': ['"Someone Like You" by Adele - Processing grief and loss', '"Hurt" by Johnny Cash - Deep emotional resonance', '"The Night We Met" by Lord Huron - Reflective melancholy']
        },
        'angry': {
            'story': 'Frustration built up as unfairness struck, showing in the clenched jaw and raised voice. The raw emotion demanded attention and understanding.',
            'quote': '"Anger is an acid that can do more harm to the vessel in which it is stored than to anything on which it is poured." - Mark Twain',
            'video': '"How to Control Your Anger" by The School of Life - Practical strategies for managing anger',
            'songs': ['"Break Stuff" by Limp Bizkit - Cathartic anger release', '"Killing in the Name" by Rage Against the Machine - Frustration outlet', '"Express Yourself" by Madonna - Channeling anger into self-expression']
        },
        'fearful': {
            'story': 'Uncertainty clouded the eyes as anxiety took hold, the shaky voice betraying inner turmoil. The body language spoke of protection and hesitation.',
            'quote': '"The only way to deal with fear is to face it head on." - Mark Twain',
            'video': '"Conquering Fear" by Tim Ferriss - Practical techniques for overcoming anxiety',
            'songs': ['"Fearless" by Taylor Swift - Overcoming fear', '"Brave" by Sara Bareilles - Finding courage', '"Roar" by Katy Perry - Empowerment through fear']
        },
        'neutral': {
            'story': 'A composed presence filled the space, with steady gaze and measured tone. The calm exterior suggested thoughtful contemplation.',
            'quote': '"Peace is not absence of conflict, it is the ability to cope with it." - Mahatma Gandhi',
            'video': '"The Benefits of Mindfulness" by Headspace - Finding peace through meditation',
            'songs': ['"Imagine" by John Lennon - Vision of peace', '"What a Wonderful World" by Louis Armstrong - Appreciation of simplicity', '"Blackbird" by The Beatles - Finding strength in stillness']
        },
        'surprised': {
            'story': 'Eyes widened in unexpected delight as surprise unfolded, the gasp escaping before the smile could form. The moment captured pure, unfiltered reaction.',
            'quote': '"The world is full of magic things, patiently waiting for our senses to grow sharper." - W.B. Yeats',
            'video': '"The Psychology of Surprise" by Vsauce - Understanding the science of surprise',
            'songs': ['"Surprise" by G-Eazy ft. Blackbear - Modern take on unexpected feelings', '"Wow" by Post Malone - Expressing amazement', '"Speechless" by Dan + Shay - Overwhelming positive surprise']
        },
        'disgust': {
            'story': 'A look of aversion crossed the face as something disagreeable presented itself. The wrinkled nose and turned head spoke volumes.',
            'quote': '"Disgust is the appropriate response to most situations in life." - Anonymous',
            'video': '"Understanding Disgust" by Crash Course Psychology - The science behind this emotion',
            'songs': ['"U + Me = Love" by P!nk - Finding beauty despite disgust', '"Shake It Off" by Taylor Swift - Moving past negative feelings', '"Toxic" by Britney Spears - Recognizing unhealthy situations']
        }
    }

    return fallbacks.get(fused_emotion, fallbacks['neutral'])

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