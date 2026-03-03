# EmotionAI — Complete Technical Documentation

> **Team Project — D7 | Final Year B.Tech**
> A multimodal emotion recognition system that analyzes both speech (audio) and facial expressions (video) to detect human emotions in real time.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset — Raw Data](#3-dataset--raw-data)
4. [Data Processing Pipeline](#4-data-processing-pipeline)
5. [Model Architecture & Training](#5-model-architecture--training)
6. [Model Testing & Fusion](#6-model-testing--fusion)
7. [Backend — Flask API](#7-backend--flask-api)
8. [Frontend — React UI](#8-frontend--react-ui)
9. [Cognitive Analysis Layer](#9-cognitive-analysis-layer)
10. [AI Recommendations Engine](#10-ai-recommendations-engine)
11. [Chatbot — Therapist Mode](#11-chatbot--therapist-mode)
12. [Music Playback — Deezer Integration](#12-music-playback--deezer-integration)
13. [File-by-File Reference](#13-file-by-file-reference)
14. [How to Run](#14-how-to-run)
15. [Limitations & Future Work](#15-limitations--future-work)

---

## 1. Project Overview

### What It Does
EmotionAI takes a video (uploaded or recorded live) and:
1. **Extracts audio** from the video using FFmpeg
2. **Analyzes facial expressions** frame-by-frame using a CNN + MobileNetV2 model
3. **Analyzes voice/speech** using MFCC features + a 2D CNN model
4. **Fuses both results** using weighted averaging to determine the dominant emotion
5. **Runs cognitive analysis** — temporal patterns, stability, micro-expressions, voice-face match
6. **Generates AI content** — personalized story, quote, book/song/video recommendations via Groq LLM
7. **Provides a therapist chatbot** that understands the user's emotional state

### Why This Approach?
- **Single-modality systems fail** — a person can smile while their voice reveals stress. Multimodal fusion catches what one sensor alone cannot.
- **Temporal analysis** — instead of analyzing just one frame, we track emotions over time to detect emotional arcs, micro-expressions, and regulation patterns.
- **Cognitive layer** — goes beyond basic "happy/sad" labels to provide meaningful insights like self-control scores, voice-face alignment, and emotional journey mapping.

### Emotions Detected (7 Classes)
| Index | Emotion   | RAVDESS Code |
|-------|-----------|-------------|
| 0     | Neutral   | 01, 02 (calm merged) |
| 1     | Happy     | 03 |
| 2     | Sad       | 04 |
| 3     | Angry     | 05 |
| 4     | Fearful   | 06 |
| 5     | Disgust   | 07 |
| 6     | Surprised | 08 |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER (Browser)                          │
│  Upload Video / Record Live → React Frontend (Vite + React)  │
└──────────────────────────┬──────────────────────────────────┘
                           │ POST /process
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Backend (app.py)                     │
│                                                              │
│  1. Save video → FFmpeg extracts audio (.wav)                │
│  2. Video Pipeline:                                          │
│     video → sample_frames() → MobileNetV2 features           │
│     → video_emotion_model.h5 → per-frame predictions         │
│  3. Audio Pipeline:                                          │
│     audio → extract_mfcc() → (300, 13, 1) features           │
│     → audio_emotion_model.h5 → per-window predictions        │
│  4. Fusion: weighted avg (audio 65% + video 35%)             │
│  5. Cognitive Layer: temporal analysis + reasoning            │
│  6. AI Layer: Groq LLM → story, quote, books, songs, video   │
│  7. Return JSON results                                      │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack
| Layer     | Technology |
|-----------|-----------|
| Frontend  | React 18 + Vite + Vanilla CSS |
| Backend   | Flask 3.0 + Flask-CORS |
| ML Models | TensorFlow/Keras 2.17 |
| Audio     | Librosa 0.10 (MFCC extraction) |
| Video     | OpenCV 4.8 (frame sampling) + MobileNetV2 |
| LLM       | Groq API (Llama 3.3 70B) |
| Music API | Deezer (free, no key) |
| FFmpeg    | Audio extraction from video |

---

## 3. Dataset — Raw Data

### 3.1 RAVDESS (Primary Dataset)
**Ryerson Audio-Visual Database of Emotional Speech and Song**
- **24 actors** (12 male, 12 female)
- **~7,356 files** — both audio (.wav) and video (.mp4)
- **8 emotions**: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **2 modalities**: speech and song
- **2 intensities**: normal and strong

**Filename Format**: `MM-VV-EE-AA-II-RR-OO.ext`
| Field | Meaning |
|-------|---------|
| MM | Modality (01=full-AV, 02=video-only, 03=audio-only) |
| VV | Vocal channel (01=speech, 02=song) |
| EE | Emotion (01-08) |
| AA | Intensity (01=normal, 02=strong) |
| II | Statement (01="Kids are talking", 02="Dogs are sitting") |
| RR | Repetition (01=1st, 02=2nd) |
| OO | Actor number (01-24, odd=male, even=female) |

### 3.2 CREMA-D (Supplementary)
**Crowd-sourced Emotional Multimodal Actors Dataset**
- Used to supplement training data for better generalization
- Stored in `data/CREMA-D/`

### 3.3 Data Directory Structure
```
data/
├── Actor_01/ to Actor_24/    ← Raw RAVDESS .mp4 files
├── CREMA-D/                  ← Supplementary dataset
├── organized/                ← Sorted by modality/emotion
│   ├── audio/speech/happy/
│   ├── audio/song/sad/
│   ├── video/speech/angry/
│   └── ...
├── audio_features/           ← Extracted MFCC .npy files
├── video_frames/             ← Extracted frame sequences .npy files
├── merged_train/             ← Final training split
├── merged_val/               ← Validation split
└── merged_test/              ← Test split
```

---

## 4. Data Processing Pipeline

All scripts are in `scripts/data_processing/`. They run in this order:

### Step 1: `organize_data.py`
**Purpose**: Sort raw RAVDESS files by modality and emotion.
- Reads RAVDESS filename format to extract emotion code
- Copies files into `data/organized/audio/{speech|song}/{emotion}/` and `data/organized/video/{speech|song}/{emotion}/`
- Maps emotion codes: 01→neutral, 02→calm, 03→happy, etc.

**Key Functions**:
| Function | What It Does |
|----------|-------------|
| `get_emotion_label(code)` | Maps RAVDESS code ("03") → emotion name ("happy") |
| `organize_video_files()` | Sorts .mp4 files into emotion folders |
| `organize_audio_files()` | Sorts .wav files into emotion folders |

### Step 2: `extract_audio_features.py`
**Purpose**: Convert raw audio into MFCC feature arrays.
- Loads each audio file with Librosa at SR=22050
- Extracts 13 MFCC coefficients with hop_length=512
- Pads/truncates to exactly **300 time frames**
- Saves as `.npy` files with shape `(300, 13, 1)`

**Why MFCC?**
MFCC (Mel-Frequency Cepstral Coefficients) capture the tonal quality of speech — they represent how the vocal tract shapes sound, which directly correlates with emotional expression.

**Parameters**:
| Parameter | Value | Why |
|-----------|-------|-----|
| SR | 22,050 Hz | Standard for speech processing |
| N_MFCC | 13 | Captures enough tonal detail without noise |
| HOP_LENGTH | 512 | ~23ms steps — fine enough for emotion shifts |
| N_FRAMES | 300 | Fixed sequence length for model input |

### Step 3: `extract_video_frames.py`
**Purpose**: Sample uniform frames from each video.
- Opens video with OpenCV
- Uniformly samples **16 frames** across the video duration
- Resizes each frame to **112×112 pixels**
- Normalizes pixel values to [0, 1]
- Saves as `.npy` with shape `(16, 112, 112, 3)`

**Why 16 frames?**
RAVDESS videos are ~3 seconds. 16 frames gives us ~5 FPS temporal coverage, enough to capture emotional transitions without being computationally expensive.

### Step 4: `merge_and_relabel_datasets.py`
**Purpose**: Merge RAVDESS + CREMA-D data and create train/val/test splits.
- Combines features from both datasets
- Remaps labels to a unified 7-class system (calm merged into neutral)
- Splits data: 70% train / 15% val / 15% test
- Saves splits to `data/merged_train/`, `merged_val/`, `merged_test/`

### Step 5: `download_and_prepare_fer_datasets.py` & `download_audio_datasets.py`
**Purpose**: Scripts to download FER (Facial Expression Recognition) datasets and additional audio datasets if needed. Used during initial data collection.

---

## 5. Model Architecture & Training

### 5.1 Audio Emotion Model (`audio_emotion_model.h5`)
**File**: `scripts/model_training/train_audio_model.py`
**Size**: 7.5 MB
**Architecture**: 2D CNN

```
Input: (300, 13, 1) — MFCC spectrogram
   ↓
Conv2D(32, 3×3, ReLU)
   ↓
MaxPooling2D(2×2)
   ↓
Conv2D(64, 3×3, ReLU)
   ↓
MaxPooling2D(2×2)
   ↓
Flatten
   ↓
Dense(128, ReLU)
   ↓
Dropout(0.5)
   ↓
Dense(7, Softmax) → [neutral, happy, sad, angry, fearful, disgust, surprised]
```

**Training Details**:
| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 50 (with EarlyStopping, patience=10) |
| Batch Size | 32 |
| LR Schedule | ReduceLROnPlateau (factor=0.5, patience=5) |
| Class Weights | Balanced (computed from sklearn) |

**Why 2D CNN for audio?**
The MFCC matrix (300×13) is treated like a 2D "image" where one axis is time and the other is frequency bands. CNNs excel at detecting local patterns in this structure — rising pitch (anger), monotone (sadness), etc.

### 5.2 Video Emotion Model (`video_emotion_model.h5`)
**File**: `scripts/model_training/train_video_model.py`
**Size**: 2 MB
**Architecture**: MobileNetV2 (frozen) + Dense Head

```
Input: Video Frame (112, 112, 3)
   ↓
MobileNetV2 (pretrained on ImageNet, frozen)
   ↓
GlobalAveragePooling2D → (1280,) feature vector
   ↓
Dense(128, ReLU)
   ↓
Dropout(0.5)
   ↓
Dense(7, Softmax)
```

**Training Details**:
| Setting | Value |
|---------|-------|
| Base Model | MobileNetV2 (ImageNet weights, frozen) |
| Optimizer | Adam (lr=0.0001) |
| Loss | Categorical Crossentropy |
| Epochs | 30 (with EarlyStopping) |
| Batch Size | 16 |

**Why MobileNetV2?**
- Pretrained on ImageNet — already understands edges, textures, and facial features
- Lightweight (3.4M params) — suitable for real-time inference
- Frozen base = faster training, less overfitting on small RAVDESS dataset

### 5.3 Fusion Model (`fusion_emotion.h5`)
**File**: `scripts/model_testing/fuse_models.py`
**Size**: 35 KB
**Method**: Weighted Probability Averaging

```
Final Prediction = 0.65 × Audio_Prediction + 0.35 × Video_Prediction
```

**Why 65/35 split?**
Audio consistently showed higher accuracy than video during testing. The model fusion uses late fusion (combining output probabilities) rather than early fusion (combining raw features). This is simpler, more robust, and easier to debug.

### 5.4 Additional Training Scripts
| Script | Purpose |
|--------|---------|
| `train_complete_pipeline.py` | End-to-end pipeline combining data loading + training + evaluation |
| `train_temporal_sequence.py` | LSTM-based temporal model for emotion sequences |
| `train_temporal_video.py` | Video temporal sequence model |
| `train_hybrid_multimodal.py` | Experimental hybrid fusion approach |
| `train_improved_temporal.py` | Enhanced temporal model with attention |
| `train_fer_model.py` | FER dataset-specific model training |
| `train_models.py` | Batch training orchestrator |
| `run_training_pipeline.py` | Automated pipeline runner |

---

## 6. Model Testing & Fusion

### `scripts/model_testing/test_model.py`
- Loads test split data
- Runs predictions on audio and video models separately
- Computes per-class accuracy, confusion matrix, and F1 scores
- Generates classification report

### `scripts/model_testing/test_all_models.py`
- Tests all saved model variants
- Compares accuracy across different training configurations
- Outputs a comparison table

### `scripts/model_testing/fuse_models.py`
- Loads both audio and video models
- Runs predictions on matching audio-video pairs
- Applies weighted fusion (65% audio, 35% video)
- Reports fused accuracy vs individual model accuracy

### `scripts/model_testing/real_time_demo.py`
- Real-time webcam + microphone emotion detection
- Uses OpenCV for face detection (Haar Cascade)
- Displays live emotion predictions with confidence bars

---

## 7. Backend — Flask API (`app.py`)

### 7.1 Model Loading (Startup)
On startup, `app.py` loads all three models:
```python
audio_model = keras.models.load_model('models/audio_emotion_model.h5')
video_model = keras.models.load_model('models/video_emotion_model.h5')
base_model  = MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
```

### 7.2 API Endpoints

#### `POST /process` — Main Analysis Endpoint
The core endpoint. Receives a video file and returns full analysis results.

**Flow**:
1. Save uploaded video to `temp_video.webm`
2. Extract audio: `ffmpeg` → `temp_audio.wav` (16kHz, mono, PCM)
3. **Video processing** (10%→40%):
   - `sample_frames()` extracts frame sequences over time
   - Each frame → MobileNetV2 → 1280-dim feature → Dense → 7-class prediction
   - Collects temporal predictions: `[happy, happy, sad, happy, ...]`
4. **Audio processing** (40%→80%):
   - `extract_mfcc()` computes sliding MFCC windows
   - Each window → 2D CNN → 7-class prediction
   - Collects temporal predictions: `[neutral, happy, happy, sad, ...]`
5. **Fusion** (80%→90%):
   - Timeline-based: majority vote across all temporal predictions
   - Computes: dominant emotion, confidence, stability, transition rate
6. **Cognitive reasoning** (90%):
   - Analyzes temporal patterns, modality agreement, intensity
7. **AI content generation** (90%→100%):
   - Sends emotion + reasoning to Groq LLM
   - Returns personalized story, quote, video, books, songs
8. Clean up temp files and return JSON

**Response JSON**:
```json
{
  "audio_emotion": "happy",
  "video_emotion": "happy",
  "fused_emotion": "happy",
  "reasoning": "Strong emotional consistency...",
  "story": "A luminous joy...",
  "quote": "Happiness is...",
  "video": {"title": "...", "channel": "...", "link": "...", "reason": "..."},
  "books": [{"title": "...", "author": "...", "reason": "..."}],
  "songs": [{"artist": "...", "title": "...", "link": "...", "explanation": "..."}],
  "audio_temporal": ["happy", "happy", "sad", ...],
  "video_temporal": ["happy", "neutral", "happy", ...],
  "audio_probs_temporal": [[0.1, 0.7, ...], ...],
  "video_probs_temporal": [[0.2, 0.6, ...], ...],
  "timeline_confidence": 0.72,
  "emotional_stability": 0.85,
  "transition_rate": 0.15,
  "emotion_distribution": {"happy": 12, "sad": 3, "neutral": 5}
}
```

#### `GET /status` — Progress Polling
Returns current processing progress (0-100%) and status message.
Frontend polls this every 2 seconds during analysis.

#### `POST /chat` — Therapist Chatbot
- Receives: user message, analysis results context, last 10 messages
- Sends to Groq LLM with a therapist system prompt
- Returns a warm, empathetic response (2-3 sentences, 1 emoji)

#### `GET /music/search` — Deezer Music Search
- Proxies Deezer's free API to avoid CORS
- Searches by artist + title
- Returns: track title, artist, 30-second MP3 preview URL, album artwork

### 7.3 Key Backend Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `extract_mfcc(audio_path)` | 52-101 | Extracts MFCC features from audio, returns `(N, 300, 13, 1)` windows |
| `cognitive_reasoning(...)` | 103-152 | Generates detailed cognitive analysis text |
| `generate_llm_content(...)` | 154-215 | Calls Groq LLM for personalized AI recommendations |
| `generate_fallback_content(emotion)` | 217-271 | Hardcoded fallback when LLM fails (per-emotion content) |
| `sample_frames(video_path)` | 273-306 | Samples 10 frame sequences over video duration |
| `process()` | 315-498 | Main analysis pipeline (see above) |
| `music_search()` | 500-521 | Deezer API proxy |
| `chat()` | 523-589 | Therapist chatbot endpoint |

---

## 8. Frontend — React UI

### 8.1 Project Structure
```
frontend/
├── index.html
├── vite.config.js          ← Dev server + proxy config
├── package.json
└── src/
    ├── main.jsx             ← React DOM entry point
    ├── App.jsx              ← Root component + state management
    ├── index.css            ← Global styles + design system
    ├── components/
    │   ├── Navbar.jsx        ← Top navigation bar
    │   ├── RecordingPanel.jsx← Upload/record video panel
    │   ├── ProcessingLoader.jsx ← Progress bar during analysis
    │   ├── EmotionCards.jsx  ← Emotion result cards (audio/video/fused)
    │   ├── TemporalChart.jsx ← Timeline emotion graphs
    │   ├── CognitiveInsights.jsx ← Advanced temporal metrics
    │   ├── AIContent.jsx     ← AI recommendations (stories, songs, video)
    │   └── Chatbot.jsx       ← Therapist chatbot panel
    └── hooks/
        ├── useMediaRecorder.js ← Camera/mic recording hook
        └── useAnalysis.js     ← Video upload + polling hook
```

### 8.2 Component Details

#### `App.jsx` — Root Component
- Manages global state: `results`, `processing`, `chatOpen`, `chatKey`
- When results arrive → shows EmotionCards, TemporalChart, CognitiveInsights, AIContent
- `handleReset()` — clears results + increments `chatKey` (resets chatbot messages)
- Hero section with project description and accuracy disclaimer

#### `Navbar.jsx` — Navigation Bar
- Shows "Team Project — D7" label
- Chat toggle button (opens/closes Chatbot panel)
- Sticky top navigation

#### `RecordingPanel.jsx` — Video Input
- **Upload mode**: Click to open file explorer, select video file
- **Record mode**: Toggle to record live via webcam + microphone
  - Shows camera mirror preview using `useMediaRecorder` hook
  - Records video + audio simultaneously
  - Saves as WebM blob
- Includes bias disclaimer about environmental factors

#### `ProcessingLoader.jsx` — Progress Visualization
- Polls `GET /status` every 2 seconds
- Displays progress percentage (0-100%)
- Shows current stage name (Extracting Audio → Processing Video → Processing Audio → Cognitive Analysis → Generating AI Response)

#### `EmotionCards.jsx` — Result Cards
- Three cards: Audio Emotion, Video Emotion, Fused Emotion
- Each shows the detected emotion with emoji and confidence

#### `TemporalChart.jsx` — Timeline Graphs
- Renders emotion probability curves over time
- Shows how each emotion's probability changes across the video
- Separate charts for audio and video modalities
- Curved lines (not linear) — spikes show when AI detects a particular emotion strongly

#### `CognitiveInsights.jsx` — Advanced Temporal Metrics
6 metrics computed in the browser from temporal data:

| Metric | Label | What It Shows |
|--------|-------|---------------|
| Mood Trend | Getting Better / Dipping Down / Steady | Overall emotional trajectory (positive → negative over time?) |
| Energy Level | Peaked early/mid/late | When emotional intensity was highest |
| Voice-Face Match | 0-100% | Whether audio and video emotions agreed at each time point |
| Self-Control | 0-100% | How quickly high-intensity emotions returned to baseline |
| Your Journey | Started X → ended Y | Dominant emotion transition across the session |
| Quick Flickers | N detected | Rapid A→B→A micro-expression patterns (fleeting real emotions) |

#### `AIContent.jsx` — AI Recommendations
- **Read Story** — LLM-generated narrative about the emotional journey
- **Short Reads** — 2-3 book recommendations with reasons
- **Quote** — Inspirational quote matching the emotion
- **Watch This** — YouTube video recommendation (embedded player if valid ID, link button otherwise)
- **Playlist** — Song recommendations with Deezer 30-second preview playback (album art, play/pause, progress bar)

#### `Chatbot.jsx` — Therapist Panel
- Fixed bottom-right panel, toggleable from Navbar
- Maintains last 10 messages in memory
- Sends context (analysis results) to backend `/chat` endpoint
- Therapist persona — warm, empathetic, validates feelings
- `key` prop resets messages on new analysis

### 8.3 Custom Hooks

#### `useMediaRecorder.js`
- Requests camera + microphone permissions via `navigator.mediaDevices.getUserMedia`
- Uses callback ref to attach stream to video element (mirror preview)
- Records using MediaRecorder API
- Returns recorded blob for upload

#### `useAnalysis.js`
- Uploads video file to `POST /process`
- Polls `GET /status` for progress updates
- Returns results when complete

### 8.4 Vite Configuration (`vite.config.js`)
```javascript
proxy: {
  '/process': 'http://localhost:5000',
  '/status':  'http://localhost:5000',
  '/chat':    'http://localhost:5000',
  '/music':   'http://localhost:5000',
}
```
All API calls from the React dev server (port 5173) are proxied to the Flask backend (port 5000).

---

## 9. Cognitive Analysis Layer

The cognitive layer is what makes this system unique compared to standard emotion detection models. Most systems just predict "happy" or "sad" — we go deeper.

### What Standard Models Do
```
Video → CNN → "Happy" (done)
```

### What Our System Does
```
Video → CNN → [happy, happy, sad, happy, neutral, happy, sad, ...] (temporal sequence)
                    ↓
        Cognitive Analysis Engine
                    ↓
  ┌─────────────────────────────────────────────────┐
  │ • Mood is improving over time (+0.3 shift)       │
  │ • Energy peaked at midpoint during "angry"       │
  │ • Voice and face matched 73% — mostly genuine    │
  │ • Self-control: 85% — quick recovery from highs  │
  │ • 2 micro-expression flickers detected           │
  │ • Journey: started neutral → ended happy         │
  └─────────────────────────────────────────────────┘
```

### How Each Metric Works

**Valence Mapping** (negative ←→ positive):
```
angry: -0.8 | disgust: -0.7 | sad: -0.6 | fearful: -0.5
neutral: 0.0 | surprised: +0.4 | happy: +0.9
```

**Arousal Mapping** (calm ←→ excited):
```
neutral: 0.1 | sad: 0.2 | disgust: 0.4 | happy: 0.6
fearful: 0.7 | surprised: 0.8 | angry: 0.9
```

These mappings are from Russell's Circumplex Model of Affect, a well-established framework in affective computing.

---

## 10. AI Recommendations Engine

### How It Works
1. After fusion determines the dominant emotion, the system builds a prompt:
   ```
   Primary Emotion: happy
   Cognitive Analysis: Strong consistency at 85%, high stability...
   Audio Timeline: happy, happy, neutral, happy
   Video Timeline: happy, neutral, happy, happy
   ```
2. Sends to **Groq API** (Llama 3.3 70B at temperature=0.8)
3. LLM returns JSON with: story, quote, video, books, songs
4. If LLM fails → `generate_fallback_content()` provides hardcoded content per emotion

### Fallback Content
Each emotion has pre-written fallback content with:
- Real YouTube video IDs (`youtu.be/VIDEO_ID` format) for reliable embedding
- Curated book recommendations
- Song recommendations with valid YouTube links

---

## 11. Chatbot — Therapist Mode

### System Prompt
```
You are a warm, empathetic AI therapist.
The person you're talking to just had their emotions analyzed.

How to behave:
- You KNOW this person's emotional state — talk naturally
- Do NOT repeat analysis data unless specifically asked
- For greetings: reply with a short warm message (1 line)
- Keep responses SHORT (2-3 sentences max)
- Use exactly 1 emoji per response
- Validate feelings, offer gentle perspective
```

### Memory
- Last 10 messages are sent as conversation history
- Full analysis results are sent as context
- `chatKey` prop resets messages when user runs a new analysis

---

## 12. Music Playback — Deezer Integration

### Why Deezer?
- **Free API** — no API key or authentication required
- **30-second previews** — direct MP3 URLs, no CORS issues
- **Album artwork** — enhances the UI with visual feedback

### Flow
```
User clicks Play on a song
        ↓
Frontend: GET /music/search?q=Pharrell+Happy
        ↓
Backend: proxies to https://api.deezer.com/search?q=Pharrell+Happy&limit=1
        ↓
Returns: { title, artist, preview (MP3 URL), album_art }
        ↓
Frontend: plays preview via HTML5 <audio> element
```

---

## 13. File-by-File Reference

### Root Directory
| File | Purpose |
|------|---------|
| `app.py` | Flask backend — all API endpoints, model inference, LLM integration |
| `requirements.txt` | Python dependencies (numpy, tensorflow, librosa, flask, groq, etc.) |
| `ffmpeg.exe` | Audio extraction tool (bundled for Windows) |
| `inspect_models.py` | Utility to inspect model layer shapes and weights |
| `test_audio.py` | Quick audio model test script |

### `models/`
| File | Size | Purpose |
|------|------|---------|
| `audio_emotion_model.h5` | 7.5 MB | 2D CNN for MFCC-based audio emotion detection |
| `video_emotion_model.h5` | 2.0 MB | MobileNetV2 + Dense head for facial emotion detection |
| `fusion_emotion.h5` | 35 KB | Fusion model weights |
| `haarcascade_frontalface_default.xml` | 930 KB | OpenCV face detection cascade |

### `scripts/data_processing/`
| File | Purpose |
|------|---------|
| `organize_data.py` | Sorts RAVDESS files by modality/emotion into organized dirs |
| `extract_audio_features.py` | Converts audio → MFCC `.npy` files (300, 13, 1) |
| `extract_video_frames.py` | Converts video → frame `.npy` files (16, 112, 112, 3) |
| `merge_and_relabel_datasets.py` | Merges RAVDESS + CREMA-D, creates train/val/test splits |
| `download_and_prepare_fer_datasets.py` | Downloads FER dataset for additional training |
| `download_audio_datasets.py` | Downloads supplementary audio datasets |

### `scripts/model_training/`
| File | Purpose |
|------|---------|
| `train_audio_model.py` | Trains the 2D CNN audio model on MFCC features |
| `train_video_model.py` | Trains the MobileNetV2 video model on frame features |
| `train_complete_pipeline.py` | Full end-to-end training pipeline |
| `train_temporal_sequence.py` | LSTM-based temporal sequence training |
| `train_temporal_video.py` | Temporal video model training |
| `train_hybrid_multimodal.py` | Experimental hybrid multimodal approach |
| `train_improved_temporal.py` | Enhanced temporal model with improvements |
| `train_fer_model.py` | FER-specific model training |
| `train_models.py` | Batch model training orchestrator |
| `run_training_pipeline.py` | Automated pipeline runner |

### `scripts/model_testing/`
| File | Purpose |
|------|---------|
| `fuse_models.py` | Evaluates weighted fusion of audio + video models |
| `test_model.py` | Tests individual model performance with metrics |
| `test_all_models.py` | Compares all model variants |
| `real_time_demo.py` | Live webcam + mic emotion detection demo |

### `scripts/utils/`
| File | Purpose |
|------|---------|
| `emotion_models.py` | CNN-LSTM model class definitions (Audio, Video, Multimodal) |
| `data_loader.py` | Data loading utilities for training scripts |
| `face_detector.py` | OpenCV face detection wrapper (Haar + DNN) |
| `multimodal_emotion.py` | Multimodal fusion model utilities |

### `frontend/src/components/`
| File | Purpose |
|------|---------|
| `Navbar.jsx` | Top nav bar with project title and chat toggle |
| `RecordingPanel.jsx` | Video upload + live recording interface |
| `ProcessingLoader.jsx` | Progress bar with stage labels |
| `EmotionCards.jsx` | Audio/Video/Fused emotion result cards |
| `TemporalChart.jsx` | Emotion probability timeline graphs |
| `CognitiveInsights.jsx` | 6 advanced temporal metrics display |
| `AIContent.jsx` | Story, quote, books, video, songs with Deezer player |
| `Chatbot.jsx` | Therapist chatbot with message history |

### `frontend/src/hooks/`
| File | Purpose |
|------|---------|
| `useMediaRecorder.js` | Camera/mic permission, recording, mirror preview |
| `useAnalysis.js` | Video upload to backend + progress polling |

---

## 14. How to Run

### Prerequisites
- Python 3.10+
- Node.js 18+
- FFmpeg (bundled as `ffmpeg.exe` or install separately)
- GROQ_API_KEY environment variable set

### Backend
```bash
cd 4-2-project
pip install -r requirements.txt
python app.py
# Server starts on http://localhost:5000
```

### Frontend
```bash
cd 4-2-project/frontend
npm install
npm run dev
# Dev server starts on http://localhost:5173
```

### Training (Optional — models are pre-trained)
```bash
# Step 1: Organize data
python scripts/data_processing/organize_data.py

# Step 2: Extract features
python scripts/data_processing/extract_audio_features.py
python scripts/data_processing/extract_video_frames.py

# Step 3: Train models
python scripts/model_training/train_audio_model.py
python scripts/model_training/train_video_model.py

# Step 4: Test fusion
python scripts/model_testing/fuse_models.py
```

---

## 15. Limitations & Future Work

### Current Limitations
- **Training data**: ~25GB from RAVDESS + CREMA-D — may not generalize to all demographics
- **Camera/lighting bias**: Poor camera quality and lighting significantly affect video predictions
- **7 emotions only**: Real human emotions are far more nuanced
- **30-second music previews**: Deezer provides only 30-second clips

### Future Improvements
- Train on larger, more diverse datasets (AffectNet, FER2013+, MELD)
- Add real-time streaming analysis (WebSocket instead of file upload)
- Implement attention mechanisms for temporal analysis
- Add language/text sentiment analysis as a third modality
- Deploy as a containerized microservice (Docker + Kubernetes)
- Use cases: HR screening, therapy sessions, customer feedback, education

---

> **Note**: This system is not 100% accurate. The goal is to make machines understand humans in various scenarios. The base model can be adapted for use cases like HR interview screening, customer sentiment analysis, therapy support, and educational engagement tracking.
