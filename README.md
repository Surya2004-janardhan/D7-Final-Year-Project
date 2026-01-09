# Emotion Detection from Audio and Video using CNN-LSTM

A comprehensive deep learning project for detecting emotions from audio and video using convolutional and recurrent neural networks.

## Dataset

**RAVDESS (Ryerson Audio-Visual Emotion Database and Speech Dataset)**
- 24 professional actors
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgusted, surprised
- Audio (speech & song) and Video (speech & song) modalities
- Total: ~1440 files

## Project Structure

```
├── data/
│   ├── organized/              # Processed data structure
│   │   ├── audio/
│   │   │   ├── speech/
│   │   │   │   └── [emotions]/
│   │   │   └── song/
│   │   │       └── [emotions]/
│   │   └── video/
│   │       ├── speech/
│   │       │   └── [emotions]/
│   │       └── song/
│   │           └── [emotions]/
│   └── [raw zip files]
│
├── organize_data.py            # Data organization script
├── data_loader.py              # Data loading and preprocessing
├── emotion_models.py           # Model architectures
├── train_models.py             # Training pipeline
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

### 1. Clone or setup the project

```bash
cd 4-2-project
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Setup & Data Preparation

### 1. Extract RAVDESS Data

Run the organization script to extract and organize raw data:

```bash
python organize_data.py
```

This creates:
- `data/organized/audio/speech/[emotions]/`
- `data/organized/audio/song/[emotions]/`
- `data/organized/video/speech/[emotions]/`
- `data/organized/video/song/[emotions]/`

### 2. Verify Data Structure

```bash
# Check organized data
ls -la data/organized/audio/speech/
ls -la data/organized/video/speech/
```

## Models Overview

### 1. **Audio Emotion CNN-LSTM**
- **Input:** MFCC features (13 coefficients)
- **Architecture:**
  - Conv2D layers (feature extraction)
  - LSTM layers (temporal modeling)
  - Dense layers (classification)
- **Output:** 8 emotion classes

### 2. **Video Emotion 3D CNN-LSTM**
- **Input:** Video frames (30 frames, 224×224×3)
- **Architecture:**
  - 3D Conv layers (spatiotemporal features)
  - LSTM layers (temporal dependencies)
  - Dense layers (classification)
- **Output:** 8 emotion classes

### 3. **Multimodal Emotion Fusion**
- **Inputs:** Audio MFCC + Video frames
- **Architecture:**
  - Parallel CNN-LSTM branches
  - Feature fusion (concatenation)
  - Final dense layers
- **Output:** 8 emotion classes

## Training

### Quick Start - Train All Models

```bash
python train_models.py
```

This will:
1. Load and preprocess RAVDESS data
2. Train audio models (speech & song)
3. Train video models (speech & song)
4. Save trained models in `models/` directory
5. Generate training plots in `plots/` directory

### Custom Training

```python
from emotion_models import AudioEmotionCNNLSTM
from data_loader import RAVDESSDataLoader

# Load data
loader = RAVDESSDataLoader()
X, y = loader.load_audio_dataset('speech')

# Build model
model = AudioEmotionCNNLSTM(num_emotions=8)
model.build_model((13, 216))

# Train
history = model.train(X_train, y_train, X_val, y_val, epochs=50)
```

## Feature Extraction

### Audio Features (MFCC)
- **Sampling rate:** 22,050 Hz
- **MFCC coefficients:** 13
- **Time steps:** Variable (padded to 216)

### Video Features
- **Frame extraction:** 30 frames per video
- **Frame size:** 224×224 pixels
- **Normalization:** Scaled to [0, 1]

## Hyperparameters

| Parameter | Audio | Video |
|-----------|-------|-------|
| Batch Size | 16 | 8 |
| Epochs | 50 | 30 |
| Learning Rate | 0.001 (default) | 0.001 (default) |
| Dropout | 0.5 | 0.5 |
| LSTM Units | 128 → 64 | 128 → 64 |

## Outputs

- **Models:** Saved in `models/` directory
  - `audio_emotion_speech.h5`
  - `audio_emotion_song.h5`
  - `video_emotion_speech.h5`
  - `video_emotion_song.h5`

- **Plots:** Saved in `plots/` directory
  - Training/validation accuracy and loss curves
  - Model performance visualization

## Performance Metrics

After training, you'll get:
- Training accuracy
- Validation accuracy
- Loss curves
- Per-emotion classification performance

## Emotion Classes

```
0: Neutral
1: Calm
2: Happy
3: Sad
4: Angry
5: Fearful
6: Disgusted
7: Surprised
```

## Advanced Usage

### Load Pre-trained Model

```python
from tensorflow.keras.models import load_model

model = load_model('models/audio_emotion_speech.h5')
predictions = model.predict(X_test)
```

### Extract and Save Features

```python
from data_loader import RAVDESSDataLoader

loader = RAVDESSDataLoader()
X, y = loader.load_audio_dataset('speech')
loader.save_features(X, y, 'features_audio.pkl')

# Load later
X_loaded, y_loaded = loader.load_features('features_audio.pkl')
```

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- scikit-learn 1.3+
- librosa 0.10+ (audio processing)
- OpenCV 4.8+ (video processing)
- NumPy, Pandas

See `requirements.txt` for complete list.

## Troubleshooting

### Out of Memory Error
- Reduce batch size in `train_models.py`
- Reduce number of frames for video: `n_frames=15`

### No Audio/Video Files Found
- Ensure `organize_data.py` completed successfully
- Check that `data/organized/` directory exists with subdirectories

### Model Not Training
- Verify data is loaded: check console output
- Ensure GPU is available: `tf.config.list_physical_devices('GPU')`

## Future Improvements

1. **Data Augmentation**
   - Audio: Time stretching, pitch shifting
   - Video: Rotation, flipping, zoom

2. **Ensemble Methods**
   - Combine audio + video predictions
   - Weighted ensemble fusion

3. **Attention Mechanisms**
   - Add attention layers to LSTM
   - Focus on important frames/features

4. **Transfer Learning**
   - Use pre-trained models (ResNet, VGG, etc.)
   - Fine-tune for emotion detection

## References

- RAVDESS Dataset: https://zenodo.org/record/1188976
- Goodfellow et al. (2015): Deep Learning
- Graves & Schmidhuber (2005): Framewise phoneme classification with bidirectional LSTM

## Author

Emotion Detection Project - CNN/LSTM based multimodal emotion recognition

## License

MIT License
