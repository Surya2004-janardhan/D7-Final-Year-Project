"""
Complete Training Pipeline for Emotion Recognition System

This script runs the full training pipeline:
1. Train FER base model on FER-2013 dataset
2. Train Temporal Video model on RAVDESS videos (using face detection)
3. (Optional) Train late fusion model combining audio + video

Prerequisites:
- FER-2013 dataset in fer-data/train and fer-data/test
- RAVDESS data in data/Actor_XX folders
- Pre-trained audio models in models/ folder
"""

import os
import sys
import argparse
from datetime import datetime


def train_fer_model(epochs=50, batch_size=64):
    """Train FER-2013 base model"""
    print("\n" + "=" * 70)
    print("STEP 1: Training FER Base Model on FER-2013 Dataset")
    print("=" * 70)
    
    from train_fer_model import FERModel, plot_training_history
    
    FER_TRAIN_DIR = 'fer-data/train'
    FER_TEST_DIR = 'fer-data/test'
    
    if not os.path.exists(FER_TRAIN_DIR):
        print(f"ERROR: FER training directory not found: {FER_TRAIN_DIR}")
        return False
    
    # Build and train
    fer_model = FERModel(num_classes=8, input_shape=(48, 48, 1))
    fer_model.build_model()
    
    print("\nModel Summary:")
    fer_model.model.summary()
    
    history = fer_model.train(
        train_dir=FER_TRAIN_DIR,
        val_dir=FER_TEST_DIR,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Plot and save
    plot_training_history(history, 'plots/fer_training.png')
    fer_model.save('models/fer_base_model.h5')
    
    # Final evaluation
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        FER_TEST_DIR,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )
    
    loss, acc = fer_model.model.evaluate(test_gen)
    print(f"\n[FER Model] Final Test Accuracy: {acc*100:.2f}%")
    
    return True


def train_temporal_video_model(epochs=50, batch_size=8, n_frames=16):
    """Train temporal video model on RAVDESS"""
    print("\n" + "=" * 70)
    print("STEP 2: Training Temporal Video Model on RAVDESS")
    print("=" * 70)
    
    from train_temporal_video import (
        TemporalEmotionModel, 
        load_ravdess_videos, 
        plot_training_history,
        NUM_EMOTIONS
    )
    from sklearn.model_selection import train_test_split
    
    # Load data
    print("\nLoading RAVDESS video data with face extraction...")
    X, y = load_ravdess_videos(data_dir='data', n_frames=n_frames)
    
    if len(X) == 0:
        print("ERROR: No video data loaded!")
        return False
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    model = TemporalEmotionModel(
        num_emotions=NUM_EMOTIONS,
        n_frames=n_frames,
        input_shape=(48, 48, 1)
    )
    
    # Use FER backbone if available
    fer_path = 'models/fer_base_model.h5'
    if os.path.exists(fer_path):
        print(f"\nUsing pre-trained FER backbone from {fer_path}")
        model.build_with_fer_backbone(fer_path)
    else:
        print("\nBuilding fresh CNN-LSTM model (no FER backbone)")
        model.build_cnn_lstm_model()
    
    print("\nModel Summary:")
    model.model.summary()
    
    # Train
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Plot and save
    plot_training_history(history, 'plots/temporal_video_training.png')
    model.save('models/temporal_video_emotion.h5')
    
    # Final evaluation
    loss, acc = model.model.evaluate(X_val, y_val)
    print(f"\n[Temporal Video Model] Final Val Accuracy: {acc*100:.2f}%")
    
    return True


def print_summary():
    """Print summary of trained models"""
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - MODEL SUMMARY")
    print("=" * 70)
    
    models_dir = 'models'
    models_info = [
        ('fer_base_model.h5', 'FER Base Model (Face Emotion Recognition)'),
        ('temporal_video_emotion.h5', 'Temporal Video Model (2D CNN + LSTM)'),
        ('audio_emotion_speech.h5', 'Audio Speech Model (Pre-existing)'),
        ('audio_emotion_song.h5', 'Audio Song Model (Pre-existing)')
    ]
    
    print("\nAvailable Models:")
    print("-" * 50)
    
    for model_file, description in models_info:
        path = os.path.join(models_dir, model_file)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✓ {description}")
            print(f"  Path: {path}")
            print(f"  Size: {size:.2f} MB\n")
        else:
            print(f"✗ {description} - NOT FOUND\n")
    
    print("\nUsage:")
    print("-" * 50)
    print("To use the multimodal system:")
    print("  from multimodal_emotion import MultimodalEmotionRecognizer")
    print("  recognizer = MultimodalEmotionRecognizer()")
    print("  recognizer.load_models()")
    print("  emotion, conf, preds = recognizer.predict_multimodal('video.mp4')")


def main():
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Models')
    parser.add_argument('--fer-only', action='store_true', help='Train only FER model')
    parser.add_argument('--video-only', action='store_true', help='Train only temporal video model')
    parser.add_argument('--epochs-fer', type=int, default=50, help='Epochs for FER training')
    parser.add_argument('--epochs-video', type=int, default=50, help='Epochs for video training')
    parser.add_argument('--batch-fer', type=int, default=64, help='Batch size for FER')
    parser.add_argument('--batch-video', type=int, default=8, help='Batch size for video')
    parser.add_argument('--n-frames', type=int, default=16, help='Number of frames for video')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"\nTraining started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    success = True
    
    # Train FER model
    if not args.video_only:
        if not train_fer_model(epochs=args.epochs_fer, batch_size=args.batch_fer):
            print("WARNING: FER model training failed or skipped")
            if args.fer_only:
                success = False
    
    # Train temporal video model
    if not args.fer_only:
        if not train_temporal_video_model(
            epochs=args.epochs_video, 
            batch_size=args.batch_video,
            n_frames=args.n_frames
        ):
            print("WARNING: Temporal video model training failed or skipped")
            success = False
    
    # Print summary
    print_summary()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\nTraining completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
