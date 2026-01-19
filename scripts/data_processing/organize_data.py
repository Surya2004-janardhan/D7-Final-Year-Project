"""
RAVDESS Data Organization Script
Organizes RAVDESS data by modality (audio/video), content (song/speech), and emotion
RAVDESS Filename Format: MM-VV-EE-AA-II-RR-OO.ext
MM = Modality (01=speech, 02=song)
VV = Vocal channel (01=speech, 02=song)
EE = Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgusted, 08=surprised)
AA = Intensity (01=normal, 02=strong)
II = Statement (01=kids, 02=too)
RR = Repetition (01=1st, 02=2nd)
OO = Actor (01-24)
"""

import os
import shutil
from pathlib import Path

# Emotion mapping
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgusted',
    '08': 'surprised'
}

# Modality mapping
MODALITY = {
    '01': 'speech',
    '02': 'song'
}

def get_emotion_label(code):
    return EMOTIONS.get(code, 'unknown')

def get_modality_label(code):
    return MODALITY.get(code, 'unknown')

def organize_video_files():
    """Organize video files"""
    data_path = Path('C:\\Users\\chint\\Desktop\\4-2-project\\data')
    organized_path = data_path / 'organized'
    
    # Process Actor directories (video files)
    for actor_dir in data_path.glob('Actor_*'):
        if actor_dir.is_dir():
            actor_num = actor_dir.name.replace('Actor_', '').zfill(2)
            
            for video_file in actor_dir.glob('*.mp4'):
                parts = video_file.stem.split('-')
                if len(parts) >= 8:
                    modality = parts[0]  # First digit (but all should be 01 for video)
                    vocal_channel = parts[1]  # speech=01, song=02
                    emotion_code = parts[2]
                    
                    emotion = get_emotion_label(emotion_code)
                    modality_type = get_modality_label(vocal_channel)
                    
                    # Create destination
                    dest_dir = organized_path / 'video' / modality_type / emotion
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    dest_file = dest_dir / f"actor_{actor_num}_{video_file.name}"
                    shutil.copy2(video_file, dest_file)
                    print(f"Organized: {video_file.name} -> {modality_type}/{emotion}/")

def organize_audio_files():
    """Organize audio files from extracted archives"""
    data_path = Path('C:\\Users\\chint\\Desktop\\4-2-project\\data')
    organized_path = data_path / 'organized'
    
    # Look for audio files (mp3 or wav) in all subdirectories
    for audio_file in data_path.rglob('*.wav'):
        if 'organized' not in str(audio_file):
            parts = audio_file.stem.split('-')
            if len(parts) >= 8:
                vocal_channel = parts[1]
                emotion_code = parts[2]
                
                emotion = get_emotion_label(emotion_code)
                modality_type = get_modality_label(vocal_channel)
                
                dest_dir = organized_path / 'audio' / modality_type / emotion
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                dest_file = dest_dir / audio_file.name
                shutil.copy2(audio_file, dest_file)
                print(f"Organized: {audio_file.name} -> audio/{modality_type}/{emotion}/")
    
    # Also look for mp3 files
    for audio_file in data_path.rglob('*.mp3'):
        if 'organized' not in str(audio_file):
            parts = audio_file.stem.split('-')
            if len(parts) >= 8:
                vocal_channel = parts[1]
                emotion_code = parts[2]
                
                emotion = get_emotion_label(emotion_code)
                modality_type = get_modality_label(vocal_channel)
                
                dest_dir = organized_path / 'audio' / modality_type / emotion
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                dest_file = dest_dir / audio_file.name
                shutil.copy2(audio_file, dest_file)
                print(f"Organized: {audio_file.name} -> audio/{modality_type}/{emotion}/")

if __name__ == '__main__':
    print("Starting RAVDESS data organization...")
    
    data_path = Path('C:\\Users\\chint\\Desktop\\4-2-project\\data')
    organized_path = data_path / 'organized'
    
    video_count = 0
    audio_count = 0
    
    # Process all Actor directories
    for actor_dir in sorted(data_path.glob('Actor_*')):
        if actor_dir.is_dir():
            print(f"Processing {actor_dir.name}...")
            
            # Process video files (.mp4)
            for video_file in actor_dir.glob('*.mp4'):
                parts = video_file.stem.split('-')
                if len(parts) >= 3:
                    vocal_channel = parts[1]  # 01=speech, 02=song
                    emotion_code = parts[2]   # 01-08 = emotions
                    
                    emotion = get_emotion_label(emotion_code)
                    modality_type = get_modality_label(vocal_channel)
                    
                    dest_dir = organized_path / 'video' / modality_type / emotion
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    dest_file = dest_dir / video_file.name
                    shutil.copy2(video_file, dest_file)
                    video_count += 1
            
            # Process audio files (.wav)
            for audio_file in actor_dir.glob('*.wav'):
                parts = audio_file.stem.split('-')
                if len(parts) >= 3:
                    vocal_channel = parts[1]  # 01=speech, 02=song
                    emotion_code = parts[2]   # 01-08 = emotions
                    
                    emotion = get_emotion_label(emotion_code)
                    modality_type = get_modality_label(vocal_channel)
                    
                    dest_dir = organized_path / 'audio' / modality_type / emotion
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    dest_file = dest_dir / audio_file.name
                    shutil.copy2(audio_file, dest_file)
                    audio_count += 1
    
    print(f"\n✓ Data organization complete!")
    print(f"Videos organized: {video_count}")
    print(f"Audio files organized: {audio_count}")
    print("\nFinal structure:")
    print("organized/")
    print("├── audio/")
    print("│   ├── speech/")
    print("│   │   ├── neutral/")
    print("│   │   ├── calm/")
    print("│   │   ├── happy/")
    print("│   │   ├── sad/")
    print("│   │   ├── angry/")
    print("│   │   ├── fearful/")
    print("│   │   ├── disgusted/")
    print("│   │   └── surprised/")
    print("│   └── song/")
    print("│       └── [emotions...]")
    print("└── video/")
    print("    ├── speech/")
    print("    │   └── [emotions...]")
    print("    └── song/")
    print("        └── [emotions...]")
