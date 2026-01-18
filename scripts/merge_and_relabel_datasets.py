import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Define emotion class mappings
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Dataset-specific mappings
dataset_mappings = {
    'FER2013+': {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happy',
        'neutral': 'neutral',
        'sad': 'sad',
        'surprise': 'surprise'
    },
    'ExpW': {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happy',
        'neutral': 'neutral',
        'sad': 'sad',
        'surprise': 'surprise'
    },
    'RAVDESS': {
        '01': 'neutral',
        '02': 'calm',  # Map calm to neutral
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fear',
        '07': 'disgust',
        '08': 'surprise'
    },
    'CREMA-D': {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fear',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad',
        'SUR': 'surprise'
    },
    'SAVEE': {
        'a': 'angry',
        'd': 'disgust',
        'f': 'fear',
        'h': 'happy',
        'n': 'neutral',
        'sa': 'sad',
        'su': 'surprise'
    },
    'Emo-DB': {
        'W': 'angry',
        'E': 'disgust',
        'A': 'fear',
        'F': 'happy',
        'N': 'neutral',
        'T': 'sad',
        'L': 'boredom'  # Map boredom to neutral or skip
    }
}

def load_fer_dataset(dataset_path, dataset_name):
    """Load FER dataset images and labels."""
    images = []
    labels = []
    mapping = dataset_mappings[dataset_name]
    
    for emotion in os.listdir(dataset_path):
        if emotion in mapping:
            emotion_path = os.path.join(dataset_path, emotion)
            if os.path.isdir(emotion_path):
                for img_file in os.listdir(emotion_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        images.append(os.path.join(emotion_path, img_file))
                        labels.append(emotion_classes.index(mapping[emotion]))
    
    return images, labels

def load_audio_dataset(dataset_path, dataset_name):
    """Load audio dataset files and labels."""
    audio_files = []
    labels = []
    mapping = dataset_mappings[dataset_name]
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                # Extract emotion from filename
                emotion_code = file.split('_')[2] if dataset_name == 'RAVDESS' else file.split('.')[0][-3:]
                if emotion_code in mapping:
                    audio_files.append(os.path.join(root, file))
                    labels.append(emotion_classes.index(mapping[emotion_code]))
    
    return audio_files, labels

def balance_dataset(images_or_files, labels, target_samples_per_class=1000):
    """Balance the dataset by undersampling to target samples per class."""
    balanced_images = []
    balanced_labels = []
    
    for class_idx in range(len(emotion_classes)):
        class_samples = [img for img, label in zip(images_or_files, labels) if label == class_idx]
        if len(class_samples) > target_samples_per_class:
            class_samples = np.random.choice(class_samples, target_samples_per_class, replace=False)
        balanced_images.extend(class_samples)
        balanced_labels.extend([class_idx] * len(class_samples))
    
    return balanced_images, balanced_labels

def merge_datasets():
    """Merge all datasets and create balanced train/val/test splits."""
    all_images = []
    all_labels = []
    
    # Load FER datasets
    fer_datasets = {
        'FER2013+': 'fer-data/fer2013plus/fer2013/train',
        'ExpW': 'expW'  # Assuming extracted images are here
    }
    
    for name, path in fer_datasets.items():
        if os.path.exists(path):
            images, labels = load_fer_dataset(path, name)
            all_images.extend(images)
            all_labels.extend(labels)
    
    # Load audio datasets
    audio_datasets = {
        'RAVDESS': 'data/RAVDESS',
        'CREMA-D': 'data/CREMA-D',
        'SAVEE': 'data/SAVEE',
        'Emo-DB': 'data/Emo-DB'
    }
    
    for name, path in audio_datasets.items():
        if os.path.exists(path):
            audio_files, labels = load_audio_dataset(path, name)
            all_images.extend(audio_files)  # Treat as files
            all_labels.extend(labels)
    
    # Balance the dataset
    balanced_files, balanced_labels = balance_dataset(all_images, all_labels)
    
    # Split into train/val/test
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        balanced_files, balanced_labels, test_size=0.3, stratify=balanced_labels, random_state=42
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Create directories and copy files
    for split, files, labels in [('train', train_files, train_labels), ('val', val_files, val_labels), ('test', test_files, test_labels)]:
        split_dir = f'data/merged_{split}'
        os.makedirs(split_dir, exist_ok=True)
        
        for class_idx in range(len(emotion_classes)):
            class_dir = os.path.join(split_dir, emotion_classes[class_idx])
            os.makedirs(class_dir, exist_ok=True)
        
        for file_path, label in zip(files, labels):
            class_name = emotion_classes[label]
            dest_dir = os.path.join(split_dir, class_name)
            shutil.copy(file_path, dest_dir)
    
    print("Datasets merged and balanced successfully!")

if __name__ == "__main__":
    merge_datasets()