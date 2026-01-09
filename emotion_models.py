"""
CNN-LSTM Model for Emotion Detection from Audio and Video
Combines audio features with video frames for multimodal emotion recognition
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

class AudioEmotionCNNLSTM:
    """CNN-LSTM model for audio emotion detection"""
    
    def __init__(self, num_emotions=8):
        self.num_emotions = num_emotions
        self.model = None
    
    def build_model(self, input_shape):
        """
        Build CNN-LSTM model for audio features (MFCC)
        Input shape: (seq_length, n_mfcc)
        """
        model = models.Sequential([
            # CNN layers to extract features
            layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Reshape for LSTM
            layers.Reshape((input_shape[0] // 4, 64 * (input_shape[1] // 4))),
            
            # LSTM layers
            layers.LSTM(128, return_sequences=True, dropout=0.5),
            layers.LSTM(64, dropout=0.5),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_emotions, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Predict emotions"""
        return self.model.predict(X)


class VideoEmotionCNNLSTM:
    """CNN-LSTM model for video emotion detection"""
    
    def __init__(self, num_emotions=8):
        self.num_emotions = num_emotions
        self.model = None
    
    def build_model(self, input_shape):
        """
        Build 3D CNN-LSTM model for video frames
        Input shape: (num_frames=8, height=160, width=160, 3)
        Ultra-optimized for RTX 2050 (minimal memory footprint)
        """
        model = models.Sequential([
            # 3D CNN layers (reduced filters for memory)
            layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling3D((1, 2, 2)),
            
            layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling3D((1, 2, 2)),
            
            # Reshape for LSTM
            layers.Reshape((input_shape[0], -1)),
            
            # LSTM layers (reduced units)
            layers.LSTM(64, return_sequences=True, dropout=0.5),
            layers.LSTM(32, dropout=0.5),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_emotions, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=8):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history


class MultimodalEmotionCNNLSTM:
    """Multimodal model combining audio and video"""
    
    def __init__(self, num_emotions=8):
        self.num_emotions = num_emotions
        self.model = None
    
    def build_model(self, audio_shape, video_shape):
        """Build multimodal model"""
        # Audio branch (CNN-LSTM)
        audio_input = layers.Input(shape=audio_shape)
        audio_x = layers.Reshape((audio_shape[0], audio_shape[1], 1))(audio_input)
        audio_x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(audio_x)
        audio_x = layers.MaxPooling2D((2, 2))(audio_x)
        audio_x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(audio_x)
        audio_x = layers.MaxPooling2D((2, 2))(audio_x)
        audio_x = layers.Reshape((audio_shape[0] // 4, 64 * (audio_shape[1] // 4)))(audio_x)
        audio_x = layers.LSTM(128, return_sequences=True, dropout=0.5)(audio_x)
        audio_x = layers.LSTM(64, dropout=0.5)(audio_x)
        
        # Video branch (3D CNN-LSTM)
        video_input = layers.Input(shape=video_shape)
        video_x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(video_input)
        video_x = layers.MaxPooling3D((1, 2, 2))(video_x)
        video_x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(video_x)
        video_x = layers.MaxPooling3D((1, 2, 2))(video_x)
        video_x = layers.Reshape((video_shape[0], -1))(video_x)
        video_x = layers.LSTM(128, return_sequences=True, dropout=0.5)(video_x)
        video_x = layers.LSTM(64, dropout=0.5)(video_x)
        
        # Fusion
        fusion = layers.Concatenate()([audio_x, video_x])
        fusion = layers.Dense(128, activation='relu')(fusion)
        fusion = layers.Dropout(0.5)(fusion)
        output = layers.Dense(self.num_emotions, activation='softmax')(fusion)
        
        model = models.Model(inputs=[audio_input, video_input], outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, X_audio, X_video, y, X_audio_val, X_video_val, y_val, 
              epochs=50, batch_size=8):
        """Train multimodal model"""
        history = self.model.fit(
            [X_audio, X_video], y,
            validation_data=([X_audio_val, X_video_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history


# Example training script
if __name__ == '__main__':
    print("Building Audio Emotion Detection Model...")
    audio_model = AudioEmotionCNNLSTM(num_emotions=8)
    audio_model.build_model((13, 216))  # MFCC shape
    print(audio_model.model.summary())
    
    print("\nBuilding Video Emotion Detection Model...")
    video_model = VideoEmotionCNNLSTM(num_emotions=8)
    video_model.build_model((30, 224, 224, 3))  # Video shape (frames, height, width, channels)
    print(video_model.model.summary())
    
    print("\nBuilding Multimodal Emotion Detection Model...")
    multimodal_model = MultimodalEmotionCNNLSTM(num_emotions=8)
    multimodal_model.build_model((13, 216), (30, 224, 224, 3))
    print(multimodal_model.model.summary())
