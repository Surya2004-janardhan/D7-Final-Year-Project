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
    """2D CNN with TimeDistributed + LSTM for video emotion detection"""
    
    def __init__(self, num_emotions=8):
        self.num_emotions = num_emotions
        self.model = None
    
    def build_model(self, input_shape):
        """
        2D CNN (TimeDistributed) + LSTM architecture
        Input shape: (num_frames=16, height=160, width=160, 3)
        Applies 2D CNN to each frame, then LSTM on temporal sequence
        """
        inputs = layers.Input(shape=input_shape)
        
        # TimeDistributed 2D CNN on each frame
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))(inputs)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        
        # x shape: (batch, 16, 128) - temporal sequence of CNN features
        
        # LSTM on temporal features
        x = layers.LSTM(64, return_sequences=True, dropout=0.3)(x)
        x = layers.LSTM(32, dropout=0.3)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_emotions, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=4):
        """Train with learning rate scheduling"""
        
        # Learning rate scheduler
        lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[lr_schedule, early_stop]
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
