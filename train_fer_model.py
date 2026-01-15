"""
Train Basic Facial Emotion Recognition Model on FER-2013 Dataset
This serves as the base face emotion feature extractor
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# FER-2013 emotion classes
FER_EMOTIONS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class FERModel:
    """
    Basic CNN model for Facial Emotion Recognition trained on FER-2013
    This model serves as a feature extractor for temporal emotion detection
    """
    
    def __init__(self, num_classes=8, input_shape=(48, 48, 1)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.feature_extractor = None
    
    def build_model(self):
        """
        Build a basic CNN architecture for FER
        Architecture: Simple but effective for grayscale face images
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Block 1
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 3
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 4
        x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_feature_extractor(self):
        """
        Create a feature extractor from the trained model
        Returns 256-dimensional feature vector per face
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Get output from second-to-last dense layer
        self.feature_extractor = models.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output  # 256-dim features
        )
        return self.feature_extractor
    
    def train(self, train_dir, val_dir, epochs=50, batch_size=64):
        """
        Train the model on FER-2013 dataset
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            shear_range=0.1
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'models/fer_base_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save(self, path='models/fer_base_model.h5'):
        """Save the model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path='models/fer_base_model.h5'):
        """Load the model"""
        self.model = keras.models.load_model(path)
        self.build_feature_extractor()
        print(f"Model loaded from {path}")


def plot_training_history(history, save_path='plots/fer_training.png'):
    """Plot training history"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training plot saved to {save_path}")


if __name__ == '__main__':
    # Paths
    FER_TRAIN_DIR = 'fer-data/train'
    FER_TEST_DIR = 'fer-data/test'
    
    print("=" * 60)
    print("Training FER-2013 Base Model")
    print("=" * 60)
    
    # Check if directories exist
    if not os.path.exists(FER_TRAIN_DIR):
        print(f"Error: Training directory not found: {FER_TRAIN_DIR}")
        exit(1)
    
    # Create model
    fer_model = FERModel(num_classes=8, input_shape=(48, 48, 1))
    fer_model.build_model()
    
    print("\nModel Architecture:")
    fer_model.model.summary()
    
    # Train
    print("\nStarting training...")
    history = fer_model.train(
        train_dir=FER_TRAIN_DIR,
        val_dir=FER_TEST_DIR,
        epochs=50,
        batch_size=64
    )
    
    # Plot results
    plot_training_history(history)
    
    # Evaluate
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        FER_TEST_DIR,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )
    
    loss, accuracy = fer_model.model.evaluate(test_generator)
    print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")
    print(f"Final Test Loss: {loss:.4f}")
    
    # Save final model
    fer_model.save('models/fer_base_model.h5')
    print("\nFER Base Model training complete!")
