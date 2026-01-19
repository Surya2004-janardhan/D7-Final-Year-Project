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
from sklearn.utils.class_weight import compute_class_weight

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
        Build EfficientNet-B0 backbone for FER
        """
        from tensorflow.keras.applications import EfficientNetB0
        inputs = layers.Input(shape=(48, 48, 3))
        base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(48, 48, 3), pooling='avg')
        base.trainable = True
        x = base(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        self.model = models.Model(inputs=inputs, outputs=outputs)
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
        Train EfficientNetB0 on FER-2013 with strong augmentation and class weights
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
                color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48, 48),
                color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(self.num_classes),
            y=train_generator.classes
        )
        class_weight_dict = dict(enumerate(class_weights))
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
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=2  # One print per epoch
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
    import shutil
    from sklearn.model_selection import train_test_split
    import glob
    # Paths - use test folder as main data since train is incomplete
    FER_DATA_DIR = 'fer-data/test'
    FER_TRAIN_DIR = 'fer-data/train_split'
    FER_VAL_DIR = 'fer-data/val_split'
    print("=" * 60)
    print("Training FER-2013 Base Model")
    print("=" * 60)
    # Check if data exists
    if not os.path.exists(FER_DATA_DIR):
        print(f"Error: Data directory not found: {FER_DATA_DIR}")
        exit(1)
    
    # Create train/val split from existing data
    print("\nCreating train/validation split from available data...")
    
    # Clean up old splits
    for split_dir in [FER_TRAIN_DIR, FER_VAL_DIR]:
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)
    
    # Get all emotion classes
    emotion_dirs = [d for d in os.listdir(FER_DATA_DIR) 
                    if os.path.isdir(os.path.join(FER_DATA_DIR, d))]
    
    print(f"Found emotion classes: {emotion_dirs}")
    
    for emotion in emotion_dirs:
        emotion_path = os.path.join(FER_DATA_DIR, emotion)
        images = glob.glob(os.path.join(emotion_path, '*.png')) + \
                 glob.glob(os.path.join(emotion_path, '*.jpg'))
        if len(images) == 0:
            print(f"Warning: No images found for {emotion}")
            continue
        # Split 80/20
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
        # Create directories and copy files
        train_emotion_dir = os.path.join(FER_TRAIN_DIR, emotion)
        val_emotion_dir = os.path.join(FER_VAL_DIR, emotion)
        os.makedirs(train_emotion_dir, exist_ok=True)
        os.makedirs(val_emotion_dir, exist_ok=True)
        for img in train_imgs:
            shutil.copy(img, train_emotion_dir)
        for img in val_imgs:
            shutil.copy(img, val_emotion_dir)
        print(f"  {emotion}: {len(train_imgs)} train, {len(val_imgs)} val")
    
    # Create model
    num_classes = len(emotion_dirs)
    fer_model = FERModel(num_classes=num_classes, input_shape=(48, 48, 1))
    fer_model.build_model()
    
    print(f"\nModel Architecture ({num_classes} classes):")
    fer_model.model.summary()
    
    # Train
    print("\nStarting training...")
    history = fer_model.train(
        train_dir=FER_TRAIN_DIR,
        val_dir=FER_VAL_DIR,
        epochs=50,
        batch_size=64
    )
    
    # Plot results
    plot_training_history(history)
    
    # Evaluate
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        FER_VAL_DIR,
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
