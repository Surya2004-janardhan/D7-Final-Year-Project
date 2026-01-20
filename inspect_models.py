"""
Inspect H5 Model Metadata
Print model summaries, parameters, and generate architecture plots
"""

import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def inspect_model(model_path, model_name):
    """Inspect a single model"""
    print(f"\n{'='*60}")
    print(f"Inspecting {model_name}")
    print(f"{'='*60}")

    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")

        # Model summary
        print("\nModel Summary:")
        model.summary()

        # Parameters
        total_params = model.count_params()
        trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
        non_trainable_params = total_params - trainable_params

        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")

        # Layers info
        print(f"\nNumber of Layers: {len(model.layers)}")
        print("Layer Details:")
        for i, layer in enumerate(model.layers):
            print(f"  {i+1}. {layer.name} ({layer.__class__.__name__}) - Trainable: {layer.trainable}")

        # Input/Output shapes
        print(f"\nInput Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")

        # Generate architecture plot
        try:
            plot_path = f'plots/{model_name.lower().replace(" ", "_")}_architecture.png'
            tf.keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
            print(f"Architecture plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not generate architecture plot: {e}")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main():
    models_dir = 'models'
    models = [
        ('audio_emotion_model.h5', 'Audio Emotion Model'),
        ('video_emotion_model.h5', 'Video Emotion Model'),
        ('fusion_emotion.h5', 'Fusion Emotion Model')
    ]

    os.makedirs('plots', exist_ok=True)

    for model_file, model_name in models:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            inspect_model(model_path, model_name)
        else:
            print(f"Model file not found: {model_path}")

if __name__ == '__main__':
    main()