"""
Inference script for the trained CNN speech model

This script loads a trained model and uses it to make predictions on new audio files.

Usage:
    python predict_speech.py --model_path speech_cnn_model.h5 --audio_file sample.wav
"""

import argparse
import json
import numpy as np
import librosa
from tensorflow import keras


def load_model_and_labels(model_path):
    """
    Load trained model and label encoder
    
    Args:
        model_path: Path to the trained model (.h5 file)
        
    Returns:
        model: Loaded Keras model
        classes: List of class names
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load labels
    label_path = model_path.replace('.h5', '_labels.json')
    with open(label_path, 'r') as f:
        label_data = json.load(f)
        classes = label_data['classes']
    
    return model, classes


def extract_features(audio_path, sample_rate=16000, duration=3, n_mfcc=40):
    """
    Extract MFCC features from audio file
    
    Args:
        audio_path: Path to audio file
        sample_rate: Sample rate for audio
        duration: Duration to pad/truncate
        n_mfcc: Number of MFCC features
        
    Returns:
        MFCC features
    """
    max_len = int(sample_rate * duration)
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
    
    # Pad or truncate
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Normalize
    mfcc_mean = np.mean(mfcc)
    mfcc_std = np.std(mfcc)
    # Add epsilon to avoid division by zero for silent audio
    mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
    
    # Add batch and channel dimensions
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    
    return mfcc


def predict(model_path, audio_file, top_k=3):
    """
    Predict the class of an audio file
    
    Args:
        model_path: Path to trained model
        audio_file: Path to audio file to classify
        top_k: Number of top predictions to show
        
    Returns:
        predictions: List of (class_name, probability) tuples
    """
    # Load model and labels
    model, classes = load_model_and_labels(model_path)
    
    # Extract features
    features = extract_features(audio_file)
    
    # Make prediction
    predictions = model.predict(features, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_predictions = [(classes[i], float(predictions[i])) for i in top_indices]
    
    return top_predictions


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Predict speech class using trained CNN model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='speech_cnn_model.h5',
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--audio_file',
        type=str,
        required=True,
        help='Path to audio file to classify'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Number of top predictions to show'
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing: {args.audio_file}")
    print("-" * 60)
    
    # Make prediction
    predictions = predict(args.model_path, args.audio_file, args.top_k)
    
    # Display results
    print("\nPredictions:")
    for i, (class_name, probability) in enumerate(predictions, 1):
        print(f"{i}. {class_name}: {probability*100:.2f}%")
    
    print("\n" + "=" * 60)
    print(f"Top prediction: {predictions[0][0]} ({predictions[0][1]*100:.2f}%)")


if __name__ == '__main__':
    main()
