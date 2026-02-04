"""
CNN-based Speech Data Training Script

This script trains a Convolutional Neural Network (CNN) on speech data.
It supports various audio formats and can be used for tasks like:
- Speech recognition
- Speaker identification
- Audio classification
- Emotion detection from speech

Usage:
    python train_speech_cnn.py --data_dir /path/to/data --epochs 50 --batch_size 32
"""

import os
import argparse
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path


class SpeechDataLoader:
    """Loads and preprocesses speech data for CNN training"""
    
    def __init__(self, data_dir, sample_rate=16000, duration=3, n_mfcc=40):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing audio files organized by class/label
            sample_rate: Target sample rate for audio files
            duration: Duration in seconds to pad/truncate audio
            n_mfcc: Number of MFCC features to extract
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.max_len = int(sample_rate * duration)
        
    def extract_features(self, audio_path):
        """
        Extract MFCC features from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            MFCC features as numpy array
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate to fixed length
            if len(audio) < self.max_len:
                audio = np.pad(audio, (0, self.max_len - len(audio)))
            else:
                audio = audio[:self.max_len]
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            
            # Normalize
            mfcc_mean = np.mean(mfcc)
            mfcc_std = np.std(mfcc)
            # Add epsilon to avoid division by zero for silent audio
            mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
            
            return mfcc
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def load_data(self):
        """
        Load all audio files from data directory
        
        Expected directory structure:
            data_dir/
                class1/
                    audio1.wav
                    audio2.wav
                class2/
                    audio3.wav
                    audio4.wav
        
        Returns:
            X: Feature array
            y: Label array
            label_encoder: Fitted LabelEncoder
        """
        X = []
        y = []
        
        # Get all subdirectories as class labels
        data_path = Path(self.data_dir)
        
        if not data_path.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist")
        
        # Supported audio formats
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        
        # Iterate through class directories
        for class_dir in sorted(data_path.iterdir()):
            if class_dir.is_dir():
                label = class_dir.name
                print(f"Processing class: {label}")
                
                # Process all audio files in this class
                audio_files = [f for f in class_dir.iterdir() 
                              if f.suffix.lower() in audio_extensions]
                
                for audio_file in audio_files:
                    features = self.extract_features(str(audio_file))
                    if features is not None:
                        X.append(features)
                        y.append(label)
        
        if len(X) == 0:
            raise ValueError("No audio files found! Please check your data directory structure.")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"Loaded {len(X)} samples from {len(label_encoder.classes_)} classes")
        print(f"Classes: {label_encoder.classes_}")
        
        # Add channel dimension for CNN
        X = X[..., np.newaxis]
        
        return X, y_encoded, label_encoder


def build_cnn_model(input_shape, num_classes):
    """
    Build a CNN model for speech classification
    
    Args:
        input_shape: Shape of input features (n_mfcc, time_steps, 1)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_model(data_dir, epochs=50, batch_size=32, validation_split=0.2, 
                model_save_path='speech_cnn_model.h5'):
    """
    Train the CNN model on speech data
    
    Args:
        data_dir: Directory containing training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        model_save_path: Path to save the trained model
    """
    print("="*60)
    print("CNN Speech Training")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    data_loader = SpeechDataLoader(data_dir)
    X, y, label_encoder = data_loader.load_data()
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    print("\n3. Building model...")
    num_classes = len(label_encoder.classes_)
    input_shape = X_train.shape[1:]
    
    model = build_cnn_model(input_shape, num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n4. Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n5. Evaluating model...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Save label encoder
    label_encoder_path = model_save_path.replace('.h5', '_labels.json')
    with open(label_encoder_path, 'w') as f:
        json.dump({
            'classes': label_encoder.classes_.tolist()
        }, f, indent=2)
    
    print(f"\n6. Model saved to: {model_save_path}")
    print(f"   Labels saved to: {label_encoder_path}")
    
    return model, history, label_encoder


def main():
    """Main function to parse arguments and start training"""
    parser = argparse.ArgumentParser(
        description='Train a CNN model on speech data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing audio files organized by class'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--validation_split',
        type=float,
        default=0.2,
        help='Fraction of data to use for validation'
    )
    
    parser.add_argument(
        '--model_save_path',
        type=str,
        default='speech_cnn_model.h5',
        help='Path to save the trained model'
    )
    
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Sample rate for audio processing'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=3,
        help='Duration in seconds to pad/truncate audio'
    )
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        model_save_path=args.model_save_path
    )


if __name__ == '__main__':
    main()
