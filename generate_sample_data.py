"""
Sample Data Generator for Testing

This script generates synthetic audio data for testing the CNN training pipeline.
It creates simple audio samples with different frequencies representing different classes.

Usage:
    python generate_sample_data.py --output_dir sample_data --num_samples 100
"""

import argparse
import os
import numpy as np
import soundfile as sf
from pathlib import Path


def generate_audio_sample(frequency, duration=3, sample_rate=16000):
    """
    Generate a simple audio sample with a specific frequency
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate
        
    Returns:
        Audio signal as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate sine wave with the given frequency
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some harmonics for richness
    audio += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
    audio += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)
    
    # Add slight noise
    noise = np.random.normal(0, 0.02, audio.shape)
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio


def generate_dataset(output_dir, num_samples_per_class=30):
    """
    Generate a sample dataset with different audio classes
    
    Args:
        output_dir: Directory to save the generated data
        num_samples_per_class: Number of samples to generate per class
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define classes with different frequency ranges
    classes = {
        'low_tone': 220,    # A3 note
        'mid_tone': 440,    # A4 note
        'high_tone': 880,   # A5 note
    }
    
    print(f"Generating sample dataset in: {output_dir}")
    print("=" * 60)
    
    for class_name, base_frequency in classes.items():
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        print(f"Generating {num_samples_per_class} samples for class: {class_name}")
        
        for i in range(num_samples_per_class):
            # Add some variation to the frequency
            frequency = base_frequency * (1 + np.random.uniform(-0.1, 0.1))
            
            # Generate audio
            audio = generate_audio_sample(frequency)
            
            # Save as WAV file
            output_file = class_dir / f"{class_name}_{i+1:03d}.wav"
            sf.write(str(output_file), audio, 16000)
        
        print(f"  ✓ Created {num_samples_per_class} samples in {class_dir}")
    
    print("\n" + "=" * 60)
    print(f"Dataset generation complete!")
    print(f"Total samples: {len(classes) * num_samples_per_class}")
    print(f"\nYou can now train the model using:")
    print(f"python train_speech_cnn.py --data_dir {output_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Generate sample audio data for testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='sample_data',
        help='Directory to save generated data'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=30,
        help='Number of samples to generate per class'
    )
    
    args = parser.parse_args()
    
    generate_dataset(args.output_dir, args.num_samples)


if __name__ == '__main__':
    main()
