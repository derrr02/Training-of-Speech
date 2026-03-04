"""
Translation App Integration Example

This script demonstrates how to integrate the trained CNN model into a translation app.
It shows language detection and can be extended with actual translation logic.

Usage:
    python translation_app_example.py --audio_file input_audio.wav
"""

import argparse
import json
import numpy as np
import librosa
from tensorflow import keras
from pathlib import Path


class LanguageDetector:
    """Language detector for translation apps"""
    
    def __init__(self, model_path='language_classifier.h5', 
                 sample_rate=16000, duration=3, n_mfcc=40):
        """
        Initialize the language detector
        
        Args:
            model_path: Path to trained model
            sample_rate: Sample rate for audio processing
            duration: Duration in seconds for audio clips
            n_mfcc: Number of MFCC features
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.max_len = int(sample_rate * duration)
        
        # Load model and labels
        self.model = keras.models.load_model(model_path)
        
        # Load labels
        label_path = model_path.replace('.h5', '_labels.json')
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            self.languages = label_data['classes']
        
        print(f"Loaded model: {model_path}")
        print(f"Supported languages: {', '.join(self.languages)}")
    
    def extract_features(self, audio_path):
        """
        Extract MFCC features from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            MFCC features as numpy array
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        
        # Pad or truncate
        if len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)))
        else:
            audio = audio[:self.max_len]
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        
        # Normalize
        mfcc_mean = np.mean(mfcc)
        mfcc_std = np.std(mfcc)
        mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
        
        # Add batch and channel dimensions
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        
        return mfcc
    
    def detect_language(self, audio_path, confidence_threshold=0.5):
        """
        Detect language from audio file
        
        Args:
            audio_path: Path to audio file
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Tuple of (language, confidence, all_predictions)
        """
        # Extract features
        features = self.extract_features(audio_path)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)[0]
        
        # Get top prediction
        top_idx = np.argmax(predictions)
        detected_language = self.languages[top_idx]
        confidence = float(predictions[top_idx])
        
        # Get all predictions sorted by confidence
        sorted_indices = np.argsort(predictions)[::-1]
        all_predictions = [
            (self.languages[i], float(predictions[i])) 
            for i in sorted_indices
        ]
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            detected_language = "unknown"
        
        return detected_language, confidence, all_predictions
    
    def process_audio_stream(self, audio_data, sr=16000):
        """
        Process streaming audio data (for real-time apps)
        
        Args:
            audio_data: Raw audio data as numpy array
            sr: Sample rate of the audio
            
        Returns:
            Detected language and confidence
        """
        # Resample if needed
        if sr != self.sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
        
        # Process in chunks
        chunk_size = self.max_len
        languages_detected = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            
            if len(chunk) < chunk_size // 2:  # Skip very short chunks
                continue
            
            # Pad if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Extract features
            mfcc = librosa.feature.mfcc(y=chunk, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc)
            mfcc_std = np.std(mfcc)
            mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
            mfcc = mfcc[np.newaxis, ..., np.newaxis]
            
            # Predict
            predictions = self.model.predict(mfcc, verbose=0)[0]
            top_idx = np.argmax(predictions)
            
            languages_detected.append({
                'language': self.languages[top_idx],
                'confidence': float(predictions[top_idx])
            })
        
        # Return most common high-confidence language
        if languages_detected:
            # Filter by confidence
            high_conf = [l for l in languages_detected if l['confidence'] > 0.7]
            
            if high_conf:
                # Get most common
                lang_counts = {}
                for item in high_conf:
                    lang = item['language']
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
                most_common = max(lang_counts, key=lang_counts.get)
                avg_confidence = np.mean([l['confidence'] for l in high_conf if l['language'] == most_common])
                
                return most_common, float(avg_confidence)
        
        return "unknown", 0.0


class TranslationApp:
    """Simple translation app using language detection"""
    
    def __init__(self, model_path='language_classifier.h5'):
        """Initialize translation app with language detector"""
        self.detector = LanguageDetector(model_path)
    
    def translate(self, audio_file, target_language='english'):
        """
        Translate audio from detected language to target language
        
        Args:
            audio_file: Path to audio file
            target_language: Target language for translation
            
        Returns:
            Translation result dictionary
        """
        print(f"\n{'='*60}")
        print("TRANSLATION APP - Language Detection")
        print(f"{'='*60}\n")
        
        # Detect source language
        print(f"Analyzing audio: {audio_file}")
        source_lang, confidence, all_predictions = self.detector.detect_language(audio_file)
        
        print(f"\n📊 Language Detection Results:")
        print(f"{'─'*60}")
        for i, (lang, prob) in enumerate(all_predictions[:5], 1):
            bar = '█' * int(prob * 30)
            print(f"{i}. {lang:15s} {prob*100:5.2f}% {bar}")
        
        print(f"\n🎯 Detected Language: {source_lang.upper()}")
        print(f"   Confidence: {confidence*100:.2f}%")
        
        # Translation logic (placeholder)
        if source_lang == "unknown":
            print(f"\n⚠️  Could not confidently detect language")
            print(f"   Confidence too low (threshold: 50%)")
            return {
                'source_language': source_lang,
                'confidence': confidence,
                'translation': None,
                'status': 'failed'
            }
        
        if source_lang.lower() == target_language.lower():
            print(f"\n✓ Source language is already {target_language}")
            return {
                'source_language': source_lang,
                'confidence': confidence,
                'translation': 'No translation needed',
                'status': 'same_language'
            }
        
        # Placeholder for actual translation
        print(f"\n🔄 Translation: {source_lang} → {target_language}")
        print(f"   (Translation logic would be implemented here)")
        print(f"   You can integrate with Google Translate API, Azure Translator, etc.")
        
        return {
            'source_language': source_lang,
            'target_language': target_language,
            'confidence': confidence,
            'translation': f'[Translated from {source_lang} to {target_language}]',
            'status': 'success'
        }
    
    def batch_translate(self, audio_files, target_language='english'):
        """
        Translate multiple audio files
        
        Args:
            audio_files: List of audio file paths
            target_language: Target language for translation
            
        Returns:
            List of translation results
        """
        results = []
        
        print(f"\n🔢 Batch Translation: {len(audio_files)} files")
        print(f"   Target Language: {target_language}")
        print(f"{'='*60}\n")
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"Processing {i}/{len(audio_files)}: {Path(audio_file).name}")
            result = self.translate(audio_file, target_language)
            results.append(result)
            print()
        
        # Summary
        successful = sum(1 for r in results if r['status'] == 'success')
        print(f"\n{'='*60}")
        print(f"Batch Translation Complete: {successful}/{len(audio_files)} successful")
        print(f"{'='*60}")
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Translation app with language detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='speech_cnn_model.h5',
        help='Path to trained language detection model'
    )
    
    parser.add_argument(
        '--audio_file',
        type=str,
        help='Path to audio file to translate'
    )
    
    parser.add_argument(
        '--audio_files',
        type=str,
        nargs='+',
        help='Multiple audio files for batch processing'
    )
    
    parser.add_argument(
        '--target_language',
        type=str,
        default='english',
        help='Target language for translation'
    )
    
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.5,
        help='Minimum confidence for language detection'
    )
    
    args = parser.parse_args()
    
    # Create translation app
    try:
        app = TranslationApp(args.model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found: {args.model_path}")
        print("\nPlease train a model first using:")
        print("python train_speech_cnn.py --data_dir your_data")
        return
    
    # Process audio
    if args.audio_files:
        # Batch mode
        app.batch_translate(args.audio_files, args.target_language)
    elif args.audio_file:
        # Single file mode
        result = app.translate(args.audio_file, args.target_language)
        
        if result['status'] == 'success':
            print(f"\n✅ Translation completed successfully!")
        elif result['status'] == 'same_language':
            print(f"\n✅ No translation needed")
        else:
            print(f"\n❌ Translation failed")
    else:
        print("Error: Please provide --audio_file or --audio_files")
        print("\nExample usage:")
        print("  python translation_app_example.py --audio_file speech.wav")
        print("  python translation_app_example.py --audio_files file1.wav file2.wav file3.wav")


if __name__ == '__main__':
    main()
