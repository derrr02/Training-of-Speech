# Training Speech Data for Translation App Using CNN

This guide will help you train a CNN model on speech data specifically for a translation application. The model can help identify languages, speakers, or speech patterns that are useful for translation.

## Overview

For a translation app, you typically want to train a CNN model to:
1. **Identify the source language** - Detect which language is being spoken
2. **Recognize speakers** - Distinguish between different speakers
3. **Classify speech segments** - Identify meaningful segments for translation

## Step-by-Step Guide

### 1. Prepare Your Speech Data

For a translation app, organize your audio files by language or category:

```
translation_data/
├── english/
│   ├── speaker1_english_001.wav
│   ├── speaker1_english_002.wav
│   ├── speaker2_english_001.wav
│   └── ...
├── spanish/
│   ├── speaker1_spanish_001.wav
│   ├── speaker1_spanish_002.wav
│   └── ...
├── french/
│   ├── speaker1_french_001.wav
│   └── ...
└── german/
    └── ...
```

**Important Tips:**
- Each subdirectory name will be treated as a class label (e.g., "english", "spanish")
- Use clear, consistent audio recordings (16kHz sample rate recommended)
- Keep audio segments between 2-5 seconds for best results
- Have at least 50-100 samples per language for decent accuracy
- More samples = better accuracy (aim for 500+ per class for production)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow (for the CNN model)
- librosa (for audio processing)
- scikit-learn (for data splitting)
- soundfile (for audio file handling)

### 3. Train Your Language Identification Model

#### Basic Training (Quick Start)

```bash
python train_speech_cnn.py --data_dir translation_data --epochs 50
```

#### Recommended Settings for Translation App

```bash
python train_speech_cnn.py \
    --data_dir translation_data \
    --epochs 100 \
    --batch_size 32 \
    --validation_split 0.2 \
    --model_save_path language_classifier.h5 \
    --sample_rate 16000 \
    --duration 3
```

**Parameter Explanation:**
- `--data_dir`: Your audio data directory
- `--epochs`: Number of training iterations (100 is good for production)
- `--batch_size`: Number of samples processed at once (32 is standard)
- `--validation_split`: Percentage of data used for testing (0.2 = 20%)
- `--model_save_path`: Where to save your trained model
- `--sample_rate`: Audio quality (16000 Hz is standard for speech)
- `--duration`: Audio clip length in seconds (3 seconds works well)

### 4. Monitor Training Progress

During training, you'll see output like:

```
============================================================
CNN Speech Training
============================================================

1. Loading data...
Processing class: english
Processing class: spanish
Processing class: french
Loaded 300 samples from 3 classes

2. Splitting data...
Training samples: 240
Validation samples: 60

3. Building model...
Total params: 1,930,371 (7.36 MB)

4. Training model...
Epoch 1/100
...
Epoch 50: val_accuracy improved to 0.9500, saving model
...

5. Evaluating model...
Validation Accuracy: 0.9500
```

**What to look for:**
- Validation accuracy should increase over time
- Training should stop automatically if accuracy plateaus (early stopping)
- Final validation accuracy above 85% is good, above 95% is excellent

### 5. Use Your Trained Model

Once training is complete, you'll have:
- `language_classifier.h5` - Your trained model
- `language_classifier_labels.json` - Language labels

#### Test Your Model

```bash
python predict_speech.py \
    --model_path language_classifier.h5 \
    --audio_file test_audio.wav
```

Example output:
```
Analyzing: test_audio.wav
------------------------------------------------------------

Predictions:
1. spanish: 94.23%
2. english: 4.12%
3. french: 1.65%

============================================================
Top prediction: spanish (94.23%)
```

### 6. Integrate with Your Translation App

Here's sample Python code to use your model in a translation app:

```python
import numpy as np
import librosa
from tensorflow import keras
import json

# Load the trained model
model = keras.models.load_model('language_classifier.h5')

# Load language labels
with open('language_classifier_labels.json', 'r') as f:
    labels_data = json.load(f)
    languages = labels_data['classes']

def detect_language(audio_file_path):
    """Detect language from audio file"""
    # Load and preprocess audio
    audio, sr = librosa.load(audio_file_path, sr=16000, duration=3)
    
    # Pad or truncate
    max_len = 16000 * 3
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Normalize
    mfcc_mean = np.mean(mfcc)
    mfcc_std = np.std(mfcc)
    mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
    
    # Add batch and channel dimensions
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    
    # Predict
    predictions = model.predict(mfcc, verbose=0)[0]
    
    # Get top language
    top_idx = np.argmax(predictions)
    detected_language = languages[top_idx]
    confidence = float(predictions[top_idx])
    
    return detected_language, confidence

# Example usage in your translation app
language, confidence = detect_language('user_speech.wav')
print(f"Detected: {language} (confidence: {confidence*100:.1f}%)")

# Now you can use the detected language for translation
if language == 'spanish':
    # Translate Spanish to English
    pass
elif language == 'french':
    # Translate French to English
    pass
```

## Tips for Better Results

### 1. Data Quality Matters
- **Clear audio**: Remove background noise when possible
- **Consistent length**: Keep audio clips similar in duration
- **Balanced dataset**: Have similar amounts of data for each language
- **Varied speakers**: Include multiple speakers for each language

### 2. Handling Real-World Audio
If your translation app receives real-time audio:

```python
# For streaming audio, process in chunks
def process_audio_stream(audio_stream):
    # Split into 3-second chunks
    chunk_size = 16000 * 3
    for i in range(0, len(audio_stream), chunk_size):
        chunk = audio_stream[i:i+chunk_size]
        language, confidence = detect_language(chunk)
        
        # Only accept high-confidence predictions
        if confidence > 0.8:
            return language
    
    return "unknown"
```

### 3. Improving Accuracy

If your model accuracy is low:

1. **Add more training data** (most important!)
2. **Increase training epochs**: Try `--epochs 150` or `--epochs 200`
3. **Adjust audio duration**: Try `--duration 5` for longer clips
4. **Clean your data**: Remove mislabeled or poor-quality audio
5. **Balance your classes**: Ensure equal samples per language

### 4. Multi-Language Support

For many languages, organize like this:

```
translation_data/
├── arabic/
├── chinese/
├── english/
├── french/
├── german/
├── hindi/
├── japanese/
├── korean/
├── portuguese/
├── russian/
└── spanish/
```

The model will automatically detect all languages in your data directory.

## Common Use Cases for Translation Apps

### Use Case 1: Language Detection Before Translation
```python
# Detect language first
language, confidence = detect_language(audio_file)

# Only translate if confident
if confidence > 0.8:
    translation = translate_audio(audio_file, source_lang=language, target_lang='english')
else:
    print("Could not confidently detect language")
```

### Use Case 2: Multi-Speaker Translation
```python
# Train separate models for:
# 1. Language detection
# 2. Speaker identification

language = detect_language(audio)
speaker = identify_speaker(audio)

# Translate with speaker context
translation = translate_with_context(audio, language, speaker)
```

### Use Case 3: Real-Time Translation
```python
# For live translation apps
def live_translate(audio_stream):
    # Detect language every 3 seconds
    language = detect_language(audio_stream[:48000])  # 3 sec at 16kHz
    
    # Translate with detected language
    return real_time_translator(audio_stream, source_lang=language)
```

## Troubleshooting

### Problem: "No audio files found"
**Solution**: Make sure your audio files are in subdirectories and have supported extensions (.wav, .mp3, .flac, .ogg, .m4a)

### Problem: Low accuracy (below 70%)
**Solution**: 
- Add more training samples (aim for 200+ per language)
- Check if audio quality is consistent
- Increase training epochs to 150-200
- Verify your audio files are correctly labeled

### Problem: Model is too large
**Solution**: The model is about 7.36 MB, which is reasonable for mobile apps. If needed, you can quantize it:

```python
import tensorflow as tf

# Convert to TensorFlow Lite (smaller size)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save compressed model
with open('language_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Problem: Slow inference
**Solution**: 
- Use TensorFlow Lite (shown above)
- Process audio on a background thread
- Cache predictions for repeated audio

## Next Steps

1. **Start with sample data**: Use `generate_sample_data.py` to test the pipeline
2. **Collect your language data**: Record or download audio in your target languages
3. **Train your model**: Use the commands above
4. **Integrate**: Add the detection code to your translation app
5. **Iterate**: Collect more data and retrain as needed

## Additional Resources

- **Dataset sources**:
  - Common Voice (Mozilla): Free multilingual speech data
  - VoxForge: Open source speech corpus
  - LibriVox: Free public domain audiobooks
  
- **For production apps**:
  - Consider using data augmentation (pitch shift, time stretch)
  - Implement confidence thresholds (e.g., only translate if >80% confident)
  - Add error handling for unsupported languages
  - Log predictions to improve your model over time

## Questions?

If you need help with specific aspects:
- Data preparation
- Model performance tuning
- Integration with your app
- Handling specific languages

Feel free to ask in the comments!
