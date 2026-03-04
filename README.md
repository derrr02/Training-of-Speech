# Training-of-Speech

A CNN-based speech data training framework for audio classification tasks.

## Features

- **CNN Architecture**: Deep convolutional neural network optimized for speech/audio data
- **MFCC Feature Extraction**: Uses Mel-frequency cepstral coefficients for audio representation
- **Flexible Input**: Supports various audio formats (WAV, MP3, FLAC, OGG, M4A)
- **Easy to Use**: Simple command-line interface
- **Model Checkpointing**: Automatically saves the best model during training
- **Early Stopping**: Prevents overfitting with early stopping callbacks

## Installation

1. Clone this repository:
```bash
git clone https://github.com/derrr02/Training-of-Speech.git
cd Training-of-Speech
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Organize your audio files in the following directory structure:

```
data/
├── class1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── class2/
│   ├── audio3.wav
│   ├── audio4.wav
│   └── ...
└── class3/
    ├── audio5.wav
    └── ...
```

Each subdirectory represents a class/category, and should contain audio files for that class.

## Usage

### Basic Training

```bash
python train_speech_cnn.py --data_dir /path/to/data
```

### Advanced Options

```bash
python train_speech_cnn.py \
    --data_dir /path/to/data \
    --epochs 100 \
    --batch_size 64 \
    --validation_split 0.2 \
    --model_save_path my_model.h5 \
    --sample_rate 16000 \
    --duration 3
```

### Parameters

- `--data_dir`: Directory containing audio files organized by class (required)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--validation_split`: Fraction of data for validation (default: 0.2)
- `--model_save_path`: Path to save the trained model (default: speech_cnn_model.h5)
- `--sample_rate`: Sample rate for audio processing (default: 16000 Hz)
- `--duration`: Duration in seconds to pad/truncate audio (default: 3)

## Model Architecture

The CNN model consists of:
- 3 Convolutional blocks with BatchNormalization and MaxPooling
- Dropout layers for regularization
- 2 Dense layers for classification
- Softmax output layer

## Output

After training, the script will save:
1. `speech_cnn_model.h5` - The trained model
2. `speech_cnn_model_labels.json` - Label encoding information

## Use Cases

This framework can be used for various speech/audio tasks:
- **Speech Recognition**: Recognize spoken words or commands
- **Speaker Identification**: Identify who is speaking
- **Emotion Detection**: Detect emotions from speech
- **Audio Classification**: Classify any type of audio data
- **Language Identification**: Identify the language being spoken
- **Translation Apps**: Detect source language for translation (see [TRANSLATION_APP_GUIDE.md](TRANSLATION_APP_GUIDE.md))

## Example: Training on Speech Commands

```bash
# Assuming you have the Google Speech Commands dataset
python train_speech_cnn.py \
    --data_dir ./speech_commands_data \
    --epochs 50 \
    --batch_size 32
```

## For Translation Apps

If you're building a translation app and need to train a model for language detection, see our detailed guide:

📖 **[Translation App Guide](TRANSLATION_APP_GUIDE.md)** - Complete tutorial for training speech models for translation

Example integration:
```bash
# Train language detection model
python train_speech_cnn.py --data_dir translation_data --model_save_path language_classifier.h5

# Use in translation app
python translation_app_example.py --audio_file speech.wav
```

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- librosa 0.10+
- NumPy 1.23+
- scikit-learn 1.2+
- soundfile 0.12+

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.