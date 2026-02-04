# Quick Start Guide

This guide will help you get started with training speech data using CNN in just a few minutes.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Your Data

### Option A: Generate Sample Data (for testing)

```bash
python generate_sample_data.py --output_dir sample_data --num_samples 30
```

This will create a `sample_data` directory with 3 classes of synthetic audio samples.

### Option B: Use Your Own Data

Organize your audio files in this structure:

```
my_data/
├── class1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── class2/
│   ├── audio3.wav
│   └── ...
└── class3/
    └── ...
```

## Step 3: Train the Model

```bash
python train_speech_cnn.py --data_dir sample_data --epochs 50
```

This will:
- Load and preprocess your audio data
- Train a CNN model
- Save the best model as `speech_cnn_model.h5`
- Save class labels as `speech_cnn_model_labels.json`

## Step 4: Make Predictions

```bash
python predict_speech.py --audio_file path/to/your/audio.wav
```

This will output the top-3 predicted classes with their probabilities.

## Advanced Usage

### Training with Custom Parameters

```bash
python train_speech_cnn.py \
    --data_dir my_data \
    --epochs 100 \
    --batch_size 64 \
    --validation_split 0.2 \
    --model_save_path my_custom_model.h5 \
    --sample_rate 16000 \
    --duration 3
```

### Prediction with Different Model

```bash
python predict_speech.py \
    --model_path my_custom_model.h5 \
    --audio_file test_audio.wav \
    --top_k 5
```

## Example Output

### Training Output
```
============================================================
CNN Speech Training
============================================================

1. Loading data...
Processing class: class1
Processing class: class2
Processing class: class3
Loaded 300 samples from 3 classes

2. Splitting data...
Training samples: 240
Validation samples: 60

3. Building model...
Model: "sequential"
...
Total params: 1,930,371 (7.36 MB)

4. Training model...
Epoch 1/50
...
Validation Accuracy: 0.9500

6. Model saved to: speech_cnn_model.h5
```

### Prediction Output
```
Analyzing: test_audio.wav
------------------------------------------------------------

Predictions:
1. class2: 87.42%
2. class1: 10.33%
3. class3: 2.25%

============================================================
Top prediction: class2 (87.42%)
```

## Common Issues

### Issue: "No audio files found"
**Solution**: Make sure your data is organized with subdirectories for each class, and contains supported audio formats (.wav, .mp3, .flac, .ogg, .m4a).

### Issue: Memory errors during training
**Solution**: Reduce the batch size with `--batch_size 16` or `--batch_size 8`.

### Issue: Low accuracy
**Solutions**:
- Increase the number of training samples
- Increase the number of epochs
- Adjust the duration parameter to match your audio length
- Ensure your data is properly labeled

## Tips for Better Results

1. **More Data**: More training samples generally lead to better accuracy
2. **Balanced Classes**: Try to have similar numbers of samples for each class
3. **Audio Quality**: Use clean, clear audio recordings
4. **Consistent Length**: If your audio has consistent length, adjust the `--duration` parameter accordingly
5. **Early Stopping**: The model uses early stopping to prevent overfitting - training will stop automatically if validation loss stops improving

## Next Steps

- Try with your own audio data
- Experiment with different hyperparameters
- Use the trained model in your applications
- Integrate with other Python projects

For more details, see the main [README.md](README.md).
