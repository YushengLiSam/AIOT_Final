# Emotion Classification from Facial Landmarks

Position-invariant emotion classification using facial landmark relative positions.

## Overview

This module provides emotion classification based on **position-invariant features** extracted from facial landmarks. The model only considers the relative geometric relationships between keypoints, making it robust to face position, rotation, and scale variations.

## Features

- **Position-Invariant Feature Extraction**: Uses relative distances, angles, and distance ratios
- **Multiple Model Support**: Random Forest, Gradient Boosting, SVM
- **106-Point Face6 Landmarks**: High-precision facial keypoints
- **8 Emotion Classes**: anger, contempt, disgust, fear, happy, neutral, sad, surprise

## Architecture

### Feature Extraction (456 dimensions)

1. **Center Distances** (106 dims): Distance from each landmark to centroid
2. **Angles** (200 dims): Angles formed by random landmark triplets  
3. **Distance Ratios** (150 dims): Scale-invariant distance ratios

### Models

- **Random Forest**: 200 trees, max_depth=20
- **Gradient Boosting**: 200 estimators, learning_rate=0.1
- **SVM**: RBF kernel, C=10.0

## Usage

### 1. Extract Landmarks from Images

```bash
python demo/landmark_extraction.py
```

This extracts Face6 landmarks from AffectNet images and saves to `data/landmarks/affectnet/`.

### 2. Train Emotion Classifier

```bash
# Train all models (Random Forest + SVM)
python src/train_emotion.py

# Train specific models
python src/train_emotion.py --model_types random_forest gradient_boosting

# Custom data path
python src/train_emotion.py --data_root data/landmarks/custom --save_dir models/custom_classifier
```

**Arguments**:
- `--data_root`: Path to landmark data directory (default: `data/landmarks/affectnet`)
- `--save_dir`: Directory to save trained models (default: `models/emotion_classifier`)
- `--model_types`: Model types to train (default: `random_forest svm`)
- `--test_size`: Test set ratio (default: 0.2)
- `--random_state`: Random seed (default: 42)

### 3. Predict Emotion

#### From pkl file:
```bash
python src/predict_emotion.py --input data/landmarks/affectnet/happy/001.pkl
```

#### From image:
```bash
python src/predict_emotion.py --input test_image.jpg --output result.jpg
```

#### Specify model:
```bash
python src/predict_emotion.py --input image.jpg --model_type svm --device cuda
```

**Arguments**:
- `--input`: Input image or pkl file path (required)
- `--model_dir`: Directory containing trained model (default: `models/emotion_classifier`)
- `--model_type`: Model to use: `random_forest`, `gradient_boosting`, `svm` (default: `random_forest`)
- `--output`: Output path for visualization (default: auto-generated)
- `--detector`: Object detector: `YOLOX`, `RTMDet` (default: `YOLOX`)
- `--device`: Device: `cpu`, `cuda` (default: `cpu`)

## File Structure

```
src/
├── emotion_features.py    # Feature extraction (LandmarkFeatureExtractor)
├── emotion_data.py        # Data loading functions
├── train_emotion.py       # Training script
└── predict_emotion.py     # Prediction script

demo/
└── landmark_extraction.py # Extract landmarks from images

models/emotion_classifier/
├── random_forest_model.pkl
├── svm_model.pkl
├── scaler.pkl
├── feature_extractor.pkl
├── label_encoder.pkl
├── *_confusion_matrix.png
└── *_classification_report.txt
```

## Example Workflow

```bash
# Step 1: Extract landmarks (run once)
python demo/landmark_extraction.py

# Step 2: Train models
python src/train_emotion.py --model_types random_forest svm

# Step 3: Predict on new images
python src/predict_emotion.py --input test.jpg --model_type random_forest
```

## Output

### Training Output
- Trained models: `models/emotion_classifier/*_model.pkl`
- Confusion matrices: `*_confusion_matrix.png`
- Classification reports: `*_classification_report.txt`
- Preprocessors: `scaler.pkl`, `feature_extractor.pkl`, `label_encoder.pkl`

### Prediction Output
- Console: Emotion, confidence, all class probabilities
- Visualization: Image with landmarks and prediction overlay

## Performance

Typical results on AffectNet (31,002 samples):

| Model | Test Accuracy | Training Time |
|-------|--------------|---------------|
| Random Forest | ~65-70% | ~2-5 min |
| Gradient Boosting | ~63-68% | ~15-30 min |
| SVM | ~60-65% | ~10-20 min |

*Actual performance depends on data quality and hyperparameters*

## Technical Details

### Why Position-Invariant Features?

Traditional CNNs learn absolute pixel patterns, which can be sensitive to face position/scale. Our approach:

1. **Translation Invariance**: Using centroid-relative distances
2. **Rotation Invariance**: Using angles between landmarks
3. **Scale Invariance**: Using distance ratios

This makes the model robust to:
- Face position in image
- Face size variations
- Camera distance changes
- Head rotation (within limits)

### Feature Computation

For N=106 landmarks:
- Center distances: O(N) = 106 features
- Angles: Sampled 200 triplets = 200 features
- Distance ratios: Sampled 100 pairs = 150 features
- **Total: 456 features**

## Dependencies

```
numpy
scikit-learn
joblib
opencv-python
matplotlib
seaborn
tqdm
loguru
rtmlib
onnxruntime
```

Install with:
```bash
pip install -r requirements.txt
```

## Notes

- First run downloads Face6 model (~60MB) and YOLOX detector (~7MB)
- GPU support requires `onnxruntime-gpu` and CUDA
- For best results, use images with clear, frontal faces
- Model performance improves with more training data

## Troubleshooting

### "No face detected"
- Ensure face is clearly visible and well-lit
- Try different detector: `--detector RTMDet`

### "Model not found"
- Run training first: `python src/train_emotion.py`
- Check model path: `--model_dir models/emotion_classifier`

### Low accuracy
- Check data quality and label correctness
- Try different model: `--model_type gradient_boosting`
- Increase training data
- Tune hyperparameters in `train_emotion.py`
