# Emotion Classification Source Code

This directory contains the core modules for facial landmark-based emotion classification.

## File Structure

### 1. `features_extraction.py` (核心特征工程)
**Purpose**: Extract position-invariant features from facial landmarks

**Key Components**:
- `LandmarkFeatureExtractor` class
  - `extract_center_distances()`: Translation-invariant (106 dims)
  - `extract_angles()`: Rotation-invariant (200 dims)
  - `extract_distance_ratios()`: Scale-invariant (150 dims)
  - `extract_features()`: Main method returning 456-dim vector
- `extract_features_from_landmarks()`: Batch feature extraction function

**Input**: Landmark arrays (n, 106, 2)
**Output**: Feature matrix (n, 456)

---

### 2. `data_loader.py` (数据加载)
**Purpose**: Load facial landmark data from pkl files

**Key Functions**:
- `load_landmark_data(data_root, emotion_labels)`: Load pkl files from directory

**Input**: Directory with emotion subdirectories containing .pkl files
**Output**: Landmark arrays, emotion labels, file paths

**Note**: Only handles data loading, feature extraction is in `features_extraction.py`

---

### 3. `train.py` (模型训练)
**Purpose**: Train emotion classification model using AutoGluon

**Key Components**:
- `prepare_dataframe()`: Convert features to AutoGluon DataFrame format
- `main()`: Complete training pipeline

**Training Pipeline**:
1. Load landmark data (from data_loader)
2. Extract features (from features_extraction)
3. Prepare DataFrame
4. Train AutoGluon models (RF, LightGBM, XGBoost, CatBoost)
5. Evaluate and save results

**Command**:
```bash
python src/train.py --data_root data/landmarks/affectnet \
                    --save_dir models/emotion_model \
                    --time_limit 600 \
                    --presets medium_quality
```

**Output**: Trained model, confusion matrix, classification report, feature importance

---

### 4. `evaluate.py` (模型评估与预测)
**Purpose**: Predict emotion using trained model

**Key Components**:
- `AutoGluonEmotionPredictor` class
  - `landmarks_to_dataframe()`: Convert landmarks to prediction format
  - `predict_from_landmarks()`: Predict from landmark array
  - `predict_from_pkl()`: Predict from pkl file
  - `predict_from_image()`: Extract landmarks then predict
- `visualize_prediction()`: Visualize results on image
- `main()`: Command-line interface

**Command**:
```bash
# From pkl file
python src/evaluate.py --model_dir models/emotion_model \
                       --input data/test.pkl

# From image
python src/evaluate.py --model_dir models/emotion_model \
                       --input image.jpg \
                       --output result.jpg
```

**Output**: Predicted emotion, confidence, probabilities, visualization

---

### 5. `utils.py` (工具函数)
**Purpose**: Utility functions for configuration and logging

**Key Functions**:
- `setup_logger()`: Configure loguru logger
- `load_config()`: Load YAML configuration
- `save_config()`: Save YAML configuration
- `ensure_dir()`: Create directory if not exists
- `save_json()` / `load_json()`: JSON file operations
- `format_time()`: Format seconds to readable string
- Other helper utilities

---

## Module Dependencies

```
features_extraction.py (独立模块)
    └─ LandmarkFeatureExtractor
    └─ extract_features_from_landmarks()

data_loader.py (独立模块)
    └─ load_landmark_data()

train.py (训练脚本)
    ├─ import: data_loader.load_landmark_data
    ├─ import: features_extraction.LandmarkFeatureExtractor
    ├─ import: features_extraction.extract_features_from_landmarks
    └─ import: utils.setup_logger

evaluate.py (评估脚本)
    └─ import: utils.setup_logger

utils.py (独立工具模块)
```

---

## Workflow

```
1. Extract landmarks (demo/landmark_extraction.py)
   └─> data/landmarks/affectnet/[emotion]/*.pkl

2. Train model (src/train.py)
   ├─> data_loader.load_landmark_data()
   ├─> features_extraction.extract_features_from_landmarks()
   ├─> prepare_dataframe()
   └─> AutoGluon.fit()
   └─> models/emotion_model/

3. Evaluate/Predict (src/evaluate.py)
   ├─> AutoGluonEmotionPredictor.load()
   ├─> predict_from_image() / predict_from_pkl()
   └─> visualize_prediction()
   └─> result.jpg
```

---

## Design Principles

### 模块职责分离
- **features_extraction.py**: 特征工程逻辑
- **data_loader.py**: 数据IO操作
- **train.py**: 训练流程编排
- **evaluate.py**: 推理和评估
- **utils.py**: 通用工具函数

### 位置不变性特征设计
- **Translation invariance**: 中心距离 (landmarks → centroid)
- **Rotation invariance**: 三点角度 (固定随机种子)
- **Scale invariance**: 距离比值 (归一化)

**Total**: 456 dimensions (106 + 200 + 150)

### 模型选择
**Fast inference, CPU-optimized**:
- RandomForest: <1ms inference, parallel trees
- LightGBM: <1ms inference, minimal memory
- XGBoost: <2ms inference, highest accuracy
- CatBoost: <2ms inference, robust to overfitting

**No GPU required**: Traditional ML models, optimized for CPU inference
**AutoGluon**: Automatic model selection and ensemble
