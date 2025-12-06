# AutoGluon Project Guide

## 📋 目录

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Usage Details](#usage-details)
5. [Configuration](#configuration)
6. [FAQ](#faq)

---

## Project Overview

This repository provides a complete AutoGluon-based automated machine learning scaffold that supports:

- ✅ Auto model selection: AutoGluon automatically selects and optimizes competitive models
- ✅ Multiple task types: classification, regression, time series, and more
- ✅ End-to-end workflow: data loading → training → prediction → evaluation
- ✅ Config-driven: manage parameters through YAML files
- ✅ Visual reports: automated generation of evaluation reports and charts

## Installation

### 1. Create a virtual environment (recommended)

```bash
# using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# or using conda
conda create -n autogluon python=3.9
conda activate autogluon
```

### 2. Install dependencies

```bash
cd autogluon_project
pip install -r requirements.txt
```

Note: the first installation may take a while because AutoGluon installs multiple dependencies.

### 3. Verify installation

```bash
python -c "import autogluon; print(autogluon.__version__)"
```

## Quick Start

### Option 1: Use the quick start script

```bash
# classification example
python quick_start.py --task classification

# regression example
python quick_start.py --task regression
```

This will generate sample data, train a model, and run evaluation automatically.

### Option 2: Use your own data

#### Step 1: Prepare data

Place your dataset files (CSV/Excel/etc.) into `data/raw/`:

```
data/raw/
├── train.csv  # training data
└── test.csv   # test data
```

Example data format:

```csv
feature1,feature2,feature3,target
1.2,cat,0.5,0
2.3,dog,0.8,1
3.4,cat,0.3,1
```

#### Step 2: Configure parameters

Edit `config/config.yaml` to set paths and options, e.g.:

```yaml
data:
  raw_data_path: "data/raw/train.csv"
  test_data_path: "data/raw/test.csv"
  target_column: "target"  # change to your target column

model:
  time_limit: 600  # training time limit (seconds)
  presets: "best_quality"  # model preset
```

#### Step 3: Train the model

```bash
python main.py --mode train --config config/config.yaml
```

#### Step 4: Evaluate the model

```bash
python main.py --mode evaluate \
  --model_path models/ag_model_20231201_120000 \
  --test_data data/raw/test.csv
```

#### Step 5: Predict

```bash
python main.py --mode predict \
  --model_path models/ag_model_20231201_120000 \
  --test_data data/raw/test.csv \
  --output results/predictions.csv
```

## Usage Details

### 1. Training mode

#### Basic training

```bash
python main.py --mode train --config config/config.yaml
```

#### Training with validation

```bash
python main.py --mode train \
  --config config/config.yaml \
  --use_validation \
  --evaluate
```

Arguments:
- `--use_validation`: automatically split training data into training and validation sets
- `--evaluate`: evaluate the model on the validation set immediately after training

#### Specify model save path

```bash
python main.py --mode train \
  --config config/config.yaml \
  --model_path models/my_custom_model
```

### 2. Prediction mode

```bash
python main.py --mode predict \
  --model_path models/ag_model_xxx \
  --test_data data/raw/test.csv \
  --output results/predictions.csv
```

The prediction output is saved as a CSV file and includes:
- original features
- predicted labels
- predicted probabilities (for classification)

### 3. Evaluation mode

```bash
python main.py --mode evaluate \
  --model_path models/ag_model_xxx \
  --test_data data/raw/test.csv
```

The evaluation report contains:
- various metrics (accuracy, F1, RMSE, etc.)
- confusion matrix (classification)
- ROC curve (classification)
- feature importance plots

### 4. List saved models

```bash
python main.py --mode list --models_dir models/
```

## Configuration

### Data configuration

```yaml
data:
  raw_data_path: "data/raw/train.csv"  # training data path
  test_data_path: "data/raw/test.csv"  # test data path
  target_column: "target"               # target column name
  train_split: 0.8                      # train split ratio
  random_state: 42                      # random seed
```

### Model configuration

```yaml
model:
  save_path: "models/"                  # model save directory
  time_limit: 600                       # training time limit (seconds)
  presets: "best_quality"               # preset quality
  eval_metric: "accuracy"               # evaluation metric
  problem_type: null                    # problem type (null = auto-detect)
  hyperparameter_tune: true             # whether to tune hyperparameters
  num_gpus: 0                          # number of GPUs
```

### Preset quality options

- `best_quality`: highest quality, longest training time
- `high_quality`: high quality
- `good_quality`: good quality
- `medium_quality`: medium quality (recommended for quick tests)
- `low_quality`: low quality, fastest training

### Evaluation metric options

**Classification**:
- `accuracy`: accuracy
- `f1`: F1 score
- `roc_auc`: ROC AUC
- `log_loss`: log loss

**Regression**:
- `rmse`: root mean squared error
- `mae`: mean absolute error
- `r2`: R² score

## FAQ

### Q1: Training takes too long — what can I do?

Answer: adjust the following parameters:

```yaml
model:
  time_limit: 300  # reduce time limit
  presets: "medium_quality"  # use a lower-quality preset
```

### Q2: Not enough memory — what can I do?

Answer:
1. Reduce the training time limit
2. Use a smaller dataset
3. Set `save_space: true` to drop intermediate models

```yaml
model:
  save_space: true
  keep_only_best: true
```

### Q3: How to use GPU?

Answer: set the following in the config file:

```yaml
model:
  num_gpus: 1  # use 1 GPU
```

Make sure GPU-enabled dependencies are installed.

### Q4: How to handle class imbalance?

Answer: AutoGluon handles class imbalance automatically in many cases, but you can also:

1. Use weighted metrics (e.g., `f1_weighted`)
2. Perform resampling during preprocessing

### Q5: What data formats are supported?

Answer: supported formats include:
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- Parquet (`.parquet`)
- JSON (`.json`)

### Q6: How to view feature importance?

Answer: in Python:

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor.load('models/ag_model_xxx')
importance = predictor.feature_importance()
print(importance)
```

### Q7: Can I customize models?

Answer: Yes — modify training parameters or pass custom hyperparameters:

```python
from src.train import ModelTrainer

trainer = ModelTrainer(config)
predictor = trainer.train(
    train_data=train_df,
    hyperparameters={
        'GBM': {},
        'CAT': {},
        'XGB': {},
        'RF': {}
    }
)
```

### Advanced usage

## Advanced usage

### Python API

In addition to the CLI, you can use the library from Python scripts:

```python
from src.utils import load_config
from src.data_loader import DataLoader
from src.train import train_model
from src.predict import ModelPredictor
from src.evaluate import evaluate_model

# Load config
config = load_config('config/config.yaml')

# Load data
loader = DataLoader(config)
train_df = loader.load_data('data/raw/train.csv')

# Train model
predictor = train_model(config, train_df)

# Predict
model = ModelPredictor(predictor.path, config)
predictions = model.predict(test_df)

# Evaluate
evaluate_model(predictor, test_df, config)
```

### Jupyter Notebook

Create notebooks under the `notebooks/` directory. Example header:

```python
import sys
sys.path.append('..')

from src.utils import load_config
from src.data_loader import DataLoader
# ... other imports
```

## Project structure

```
autogluon_project/
├── config/              # configuration files
│   └── config.yaml
├── data/               # data directory
│   ├── raw/           # raw data
│   └── processed/     # processed data
├── models/            # saved models
├── src/               # source code
│   ├── data_loader.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/         # Jupyter notebooks
├── logs/             # log files
├── results/          # output results
├── main.py           # CLI entrypoint
├── quick_start.py    # quick start script
└── requirements.txt  # dependencies
```

## Getting help

View CLI help:

```bash
python main.py --help
```

View help for a specific mode:

```bash
python main.py --mode train --help
```

## More resources

- https://auto.gluon.ai/ (AutoGluon documentation)
- https://github.com/autogluon/autogluon (AutoGluon GitHub)
- https://auto.gluon.ai/stable/tutorials/index.html (Tutorials)
