# AutoGluon Machine Learning Project

This repository is a complete AutoGluon-based machine learning project scaffold that supports automated model training, prediction, and evaluation.

## Project Structure

```
autogluon_project/
├── data/                  # data directory
│   ├── raw/              # raw data
│   └── processed/        # processed data
├── models/               # saved trained models
├── config/               # configuration files
│   └── config.yaml       # main configuration file
├── src/                  # source code
│   ├── data_loader.py    # data loading & preprocessing
│   ├── train.py          # model training
│   ├── predict.py        # model prediction
│   ├── evaluate.py       # model evaluation
│   └── utils.py          # helper utilities
├── notebooks/            # Jupyter notebooks
├── logs/                 # log files
├── results/              # results and outputs
├── main.py               # CLI entrypoint
├── requirements.txt      # project dependencies
└── README.md             # project overview
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Configure the project

Edit `config/config.yaml` to set data paths, model parameters, and other options.

### 2. Prepare data

Put your datasets into `data/raw/`. Supported formats include CSV, Excel, Parquet, JSON.

### 3. Train a model

```bash
python main.py --mode train --config config/config.yaml
```

### 4. Make predictions

```bash
python main.py --mode predict --config config/config.yaml --model_path models/your_model
```

### 5. Evaluate a model

```bash
python main.py --mode evaluate --config config/config.yaml --model_path models/your_model
```

## Features

- ✅ AutoML: leverage AutoGluon to find competitive models automatically
- ✅ Multi-task support: classification, regression, and more
- ✅ Data preprocessing: basic missing value handling and feature inspection
- ✅ Model evaluation: standard metrics and visualization helpers
- ✅ Configuration management: YAML-driven project configuration
- ✅ Logging: detailed logs for training and inference

## Supported Task Types (AutoGluon)

- **Tabular**: classification, regression
- **Vision**: image classification, object detection
- **Text**: text classification, sentiment analysis
- **Time Series**: forecasting
- **Multimodal**: combined data types

## Configuration Options

Key sections in the config file:
- `data`: data related settings (paths, target column, etc.)
- `model`: model training options (time limits, presets)
- `training`: training-related settings
- `evaluation`: evaluation metrics and options

## Examples

See the notebooks in `notebooks/` for example workflows and demonstrations.

## Notes

1. AutoGluon may download pre-trained components; the first run can take longer.
2. We recommend at least 4GB of RAM for training small to medium models.
3. Training time limits can be adjusted in the configuration file.

## References

- https://auto.gluon.ai/ (AutoGluon official docs)
- https://github.com/autogluon/autogluon (AutoGluon on GitHub)
- https://auto.gluon.ai/stable/tutorials/index.html (Tutorials)

## License

MIT
