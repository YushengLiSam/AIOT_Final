"""
Quick start examples.
Demonstrates how to quickly train and predict using this project.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Import project modules
from src.utils import load_config, setup_logger
from src.train import train_model
from src.predict import ModelPredictor
from src.evaluate import evaluate_model


def create_sample_classification_data():
    """Create a sample classification dataset."""
    print("Creating sample classification dataset...")
    
    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save data
    train_df.to_csv('data/raw/train.csv', index=False)
    test_df.to_csv('data/raw/test.csv', index=False)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Data saved to data/raw/")
    
    return train_df, test_df


def create_sample_regression_data():
    """Create a sample regression dataset."""
    print("Creating sample regression dataset...")
    
    # Generate data
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save data
    train_df.to_csv('data/raw/train.csv', index=False)
    test_df.to_csv('data/raw/test.csv', index=False)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Data saved to data/raw/")
    
    return train_df, test_df


def quick_start_classification():
    """Quick start for a classification task."""
    print("\n" + "="*60)
    print("AutoGluon classification quick start example")
    print("="*60 + "\n")
    
    # 1. Create sample data
    train_df, test_df = create_sample_classification_data()
    
    # 2. Load config
    config = load_config('config/config.yaml')
    setup_logger(config)
    
    # 3. Adjust config for a quick demo
    config['model']['time_limit'] = 60  # limit training to 60 seconds
    config['model']['presets'] = 'medium_quality'  # use medium-quality preset
    
    # 4. Train model
    print("\nTraining model...")
    predictor = train_model(
        config=config,
        train_data=train_df,
        model_path='models/quick_start_classification'
    )
    
    # 5. Evaluate model
    print("\nEvaluating model...")
    evaluate_model(predictor, test_df, config)
    
    # 6. Predict
    print("\nRunning prediction...")
    model_predictor = ModelPredictor('models/quick_start_classification', config)
    predictions = model_predictor.predict(test_df.drop(columns=['target']))
    
    print(f"\nPrediction samples (first 10):")
    print(predictions.head(10))
    
    print("\n" + "="*60)
    print("Quick start example finished!")
    print("="*60)


def quick_start_regression():
    """Quick start for a regression task."""
    print("\n" + "="*60)
    print("AutoGluon regression quick start example")
    print("="*60 + "\n")
    
    # 1. Create sample data
    train_df, test_df = create_sample_regression_data()
    
    # 2. Load config
    config = load_config('config/config.yaml')
    setup_logger(config)
    
    # 3. Adjust config for a quick demo
    config['model']['time_limit'] = 60  # limit training to 60 seconds
    config['model']['presets'] = 'medium_quality'  # use medium-quality preset
    config['model']['eval_metric'] = 'rmse'  # regression uses RMSE
    config['model']['problem_type'] = 'regression'  # explicitly set regression
    
    # 4. Train model
    print("\nTraining model...")
    predictor = train_model(
        config=config,
        train_data=train_df,
        model_path='models/quick_start_regression'
    )
    
    # 5. Evaluate model
    print("\nEvaluating model...")
    evaluate_model(predictor, test_df, config)
    
    # 6. Predict
    print("\nRunning prediction...")
    model_predictor = ModelPredictor('models/quick_start_regression', config)
    predictions = model_predictor.predict(test_df.drop(columns=['target']))
    
    print(f"\nPrediction samples (first 10):")
    print(predictions.head(10))
    
    print("\nTrue vs Predicted (first 10):")
    comparison = pd.DataFrame({
        'True': test_df['target'].values[:10],
        'Predicted': predictions.values[:10]
    })
    print(comparison)
    
    print("\n" + "="*60)
    print("Quick start example finished!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoGluon quick start example')
    parser.add_argument(
        '--task',
        type=str,
        default='classification',
        choices=['classification', 'regression'],
        help='Task type: classification or regression'
    )
    
    args = parser.parse_args()
    
    if args.task == 'classification':
        quick_start_classification()
    else:
        quick_start_regression()
