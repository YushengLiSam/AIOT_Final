"""
Main entrypoint for the AutoGluon project.
Provides a CLI for training, prediction and evaluation.
"""

import os
import sys
import argparse
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    load_config, setup_logger, validate_config, 
    ensure_dir, Timer, list_models
)
from src.data_loader import DataLoader, load_and_prepare_data
from src.train import train_model
from src.predict import predict_from_model
from src.evaluate import evaluate_model
from autogluon.tabular import TabularPredictor
from loguru import logger


def train_mode(args, config):
    """Run training mode."""
    logger.info("="*60)
    logger.info("Mode: train")
    logger.info("="*60)
    
    with Timer("Data Loading"):
        # Load data
        loader = DataLoader(config)
        raw_data_path = config['data']['raw_data_path']
        
        if not os.path.exists(raw_data_path):
            logger.error(f"Data file not found: {raw_data_path}")
            return
        
        # Load and preprocess data
        df = loader.load_data(raw_data_path)
        df = loader.preprocess_data(df, is_train=True)

        # Split dataset
        train_df, val_df = loader.split_data(df)
    
    with Timer("Model Training"):
        # Train model
        predictor = train_model(
            config=config,
            train_data=train_df,
            tuning_data=val_df if args.use_validation else None,
            model_path=args.model_path
        )
    
    # Evaluate model if requested
    if args.evaluate and val_df is not None:
        logger.info("\n" + "="*60)
        logger.info("Evaluating on validation set")
        logger.info("="*60)
        evaluate_model(predictor, val_df, config)
    
    logger.info("\nTraining finished!")
    logger.info(f"Model saved to: {predictor.path}")


def predict_mode(args, config):
    """Run prediction mode."""
    logger.info("="*60)
    logger.info("Mode: predict")
    logger.info("="*60)
    
    # Check model path
    if not args.model_path:
        logger.error("Predict mode requires --model_path")
        return
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    with Timer("Data Loading"):
        # Load test data
        test_data_path = args.test_data or config['data'].get('test_data_path')
        
        if not test_data_path or not os.path.exists(test_data_path):
            logger.error(f"Test data file not found: {test_data_path}")
            return
        
        loader = DataLoader(config)
        test_df = loader.load_data(test_data_path)
        test_df = loader.preprocess_data(test_df, is_train=False)
    
    with Timer("Model Prediction"):
        # Predict
        save_path = args.output or os.path.join(
            config.get('evaluation', {}).get('report_path', 'results/'),
            'predictions.csv'
        )
        
        results = predict_from_model(
            model_path=args.model_path,
            data=test_df,
            config=config,
            save_path=save_path
        )
    
    logger.info("\nPrediction finished!")
    logger.info(f"Results saved to: {save_path}")


def evaluate_mode(args, config):
    """Run evaluation mode."""
    logger.info("="*60)
    logger.info("Mode: evaluate")
    logger.info("="*60)
    
    # Check model path
    if not args.model_path:
        logger.error("Evaluate mode requires --model_path")
        return
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    with Timer("Data Loading"):
        # Load test data
        test_data_path = args.test_data or config['data'].get('test_data_path')
        
        if not test_data_path or not os.path.exists(test_data_path):
            logger.error(f"Test data file not found: {test_data_path}")
            return
        
        loader = DataLoader(config)
        test_df = loader.load_data(test_data_path)
        
        # Ensure the test data contains the target column
        target_column = config['data']['target_column']
        if target_column not in test_df.columns:
            logger.error(f"Test data does not contain target column: {target_column}")
            return
    
    with Timer("Load Model"):
        # Load model
        predictor = TabularPredictor.load(args.model_path)
    
    with Timer("Model Evaluation"):
        # Evaluate
        evaluate_model(predictor, test_df, config)
    
    logger.info("\nEvaluation finished!")


def list_models_mode(args):
    """List all saved models."""
    logger.info("="*60)
    logger.info("Saved models")
    logger.info("="*60)
    
    models_dir = args.models_dir or "models/"
    models = list_models(models_dir)
    
    if not models:
        logger.info("No models found")
        return
    
    from datetime import datetime
    
    for i, model in enumerate(models, 1):
        logger.info(f"\n{i}. {model['name']}")
        logger.info(f"   Path: {model['path']}")
        logger.info(f"   Size: {model['size']}")
        logger.info(f"   Modified: {datetime.fromtimestamp(model['modified']).strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main entrypoint for the CLI."""
        parser = argparse.ArgumentParser(
                description="AutoGluon automated machine learning project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Train a model:
        python main.py --mode train --config config/config.yaml

    Train with validation and evaluate:
        python main.py --mode train --config config/config.yaml --use_validation --evaluate

    Predict:
        python main.py --mode predict --model_path models/ag_model_xxx --test_data data/raw/test.csv

    Evaluate:
        python main.py --mode evaluate --model_path models/ag_model_xxx --test_data data/raw/test.csv

    List saved models:
        python main.py --mode list
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'predict', 'evaluate', 'list'],
        help='Mode to run: train, predict, evaluate, list'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Config file path (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        help='Model path (for predict/evaluate)'
    )
    
    parser.add_argument(
        '--test_data',
        type=str,
        help='Test data path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path'
    )
    
    parser.add_argument(
        '--use_validation',
        action='store_true',
        help='Use validation set during training'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model immediately after training'
    )
    
    parser.add_argument(
        '--models_dir',
        type=str,
        default='models/',
        help='Models directory (used for list mode)'
    )
    
    args = parser.parse_args()
    
    # List mode does not require configuration file
    if args.mode == 'list':
        list_models_mode(args)
        return
    
    # Load configuration
    try:
        config = load_config(args.config)

        # Validate configuration
        if not validate_config(config):
            logger.error("Configuration validation failed")
            return

        # Setup logging
        setup_logger(config)

    except Exception as e:
        print(f"Failed to load config file: {e}")
        return
    
    # Execute action according to selected mode
    try:
        if args.mode == 'train':
            train_mode(args, config)
        elif args.mode == 'predict':
            predict_mode(args, config)
        elif args.mode == 'evaluate':
            evaluate_mode(args, config)
    
    except KeyboardInterrupt:
        logger.warning("\nOperation interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
