"""
Emotion Classification Training with AutoGluon
Automatically train and select best model for emotion classification
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from autogluon.tabular import TabularPredictor
from loguru import logger

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_landmark_data, load_feature_data
from src.features_extraction import LandmarkFeatureExtractor, extract_features_from_landmarks
from src.feature_selection import select_discriminative_features, load_feature_config


def stratified_split_by_class(df, test_size, val_size, random_state=42):
    """
    Perform stratified sampling for each class separately to ensure uniform distribution
    across training, validation, and test sets.
    
    Args:
        df: Complete dataset
        test_size: Test set ratio
        val_size: Validation set ratio (relative to total dataset)
        random_state: Random seed
    
    Returns:
        train_df, val_df, test_df: Split datasets
    """
    from sklearn.model_selection import train_test_split
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    emotion_col = 'emotion'
    emotions = df[emotion_col].unique()
    
    logger.info("Stratified split by class:")
    logger.info(f"{'Emotion':<12} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
    logger.info("-" * 50)
    
    for emotion in sorted(emotions):
        # Get all samples for current class
        emotion_df = df[df[emotion_col] == emotion].copy()
        n_total = len(emotion_df)
        
        # First split: separate test set
        train_val_df, test_df_class = train_test_split(
            emotion_df, 
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        # Second split: separate validation set from training set
        val_ratio = val_size / (1 - test_size)
        train_df_class, val_df_class = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True
        )
        
        train_dfs.append(train_df_class)
        val_dfs.append(val_df_class)
        test_dfs.append(test_df_class)
        
        # Print distribution for each class
        logger.info(f"{emotion:<12} {n_total:<8} {len(train_df_class):<8} "
                   f"{len(val_df_class):<8} {len(test_df_class):<8}")
    
    # Merge all classes and shuffle
    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
    
    logger.info("-" * 50)
    logger.info(f"{'TOTAL':<12} {len(df):<8} {len(train_df):<8} "
               f"{len(val_df):<8} {len(test_df):<8}")
    
    return train_df, val_df, test_df


def prepare_dataframe(X_features, y_labels, feature_extractor):
    """
    Convert features and labels to DataFrame for AutoGluon
    """
    # Create feature column names
    # Note: Using 68 face keypoints from COCO wholebody (indices 23-90)
    n_center = 68 if feature_extractor.use_distances else 0
    
    # If using fixed feature selection, use the actual number of selected features
    if feature_extractor.feature_config is not None:
        n_angles = len(feature_extractor.angle_triplets) if feature_extractor.use_angles and feature_extractor.angle_triplets else 0
        n_ratios = len(feature_extractor.ratio_quadruplets) if feature_extractor.use_ratios and feature_extractor.ratio_quadruplets else 0
    else:
        # Legacy: random sampling
        n_angles = 200 if feature_extractor.use_angles else 0
        n_ratios = 150 if feature_extractor.use_ratios else 0
    
    columns = []
    if feature_extractor.use_distances:
        columns += [f'center_dist_{i}' for i in range(n_center)]
    if feature_extractor.use_angles:
        columns += [f'angle_{i}' for i in range(n_angles)]
    if feature_extractor.use_ratios:
        columns += [f'ratio_{i}' for i in range(n_ratios)]
    
    # Create DataFrame
    df = pd.DataFrame(X_features, columns=columns)
    df['emotion'] = y_labels
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Train emotion classifier with AutoGluon')
    parser.add_argument('--data_root', type=str, default='data/landmarks/affectnet',
                       help='Path to landmark data directory')
    parser.add_argument('--save_dir', type=str, default='models/emotion_classifier3',
                       help='Directory to save trained model')
    parser.add_argument('--val_size', type=float, default=0.3,
                       help='Validation set size ratio (from training data, default: 0.3)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio')
    parser.add_argument('--time_limit', type=int, default=3000,
                       help='Time limit for training in seconds (default: 3000=50min)')
    parser.add_argument('--presets', type=str, default='medium_quality',
                       choices=['best_quality', 'high_quality', 'good_quality', 
                               'medium_quality', 'optimize_for_deployment'],
                       help='AutoGluon preset quality level')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--exclude_models', type=str, nargs='+',
                       default=['NN_TORCH', 'FASTAI', 'AG_TEXT_NN', 'VW'],
                       help='Models to exclude (default: exclude slow neural networks)')
    parser.add_argument('--num_cpus', type=int, default=16,
                       help='Number of CPU cores to use (default: 16)')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use (default: 1)')
    parser.add_argument('--use_feature_selection', type=bool, default=True,
                       help='Use ANOVA F-test based feature selection for angles and ratios')
    parser.add_argument('--n_samples_per_class', type=int, default=100,
                       help='Number of samples per class for feature selection (default: 100)')
    parser.add_argument('--n_angle_features', type=int, default=200,
                       help='Number of angle features to select (default: 200)')
    parser.add_argument('--n_ratio_features', type=int, default=200,
                       help='Number of ratio features to select (default: 200)')
    
    args = parser.parse_args()
    
    # Emotion labels (8 classes from AffectNet)
    emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 
                     'happy', 'neutral', 'sad', 'surprise']
    
    # Step 1: Load landmark data
    logger.info("="*60)
    logger.info("STEP 1: Loading landmark data")
    logger.info("="*60)
    X_landmarks, y_labels, file_paths = load_landmark_data(args.data_root, emotion_labels)
    
    if len(X_landmarks) == 0:
        logger.error("No data loaded!")
        return
    
    # Step 1.5: Feature selection (if enabled)
    feature_config = None
    feature_config_path = Path(args.data_root) / 'feature_config.txt'
    
    if args.use_feature_selection:
        logger.info("\n" + "="*60)
        logger.info("STEP 1.5: Discriminative Feature Selection (ANOVA F-test)")
        logger.info("="*60)
        
        # Check if feature_config.txt already exists
        if feature_config_path.exists():
            logger.info(f"Found existing feature config: {feature_config_path}")
            logger.info("Loading existing feature configuration...")
            feature_config = load_feature_config(str(feature_config_path))
            logger.info(f"  Loaded {len(feature_config['angle_triplets'])} angle features")
            logger.info(f"  Loaded {len(feature_config['ratio_quadruplets'])} ratio features")
        else:
            logger.info(f"No existing feature config found")
            logger.info("Performing feature selection...")
            
            feature_config = select_discriminative_features(
                X_landmarks=X_landmarks,
                y_labels=y_labels,
                emotion_labels=emotion_labels,
                data_root=args.data_root,  # Will auto-save to data_root/feature_config.txt
                n_samples_per_class=args.n_samples_per_class,
                n_angle_features=args.n_angle_features,
                n_ratio_features=args.n_ratio_features,
                random_state=args.random_state
            )
            
            logger.info(f"Feature configuration saved to: {feature_config_path}")
    
    # Step 2: Extract position-invariant features
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Extracting position-invariant features")
    logger.info("="*60)
    feature_extractor = LandmarkFeatureExtractor(
        use_distances=True,
        use_angles=True,
        use_ratios=True,
        feature_config=feature_config  # Pass feature config if available
    )
    X_features = extract_features_from_landmarks(X_landmarks, feature_extractor)
    
    # Step 3: Create DataFrame
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Preparing data for AutoGluon")
    logger.info("="*60)
    
    df = prepare_dataframe(X_features, y_labels, feature_extractor)
    
    # Add file paths to DataFrame for tracking
    df['file_path'] = file_paths
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    
    # Step 4: Split data into train/val/test (stratified by class)
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Stratified splitting by class (train/val/test)")
    logger.info("="*60)
    
    # Use custom stratified split function to ensure uniform split for each class
    train_df, val_df, test_df = stratified_split_by_class(
        df, 
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    logger.info(f"\nOverall split:")
    logger.info(f"  Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify distribution
    logger.info(f"\nClass distribution verification:")
    for emotion in sorted(df['emotion'].unique()):
        train_count = (train_df['emotion'] == emotion).sum()
        val_count = (val_df['emotion'] == emotion).sum()
        test_count = (test_df['emotion'] == emotion).sum()
        total_count = (df['emotion'] == emotion).sum()
        logger.info(f"  {emotion}: {train_count}/{val_count}/{test_count} "
                   f"(ratios: {train_count/total_count:.1%}/{val_count/total_count:.1%}/{test_count/total_count:.1%})")
    
    # Step 5: Save dataset split information
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Saving dataset split information")
    logger.info("="*60)
    
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save train/val/test split with file paths
    split_info = []
    for idx in train_df.index:
        split_info.append({
            'file_path': df.loc[idx, 'file_path'],
            'emotion': df.loc[idx, 'emotion'],
            'split': 'train'
        })
    for idx in val_df.index:
        split_info.append({
            'file_path': df.loc[idx, 'file_path'],
            'emotion': df.loc[idx, 'emotion'],
            'split': 'val'
        })
    for idx in test_df.index:
        split_info.append({
            'file_path': df.loc[idx, 'file_path'],
            'emotion': df.loc[idx, 'emotion'],
            'split': 'test'
        })
    
    split_df = pd.DataFrame(split_info)
    split_file = save_path / 'dataset_split.csv'
    split_df.to_csv(split_file, index=False)
    logger.info(f"Dataset split information saved to {split_file}")
    logger.info(f"  Total: {len(split_df)} samples")
    logger.info(f"  Train: {(split_df['split']=='train').sum()} samples")
    logger.info(f"  Val: {(split_df['split']=='val').sum()} samples")
    logger.info(f"  Test: {(split_df['split']=='test').sum()} samples")
    
    # Remove file_path column from train/val/test DataFrames for training
    train_df = train_df.drop(columns=['file_path'])
    val_df = val_df.drop(columns=['file_path'])
    test_df = test_df.drop(columns=['file_path'])
    
    # Step 6: Train with AutoGluon
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Training with AutoGluon")
    logger.info("="*60)
    
    logger.info(f"Preset: {args.presets}")
    logger.info(f"Time limit: {args.time_limit} seconds")
    logger.info(f"Excluded models: {args.exclude_models}")
    logger.info(f"Model will be saved to: {save_path}")
    logger.info("Note: Focus on fast inference models (RF, LightGBM, XGBoost, CatBoost)")
    logger.info(f"Using validation set for model selection and early stopping")
    
    start_time = time.time()
    
    # Enhanced regularization settings
    # Add hyperparameter configurations for better generalization
    hyperparameters = {
        'GBM': [
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            {},
            'GBMLarge',
        ],
        'CAT': {},
        'XGB': {},
        'RF': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
        ],
        'XT': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
        ],
    }
    
    # AG args for fit - enable early stopping and increase regularization
    ag_args_fit = {
        'num_cpus': args.num_cpus,
        'num_gpus': args.num_gpus,
    }
    
    # Hyperparameter tune kwargs for more aggressive regularization
    hyperparameter_tune_kwargs = {
        'num_trials': 5,  # Number of HPO trials per model
        'scheduler': 'local',
        'searcher': 'auto',
    }
    
    predictor = TabularPredictor(
        label='emotion',
        path=str(save_path),
        problem_type='multiclass',
        eval_metric='accuracy',
        verbosity=2
    ).fit(
        train_data=train_df,
        tuning_data=val_df,  # Use validation set for hyperparameter tuning
        time_limit=args.time_limit,
        presets=args.presets,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        ag_args_fit=ag_args_fit,
        excluded_model_types=args.exclude_models,  # Exclude slow models
        num_bag_folds=8,  # Increased from 5 to 8 for stronger regularization
        num_stack_levels=1,
        use_bag_holdout=True,  # Required to use tuning_data with bagging
    )
    
    train_time = time.time() - start_time
    logger.info(f"\nTraining completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    
    # Step 7: Evaluate on validation and test sets
    logger.info("\n" + "="*60)
    logger.info("STEP 7: Detailed Model Evaluation")
    logger.info("="*60)
    
    # Get detailed model information
    model_info = predictor.info()
    trained_models = [m for m in model_info['model_info'].keys() if not m.endswith('_BAG')]
    
    logger.info(f"\nTotal models trained: {len(trained_models)}")
    logger.info(f"Models: {trained_models}")
    
    # Evaluate each model on train, val, and test sets
    logger.info("\n" + "-"*80)
    logger.info("Individual Model Performance:")
    logger.info("-"*80)
    logger.info(f"{'Model':<25} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Train Time':<15} {'Infer Time':<15}")
    logger.info("-"*80)
    
    model_performances = []
    
    for model_name in trained_models:
        try:
            # Get model training time
            model_fit_time = model_info['model_info'][model_name].get('fit_time', 0)
            
            # Train accuracy
            train_pred = predictor.predict(train_df, model=model_name)
            train_acc = accuracy_score(train_df['emotion'], train_pred)
            
            # Validation accuracy
            val_pred = predictor.predict(val_df, model=model_name)
            val_acc = accuracy_score(val_df['emotion'], val_pred)
            
            # Test accuracy and inference time
            import time as time_module
            start_infer = time_module.time()
            test_pred = predictor.predict(test_df, model=model_name)
            infer_time = time_module.time() - start_infer
            test_acc = accuracy_score(test_df['emotion'], test_pred)
            
            # Calculate average inference time per sample
            avg_infer_time = (infer_time / len(test_df)) * 1000  # ms per sample
            
            logger.info(f"{model_name:<25} {train_acc:<12.4f} {val_acc:<12.4f} {test_acc:<12.4f} "
                       f"{model_fit_time:<15.2f} {avg_infer_time:<15.2f}")
            
            # Per-class accuracy for each dataset
            per_class_acc = {}
            for emotion in sorted(emotion_labels):
                # Train set per-class accuracy
                train_mask = train_df['emotion'] == emotion
                train_class_acc = accuracy_score(
                    train_df[train_mask]['emotion'], 
                    train_pred[train_mask]
                ) if train_mask.sum() > 0 else 0.0
                
                # Val set per-class accuracy
                val_mask = val_df['emotion'] == emotion
                val_class_acc = accuracy_score(
                    val_df[val_mask]['emotion'], 
                    val_pred[val_mask]
                ) if val_mask.sum() > 0 else 0.0
                
                # Test set per-class accuracy
                test_mask = test_df['emotion'] == emotion
                test_class_acc = accuracy_score(
                    test_df[test_mask]['emotion'], 
                    test_pred[test_mask]
                ) if test_mask.sum() > 0 else 0.0
                
                per_class_acc[emotion] = {
                    'train': train_class_acc,
                    'val': val_class_acc,
                    'test': test_class_acc
                }
            
            model_performances.append({
                'model': model_name,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'fit_time': model_fit_time,
                'infer_time_ms': avg_infer_time,
                'per_class_acc': per_class_acc
            })
            
        except Exception as e:
            logger.warning(f"Could not evaluate model {model_name}: {e}")
    
    logger.info("-"*80)
    
    # Per-class accuracy details for each model
    logger.info("\n" + "="*80)
    logger.info("Per-Class Accuracy for Each Model")
    logger.info("="*80)
    
    for perf in model_performances:
        logger.info(f"\nModel: {perf['model']}")
        logger.info(f"{'-'*80}")
        logger.info(f"{'Emotion':<12} {'Train Acc':<15} {'Val Acc':<15} {'Test Acc':<15}")
        logger.info(f"{'-'*80}")
        
        for emotion in sorted(emotion_labels):
            class_acc = perf['per_class_acc'][emotion]
            logger.info(f"{emotion:<12} {class_acc['train']:<15.4f} {class_acc['val']:<15.4f} {class_acc['test']:<15.4f}")
        
        logger.info(f"{'-'*80}")
        logger.info(f"{'Overall':<12} {perf['train_acc']:<15.4f} {perf['val_acc']:<15.4f} {perf['test_acc']:<15.4f}")
    
    # Overall performance
    logger.info("\n" + "="*60)
    logger.info("Overall Performance (Best Model Ensemble)")
    logger.info("="*60)
    
    # Evaluate on validation set
    logger.info("\nEvaluating on validation set...")
    val_performance = predictor.evaluate(val_df, silent=False)
    logger.info(f"Validation Performance: {val_performance}")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_performance = predictor.evaluate(test_df, silent=False)
    logger.info(f"Test Performance: {test_performance}")
    
    # Get leaderboard on test set
    leaderboard = predictor.leaderboard(test_df, silent=True)
    logger.info(f"\nModel Leaderboard (on test set):\n{leaderboard.to_string()}")
    
    # Get predictions
    y_pred = predictor.predict(test_df)
    y_true = test_df['emotion']
    
    # Classification report
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    test_acc = accuracy_score(y_true, y_pred)
    logger.info(f"\nTest Accuracy: {test_acc:.4f}")
    
    logger.info("\nClassification Report:")
    report = classification_report(y_true, y_pred)
    logger.info(f"\n{report}")
    
    # Save detailed classification report
    report_path = save_path / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"AutoGluon Emotion Classification\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  Preset: {args.presets}\n")
        f.write(f"  Time Limit: {args.time_limit}s\n")
        f.write(f"  Total Training Time: {train_time:.2f}s ({train_time/60:.2f}min)\n\n")
        
        f.write(f"Dataset Split:\n")
        f.write(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)\n")
        f.write(f"  Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)\n")
        f.write(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)\n\n")
        
        f.write(f"Individual Model Performance:\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'Model':<25} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Fit Time(s)':<15} {'Infer(ms)':<15}\n")
        f.write(f"{'-'*80}\n")
        for perf in model_performances:
            f.write(f"{perf['model']:<25} {perf['train_acc']:<12.4f} {perf['val_acc']:<12.4f} "
                   f"{perf['test_acc']:<12.4f} {perf['fit_time']:<15.2f} {perf['infer_time_ms']:<15.2f}\n")
        f.write(f"{'-'*80}\n\n")
        
        f.write(f"Per-Class Accuracy Details:\n")
        f.write(f"="*80 + "\n")
        for perf in model_performances:
            f.write(f"\nModel: {perf['model']}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{'Emotion':<12} {'Train Acc':<15} {'Val Acc':<15} {'Test Acc':<15}\n")
            f.write(f"{'-'*80}\n")
            for emotion in sorted(emotion_labels):
                class_acc = perf['per_class_acc'][emotion]
                f.write(f"{emotion:<12} {class_acc['train']:<15.4f} {class_acc['val']:<15.4f} {class_acc['test']:<15.4f}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{'Overall':<12} {perf['train_acc']:<15.4f} {perf['val_acc']:<15.4f} {perf['test_acc']:<15.4f}\n\n")
        
        f.write(f"Overall Performance (Best Model/Ensemble):\n")
        f.write(f"  Validation Accuracy: {val_performance.get('accuracy', 'N/A')}\n")
        f.write(f"  Test Accuracy: {test_acc:.4f}\n\n")
        f.write(f"Best Model: {predictor.get_model_best()}\n\n")
        f.write(f"Classification Report (Test Set):\n{report}\n")
    logger.info(f"Detailed report saved to {report_path}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=emotion_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = save_path / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Save feature extractor for prediction
    joblib.dump(feature_extractor, save_path / 'feature_extractor.pkl')
    logger.info(f"Feature extractor saved")
    
    # Feature importance
    try:
        logger.info("\nFeature Importance (Top 20) in the best model:")
        importance = predictor.feature_importance(test_df, silent=True, model=predictor.get_model_best(), sample_size=2000)
        logger.info(f"\n{importance.head(20).to_string()}")
        
        importance_path = save_path / 'feature_importance.csv'
        importance.to_csv(importance_path)
        logger.info(f"Feature importance saved to {importance_path}")
    except Exception as e:
        logger.warning(f"Could not compute feature importance: {e}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Dataset: {len(df)} total samples")
    logger.info(f"  - Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  - Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  - Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    logger.info(f"\nBest Model: {predictor.get_model_best()}")
    logger.info(f"Validation Accuracy: {val_performance.get('accuracy', 'N/A')}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"\nTotal Training Time: {train_time:.2f}s ({train_time/60:.2f}min)")
    logger.info(f"Model saved to: {save_path}")
    logger.info(f"\nTo predict with this model, use:")
    logger.info(f"  python src/evaluate.py --model_dir {save_path} --input <image.jpg>")


if __name__ == '__main__':
    from src.utils import setup_logger
    from datetime import datetime
    import sys
    from loguru import logger as log
    import argparse
    
    # Parse args early to get save_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='models/emotion_classifier')
    temp_args, _ = parser.parse_known_args()
    
    # Create model directory
    model_dir = Path(temp_args.save_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger with file output in model directory
    log.remove()  # Remove default handler
    
    # Console output
    log.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # File output - save to model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = model_dir / f"training_{timestamp}.log"
    
    log.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        encoding="utf-8"
    )
    
    log.info(f"Log file: {log_file}")
    log.info(f"Model directory: {model_dir}")
    log.info("="*60)
    
    main()
