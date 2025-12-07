"""
Feature Selection for Emotion Classification
Select discriminative angle and ratio features using ANOVA F-test
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from loguru import logger
from sklearn.feature_selection import f_classif
import itertools
from multiprocessing import Pool, cpu_count
from functools import partial


def _compute_angles_only(landmarks: np.ndarray) -> np.ndarray:
    """Helper function for multiprocessing: compute only angle values"""
    angles, _ = compute_all_angle_features(landmarks)
    return angles


def _compute_ratios_only(landmarks: np.ndarray) -> np.ndarray:
    """Helper function for multiprocessing: compute only ratio values"""
    ratios, _ = compute_all_ratio_features(landmarks)
    return ratios


def compute_all_angle_features(landmarks: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Compute all possible angle features from landmark triplets
    
    Args:
        landmarks: (n_points, 2) array of normalized landmarks
    
    Returns:
        angles: Array of all angle features
        triplets: List of (i, j, k) triplet indices
    """
    n_points = len(landmarks)
    angles = []
    triplets = []
    
    # Generate all possible triplets (combinations of 3 points)
    for i, j, k in itertools.combinations(range(n_points), 3):
        v1 = landmarks[i] - landmarks[j]
        v2 = landmarks[k] - landmarks[j]
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        angles.append(angle)
        triplets.append((i, j, k))
    
    return np.array(angles), triplets


def compute_all_ratio_features(landmarks: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Compute all possible ratio features from landmark quadruplets
    
    Args:
        landmarks: (n_points, 2) array of normalized landmarks
    
    Returns:
        ratios: Array of all ratio features
        quadruplets: List of (i, j, k, m) quadruplet indices
    """
    n_points = len(landmarks)
    ratios = []
    quadruplets = []
    
    # Generate all possible quadruplets (combinations of 4 points)
    for i, j, k, m in itertools.combinations(range(n_points), 4):
        dist1 = np.linalg.norm(landmarks[i] - landmarks[j])
        dist2 = np.linalg.norm(landmarks[k] - landmarks[m])
        
        ratio = dist1 / (dist2 + 1e-8)
        
        ratios.append(ratio)
        quadruplets.append((i, j, k, m))
    
    return np.array(ratios), quadruplets


def select_discriminative_features(
    X_landmarks: List[np.ndarray],
    y_labels: List[str],
    emotion_labels: List[str],
    data_root: str = None,
    n_samples_per_class: int = 500,
    n_angle_features: int = 200,
    n_ratio_features: int = 200,
    random_state: int = 42,
    file_paths: List[str] = None,
    n_jobs: int = 16
) -> Dict:
    """
    Select discriminative angle and ratio features using ANOVA F-test
    
    Args:
        X_landmarks: List of landmark arrays (n_samples, n_points, 2)
        y_labels: List of emotion labels
        emotion_labels: List of all emotion classes
        data_root: Path to landmark data directory (will save config here)
        n_samples_per_class: Number of samples to use per class
        n_angle_features: Number of angle features to select
        n_ratio_features: Number of ratio features to select
        random_state: Random seed
    
    Returns:
        feature_config: Dictionary containing selected feature indices
    """
    np.random.seed(random_state)
    
    logger.info("="*70)
    logger.info("Feature Selection using ANOVA F-test")
    logger.info("="*70)
    
    # Step 1: Sample data from each class
    logger.info(f"\nStep 1: Sampling {n_samples_per_class} samples per class")
    sampled_landmarks = []
    sampled_labels = []
    
    for emotion in emotion_labels:
        # Get all samples for this emotion
        emotion_indices = [i for i, label in enumerate(y_labels) if label == emotion]
        
        # Sample n_samples_per_class (or all if less than n_samples_per_class)
        n_to_sample = min(n_samples_per_class, len(emotion_indices))
        sampled_indices = np.random.choice(emotion_indices, n_to_sample, replace=False)
        
        sampled_landmarks.extend([X_landmarks[i] for i in sampled_indices])
        sampled_labels.extend([emotion] * n_to_sample)
        
        logger.info(f"  {emotion}: sampled {n_to_sample} / {len(emotion_indices)} samples")
    
    logger.info(f"Total sampled: {len(sampled_landmarks)} samples")
    
    # Step 2: Compute all angle features for all samples
    logger.info(f"\nStep 2: Computing all possible angle features...")
    logger.info(f"  Number of possible triplets (C(68,3)): {68*67*66//6:,}")
    
    # Compute angle features for first sample to get feature indices
    first_angles, angle_triplets = compute_all_angle_features(sampled_landmarks[0])
    n_angle_combinations = len(angle_triplets)
    logger.info(f"  Actual angle features to compute: {n_angle_combinations:,}")
    
    # Compute angle features for all samples (parallel)
    all_angle_features = np.zeros((len(sampled_landmarks), n_angle_combinations))
    
    logger.info(f"  Using {n_jobs} CPU cores for parallel computation")
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_compute_angles_only, sampled_landmarks),
            total=len(sampled_landmarks),
            desc="Computing angle features"
        ))
    
    for i, angles in enumerate(results):
        all_angle_features[i] = angles
    
    logger.info(f"  Angle feature matrix shape: {all_angle_features.shape}")
    
    # Step 3: Compute all ratio features for all samples
    logger.info(f"\nStep 3: Computing all possible ratio features...")
    logger.info(f"  Number of possible quadruplets (C(68,4)): {68*67*66*65//24:,}")
    
    # Compute ratio features for first sample to get feature indices
    first_ratios, ratio_quadruplets = compute_all_ratio_features(sampled_landmarks[0])
    n_ratio_combinations = len(ratio_quadruplets)
    logger.info(f"  Actual ratio features to compute: {n_ratio_combinations:,}")
    
    # Compute ratio features for all samples (parallel)
    all_ratio_features = np.zeros((len(sampled_landmarks), n_ratio_combinations))
    
    logger.info(f"  Using {n_jobs} CPU cores for parallel computation")
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_compute_ratios_only, sampled_landmarks),
            total=len(sampled_landmarks),
            desc="Computing ratio features"
        ))
    
    for i, ratios in enumerate(results):
        all_ratio_features[i] = ratios
    
    logger.info(f"  Ratio feature matrix shape: {all_ratio_features.shape}")
    
    # Step 4: ANOVA F-test for angle features
    logger.info(f"\nStep 4: Selecting top {n_angle_features} angle features using ANOVA F-test...")
    
    # Convert labels to numeric
    label_to_idx = {label: idx for idx, label in enumerate(emotion_labels)}
    y_numeric = np.array([label_to_idx[label] for label in sampled_labels])
    
    # Compute F-statistics for all angle features
    f_scores_angles, p_values_angles = f_classif(all_angle_features, y_numeric)
    
    # Handle NaN values (constant features)
    f_scores_angles = np.nan_to_num(f_scores_angles, nan=0.0)
    
    # Select top N features
    top_angle_indices = np.argsort(f_scores_angles)[-n_angle_features:][::-1]
    selected_angle_triplets = [angle_triplets[i] for i in top_angle_indices]
    selected_angle_f_scores = f_scores_angles[top_angle_indices]
    
    logger.info(f"  Selected {len(selected_angle_triplets)} angle features")
    logger.info(f"  F-score range: {selected_angle_f_scores.min():.2f} - {selected_angle_f_scores.max():.2f}")
    logger.info(f"  Top 5 angle features:")
    for i in range(min(5, len(selected_angle_triplets))):
        triplet = selected_angle_triplets[i]
        f_score = selected_angle_f_scores[i]
        logger.info(f"    Triplet {triplet}: F-score = {f_score:.2f}")
    
    # Step 5: ANOVA F-test for ratio features
    logger.info(f"\nStep 5: Selecting top {n_ratio_features} ratio features using ANOVA F-test...")
    
    # Compute F-statistics for all ratio features
    f_scores_ratios, p_values_ratios = f_classif(all_ratio_features, y_numeric)
    
    # Handle NaN values
    f_scores_ratios = np.nan_to_num(f_scores_ratios, nan=0.0)
    
    # Select top N features
    top_ratio_indices = np.argsort(f_scores_ratios)[-n_ratio_features:][::-1]
    selected_ratio_quadruplets = [ratio_quadruplets[i] for i in top_ratio_indices]
    selected_ratio_f_scores = f_scores_ratios[top_ratio_indices]
    
    logger.info(f"  Selected {len(selected_ratio_quadruplets)} ratio features")
    logger.info(f"  F-score range: {selected_ratio_f_scores.min():.2f} - {selected_ratio_f_scores.max():.2f}")
    logger.info(f"  Top 5 ratio features:")
    for i in range(min(5, len(selected_ratio_quadruplets))):
        quadruplet = selected_ratio_quadruplets[i]
        f_score = selected_ratio_f_scores[i]
        logger.info(f"    Quadruplet {quadruplet}: F-score = {f_score:.2f}")
    
    # Step 6: Create feature configuration
    feature_config = {
        'angle_triplets': selected_angle_triplets,
        'ratio_quadruplets': selected_ratio_quadruplets,
        'angle_f_scores': selected_angle_f_scores.tolist(),
        'ratio_f_scores': selected_ratio_f_scores.tolist(),
        'n_samples_per_class': n_samples_per_class,
        'random_state': random_state,
        'emotion_labels': emotion_labels
    }
    
    logger.info(f"\n" + "="*70)
    logger.info("Feature Selection Summary")
    logger.info("="*70)
    logger.info(f"Selected {len(selected_angle_triplets)} angle features")
    logger.info(f"Selected {len(selected_ratio_quadruplets)} ratio features")
    logger.info(f"Total: {len(selected_angle_triplets) + len(selected_ratio_quadruplets)} geometric features")
    logger.info(f"Plus 68 center distance features")
    logger.info(f"Total feature dimension: {68 + len(selected_angle_triplets) + len(selected_ratio_quadruplets)}")
    
    # Auto-save configuration to data_root if provided
    if data_root is not None:
        data_root_path = Path(data_root)
        
        # Save human-readable text format only
        txt_path = data_root_path / 'feature_config.txt'
        save_feature_config_txt(feature_config, str(txt_path))
        logger.info(f"Feature configuration saved to: {txt_path}")
    
    return feature_config


def save_feature_config_txt(feature_config: Dict, save_path: str):
    """Save feature configuration in human-readable text format"""
    save_path = Path(save_path)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Emotion Classification Feature Configuration\n")
        f.write("="*70 + "\n\n")
        
        # Metadata
        f.write("Metadata:\n")
        f.write(f"  Random State: {feature_config.get('random_state', 'N/A')}\n")
        f.write(f"  Samples per Class: {feature_config.get('n_samples_per_class', 'N/A')}\n")
        f.write(f"  Emotion Labels: {', '.join(feature_config.get('emotion_labels', []))}\n")
        f.write("\n")
        
        # Angle features
        angle_triplets = feature_config.get('angle_triplets', [])
        angle_f_scores = feature_config.get('angle_f_scores', [])
        
        f.write("="*70 + "\n")
        f.write(f"Angle Features (Total: {len(angle_triplets)})\n")
        f.write("="*70 + "\n")
        f.write("Format: Index, Point1, Point2, Point3, F-Score\n")
        f.write("-"*70 + "\n")
        
        for idx, (triplet, f_score) in enumerate(zip(angle_triplets, angle_f_scores)):
            i, j, k = triplet
            f.write(f"{idx+1:4d}, {i:3d}, {j:3d}, {k:3d}, {f_score:10.4f}\n")
        
        f.write("\n")
        
        # Ratio features
        ratio_quadruplets = feature_config.get('ratio_quadruplets', [])
        ratio_f_scores = feature_config.get('ratio_f_scores', [])
        
        f.write("="*70 + "\n")
        f.write(f"Ratio Features (Total: {len(ratio_quadruplets)})\n")
        f.write("="*70 + "\n")
        f.write("Format: Index, Point1, Point2, Point3, Point4, F-Score\n")
        f.write("-"*70 + "\n")
        
        for idx, (quadruplet, f_score) in enumerate(zip(ratio_quadruplets, ratio_f_scores)):
            i, j, k, m = quadruplet
            f.write(f"{idx+1:4d}, {i:3d}, {j:3d}, {k:3d}, {m:3d}, {f_score:10.4f}\n")
        
        f.write("\n")
        f.write("="*70 + "\n")
        f.write("End of Configuration\n")
        f.write("="*70 + "\n")
    
    logger.info(f"Human-readable configuration saved to: {save_path}")





def load_feature_config(config_path: str) -> Dict:
    """Load feature configuration from text file"""
    config_path = Path(config_path)
    
    # Ensure .txt extension
    if config_path.suffix != '.txt':
        config_path = config_path.with_suffix('.txt')
    
    if not config_path.exists():
        raise FileNotFoundError(f"Feature config not found at {config_path}")
    
    logger.info(f"Loading feature configuration from: {config_path}")
    return load_feature_config_from_txt(str(config_path))


def load_feature_config_from_txt(txt_path: str) -> Dict:
    """Load feature configuration from human-readable text file"""
    txt_path = Path(txt_path)
    
    angle_triplets = []
    ratio_quadruplets = []
    angle_f_scores = []
    ratio_f_scores = []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse file
    section = None
    for line in lines:
        line = line.strip()
        
        if 'Angle Features' in line:
            section = 'angle'
            continue
        elif 'Ratio Features' in line:
            section = 'ratio'
            continue
        elif line.startswith('=') or line.startswith('-') or not line:
            continue
        elif line.startswith('Format:') or line.startswith('Metadata:'):
            continue
        
        # Parse data lines
        if section == 'angle' and ',' in line:
            try:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    idx = int(parts[0])
                    i, j, k = int(parts[1]), int(parts[2]), int(parts[3])
                    f_score = float(parts[4])
                    angle_triplets.append((i, j, k))
                    angle_f_scores.append(f_score)
            except (ValueError, IndexError):
                continue
        
        elif section == 'ratio' and ',' in line:
            try:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    idx = int(parts[0])
                    i, j, k, m = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    f_score = float(parts[5])
                    ratio_quadruplets.append((i, j, k, m))
                    ratio_f_scores.append(f_score)
            except (ValueError, IndexError):
                continue
    
    feature_config = {
        'angle_triplets': angle_triplets,
        'ratio_quadruplets': ratio_quadruplets,
        'angle_f_scores': angle_f_scores,
        'ratio_f_scores': ratio_f_scores,
        'emotion_labels': ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    }
    
    logger.info(f"Loaded {len(angle_triplets)} angle features and {len(ratio_quadruplets)} ratio features from text file")
    
    return feature_config


if __name__ == '__main__':
    """Test feature selection and feature extraction"""
    import sys
    import argparse
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.data_loader import load_landmark_data
    
    parser = argparse.ArgumentParser(description='Feature selection using ANOVA F-test')
    parser.add_argument('--landmark_root', type=str, default='data/landmarks/affectnet',
                       help='Root directory containing landmark files')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples per class for feature selection')
    parser.add_argument('--n_angle_features', type=int, default=200,
                       help='Number of angle features to select')
    parser.add_argument('--n_ratio_features', type=int, default=200,
                       help='Number of ratio features to select')
    parser.add_argument('--n_jobs', type=int, default=16,
                       help='Number of CPU cores to use (default: 16)')
    
    args = parser.parse_args()
    
    emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 
                     'happy', 'neutral', 'sad', 'surprise']
    
    logger.info("Feature Selection using ANOVA F-test")
    logger.info("="*70)
    
    # Load landmark data
    logger.info("Loading landmark data...")
    X_landmarks, y_labels, file_paths = load_landmark_data(
        args.landmark_root,
        emotion_labels
    )
    
    # Perform feature selection (will auto-save to landmark_root/feature_config.txt)
    feature_config = select_discriminative_features(
        X_landmarks,
        y_labels,
        emotion_labels,
        data_root=args.landmark_root,
        n_samples_per_class=args.n_samples,
        n_angle_features=args.n_angle_features,
        n_ratio_features=args.n_ratio_features,
        random_state=42,
        file_paths=file_paths,
        n_jobs=args.n_jobs
    )
    
    logger.info("\n" + "="*70)
    logger.info("Feature selection completed!")
    logger.info(f"Configuration saved to: {Path(args.landmark_root) / 'feature_config.txt'}")
    logger.info("="*70)
