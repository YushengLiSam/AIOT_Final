"""
Data loading module for emotion classification.
Loads facial landmark data from pkl files.
Note: Feature extraction is in features_extraction.py
"""

import pickle
import numpy as np
from typing import Optional, List
from pathlib import Path
from tqdm import tqdm
from loguru import logger


def load_landmark_data(data_root: str, emotion_labels: Optional[List[str]] = None):
    """
    Load facial landmark data from pkl files
    Args:
        data_root: Path to landmark data directory
        emotion_labels: List of emotion labels to load (None = load all)
    Returns:
        X: List of landmark arrays
        y: List of emotion labels
        file_paths: List of file paths
    """
    data_root = Path(data_root)
    
    if emotion_labels is None:
        emotion_labels = [d.name for d in data_root.iterdir() if d.is_dir()]
    
    X = []
    y = []
    file_paths = []
    
    logger.info(f"Loading landmark data from {data_root}")
    
    for emotion in emotion_labels:
        emotion_dir = data_root / emotion
        if not emotion_dir.exists():
            logger.warning(f"{emotion_dir} does not exist, skipping")
            continue
        
        pkl_files = list(emotion_dir.glob("*.pkl"))
        logger.info(f"Loading {len(pkl_files)} samples from {emotion}")
        
        for pkl_file in tqdm(pkl_files, desc=f"Loading {emotion}"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                landmarks = data['head_keypoints']
                
                if landmarks is None or len(landmarks) == 0:
                    continue
                
                X.append(landmarks)
                y.append(emotion)
                file_paths.append(str(pkl_file))
                
            except Exception as e:
                logger.warning(f"Error loading {pkl_file}: {e}")
                continue
    
    logger.info(f"Total samples loaded: {len(X)}")
    if len(y) > 0:
        logger.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y, file_paths


def load_feature_data(data_root: str, emotion_labels: Optional[List[str]] = None):
    """
    Load pre-extracted feature data from pkl files
    This is much faster than loading landmarks and extracting features
    
    Args:
        data_root: Path to feature data directory (e.g., data/landmarks/affectnet_features)
        emotion_labels: List of emotion labels to load (None = load all)
    
    Returns:
        X: numpy array of features (n_samples, n_features)
        y: List of emotion labels
        file_paths: List of original file paths
    """
    data_root = Path(data_root)
    
    if emotion_labels is None:
        emotion_labels = [d.name for d in data_root.iterdir() if d.is_dir()]
    
    X = []
    y = []
    file_paths = []
    
    logger.info(f"Loading pre-extracted features from {data_root}")
    
    for emotion in emotion_labels:
        emotion_dir = data_root / emotion
        if not emotion_dir.exists():
            logger.warning(f"{emotion_dir} does not exist, skipping")
            continue
        
        pkl_files = list(emotion_dir.glob("*.pkl"))
        logger.info(f"Loading {len(pkl_files)} samples from {emotion}")
        
        for pkl_file in tqdm(pkl_files, desc=f"Loading {emotion}"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                features = data['features']
                
                if features is None or len(features) == 0:
                    continue
                
                X.append(features)
                y.append(emotion)
                file_paths.append(data.get('original_landmark_path', str(pkl_file)))
                
            except Exception as e:
                logger.warning(f"Error loading {pkl_file}: {e}")
                continue
    
    # Convert to numpy array
    X = np.array(X)
    
    logger.info(f"Total samples loaded: {len(X)}")
    logger.info(f"Feature shape: {X.shape}")
    if len(y) > 0:
        logger.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y, file_paths

