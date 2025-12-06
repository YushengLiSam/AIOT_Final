"""
Emotion classification module - Feature extraction
Extract position-invariant features from facial landmarks
"""

import numpy as np
from loguru import logger


class LandmarkFeatureExtractor:
    """Extract position-invariant features from facial landmarks"""
    
    def __init__(self, use_distances=True, use_angles=True, use_ratios=True):
        """
        Args:
            use_distances: Use pairwise distances between landmarks
            use_angles: Use angles formed by landmark triplets
            use_ratios: Use distance ratios (scale-invariant)
        """
        self.use_distances = use_distances
        self.use_angles = use_angles
        self.use_ratios = use_ratios
        
    def extract_pairwise_distances(self, landmarks):
        """Compute all pairwise Euclidean distances between landmarks"""
        n_points = len(landmarks)
        distances = []
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(landmarks[i] - landmarks[j])
                distances.append(dist)
        
        return np.array(distances)
    
    def extract_angles(self, landmarks, n_samples=200):
        """Compute angles formed by landmark triplets"""
        n_points = len(landmarks)
        angles = []
        
        np.random.seed(42)  # Fixed seed for reproducibility
        for _ in range(n_samples):
            i, j, k = np.random.choice(n_points, 3, replace=False)
            
            v1 = landmarks[i] - landmarks[j]
            v2 = landmarks[k] - landmarks[j]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        return np.array(angles)
    
    def extract_distance_ratios(self, landmarks, n_samples=100):
        """Compute ratios of distances (scale-invariant)"""
        n_points = len(landmarks)
        ratios = []
        
        np.random.seed(42)
        for _ in range(n_samples):
            i, j, k, m = np.random.choice(n_points, 4, replace=False)
            
            dist1 = np.linalg.norm(landmarks[i] - landmarks[j])
            dist2 = np.linalg.norm(landmarks[k] - landmarks[m])
            
            ratio = dist1 / (dist2 + 1e-8)
            ratios.append(ratio)
        
        return np.array(ratios)
    
    def extract_center_distances(self, landmarks):
        """Compute distances from each landmark to the centroid"""
        centroid = np.mean(landmarks, axis=0)
        distances = [np.linalg.norm(lm - centroid) for lm in landmarks]
        return np.array(distances)
    
    def extract_features(self, landmarks):
        """
        Extract all selected features from landmarks
        Args:
            landmarks: (n_landmarks, 2) normalized coordinates
        Returns: 1D feature vector
        """
        features = []
        
        if self.use_distances:
            center_dists = self.extract_center_distances(landmarks)
            features.append(center_dists)
            
        if self.use_angles:
            angles = self.extract_angles(landmarks, n_samples=200)
            features.append(angles)
            
        if self.use_ratios:
            ratios = self.extract_distance_ratios(landmarks, n_samples=150)
            features.append(ratios)
        
        feature_vector = np.concatenate(features)
        return feature_vector


def extract_features_from_landmarks(X_landmarks: list, feature_extractor: 'LandmarkFeatureExtractor'):
    """
    Extract features from all landmark samples
    Args:
        X_landmarks: List of landmark arrays
        feature_extractor: LandmarkFeatureExtractor instance
    Returns:
        X_features: (n_samples, n_features) array
    """
    from tqdm import tqdm
    
    logger.info("Extracting position-invariant features...")
    
    X_features = []
    for landmarks in tqdm(X_landmarks, desc="Feature extraction"):
        features = feature_extractor.extract_features(landmarks)
        X_features.append(features)
    
    X_features = np.array(X_features)
    logger.info(f"Feature shape: {X_features.shape}")
    
    return X_features
