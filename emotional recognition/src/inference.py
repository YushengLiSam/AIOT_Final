"""
Emotion Classification Inference
Multi-model ensemble inference with landmark extraction and visualization
"""

import os
import argparse
import time
import cv2
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Dict, Tuple, Union
from loguru import logger
from autogluon.tabular import TabularPredictor
from rtmlib.tools.solution import Custom

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features_extraction import LandmarkFeatureExtractor, extract_features_from_landmarks


class MultiModelEmotionPredictor:
    """Multi-model emotion predictor with landmark extraction"""
    
    def __init__(self, model_dirs: List[str], device: str = 'cpu', model_name: str = None, confidence_threshold: float = 0.3):
        """
        Initialize predictor with multiple models
        
        Args:
            model_dirs: List of model directory paths
            device: Device for inference ('cpu' or 'cuda')
            model_name: Specific model name to use (e.g., 'WeightedEnsemble_L3', 'LightGBM_BAG_L1')
                       If None, use the best model from AutoGluon
            confidence_threshold: Minimum confidence threshold for valid prediction.
                                 If max confidence < threshold, return 'none' (no valid face detected)
        """
        self.model_dirs = [Path(d) for d in model_dirs]
        self.device = device
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.predictors = []
        self.feature_extractors = []
        self.model_names = []
        
        # Load all models
        logger.info(f"Loading {len(model_dirs)} model(s)...")
        for model_dir in self.model_dirs:
            if not model_dir.exists():
                logger.warning(f"Model directory not found: {model_dir}")
                continue
            
            try:
                # Load AutoGluon predictor
                predictor = TabularPredictor.load(str(model_dir))
                
                # Load feature extractor
                feature_extractor_path = model_dir / 'feature_extractor.pkl'
                feature_extractor = joblib.load(feature_extractor_path)
                
                # Try to load feature config if available
                feature_config_path = model_dir / 'feature_config.txt'
                
                feature_config = None
                if feature_config_path.exists():
                    logger.info(f"    Loading feature config from: {feature_config_path}")
                    from src.feature_selection import load_feature_config
                    feature_config = load_feature_config(str(feature_config_path))
                
                if feature_config is not None:
                    # Update feature extractor with config
                    feature_extractor.feature_config = feature_config
                    feature_extractor.angle_triplets = feature_config.get('angle_triplets', None)
                    feature_extractor.ratio_quadruplets = feature_config.get('ratio_quadruplets', None)
                    if feature_extractor.angle_triplets:
                        logger.info(f"      Using {len(feature_extractor.angle_triplets)} fixed angle triplets")
                    if feature_extractor.ratio_quadruplets:
                        logger.info(f"      Using {len(feature_extractor.ratio_quadruplets)} fixed ratio quadruplets")
                
                self.predictors.append(predictor)
                self.feature_extractors.append(feature_extractor)
                self.model_names.append(model_dir.name)
                
                # Show available models and best model
                best_model = predictor.get_model_best()
                logger.info(f"  ✓ Loaded model from {model_dir}")
                logger.info(f"    Best model: {best_model}")
                if self.model_name:
                    logger.info(f"    Will use specified model: {self.model_name}")
                else:
                    logger.info(f"    Will use best model: {best_model}")
            except Exception as e:
                logger.error(f"  ✗ Failed to load model from {model_dir}: {e}")
        
        if len(self.predictors) == 0:
            raise ValueError("No models loaded successfully!")
        
        logger.info(f"Successfully loaded {len(self.predictors)} model(s)")
        logger.info(f"Confidence threshold: {self.confidence_threshold:.2f} (predictions below this will be classified as 'none')")
        
        # Initialize landmark detector
        logger.info("Initializing landmark detector...")
        
        self.landmark_detector = Custom(
            det_class='YOLOX',
            det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
            det_input_size=(640, 640),
            pose_class='RTMPose',
            # pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.zip',
            pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip',
            pose_input_size=(192, 256),
            backend='onnxruntime',
            device=device
        )
        logger.info("Landmark detector initialized")
    
    def extract_landmarks(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Extract facial landmarks from image
        
        Args:
            image: Input image (BGR)
        
        Returns:
            landmarks: Normalized face landmarks (68, 2) - extracted from COCO wholebody indices 23-90
            landmarks_px: Pixel coordinates (68, 2)
            bbox: Bounding box [x1, y1, x2, y2]
            extract_time: Extraction time in seconds
        """
        start_time = time.time()
        
        # Detect landmarks (133 wholebody keypoints)
        keypoints, scores = self.landmark_detector(image)
        
        extract_time = time.time() - start_time
        
        if keypoints is None or len(keypoints) == 0:
            raise ValueError("No face detected in image")
        
        # Use first detected person
        all_keypoints = keypoints[0]  # (133, 2)
        
        # Extract only face keypoints (indices 23-90, total 68 points)
        # COCO wholebody format: 0-22 body, 23-90 face, 91-132 hands
        face_keypoints_px = all_keypoints[23:91]  # (68, 2)
        landmarks_px = face_keypoints_px
        
        # Get bounding box from face keypoints only
        x_min, y_min = landmarks_px.min(axis=0)
        x_max, y_max = landmarks_px.max(axis=0)
        margin = 20
        bbox = np.array([
            max(0, x_min - margin),
            max(0, y_min - margin),
            min(image.shape[1], x_max + margin),
            min(image.shape[0], y_max + margin)
        ])
        
        # Normalize landmarks
        h, w = image.shape[:2]
        landmarks = landmarks_px.copy()
        landmarks[:, 0] /= w
        landmarks[:, 1] /= h
        
        return landmarks, landmarks_px, bbox, extract_time
    
    def predict(self, landmarks: np.ndarray, model_idx: int = 0) -> Tuple[str, float, Dict[str, float], float, bool]:
        """
        Predict emotion using specified model
        
        Args:
            landmarks: Normalized face landmarks (68, 2) - extracted from COCO wholebody
            model_idx: Index of model to use
        
        Returns:
            emotion: Predicted emotion ('none' if confidence < threshold)
            confidence: Confidence score
            all_probs: All class probabilities
            infer_time: Inference time in seconds
            is_none: True if classified as 'none' due to low confidence
        """
        start_time = time.time()
        
        # Extract features
        feature_extractor = self.feature_extractors[model_idx]
        features = feature_extractor.extract_features(landmarks)
        
        # Create DataFrame with proper feature count based on 68 face keypoints
        # If using fixed feature selection, use actual number of selected features
        if feature_extractor.feature_config is not None:
            n_center = 68 if feature_extractor.use_distances else 0
            n_angles = len(feature_extractor.angle_triplets) if feature_extractor.use_angles and feature_extractor.angle_triplets else 0
            n_ratios = len(feature_extractor.ratio_quadruplets) if feature_extractor.use_ratios and feature_extractor.ratio_quadruplets else 0
        else:
            # Legacy: random sampling
            n_center = 68 if feature_extractor.use_distances else 0
            n_angles = 200 if feature_extractor.use_angles else 0
            n_ratios = 150 if feature_extractor.use_ratios else 0
        
        columns = []
        if feature_extractor.use_distances:
            columns += [f'center_dist_{i}' for i in range(n_center)]
        if feature_extractor.use_angles:
            columns += [f'angle_{i}' for i in range(n_angles)]
        if feature_extractor.use_ratios:
            columns += [f'ratio_{i}' for i in range(n_ratios)]
        
        df = pd.DataFrame([features], columns=columns)
        
        # Predict
        predictor = self.predictors[model_idx]
        
        # Use specific model if specified, otherwise use best model
        if self.model_name:
            emotion = predictor.predict(df, model=self.model_name)[0]
            probs = predictor.predict_proba(df, model=self.model_name)
        else:
            emotion = predictor.predict(df)[0]
            probs = predictor.predict_proba(df)
        
        all_probs = probs.iloc[0].to_dict()
        confidence = all_probs[emotion]
        
        # Check if confidence is below threshold
        is_none = False
        if confidence < self.confidence_threshold:
            # Find max confidence among all classes
            max_confidence = max(all_probs.values())
            if max_confidence < self.confidence_threshold:
                emotion = 'none'
                confidence = max_confidence
                is_none = True
        
        infer_time = time.time() - start_time
        
        return emotion, confidence, all_probs, infer_time, is_none
    
    def predict_multi_model(self, landmarks: np.ndarray) -> List[Dict]:
        """
        Predict using all models
        
        Args:
            landmarks: Normalized face landmarks (68, 2) from COCO wholebody
        
        Returns:
            results: List of prediction results for each model
        """
        results = []
        for i in range(len(self.predictors)):
            emotion, confidence, all_probs, infer_time, is_none = self.predict(landmarks, model_idx=i)
            results.append({
                'model': self.model_names[i],
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': all_probs,
                'infer_time': infer_time,
                'is_none': is_none
            })
        return results
    
    def visualize_result(self, image: np.ndarray, landmarks_px: np.ndarray, 
                        bbox: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Visualize prediction results on image
        
        Args:
            image: Input image (BGR)
            landmarks_px: Landmarks in pixel coordinates
            bbox: Bounding box
            results: Prediction results from all models
        
        Returns:
            annotated_image: Image with annotations
        """
        img = image.copy()
        
        # Check if any result is classified as 'none'
        any_none = any(result.get('is_none', False) for result in results)
        
        if not any_none:
            # Only draw bounding box and landmarks if NOT classified as 'none'
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw landmarks
            for x, y in landmarks_px:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
            
            # Draw predictions
            y_offset = max(10, y1 - 10)
            for i, result in enumerate(results):
                emotion = result['emotion']
                confidence = result['confidence']
                model_name = result['model']
                
                text = f"{model_name}: {emotion.upper()} ({confidence:.2f})"
                
                # Background rectangle for text
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y_offset - text_h - 5), (x1 + text_w + 10, y_offset + 5), 
                             (0, 255, 0), -1)
                
                # Text
                cv2.putText(img, text, (x1 + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 0, 0), 2, cv2.LINE_AA)
                
                y_offset += text_h + 15
        else:
            # If classified as 'none', show warning in center of image
            h, w = img.shape[:2]
            text = "LOW CONFIDENCE - NO VALID EMOTION DETECTED"
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            # Background rectangle
            padding = 20
            cv2.rectangle(img, 
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (0, 0, 255), -1)
            
            # Text
            cv2.putText(img, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Show confidence values below
            y_offset = text_y + text_size[1] + padding + 30
            for result in results:
                conf_text = f"{result['model']}: max conf = {result['confidence']:.3f}"
                conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                conf_x = (w - conf_size[0]) // 2
                cv2.putText(img, conf_text, (conf_x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                y_offset += 25
        
        return img


def process_image(predictor: MultiModelEmotionPredictor, image_path: str, 
                 save_result: bool = False, output_dir: str = None) -> Dict:
    """
    Process single image
    
    Args:
        predictor: MultiModelEmotionPredictor instance
        image_path: Path to input image
        save_result: Whether to save visualization
        output_dir: Output directory for visualization
    
    Returns:
        result: Processing result dictionary
    """
    image_path = Path(image_path)
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    try:
        # Extract landmarks
        landmarks, landmarks_px, bbox, extract_time = predictor.extract_landmarks(img)
        
        # Predict with all models
        predict_start = time.time()
        results = predictor.predict_multi_model(landmarks)
        predict_time = time.time() - predict_start
        
        total_time = extract_time + predict_time
        
        # Log results
        logger.info(f"\n{'='*70}")
        logger.info(f"Image: {image_path.name}")
        logger.info(f"{'='*70}")
        logger.info(f"Landmark extraction time: {extract_time*1000:.2f} ms")
        logger.info(f"Classification time: {predict_time*1000:.2f} ms")
        logger.info(f"Total inference time: {total_time*1000:.2f} ms")
        logger.info(f"\nPredictions:")
        for result in results:
            none_indicator = " [LOW CONFIDENCE - CLASSIFIED AS NONE]" if result.get('is_none', False) else ""
            logger.info(f"  {result['model']}: {result['emotion'].upper()} "
                       f"(confidence: {result['confidence']:.4f}, "
                       f"time: {result['infer_time']*1000:.2f}ms){none_indicator}")
        
        # Save visualization
        if save_result:
            if output_dir is None:
                output_dir = Path("inference_results")
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            vis_img = predictor.visualize_result(img, landmarks_px, bbox, results)
            output_path = output_dir / f"{image_path.stem}_result.jpg"
            cv2.imwrite(str(output_path), vis_img)
            logger.info(f"Visualization saved to: {output_path}")
        
        return {
            'image_path': str(image_path),
            'extract_time': extract_time,
            'predict_time': predict_time,
            'total_time': total_time,
            'predictions': results
        }
        
    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Multi-model emotion inference')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Model directory paths (can specify multiple)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--save_viz', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Specific model name to use (e.g., WeightedEnsemble_L3, LightGBM_BAG_L1). '
                            'If not specified, uses the best model. Use --list_models to see available models.')
    parser.add_argument('--confidence_threshold', type=float, default=0.3,
                       help='Minimum confidence threshold for valid prediction. '
                            'If max confidence < threshold, classify as "none" (default: 0.3)')
    parser.add_argument('--list_models', action='store_true',
                       help='List all available models in the model directory and exit')
    
    args = parser.parse_args()
    
    # List models and exit if requested
    if args.list_models:
        logger.info("="*70)
        logger.info("Available Models")
        logger.info("="*70)
        for model_dir in args.models:
            model_path = Path(model_dir)
            if model_path.exists():
                try:
                    predictor_temp = TabularPredictor.load(str(model_path))
                    logger.info(f"\nModel directory: {model_dir}")
                    logger.info(f"Best model: {predictor_temp.get_model_best()}")
                    logger.info(f"\nAll available models:")
                    leaderboard = predictor_temp.leaderboard(silent=True)
                    for idx, row in leaderboard.iterrows():
                        logger.info(f"  - {row['model']} (score: {row['score_val']:.4f})")
                except Exception as e:
                    logger.error(f"Error loading model from {model_dir}: {e}")
            else:
                logger.error(f"Model directory not found: {model_dir}")
        return
    
    # Initialize predictor
    logger.info("="*70)
    logger.info("Emotion Inference")
    logger.info("="*70)
    predictor = MultiModelEmotionPredictor(args.models, device=args.device, 
                                          model_name=args.model_name,
                                          confidence_threshold=args.confidence_threshold)
    
    # Detect input type
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        logger.info(f"\nProcessing single image: {input_path}")
        result = process_image(predictor, str(input_path), args.save_viz, args.output)
        
    elif input_path.is_dir():
        # Directory of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"\nProcessing directory: {input_path}")
        logger.info(f"Found {len(image_files)} images")
        
        all_results = []
        for i, img_path in enumerate(image_files, 1):
            logger.info(f"\n[{i}/{len(image_files)}] Processing {img_path.name}...")
            result = process_image(predictor, str(img_path), args.save_viz, args.output)
            if result:
                all_results.append(result)
        
        # Summary statistics
        if all_results:
            logger.info("\n" + "="*70)
            logger.info("BATCH PROCESSING SUMMARY")
            logger.info("="*70)
            
            total_images = len(all_results)
            
            # Count 'none' classifications
            none_count = 0
            for result in all_results:
                for pred in result['predictions']:
                    if pred.get('is_none', False):
                        none_count += 1
                        break  # Count each image only once
            
            valid_count = total_images - none_count
            
            avg_extract_time = np.mean([r['extract_time'] for r in all_results]) * 1000
            avg_predict_time = np.mean([r['predict_time'] for r in all_results]) * 1000
            avg_total_time = np.mean([r['total_time'] for r in all_results]) * 1000
            
            logger.info(f"Successfully processed: {total_images} images")
            logger.info(f"  - Valid emotion predictions: {valid_count}")
            logger.info(f"  - Low confidence (classified as 'none'): {none_count}")
            logger.info(f"Average landmark extraction time: {avg_extract_time:.2f} ms")
            logger.info(f"Average classification time: {avg_predict_time:.2f} ms")
            logger.info(f"Average total time: {avg_total_time:.2f} ms")
            logger.info(f"Throughput: {1000/avg_total_time:.2f} images/second")
            
            if args.save_viz:
                logger.info(f"\nAll visualizations saved to: {args.output}")
    
    else:
        logger.error(f"Invalid input path: {input_path}")
        return
    
    logger.info("\n" + "="*70)
    logger.info("Inference completed!")
    logger.info("="*70)


if __name__ == '__main__':
    from loguru import logger
    import sys
    
    # Setup logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    main()
