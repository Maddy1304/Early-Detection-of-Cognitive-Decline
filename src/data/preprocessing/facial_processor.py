"""
Facial expression data preprocessing for emotion analysis.

This module handles facial data processing including:
- Face detection and alignment
- Facial landmark extraction
- Emotion recognition preprocessing
- Micro-expression detection
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from PIL import Image
import torch
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform
import warnings

logger = logging.getLogger(__name__)


class FacialProcessor:
    """Facial expression data processor for emotion analysis and cognitive decline detection."""
    
    def __init__(
        self,
        face_detection_model: str = "haar",  # haar, dnn
        landmark_model_path: Optional[str] = None,
        emotion_model_path: Optional[str] = None,
        image_size: Tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.5,
        max_faces: int = 1
    ):
        """
        Initialize facial processor.
        
        Args:
            face_detection_model: Face detection model to use
            landmark_model_path: Path to facial landmark model
            emotion_model_path: Path to emotion recognition model
            image_size: Target image size for processing
            confidence_threshold: Confidence threshold for face detection
            max_faces: Maximum number of faces to detect
        """
        self.face_detection_model = face_detection_model
        self.landmark_model_path = landmark_model_path
        self.emotion_model_path = emotion_model_path
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.max_faces = max_faces
        
        # Initialize face detection
        self._initialize_face_detection()
        
        # Initialize facial landmarks
        self._initialize_landmarks()
        
        # Initialize emotion recognition
        self._initialize_emotion_recognition()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _initialize_face_detection(self):
        """Initialize face detection model."""
        if self.face_detection_model == "haar":
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        elif self.face_detection_model == "dnn":
            # DNN-based face detection
            self.face_net = cv2.dnn.readNetFromTensorflow(
                'opencv_face_detector_uint8.pb',
                'opencv_face_detector.pbtxt'
            )
        else:
            # Default to Haar cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.face_detection_model = "haar"
    
    def _initialize_landmarks(self):
        """Initialize facial landmark detection."""
        # Simplified landmark detection using basic geometric features
        # In a real implementation, you would use a proper landmark detection model
        self.landmark_predictor = None
        logger.info("Using simplified landmark detection (geometric features)")
    
    def _initialize_emotion_recognition(self):
        """Initialize emotion recognition model."""
        if self.emotion_model_path:
            try:
                self.emotion_model = torch.load(self.emotion_model_path, map_location='cpu')
                self.emotion_model.eval()
                self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            except Exception as e:
                logger.warning(f"Could not load emotion model: {e}")
                self.emotion_model = None
        else:
            self.emotion_model = None
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"Loaded image: {image_path}, shape: {image.shape}")
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image.
        
        Args:
            image: Input image
            
        Returns:
            List of face detection results
        """
        faces = []
        
        if self.face_detection_model == "haar":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            detections = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in detections:
                face = {
                    'bbox': [x, y, w, h],
                    'confidence': 1.0,  # Haar doesn't provide confidence
                    'landmarks': None
                }
                faces.append(face)
        
        elif self.face_detection_model == "dnn":
            # DNN-based face detection
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    face = {
                        'bbox': [x1, y1, x2-x1, y2-y1],
                        'confidence': confidence,
                        'landmarks': None
                    }
                    faces.append(face)
        
        return faces[:self.max_faces]
    
    def extract_landmarks(self, image: np.ndarray, face: Dict) -> np.ndarray:
        """
        Extract facial landmarks using simplified geometric features.
        
        Args:
            image: Input image
            face: Face detection result
            
        Returns:
            Facial landmarks as numpy array
        """
        x, y, w, h = face['bbox']
        face_roi = image[y:y+h, x:x+w]
        
        # Simplified landmark detection using basic geometric features
        # In a real implementation, you would use a proper landmark detection model
        landmarks = []
        
        # Create basic facial landmarks based on face geometry
        # These are approximate positions for demonstration
        center_x, center_y = w // 2, h // 2
        
        # Eye positions (approximate)
        left_eye_x = center_x - w // 4
        right_eye_x = center_x + w // 4
        eye_y = center_y - h // 6
        
        # Nose position
        nose_x = center_x
        nose_y = center_y
        
        # Mouth position
        mouth_x = center_x
        mouth_y = center_y + h // 4
        
        # Create basic landmark points
        landmarks = [
            [left_eye_x, eye_y],      # Left eye
            [right_eye_x, eye_y],     # Right eye
            [nose_x, nose_y],         # Nose
            [mouth_x, mouth_y],       # Mouth
            [x, y],                   # Top-left corner
            [x + w, y],               # Top-right corner
            [x, y + h],               # Bottom-left corner
            [x + w, y + h]            # Bottom-right corner
        ]
        
        return np.array(landmarks)
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align face using facial landmarks.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Aligned face image
        """
        if len(landmarks) == 0:
            return image
        
        # Define reference points for alignment (eyes and nose)
        if len(landmarks) >= 68:  # 68-point model
            # Left eye center
            left_eye = np.mean(landmarks[36:42], axis=0)
            # Right eye center
            right_eye = np.mean(landmarks[42:48], axis=0)
            # Nose tip
            nose = landmarks[30]
        else:
            # Use first few landmarks as reference
            left_eye = landmarks[0] if len(landmarks) > 0 else [0, 0]
            right_eye = landmarks[1] if len(landmarks) > 1 else [1, 0]
            nose = landmarks[2] if len(landmarks) > 2 else [0.5, 0.5]
        
        # Calculate rotation angle
        eye_vector = right_eye - left_eye
        angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
        
        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        return aligned_image
    
    def extract_geometric_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract geometric features from facial landmarks.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Dictionary of geometric features
        """
        features = {}
        
        if len(landmarks) == 0:
            return features
        
        # Eye features
        if len(landmarks) >= 68:
            # Left eye aspect ratio
            left_eye_landmarks = landmarks[36:42]
            left_ear = self._calculate_eye_aspect_ratio(left_eye_landmarks)
            features['left_eye_aspect_ratio'] = left_ear
            
            # Right eye aspect ratio
            right_eye_landmarks = landmarks[42:48]
            right_ear = self._calculate_eye_aspect_ratio(right_eye_landmarks)
            features['right_eye_aspect_ratio'] = right_ear
            
            # Average eye aspect ratio
            features['eye_aspect_ratio'] = (left_ear + right_ear) / 2
            
            # Mouth aspect ratio
            mouth_landmarks = landmarks[48:68]
            mar = self._calculate_mouth_aspect_ratio(mouth_landmarks)
            features['mouth_aspect_ratio'] = mar
            
            # Eyebrow features
            left_eyebrow = landmarks[17:22]
            right_eyebrow = landmarks[22:27]
            features['eyebrow_angle'] = self._calculate_eyebrow_angle(left_eyebrow, right_eyebrow)
            
            # Nose features
            nose_landmarks = landmarks[27:36]
            features['nose_width'] = np.linalg.norm(nose_landmarks[0] - nose_landmarks[4])
            features['nose_height'] = np.linalg.norm(nose_landmarks[1] - nose_landmarks[5])
        
        # Overall face geometry
        if len(landmarks) > 0:
            # Face width and height
            face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
            face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
            features['face_width'] = face_width
            features['face_height'] = face_height
            features['face_aspect_ratio'] = face_width / face_height if face_height > 0 else 0
            
            # Face symmetry
            features['face_symmetry'] = self._calculate_face_symmetry(landmarks)
        
        return features
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """Calculate eye aspect ratio."""
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Vertical distances
        vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Eye aspect ratio
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def _calculate_mouth_aspect_ratio(self, mouth_landmarks: np.ndarray) -> float:
        """Calculate mouth aspect ratio."""
        if len(mouth_landmarks) < 20:
            return 0.0
        
        # Vertical distances
        vertical_1 = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])
        vertical_2 = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])
        
        # Horizontal distance
        horizontal = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])
        
        # Mouth aspect ratio
        mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return mar
    
    def _calculate_eyebrow_angle(self, left_eyebrow: np.ndarray, right_eyebrow: np.ndarray) -> float:
        """Calculate eyebrow angle."""
        if len(left_eyebrow) < 5 or len(right_eyebrow) < 5:
            return 0.0
        
        # Calculate eyebrow slopes
        left_slope = (left_eyebrow[-1][1] - left_eyebrow[0][1]) / (left_eyebrow[-1][0] - left_eyebrow[0][0])
        right_slope = (right_eyebrow[-1][1] - right_eyebrow[0][1]) / (right_eyebrow[-1][0] - right_eyebrow[0][0])
        
        # Calculate angle difference
        angle_diff = np.abs(np.arctan(left_slope) - np.arctan(right_slope)) * 180 / np.pi
        return angle_diff
    
    def _calculate_face_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate face symmetry."""
        if len(landmarks) == 0:
            return 0.0
        
        # Find face center
        face_center_x = np.mean(landmarks[:, 0])
        
        # Calculate symmetry for each landmark
        symmetry_scores = []
        for landmark in landmarks:
            # Find corresponding point on other side
            distance_from_center = landmark[0] - face_center_x
            corresponding_x = face_center_x - distance_from_center
            
            # Find closest landmark to corresponding point
            distances = np.abs(landmarks[:, 0] - corresponding_x)
            closest_idx = np.argmin(distances)
            closest_landmark = landmarks[closest_idx]
            
            # Calculate symmetry score
            symmetry_score = 1.0 / (1.0 + np.linalg.norm(landmark - closest_landmark))
            symmetry_scores.append(symmetry_score)
        
        return np.mean(symmetry_scores)
    
    def extract_texture_features(self, image: np.ndarray, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features from facial regions.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Dictionary of texture features
        """
        features = {}
        
        if len(landmarks) == 0:
            return features
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Extract features from different facial regions
        if len(landmarks) >= 68:
            # Forehead region
            forehead_landmarks = landmarks[17:22]  # Eyebrow landmarks
            forehead_features = self._extract_region_texture(gray, forehead_landmarks)
            features.update({f'forehead_{k}': v for k, v in forehead_features.items()})
            
            # Eye region
            eye_landmarks = np.concatenate([landmarks[36:42], landmarks[42:48]])
            eye_features = self._extract_region_texture(gray, eye_landmarks)
            features.update({f'eye_{k}': v for k, v in eye_features.items()})
            
            # Mouth region
            mouth_landmarks = landmarks[48:68]
            mouth_features = self._extract_region_texture(gray, mouth_landmarks)
            features.update({f'mouth_{k}': v for k, v in mouth_features.items()})
        
        return features
    
    def _extract_region_texture(self, image: np.ndarray, landmarks: np.ndarray) -> Dict[str, float]:
        """Extract texture features from a specific region."""
        features = {}
        
        if len(landmarks) == 0:
            return features
        
        # Create bounding box for region
        x_min, y_min = np.min(landmarks, axis=0).astype(int)
        x_max, y_max = np.max(landmarks, axis=0).astype(int)
        
        # Ensure bounds are within image
        h, w = image.shape
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return features
        
        # Extract region
        region = image[y_min:y_max, x_min:x_max]
        
        if region.size == 0:
            return features
        
        # Calculate texture features
        features['mean'] = np.mean(region)
        features['std'] = np.std(region)
        features['variance'] = np.var(region)
        features['skewness'] = self._calculate_skewness(region)
        features['kurtosis'] = self._calculate_kurtosis(region)
        
        # Local Binary Pattern (simplified)
        features['lbp_uniformity'] = self._calculate_lbp_uniformity(region)
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_lbp_uniformity(self, image: np.ndarray) -> float:
        """Calculate Local Binary Pattern uniformity (simplified)."""
        if image.size == 0:
            return 0.0
        
        # Simplified LBP calculation
        h, w = image.shape
        uniformity = 0
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                pattern = 0
                pattern |= (image[i-1, j-1] > center) << 7
                pattern |= (image[i-1, j] > center) << 6
                pattern |= (image[i-1, j+1] > center) << 5
                pattern |= (image[i, j+1] > center) << 4
                pattern |= (image[i+1, j+1] > center) << 3
                pattern |= (image[i+1, j] > center) << 2
                pattern |= (image[i+1, j-1] > center) << 1
                pattern |= (image[i, j-1] > center) << 0
                
                # Count transitions
                transitions = bin(pattern ^ ((pattern << 1) | (pattern >> 7))).count('1')
                if transitions <= 2:
                    uniformity += 1
        
        return uniformity / ((h-2) * (w-2)) if (h-2) * (w-2) > 0 else 0.0
    
    def recognize_emotion(self, image: np.ndarray, face: Dict) -> Dict[str, float]:
        """
        Recognize emotions in face.
        
        Args:
            image: Input image
            face: Face detection result
            
        Returns:
            Dictionary of emotion probabilities
        """
        if self.emotion_model is None:
            return {}
        
        x, y, w, h = face['bbox']
        face_roi = image[y:y+h, x:x+w]
        
        # Preprocess face for emotion recognition
        face_pil = Image.fromarray(face_roi)
        face_tensor = self.transform(face_pil).unsqueeze(0)
        
        # Predict emotions
        with torch.no_grad():
            outputs = self.emotion_model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().numpy()
        
        # Create emotion dictionary
        emotions = {}
        for i, emotion in enumerate(self.emotion_classes):
            emotions[emotion] = float(probabilities[i])
        
        return emotions
    
    def detect_micro_expressions(self, landmarks_sequence: List[np.ndarray]) -> Dict[str, float]:
        """
        Detect micro-expressions from landmark sequence.
        
        Args:
            landmarks_sequence: Sequence of facial landmarks
            
        Returns:
            Dictionary of micro-expression features
        """
        features = {}
        
        if len(landmarks_sequence) < 2:
            return features
        
        # Calculate landmark movements
        movements = []
        for i in range(1, len(landmarks_sequence)):
            if len(landmarks_sequence[i]) > 0 and len(landmarks_sequence[i-1]) > 0:
                movement = np.linalg.norm(
                    landmarks_sequence[i] - landmarks_sequence[i-1], axis=1
                )
                movements.append(movement)
        
        if not movements:
            return features
        
        movements = np.array(movements)
        
        # Micro-expression features
        features['movement_mean'] = np.mean(movements)
        features['movement_std'] = np.std(movements)
        features['movement_max'] = np.max(movements)
        features['movement_variance'] = np.var(movements)
        
        # Detect sudden movements (potential micro-expressions)
        movement_threshold = np.mean(movements) + 2 * np.std(movements)
        sudden_movements = np.sum(movements > movement_threshold)
        features['sudden_movements'] = sudden_movements
        features['micro_expression_intensity'] = sudden_movements / len(movements)
        
        return features
    
    def process_facial_data(self, image: np.ndarray, extract_features: bool = True) -> Dict:
        """
        Complete facial data processing pipeline.
        
        Args:
            image: Input image
            extract_features: Whether to extract features
            
        Returns:
            Dictionary containing processed data and features
        """
        result = {'raw_image': image}
        
        # Detect faces
        faces = self.detect_faces(image)
        result['faces'] = faces
        
        if not faces:
            logger.warning("No faces detected in image")
            return result
        
        # Process first face
        face = faces[0]
        result['primary_face'] = face
        
        # Extract landmarks
        landmarks = self.extract_landmarks(image, face)
        result['landmarks'] = landmarks
        
        # Align face
        aligned_face = self.align_face(image, landmarks)
        result['aligned_face'] = aligned_face
        
        if extract_features:
            # Extract features
            result['geometric_features'] = self.extract_geometric_features(landmarks)
            result['texture_features'] = self.extract_texture_features(image, landmarks)
            result['emotions'] = self.recognize_emotion(image, face)
        
        return result
    
    def augment_facial_data(self, image: np.ndarray, techniques: List[str] = None) -> np.ndarray:
        """
        Apply data augmentation techniques to facial data.
        
        Args:
            image: Input image
            techniques: List of augmentation techniques to apply
            
        Returns:
            Augmented image
        """
        if techniques is None:
            techniques = ['rotation', 'translation', 'scaling', 'brightness_change', 'contrast_change']
        
        augmented_image = image.copy()
        
        for technique in techniques:
            if technique == 'rotation':
                # Random rotation
                angle = np.random.uniform(-10, 10)
                h, w = augmented_image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                augmented_image = cv2.warpAffine(augmented_image, rotation_matrix, (w, h))
                
            elif technique == 'translation':
                # Random translation
                tx = np.random.uniform(-10, 10)
                ty = np.random.uniform(-10, 10)
                h, w = augmented_image.shape[:2]
                translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                augmented_image = cv2.warpAffine(augmented_image, translation_matrix, (w, h))
                
            elif technique == 'scaling':
                # Random scaling
                scale = np.random.uniform(0.9, 1.1)
                h, w = augmented_image.shape[:2]
                center = (w // 2, h // 2)
                scale_matrix = cv2.getRotationMatrix2D(center, 0, scale)
                augmented_image = cv2.warpAffine(augmented_image, scale_matrix, (w, h))
                
            elif technique == 'brightness_change':
                # Random brightness change
                brightness = np.random.uniform(-30, 30)
                augmented_image = cv2.convertScaleAbs(augmented_image, alpha=1.0, beta=brightness)
                
            elif technique == 'contrast_change':
                # Random contrast change
                contrast = np.random.uniform(0.8, 1.2)
                augmented_image = cv2.convertScaleAbs(augmented_image, alpha=contrast, beta=0)
        
        return augmented_image
