"""
Phase 6: Production Inference Pipeline

Handles model loading, preprocessing, and batch inference for production deployment.
Supports both single image and batch predictions with comprehensive error handling.

Usage:
    from src.inference import InferenceEngine
    
    engine = InferenceEngine(model_path='checkpoints/best_model.pt')
    predictions = engine.predict_single('path/to/image.jpg')
    
    # Batch prediction
    predictions = engine.predict_batch(['img1.jpg', 'img2.jpg', ...])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import cv2
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transfer_learning import TransferLearningModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    image_path: str
    predicted_class: str
    predicted_id: int
    confidence: float  # Softmax probability
    class_probabilities: Dict[str, float]
    inference_time_ms: float


class ImagePreprocessor:
    """Handle image loading and preprocessing for inference."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """Initialize preprocessor."""
        self.target_size = target_size
        logger.info(f"ImagePreprocessor initialized with target size {target_size}")
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load and validate image."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference."""
        # Resize
        img_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to torch tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor


class InferenceEngine:
    """Production-ready inference engine."""
    
    # Skin cancer classes
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    CLASS_DESCRIPTIONS = {
        'akiec': 'Actinic keratosis',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevus',
        'vasc': 'Vascular lesion'
    }
    
    def __init__(
        self,
        model_path: Union[str, Path],
        model_type: str = 'resnet50',
        device: Optional[torch.device] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """Initialize inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: 'resnet50' or 'efficientnet_b3'
            device: torch.device (defaults to CUDA if available, else CPU)
            target_size: Input image size for model
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_size = target_size
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(target_size)
        self.model = self._load_model()
        
        logger.info(f"InferenceEngine initialized on {self.device}")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self) -> nn.Module:
        """Load and validate trained model."""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create model architecture
        if self.model_type == 'resnet50':
            model = TransferLearningModel.create_resnet50(
                num_classes=len(self.CLASS_NAMES),
                pretrained=False,
                freeze_backbone=False
            )
        elif self.model_type == 'efficientnet_b3':
            model = TransferLearningModel.create_efficientnet_b3(
                num_classes=len(self.CLASS_NAMES),
                pretrained=False,
                freeze_backbone=False
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        # Set to eval mode
        model = model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
    
    @torch.no_grad()
    def predict_single(
        self,
        image_path: Union[str, Path],
        return_raw_logits: bool = False
    ) -> PredictionResult:
        """Predict on single image.
        
        Args:
            image_path: Path to image
            return_raw_logits: Whether to include raw logits in output
        
        Returns:
            PredictionResult with prediction details
        """
        import time
        start_time = time.time()
        
        try:
            # Load and preprocess
            image = self.preprocessor.load_image(image_path)
            image_tensor = self.preprocessor.preprocess(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Inference
            logits = self.model(image_tensor)
            
            # Get probabilities
            probs = F.softmax(logits, dim=1)[0]
            predicted_id = int(torch.argmax(probs).item())
            confidence = probs[predicted_id].item()
            
            # Prepare class probabilities
            class_probs = {
                self.CLASS_NAMES[i]: probs[i].item()
                for i in range(len(self.CLASS_NAMES))
            }
            
            inference_time = (time.time() - start_time) * 1000
            
            result = PredictionResult(
                image_path=str(image_path),
                predicted_class=self.CLASS_NAMES[predicted_id],
                predicted_id=predicted_id,
                confidence=confidence,
                class_probabilities=class_probs,
                inference_time_ms=inference_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during inference on {image_path}: {str(e)}")
            raise
    
    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32
    ) -> List[PredictionResult]:
        """Predict on batch of images.
        
        Args:
            image_paths: List of image paths
            batch_size: Number of images per batch
        
        Returns:
            List of PredictionResult
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_paths = []
            
            # Load images
            for path in batch_paths:
                try:
                    image = self.preprocessor.load_image(path)
                    image_tensor = self.preprocessor.preprocess(image)
                    batch_images.append(image_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    logger.error(f"Skipping {path}: {str(e)}")
            
            if not batch_images:
                continue
            
            # Stack batch
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Inference
            import time
            start_time = time.time()
            logits = self.model(batch_tensor)
            inference_time = (time.time() - start_time) * 1000 / len(valid_paths)
            
            # Process predictions
            probs = F.softmax(logits, dim=1)
            
            for j, path in enumerate(valid_paths):
                prob_dist = probs[j]
                predicted_id = torch.argmax(prob_dist).item()
                confidence = prob_dist[int(predicted_id)].item()
                
                class_probs = {
                    self.CLASS_NAMES[k]: prob_dist[k].item()
                    for k in range(len(self.CLASS_NAMES))
                }
                
                result = PredictionResult(
                    image_path=str(path),
                    predicted_class=self.CLASS_NAMES[predicted_id],
                    predicted_id=predicted_id,
                    confidence=confidence,
                    class_probabilities=class_probs,
                    inference_time_ms=inference_time
                )
                results.append(result)
        
        return results
    
    def get_class_description(self, class_id: int) -> str:
        """Get human-readable description for class."""
        class_name = self.CLASS_NAMES[class_id]
        return self.CLASS_DESCRIPTIONS.get(class_name, class_name)


def format_prediction_for_api(result: PredictionResult) -> Dict:
    """Format prediction result for API response."""
    return {
        'prediction': {
            'class': result.predicted_class,
            'class_id': result.predicted_id,
            'confidence': round(result.confidence, 4),
            'confidence_percent': round(result.confidence * 100, 2)
        },
        'probabilities': {
            k: round(v, 4) for k, v in result.class_probabilities.items()
        },
        'metadata': {
            'inference_time_ms': round(result.inference_time_ms, 2),
            'image_path': result.image_path
        }
    }
