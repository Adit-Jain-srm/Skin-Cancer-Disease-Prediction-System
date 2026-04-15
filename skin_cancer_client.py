"""
Skin Cancer Detection API - Python Client Library

A simple client library for interacting with the Skin Cancer Detection API.

Usage:
    from skin_cancer_client import SkinCancerAPIClient
    
    client = SkinCancerAPIClient(api_url="http://localhost:5000")
    result = client.predict_image("skin_lesion.jpg")
    print(f"Disease: {result['top_class']}, Confidence: {result['confidence']:.2%}")
"""

import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinCancerAPIClient:
    """Client for Skin Cancer Detection API."""
    
    # Class labels
    CLASS_NAMES = [
        'Melanoma',                    # 0
        'Melanocytic nevus',          # 1
        'Basal cell carcinoma',       # 2
        'Actinic keratosis',          # 3
        'Benign keratosis',           # 4
        'Dermatofibroma',             # 5
        'Vascular lesion'             # 6
    ]
    
    def __init__(
        self,
        api_url: str = "http://localhost:5000",
        timeout: int = 30
    ):
        """
        Initialize the API client.
        
        Args:
            api_url: Base URL of the API (default: localhost:5000)
            timeout: Request timeout in seconds (default: 30)
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Check API connectivity
        try:
            response = self.health_check()
            logger.info(f"✓ Connected to API at {self.api_url}")
        except Exception as e:
            logger.warning(f"⚠ Could not connect to API: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status dictionary
            
        Example:
            >>> client.health_check()
            {'status': 'healthy', 'timestamp': '2026-04-12T16:44:27'}
        """
        response = self.session.get(
            f"{self.api_url}/api/health",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.
        
        Returns:
            Model info dictionary with architecture, accuracy, classes
            
        Example:
            >>> info = client.get_model_info()
            >>> print(f"Model: {info['model_type']}, Accuracy: {info['test_accuracy']:.2%}")
        """
        response = self.session.get(
            f"{self.api_url}/api/info",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def predict_image(
        self,
        image_path: str,
        confidence_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Predict disease from a single image.
        
        Args:
            image_path: Path to image file (JPG, PNG)
            confidence_threshold: Minimum confidence to return (0.0-1.0)
        
        Returns:
            Dictionary with predictions:
            {
                'top_class': 'class_name',
                'confidence': 0.95,
                'predicted_id': 0,
                'all_predictions': {...},
                'inference_time_ms': 52.3
            }
            
        Example:
            >>> result = client.predict_image("lesion.jpg")
            >>> print(f"Predicted: {result['top_class']} ({result['confidence']:.2%})")
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = self.session.post(
                f"{self.api_url}/api/predict",
                files=files,
                timeout=self.timeout
            )
        
        response.raise_for_status()
        result = response.json()
        
        # Add human-readable class name
        if 'predicted_id' in result:
            result['top_class'] = self.CLASS_NAMES[result['predicted_id']]
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[str],
        return_all_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Predict diseases from multiple images.
        
        Args:
            image_paths: List of paths to image files
            return_all_predictions: Include all class probabilities (slower)
        
        Returns:
            Dictionary with batch results:
            {
                'predictions': [
                    {'image': 'file.jpg', 'predicted_id': 0, 'confidence': 0.95},
                    ...
                ],
                'summary': {...},
                'inference_time_ms': 150.2
            }
            
        Example:
            >>> results = client.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
            >>> print(f"Processed {len(results['predictions'])} images")
        """
        files = []
        for path in image_paths:
            image_path = Path(path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            files.append(('images', open(image_path, 'rb')))
        
        try:
            params = {'return_all_predictions': str(return_all_predictions).lower()}
            response = self.session.post(
                f"{self.api_url}/api/predict-batch",
                files=files,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Add human-readable class names
            for pred in result.get('predictions', []):
                if 'predicted_id' in pred:
                    pred['top_class'] = self.CLASS_NAMES[pred['predicted_id']]
            
            return result
        finally:
            # Close all file handles
            for _, file_obj in files:
                file_obj.close()
    
    def predict_from_bytes(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg"
    ) -> Dict[str, Any]:
        """
        Predict disease from image bytes.
        
        Args:
            image_bytes: Image data in bytes
            filename: Filename hint (used for validation)
        
        Returns:
            Prediction result (see predict_image for structure)
            
        Example:
            >>> with open('lesion.jpg', 'rb') as f:
            ...     result = client.predict_from_bytes(f.read(), 'lesion.jpg')
        """
        files = {'image': (filename, image_bytes)}
        response = self.session.post(
            f"{self.api_url}/api/predict-from-bytes",
            files=files,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        if 'predicted_id' in result:
            result['top_class'] = self.CLASS_NAMES[result['predicted_id']]
        
        return result
    
    def close(self):
        """Close the session and cleanup resources."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Example 1: Health check
    print("=" * 70)
    print("EXAMPLE 1: Health Check")
    print("=" * 70)
    with SkinCancerAPIClient() as client:
        status = client.health_check()
        print(f"Status: {json.dumps(status, indent=2)}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Model Information")
    print("=" * 70)
    with SkinCancerAPIClient() as client:
        info = client.get_model_info()
        print(f"Model: {info.get('model_type')}")
        print(f"Accuracy: {info.get('test_accuracy', 0):.2%}")
        print(f"Classes: {', '.join(info.get('class_names', []))}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Single Image Prediction")
    print("=" * 70)
    print("Usage: client.predict_image('path/to/image.jpg')")
    print("Returns: {'top_class': 'Melanoma', 'confidence': 0.95, ...}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Batch Prediction")
    print("=" * 70)
    print("Usage: client.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])")
    print("Returns: {'predictions': [...], 'summary': {...}}")
