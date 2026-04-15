"""
Phase 6: Production Inference API Server

Flask-based REST API for serving skin cancer predictions.
Handles image upload, inference, and structured JSON responses.

Usage:
    python deploy_api.py --model-path checkpoints/best_model.pt --port 5000
    
API Endpoints:
    POST /api/predict - Single image prediction
    POST /api/predict-batch - Batch predictions
    GET /api/health - Health check
    GET /api/info - Model info
"""

import flask
from flask import Flask, request, jsonify
import logging
import argparse
from pathlib import Path
from typing import Dict
import io
from PIL import Image
import numpy as np
import time
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import InferenceEngine, format_prediction_for_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU detection and logging
def log_gpu_info():
    """Log GPU availability and details."""
    if torch.cuda.is_available():
        logger.info("✅ GPU DETECTED AND AVAILABLE")
        logger.info(f"   Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("⚠️  GPU NOT AVAILABLE - Using CPU (slower inference)")
        logger.info("   Install CUDA and torch[cuda] for 10-50x faster predictions")


class PredictionAPI:
    """Flask-based prediction API server."""
    
    def __init__(self, model_path: str, model_type: str = 'resnet50'):
        """Initialize API server.
        
        Args:
            model_path: Path to trained model
            model_type: Model architecture type
        """
        self.app = Flask(__name__)
        self.model_path = model_path
        self.model_type = model_type
        
        # Initialize inference engine
        logger.info(f"Loading inference engine from {model_path}")
        self.engine = InferenceEngine(
            model_path=model_path,
            model_type=model_type
        )
        logger.info("Inference engine loaded successfully")
        
        # Configure CORS and request handling
        self.app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API endpoints."""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'model': self.model_type,
                'timestamp': time.time()
            }), 200
        
        @self.app.route('/api/info', methods=['GET'])
        def model_info():
            """Get model information."""
            return jsonify({
                'model_type': self.model_type,
                'model_path': str(self.model_path),
                'classes': self.engine.CLASS_NAMES,
                'classes_detailed': {
                    name: self.engine.CLASS_DESCRIPTIONS.get(name, name)
                    for name in self.engine.CLASS_NAMES
                },
                'target_size': self.engine.target_size,
                'device': str(self.engine.device)
            }), 200
        
        @self.app.route('/api/predict', methods=['POST'])
        def predict_single():
            """Single image prediction endpoint.
            
            Expected: multipart/form-data with 'image' file
            Returns: JSON with prediction details
            """
            try:
                # Validate request
                if 'image' not in request.files:
                    return jsonify({
                        'error': 'No image provided',
                        'details': 'Please upload an image with form parameter "image"'
                    }), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({'error': 'Empty filename'}), 400
                
                # Validate file type
                allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
                if file.filename and '.' not in file.filename:
                    return jsonify({'error': 'Invalid filename'}), 400
                if not file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                    return jsonify({
                        'error': 'Invalid file type',
                        'allowed_types': list(allowed_extensions)
                    }), 400
                
                # Load image from bytes
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Save temporarily for inference
                temp_path = Path(f'/tmp/{file.filename}')
                temp_path.parent.mkdir(exist_ok=True)
                image.save(temp_path)
                
                # Run inference
                result = self.engine.predict_single(temp_path)
                
                # Format response
                response = {
                    'success': True,
                    'prediction': format_prediction_for_api(result),
                    'filename': file.filename
                }
                
                # Cleanup
                temp_path.unlink()
                
                return jsonify(response), 200
                
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/predict-batch', methods=['POST'])
        def predict_batch():
            """Batch prediction endpoint.
            
            Expected: JSON with 'image_paths' list
            Returns: JSON array of predictions
            """
            try:
                data = request.get_json()
                
                if not data or 'image_paths' not in data:
                    return jsonify({
                        'error': 'Missing image_paths',
                        'expected_format': {'image_paths': ['path1.jpg', 'path2.jpg']}
                    }), 400
                
                image_paths = data['image_paths']
                if not isinstance(image_paths, list):
                    return jsonify({'error': 'image_paths must be a list'}), 400
                
                if len(image_paths) == 0:
                    return jsonify({'error': 'image_paths is empty'}), 400
                
                # Run batch inference
                results = self.engine.predict_batch(image_paths)
                
                # Format response
                response = {
                    'success': True,
                    'count': len(results),
                    'predictions': [
                        format_prediction_for_api(result) for result in results
                    ]
                }
                
                return jsonify(response), 200
                
            except Exception as e:
                logger.error(f"Error during batch prediction: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/predict-from-bytes', methods=['POST'])
        def predict_from_bytes():
            """Prediction from image bytes (base64 or raw).
            
            Expected: multipart/form-data with 'image' field
            Returns: JSON with prediction
            """
            try:
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                file = request.files['image']
                
                # Load and process image
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Convert PIL to numpy
                import cv2
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Temporary save and predict
                temp_path = Path(f'/tmp/{int(time.time())}.jpg')
                temp_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(temp_path), image_np)
                
                result = self.engine.predict_single(temp_path)
                temp_path.unlink()
                
                return jsonify({
                    'success': True,
                    'prediction': format_prediction_for_api(result)
                }), 200
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.errorhandler(404)
        def not_found(e):
            """Handle 404 errors."""
            return jsonify({
                'error': 'Endpoint not found',
                'available_endpoints': [
                    'GET /api/health',
                    'GET /api/info',
                    'POST /api/predict (multipart)',
                    'POST /api/predict-batch (JSON)',
                    'POST /api/predict-from-bytes (multipart)'
                ]
            }), 404
        
        @self.app.errorhandler(500)
        def server_error(e):
            """Handle 500 errors."""
            return jsonify({
                'error': 'Internal server error',
                'details': str(e)
            }), 500
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Start API server.
        
        Args:
            host: Host address
            port: Port number
            debug: Enable debug mode
        """
        logger.info(f"Starting API server on {host}:{port}")
        logger.info(f"Model: {self.model_type}")
        logger.info(f"Available endpoints:")
        logger.info(f"  - GET /api/health")
        logger.info(f"  - GET /api/info")
        logger.info(f"  - POST /api/predict (file upload)")
        logger.info(f"  - POST /api/predict-batch (JSON)")
        logger.info(f"  - POST /api/predict-from-bytes (file upload)")
        
        self.app.run(host=host, port=port, debug=debug)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Skin cancer prediction API server')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-type', type=str, default='resnet50',
                       choices=['resnet50', 'efficientnet_b3'],
                       help='Model architecture type')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port number')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Log GPU availability
    log_gpu_info()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        exit(1)
    
    # Create and run API
    api = PredictionAPI(
        model_path=str(model_path),
        model_type=args.model_type
    )
    
    api.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
