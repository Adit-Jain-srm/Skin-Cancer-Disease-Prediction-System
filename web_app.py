"""
Integrated Web Application - Skin Cancer Detection System
Combines Flask API with modern React-like frontend

Usage:
    python web_app.py --model-path checkpoints/best_model.pt --port 5000
"""

from flask import Flask, send_file, send_from_directory, request, jsonify
from flask_cors import CORS
import logging
import argparse
from pathlib import Path
import io
from PIL import Image
import numpy as np
import time
import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import InferenceEngine, format_prediction_for_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
LEGACY_FRONTEND = ROOT_DIR / "frontend.html"

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


class SkinCancerWebApp:
    """Integrated web application with API and UI."""
    
    def __init__(self, model_path: str, model_type: str = 'resnet50'):
        """Initialize web application."""
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.model_path = model_path
        self.model_type = model_type
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.engine = InferenceEngine(
            model_path=model_path,
            model_type=model_type
        )
        logger.info("Model loaded successfully")
        
        # Configure app
        self.app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register all routes."""
        
        # Serve frontend (Vite build if present, else legacy single-file HTML)
        @self.app.route('/')
        def index():
            """Serve main frontend."""
            built = FRONTEND_DIST / "index.html"
            if built.is_file():
                return send_file(built, mimetype='text/html; charset=utf-8')
            if LEGACY_FRONTEND.is_file():
                return send_file(LEGACY_FRONTEND, mimetype='text/html; charset=utf-8')
            return jsonify({'error': 'No frontend build or frontend.html'}), 503

        @self.app.route('/favicon.ico')
        def favicon():
            """Avoid 404 noise; serve icon from build output if present."""
            p = FRONTEND_DIST / 'favicon.ico'
            if p.is_file():
                return send_file(p)
            return ('', 204)

        @self.app.route('/sw.js')
        def service_worker_absent():
            """Browsers / extensions probe for a service worker; no 500, no HTML fallback."""
            return ('', 204)

        @self.app.route('/assets/<path:path>')
        def vite_assets(path):
            """Vite-bundled scripts and styles."""
            assets_dir = FRONTEND_DIST / 'assets'
            if not assets_dir.is_dir():
                return jsonify({'error': 'Not found'}), 404
            return send_from_directory(assets_dir, path)
        
        # API: Health check
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'model': self.model_type,
                'timestamp': time.time()
            }), 200
        
        # API: Model info
        @self.app.route('/api/info', methods=['GET'])
        def model_info():
            """Get model information."""
            return jsonify({
                'model_type': self.model_type,
                'classes': self.engine.CLASS_NAMES,
                'target_size': self.engine.target_size,
                'device': str(self.engine.device),
                'parameters': 24560711 if self.model_type == 'resnet50' else 0
            }), 200
        
        # API: Config (for frontend to detect GPU)
        @self.app.route('/api/config', methods=['GET'])
        def config():
            """Get frontend configuration."""
            gpu_available = torch.cuda.is_available()
            return jsonify({
                'gpu_available': gpu_available,
                'device': 'GPU' if gpu_available else 'CPU',
                'api_url': '/api'
            }), 200
        
        # API: Single prediction
        @self.app.route('/api/predict', methods=['POST'])
        def predict_single():
            """Single image prediction."""
            if 'image' not in request.files:
                return jsonify({'error': 'No image provided'}), 400
            
            try:
                file = request.files['image']
                
                # Load image
                img_pil = Image.open(io.BytesIO(file.read()))
                
                # Convert PIL to numpy (RGB)
                img_np = np.array(img_pil.convert('RGB'))
                
                # Preprocess
                img_tensor = self.engine.preprocessor.preprocess(img_np)
                
                # Predict
                start_time = time.time()
                img_batch = img_tensor.unsqueeze(0).to(self.engine.device)
                logits = self.engine.model(img_batch)
                inference_time = (time.time() - start_time) * 1000
                
                # Get probabilities
                probs = F.softmax(logits, dim=1)[0]
                predicted_id = int(torch.argmax(probs).item())
                confidence = float(probs[predicted_id].item())
                
                # Prepare response
                class_probs = {
                    self.engine.CLASS_NAMES[i]: float(probs[i].item())
                    for i in range(len(self.engine.CLASS_NAMES))
                }
                
                return jsonify({
                    'filename': file.filename,
                    'prediction': {
                        'metadata': {
                            'image_path': file.filename,
                            'inference_time_ms': inference_time
                        },
                        'prediction': {
                            'class': self.engine.CLASS_NAMES[predicted_id],
                            'class_id': predicted_id,
                            'confidence': confidence,
                            'confidence_percent': confidence * 100
                        },
                        'probabilities': class_probs
                    },
                    'success': True
                }), 200
            
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return jsonify({'error': str(e), 'success': False}), 500
        
        # API: Batch prediction
        @self.app.route('/api/predict-batch', methods=['POST'])
        def predict_batch():
            """Batch image prediction."""
            if 'images' not in request.files:
                return jsonify({'error': 'No images provided'}), 400
            
            try:
                files = request.files.getlist('images')
                results = []
                
                for file in files:
                    img_pil = Image.open(io.BytesIO(file.read()))
                    img_np = np.array(img_pil.convert('RGB'))
                    
                    # Preprocess
                    img_tensor = self.engine.preprocessor.preprocess(img_np)
                    
                    start_time = time.time()
                    img_batch = img_tensor.unsqueeze(0).to(self.engine.device)
                    logits = self.engine.model(img_batch)
                    inference_time = (time.time() - start_time) * 1000
                    
                    # Get probabilities
                    probs = F.softmax(logits, dim=1)[0]
                    predicted_id = int(torch.argmax(probs).item())
                    confidence = float(probs[predicted_id].item())
                    
                    results.append({
                        'filename': file.filename,
                        'prediction': {
                            'class': self.engine.CLASS_NAMES[predicted_id],
                            'confidence': confidence,
                            'confidence_percent': confidence * 100
                        },
                        'inference_time_ms': inference_time
                    })
                
                return jsonify({
                    'predictions': results,
                    'count': len(results),
                    'success': True
                }), 200
            
            except Exception as e:
                logger.error(f"Error during batch prediction: {e}")
                return jsonify({'error': str(e), 'success': False}), 500

        @self.app.route('/<path:path>')
        def dist_public(path):
            """Other static files from the Vite build (e.g. favicon.svg). Registered last so /api/* wins."""
            if path.startswith('api') or path.startswith('assets'):
                return jsonify({'error': 'Not found'}), 404
            base = FRONTEND_DIST.resolve()
            candidate = (FRONTEND_DIST / path).resolve()
            try:
                candidate.relative_to(base)
            except ValueError:
                return jsonify({'error': 'Not found'}), 404
            if candidate.is_file():
                return send_file(candidate)
            return jsonify({'error': 'Not found'}), 404
        
        @self.app.errorhandler(404)
        def not_found(e):
            if request.path.startswith('/api'):
                return jsonify({'error': 'Not found'}), 404
            suffix = Path(request.path).suffix.lower()
            static_exts = {
                '.ico', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp',
                '.js', '.css', '.map', '.woff', '.woff2', '.ttf', '.eot',
            }
            if suffix in static_exts:
                return '', 404
            built = FRONTEND_DIST / 'index.html'
            if built.is_file():
                return send_file(built, mimetype='text/html; charset=utf-8')
            if LEGACY_FRONTEND.is_file():
                return send_file(LEGACY_FRONTEND, mimetype='text/html; charset=utf-8')
            return jsonify({'error': 'Not found'}), 404
        
        @self.app.errorhandler(500)
        def server_error(e):
            logger.error(f"Server error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the web application."""
        logger.info(f"Starting web application on {host}:{port}")
        logger.info(f"Open browser at http://localhost:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Skin Cancer Detection Web Application')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-type', type=str, default='resnet50',
                       help='Model architecture type')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port number')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Log GPU info
    log_gpu_info()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        exit(1)
    
    # Create and run app
    web_app = SkinCancerWebApp(
        model_path=str(model_path),
        model_type=args.model_type
    )
    
    web_app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

