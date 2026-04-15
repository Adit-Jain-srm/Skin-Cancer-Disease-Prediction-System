"""
Web UI Module
Flask-based web interface for image upload and prediction.
"""

import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class FlaskApp:
    """
    Flask web application for skin lesion prediction.
    
    Routes:
    - GET /: Home page with upload form
    - POST /predict: Upload image and get prediction
    - GET /results: Display prediction results
    """
    
    def __init__(self, model_path: str, max_file_size_mb: int = 10):
        """
        Initialize Flask app.
        
        Args:
            model_path: Path to saved model
            max_file_size_mb: Maximum allowed file size in MB
        """
        self.model_path = model_path
        self.max_file_size_mb = max_file_size_mb
        self.app = None
        self.model = None
        logger.info(f"FlaskApp initialized with model {model_path}")
    
    def create_app(self):
        """Create and configure Flask app."""
        # TODO: Implement using Flask
        # 1. Load model from model_path
        # 2. Set up routes (/, /predict, /results)
        # 3. Configure upload folder and file validation
        pass
    
    def validate_upload(self, file) -> Tuple[bool, str]:
        """
        Validate uploaded file.
        
        Args:
            file: Uploaded file object
            
        Returns:
            (is_valid, error_message)
        """
        if file is None:
            return False, "No file uploaded"
        
        if file.filename == "":
            return False, "No file selected"
        
        if not self._is_allowed_extension(file.filename):
            return False, "Invalid file format. Allowed: JPG, PNG"
        
        # TODO: Check file size
        return True, ""
    
    def _is_allowed_extension(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        allowed = {'.jpg', '.jpeg', '.png'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed
    
    def predict_from_upload(self, file) -> Dict:
        """
        Run prediction on uploaded file.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Dictionary with prediction, confidence, and metadata
        """
        is_valid, error_msg = self.validate_upload(file)
        if not is_valid:
            return {"success": False, "error": error_msg}
        
        try:
            # TODO: Preprocess image
            # TODO: Run inference
            # TODO: Return results
            pass
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run(self, host: str = "localhost", port: int = 5000, debug: bool = False):
        """Run Flask development server."""
        logger.info(f"Starting Flask app on {host}:{port}")
        # TODO: Start Flask app
        pass


# TODO: Add HTML templates (base.html, upload.html, results.html)
# TODO: Add CSS styling
# TODO: Add JavaScript for client-side validation
