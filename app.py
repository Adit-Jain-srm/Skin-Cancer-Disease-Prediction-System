"""
app.py - Flask Web Application
Runs the Flask web server for skin lesion prediction UI.

Usage:
    python app.py --port 5000
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.app import FlaskApp
from src.utils import setup_logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Run Flask web application"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run Flask server on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind Flask server to"
    )
    
    args = parser.parse_args()
    
    setup_logging("INFO")
    logger.info("Starting Flask web application...")
    
    # TODO: Create and run Flask app
    # app = FlaskApp(args.model)
    # app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
