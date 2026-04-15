"""
predict.py - Model Prediction Script
Makes predictions on new skin lesion images.

Usage:
    python predict.py --image <path> --model models/best_model.pt
    python predict.py --image sample.jpg
    python predict.py --batch <folder> --model models/best_model.pt
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import setup_logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Run predictions on skin lesion images"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image for prediction"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to folder for batch prediction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to trained model"
    )
    
    args = parser.parse_args()
    
    setup_logging("INFO")
    logger.info("Starting predictions...")
    
    if not args.image and not args.batch:
        logger.error("Provide either --image or --batch argument")
        sys.exit(1)
    
    # TODO: Implement prediction
    logger.info("Prediction completed")

if __name__ == "__main__":
    main()
