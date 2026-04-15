"""
evaluate.py - Model Evaluation Script
Evaluates model performance on test set.

Usage:
    python evaluate.py --model models/best_model.pt
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import setup_logging, ensure_directory

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate skin cancer classification model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    
    args = parser.parse_args()
    
    setup_logging("INFO")
    ensure_directory("reports")
    
    logger.info(f"Evaluating model: {args.model}")
    # TODO: Load model, compute metrics, save report
    logger.info("Evaluation completed")

if __name__ == "__main__":
    main()
