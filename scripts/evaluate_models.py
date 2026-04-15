"""
Phase 5: Model Evaluation and Selection

Evaluates trained models and selects the best performer based on:
- Overall test accuracy
- Per-class metrics (precision, recall, F1)
- Balanced performance across minority classes
- Computational efficiency

Usage:
    python evaluate_models.py --model resnet50
    python evaluate_models.py --model best_model --select-by balanced_accuracy
"""

import torch
import torch.nn as nn
import logging
import argparse
import json
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    balanced_accuracy_score,
    f1_score
)

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.transfer_learning import TransferLearningModel
from src.dataset import DatasetManager
from src.data_loader import HAM10000DataLoader
from src.metrics import MetricComputer
from src.enhanced_augmentation import AugmentedDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(
        self,
        model_path: str,
        model_name: str,
        device: Optional[torch.device] = None,
        results_dir: str = 'evaluation_results'
    ):
        """Initialize evaluator."""
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Device: {self.device}")
    
    def load_model(self) -> nn.Module:
        """Load trained model."""
        logger.info("Loading model...")
        
        if self.model_name == 'resnet50':
            model = TransferLearningModel.create_resnet50(
                num_classes=7,
                pretrained=False,
                freeze_backbone=False
            )
        elif self.model_name == 'efficientnet_b3':
            model = TransferLearningModel.create_efficientnet_b3(
                num_classes=7,
                pretrained=False,
                freeze_backbone=False
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Load weights
        if self.model_path.exists():
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info(f"✓ Model loaded from {self.model_path}")
        else:
            logger.warning(f"Model file not found: {self.model_path}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate(
        self,
        model: nn.Module,
        test_loader,
        class_names: List[str]
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            class_names: Class name mappings
        
        Returns:
            Dictionary with comprehensive metrics
        """
        
        logger.info("\nEvaluating on test set...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Overall metrics
        accuracy = (all_preds == all_labels).mean() * 100
        balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100
        
        # Per-class metrics
        report: Dict[str, Any] = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EVALUATION RESULTS - {self.model_name}")
        logger.info(f"{'=' * 70}")
        logger.info(f"\nOverall Performance:")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        logger.info(f"  Balanced Accuracy: {balanced_acc:.2f}%")
        
        logger.info(f"\nPer-Class Metrics:")
        logger.info(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        logger.info("-" * 56)
        
        for class_name in class_names:
            if class_name in report:
                metrics: Dict[str, Any] = report[class_name]  # type: ignore
                logger.info(
                    f"{class_name:<10} "
                    f"{metrics['precision']:<12.4f} "
                    f"{metrics['recall']:<12.4f} "
                    f"{metrics['f1-score']:<12.4f} "
                    f"{int(metrics['support']):<10}"
                )
        
        # Weighted and macro averages
        logger.info("-" * 56)
        weighted_avg: Dict[str, Any] = report['weighted avg']  # type: ignore
        logger.info(
            f"{'Weighted':<10} "
            f"{weighted_avg['precision']:<12.4f} "
            f"{weighted_avg['recall']:<12.4f} "
            f"{weighted_avg['f1-score']:<12.4f}"
        )
        
        # Minority class performance
        minority_classes = ['df', 'nv']  # Usually the imbalanced classes
        minority_f1_scores = []
        for cls in minority_classes:
            if cls in report:
                cls_metrics: Dict[str, Any] = report[cls]  # type: ignore
                minority_f1_scores.append(cls_metrics['f1-score'])
        
        if minority_f1_scores:
            avg_minority_f1 = np.mean(minority_f1_scores)
            logger.info(f"\nMinority Classes Avg F1: {avg_minority_f1:.4f}")
        
        logger.info(f"{'=' * 70}\n")
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'per_class_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probs.tolist(),
            'class_names': class_names
        }
    
    def plot_confusion_matrix(
        self,
        eval_results: Dict,
        save_path: str = None
    ):
        """Plot and save confusion matrix."""
        conf_matrix = np.array(eval_results['confusion_matrix'])
        class_names = eval_results['class_names']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.results_dir / f'{self.model_name}_confusion_matrix.png'
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"✓ Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_per_class_metrics(
        self,
        eval_results: Dict,
        save_path: str = None
    ):
        """Plot per-class metrics."""
        report = eval_results['per_class_report']
        class_names = eval_results['class_names']
        
        metrics = ['precision', 'recall', 'f1-score']
        data = {
            metric: [report[cls][metric] for cls in class_names]
            for metric in metrics
        }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(class_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, data[metric], width, label=metric)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Metrics - {self.model_name}')
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        if save_path is None:
            save_path = self.results_dir / f'{self.model_name}_per_class_metrics.png'
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"✓ Per-class metrics plot saved to {save_path}")
        plt.close()
    
    def save_evaluation_report(self, eval_results: Dict):
        """Save evaluation report to JSON."""
        report_path = self.results_dir / f'{self.model_name}_evaluation.json'
        
        # Create JSON-serializable version
        serializable_results = {
            'model': self.model_name,
            'accuracy': float(eval_results['accuracy']),
            'balanced_accuracy': float(eval_results['balanced_accuracy']),
            'per_class_report': eval_results['per_class_report'],
            'confusion_matrix': eval_results['confusion_matrix'],
        }
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"✓ Evaluation report saved to {report_path}")
    
    def run_evaluation(self, test_loader, class_names: List[str]) -> Dict:
        """Execute full evaluation pipeline."""
        
        # Load model
        model = self.load_model()
        
        # Evaluate
        eval_results = self.evaluate(model, test_loader, class_names)
        
        # Plot results
        self.plot_confusion_matrix(eval_results)
        self.plot_per_class_metrics(eval_results)
        
        # Save report
        self.save_evaluation_report(eval_results)
        
        return eval_results


class ModelSelector:
    """Select best model from grid search results."""
    
    @staticmethod
    def select_best(
        results_json: str,
        select_by: str = 'accuracy'
    ) -> Dict:
        """
        Select best model from grid search results.
        
        Args:
            results_json: Path to grid search results JSON
            select_by: 'accuracy', 'balanced_accuracy', or 'f1'
        
        Returns:
            Best configuration
        """
        
        with open(results_json, 'r') as f:
            data = json.load(f)
        
        results = data['results']
        
        if select_by == 'accuracy':
            best = max(results, key=lambda x: x.get('test_accuracy', 0))
        elif select_by == 'balanced_accuracy':
            # Prefer balanced accuracy for imbalanced datasets
            best = max(results, key=lambda x: x.get('balanced_accuracy', 0))
        else:
            raise ValueError(f"Unknown selection criteria: {select_by}")
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"BEST MODEL SELECTED (by {select_by})")
        logger.info(f"{'=' * 70}")
        logger.info(f"Model: {best['model']}")
        logger.info(f"Learning Rate: {best['lr']}")
        logger.info(f"Batch Size: {best['batch_size']}")
        logger.info(f"Weight Decay: {best['weight_decay']}")
        logger.info(f"Augmentation: {best['augmentation']}")
        logger.info(f"Test Accuracy: {best['test_accuracy']:.2f}%")
        logger.info(f"{'=' * 70}\n")
        
        return best


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate and select best transfer learning model'
    )
    
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'efficientnet_b3'],
                       help='Model to evaluate')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model weights')
    parser.add_argument('--data-dir', type=str, default='Dataset',
                       help='Path to HAM10000 dataset')
    parser.add_argument('--select-by', type=str, default='accuracy',
                       choices=['accuracy', 'balanced_accuracy'],
                       help='Selection criteria for best model')
    parser.add_argument('--results-dir', type=str, default='evaluation_results',
                       help='Results directory')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("PHASE 5: MODEL EVALUATION & SELECTION")
    logger.info("=" * 70)
    
    # Load dataset
    logger.info("\nLoading test set...")
    dm = DatasetManager(
        dataset_dir=args.data_dir,
        target_size=(224, 224)
    )
    dm.load_metadata('HAM10000_metadata.csv')
    
    data_loader = HAM10000DataLoader(
        dm,
        train_split=0.70,
        val_split=0.15,
        batch_size=32,
        num_workers=4,
        random_state=42
    )
    
    # Get test loader
    test_loader = data_loader.get_test_loader()
    test_labels = data_loader.test_metadata['dx'].values
    class_names = data_loader.unique_classes
    
    # Evaluate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = f'checkpoints/best_model.pt'
    
    evaluator = ModelEvaluator(
        model_path=model_path,
        model_name=args.model,
        device=device,
        results_dir=args.results_dir
    )
    
    eval_results = evaluator.run_evaluation(test_loader, class_names)
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ EVALUATION COMPLETE")
    logger.info(f"Results saved to: {args.results_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
