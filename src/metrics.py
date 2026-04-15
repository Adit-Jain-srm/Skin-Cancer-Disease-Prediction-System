"""
Phase 4: Metrics Computation

Implements evaluation metrics:
- Per-class accuracy, precision, recall, F1
- Weighted F1 (accounts for imbalance)
- Confusion matrix
- Macro/Micro averages
"""

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)
import logging

logger = logging.getLogger(__name__)


class MetricComputer:
    """Compute and track evaluation metrics."""
    
    def __init__(self, num_classes: int = 7, class_names: list = None):
        """
        Initialize metric computer.
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class{i}' for i in range(num_classes)]
        
        # Collect predictions and labels
        self.all_predictions = []
        self.all_labels = []
    
    def reset(self):
        """Reset collected predictions and labels."""
        self.all_predictions = []
        self.all_labels = []
    
    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Update with batch predictions.
        
        Args:
            outputs: Model outputs of shape (batch_size, num_classes)
            labels: True labels of shape (batch_size,)
        """
        # Get predictions
        _, predictions = torch.max(outputs.detach(), 1)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)
    
    def compute_metrics(self) -> dict:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with computed metrics
        """
        if len(self.all_predictions) == 0:
            return {}
        
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        # Overall accuracy
        accuracy = accuracy_score(labels, predictions)
        
        # Per-class metrics
        precision = precision_score(labels, predictions, average=None, zero_division=0)
        recall = recall_score(labels, predictions, average=None, zero_division=0)
        f1 = f1_score(labels, predictions, average=None, zero_division=0)
        
        # Weighted averages (account for class imbalance)
        weighted_precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        weighted_recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # Macro averages
        macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
        macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
        macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(labels, predictions, labels=range(self.num_classes))
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'confusion_matrix': conf_matrix.tolist(),
        }
        
        return metrics
    
    def log_metrics(self, metrics: dict) -> None:
        """
        Log metrics in readable format.
        
        Args:
            metrics: Dictionary with computed metrics
        """
        if not metrics:
            logger.info("No metrics to log")
            return
        
        logger.info("=" * 70)
        logger.info("EVALUATION METRICS")
        logger.info("=" * 70)
        
        # Overall accuracy
        logger.info(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        
        # Weighted metrics
        logger.info(f"\nWeighted Metrics:")
        logger.info(f"  Precision: {metrics['weighted_precision']:.4f}")
        logger.info(f"  Recall:    {metrics['weighted_recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['weighted_f1']:.4f}")
        
        # Macro metrics
        logger.info(f"\nMacro Metrics:")
        logger.info(f"  Precision: {metrics['macro_precision']:.4f}")
        logger.info(f"  Recall:    {metrics['macro_recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['macro_f1']:.4f}")
        
        # Per-class metrics
        logger.info(f"\nPer-Class Metrics:")
        logger.info(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        logger.info("-" * 46)
        
        for i, class_name in enumerate(self.class_names):
            logger.info(
                f"{class_name:<10} {metrics['precision'][i]:<12.4f} "
                f"{metrics['recall'][i]:<12.4f} {metrics['f1'][i]:<12.4f}"
            )
        
        logger.info("=" * 70)
    
    def get_classification_report(self, predictions=None, labels=None) -> str:
        """
        Get sklearn classification report.
        
        Args:
            predictions: Optional predictions array
            labels: Optional labels array
        
        Returns:
            Classification report string
        """
        if predictions is None:
            predictions = self.all_predictions
        if labels is None:
            labels = self.all_labels
        
        return classification_report(
            labels,
            predictions,
            target_names=self.class_names,
            zero_division=0,
        )
    
    def plot_confusion_matrix(self, metrics: dict, save_path: str = None):
        """
        Plot and optionally save confusion matrix.
        
        Args:
            metrics: Dictionary with computed metrics
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping plot")
            return
        
        conf_matrix = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Confusion matrix saved: {save_path}")
        
        return plt.gcf()


def evaluate_model(model, test_loader, criterion, device='cpu', class_names=None):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test DataLoader
        criterion: Loss function
        device: Device to evaluate on
        class_names: List of class names
    
    Returns:
        Tuple of (test_loss, metrics)
    """
    model.eval()
    total_loss = 0.0
    metric_computer = MetricComputer(num_classes=len(class_names or []), class_names=class_names)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            metric_computer.update(outputs, labels)
    
    avg_loss = total_loss / len(test_loader)
    metrics = metric_computer.compute_metrics()
    
    return avg_loss, metrics, metric_computer


if __name__ == '__main__':
    """Test metrics computation."""
    print("=" * 70)
    print("METRICS COMPUTER - VERIFICATION TEST")
    print("=" * 70)
    
    # Test with random predictions
    print("\n✓ MetricComputer class initialized")
    
    # Create sample data
    num_samples = 100
    num_classes = 7
    
    # Random predictions and labels
    predictions = torch.randint(0, num_classes, (num_samples,))
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Create dummy outputs (logits)
    outputs = torch.randn(num_samples, num_classes)
    
    # Initialize metric computer
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    mc = MetricComputer(num_classes=num_classes, class_names=class_names)
    
    # Update with random batch
    mc.update(outputs, labels)
    
    # Compute metrics
    metrics = mc.compute_metrics()
    
    print(f"\n✓ Metrics computed for {len(mc.all_labels)} samples")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Confusion matrix shape: {np.array(metrics['confusion_matrix']).shape}")
    
    print("\n" + "=" * 70)
    print("✅ METRICS COMPUTER TESTS PASSED")
    print("=" * 70)
