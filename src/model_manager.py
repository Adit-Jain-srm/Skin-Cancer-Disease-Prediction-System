"""
Model Management Module

Handles:
- Model saving/loading
- Checkpoint organization
- Model versioning  
- Results persistence
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model checkpoints and results."""
    
    def __init__(self, models_dir: str = 'models', results_dir: str = 'results'):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory for trained models
            results_dir: Directory for results and logs
        """
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"ModelManager initialized")
        logger.info(f"  Models dir: {self.models_dir.absolute()}")
        logger.info(f"  Results dir: {self.results_dir.absolute()}")
    
    def save_model(
        self,
        model: torch.nn.Module,
        name: str,
        metrics: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save trained model.
        
        Args:
            model: PyTorch model
            name: Model name (e.g., 'best_model', 'phase_a_model')
            metrics: Performance metrics
            metadata: Additional metadata
        
        Returns:
            Path to saved model
        """
        model_path = self.models_dir / f"{name}.pt"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'metadata': metadata or {},
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"✓ Model saved: {model_path}")
        
        # Save metadata as JSON
        meta_path = self.models_dir / f"{name}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        logger.info(f"✓ Metadata saved: {meta_path}")
        return model_path
    
    def load_model(self, model_obj: torch.nn.Module, name: str, device: str = 'cpu') -> torch.nn.Module:
        """
        Load trained model.
        
        Args:
            model_obj: Model object to load weights into
            name: Model name
            device: Device to load onto
        
        Returns:
            Model with loaded weights
        """
        model_path = self.models_dir / f"{name}.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        model_obj.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"✓ Model loaded: {model_path}")
        logger.info(f"  Saved at: {checkpoint['timestamp']}")
        if checkpoint['metrics']:
            logger.info(f"  Metrics: {checkpoint['metrics']}")
        
        return model_obj
    
    def save_training_results(
        self,
        phase: str,
        results: Dict,
        history: Dict,
        metrics: Dict,
    ) -> Path:
        """
        Save comprehensive training results.
        
        Args:
            phase: Training phase (e.g., 'phase_a', 'phase_b')
            results: Training results
            history: Training history (losses, accuracies)
            metrics: Evaluation metrics
        
        Returns:
            Path to results file
        """
        # Create results dictionary
        full_results = {
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'training_results': results,
            'training_history': history,
            'evaluation_metrics': metrics,
        }
        
        # Save as JSON
        results_path = self.results_dir / f"{phase}_results.json"
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        logger.info(f"✓ Results saved: {results_path}")
        return results_path
    
    def save_confusion_matrix(self, confusion_matrix: np.ndarray, name: str) -> Path:
        """
        Save confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            name: Matrix name
        
        Returns:
            Path to saved matrix
        """
        matrix_path = self.results_dir / f"{name}_confusion_matrix.npy"
        np.save(matrix_path, confusion_matrix)
        logger.info(f"✓ Confusion matrix saved: {matrix_path}")
        return matrix_path
    
    def get_best_model_path(self) -> Optional[Path]:
        """Get path to best model if it exists."""
        best_model = self.models_dir / "best_model.pt"
        return best_model if best_model.exists() else None
    
    def list_models(self) -> list:
        """List all saved models."""
        models = list(self.models_dir.glob("*.pt"))
        logger.info(f"Found {len(models)} trained models:")
        for m in sorted(models):
            logger.info(f"  - {m.name}")
        return sorted(models)
    
    def list_results(self) -> list:
        """List all saved results."""
        results = list(self.results_dir.glob("*.json"))
        logger.info(f"Found {len(results)} result files:")
        for r in sorted(results):
            logger.info(f"  - {r.name}")
        return sorted(results)
    
    def verify_training_completion(self, phase: str) -> Tuple[bool, str]:
        """
        Verify if training completed for a phase.
        
        Args:
            phase: Training phase (e.g., 'phase_a')
        
        Returns:
            Tuple of (completed: bool, message: str)
        """
        model_file = self.models_dir / f"{phase}_model.pt"
        results_file = self.results_dir / f"{phase}_results.json"
        
        model_exists = model_file.exists()
        results_exists = results_file.exists()
        
        if model_exists and results_exists:
            return True, f"✓ Phase complet: model and results saved"
        elif model_exists:
            return False, f"✗ Model saved but results missing: {results_file}"
        elif results_exists:
            return False, f"✗ Results saved but model missing: {model_file}"
        else:
            return False, f"✗ Neither model nor results found for {phase}"
    
    def get_training_summary(self) -> Dict:
        """Get summary of all training progress."""
        all_models = self.list_models()
        all_results = self.list_results()
        
        summary = {
            'total_models': len(all_models),
            'total_results': len(all_results),
            'models': [m.name for m in all_models],
            'results': [r.name for r in all_results],
        }
        
        # Check for best model
        best_model = self.get_best_model_path()
        if best_model:
            summary['best_model'] = best_model.name
            summary['best_model_path'] = str(best_model.absolute())
        
        return summary
