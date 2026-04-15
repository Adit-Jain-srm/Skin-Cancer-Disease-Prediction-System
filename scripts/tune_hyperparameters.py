"""
Phase 5: Hyperparameter Grid Search Orchestrator

Systematically evaluates different hyperparameter combinations
for both ResNet50 and EfficientNet-B3 models.

Usage:
    python tune_hyperparameters.py --models resnet50 efficientnet_b3 --quick
    python tune_hyperparameters.py --models resnet50 --comprehensive
"""

import torch
import subprocess
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import itertools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Grid search for optimal hyperparameters."""
    
    def __init__(self, results_dir: str = 'tuning_results'):
        """Initialize tuner."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def get_quick_configs(self) -> List[Dict]:
        """Quick tuning: minimal grid (2-3 options per param)."""
        return [
            # ResNet50 configs
            {
                'model': 'resnet50',
                'lr': 1e-3,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'augmentation': 'medium',
            },
            {
                'model': 'resnet50',
                'lr': 5e-4,
                'batch_size': 32,
                'weight_decay': 5e-5,
                'augmentation': 'strong',
            },
            # EfficientNet-B3 configs
            {
                'model': 'efficientnet_b3',
                'lr': 1e-3,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'augmentation': 'medium',
            },
            {
                'model': 'efficientnet_b3',
                'lr': 5e-4,
                'batch_size': 32,
                'weight_decay': 5e-5,
                'augmentation': 'strong',
            },
        ]
    
    def get_standard_configs(self) -> List[Dict]:
        """Standard tuning: 5-6 options per important param."""
        
        learning_rates = [1e-3, 5e-4, 1e-4]
        batch_sizes = [32, 48]
        weight_decays = [1e-4, 5e-5, 1e-5]
        augmentations = ['light', 'medium', 'strong']
        models = ['resnet50', 'efficientnet_b3']
        
        configs = []
        
        # ResNet50: 3 LR × 2 BS × 2 WD × 2 AUG = 24 configs
        for lr, bs, wd, aug in itertools.product(
            learning_rates[:2],  # 2 LR
            batch_sizes,         # 2 BS
            weight_decays[:2],   # 2 WD
            augmentations[1:]    # 2 AUG (medium, strong)
        ):
            configs.append({
                'model': 'resnet50',
                'lr': lr,
                'batch_size': bs,
                'weight_decay': wd,
                'augmentation': aug,
            })
        
        # EfficientNet-B3: subset (same as above)
        for lr, bs, wd, aug in itertools.product(
            learning_rates[:2],
            batch_sizes,
            weight_decays[:2],
            augmentations[1:]
        ):
            configs.append({
                'model': 'efficientnet_b3',
                'lr': lr,
                'batch_size': bs,
                'weight_decay': wd,
                'augmentation': aug,
            })
        
        return configs
    
    def get_comprehensive_configs(self) -> List[Dict]:
        """Comprehensive tuning: full grid."""
        
        learning_rates = [1e-3, 5e-4, 1e-4, 1e-5]
        batch_sizes = [32, 48, 64]
        weight_decays = [1e-4, 5e-5, 1e-5]
        augmentations = ['light', 'medium', 'strong']
        models = ['resnet50', 'efficientnet_b3']
        
        configs = []
        
        for model in models:
            for lr, bs, wd, aug in itertools.product(
                learning_rates,
                batch_sizes,
                weight_decays,
                augmentations
            ):
                configs.append({
                    'model': model,
                    'lr': lr,
                    'batch_size': bs,
                    'weight_decay': wd,
                    'augmentation': aug,
                })
        
        return configs
    
    def run_training(
        self,
        config: Dict,
        config_id: int,
        total_configs: int,
        epochs: int = 50  # Reduced for grid search
    ) -> Dict:
        """
        Run single training with given config.
        
        Args:
            config: Hyperparameter configuration
            config_id: Config number in grid
            total_configs: Total configs to try
            epochs: Number of training epochs
        
        Returns:
            Results dictionary with test accuracy and other metrics
        """
        
        logger.info(f"\n[{config_id}/{total_configs}] Running: {config}")
        
        # Build command
        cmd = [
            'python', 'train_transfer_learning.py',
            '--model', config['model'],
            '--lr', str(config['lr']),
            '--batch-size', str(config['batch_size']),
            '--weight-decay', str(config['weight_decay']),
            '--augmentation', config['augmentation'],
            '--epochs', str(epochs),
            '--patience', '5',  # Reduced patience for faster grid search
            '--results-dir', str(self.results_dir / f"config_{config_id}")
        ]
        
        try:
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
                return {
                    'config_id': config_id,
                    **config,
                    'test_accuracy': 0.0,
                    'status': 'failed',
                    'error': result.stderr[:200]
                }
            
            # Parse output to extract test accuracy
            for line in result.stdout.split('\n'):
                if 'Test Accuracy:' in line:
                    try:
                        accuracy = float(line.split(':')[1].strip().replace('%', ''))
                        logger.info(f"✓ Test Accuracy: {accuracy:.2f}%")
                        
                        return {
                            'config_id': config_id,
                            **config,
                            'test_accuracy': accuracy,
                            'status': 'success',
                            'timestamp': datetime.now().isoformat()
                        }
                    except Exception as e:
                        logger.error(f"Could not parse accuracy: {e}")
            
            logger.warning("Could not extract test accuracy from output")
            return {
                'config_id': config_id,
                **config,
                'test_accuracy': 0.0,
                'status': 'incomplete'
            }
        
        except subprocess.TimeoutExpired:
            logger.error("Training timeout (10 minutes)")
            return {
                'config_id': config_id,
                **config,
                'test_accuracy': 0.0,
                'status': 'timeout'
            }
        
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {
                'config_id': config_id,
                **config,
                'test_accuracy': 0.0,
                'status': 'error',
                'error': str(e)[:200]
            }
    
    def tune(
        self,
        models: List[str],
        grid_type: str = 'quick',
        epochs: int = 50
    ):
        """
        Execute grid search.
        
        Args:
            models: List of model names to tune
            grid_type: 'quick', 'standard', or 'comprehensive'
            epochs: Training epochs per config
        """
        
        # Get configurations
        if grid_type == 'quick':
            all_configs = self.get_quick_configs()
        elif grid_type == 'standard':
            all_configs = self.get_standard_configs()
        elif grid_type == 'comprehensive':
            all_configs = self.get_comprehensive_configs()
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        # Filter by requested models
        configs = [c for c in all_configs if c['model'] in models]
        
        logger.info("=" * 70)
        logger.info(f"HYPERPARAMETER GRID SEARCH")
        logger.info(f"Grid type: {grid_type}")
        logger.info(f"Total configs: {len(configs)}")
        logger.info(f"Epochs per config: {epochs}")
        logger.info("=" * 70)
        
        # Run grid search
        results = []
        for i, config in enumerate(configs, 1):
            result = self.run_training(config, i, len(configs), epochs=epochs)
            results.append(result)
            
            # Save intermediate results
            self.save_results(results)
        
        # Sort by test accuracy descending
        results.sort(key=lambda x: x['test_accuracy'], reverse=True)
        
        # Report top configurations
        logger.info("\n" + "=" * 70)
        logger.info("TOP CONFIGURATIONS")
        logger.info("=" * 70)
        
        for i, result in enumerate(results[:5], 1):
            logger.info(f"\n{i}. Model: {result['model']}")
            logger.info(f"   LR: {result['lr']}, BS: {result['batch_size']}, "
                       f"WD: {result['weight_decay']}, Aug: {result['augmentation']}")
            logger.info(f"   Test Accuracy: {result['test_accuracy']:.2f}%")
            if 'status' in result and result['status'] != 'success':
                logger.info(f"   Status: {result['status']}")
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✓ Grid search complete. Results saved to {self.results_dir}")
        logger.info("=" * 70)
        
        return results
    
    def save_results(self, results: List[Dict]):
        """Save results to JSON."""
        results_path = self.results_dir / 'grid_search_results.json'
        
        # Sort by accuracy descending
        sorted_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_configs': len(results),
                'results': sorted_results,
                'best_config': sorted_results[0] if sorted_results else None
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter Grid Search for Transfer Learning Models'
    )
    
    parser.add_argument('--models', nargs='+',
                       default=['resnet50', 'efficientnet_b3'],
                       choices=['resnet50', 'efficientnet_b3'],
                       help='Models to tune')
    
    grid_group = parser.add_mutually_exclusive_group()
    grid_group.add_argument('--quick', action='store_const', dest='grid_type',
                           const='quick', help='Quick grid (2-3 options each)')
    grid_group.add_argument('--standard', action='store_const', dest='grid_type',
                           const='standard', help='Standard grid (~24 configs per model)')
    grid_group.add_argument('--comprehensive', action='store_const', dest='grid_type',
                           const='comprehensive', help='Full grid (~288 configs per model)')
    
    parser.set_defaults(grid_type='quick')
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epochs per config')
    parser.add_argument('--results-dir', type=str, default='tuning_results',
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Run tuning
    tuner = HyperparameterTuner(results_dir=args.results_dir)
    results = tuner.tune(
        models=args.models,
        grid_type=args.grid_type,
        epochs=args.epochs
    )


if __name__ == '__main__':
    main()
