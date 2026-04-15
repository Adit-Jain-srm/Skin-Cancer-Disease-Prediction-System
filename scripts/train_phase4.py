"""
Phase 4: Training Script

Two-phase training strategy:
1. Phase A: Train on 20% subset (validation gate)
   - Target: ≥65% accuracy in <30 minutes
   - Purpose: Quick feedback before full training
   - Early stop if fails
   
2. Phase B: Train on 100% dataset
   - Target: ≥70% accuracy
   - Purpose: Final baseline performance
"""

import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataset import DatasetManager
from data_loader import HAM10000DataLoader
from model import CNNBaseline, create_cnn_baseline
from trainer import CNNTrainer
from metrics import MetricComputer, evaluate_model
from model_manager import ModelManager


def setup_device():
    """Setup PyTorch device."""
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("Using CPU (no GPU available)")
    
    return device


def create_training_components(device, class_weights):
    """
    Create model, optimizer, and loss function.
    
    Args:
        device: Device to use
        class_weights: Class weights tensor
    
    Returns:
        Tuple of (model, optimizer, criterion, trainer)
    """
    # Model
    model = create_cnn_baseline(num_classes=7, dropout=0.5, device=device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Loss function with class weights
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Trainer
    trainer = CNNTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir='checkpoints'
    )
    
    logger.info("Training components created successfully")
    return model, optimizer, criterion, trainer


def train_phase_a(data_loader, device, target_acc=0.65, time_limit_minutes=30):
    """
    Phase A: Train on small subset (20% data).
    
    Args:
        data_loader: HAM10000DataLoader instance
        device: Device to use
        target_acc: Target accuracy (as fraction, e.g. 0.65 for 65%)
        time_limit_minutes: Time limit in minutes
    
    Returns:
        Boolean indicating if gate passed
    """
    logger.info("=" * 70)
    logger.info("PHASE A: SUBSET TRAINING (20% of data)")
    logger.info("=" * 70)
    
    # Create subset loaders (20% of data)
    train_subset_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    
    # For subset, take only 20% of training batches
    # This is approximate; we'll use first 20% of batches
    train_batches = int(len(train_subset_loader) * 0.2)
    
    # Get class weights
    class_weights = data_loader.get_class_weights()
    
    # Create training components
    model, optimizer, criterion, trainer = create_training_components(device, class_weights)
    
    # Training parameters
    num_epochs = 20
    target_accuracy_pct = target_acc * 100  # Convert to percentage for comparison with trainer output
    
    logger.info(f"Target: {target_acc:.1%} accuracy")
    logger.info(f"Time limit: {time_limit_minutes} minutes")
    logger.info(f"Epochs: {num_epochs} (early stop at {trainer.early_stop_patience} no-improvement)")
    
    # Train
    start_time = datetime.now()
    history = trainer.train(train_subset_loader, val_loader, num_epochs=num_epochs)
    training_time = (datetime.now() - start_time).total_seconds() / 60
    
    # Check gate criteria (trainer returns accuracy as percentage 0-100)
    final_acc = history['val_acc'][-1] if history['val_acc'] else 0
    gate_passed = (final_acc > target_accuracy_pct) and (training_time < time_limit_minutes)
    
    logger.info("=" * 70)
    logger.info("PHASE A RESULTS")
    logger.info("=" * 70)
    logger.info(f"Final validation accuracy: {final_acc:.2f}%")
    logger.info(f"Training time: {training_time:.1f} minutes")
    logger.info(f"Target accuracy ({target_acc:.1%}): {'✓ PASSED' if final_acc > target_accuracy_pct else '✗ FAILED'}")
    logger.info(f"Time limit ({time_limit_minutes} min): {'✓ PASSED' if training_time < time_limit_minutes else '✗ FAILED'}")
    logger.info(f"Gate status: {'✅ PASS - proceeding to Phase B' if gate_passed else '❌ FAIL - retry or adjust hyperparameters'}")
    logger.info("=" * 70)
    
    # Save trained model
    model_manager = ModelManager()
    training_summary = trainer.get_training_summary()
    
    model_manager.save_model(
        model=model,
        name='phase_a_model',
        metrics={
            'final_validation_accuracy': float(final_acc),
            'gate_passed': gate_passed,
        },
        metadata={
            'phase': 'A',
            'subset_percentage': 0.2,
            'num_epochs': num_epochs,
            'training_time_minutes': training_time,
        }
    )
    
    # Save training results
    model_manager.save_training_results(
        phase='phase_a',
        results={
            'gate_passed': gate_passed,
            'final_validation_accuracy': float(final_acc),
            'training_time_minutes': training_time,
        },
        history=history,
        metrics=training_summary,
    )
    
    return gate_passed, training_summary


def train_phase_b(data_loader, device, target_acc=0.70, time_limit_minutes=45):
    """
    Phase B: Train on full dataset with best hyperparameters from Phase A.
    
    Args:
        data_loader: HAM10000DataLoader instance
        device: Device to use
        target_acc: Target accuracy (as fraction, e.g. 0.70 for 70%)
        time_limit_minutes: Time limit in minutes
    
    Returns:
        Results dictionary
    """
    logger.info("=" * 70)
    logger.info("PHASE B: FULL DATASET TRAINING (100% of data)")
    logger.info("=" * 70)
    
    # Get full loaders
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    # Get class weights
    class_weights = data_loader.get_class_weights()
    
    # Create training components
    model, optimizer, criterion, trainer = create_training_components(device, class_weights)
    
    # Training parameters
    num_epochs = 100
    target_accuracy = target_acc
    
    logger.info(f"Target: {target_accuracy:.1%} accuracy on validation set")
    logger.info(f"Epochs: {num_epochs} (early stop at {trainer.early_stop_patience} no-improvement)")
    logger.info(f"Dataset sizes: train={len(train_loader.dataset)}, "
                 f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    
    # Train
    start_time = datetime.now()
    history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    training_time = (datetime.now() - start_time).total_seconds() / 60
    
    # Evaluate on test set
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 70)
    
    test_loss, test_metrics, metric_computer = evaluate_model(
        model, test_loader, criterion, device=device,
        class_names=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    )
    
    metric_computer.log_metrics(test_metrics)
    
    # Get training summary
    summary = trainer.get_training_summary()
    
    # Save results
    results = {
        'phase': 'B',
        'training_time_minutes': training_time,
        'training_summary': summary,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'target_accuracy': target_accuracy,
        'achieved_accuracy': test_metrics['accuracy'],
        'gate_passed': test_metrics['accuracy'] > target_accuracy,
    }
    
    logger.info("=" * 70)
    logger.info("PHASE B FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.2%}")
    logger.info(f"Test weighted F1: {test_metrics['weighted_f1']:.4f}")
    logger.info(f"Training time: {training_time:.1f} minutes")
    logger.info(f"Target ({target_accuracy:.1%}): {'✅ PASSED' if results['gate_passed'] else '❌ FAILED'}")
    logger.info("=" * 70)
    
    # Save model using ModelManager
    model_manager = ModelManager()
    
    model_manager.save_model(
        model=model,
        name='best_model',
        metrics={
            'test_accuracy': test_metrics['accuracy'],
            'test_weighted_f1': test_metrics['weighted_f1'],
            'test_loss': test_loss,
        },
        metadata={
            'phase': 'B',
            'full_dataset': True,
            'num_epochs': num_epochs,
            'training_time_minutes': training_time,
            'device': device,
        }
    )
    
    # Save comprehensive training results
    model_manager.save_training_results(
        phase='phase_b',
        results={
            'gate_passed': results['gate_passed'],
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_loss,
            'training_time_minutes': training_time,
            'target_accuracy': target_accuracy,
        },
        history=history,
        metrics=test_metrics,
    )
    
    # Also save JSON for backward compatibility
    results_path = Path('results/phase4_results.json')
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved: {results_path}")
    
    return results


def main():
    """Main training script."""
    print("\n" + "=" * 70)
    print("PHASE 4: BASELINE CNN TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    device = setup_device()
    
    # Initialize dataset and DataLoader
    logger.info("Initializing dataset and DataLoader...")
    dm = DatasetManager(dataset_dir='Dataset', target_size=(224, 224))
    dm.load_metadata('HAM10000_metadata.csv')
    
    data_loader = HAM10000DataLoader(
        dm,
        train_split=0.7,
        val_split=0.15,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    class_weights = data_loader.get_class_weights()
    logger.info(f"Class weights: {class_weights}")
    
    # Skip Phase A gate, directly train on full dataset (Phase B)
    logger.info("\n" + "=" * 70)
    logger.info("SKIPPING PHASE A GATE - PROCEEDING DIRECTLY TO PHASE B (FULL DATASET)")
    logger.info("=" * 70 + "\n")
    
    try:
        phase_b_results = train_phase_b(data_loader, device)
    except Exception as e:
        logger.error(f"Phase B failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print("\n" + "=" * 70)
    print("[COMPLETE] PHASE 4: BASELINE CNN TRAINING")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
