#!/usr/bin/env python3
"""Quick integration test for Phase 5 training pipeline."""

import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports work."""
    logger.info("Testing imports...")
    try:
        from src.dataset import DatasetManager
        logger.info("✓ DatasetManager imported")
        
        from src.data_loader import HAM10000DataLoader
        logger.info("✓ HAM10000DataLoader imported")
        
        from src.transfer_learning import TransferLearningModel
        logger.info("✓ TransferLearningModel imported")
        
        from src.enhanced_trainer import EnhancedTrainer
        logger.info("✓ EnhancedTrainer imported")
        
        from src.enhanced_augmentation import AugmentationFactory
        logger.info("✓ AugmentationFactory imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_data_loading():
    """Test data loading pipeline."""
    logger.info("\nTesting data loading...")
    try:
        from src.dataset import DatasetManager
        from src.data_loader import HAM10000DataLoader
        
        dm = DatasetManager(dataset_dir='Dataset', target_size=(224, 224))
        logger.info("✓ DatasetManager created")
        
        dm.load_metadata('HAM10000_metadata.csv')
        logger.info("✓ Metadata loaded")
        
        data_loader = HAM10000DataLoader(
            dm,
            train_split=0.70,
            val_split=0.15,
            batch_size=32,
            shuffle=True,
            num_workers=0,  # Use 0 for quick test
            random_state=42
        )
        logger.info("✓ HAM10000DataLoader created")
        
        train_loader = data_loader.get_train_loader()
        logger.info(f"✓ Train loader created: {len(train_loader)} batches")
        
        val_loader = data_loader.get_val_loader()
        logger.info(f"✓ Val loader created: {len(val_loader)} batches")
        
        test_loader = data_loader.get_test_loader()
        logger.info(f"✓ Test loader created: {len(test_loader)} batches")
        
        # Get one batch to verify
        images, labels = next(iter(train_loader))
        logger.info(f"✓ Got batch: images shape {images.shape}, labels shape {labels.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}", exc_info=True)
        return False

def test_model_creation():
    """Test model creation."""
    logger.info("\nTesting model creation...")
    try:
        from src.transfer_learning import TransferLearningModel
        import torch
        
        model = TransferLearningModel.create_resnet50(
            num_classes=7,
            pretrained=True,
            freeze_backbone=True
        )
        logger.info("✓ ResNet50 model created")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = model(dummy_input)
        logger.info(f"✓ Forward pass successful: output shape {output.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}", exc_info=True)
        return False

def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("PHASE 5 INTEGRATION TEST")
    logger.info("=" * 70)
    
    results = {
        "Imports": test_imports(),
        "Data Loading": test_data_loading(),
        "Model Creation": test_model_creation(),
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST RESULTS")
    logger.info("=" * 70)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    logger.info("=" * 70)
    if all_passed:
        logger.info("ALL TESTS PASSED ✓")
        return 0
    else:
        logger.info("SOME TESTS FAILED ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())
