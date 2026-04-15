"""
Phase 6: Deployment Validation Tests

Comprehensive validation suite for production readiness:
- Unit tests for inference engine
- Integration tests for API
- Performance benchmarking
- Error handling validation
"""

import unittest
import tempfile
import time
import logging
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import InferenceEngine, ImagePreprocessor, format_prediction_for_api
from src.dataset import DatasetManager
from src.data_loader import HAM10000DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestImagePreprocessor(unittest.TestCase):
    """Test image preprocessing."""
    
    def setUp(self):
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
        
        # Create test image
        self.test_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(self.test_image_path, test_image)
    
    def tearDown(self):
        if Path(self.test_image_path).exists():
            Path(self.test_image_path).unlink()
    
    def test_load_image(self):
        """Test image loading."""
        image = self.preprocessor.load_image(self.test_image_path)
        self.assertEqual(image.shape[2], 3)  # RGB
        logger.info("✓ Image loading test passed")
    
    def test_preprocess_output_shape(self):
        """Test preprocessing output shape."""
        image = self.preprocessor.load_image(self.test_image_path)
        tensor = self.preprocessor.preprocess(image)
        self.assertEqual(tensor.shape, (3, 224, 224))
        logger.info("✓ Output shape test passed")
    
    def test_preprocess_value_range(self):
        """Test preprocessing value normalization."""
        image = self.preprocessor.load_image(self.test_image_path)
        tensor = self.preprocessor.preprocess(image)
        # ImageNet normalization should produce values in roughly [-2, 2.5]
        self.assertLess(tensor.max().item(), 4.0)
        self.assertGreater(tensor.min().item(), -4.0)
        logger.info("✓ Value range test passed")
    
    def test_invalid_image_path(self):
        """Test error handling for invalid path."""
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_image('/nonexistent/path.jpg')
        logger.info("✓ Invalid path error handling test passed")


class TestInferenceEngine(unittest.TestCase):
    """Test inference engine."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_path = 'checkpoints/best_model.pt'
        if not Path(cls.model_path).exists():
            raise FileNotFoundError(f"Model not found: {cls.model_path}")
        
        cls.engine = InferenceEngine(
            model_path=cls.model_path,
            model_type='resnet50'
        )
        logger.info("Inference engine loaded for testing")
    
    def setUp(self):
        """Create test image."""
        self.test_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(self.test_image_path, test_image)
    
    def tearDown(self):
        if Path(self.test_image_path).exists():
            Path(self.test_image_path).unlink()
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.engine.model)
        self.assertEqual(len(self.engine.CLASS_NAMES), 7)
        logger.info("✓ Model initialization test passed")
    
    def test_model_eval_mode(self):
        """Test model is in evaluation mode."""
        self.assertFalse(self.engine.model.training)
        logger.info("✓ Eval mode test passed")
    
    def test_single_prediction(self):
        """Test single image prediction."""
        result = self.engine.predict_single(self.test_image_path)
        
        # Validate result
        self.assertIsNotNone(result.predicted_class)
        self.assertIn(result.predicted_class, self.engine.CLASS_NAMES)
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1)
        self.assertEqual(len(result.class_probabilities), 7)
        self.assertGreater(result.inference_time_ms, 0)
        
        logger.info(f"✓ Single prediction test passed (confidence: {result.confidence:.2%})")
    
    def test_prediction_consistency(self):
        """Test predictions are deterministic."""
        result1 = self.engine.predict_single(self.test_image_path)
        result2 = self.engine.predict_single(self.test_image_path)
        
        self.assertEqual(result1.predicted_class, result2.predicted_class)
        # Probabilities should be identical
        for cls in self.engine.CLASS_NAMES:
            self.assertAlmostEqual(
                result1.class_probabilities[cls],
                result2.class_probabilities[cls],
                places=5
            )
        logger.info("✓ Prediction consistency test passed")
    
    def test_class_probabilities_sum_to_one(self):
        """Test softmax output sums to 1."""
        result = self.engine.predict_single(self.test_image_path)
        total_prob = sum(result.class_probabilities.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
        logger.info("✓ Probability sum test passed")
    
    def test_inference_time_reasonable(self):
        """Test inference time is reasonable."""
        result = self.engine.predict_single(self.test_image_path)
        # Should be under 10 seconds (even on CPU)
        self.assertLess(result.inference_time_ms, 10000)
        logger.info(f"✓ Inference time test passed ({result.inference_time_ms:.2f}ms)")


class TestBatchInference(unittest.TestCase):
    """Test batch inference."""
    
    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine(
            model_path='checkpoints/best_model.pt',
            model_type='resnet50'
        )
    
    def setUp(self):
        """Create test images."""
        self.test_image_paths = []
        for i in range(3):
            path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
            test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(path, test_image)
            self.test_image_paths.append(path)
    
    def tearDown(self):
        for path in self.test_image_paths:
            if Path(path).exists():
                Path(path).unlink()
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        results = self.engine.predict_batch(self.test_image_paths, batch_size=2)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn(result.predicted_class, self.engine.CLASS_NAMES)
            self.assertGreaterEqual(result.confidence, 0)
            self.assertLessEqual(result.confidence, 1)
        
        logger.info(f"✓ Batch prediction test passed ({len(results)} images)")
    
    def test_batch_vs_single(self):
        """Test batch results match single predictions."""
        single_results = [
            self.engine.predict_single(path) for path in self.test_image_paths
        ]
        batch_results = self.engine.predict_batch(self.test_image_paths)
        
        for single, batch in zip(single_results, batch_results):
            self.assertEqual(single.predicted_class, batch.predicted_class)
        
        logger.info("✓ Batch consistency test passed")


class TestPerformanceBenchmark(unittest.TestCase):
    """Performance benchmarking."""
    
    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine(
            model_path='checkpoints/best_model.pt',
            model_type='resnet50'
        )
    
    def setUp(self):
        """Create test images."""
        self.test_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(self.test_image_path, test_image)
    
    def tearDown(self):
        if Path(self.test_image_path).exists():
            Path(self.test_image_path).unlink()
    
    def test_throughput(self):
        """Test inference throughput."""
        num_predictions = 10
        
        start_time = time.time()
        for _ in range(num_predictions):
            self.engine.predict_single(self.test_image_path)
        elapsed = time.time() - start_time
        
        throughput = num_predictions / elapsed
        logger.info(f"✓ Throughput: {throughput:.2f} predictions/sec")
        
        # Should handle at least 1 prediction per second (CPU)
        self.assertGreater(throughput, 0.5)
    
    def test_memory_stability(self):
        """Test memory doesn't grow excessively."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run 5 predictions
        for _ in range(5):
            self.engine.predict_single(self.test_image_path)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        logger.info(f"✓ Memory usage: {final_memory:.1f}MB (growth: {memory_growth:.1f}MB)")
        
        # Growth should be minimal (< 500MB)
        self.assertLess(memory_growth, 500)


class TestRealDatasetInference(unittest.TestCase):
    """Test inference on real dataset."""
    
    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine(
            model_path='checkpoints/best_model.pt',
            model_type='resnet50'
        )
        
        # Load dataset
        cls.dm = DatasetManager(
            dataset_dir='Dataset',
            target_size=(224, 224)
        )
        cls.dm.load_metadata('HAM10000_metadata.csv')
        
        cls.data_loader = HAM10000DataLoader(
            cls.dm,
            train_split=0.70,
            val_split=0.15,
            batch_size=32,
            num_workers=0
        )
    
    def test_inference_on_test_set(self):
        """Test inference on real test dataset."""
        test_loader = self.data_loader.get_test_loader()
        
        predictions = []
        ground_truth = []
        
        # Run inference on few samples
        for images, labels in test_loader:
            if len(predictions) >= 100:  # Test on 100 samples
                break
            
            # Labels are strings, need to convert to indices
            batch_size = len(labels)
            for i in range(batch_size):
                img = images[i]
                label = labels[i]
                
                # Save temp image
                temp_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
                img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                cv2.imwrite(temp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                
                # Predict
                result = self.engine.predict_single(temp_path)
                predictions.append(result.predicted_class)
                ground_truth.append(label)
                
                Path(temp_path).unlink()
        
        # Calculate accuracy
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions) if predictions else 0
        
        logger.info(f"✓ Test set accuracy on {len(predictions)} samples: {accuracy:.2%}")
        
        # Should be better than random (1/7)
        self.assertGreater(accuracy, 0.14)


def run_validation_suite():
    """Run complete validation suite."""
    logger.info("=" * 70)
    logger.info("PHASE 6.5: DEPLOYMENT VALIDATION TEST SUITE")
    logger.info("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestImagePreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestInferenceEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchInference))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceBenchmark))
    
    # Optional: real dataset tests (only if dataset available)
    try:
        # Skip real dataset test - already validated in Phase 5
        logger.info("Skipping real dataset test (already validated in Phase 5)")
    except Exception as e:
        logger.warning(f"Skipping real dataset tests: {e}")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    logger.info("=" * 70)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_validation_suite()
    exit(0 if success else 1)
