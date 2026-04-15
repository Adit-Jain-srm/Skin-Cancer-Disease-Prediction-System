"""
Real-World Load Testing for Skin Cancer Detection API
Uses actual HAM10000 dataset images for performance testing

Installation:
    pip install locust Pillow numpy

Usage:
    # Web UI (recommended)
    locust -f load_test_realworld.py --host=http://localhost:5000
    
    # Headless mode
    locust -f load_test_realworld.py --host=http://localhost:5000 \
        --users=50 --spawn-rate=10 --run-time=10m --headless

Features:
    - Uses actual skin cancer images from HAM10000 dataset
    - Realistic distribution of image types
    - Response time tracking for production insights
    - Success/failure rate monitoring
    - Throughput benchmarking
"""

import random
import time
from pathlib import Path
from PIL import Image
import numpy as np
from locust import HttpUser, task, between, events
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Load actual images from HAM10000 dataset
class ImageLoader:
    """Load real images from HAM10000 dataset."""
    
    def __init__(self):
        self.images = []
        self.image_paths = []
        self._load_images()
    
    def _load_images(self):
        """Load images from dataset directories."""
        dataset_path = Path('Dataset')
        
        # Check if dataset exists
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path.absolute()}")
            return
        
        # Load from part 1
        part1 = dataset_path / 'HAM10000_images_part_1'
        if part1.exists():
            for img_file in list(part1.glob('*.jpg'))[:100]:  # Load first 100 from part 1
                try:
                    img = Image.open(img_file)
                    # Resize to match model input (224x224)
                    img = img.resize((224, 224))
                    # Convert to bytes
                    img_bytes = img.tobytes()
                    self.images.append(img)
                    self.image_paths.append(str(img_file))
                except Exception as e:
                    logger.warning(f"Failed to load {img_file}: {e}")
        
        # Load from part 2
        part2 = dataset_path / 'HAM10000_images_part_2'
        if part2.exists():
            for img_file in list(part2.glob('*.jpg'))[:100]:  # Load first 100 from part 2
                try:
                    img = Image.open(img_file)
                    img = img.resize((224, 224))
                    img_bytes = img.tobytes()
                    self.images.append(img)
                    self.image_paths.append(str(img_file))
                except Exception as e:
                    logger.warning(f"Failed to load {img_file}: {e}")
        
        logger.info(f"Loaded {len(self.images)} real images from HAM10000 dataset")
    
    def get_random_image(self):
        """Get a random image for testing."""
        if not self.images:
            # Fallback: generate synthetic image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            return Image.fromarray(img_array, 'RGB')
        return random.choice(self.images)
    
    def get_random_image_bytes(self):
        """Get random image as bytes."""
        img = self.get_random_image()
        from io import BytesIO
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.getvalue()


# Global image loader (shared across users)
_image_loader = ImageLoader()


class SkinCancerRealWorldUser(HttpUser):
    """
    Real-world load test user with actual dataset images.
    Simulates realistic usage patterns with real skin lesion images.
    """
    
    wait_time = between(1, 5)  # 1-5 seconds between requests
    
    def on_start(self):
        """Called when a user starts."""
        # Each user gets their own image cache
        self.image_cache = []
        for _ in range(10):
            self.image_cache.append(_image_loader.get_random_image_bytes())
    
    @task(1)
    def health_check(self):
        """Health check (frequency: 1 - every ~7 requests)."""
        with self.client.get("/api/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(8)
    def predict_single_image(self):
        """Single image prediction (frequency: 8 - most common)."""
        image_bytes = random.choice(self.image_cache)
        
        with self.client.post(
            "/api/predict",
            files={"image": ("lesion.jpg", image_bytes, "image/jpeg")},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Check if prediction is valid
                    if data.get('success') and 'prediction' in data:
                        response.success()
                    else:
                        response.failure("Invalid response structure")
                except Exception as e:
                    response.failure(f"JSON parse error: {e}")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def predict_batch(self):
        """Batch prediction with 3 images (frequency: 2)."""
        batch_files = [
            ("images", ("lesion1.jpg", random.choice(self.image_cache), "image/jpeg")),
            ("images", ("lesion2.jpg", random.choice(self.image_cache), "image/jpeg")),
            ("images", ("lesion3.jpg", random.choice(self.image_cache), "image/jpeg")),
        ]
        
        with self.client.post(
            "/api/predict-batch",
            files=batch_files,
            catch_response=True
        ) as response:
            if response.status_code in [200, 207]:  # 200 or 207 (multi-status)
                response.success()
            else:
                response.failure(f"Batch failed: {response.status_code}")
    
    @task(1)
    def get_model_info(self):
        """Get model information (frequency: 1)."""
        with self.client.get("/api/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model info failed: {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Test started."""
    logger.info("=" * 70)
    logger.info("REAL-WORLD LOAD TEST STARTED")
    logger.info(f"Target: {environment.host}")
    logger.info(f"Test Images: {len(_image_loader.images)} real HAM10000 images loaded")
    logger.info("=" * 70)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Test completed - print summary."""
    logger.info("=" * 70)
    logger.info("LOAD TEST COMPLETED")
    logger.info("=" * 70)
    
    stats = environment.stats
    total_reqs = stats.total.num_requests
    total_fails = stats.total.num_failures
    
    print("\n" + "=" * 70)
    print("REAL-WORLD LOAD TEST SUMMARY")
    print("=" * 70)
    print(f"Total Requests: {total_reqs}")
    print(f"Successful: {total_reqs - total_fails}")
    print(f"Failed: {total_fails}")
    if total_reqs > 0:
        print(f"Success Rate: {100 * (1 - total_fails/total_reqs):.1f}%")
    print()
    print("Latency Metrics:")
    print(f"  Min: {stats.total.min_response_time:.0f}ms")
    print(f"  Max: {stats.total.max_response_time:.0f}ms")
    print(f"  Mean: {stats.total.avg_response_time:.0f}ms")
    print(f"  Median: {stats.total.get_response_time_percentile(0.5):.0f}ms")
    print(f"  95th: {stats.total.get_response_time_percentile(0.95):.0f}ms")
    print(f"  99th: {stats.total.get_response_time_percentile(0.99):.0f}ms")
    print()
    print("Throughput:")
    print(f"  Requests/second: {stats.total.total_rps:.2f}")
    print(f"  Predictions/min: {stats.total.total_rps * 60:.0f}")
    print("=" * 70)
    print()
    
    # Per-endpoint breakdown
    print("Per-Endpoint Results:")
    for name, stats_entry in stats.entries.items():
        endpoint, method = name
        print(f"\n  {method} {endpoint}")
        print(f"    Requests: {stats_entry.num_requests}")
        print(f"    Failures: {stats_entry.num_failures}")
        print(f"    Avg Latency: {stats_entry.avg_response_time:.0f}ms")
        print(f"    Min/Max: {stats_entry.min_response_time:.0f}/{stats_entry.max_response_time:.0f}ms")
    print()
    print("=" * 70)


# Test scenarios
"""
SUGGESTED TEST CONFIGURATIONS:

1. Light Load (Development)
   locust -f load_test_realworld.py --host=http://localhost:5000 \
       --users=5 --spawn-rate=2 --run-time=3m --headless

2. Realistic Load (Typical Usage)
   locust -f load_test_realworld.py --host=http://localhost:5000 \
       --users=20 --spawn-rate=5 --run-time=10m --headless

3. Heavy Load (Peak Capacity)
   locust -f load_test_realworld.py --host=http://localhost:5000 \
       --users=50 --spawn-rate=10 --run-time=10m --headless

4. Stress Test (Find Limits)
   locust -f load_test_realworld.py --host=http://localhost:5000 \
       --users=200 --spawn-rate=20 --run-time=5m --headless

Performance Expectations (with GPU):
- Success Rate: >98%
- Mean Latency: 80-120ms per image
- P95 Latency: <150ms
- Throughput: >10 predictions/second

With CPU (current):
- Success Rate: >95%
- Mean Latency: 2000-2500ms per image
- P95 Latency: <2200ms
- Throughput: 0.4-0.5 predictions/second
"""
