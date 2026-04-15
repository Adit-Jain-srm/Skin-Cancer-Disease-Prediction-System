"""
Load Testing for Skin Cancer Detection API
Uses Locust for performance and stress testing

Installation:
    pip install locust

Usage:
    # Web UI (recommended)
    locust -f load_test.py --host=http://localhost:5000
    
    # Headless mode
    locust -f load_test.py --host=http://localhost:5000 \
        --users=100 --spawn-rate=10 --run-time=5m --headless
    
    # Custom parameters
    locust -f load_test.py --host=http://localhost:5000 \
        --users=50 --spawn-rate=5 --run-time=10m \
        --csv=results --headless

Note: You need sample images in the 'sample_images/' directory for testing
"""

import random
import time
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

from locust import HttpUser, task, between, events
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Generate sample test images
def generate_test_image(size=(224, 224)):
    """
    Generate a random test image for load testing.
    
    Args:
        size: Image dimensions (default: 224x224)
    
    Returns:
        Image bytes (PNG format)
    """
    # Create random RGB image
    img_array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    
    # Save to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


class SkinCancerAPIUser(HttpUser):
    """
    Load test user that simulates API usage patterns.
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts."""
        self.test_images = [generate_test_image() for _ in range(5)]
        logger.debug(f"User started with {len(self.test_images)} test images")
    
    @task(1)
    def health_check(self):
        """Health check endpoint (frequency: 1)."""
        with self.client.get("/api/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def model_info(self):
        """Get model information (frequency: 1)."""
        with self.client.get("/api/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(5)
    def predict_single_image(self):
        """
        Predict on single image (frequency: 5 - most common operation).
        """
        image_bytes = random.choice(self.test_images)
        
        with self.client.post(
            "/api/predict",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'confidence' in data and 'predicted_id' in data:
                        response.success()
                    else:
                        response.failure("Missing required fields in response")
                except ValueError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def predict_batch(self):
        """
        Predict batch of images (frequency: 2).
        """
        files = [
            ("images", ("test1.jpg", random.choice(self.test_images), "image/jpeg")),
            ("images", ("test2.jpg", random.choice(self.test_images), "image/jpeg")),
            ("images", ("test3.jpg", random.choice(self.test_images), "image/jpeg")),
        ]
        
        with self.client.post(
            "/api/predict-batch",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'predictions' in data:
                        response.success()
                    else:
                        response.failure("Missing 'predictions' in response")
                except ValueError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def predict_from_bytes(self):
        """
        Predict from bytes (frequency: 1).
        """
        image_bytes = random.choice(self.test_images)
        
        with self.client.post(
            "/api/predict-from-bytes",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'confidence' in data:
                        response.success()
                    else:
                        response.failure("Missing 'confidence' in response")
                except ValueError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")


# Event handlers for test statistics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called at test start."""
    logger.info("=" * 70)
    logger.info("LOAD TEST STARTED")
    logger.info(f"Target: {environment.host}")
    logger.info("=" * 70)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    logger.info("=" * 70)
    logger.info("LOAD TEST COMPLETED")
    logger.info("=" * 70)
    
    # Print statistics
    stats = environment.stats
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Success Rate: {100 * (1 - stats.total.failure_rate):.2f}%")
    print(f"Median Response Time: {stats.total.get_response_time_percentile(0.5):.0f}ms")
    print(f"95th Percentile: {stats.total.get_response_time_percentile(0.95):.0f}ms")
    print(f"99th Percentile: {stats.total.get_response_time_percentile(0.99):.0f}ms")
    print(f"Min Response Time: {stats.total.min_response_time:.0f}ms")
    print(f"Max Response Time: {stats.total.max_response_time:.0f}ms")
    print(f"Mean Response Time: {stats.total.avg_response_time:.0f}ms")
    print(f"Requests/Second: {stats.total.total_rps:.2f}")
    print("=" * 70)


# Test scenarios (can be run with different configurations)
"""
EXAMPLE TEST SCENARIOS:

1. Light Load Test (Development)
   Users: 10
   Spawn Rate: 2/sec
   Duration: 2 minutes
   Expected: Identify basic performance issues
   
   locust -f load_test.py --host=http://localhost:5000 \
       --users=10 --spawn-rate=2 --run-time=2m --headless

2. Moderate Load Test (Staging)
   Users: 50
   Spawn Rate: 5/sec
   Duration: 5 minutes
   Expected: Identify bottlenecks at moderate load
   
   locust -f load_test.py --host=http://localhost:5000 \
       --users=50 --spawn-rate=5 --run-time=5m --headless

3. Heavy Load Test (Production Simulation)
   Users: 100
   Spawn Rate: 10/sec
   Duration: 10 minutes
   Expected: Test system limits and auto-healing
   
   locust -f load_test.py --host=http://localhost:5000 \
       --users=100 --spawn-rate=10 --run-time=10m --headless

4. Stress Test (Breaking Point)
   Users: 500
   Spawn Rate: 50/sec
   Duration: 5 minutes
   Expected: Find system breaking point
   
   locust -f load_test.py --host=http://localhost:5000 \
       --users=500 --spawn-rate=50 --run-time=5m --headless

5. Spike Test (Traffic Spike)
   Phase 1: 10 users for 1 minute
   Phase 2: 500 users suddenly added
   Phase 3: 2 minutes sustained
   Expected: Test recovery from sudden traffic spike
   
   # Requires custom test plan (see advanced scenarios)

PERFORMANCE TARGETS (Based on Phase 6 benchmarks):
- Throughput: 20+ predictions/second
- Median Latency: <60ms
- 95th Percentile: <100ms
- Success Rate: 99%+
- Memory: <1GB (all users combined)

FAILURE ANALYSIS:
- Connection refused: API not running
- Timeout errors: API overload or slow processing
- 5xx errors: Application errors
- Success rate <99%: Stability issues
"""
