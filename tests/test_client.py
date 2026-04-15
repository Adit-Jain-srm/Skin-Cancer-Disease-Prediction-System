"""Test script for Python client integration testing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

from skin_cancer_client import SkinCancerAPIClient
from PIL import Image
import numpy as np
import time

print("="*70)
print("PYTHON CLIENT TEST - SKIN CANCER API")
print("="*70)
print()

# Initialize client
print("1. Initializing client...")
client = SkinCancerAPIClient(api_url="http://localhost:5000")
print("✅ Client connected")
print()

# Test 1: Health check
print("2. Testing /api/health endpoint...")
try:
    response = client.health_check()
    print(f"✅ API is healthy")
    print(f"   Status: {response.get('status')}")
    print(f"   Timestamp: {response.get('timestamp')}")
except Exception as e:
    print(f"❌ Health check failed: {e}")
    sys.exit(1)
print()

# Test 2: Model info
print("3. Testing /api/info endpoint...")
try:
    info = client.get_model_info()
    print(f"✅ Model info retrieved")
    print(f"   Model type: {info.get('model_type')}")
    print(f"   Device: {info.get('device')}")
    print(f"   Classes: {len(info.get('classes', []))} lesion types")
    print(f"   Target size: {info.get('target_size')}")
except Exception as e:
    print(f"❌ Model info failed: {e}")
    sys.exit(1)
print()

# Test 3: Create test image and predict
print("4. Creating synthetic test image...")
# Create a 224x224 random image (matching model's target size)
img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array, 'RGB')
img_path = Path(f"temp_test_image_{int(time.time()*1000)}.jpg")
img.save(img_path)
print(f"✅ Test image created")
print()

# Test 4: Single prediction
print("5. Testing single image prediction...")
try:
    start = time.time()
    result = client.predict_image(str(img_path))
    elapsed = time.time() - start
    
    print(f"✅ Prediction successful ({elapsed:.3f}s)")
    print(f"   Response: {result}")
    print(f"   Top class: {result.get('top_class')}")
    confidence = result.get('confidence')
    if confidence is not None:
        print(f"   Confidence: {confidence:.2%}")
    else:
        print(f"   Confidence: {confidence}")
    print(f"   All predictions:")
    for cls, conf in result.get('all_predictions', {}).items():
        print(f"      {cls}: {conf:.2%}")
except Exception as e:
    print(f"❌ Single prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 5: Batch prediction
print("6. Testing batch prediction (5 images)...")
try:
    # Create 5 test images
    batch_images = []
    for i in range(5):
        img_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        batch_images.append(Image.fromarray(img_arr, 'RGB'))
    
    # For batch, we'll use predict_image in a loop
    start = time.time()
    batch_results = []
    for i, img in enumerate(batch_images):
        tmp_path = Path(f"temp_batch_img_{int(time.time()*1000)}_{i}.jpg")
        img.save(tmp_path)
        result = client.predict_image(str(tmp_path))
        batch_results.append(result)
    
    elapsed = time.time() - start
    
    print(f"✅ Batch prediction successful ({elapsed:.3f}s)")
    print(f"   Processed: 5 images")
    print(f"   Average time per image: {elapsed/5:.3f}s")
    throughput = 5 / elapsed
    print(f"   Throughput: {throughput:.2f} predictions/sec")
except Exception as e:
    print(f"❌ Batch prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("="*70)
print("✅ ALL CLIENT TESTS PASSED")
print("="*70)
