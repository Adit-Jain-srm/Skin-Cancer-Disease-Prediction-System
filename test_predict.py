#!/usr/bin/env python3
"""
Test script to verify model loading and prediction function
"""
import sys
from pathlib import Path
from PIL import Image
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import predict_image

def test_model_loading():
    """Test that model loads correctly"""
    print("🔍 Testing model loading...")
    model_path = "model/model.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found at {model_path}")
        return False
    
    try:
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_predict_function():
    """Test predict_image function"""
    print("\n🔍 Testing predict_image function...")
    
    try:
        # Create a dummy image
        dummy_image = Image.new('RGB', (256, 256), color='red')
        
        # Test prediction
        result = predict_image(
            dummy_image,
            model_path="model/model.pth",
            device=torch.device("cpu")
        )
        
        print(f"✅ Prediction function works!")
        print(f"   Result type: {type(result)}")
        print(f"   Result keys: {result.keys()}")
        print(f"   Predicted label: {result['label']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        
        # Validate result structure
        if 'label' not in result or 'confidence' not in result:
            print("❌ Result missing required keys")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error in predict_image: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SKIN DISEASE DETECTION - TEST SUITE")
    print("=" * 50)
    
    model_ok = test_model_loading()
    predict_ok = test_predict_function()
    
    print("\n" + "=" * 50)
    if model_ok and predict_ok:
        print("✅ ALL TESTS PASSED!")
        print("Ready to run: streamlit run app.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above")
    print("=" * 50)
