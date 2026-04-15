#!/usr/bin/env python3
"""
Test script to verify the fixed model loading and prediction
"""
import sys
from pathlib import Path
from PIL import Image

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_model, predict_image, CLASS_NAMES

def test_model_loading():
    """Test that model loads correctly"""
    print("🔍 Testing model loading...")
    try:
        model = load_model()
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model)}")
        print(f"   Final layer output: {model.fc.out_features} classes")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_class_names():
    """Test that CLASS_NAMES are correct"""
    print("\n🔍 Testing CLASS_NAMES...")
    expected = {
        0: "Actinic Keratoses",
        1: "Basal Cell Carcinoma",
        2: "Benign Keratosis",
        3: "Dermatofibroma",
        4: "Melanoma",
        5: "Melanocytic Nevi",
        6: "Vascular Lesions"
    }
    if CLASS_NAMES == expected:
        print(f"✅ CLASS_NAMES correct (full medical names)")
        for idx, name in CLASS_NAMES.items():
            print(f"   {idx}: {name}")
        return True
    else:
        print(f"❌ CLASS_NAMES incorrect")
        return False

def test_predict_function():
    """Test predict_image function"""
    print("\n🔍 Testing predict_image function...")

    try:
        # Create a dummy image
        dummy_image = Image.new('RGB', (256, 256), color='red')

        # Test prediction
        label, confidence = predict_image(dummy_image)

        print(f"✅ Prediction function works!")
        print(f"   Return type: {type((label, confidence))}")
        print(f"   Predicted label: {label}")
        print(f"   Confidence: {confidence:.2f}%")

        # Validate return types
        if not isinstance(label, str):
            print("❌ Label should be string")
            return False
        if not isinstance(confidence, (int, float)):
            print("❌ Confidence should be number")
            return False
        if not (0 <= confidence <= 100):
            print("❌ Confidence should be between 0-100")
            return False

        return True
    except Exception as e:
        print(f"❌ Error in predict_image: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SKIN DISEASE DETECTION - PRODUCTION FIX TEST")
    print("=" * 60)

    model_ok = test_model_loading()
    class_ok = test_class_names()
    predict_ok = test_predict_function()

    print("\n" + "=" * 60)
    if model_ok and class_ok and predict_ok:
        print("✅ ALL TESTS PASSED!")
        print("Ready for production deployment")
        print("\nTo run the app:")
        print("  streamlit run app.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above")
    print("=" * 60)