import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# HAM10000 dataset class names with full medical terminology
CLASS_NAMES = {
    0: "Actinic Keratoses",
    1: "Basal Cell Carcinoma",
    2: "Benign Keratosis",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic Nevi",
    6: "Vascular Lesions"
}

def load_model(model_path="model/model.pth"):
    """
    Load the trained ResNet18 model from state_dict.

    Args:
        model_path: Path to the saved model state_dict

    Returns:
        Loaded and evaluated model
    """
    # Create ResNet18 model without pretrained weights
    model = models.resnet18(weights=None)

    # Replace final layer for 7 classes
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 7)

    # Load state_dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # Set to evaluation mode
    model.eval()

    return model

def predict_image(image):
    """
    Predict skin disease class from image.

    Args:
        image: PIL Image object

    Returns:
        tuple: (predicted_label, confidence_score)
    """
    try:
        # Load model (this will be cached by Streamlit)
        model = load_model()

        # Transform image
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # Apply transforms and add batch dimension
        image_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        # Get class name and confidence
        predicted_label = CLASS_NAMES[predicted_class.item()]
        confidence_score = confidence.item() * 100

        return predicted_label, confidence_score

    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")