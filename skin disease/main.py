import torch
import torch.nn.functional as F
from torchvision import models

from utils import preprocess_image


class SkinDiseaseDetector:

    def __init__(self):

        self.model = models.resnet18()

        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 7)

        self.model.load_state_dict(
            torch.load("model/skin_model.pth", map_location="cpu")
        )

        self.model.eval()

        self.classes = [
            "Actinic keratoses",
            "Basal cell carcinoma",
            "Benign keratosis",
            "Dermatofibroma",
            "Melanoma",
            "Melanocytic nevus",
            "Vascular lesion"
        ]


    def predict(self, image):

        img = preprocess_image(image)

        outputs = self.model(img)

        probs = F.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)

        return {
            "disease_name": self.classes[pred],
            "confidence": float(confidence * 100)
        }