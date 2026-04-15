"""
Phase 5: Transfer Learning Models

Implements ResNet50 and EfficientNet-B3 with fine-tuning strategies
for HAM10000 skin lesion classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TransferLearningModel:
    """Factory class for creating transfer learning models."""
    
    @staticmethod
    def create_resnet50(
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ) -> nn.Module:
        """
        Create ResNet50 with transfer learning.
        
        Args:
            num_classes: Number of output classes (7 for HAM10000)
            pretrained: Use ImageNet pre-trained weights
            freeze_backbone: Freeze all layers except final FC layer initially
        
        Returns:
            ResNet50 model with modified final layer
        """
        logger.info(f"Creating ResNet50 (pretrained={pretrained}, freeze={freeze_backbone})")
        
        model = models.resnet50(pretrained=pretrained)
        
        # Replace final FC layer
        in_features = model.fc.in_features if hasattr(model.fc, 'in_features') else 2048
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(int(in_features), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in model.layer1.parameters():
                param.requires_grad = False
            for param in model.layer2.parameters():
                param.requires_grad = False
            logger.info("ResNet50: Froze layers 1-2, unfroze layer3-4 + FC")
        
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"ResNet50: {trainable_params:,} / {total_params:,} parameters trainable")
        
        return model
    
    @staticmethod
    def create_efficientnet_b3(
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ) -> nn.Module:
        """
        Create EfficientNet-B3 with transfer learning.
        
        Args:
            num_classes: Number of output classes (7 for HAM10000)
            pretrained: Use ImageNet pre-trained weights
            freeze_backbone: Freeze backbone initially for fine-tuning
        
        Returns:
            EfficientNet-B3 model with modified classification head
        """
        logger.info(f"Creating EfficientNet-B3 (pretrained={pretrained}, freeze={freeze_backbone})")
        
        model = models.efficientnet_b3(pretrained=pretrained)
        
        # Replace final classifier
        in_features = model.classifier[1].in_features if hasattr(model.classifier[1], 'in_features') else 1536
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(int(in_features), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in model.features[:6].parameters():
                param.requires_grad = False
            logger.info("EfficientNet-B3: Froze first 6 blocks, unfroze last blocks + classifier")
        
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"EfficientNet-B3: {trainable_params:,} / {total_params:,} parameters trainable")
        
        return model
    
    @staticmethod
    def unfreeze_backbone(model: nn.Module, model_name: str = 'resnet50') -> None:
        """
        Unfreeze entire backbone for fine-tuning in later epochs.
        
        Args:
            model: The transfer learning model
            model_name: 'resnet50' or 'efficientnet_b3'
        """
        if model_name == 'resnet50':
            for param in model.layer1.parameters():
                param.requires_grad = True
            for param in model.layer2.parameters():
                param.requires_grad = True
            logger.info("ResNet50: Unfroze all backbone layers for fine-tuning")
        
        elif model_name == 'efficientnet_b3':
            for param in model.features.parameters():
                param.requires_grad = True
            logger.info("EfficientNet-B3: Unfroze all backbone layers for fine-tuning")
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Tuple[int, int]:
        """
        Count total and trainable parameters.
        
        Args:
            model: PyTorch model
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total


if __name__ == '__main__':
    """Test transfer learning models."""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("TRANSFER LEARNING MODELS - VERIFICATION TEST")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")
    
    # Test ResNet50
    print("Testing ResNet50...")
    resnet = TransferLearningModel.create_resnet50(num_classes=7, pretrained=True, freeze_backbone=True)
    resnet = resnet.to(device)
    
    x = torch.randn(4, 3, 224, 224).to(device)
    y = resnet(x)
    assert y.shape == (4, 7), f"Expected (4, 7), got {y.shape}"
    print(f"✓ ResNet50 forward pass successful: {y.shape}\n")
    
    # Test EfficientNet-B3
    print("Testing EfficientNet-B3...")
    efficientnet = TransferLearningModel.create_efficientnet_b3(num_classes=7, pretrained=True, freeze_backbone=True)
    efficientnet = efficientnet.to(device)
    
    y = efficientnet(x)
    assert y.shape == (4, 7), f"Expected (4, 7), got {y.shape}"
    print(f"✓ EfficientNet-B3 forward pass successful: {y.shape}\n")
    
    # Test parameter counts
    print("=" * 70)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 70)
    
    resnet_train, resnet_total = TransferLearningModel.count_parameters(resnet)
    eff_train, eff_total = TransferLearningModel.count_parameters(efficientnet)
    
    print(f"\nResNet50:")
    print(f"  Trainable: {resnet_train:,}")
    print(f"  Total: {resnet_total:,}")
    print(f"  Frozen: {resnet_total - resnet_train:,}\n")
    
    print(f"EfficientNet-B3:")
    print(f"  Trainable: {eff_train:,}")
    print(f"  Total: {eff_total:,}")
    print(f"  Frozen: {eff_total - eff_train:,}\n")
    
    # Test unfreezing
    print("Testing backbone unfreezing...")
    TransferLearningModel.unfreeze_backbone(resnet, 'resnet50')
    resnet_train_unfrozen, _ = TransferLearningModel.count_parameters(resnet)
    assert resnet_train_unfrozen > resnet_train, "Unfrozing should increase trainable params"
    print(f"✓ ResNet50 unfrozen: {resnet_train} → {resnet_train_unfrozen} trainable params\n")
    
    print("=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
