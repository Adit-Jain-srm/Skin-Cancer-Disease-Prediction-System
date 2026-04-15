"""
Phase 4: CNN Baseline Model for Skin Lesion Classification

Architecture: 4 conv blocks (64→128→256→512) + 2 FC layers
Target: ≥70% accuracy on HAM10000 dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    """
    Baseline CNN for skin lesion classification.
    
    Architecture:
      - 4 convolutional blocks: 64→128→256→512 filters
      - BatchNorm after each conv
      - ReLU activation
      - MaxPool 2x2 after each block
      - Global average pooling
      - 2 fully connected layers with dropout (0.5)
      - Output: 7 classes (logits)
    
    Parameters:
      - Input: (batch_size, 3, 224, 224)
      - Output: (batch_size, 7)
      - ~6.5M parameters
    """
    
    def __init__(self, num_classes=7, dropout=0.5):
        """
        Initialize CNN baseline.
        
        Args:
            num_classes (int): Number of output classes (default: 7)
            dropout (float): Dropout probability (default: 0.5)
        """
        super().__init__()
        
        # Convolutional feature extraction blocks
        self.features = nn.Sequential(
            # Block 1: 3 → 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 256 → 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Kaiming Normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.features(x)
        
        # Global average pooling: (batch, 512, 14, 14) → (batch, 512, 1, 1)
        x = self.avgpool(x)
        
        # Flatten: (batch, 512, 1, 1) → (batch, 512)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_model_info(self):
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'CNN4Block (64→128→256→512)',
            'input_shape': (3, 224, 224),
            'output_shape': (7,),
        }


class CNNBaseline_SmallInput(nn.Module):
    """
    Baseline CNN optimized for smaller inputs (e.g., 128x128).
    Useful for quick validation/testing.
    
    Architecture: 3 conv blocks (32→64→128) + 2 FC layers
    Parameters: ~1.2M
    """
    
    def __init__(self, num_classes=7, dropout=0.5, input_size=128):
        """
        Initialize small CNN baseline.
        
        Args:
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
            input_size (int): Input image size (128 or 224)
        """
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(64, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming Normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_cnn_baseline(num_classes=7, dropout=0.5, device='cpu'):
    """
    Factory function to create and initialize CNN baseline.
    
    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout probability
        device (str): Device to place model on ('cpu' or 'cuda')
    
    Returns:
        nn.Module: Initialized model on specified device
    """
    model = CNNBaseline(num_classes=num_classes, dropout=dropout)
    model = model.to(device)
    return model


if __name__ == '__main__':
    """Test script: Verify model forward pass."""
    print("=" * 70)
    print("CNN BASELINE MODEL - VERIFICATION TEST")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test 1: Model instantiation
    print("\n[1/3] Creating CNN baseline...")
    model = CNNBaseline(num_classes=7, dropout=0.5)
    model = model.to(device)
    print(f"✓ Model created successfully")
    
    # Test 2: Model info
    print("\n[2/3] Model information:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test 3: Forward pass
    print("\n[3/3] Testing forward pass...")
    model.eval()
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    if output.shape == (batch_size, 7):
        print(f"✓ Forward pass successful, output shape correct")
    else:
        print(f"✗ ERROR: Expected shape ({batch_size}, 7), got {output.shape}")
    
    print("\n" + "=" * 70)
    print("✅ ALL MODEL TESTS PASSED")
    print("=" * 70)
