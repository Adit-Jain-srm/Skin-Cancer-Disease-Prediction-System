"""
Phase 5: Enhanced Data Augmentation with Class-Balanced Sampling

Implements:
- Albumentations-based augmentation (elastic transforms, CLAHE, etc.)
- Class-balanced batch sampling
- Weighted random sampling for minority classes
- Augmentation strategies: light, medium, strong
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional, Dict, Callable
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AugmentationPipeline:
    """Factory for creating augmentation pipelines."""
    
    @staticmethod
    def light(image_size: int = 224) -> A.Compose:
        """Light augmentation - minimal, always-safe transforms."""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.3),
            A.GaussNoise(p=0.1),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ], bbox_params=None)
    
    @staticmethod
    def medium(image_size: int = 224) -> A.Compose:
        """Medium augmentation - balanced aggressive transforms."""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Transpose(p=0.3),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.ElasticTransform(p=0.1),
            ], p=0.3),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=0.0625,
                rotate=(-45, 45),
                p=0.4
            ),
            A.OneOf([
                A.CLAHE(p=0.3),
                A.Equalize(p=0.1),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.1),
            ], p=0.3),
            A.CoarseDropout(max_holes=8, p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ], bbox_params=None)
    
    @staticmethod
    def strong(image_size: int = 224) -> A.Compose:
        """Strong augmentation - aggressive transforms for limited data."""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.2),
                A.ElasticTransform(p=0.2),
            ], p=0.5),
            A.Affine(
                scale=(0.7, 1.3),
                translate_percent=0.0625,
                rotate=(-45, 45),
                p=0.7
            ),
            A.OneOf([
                A.CLAHE(p=0.3),
                A.Equalize(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.2),
            ], p=0.6),
            A.CoarseDropout(max_holes=16, p=0.4),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ], bbox_params=None)


class BalancedDataLoader:
    """Create class-balanced data loaders with weighted sampling."""
    
    @staticmethod
    def create_balanced_loader(
        dataset: Dataset,
        class_counts: Dict[int, int],
        batch_size: int = 32,
        balance_method: str = 'weighted',
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        """Create balanced DataLoader with weighted sampling."""
        num_samples = len(dataset)
        
        # Get labels from dataset
        if hasattr(dataset, 'labels'):
            labels = dataset.labels
        else:
            labels = [dataset[i][1] for i in range(num_samples)]
        
        labels = np.array(labels)
        
        if balance_method == 'weighted':
            # Compute weights: inverse of class frequency
            class_weights = {}
            for cls_id, count in class_counts.items():
                if count > 0:
                    class_weights[cls_id] = 1.0 / count
                else:
                    class_weights[cls_id] = 1.0
            
            # Normalize weights
            max_weight = max(class_weights.values())
            class_weights = {k: v / max_weight for k, v in class_weights.items()}
            
            # Assign weight to each sample
            weights = np.array([class_weights[int(label)] for label in labels])
            
            logger.info(f"Weighted sampling class weights: {class_weights}")
            logger.info(f"Weight distribution: min={weights.min():.4f}, max={weights.max():.4f}")
            
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=num_samples,
                replacement=True
            )
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=torch.cuda.is_available()
            )
        
        elif balance_method == 'oversampling':
            # Simple oversampling: repeat minority class samples
            unique_classes = np.unique(labels)
            max_count = max(class_counts.values())
            
            indices = []
            for i in range(num_samples):
                label = int(labels[i])
                repeat_count = max_count // max(class_counts[label], 1)
                indices.extend([i] * repeat_count)
            
            logger.info(f"Oversampling: {num_samples} → {len(indices)} samples")
            
            # Shuffle
            if shuffle:
                np.random.shuffle(indices)
            
            loader = DataLoader(
                torch.utils.data.Subset(dataset, indices),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=torch.cuda.is_available()
            )
        
        else:
            raise ValueError(f"Unknown balance method: {balance_method}")
        
        return loader


class AugmentedDataset(Dataset):
    """Wrapper Dataset with augmentation."""
    
    def __init__(
        self,
        dataset: Dataset,
        augmentation: Optional[A.Compose] = None,
        is_train: bool = True
    ):
        """Initialize augmented dataset."""
        self.dataset = dataset
        self.augmentation = augmentation
        self.is_train = is_train
        
        # Get labels from base dataset
        if hasattr(dataset, 'labels'):
            self.labels = dataset.labels
        else:
            self.labels = [dataset[i][1] for i in range(len(dataset))]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Apply augmentation to image and get label."""
        image, label = self.dataset[idx]
        
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Apply augmentation
        if self.augmentation:
            image = self.augmentation(image=image)['image']
        else:
            # Minimal preprocessing
            image = torch.tensor(image, dtype=torch.float32)
        
        return image, label


def create_augmented_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    train_labels: List[int],
    val_labels: List[int],
    test_labels: List[int],
    augmentation_level: str = 'medium',
    batch_size: int = 32,
    balance_train: bool = True,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create augmented data loaders for train/val/test."""
    
    # Get augmentation pipeline
    if augmentation_level == 'light':
        aug_pipeline = AugmentationPipeline.light(image_size)
    elif augmentation_level == 'medium':
        aug_pipeline = AugmentationPipeline.medium(image_size)
    elif augmentation_level == 'strong':
        aug_pipeline = AugmentationPipeline.strong(image_size)
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    # Wrap datasets with augmentation
    train_aug = AugmentedDataset(train_dataset, augmentation=aug_pipeline, is_train=True)
    val_aug = AugmentedDataset(val_dataset, augmentation=None, is_train=False)
    test_aug = AugmentedDataset(test_dataset, augmentation=None, is_train=False)
    
    # Compute class counts
    train_labels_array = np.array(train_labels)
    unique, counts = np.unique(train_labels_array, return_counts=True)
    train_class_counts = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
    
    logger.info(f"Training set class distribution: {train_class_counts}")
    logger.info(f"Augmentation level: {augmentation_level}")
    
    # Create loaders
    if balance_train:
        train_loader = BalancedDataLoader.create_balanced_loader(
            dataset=train_aug,
            class_counts=train_class_counts,
            batch_size=batch_size,
            balance_method='weighted',
            num_workers=num_workers,
            shuffle=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_aug,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available()
        )
    
    val_loader = DataLoader(
        val_aug,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_aug,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """Test augmentation pipelines."""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("ENHANCED DATA AUGMENTATION - VERIFICATION TEST")
    print("=" * 70 + "\n")
    
    # Test augmentation pipelines
    print("Testing augmentation pipelines...")
    
    light_aug = AugmentationPipeline.light(image_size=224)
    medium_aug = AugmentationPipeline.medium(image_size=224)
    strong_aug = AugmentationPipeline.strong(image_size=224)
    
    print("✓ All augmentation pipelines created successfully\n")
    
    # Test dummy data
    dummy_image = np.random.rand(256, 256, 3).astype(np.float32)
    
    print("Testing light augmentation...")
    result = light_aug(image=dummy_image)
    assert 'image' in result and isinstance(result['image'], torch.Tensor)
    print(f"✓ Light augmentation output: {result['image'].shape}\n")
    
    print("Testing medium augmentation...")
    result = medium_aug(image=dummy_image)
    assert 'image' in result and isinstance(result['image'], torch.Tensor)
    print(f"✓ Medium augmentation output: {result['image'].shape}\n")
    
    print("Testing strong augmentation...")
    result = strong_aug(image=dummy_image)
    assert 'image' in result and isinstance(result['image'], torch.Tensor)
    print(f"✓ Strong augmentation output: {result['image'].shape}\n")
    
    # Test class weights
    print("Testing class weight computation...")
    class_counts = {0: 100, 1: 50, 2: 10, 3: 200}
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    max_weight = max(class_weights.values())
    class_weights_norm = {k: v / max_weight for k, v in class_weights.items()}
    
    print("Class counts:", class_counts)
    print("Normalized weights:", {k: f"{v:.4f}" for k, v in class_weights_norm.items()})
    print("✓ Class weight computation successful\n")
    
    print("=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
