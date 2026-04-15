"""
Data Loader Module
Provides PyTorch DataLoader for HAM10000 dataset with stratified splitting
and augmentation pipeline.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class HAM10000Dataset(Dataset):
    """Custom PyTorch Dataset for HAM10000 images."""
    
    def __init__(
        self,
        metadata: pd.DataFrame,
        dataset_manager,
        augment: bool = False,
        class_to_id: Optional[dict] = None
    ):
        """
        Initialize dataset.
        
        Args:
            metadata: DataFrame with image_id, dx (class label)
            dataset_manager: DatasetManager instance for preprocessing
            augment: Whether to apply augmentation (True for training)
            class_to_id: Mapping of disease class names to integer IDs
        """
        self.metadata = metadata.reset_index(drop=True)
        self.dm = dataset_manager
        self.augment = augment
        
        # Class mapping (ensure consistent encoding)
        if class_to_id is None:
            unique_classes = sorted(self.metadata['dx'].unique())
            self.class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
        else:
            self.class_to_id = class_to_id
        
        logger.info(f"Dataset: {len(self.metadata)} samples, augment={augment}")
        logger.info(f"Class mapping: {self.class_to_id}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get single image and label.
        
        Returns:
            (image_tensor, label_id) where image has shape (3, 224, 224)
        """
        row = self.metadata.iloc[idx]
        image_id = row['image_id']
        label_str = row['dx']
        
        # Find image path (try part_1, fall back to part_2)
        img_path_1 = self.dm.dataset_dir / "HAM10000_images_part_1" / f"{image_id}.jpg"
        img_path_2 = self.dm.dataset_dir / "HAM10000_images_part_2" / f"{image_id}.jpg"
        
        if img_path_1.exists():
            img_path = str(img_path_1)
        elif img_path_2.exists():
            img_path = str(img_path_2)
        else:
            raise FileNotFoundError(f"Image not found: {image_id}")
        
        # Preprocess image
        try:
            img = self.dm.preprocess_image(img_path)
        except Exception as e:
            logger.error(f"Failed to load {image_id}: {e}")
            raise
        
        # Apply augmentation if training
        if self.augment:
            img = self.dm.augment_image(img, augment=True)
        
        # Convert to tensor: HWC → CHW
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, 224, 224)
        
        # Encode label as integer
        label_id = self.class_to_id[label_str]
        
        return img_tensor, label_id


class HAM10000DataLoader:
    """PyTorch DataLoader wrapper for HAM10000 dataset."""
    
    def __init__(
        self,
        dataset_manager,
        train_split: float = 0.7,
        val_split: float = 0.15,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        random_state: int = 42
    ):
        """
        Initialize data loader with stratified splitting.
        
        Args:
            dataset_manager: Initialized DatasetManager instance
            train_split: Fraction for training (0.7 = 70%)
            val_split: Fraction for validation (0.15 = 15%)
            batch_size: Batch size for DataLoader
            shuffle: Shuffle training data
            num_workers: Number of workers for parallel loading
            random_state: Random seed for reproducibility
        """
        self.dm = dataset_manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.random_state = random_state
        # GPU support: only pin memory if CUDA is available
        self.pin_memory = torch.cuda.is_available()
        
        # Validate splits
        test_split = 1.0 - train_split - val_split
        assert test_split > 0, "test_split must be positive"
        assert train_split + val_split + test_split == 1.0
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Load metadata
        logger.info("Loading metadata for stratified split...")
        self.metadata = dataset_manager.load_metadata('HAM10000_metadata.csv')
        
        # Get unique classes for stratification
        self.unique_classes = sorted(self.metadata['dx'].unique())
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.unique_classes)}
        self.id_to_class = {idx: cls for cls, idx in self.class_to_id.items()}
        
        logger.info(f"Classes: {self.unique_classes}")
        logger.info(f"Class mapping: {self.class_to_id}")
        
        # Perform stratified split at LESION level (critical!)
        logger.info("Performing stratified split at lesion level...")
        self._perform_stratified_split()
        
        # Compute class weights
        self._compute_class_weights()
    
    def _perform_stratified_split(self) -> None:
        """
        Stratified split at LESION level to prevent data leakage.
        
        Important: We split by unique lesions (not images), so all images
        from the same lesion stay in the same split.
        """
        # Group by lesion_id to get unique lesions
        unique_lesions = self.metadata.groupby('lesion_id').first().reset_index()
        
        logger.info(f"Unique lesions: {len(unique_lesions)}")
        logger.info(f"Samples per lesion: min={unique_lesions.groupby('lesion_id').size().min()}, "
                   f"max={unique_lesions.groupby('lesion_id').size().max()}")
        
        # Stratified split: lesion level
        indices = np.arange(len(unique_lesions))
        
        # Split 1: train+val vs test (stratified by class)
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_split,
            random_state=self.random_state,
            stratify=unique_lesions['dx']
        )
        
        # Split 2: train vs val (stratified by class)
        adjusted_val_size = self.val_split / (self.train_split + self.val_split)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=adjusted_val_size,
            random_state=self.random_state,
            stratify=unique_lesions.iloc[train_val_indices]['dx']
        )
        
        # Get lesion IDs for each split
        train_lesions = unique_lesions.iloc[train_indices]['lesion_id'].values
        val_lesions = unique_lesions.iloc[val_indices]['lesion_id'].values
        test_lesions = unique_lesions.iloc[test_indices]['lesion_id'].values
        
        # Map back to all images (multiple images per lesion)
        self.train_metadata = self.metadata[self.metadata['lesion_id'].isin(train_lesions)].reset_index(drop=True)
        self.val_metadata = self.metadata[self.metadata['lesion_id'].isin(val_lesions)].reset_index(drop=True)
        self.test_metadata = self.metadata[self.metadata['lesion_id'].isin(test_lesions)].reset_index(drop=True)
        
        logger.info(f"\n✓ Stratified split complete:")
        logger.info(f"  Train: {len(self.train_metadata)} images ({len(train_lesions)} lesions)")
        logger.info(f"  Val:   {len(self.val_metadata)} images ({len(val_lesions)} lesions)")
        logger.info(f"  Test:  {len(self.test_metadata)} images ({len(test_lesions)} lesions)")
        
        # Verify no data leakage (no overlap in lesions)
        train_set = set(self.train_metadata['lesion_id'])
        val_set = set(self.val_metadata['lesion_id'])
        test_set = set(self.test_metadata['lesion_id'])
        
        assert len(train_set & val_set) == 0, "Data leakage: train/val overlap"
        assert len(train_set & test_set) == 0, "Data leakage: train/test overlap"
        assert len(val_set & test_set) == 0, "Data leakage: val/test overlap"
        
        logger.info("✓ No data leakage detected")
        
        # Log class distribution per split
        logger.info("\nClass distribution:")
        for split_name, split_data in [
            ('Train', self.train_metadata),
            ('Val', self.val_metadata),
            ('Test', self.test_metadata)
        ]:
            dist = split_data['dx'].value_counts()
            logger.info(f"\n{split_name}:")
            for cls in self.unique_classes:
                count = dist.get(cls, 0)
                pct = (count / len(split_data)) * 100
                logger.info(f"  {cls:10s}: {count:5d} ({pct:5.1f}%)")
    
    def _compute_class_weights(self) -> None:
        """Compute class weights for loss function (inverse frequency)."""
        class_counts = self.train_metadata['dx'].value_counts()
        
        # Inverse frequency weighting
        weights = {}
        for cls in self.unique_classes:
            count = class_counts.get(cls, 1)
            weight = 1.0 / count
            weights[cls] = weight
        
        # Normalize to sum=1
        total = sum(weights.values())
        weights = {cls: w / total for cls, w in weights.items()}
        
        # Convert to tensor in class ID order
        weight_list = [weights[self.id_to_class[i]] for i in range(len(self.unique_classes))]
        self.class_weights_tensor = torch.tensor(weight_list, dtype=torch.float32)
        
        logger.info("\nClass weights (inverse frequency):")
        for cls_id, cls_name in self.id_to_class.items():
            weight = self.class_weights_tensor[cls_id].item()
            logger.info(f"  {cls_name:10s}: {weight:.4f}")
    
    def get_train_loader(self) -> DataLoader:
        """Return DataLoader for training (with augmentation)."""
        dataset = HAM10000Dataset(
            self.train_metadata,
            self.dm,
            augment=True,
            class_to_id=self.class_to_id
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_val_loader(self) -> DataLoader:
        """Return DataLoader for validation (no augmentation)."""
        dataset = HAM10000Dataset(
            self.val_metadata,
            self.dm,
            augment=False,
            class_to_id=self.class_to_id
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_test_loader(self) -> DataLoader:
        """Return DataLoader for testing (no augmentation)."""
        dataset = HAM10000Dataset(
            self.test_metadata,
            self.dm,
            augment=False,
            class_to_id=self.class_to_id
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Return class weights for weighted loss functions."""
        return self.class_weights_tensor
    
    def get_id_to_class(self) -> dict:
        """Return mapping of class ID to class name."""
        return self.id_to_class
    
    def get_class_to_id(self) -> dict:
        """Return mapping of class name to class ID."""
        return self.class_to_id
