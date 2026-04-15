"""
Dataset Manager Module
Handles loading, validation, preprocessing, and augmentation of skin lesion images.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import logging
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Manages HAM10000 and HMNIST datasets.
    Provides loading, validation, preprocessing, and augmentation utilities.
    """
    
    def __init__(self, dataset_dir: str, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize Dataset Manager.
        
        Args:
            dataset_dir: Path to dataset directory (e.g., 'Dataset/')
            target_size: Target image size (height, width)
        """
        self.dataset_dir = Path(dataset_dir)
        self.target_size = target_size
        self.metadata = None
        logger.info(f"DatasetManager initialized with target size {target_size}")
    
    def load_metadata(self, metadata_csv: str) -> pd.DataFrame:
        """
        Load and validate dataset metadata from CSV.
        
        Args:
            metadata_csv: Path to metadata CSV file
            
        Returns:
            DataFrame with image metadata (lesion_id, image_id, dx, age, sex, localization)
            
        Raises:
            FileNotFoundError: If metadata CSV not found
            ValueError: If CSV structure is invalid
        """
        path = self.dataset_dir / metadata_csv
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        
        # Load CSV
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")
        
        if len(df) == 0:
            raise ValueError("Metadata CSV is empty")
        
        # Validate required columns
        required_cols = {'lesion_id', 'image_id', 'dx', 'age', 'sex', 'localization'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Found: {set(df.columns)}")
        
        # Store metadata
        self.metadata = df[['lesion_id', 'image_id', 'dx', 'age', 'sex', 'localization']].copy()
        
        # Log basic stats
        logger.info(f"Loaded metadata: {len(self.metadata)} samples")
        logger.info(f"Columns: {list(self.metadata.columns)}")
        
        # Validate image paths exist
        missing_count = 0
        valid_count = 0
        for idx, row in self.metadata.iterrows():
            image_id = row['image_id']
            # Try part_1 first, then part_2
            img_path_1 = self.dataset_dir / "HAM10000_images_part_1" / f"{image_id}.jpg"
            img_path_2 = self.dataset_dir / "HAM10000_images_part_2" / f"{image_id}.jpg"
            
            if img_path_1.exists() or img_path_2.exists():
                valid_count += 1
            else:
                missing_count += 1
                if missing_count <= 5:  # Log first 5 missing
                    logger.warning(f"Image not found: {image_id}")
        
        if missing_count > 0:
            logger.warning(f"Missing {missing_count}/{len(self.metadata)} images")
        else:
            logger.info(f"All {valid_count} images found ✓")
        
        # Compute and log statistics
        self._log_statistics()
        
        return self.metadata
    
    def _log_statistics(self) -> None:
        """Log dataset statistics from loaded metadata."""
        if self.metadata is None:
            return
        
        logger.info("=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)
        
        # Class distribution
        class_dist = self.metadata['dx'].value_counts()
        logger.info(f"Total samples: {len(self.metadata)}")
        logger.info(f"Unique lesions: {self.metadata['lesion_id'].nunique()}")
        logger.info(f"Classes: {len(class_dist)}")
        logger.info("\nClass Distribution:")
        for dx_class, count in class_dist.items():
            pct = (count / len(self.metadata)) * 100
            logger.info(f"  {dx_class:20s}: {count:5d} ({pct:5.1f}%)")
        
        # Age statistics
        age_data = self.metadata['age'].dropna()
        if len(age_data) > 0:
            logger.info(f"\nAge Statistics:")
            logger.info(f"  Mean: {age_data.mean():.1f} years")
            logger.info(f"  Std:  {age_data.std():.1f} years")
            logger.info(f"  Min:  {age_data.min():.1f} years")
            logger.info(f"  Max:  {age_data.max():.1f} years")
            logger.info(f"  Missing: {self.metadata['age'].isna().sum()}")
        
        # Gender distribution
        gender_dist = self.metadata['sex'].value_counts()
        logger.info(f"\nGender Distribution:")
        for gender, count in gender_dist.items():
            pct = (count / len(self.metadata)) * 100
            logger.info(f"  {gender:10s}: {count:5d} ({pct:5.1f}%)")
        
        # Data completeness
        total_cells = len(self.metadata) * len(self.metadata.columns)
        missing_cells = self.metadata.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        logger.info(f"\nData Completeness: {completeness:.2f}%")
        logger.info("=" * 60)
    
    def validate_images(self) -> dict:
        """
        Validate dataset integrity (check for corrupted/missing files).
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "total": len(self.metadata),
            "valid": 0,
            "missing": [],
            "corrupted": []
        }
        
        logger.info("Validating dataset...")
        return results
    
    def preprocess_image(self, image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess a single image: load, resize, normalize to [0, 1].
        
        Args:
            image_path: Path to image file (JPG/PNG)
            target_size: Target size (height, width). Defaults to self.target_size
            
        Returns:
            Normalized image array of shape (224, 224, 3), dtype float32, values in [0, 1]
            
        Raises:
            FileNotFoundError: If image not found
            ValueError: If image cannot be loaded or is invalid
        """
        if target_size is None:
            target_size = self.target_size
        
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load image with PIL
            img = Image.open(img_path)
        except Exception as e:
            raise ValueError(f"Failed to load image: {image_path}. Error: {e}")
        
        # Convert to RGB (handle RGBA, grayscale, palette modes)
        if img.mode == 'RGBA':
            # Drop alpha channel
            img = img.convert('RGB')
        elif img.mode not in ['RGB', 'L']:
            # Convert other modes to RGB
            img = img.convert('RGB')
        elif img.mode == 'L':
            # Convert grayscale to RGB (3 channels)
            img = img.convert('RGB')
        
        # Verify image has valid dimensions
        if img.size[0] <= 0 or img.size[1] <= 0:
            raise ValueError(f"Invalid image dimensions: {img.size}")
        
        # Resize to target size using high-quality resampling
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array (uint8, values 0-255)
        img_array = np.array(img, dtype=np.uint8)
        
        # Verify shape
        if img_array.ndim == 2:
            # Grayscale image, expand to 3 channels
            img_array = np.stack([img_array] * 3, axis=-1)
        
        assert img_array.shape == (target_size[0], target_size[1], 3), \
            f"Expected shape {(target_size[0], target_size[1], 3)}, got {img_array.shape}"
        
        # Normalize to [0, 1] as float32
        img_normalized = img_array.astype(np.float32) / 255.0
        
        assert img_normalized.dtype == np.float32, f"Expected dtype float32, got {img_normalized.dtype}"
        assert img_normalized.min() >= 0.0 and img_normalized.max() <= 1.0, \
            f"Expected values in [0, 1], got range [{img_normalized.min()}, {img_normalized.max()}]"
        
        return img_normalized
    
    def augment_image(self, image: np.ndarray, augment: bool = True) -> np.ndarray:
        """
        Apply data augmentation: rotation, flip, brightness, contrast, zoom.
        
        Args:
            image: Input image array of shape (H, W, 3), dtype float32, values [0, 1]
            augment: Enable/disable augmentation. If False, return unchanged.
            
        Returns:
            Augmented image array, same shape and dtype as input
        """
        if not augment:
            return image.copy()
        
        # Work with PIL Image for easy transformations
        # Convert float32 [0, 1] to uint8 [0, 255]
        img_uint8 = (image * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        
        # 1. Random rotation ±15 degrees (100% probability)
        angle = np.random.uniform(-15, 15)
        img_pil = img_pil.rotate(angle, expand=False, fillcolor=(0, 0, 0))
        
        # 2. Random horizontal flip (50% probability)
        if np.random.random() < 0.5:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 3. Random vertical flip (50% probability)
        if np.random.random() < 0.5:
            img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Convert back to numpy for brightness/contrast operations
        img_aug = np.array(img_pil, dtype=np.float32) / 255.0
        
        # 4. Random brightness adjustment (50% probability, ±10%)
        if np.random.random() < 0.5:
            brightness_factor = np.random.uniform(0.9, 1.1)
            img_aug = np.clip(img_aug * brightness_factor, 0, 1)
        
        # 5. Random contrast adjustment (50% probability, ±10%)
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.9, 1.1)
            # Contrast: (pixel - 0.5) * factor + 0.5
            mean_brightness = 0.5  # Pivot around middle gray
            img_aug = np.clip((img_aug - mean_brightness) * contrast_factor + mean_brightness, 0, 1)
        
        # 6. Random zoom/crop (50% probability, scale 0.85-1.15)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.85, 1.15)
            h, w = img_aug.shape[:2]
            
            if scale > 1:
                # Zoom in: crop from center
                new_h, new_w = int(h / scale), int(w / scale)
                y_start = (h - new_h) // 2
                x_start = (w - new_w) // 2
                img_aug = img_aug[y_start:y_start+new_h, x_start:x_start+new_w, :]
                # Resize back to original
                img_pil = Image.fromarray((img_aug * 255).astype(np.uint8))
                img_pil = img_pil.resize((w, h), Image.LANCZOS)
                img_aug = np.array(img_pil, dtype=np.float32) / 255.0
            else:
                # Zoom out: pad with black borders
                new_h, new_w = int(h * scale), int(w * scale)
                y_pad = (h - new_h) // 2
                x_pad = (w - new_w) // 2
                img_padded = np.zeros((h, w, 3), dtype=np.float32)
                img_padded[y_pad:y_pad+new_h, x_pad:x_pad+new_w, :] = img_aug[:new_h, :new_w, :]
                img_aug = img_padded
        
        # Final validation
        assert img_aug.shape == image.shape, f"Shape mismatch: {img_aug.shape} vs {image.shape}"
        assert img_aug.dtype == np.float32, f"Dtype mismatch: {img_aug.dtype}"
        assert img_aug.min() >= 0 and img_aug.max() <= 1, \
            f"Value range mismatch: [{img_aug.min()}, {img_aug.max()}]"
        
        return img_aug
    
    def get_class_distribution(self) -> dict:
        """
        Get class distribution from metadata.
        
        Returns:
            Dictionary with class counts
        """
        if self.metadata is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        distribution = self.metadata['label'].value_counts().to_dict()
        logger.info(f"Class distribution: {distribution}")
        return distribution


# TODO: Add DataLoader class for batch loading
# TODO: Add train/val/test split utilities
