#!/usr/bin/env python3
"""
Comprehensive HAM10000 Dataset Analysis Script
Analyzes metadata, class distribution, image validity, and data quality
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import Counter
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = Path(__file__).parent / "Dataset"
METADATA_FILE = DATASET_PATH / "HAM10000_metadata.csv"
IMAGES_DIR_1 = DATASET_PATH / "HAM10000_images_part_1"
IMAGES_DIR_2 = DATASET_PATH / "HAM10000_images_part_2"

print("="*80)
print("HAM10000 DATASET COMPREHENSIVE ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD AND INSPECT METADATA
# ============================================================================
print("\n1. METADATA STRUCTURE & BASIC INFO")
print("-"*80)

try:
    df = pd.read_csv(METADATA_FILE)
    print(f"✓ Metadata file loaded successfully")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
except Exception as e:
    print(f"✗ Error loading metadata: {e}")
    exit(1)

# Display column info
print("\nColumn Information:")
for col in df.columns:
    dtype = df[col].dtype
    non_null = df[col].notna().sum()
    null_count = df[col].isna().sum()
    print(f"  {col:15} | Type: {str(dtype):10} | Non-null: {non_null:5} | Null: {null_count:5}")

# Show sample rows
print("\nFirst 5 Sample Rows:")
print(df.head(5).to_string())

# ============================================================================
# 2. DATA TYPES & BASIC STATISTICS
# ============================================================================
print("\n2. DATA QUALITY ASSESSMENT")
print("-"*80)

print("\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values detected")
else:
    for col, count in missing[missing > 0].items():
        pct = (count / len(df)) * 100
        print(f"  {col}: {count} ({pct:.2f}%)")

print("\nAge Statistics (non-null values):")
age_stats = df['age'].describe()
print(f"  Count:  {age_stats['count']:.0f}")
print(f"  Mean:   {age_stats['mean']:.2f}")
print(f"  Std:    {age_stats['std']:.2f}")
print(f"  Min:    {age_stats['min']:.0f}")
print(f"  25%:    {age_stats['25%']:.0f}")
print(f"  50%:    {age_stats['50%']:.0f}")
print(f"  75%:    {age_stats['75%']:.0f}")
print(f"  Max:    {age_stats['max']:.0f}")

print("\nGender Distribution:")
sex_counts = df['sex'].value_counts()
for sex, count in sex_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {sex:10}: {count:5} ({pct:6.2f}%)")
unknown = len(df) - sex_counts.sum()
if unknown > 0:
    print(f"  Unknown:  {unknown:5} ({(unknown/len(df))*100:6.2f}%)")

print("\nBody Localization Distribution (Top 10):")
localization_counts = df['localization'].value_counts()
for idx, (loc, count) in enumerate(localization_counts.head(10).items(), 1):
    pct = (count / len(df)) * 100
    print(f"  {idx:2}. {loc:20}: {count:5} ({pct:6.2f}%)")
if len(localization_counts) > 10:
    other_sum = localization_counts[10:].sum()
    print(f"  Others ({len(localization_counts)-10} locations): {other_sum:5} ({(other_sum/len(df))*100:6.2f}%)")

# ============================================================================
# 3. CLASS DISTRIBUTION (DIAGNOSIS)
# ============================================================================
print("\n3. CLASS DISTRIBUTION (DIAGNOSIS TYPES)")
print("-"*80)

class_dist = df['dx'].value_counts().sort_values(ascending=False)
print(f"\nDisease Type Distribution:")
print(f"{'Disease Type':<20} {'Count':>6} {'Percentage':>12} {'Bar Chart':>40}\n")

diagnosis_mapping = {
    'nv': 'Nevus',
    'mel': 'Melanoma',
    'bkl': 'Keratosis/Benign',
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'vasc': 'Vascular Lesion',
    'df': 'Dermatofibroma'
}

total_samples = len(df)
for disease, count in class_dist.items():
    pct = (count / total_samples) * 100
    bar_length = int(pct / 2)
    bar = "█" * bar_length
    label = diagnosis_mapping.get(disease, disease)
    print(f"{label:<20} {count:>6} {pct:>11.2f}% {bar}")

# Calculate imbalance ratio
max_class = class_dist.max()
min_class = class_dist.min()
imbalance_ratio = max_class / min_class
print(f"\nClass Imbalance Analysis:")
print(f"  Largest class:  {class_dist.idxmax():10} ({max_class:5} samples)")
print(f"  Smallest class: {class_dist.idxmin():10} ({min_class:5} samples)")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
print(f"  Status: {'HIGHLY IMBALANCED' if imbalance_ratio > 3 else 'MODERATELY IMBALANCED' if imbalance_ratio > 1.5 else 'RELATIVELY BALANCED'}")

# ============================================================================
# 4. DIAGNOSIS TYPE CONFIRMATION
# ============================================================================
print("\n4. DIAGNOSIS CONFIRMATION METHOD")
print("-"*80)

dx_type_dist = df['dx_type'].value_counts()
print("\nDiagnosis Confirmation Type:")
for dtype, count in dx_type_dist.items():
    pct = (count / len(df)) * 100
    print(f"  {dtype:15}: {count:5} ({pct:6.2f}%)")

# ============================================================================
# 5. IMAGE PATH ANALYSIS & VALIDATION
# ============================================================================
print("\n5. IMAGE PATH & EXISTENCE ANALYSIS")
print("-"*80)

print(f"\nImage Directories:")
print(f"  Part 1: {IMAGES_DIR_1}")
print(f"  Part 2: {IMAGES_DIR_2}")

part1_exists = IMAGES_DIR_1.exists()
part2_exists = IMAGES_DIR_2.exists()
print(f"  Part 1 exists: {'✓' if part1_exists else '✗'}")
print(f"  Part 2 exists: {'✓' if part2_exists else '✗'}")

# Get all image files
all_images = {}
if part1_exists:
    images_p1 = list(IMAGES_DIR_1.glob("ISIC_*.jpg"))
    all_images.update({img.stem: img for img in images_p1})
    print(f"  Files in Part 1: {len(images_p1)}")

if part2_exists:
    images_p2 = list(IMAGES_DIR_2.glob("ISIC_*.jpg"))
    all_images.update({img.stem: img for img in images_p2})
    print(f"  Files in Part 2: {len(images_p2)}")

print(f"\n  Total unique image files: {len(all_images)}")

# Validate image IDs against metadata
print("\nImage-Metadata Validation:")
metadata_image_ids = set(df['image_id'].unique())
file_image_ids = set(all_images.keys())

matching = metadata_image_ids & file_image_ids
missing_from_files = metadata_image_ids - file_image_ids
extra_files = file_image_ids - metadata_image_ids

print(f"  Image IDs in metadata: {len(metadata_image_ids)}")
print(f"  Image files available: {len(file_image_ids)}")
print(f"  Matching:             {len(matching)}")
print(f"  Missing from files:   {len(missing_from_files)}")
print(f"  Extra files:          {len(extra_files)}")

if missing_from_files:
    print("  ⚠ Sample missing images:")
    for img_id in list(missing_from_files)[:5]:
        print(f"    - {img_id}")
    if len(missing_from_files) > 5:
        print(f"    ... and {len(missing_from_files)-5} more")

# ============================================================================
# 6. IMAGE FORMAT & RESOLUTION ANALYSIS
# ============================================================================
print("\n6. IMAGE FORMAT & RESOLUTION ANALYSIS")
print("-"*80)

resolutions = Counter()
formats = Counter()
file_sizes = []
invalid_images = []

print("\nScanning image properties (this may take a moment)...")
count = 0
for img_id, img_path in all_images.items():
    count += 1
    if count % 1000 == 0:
        print(f"  Processed {count}/{len(all_images)} images...")
    
    try:
        img = Image.open(img_path)
        resolution = img.size  # (width, height)
        file_format = img.format
        file_size = img_path.stat().st_size / (1024 * 1024)  # MB
        
        resolutions[resolution] += 1
        formats[file_format] += 1
        file_sizes.append(file_size)
    except Exception as e:
        invalid_images.append((img_id, str(e)))

print(f"✓ Scanned {count} images")

print(f"\nImage Format Distribution:")
for fmt, count in formats.most_common():
    print(f"  {fmt:10}: {count:5} files")

print(f"\nImage Resolution Distribution:")
for res, count in resolutions.most_common():
    print(f"  {res} pixels: {count:5} files")

if file_sizes:
    file_sizes = np.array(file_sizes)
    print(f"\nFile Size Statistics:")
    print(f"  Mean:     {file_sizes.mean():.2f} MB")
    print(f"  Median:   {np.median(file_sizes):.2f} MB")
    print(f"  Std:      {file_sizes.std():.2f} MB")
    print(f"  Min:      {file_sizes.min():.2f} MB")
    print(f"  Max:      {file_sizes.max():.2f} MB")

if invalid_images:
    print(f"\n⚠ Invalid/Corrupted Images: {len(invalid_images)}")
    for img_id, error in invalid_images[:5]:
        print(f"  {img_id}: {error}")
else:
    print(f"\n✓ All images are valid")

# ============================================================================
# 7. LESION ID & IMAGE MULTIPLICITY ANALYSIS
# ============================================================================
print("\n7. LESION & IMAGE RELATIONSHIP ANALYSIS")
print("-"*80)

unique_lesions = df['lesion_id'].nunique()
unique_images = df['image_id'].nunique()
total_records = len(df)

print(f"\nData Structure:")
print(f"  Total records:      {total_records}")
print(f"  Unique lesion IDs:  {unique_lesions}")
print(f"  Unique image IDs:   {unique_images}")
print(f"  Avg images/lesion:  {total_records/unique_lesions:.2f}")

# Check how many images per lesion
lesion_counts = df.groupby('lesion_id').size()
print(f"\nImages per Lesion Distribution:")
print(f"  Min:    {lesion_counts.min()}")
print(f"  Max:    {lesion_counts.max()}")
print(f"  Mean:   {lesion_counts.mean():.2f}")
print(f"  Median: {lesion_counts.median():.0f}")

mult_dist = lesion_counts.value_counts().sort_index()
print(f"\n  Lesions with N images:")
for n, count in mult_dist.items():
    pct = (count / unique_lesions) * 100
    print(f"    {n} image(s):  {count:5} lesions ({pct:6.2f}%)")

# ============================================================================
# 8. CROSS-VALIDATION: Check consistency within lesions
# ============================================================================
print("\n8. DATA CONSISTENCY CHECK")
print("-"*80)

inconsistencies = []
for lesion_id in df['lesion_id'].unique():
    lesion_data = df[df['lesion_id'] == lesion_id]
    
    # Check if all records for a lesion have same diagnosis
    if lesion_data['dx'].nunique() > 1:
        inconsistencies.append(f"Lesion {lesion_id}: Multiple diagnoses ({set(lesion_data['dx'])})")
    
    # Check if all records for a lesion have same age
    if lesion_data['age'].nunique() > 1:
        inconsistencies.append(f"Lesion {lesion_id}: Multiple ages ({set(lesion_data['age'])})")
    
    # Check if all records for a lesion have same sex
    if lesion_data['sex'].nunique() > 1:
        inconsistencies.append(f"Lesion {lesion_id}: Multiple sexes ({set(lesion_data['sex'])})")

if inconsistencies:
    print(f"⚠ Found {len(inconsistencies)} inconsistencies:")
    for issue in inconsistencies[:5]:
        print(f"  - {issue}")
    if len(inconsistencies) > 5:
        print(f"  ... and {len(inconsistencies)-5} more")
else:
    print("✓ All data is consistent within lesion groups")

# ============================================================================
# 9. PREPROCESSING REQUIREMENTS
# ============================================================================
print("\n9. PREPROCESSING REQUIREMENTS & RECOMMENDATIONS")
print("-"*80)

recommendations = []

if imbalance_ratio > 3:
    recommendations.append("• CLASS IMBALANCE: Use class weights during training OR apply oversampling/undersampling")

if len(missing_from_files) > 0:
    recommendations.append(f"• MISSING IMAGES: Remove {len(missing_from_files)} records with missing image files")

if invalid_images:
    recommendations.append(f"• CORRUPTED IMAGES: Remove {len(invalid_images)} corrupted image files")

if len(set(resolutions)) > 1:
    recommendations.append("• VARIABLE RESOLUTIONS: Resize all images to uniform resolution (e.g., 224x224, 448x448)")

recommendations.append("• NORMALIZATION: Apply standard preprocessing (resize, normalize pixel values to [0,1] or [-1,1])")
recommendations.append("• DATA AUGMENTATION: Apply rotation, flip, brightness, contrast adjustments for generalization")
recommendations.append("• TRAIN/VAL/TEST SPLIT: Use stratified split to maintain class distribution across sets")

print("\nKey Recommendations:")
for rec in recommendations:
    print(rec)

# ============================================================================
# 10. SUMMARY STATISTICS TABLE
# ============================================================================
print("\n10. SUMMARY STATISTICS TABLE")
print("-"*80)

summary_data = []
for disease, count in class_dist.items():
    pct = (count / total_samples) * 100
    label = diagnosis_mapping.get(disease, disease)
    summary_data.append({
        'Disease Type': label,
        'Code': disease,
        'Count': count,
        'Percentage': f"{pct:.2f}%",
        'Images Available': count if disease in [d for d in df['dx'].unique()] else 0
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\n{'TOTAL':<20} {total_samples:>6}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print(f"\n📊 DATASET OVERVIEW:")
print(f"  • Total samples (records): {total_samples}")
print(f"  • Unique lesions: {unique_lesions}")
print(f"  • Unique images: {unique_images}")
print(f"  • Disease classes: {len(class_dist)}")
print(f"  • Class imbalance ratio: {imbalance_ratio:.2f}:1 ({'IMBALANCED' if imbalance_ratio > 1.5 else 'BALANCED'})")
print(f"  • Image availability: {len(matching)}/{len(metadata_image_ids)} ({(len(matching)/len(metadata_image_ids)*100):.1f}%)")
print(f"  • Data quality: {'✓ GOOD' if not inconsistencies and not invalid_images else '⚠ ISSUES FOUND'}")

print("\n" + "="*80)
