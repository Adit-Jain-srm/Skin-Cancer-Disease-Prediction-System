## Skin Cancer Disease Prediction System

An AI-based application that assists in **early detection and classification of skin cancer** from dermoscopic or skin-lesion images using **Convolutional Neural Networks (CNNs)** and classic image preprocessing techniques.

This project is developed as an **SEPM (Software Engineering Project Management) / academic project**.

---

## 1. Problem Statement

Skin diseases are among the most common health concerns worldwide and require **prompt and accurate diagnosis**. Conventional diagnosis depends heavily on dermatologist expertise and is affected by:

- Low contrast between lesions and surrounding skin  
- Visual similarity between healthy and diseased regions  
- Limited availability of specialists in remote areas  

This project aims to provide an **automated, computer-aided diagnostic system** that:

- Preprocesses skin images (noise reduction, grayscale conversion, enhancement)  
- Extracts important visual features  
- Uses a **CNN model** to classify the skin lesion type  
- Supports clinicians with consistent, reliable prediction results and confidence scores  

> **Note:** The system is intended as a **decision-support tool**, not a replacement for professional medical diagnosis.

---

## 2. Objectives

- **Early detection** of skin cancer from lesion images  
- **Automated preprocessing**: resizing, normalization, denoising, augmentation  
- **CNN-based classification** of skin disease categories (e.g., benign vs malignant / multiple classes)  
- **User-friendly interface** for image upload and prediction  
- Provide **performance metrics** (accuracy, precision, recall) for evaluation  

---

## 3. System Overview

### 3.1 High-Level Workflow

1. **Image Upload** (user selects or captures a lesion image)  
2. **Image Preprocessing**  
   - Resize to `224×224`  
   - Normalize pixel values to \([0, 1]\)  
   - Denoise and enhance contrast  
   - Data augmentation during training  
3. **Feature Extraction & CNN Classification**  
4. **Prediction & Result Display**  
   - Predicted class (e.g., melanoma / nevus / benign)  
   - Confidence score  
5. **(Optional)** Store or log prediction for offline analysis (if enabled)

### 3.2 Core Modules

- **Dataset Manager**
  - Reads and validates labelled datasets (e.g., HAM10000)
  - Manages train/validation/test splits
- **Image Preprocessing Module**
  - Resizing, normalization, noise removal, augmentation
- **CNN Model Module**
  - Defines CNN or transfer-learning architecture
  - Training, validation, evaluation
- **Prediction & Result Module**
  - Loads saved model
  - Performs inference on new images
  - Computes confidence scores
- **User Interface**
  - CLI or web UI (e.g., Flask) for image upload and result viewing

---

## 4. Features

- **Upload skin lesion images**
- **Automatic preprocessing** (resize, normalize, denoise, augment)
- **Trainable CNN model**
- **Prediction with confidence score**
- **Evaluation metrics**: accuracy, precision, recall
- **Extensible architecture** to plug in advanced models (ResNet, EfficientNet, etc.)

---

## 5. Requirements

### 5.1 Software / Tools

- **OS**: Windows 11 (64-bit) or compatible
- **Language**: Python 3.13+
- **Core Libraries**:
  - `torch` 2.6.0 (with CUDA 12.4 support)
  - `torchvision` 0.21.0
  - `torchaudio` 2.6.0
  - `opencv-python`
  - `numpy`
  - `scikit-learn`
  - `matplotlib` / `seaborn` (for plots)
  - `flask` (web UI)
  - `albumentations` (image augmentation)
- **Environment**: `venv` (recommended)
- **GPU Support**: NVIDIA CUDA 12.4 (recommended for training speedup)

### 5.2 Hardware

- **Minimum**: 8GB RAM, multi-core CPU
- **Recommended**: 16GB+ RAM, NVIDIA GPU (RTX 3050 Ti or better) with 4GB+ VRAM
- **Storage**: 5GB+ for dataset and models

---

## 6. Installation & Setup

### 6.1 Prerequisites

Ensure Python 3.13+ is installed:
```bash
python --version
```

### 6.2 Clone & Navigate

```bash
git clone <repository-url>
cd Skin-Cancer-Disease-Prediction-System
```

### 6.3 Create Virtual Environment

```bash
python -m venv .venv
```

**Activate on Windows:**
```powershell
.venv\Scripts\Activate.ps1
```

**Activate on macOS/Linux:**
```bash
source .venv/bin/activate
```

### 6.4 Install Dependencies with GPU Support

**For NVIDIA GPU (CUDA 12.4) - Recommended:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**For CPU-only (slower training):**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 6.5 Verify GPU Setup

```bash
python check_gpu.py
```

Expected output when GPU is available:
```
✓ CUDA Available: True
✓ GPU Count: 1
  GPU 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU
  Memory: 4.00 GB
```

---

## 7. Usage

### 7.1 Training the Model

**Use the virtual environment Python:**
```bash
.venv\Scripts\python train_phase4.py
```

Or if venv is activated:
```bash
python train_phase4.py
```

### 7.2 Making Predictions

```bash
python predict.py --image_path path/to/image.jpg
```

### 7.3 Running Tests

```bash
pytest tests/ -v
```

---

## 8. Project Structure

```
Skin-Cancer-Disease-Prediction-System/
├── src/                          # Core modules
│   ├── model.py                  # CNN architecture
│   ├── trainer.py                # Training loop
│   ├── data_loader.py            # Dataset loading (with GPU support)
│   ├── metrics.py                # Evaluation metrics
│   └── utils.py                  # Helper functions
├── checkpoints/                  # Saved model checkpoints
├── Dataset/                      # Training data (HAM10000)
├── train_phase4.py               # Main training script
├── predict.py                    # Inference script
├── check_gpu.py                  # GPU verification
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 9. GPU Troubleshooting

### Issue: "pin_memory argument is set as true but no accelerator is found"

**Solution:** This is a warning that pin_memory=True is set without GPU. The code now automatically detects GPU availability and sets pin_memory only when CUDA is available.

### Issue: CUDA not detected despite having GPU

**Solution:** Ensure you installed PyTorch with CUDA support:
```bash
# Uninstall CPU version
pip uninstall torch torchvision torchaudio -y

# Install CUDA 12.4 version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify
python check_gpu.py
```

---

---

## 10. Limitations & Disclaimer

- Predictions are based solely on the dataset used for training.  
- **Not a certified medical device** and must not be used as the sole basis for any treatment decisions.  
- Users (especially medical practitioners) should treat this as a **supporting tool** only.

---

## 11. Future Work

- Add more skin conditions (eczema, psoriasis, acne, etc.)
- Cloud or container-based deployment (e.g., Docker + cloud GPU)
- Android mobile app for on-device or cloud-assisted prediction
- User accounts and prediction history
- Real-time camera capture in UI

---

## 12. Acknowledgements

- Public skin lesion datasets such as **HAM10000**
- Research literature and reference paper(s) included in `References/`
- Academic guides and mentors for the SEPM project

