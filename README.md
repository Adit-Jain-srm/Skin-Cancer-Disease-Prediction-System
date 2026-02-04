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
- **Language**: Python 3.x
- **Core Libraries (example stack)**:
  - `tensorflow` or `torch`
  - `opencv-python`
  - `numpy`
  - `scikit-learn`
  - `matplotlib` / `seaborn` (for plots)
  - `flask` (if using web UI)
- **Environment**: `venv` / `conda` (recommended)
- **Optional**: GPU support via CUDA for faster training

### 5.2 Hardware

- Minimum: 8 GB RAM, dual-core CPU  
- Recommended: 16 GB RAM, NVIDIA GPU with ≥4 GB VRAM for deep learning training

---

## 6. Project Structure (Proposed)

This is a suggested structure for the repository:

```text
Skin-Cancer-Disease-Prediction-System/
│
├─ Dataset/
│  ├─ raw/                  # Original images (if stored locally)
│  ├─ processed/            # Preprocessed images (if cached)
│  ├─ metadata.csv          # Labels / metadata (e.g., HAM10000 CSV)
│  └─ Placeholder.md        # Dataset documentation / links
│
├─ src/
│  ├─ data/
│  │  ├─ dataset_manager.py
│  │  └─ dataloaders.py
│  ├─ preprocessing/
│  │  ├─ filters.py
│  │  └─ augmentations.py
│  ├─ models/
│  │  ├─ cnn_baseline.py
│  │  └─ transfer_learning.py
│  ├─ training/
│  │  ├─ train.py
│  │  └─ evaluate.py
│  ├─ inference/
│  │  └─ predict.py
│  └─ ui/
│     ├─ app.py             # Flask app (if used)
│     └─ templates/         # HTML templates
│
├─ notebooks/
│  ├─ EDA.ipynb             # Exploratory data analysis
│  ├─ Baseline_Model.ipynb
│  └─ Experiments_*.ipynb
│
├─ References/
│  ├─ SKIN CANCER DISEASE PREDICTION SYSTEM_ Adit Jain.docx
│  └─ IJCRT25A4490 (1).pdf
│
├─ models/
│  └─ best_model.h5 or best_model.pt
│
├─ reports/
│  ├─ sepm_plan.md
│  ├─ srs.md
│  └─ final_report.md
│
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## 7. Installation

### 7.1 Clone the Repository

```bash
git clone "<your-repo-url>.git"
cd "Skin-Cancer-Disease-Prediction-System"
```

### 7.2 Create and Activate Virtual Environment (Windows / PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 7.3 Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Adjust `requirements.txt` based on the final chosen stack (TensorFlow or PyTorch).

---

## 8. Usage

### 8.1 Training the Model

Example (to be aligned with your actual `train.py`):

```bash
python -m src.training.train ^
  --data_dir "Dataset" ^
  --metadata "Dataset/metadata.csv" ^
  --epochs 20 ^
  --batch_size 32 ^
  --model_out "models/best_model.h5"
```

### 8.2 Evaluating the Model

```bash
python -m src.training.evaluate --model_path "models/best_model.h5"
```

### 8.3 Running Prediction (CLI)

```bash
python -m src.inference.predict ^
  --model_path "models/best_model.h5" ^
  --image_path "path/to/image.jpg"
```

Output:

- Predicted disease class  
- Confidence score (e.g., 0.92)

### 8.4 Running Web UI (Optional)

```bash
python -m src.ui.app
```

Then open `http://127.0.0.1:5000` in a browser, upload an image, and view prediction.

---

## 9. Functional & Non-Functional Requirements (Summary)

- **Functional**
  - Load labelled skin disease dataset (HAM10000 etc.)
  - Validate dataset formats and paths
  - Preprocess images (resize `224×224`, normalize, denoise, augment)
  - Train, validate, and test CNN model
  - Predict skin disease class for a new image
  - Display prediction with confidence score

- **Non-Functional**
  - **Performance**: prediction time < 5 seconds on standard hardware
  - **Usability**: simple upload interface, minimal clicks
  - **Reliability**: handle invalid images gracefully; consistent results
  - **Security & Privacy**: no permanent storage of images unless explicitly enabled; no personal data collection
  - **Scalability**: support additional disease classes and advanced architectures

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

