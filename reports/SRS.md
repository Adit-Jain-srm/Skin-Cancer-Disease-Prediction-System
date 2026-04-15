# Software Requirements Specification (SRS)
## Skin Cancer Disease Prediction System

**Version**: 1.0  
**Date**: 2026-04-08  
**Status**: Approved  

---

## 1. Introduction

### 1.1 Purpose
This SRS document specifies the functional and non-functional requirements for the **Skin Cancer Disease Prediction System**, an AI-based medical image analysis application for early detection and classification of skin cancer from dermoscopic images.

### 1.2 Scope
The system encompasses:
- Image upload and validation
- Automated preprocessing (resizing, normalization, denoising, augmentation)
- CNN-based classification of skin lesion types
- User interface (Flask web app or CLI)
- Performance metrics reporting
- Model inference and confidence scoring

**Out of Scope** (future enhancements):
- Mobile app (iOS/Android)
- Cloud deployment on Azure/AWS
- Multi-user authentication and audit logs
- Real-time video analysis

### 1.3 Document Organization
1. Functional Requirements (FR)
2. Non-Functional Requirements (NFR)
3. Use Cases
4. System Constraints

---

## 2. Functional Requirements

| ID | Requirement | Description | Priority |
|------|-----------------------|-------------------------------------|----------|
| **FR1** | Image Upload | User can upload a single image (JPG, PNG) via web UI or command-line | **HIGH** |
| **FR2** | Image Validation | System validates image format, resolution, and file size (max 10MB) | **HIGH** |
| **FR3** | Image Preprocessing | System resizes image to 224×224, normalizes, denoises, and applies contrast enhancement | **HIGH** |
| **FR4** | Data Augmentation | System applies random rotation (±15°), flip, crop, and zoom during training | **HIGH** |
| **FR5** | CNN Model Training | System trains a CNN model (baseline or transfer learning) on labeled dataset | **HIGH** |
| **FR6** | Model Evaluation | System computes accuracy, precision, recall, F1-score, and confusion matrix | **HIGH** |
| **FR7** | Single Image Prediction | System predicts skin lesion class for uploaded image within 5 seconds | **HIGH** |
| **FR8** | Confidence Score | System returns prediction confidence (0–100%) alongside classification | **HIGH** |
| **FR9** | Class Distribution Report | System provides per-class accuracy and visualization (confusion matrix) | **MEDIUM** |
| **FR10** | Model Persistence | System saves/loads trained model to/from disk for inference | **HIGH** |
| **FR11** | Batch Prediction | System allows prediction on multiple images in a batch | **MEDIUM** |
| **FR12** | Web UI | Flask-based interface with form for image upload and result display | **MEDIUM** |
| **FR13** | CLI Tool | Command-line interface for prediction: `python predict.py --image <path>` | **MEDIUM** |

---

## 3. Non-Functional Requirements

| ID | Requirement | Acceptance Criteria |
|--------|--------------------------------|----------------------------------------------|
| **NFR1** | Accuracy | ≥ 85% on validation set (multi-class) |
| **NFR2** | Latency | Prediction time ≤ 5 seconds per image (CPU) |
| **NFR3** | Memory Usage | Model + inference ≤ 1GB RAM |
| **NFR4** | Image Resolution | Support images 100×100 to 4000×4000 pixels |
| **NFR5** | Usability | UI learns new user < 2 minutes; no training required |
| **NFR6** | Code Quality | PEP8 compliant, type hints, test coverage ≥ 70% |
| **NFR7** | Documentation | User manual + technical docs for all modules |
| **NFR8** | Robustness | Graceful error handling for invalid/corrupted images |
| **NFR9** | Reproducibility | Fixed random seed for model training; documented hyperparameters |
| **NFR10** | Batch Efficiency | Batch prediction ≥ 10× faster than sequential |

---

## 4. Use Cases

### UC1: Upload and Predict (Single Image)

**Actor**: End User  
**Precondition**: User has web browser open to Flask UI or command-line access  

**Main Flow**:
1. User uploads an image file (JPG/PNG)
2. System validates image format and size
3. System preprocesses image (resize, normalize, denoise)
4. System runs CNN inference
5. System displays prediction class and confidence score
6. User reviews result

**Alternate Flow** (invalid image):
- System displays error message: "Invalid image format or size exceeds 10MB"

---

### UC2: Train Model

**Actor**: ML Engineer  
**Precondition**: HAM10000 dataset is in `/Dataset/` with metadata CSV  

**Main Flow**:
1. Engineer runs training script: `python train.py --epochs 50 --batch_size 32`
2. System loads dataset and applies preprocessing + augmentation
3. System trains CNN model on GPU/CPU
4. System evaluates on validation set every N epochs
5. System saves best model checkpoint
6. System logs metrics (loss, accuracy) and saves to `reports/results.csv`

---

### UC3: Evaluate Model Performance

**Actor**: Tester / QA Engineer  
**Precondition**: Trained model exists; test dataset available  

**Main Flow**:
1. QA runs evaluation script: `python evaluate.py --model_path models/best_model.pth`
2. System loads test set and trained model
3. System computes confusion matrix, per-class metrics
4. System generates visualization and report in `reports/evaluation_report.md`
5. QA verifies accuracy meets NFR1 (≥ 85%)

---

## 5. System Constraints

### 5.1 Technical Constraints
- **Python 3.9+** required
- **TensorFlow 2.x** or **PyTorch 1.x** for model framework
- **No GPU required** (CPU inference acceptable with 5s latency)
- **Windows/Linux/macOS** compatible

### 5.2 Data Constraints
- **Input**: HAM10000 dataset (~10,000 labeled images)
- **Classes**: Up to 7 disease categories (melanoma, nevus, benign keratosis, etc.)
- **Image formats**: JPG, PNG only
- **Max file size**: 10 MB per image

### 5.3 Operational Constraints
- System deployed on **local machine** (no cloud backend)
- Single-threaded inference (concurrent requests not supported)
- Model training on CPU may take **4–8 hours** depending on hardware

---

## 6. Dependencies & Assumptions

### 6.1 Assumptions
- Dataset is clean and balanced (or mitigated with class weighting)
- Users upload valid skin lesion images (not random photos)
- Medical expertise not required to interpret predictions (confidence score provided)
- No real-time updates required; batch retraining acceptable

### 6.2 External Dependencies
- TensorFlow/PyTorch (ML framework)
- Flask (web framework)
- NumPy, Pandas, Scikit-learn (data processing)
- Matplotlib, Seaborn (visualization)

---

## 7. Success Criteria

**Project Success** if all of the following are **true** by end of Week 10:

1. ✓ **Accuracy**: Model achieves ≥ 85% on held-out test set  
2. ✓ **Latency**: Single prediction ≤ 5 seconds (CPU)  
3. ✓ **UI Ready**: Flask UI or CLI fully functional with no crashes  
4. ✓ **Documentation**: SRS, design doc, user manual, test report completed  
5. ✓ **Test Coverage**: All functional requirements (FR1–FR13) validated with test cases  
6. ✓ **Reproducibility**: Model, code, and results are reproducible; random seed fixed  

---

## 8. Change Log

| Date | Version | Author | Change |
|------|---------|--------|--------|
| 2026-04-08 | 1.0 | SEPM Team | Initial SRS approved |

