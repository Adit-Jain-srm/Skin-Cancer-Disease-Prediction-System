# Skin Cancer Disease Prediction System

A research-oriented **skin lesion classification** pipeline built with PyTorch. The system preprocesses dermoscopic or clinical lesion images, runs a trained **ResNet50** (transfer learning) classifier on the **HAM10000** label set, and exposes both a **command-line interface** and an **integrated web application** (Flask API + modern React frontend).

The project is suitable as a **software engineering or machine learning capstone** and as a baseline for experimentation with data augmentation, metrics, and deployment patterns.

> **Medical disclaimer:** This software is for **education and decision-support research only**. It is not a medical device, is not FDA-cleared or CE-marked, and must **not** replace examination by a qualified clinician.

---

## Capabilities

| Area | Description |
|------|-------------|
| **Classification** | Seven HAM10000 classes (e.g., melanoma, melanocytic nevus, BCC, AK, benign keratosis, dermatofibroma, vascular lesion). |
| **Inference** | Single-image prediction with per-class probabilities and confidence. |
| **Training** | Transfer learning, augmentation (Albumentations), and configurable training scripts under `scripts/`. |
| **Web UI** | Image upload, live results, session history, and GPU/CPU device indicator when served via `web_app.py`. |
| **API** | JSON endpoints for health, config, and prediction (`deploy_api.py` for API-only deployments). |

---

## Architecture

- **Core library (`src/`)** — Data loading, preprocessing, ResNet50 wiring, training loops, metrics, and `InferenceEngine` for production-style loading of checkpoints.
- **Training & evaluation (`scripts/`)** — End-to-end training, evaluation, and prediction entry points.
- **Web stack** — `web_app.py` serves the UI and implements `/api/*` routes. The UI is a **Vite + React + TypeScript** app in `frontend/` (Framer Motion, accessible motion defaults). After `npm run build`, Flask serves `frontend/dist/`; if no build is present, it falls back to `frontend.html`.

```text
Browser  →  Flask (web_app.py)  →  InferenceEngine  →  checkpoint (.pt)
                 ↓
            /api/config, /api/predict, …
```

---

## Requirements

- **Python** 3.13+
- **PyTorch** 2.6 / **torchvision** 0.21 (CUDA 12.4 wheels recommended on NVIDIA hardware)
- **Node.js** 20+ (only for building the React frontend)
- **Hardware** — CPU-only is supported; **GPU strongly recommended** for training and faster inference (typical laptop dGPU with 4GB+ VRAM is sufficient for inference).

Full Python dependencies are listed in `requirements.txt`.

---

## Installation

### 1. Clone and enter the repository

```bash
git clone <repository-url>
cd Skin-Cancer-Disease-Prediction-System
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch, then the rest of the stack

**NVIDIA GPU (CUDA 12.4):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**CPU only:**

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 4. Verify GPU (optional)

```bash
python scripts/check_gpu.py
```

---

## Usage

### Train a model

From the repository root (with the virtual environment activated):

```bash
python scripts/train_phase4.py
```

Other training and evaluation scripts live under `scripts/` (e.g., transfer learning, abbreviated runs, hyperparameter tuning).

### Run CLI prediction

```bash
python scripts/predict.py --image_path path/to/image.jpg
```

Adjust arguments per the script’s `--help` output.

### Run the integrated web application

1. **Build the frontend** (once, or after UI changes):

   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

2. **Start the server** (requires a trained checkpoint):

   ```bash
   python web_app.py --model-path checkpoints/best_model.pt --port 5000
   ```

3. Open **http://localhost:5000** in a browser.

**Local development** (hot reload for UI): run Flask as above, then in another terminal:

```bash
cd frontend
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:5000` by default.

### API-only deployment

For deployments that expose only the REST API, use `deploy_api.py` (see `DEPLOYMENT_QUICKSTART.md` for operational detail).

### Tests

```bash
pytest tests/ -v
```

---

## Project structure

```text
Skin-Cancer-Disease-Prediction-System/
├── src/                    # Core ML code (models, data, training, inference)
├── scripts/                # Training, prediction, evaluation, utilities
├── tests/                  # Pytest suite
├── frontend/               # Vite + React + TypeScript UI
├── checkpoints/            # Saved model weights (.pt)
├── web_app.py              # Flask app: API + static UI
├── deploy_api.py           # API-focused Flask entrypoint
├── frontend.html           # Legacy single-page fallback (no Node build)
├── requirements.txt
├── DEPLOYMENT_QUICKSTART.md
└── README.md
```

---

## Troubleshooting

### `pin_memory` warning without GPU

If you see warnings about `pin_memory` without an accelerator, they are benign on CPU; the dataloaders are intended to disable `pin_memory` when CUDA is unavailable.

### CUDA not detected

Reinstall PyTorch with the correct CUDA wheel index (see Installation), then run `python scripts/check_gpu.py`.

### Web UI shows the old page or missing assets

Run `npm run build` inside `frontend/` so `frontend/dist/` exists. Ensure you restart `web_app.py` after rebuilding.

---

## Limitations

- Model quality depends on training data, splits, and hyperparameters; reported metrics in your own runs are authoritative for your checkpoint.
- Class imbalance and dataset bias (common in dermatology datasets) can affect real-world behavior.
- The web server bundled with Flask is intended for **development and demos**; use a production WSGI server and hardening for public deployment.

---

## Future work

- Additional lesion categories and external validation cohorts.
- Container images and cloud GPU deployment automation.
- Mobile or edge clients consuming the same API.
- Auditable logging and optional authenticated prediction history.

---

## Acknowledgements

- **HAM10000** and related public dermatology imaging resources used for research and education.
- Course staff, mentors, and peers supporting the original SEPM / academic project context.
- Open-source ecosystem: PyTorch, Flask, Vite, React, Framer Motion.
