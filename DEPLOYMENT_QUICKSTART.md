# Deployment & Quick Start Guide

**Last Updated:** April 15, 2026  
**System Status:** ✅ READY FOR PRODUCTION

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
cd "Skin-Cancer-Disease-Prediction-System"
pip install -r requirements.txt
```

### Step 2: Run the Web App
```bash
python web_app.py --port 5000
```

### Step 3: Open in Browser
```
http://localhost:5000
```

✅ **Done!** Web app is now running with both API and UI.

---

## 📋 Three Deployment Options

### Option A: All-in-One Web App (Recommended)

**Best for:** Development, testing, small deployments

```bash
python web_app.py \
    --model-path checkpoints/best_model.pt \
    --port 5000 \
    --host 0.0.0.0
```

**What you get:**
- Frontend at `http://localhost:5000`
- API endpoints at `http://localhost:5000/api/*`
- Combined in single process
- Perfect for testing and demos

**Endpoints:**
- `GET /` - Web interface
- `GET /api/health` - Health check  
- `GET /api/info` - Model information
- `GET /api/config` - GPU configuration
- `POST /api/predict` - Single image prediction
- `POST /api/predict-batch` - Batch images

---

### Option B: Separate API Server

**Best for:** When API needs to be independent

```bash
# Terminal 1: Start API server
python deploy_api.py \
    --model-path checkpoints/best_model.pt \
    --port 5000 \
    --host 0.0.0.0

# Terminal 2: Serve frontend (use any static server)
# Option 1: Python
python -m http.server 3000 --directory .
# Option 2: Node
npx http-server -p 3000
# Option 3: Any web server (nginx, Apache, etc.)
```

**Access:**
- Frontend: `http://localhost:3000/frontend.html`
- API: `http://localhost:5000/api/*`

---

### Option C: Docker Container

**Best for:** Production deployment, cloud platforms

```bash
# Build image
docker build -t skin-cancer-api:latest .

# Run container
docker run \
    --name skin-cancer-api \
    -p 5000:5000 \
    --rm \
    skin-cancer-api

# With GPU support (NVIDIA)
docker run \
    --name skin-cancer-api \
    -p 5000:5000 \
    --gpus all \
    --rm \
    skin-cancer-api
```

**Access:** `http://localhost:5000`

---

### Option D: Kubernetes Cluster

**Best for:** Production-scale deployments with auto-scaling

```bash
# Create namespace
kubectl create namespace skin-cancer

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check deployment
kubectl get pods -n skin-cancer
kubectl logs -n skin-cancer -l app=skin-cancer-api

# Port forward for testing
kubectl port-forward -n skin-cancer svc/skin-cancer-api 5000:5000

# Access at http://localhost:5000
```

**With GPU support:**
```bash
kubectl apply -f k8s/deployment-gpu.yaml
# (Requires NVIDIA GPU Operator on cluster)
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Model path
export MODEL_PATH=checkpoints/best_model.pt

# Server settings
export HOST=0.0.0.0
export PORT=5000
export DEBUG=False

# GPU settings  
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export TORCH_HOME=/tmp/torch   # PyTorch cache
```

### Config File (config.yaml)

```yaml
model:
  path: checkpoints/best_model.pt
  input_size: 224
  batch_size: 32
  device: auto  # or 'cuda', 'cpu'

server:
  host: 0.0.0.0
  port: 5000
  debug: false
  workers: 1

inference:
  timeout: 30
  max_batch_size: 100
  cache_enabled: true
```

---

## 🧪 Testing & Verification

### 1. Health Check
```bash
curl http://localhost:5000/api/health
# Response: {"status": "healthy"}
```

### 2. Model Info
```bash
curl http://localhost:5000/api/info
# Response: {"model": "ResNet50", "accuracy": 0.8029, "classes": 7, ...}
```

### 3. GPU Config
```bash
curl http://localhost:5000/api/config
# Response: {"gpu_available": true, "cuda_version": "12.1", ...}
```

### 4. Single Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
    -F "image=@test_image.jpg"

# Response:
# {
#   "success": true,
#   "prediction": {
#     "class": "Melanoma",
#     "confidence": 0.95,
#     "all_predictions": {...}
#   },
#   "processing_time_ms": 2145
# }
```

### 5. Batch Prediction
```bash
curl -X POST http://localhost:5000/api/predict-batch \
    -F "images=@image1.jpg" \
    -F "images=@image2.jpg" \
    -F "images=@image3.jpg"

# Response: Array of predictions
```

### 6. Real-World Load Test
```bash
python -m locust -f load_test_realworld.py \
    --host=http://localhost:5000 \
    --users=5 \
    --spawn-rate=1 \
    --run-time=60s \
    --headless
```

---

## 📊 Performance Tuning

### CPU Performance (Development)
```python
# Current: 2.1-4.1s per prediction
# Expected load: 0.3-0.5 predictions/second

# Optimization options:
import torch
torch.set_num_threads(8)  # Use more CPU threads
torch.set_num_interop_threads(8)
```

### GPU Performance (Production)
```bash
# Install CUDA 12.1 and NVIDIA drivers
# https://developer.nvidia.com/cuda-12-1-0-download-archive

# Install cuDNN
# https://developer.nvidia.com/cudnn

# Install GPU-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Expected: 80-120ms per prediction
# Expected load: 8-12 predictions/second
```

### Model Acceleration (Optional)
```python
# Export to ONNX for 3-5x speedup
import torch
import torch.onnx

model = torch.load('checkpoints/best_model.pt')
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")

# Or TensorRT (5-10x speedup with GPU)
# https://docs.nvidia.com/deeplearning/tensorrt/
```

---

## 🔐 Security Checklist

- [ ] Model checkpoint stored securely
- [ ] API endpoints protected (if needed)
- [ ] Input validation enabled
- [ ] File size limits enforced (50MB)
- [ ] CORS configured correctly
- [ ] Error messages don't leak internals
- [ ] Logging doesn't expose sensitive data
- [ ] Rate limiting implemented (optional)

---

## 📈 Monitoring & Logging

### Check Logs
```bash
# Web app logs
tail -f logs/web_app.log

# API logs
tail -f logs/api.log

# GPU usage
nvidia-smi -l 1
```

### Kubernetes Logging
```bash
# Real-time logs
kubectl logs -f deployment/skin-cancer-api -n skin-cancer

# All pod logs
kubectl logs -l app=skin-cancer-api -n skin-cancer --all-containers=true
```

### Prometheus Metrics (Optional)
```bash
# Add to requirements.txt
pip install prometheus-client

# Metrics endpoint: http://localhost:5000/metrics
```

---

## 🛠️ Troubleshooting

### Problem: Port Already in Use
```bash
# Option 1: Use different port
python web_app.py --port 8000

# Option 2: Kill process
lsof -i :5000  # Find process
kill -9 <PID>
```

### Problem: GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False:
# 1. Install NVIDIA drivers
# 2. Install CUDA 12.1
# 3. Reinstall PyTorch with CUDA support
```

### Problem: Model Checkpoint Not Found
```bash
# Verify file exists
ls -lh checkpoints/best_model.pt

# If missing, download from your source
# Place in: checkpoints/best_model.pt

# Set correct path
python web_app.py --model-path checkpoints/best_model.pt
```

### Problem: Memory Issues
```bash
# Reduce batch size
python web_app.py --batch-size 16  # Default 32

# Or reduce image size
python web_app.py --image-size 128  # Default 224
```

### Problem: Slow Inference
```bash
# Check if GPU is being used
nvidia-smi

# If not using GPU but available:
# Reinstall torch with CUDA support
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Check device in logs
# Should see: "🔥 GPU DETECTED AND AVAILABLE"
```

---

## 📱 API Curl Examples

### Upload and Predict
```bash
curl -X POST http://localhost:5000/api/predict \
    -H "Content-Type: multipart/form-data" \
    -F "image=@/path/to/image.jpg"
```

### Batch Processing
```bash
curl -X POST http://localhost:5000/api/predict-batch \
    -F "images=@image1.jpg" \
    -F "images=@image2.jpg" \
    -F "images=@image3.jpg"
```

### Check Server Status
```bash
curl http://localhost:5000/api/health
curl http://localhost:5000/api/info
curl http://localhost:5000/api/config
```

---

## 🌐 Browser Access

### Web Interface
```
http://localhost:5000
```

### Features Available:
- ✅ Image upload (drag & drop or click)
- ✅ Real-time prediction analysis
- ✅ Confidence visualization
- ✅ Analysis history
- ✅ Session statistics
- ✅ GPU/CPU status indicator

---

## 📦 Dependencies

### Core Requirements
- Python 3.11+
- PyTorch 2.0+
- Flask 3.0+
- Pillow 10.0+
- numpy 1.24+

### Optional
- CUDA 12.1 (for GPU)
- cuDNN 8.x (for GPU)
- Docker (for containerization)
- Kubernetes (for orchestration)

See `requirements.txt` for complete list.

---

## 🎯 Deployment Scenarios

### Scenario 1: Local Development
```bash
python web_app.py --port 5000
# Access: http://localhost:5000
```

### Scenario 2: Small Team (Docker)
```bash
docker run -p 5000:5000 skin-cancer-api:latest
# Access: http://localhost:5000
```

### Scenario 3: Production (Kubernetes)
```bash
kubectl apply -f k8s/deployment.yaml
# Auto-scales, self-healing, rolling updates
```

### Scenario 4: High Performance (GPU + Kubernetes)
```bash
kubectl apply -f k8s/deployment-gpu.yaml
# 10-20x faster inference with GPU
```

---

## 📊 Performance Expectations

| Metric | CPU | GPU |
|--------|-----|-----|
| Latency (single) | 2.1-2.5s | 80-120ms |
| Throughput | 0.4-0.5 req/s | 8-12 req/s |
| Cost/query | ~0.01¢ (CPU) | ~0.001¢ (GPU) |
| Startup time | 1-2s | 2-3s |
| Memory | ~2GB | ~4GB |

---

## 🔄 Updates & Maintenance

### Update Model
```bash
# Place new model in:
checkpoints/best_model_v2.pt

# Run with new model:
python web_app.py --model-path checkpoints/best_model_v2.pt
```

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Update Container
```bash
docker build -t skin-cancer-api:v2 .
docker run -p 5000:5000 skin-cancer-api:v2
```

---

## 📞 Support

**System Status:** ✅ PRODUCTION READY

For issues:
1. Check logs: `tail -f logs/*.log`
2. Run health check: `curl http://localhost:5000/api/health`
3. Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
4. Test with sample image
5. Check troubleshooting section above

---

## ✅ Deployment Checklist

- [ ] Python 3.11+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model checkpoint exists (`checkpoints/best_model.pt`)
- [ ] Port 5000 is available (or use different port)
- [ ] Web app starts without errors
- [ ] Health check passes (`http://localhost:5000/api/health`)
- [ ] Frontend loads (`http://localhost:5000`)
- [ ] Test prediction works
- [ ] GPU detected (if available): `torch.cuda.is_available() == True`

---

## 🎉 You're Ready!

All deployment options are ready to use. Choose one based on your needs:

1. **Quick Test** → `Option A: Web App`
2. **Development** → `Option A or B`
3. **Production Small** → `Option C: Docker`
4. **Production Scale** → `Option D: Kubernetes`

Start with:
```bash
python web_app.py
# Then open http://localhost:5000
```

**Happy deploying! 🚀**
