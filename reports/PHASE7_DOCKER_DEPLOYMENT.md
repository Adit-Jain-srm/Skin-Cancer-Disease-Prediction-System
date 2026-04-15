# Phase 7: Docker Deployment Guide

## Overview
This guide covers containerizing and deploying the Skin Cancer Detection API in Docker for production environments.

## Prerequisites
- Docker 20.10+ installed and running
- Docker Compose 2.0+ installed
- ~2GB disk space for base image
- ~800MB VRAM or RAM for inference

## Quick Start (5 minutes)

### 1. Build Docker Image
```bash
docker build -t skin-cancer-api:latest .
```

**Build output check:**
- Image size: ~2.5GB (includes PyTorch 2.6, CUDA libraries)
- Base: python:3.13-slim (138MB)
- Dependencies: ~2.3GB
- Application code: ~5MB
- Model checkpoint: ~100MB

### 2. Run Single Container
```bash
docker run -d \
  --name skin-cancer-api \
  -p 5000:5000 \
  -v $(pwd)/logs:/app/logs \
  skin-cancer-api:latest
```

### 3. Test API
```bash
# Health check
curl http://localhost:5000/health

# Get model info
curl http://localhost:5000/info

# Predict on sample image
curl -X POST http://localhost:5000/predict \
  -F "image=@sample_image.jpg"
```

## Docker Compose Deployment (Recommended)

### 1. Set Environment Variables
```bash
cp .env.template .env
# Edit .env with your configuration
nano .env
```

### 2. Start Services
```bash
docker-compose up -d
```

**Services started:**
- API server on port 5000
- Prometheus metrics on port 9090 (optional)

### 3. Check Status
```bash
docker-compose ps
docker-compose logs -f api
```

### 4. Stop Services
```bash
docker-compose down
```

## Production Configuration

### 1. Environment Setup
```bash
# Copy and customize environment file
cp .env.template .env
```

**Key settings to configure:**
- `FLASK_ENV=production` ✓ (prevents debug mode)
- `DEVICE=cpu|cuda` (choose based on hardware)
- `CORS_ORIGINS` (restrict to your domain)
- `LOG_LEVEL=INFO` (or WARNING for less verbose)

### 2. Performance Tuning
```bash
# For high-throughput scenarios:
API_WORKERS=8          # Increase based on CPU cores
BATCH_SIZE=64          # Larger batches for efficiency
```

### 3. Security Hardening
```bash
# Update CORS_ORIGINS in .env
CORS_ORIGINS=https://yourdomain.com

# Run with read-only filesystem for model directory
docker run -d \
  --read-only \
  --tmpfs /tmp \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -v $(pwd)/logs:/app/logs \
  skin-cancer-api:latest
```

## Health Checks

### Built-in Health Check
```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' skin-cancer-api
```

### Manual Health Tests
```bash
# Test health endpoint
curl -s http://localhost:5000/health | jq .

# Test model info
curl -s http://localhost:5000/info | jq .

# Test inference
curl -X POST http://localhost:5000/predict \
  -F "image=@test_image.jpg" | jq '.predictions'
```

## Monitoring

### Prometheus Metrics
- Access at: http://localhost:9090
- Metrics endpoint: http://localhost:5000/metrics (if enabled)

### View Logs
```bash
# Real-time logs
docker-compose logs -f api

# Last 100 lines
docker logs --tail 100 skin-cancer-api

# Save logs to file
docker logs skin-cancer-api > api.log 2>&1
```

## Troubleshooting

### Container fails to start
```bash
# Check logs
docker logs skin-cancer-api

# Common issues:
# - Port 5000 already in use: change port mapping
# - Insufficient memory: increase Docker memory allocation
# - Model checkpoint missing: verify checkpoints/ directory
```

### Out of Memory
```bash
# Increase Docker memory limit (edit docker-compose.yml)
deploy:
  resources:
    limits:
      memory: 4G
```

### Slow predictions
```bash
# Check if using CPU or GPU
docker exec skin-cancer-api python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# For GPU support, use nvidia-docker
nvidia-docker run -d \
  --gpus all \
  -p 5000:5000 \
  skin-cancer-api:latest
```

## Deployment Patterns

### Single Machine (Development/Testing)
```bash
docker-compose up     # Single machine deployment
```

### Load Balanced (Production)
```bash
# Option 1: Docker Swarm
docker swarm init
docker service create --name api --publish 5000:5000 skin-cancer-api:latest

# Option 2: Kubernetes (see kubernetes-deployment.yaml)
kubectl apply -f kubernetes-deployment.yaml
```

### Cloud Deployment

#### AWS ECS
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag skin-cancer-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/skin-cancer-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/skin-cancer-api:latest

# Deploy via cloudformation or AWS console
```

#### Azure Container Instances
```bash
az acr build --registry <registry-name> --image skin-cancer-api:latest .
az container create --resource-group <rg> --name skin-cancer-api --image <registry>.azurecr.io/skin-cancer-api:latest --port 5000
```

#### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/<project>/skin-cancer-api
gcloud run deploy skin-cancer-api --image gcr.io/<project>/skin-cancer-api --port 5000
```

## API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/info` | GET | Model info and statistics |
| `/predict` | POST | Single image prediction |
| `/batch-predict` | POST | Multiple images prediction |
| `/predict-from-bytes` | POST | Predict from image bytes |

## Cleanup

### Remove containers
```bash
docker-compose down          # Remove services
docker container prune       # Remove stopped containers
```

### Remove images
```bash
docker rmi skin-cancer-api:latest
docker image prune           # Remove unused images
```

### Full cleanup
```bash
docker system prune -a       # Remove all unused images/containers (CAREFUL!)
```

## Performance Benchmarks

Based on Phase 6 testing with CPU inference:
- **Throughput**: 20-24 predictions/second
- **Latency**: ~50-60ms per image
- **Memory usage**: ~800MB peak, ~10MB growth
- **Model size**: ~100MB
- **Container startup**: ~5 seconds

For higher throughput, consider:
- GPU inference (10x faster)
- Batch processing
- Load balancing with multiple replicas
- Caching layer (Redis)

## Next Steps

1. **Test locally**: `docker-compose up && curl http://localhost:5000/health`
2. **Push to registry**: `docker push <registry>/skin-cancer-api:latest`
3. **Deploy to cloud**: Use cloud provider's container service
4. **Set up monitoring**: Configure Prometheus and Grafana
5. **Enable CI/CD**: GitHub Actions or similar for auto-deployment

## Support

For issues or questions:
- Check logs: `docker logs skin-cancer-api`
- Test endpoints: `curl http://localhost:5000/info`
- Review configuration: Check `.env` file
- Consult Phase 6 documentation for API details
