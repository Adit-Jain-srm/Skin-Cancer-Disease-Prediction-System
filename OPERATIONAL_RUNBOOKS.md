# Operational Runbooks - Skin Cancer Detection API

Production-ready operational procedures for managing the Skin Cancer Detection API in production environments.

---

## 1. Common Operational Tasks

### Start the API (Docker Compose - Local/Staging)

**Prerequisites:**
- Docker and Docker Compose installed
- .env file configured
- checkpoints/best_model.pt present

**Steps:**
```bash
# 1. Navigate to project directory
cd /path/to/Skin-Cancer-Disease-Prediction-System

# 2. Start services
docker-compose up -d

# 3. Verify API is running
curl http://localhost:5000/health

# 4. Check logs
docker-compose logs -f api

# 5. Get API info
curl http://localhost:5000/info | jq '.'
```

**Verification:**
- ✅ API responds to /health endpoint
- ✅ Status code 200
- ✅ Response includes timestamp and status

**Time to Ready:** 5-10 seconds

---

### Deploy to Kubernetes (Production)

**Prerequisites:**
- Kubernetes cluster running
- kubectl configured
- Container image pushed to registry
- PersistentVolumes available

**Steps:**
```bash
# 1. Create namespace
kubectl create namespace skin-cancer-detection

# 2. Create ConfigMap for environment
kubectl create configmap api-config \
  --from-literal=MODEL_NAME=resnet50 \
  --from-literal=DEVICE=cpu \
  -n skin-cancer-detection

# 3. Deploy application
kubectl apply -f kubernetes-deployment.yaml

# 4. Wait for rollout
kubectl rollout status deployment/skin-cancer-api -n skin-cancer-detection

# 5. Verify pods are running
kubectl get pods -n skin-cancer-detection

# 6. Check service
kubectl get svc -n skin-cancer-detection

# 7. Get LoadBalancer IP (may take 1-2 min)
kubectl get svc skin-cancer-api-svc -n skin-cancer-detection

# 8. Test API
curl http://<EXTERNAL-IP>:80/health
```

**Verification:**
- ✅ All pods in Running state
- ✅ Service has external IP/hostname
- ✅ API responds to requests
- ✅ All replicas healthy

**Time to Ready:** 30-60 seconds

---

### Monitor API Performance

**Dashboard Access:**
```bash
# Prometheus
http://localhost:9090

# Grafana
http://localhost:3000  # (admin/admin by default)

# AlertManager
http://localhost:9093
```

**Key Metrics to Check:**
1. **Request Rate**: requests/second (target: 20+)
2. **Error Rate**: % failed (target: <1%)
3. **Latency**: p95 (<100ms), p99 (<150ms)
4. **Memory**: <1GB
5. **CPU**: <80%
6. **Model Accuracy**: 80%+ on validation set

**Manual Check:**
```bash
# Check API response time
time curl -s http://localhost:5000/info | jq '.test_accuracy'

# Check container metrics
docker stats skin-cancer-api

# Kubernetes metrics
kubectl top pods -n skin-cancer-detection
kubectl top nodes
```

---

### Scale the API (Kubernetes)

**Horizontal Scaling (More Replicas):**
```bash
# Manual scaling
kubectl scale deployment skin-cancer-api --replicas=5 -n skin-cancer-detection

# Verify scaling
kubectl get pods -n skin-cancer-detection

# Check HPA status
kubectl get hpa -n skin-cancer-detection
```

**Auto-scaling Information:**
- Min replicas: 3
- Max replicas: 10
- Scale up at: 70% CPU
- Scale down at: 30% CPU

---

### Update Model Weights

**Procedure:**
```bash
# 1. Create new checkpoint with different name
cp checkpoints/best_model.pt checkpoints/best_model_v2.pt

# 2. Update deployment to use new model
kubectl set env deployment/skin-cancer-api \
  MODEL_PATH=./checkpoints/best_model_v2.pt \
  -n skin-cancer-detection

# 3. Monitor rollout
kubectl rollout status deployment/skin-cancer-api -n skin-cancer-detection

# 4. Rollback if needed (see rollback section)
```

**Verification:**
- ✅ API /info endpoint shows new model info
- ✅ Predictions working
- ✅ No error spikes
- ✅ Accuracy meets expectations

---

### View Logs

**Docker Compose:**
```bash
# Stream logs
docker-compose logs -f api

# Last 100 lines
docker logs --tail 100 skin-cancer-api

# Save to file
docker logs skin-cancer-api > api_logs.txt 2>&1

# Filter by level
docker logs skin-cancer-api | grep ERROR
```

**Kubernetes:**
```bash
# Stream logs
kubectl logs -f deployment/skin-cancer-api -n skin-cancer-detection

# Logs from specific pod
kubectl logs <pod-name> -n skin-cancer-detection

# Previous logs (if crashed)
kubectl logs -p <pod-name> -n skin-cancer-detection

# All pods at once
kubectl logs -l app=skin-cancer-api -n skin-cancer-detection --all-containers=true
```

---

## 2. Troubleshooting Guide

### Issue: API Returns 500 Errors

**Diagnosis:**
```bash
# Check logs for errors
docker logs skin-cancer-api | tail -50

# Check if model file exists
docker exec skin-cancer-api ls -lh checkpoints/best_model.pt

# Test model loading
docker exec skin-cancer-api python -c "
from src.inference import InferenceEngine
engine = InferenceEngine('checkpoints/best_model.pt')
print('✓ Model loaded successfully')
"
```

**Solutions:**
1. Verify model checkpoint exists
2. Check file permissions
3. Review application logs
4. Restart container: `docker-compose restart api`
5. Check available memory: `docker stats`

### Issue: Slow Predictions

**Diagnosis:**
```bash
# Check response time
time curl -X POST http://localhost:5000/predict -F "image=@test.jpg"

# Monitor container memory
docker stats skin-cancer-api

# Check CPU usage
docker stats --no-stream
```

**Solutions:**
1. **Increase resources**: Update docker-compose.yml memory limit
2. **Enable GPU**: Set `DEVICE=cuda` in .env
3. **Reduce batch size**: Lower if using batch predictions
4. **Scale horizontally**: Add more replicas
5. **Check system**: Free up host memory/CPU

### Issue: API Won't Start

**Diagnosis:**
```bash
# Check logs
docker logs skin-cancer-api

# All logs from failed start
docker-compose logs api

# Check port is not in use
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows
```

**Solutions:**
1. **Port conflict**: Change port in docker-compose.yml
2. **Model missing**: Verify checkpoints/ directory
3. **Dependency issue**: Rebuild image: `docker build -t skin-cancer-api:latest .`
4. **Memory issue**: Increase Docker memory limit
5. **Network issue**: Check firewall settings

### Issue: High Memory Usage

**Diagnosis:**
```bash
# Check peak memory
docker stats --no-stream skin-cancer-api

# Monitor over time
watch -n 1 'docker stats --no-stream skin-cancer-api'

# Check for memory leaks
docker exec skin-cancer-api ps aux | grep python
```

**Solutions:**
1. **Reduce batch size**: Smaller batches use less memory
2. **Limit workers**: Reduce API_WORKERS in .env
3. **Increase available memory**: Host system memory
4. **Restart container**: May clear accumulated memory
5. **Check for leaks**: Monitor over time, should stabilize

---

## 3. Maintenance Procedures

### Backup Model Weights

**Regular Backups:**
```bash
# Local backup
cp checkpoints/best_model.pt /backup/best_model_$(date +%Y%m%d).pt

# To cloud storage (AWS S3 example)
aws s3 cp checkpoints/best_model.pt s3://my-bucket/models/best_model_$(date +%Y%m%d).pt

# Kubernetes PVC backup
kubectl get pvc -n skin-cancer-detection
# Backup via cloud provider's snapshot feature
```

**Retention Policy:**
- Keep last 7 daily backups
- Keep last 4 weekly backups
- Keep last 12 monthly backups
- Store in geographically diverse locations

---

### Update API Image

**Docker:**
```bash
# 1. Build new image
docker build -t skin-cancer-api:v2.0 .

# 2. Test locally
docker run -p 5000:5000 skin-cancer-api:v2.0

# 3. Push to registry
docker push registry.example.com/skin-cancer-api:v2.0

# 4. Update docker-compose
# Edit docker-compose.yml, change image tag

# 5. Restart with new image
docker-compose up -d api

# 6. Verify
curl http://localhost:5000/health
```

**Kubernetes:**
```bash
# 1. Tag and push new image
docker build -t registry.example.com/skin-cancer-api:v2.0 .
docker push registry.example.com/skin-cancer-api:v2.0

# 2. Update deployment
kubectl set image deployment/skin-cancer-api \
  skin-cancer-api=registry.example.com/skin-cancer-api:v2.0 \
  -n skin-cancer-detection

# 3. Monitor rollout
kubectl rollout status deployment/skin-cancer-api -n skin-cancer-detection

# 4. Verify
kubectl get pods -n skin-cancer-detection
```

---

### Database Maintenance (if using logs database)

```bash
# Vacuum database for space
sqlite3 logs.db "VACUUM;"

# Check database size
du -h logs.db

# Backup database
cp logs.db logs_$(date +%Y%m%d).db.backup

# Archive old logs
find logs/ -type f -mtime +30 -exec gzip {} \;
```

---

## 4. Disaster Recovery

### Rollback to Previous Version

**Docker:**
```bash
# 1. Stop current container
docker-compose down

# 2. Restore previous image
docker tag skin-cancer-api:previous skin-cancer-api:latest

# 3. Start with previous
docker-compose up -d

# 4. Verify
curl http://localhost:5000/health
```

**Kubernetes:**
```bash
# 1. View rollout history
kubectl rollout history deployment/skin-cancer-api -n skin-cancer-detection

# 2. Rollback to previous
kubectl rollout undo deployment/skin-cancer-api -n skin-cancer-detection

# 3. Rollback to specific revision
kubectl rollout undo deployment/skin-cancer-api --to-revision=2 -n skin-cancer-detection

# 4. Verify
kubectl rollout status deployment/skin-cancer-api -n skin-cancer-detection
```

### Recovery from Data Loss

**Model Checkpoint Recovery:**
```bash
# 1. Restore from backup
aws s3 cp s3://my-bucket/models/best_model_20260401.pt checkpoints/best_model.pt

# 2. Verify model integrity
python -c "
import torch
model = torch.load('checkpoints/best_model.pt')
print('✓ Model loaded successfully')
"

# 3. Restart API
docker-compose restart api
```

---

## 5. Performance Optimization

### Optimize for Throughput

```bash
# Update docker-compose.yml
environment:
  - API_WORKERS=8           # Increase web workers
  - BATCH_SIZE=64           # Larger batches
  - DEVICE=cuda             # Use GPU if available
```

### Optimize for Latency

```bash
# Update docker-compose.yml
environment:
  - API_WORKERS=2           # Fewer workers, less context switching
  - BATCH_SIZE=1            # Real-time processing
  - DEVICE=cuda             # Use GPU for faster inference
```

### Load Balancing

```bash
# Multiple replicas behind load balancer
kubectl scale deployment skin-cancer-api --replicas=5 -n skin-cancer-detection

# Load balancer distributes requests
# Monitor with:
kubectl top pods -n skin-cancer-detection
```

---

## 6. Health Checks & Alerts

### Manual Health Check

```bash
#!/bin/bash
API_URL="http://localhost:5000"

echo "Checking API health..."

# Health endpoint
health=$(curl -s $API_URL/health)
if [ $? -eq 0 ]; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
    exit 1
fi

# Model info
info=$(curl -s $API_URL/info)
if [ $? -eq 0 ]; then
    echo "✓ Model accessible"
else
    echo "✗ Model not accessible"
    exit 1
fi

# Inference test
test_result=$(curl -s -X POST $API_URL/predict -F "image=@test_image.jpg")
if [ $? -eq 0 ]; then
    echo "✓ Inference working"
else
    echo "✗ Inference failed"
    exit 1
fi

echo "✓ All health checks passed"
```

### Common Alerts to Configure

1. **API Down**: up{job="skin-cancer-api"} == 0
2. **High Error Rate**: error_rate > 5%
3. **High Latency**: p95_latency > 200ms
4. **High Memory**: memory_usage > 90%
5. **High CPU**: cpu_usage > 80%
6. **Out of Memory**: container_oom_kills > 0

---

## 7. Contact & Escalation

### On-Call Procedures

**Severity Levels:**
- **Critical** (SIRI): API down, >5% error rate → Page on-call
- **High** (< 200ms latency): High latency → Email alert
- **Medium**: Memory >80%: Slack notification
- **Low**: Performance trending: Weekly report

**Contact Chain:**
1. Primary on-call: [Contact info]
2. Secondary backup: [Contact info]
3. Management escalation: [Contact info]

---

## Quick Commands Reference

```bash
# Docker Compose
docker-compose up -d          # Start
docker-compose down           # Stop
docker compose logs -f api    # Logs
docker-compose ps            # Status
docker-compose restart api   # Restart

# Kubernetes
kubectl apply -f kubernetes-deployment.yaml     # Deploy
kubectl get pods -n skin-cancer-detection      # Check pods
kubectl logs -f deployment/skin-cancer-api     # Logs
kubectl delete -f kubernetes-deployment.yaml   # Remove

# API Testing
curl http://localhost:5000/health      # Health
curl http://localhost:5000/info        # Info
curl -X POST http://localhost:5000/predict -F "image=@img.jpg"  # Predict

# Monitoring
docker stats skin-cancer-api           # Docker metrics
kubectl top pods -n skin-cancer-detection     # K8s metrics
```

---

**Last Updated**: 2026-04-12  
**Document Version**: 1.0  
**Status**: Production Ready
