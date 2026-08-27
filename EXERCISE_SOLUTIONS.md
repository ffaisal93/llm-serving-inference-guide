# Exercise Solutions

This document provides solutions to exercises in each learning topic.

## Topic 1: Basic Model Serving

### Exercise 1: Change the Model
**Solution:**
```python
# In model_loader.py, change:
model = LLMModel(
    model_name="distilgpt2",  # Smaller, faster model
    device="cpu"
)
```

### Exercise 2: Modify Parameters
**Solution:**
```python
# In app.py, test different temperatures:
# Temperature 0.1 = more deterministic
# Temperature 2.0 = more creative
payload = {
    "prompt": "The future of AI is",
    "temperature": 0.1,  # Try different values
    "max_length": 100
}
```

### Exercise 3: Add Logging
**Solution:**
```python
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    start_time = time.time()
    logger.info(f"Request received: prompt='{request.prompt[:50]}...'")
    
    # ... generation code ...
    
    latency = (time.time() - start_time) * 1000
    logger.info(f"Request completed: latency={latency:.2f}ms, tokens={num_tokens}")
    return response
```

### Exercise 4: Error Handling
**Solution:**
```python
@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if len(request.prompt) > 1000:
            raise HTTPException(status_code=400, detail="Prompt too long")
        
        # Generation code...
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Exercise 5: Batch Requests
**Solution:**
```python
class BatchGenerationRequest(BaseModel):
    prompts: List[str] = Field(..., min_items=1, max_items=10)

@app.post("/generate_batch")
async def generate_batch(request: BatchGenerationRequest):
    results = []
    for prompt in request.prompts:
        generated = model.generate(prompt=prompt, max_length=50)
        results.append({"prompt": prompt, "generated": generated})
    return {"results": results}
```

---

## Topic 2: Docker

### Exercise 1: Build Basic Image
**Solution:**
```bash
cd 01_basic_serving
docker build -f ../02_docker/Dockerfile.basic -t llm-serving:basic .
docker run -p 8000:8000 llm-serving:basic
```

### Exercise 2: Optimize Image Size
**Solution:**
```dockerfile
# Use multi-stage build
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

### Exercise 3: Multi-stage Build
**Solution:** See `02_docker/Dockerfile.optimized`

### Exercise 4: GPU Support
**Solution:** See `02_docker/Dockerfile.gpu`

### Exercise 5: Docker Compose
**Solution:** See `02_docker/docker-compose.yml`

---

## Topic 3: Kubernetes

### Exercise 1: Basic Deployment
**Solution:**
```bash
kubectl apply -f 03_kubernetes/deployment.yaml
kubectl get pods -n llm-serving
kubectl logs -f deployment/llm-serving -n llm-serving
```

### Exercise 2: Health Checks
**Solution:**
```yaml
# Already in deployment.yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
```

### Exercise 3: Resource Limits
**Solution:**
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Exercise 4: ConfigMap
**Solution:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-config
data:
  model_name: "gpt2"
  max_length: "100"
---
env:
- name: MODEL_NAME
  valueFrom:
    configMapKeyRef:
      name: llm-config
      key: model_name
```

### Exercise 5: Scaling
**Solution:**
```bash
kubectl scale deployment llm-serving --replicas=3 -n llm-serving
kubectl get pods -n llm-serving
```

### Exercise 6: Rolling Update
**Solution:**
```bash
# Update image
kubectl set image deployment/llm-serving \
  llm-serving=llm-serving:v2 -n llm-serving

# Watch rollout
kubectl rollout status deployment/llm-serving -n llm-serving

# Rollback if needed
kubectl rollout undo deployment/llm-serving -n llm-serving
```

---

## Topic 4: Load Testing

### Exercise 1: Baseline Test
**Solution:**
```bash
python measure_latency.py --url http://localhost:8000 --requests 10
```

### Exercise 2: Ramp-up Test
**Solution:**
```bash
# In Locust UI, use "Spawn rate" to gradually increase users
# Or use command line:
locust -f locust_test.py \
  --host=http://localhost:8000 \
  --headless \
  --users 50 \
  --spawn-rate 5 \
  --run-time 5m
```

### Exercise 3: Compare Models
**Solution:**
```bash
# Test model A
python measure_latency.py --url http://localhost:8000 --requests 100 > model_a.txt

# Change model, restart server
# Test model B
python measure_latency.py --url http://localhost:8001 --requests 100 > model_b.txt

# Compare results
diff model_a.txt model_b.txt
```

### Exercise 4: Find Bottleneck
**Solution:**
```python
# Add profiling to app.py
import cProfile
import pstats

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    profiler = cProfile.Profile()
    profiler.enable()
    
    # ... generation code ...
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 slowest functions
```

### Exercise 5: Stress Test
**Solution:**
```bash
# Keep increasing users until system breaks
locust -f locust_test.py \
  --host=http://localhost:8000 \
  --headless \
  --users 100 \
  --spawn-rate 10 \
  --run-time 10m
```

---

## Topic 5: vLLM Serving

### Exercise 1: Compare Performance
**Solution:**
```python
# Test HuggingFace
import time
start = time.time()
# ... HuggingFace generation ...
hf_time = time.time() - start

# Test vLLM
start = time.time()
# ... vLLM generation ...
vllm_time = time.time() - start

print(f"HuggingFace: {hf_time:.2f}s")
print(f"vLLM: {vllm_time:.2f}s")
print(f"Speedup: {hf_time/vllm_time:.2f}x")
```

### Exercise 2: Tune Parameters
**Solution:**
```python
# Test different GPU memory utilization
for util in [0.7, 0.8, 0.9, 0.95]:
    llm = LLM(
        model="gpt2",
        gpu_memory_utilization=util
    )
    # Measure throughput
    # Find optimal value
```

### Exercise 3: Test Continuous Batching
**Solution:**
```python
# Send requests at different rates
import asyncio

async def send_requests(rate_per_second=10):
    for i in range(100):
        # Send request
        await asyncio.sleep(1/rate_per_second)
        # Monitor GPU utilization
```

### Exercise 4: Monitor Metrics
**Solution:**
```python
# Add Prometheus metrics to vLLM server
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('vllm_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('vllm_request_duration_seconds', 'Request duration')

@app.post("/complete")
async def complete(request: CompletionRequest):
    REQUEST_COUNT.inc()
    with REQUEST_DURATION.time():
        # ... generation ...
        pass
```

---

## Topic 6: Autoscaling

### Exercise 1: Basic HPA
**Solution:**
```bash
kubectl apply -f 06_autoscaling/hpa.yaml
kubectl get hpa -n llm-serving
kubectl describe hpa llm-serving-hpa -n llm-serving
```

### Exercise 2: Custom Metrics
**Solution:**
```yaml
# Requires Prometheus Adapter
metrics:
- type: Pods
  pods:
    metric:
      name: http_requests_per_second
    target:
      type: AverageValue
      averageValue: "10"
```

### Exercise 3: Scaling Policies
**Solution:**
```yaml
behavior:
  scaleUp:
    policies:
    - type: Percent
      value: 50  # Increase by 50%
      periodSeconds: 30
  scaleDown:
    stabilizationWindowSeconds: 300  # Wait 5 min
    policies:
    - type: Percent
      value: 25  # Decrease by 25%
```

### Exercise 4: Load Test
**Solution:**
```bash
# Terminal 1: Watch HPA
kubectl get hpa -w -n llm-serving

# Terminal 2: Generate load
locust -f locust_test.py --host=http://<service-url> --users 50

# Terminal 3: Watch pods
kubectl get pods -w -n llm-serving
```

---

## Topic 7: Canary Deployments

### Exercise 1: Basic Canary
**Solution:**
```bash
kubectl apply -f 07_canary_deployments/canary-deployment.yaml
# Traffic splits 90/10 based on replica count
```

### Exercise 2: Gradual Rollout
**Solution:**
```bash
# Phase 1: 10% (already done)
# Phase 2: 25%
kubectl scale deployment llm-serving-canary --replicas=3 -n llm-serving
kubectl scale deployment llm-serving-stable --replicas=7 -n llm-serving

# Phase 3: 50%
kubectl scale deployment llm-serving-canary --replicas=5 -n llm-serving
kubectl scale deployment llm-serving-stable --replicas=5 -n llm-serving

# Phase 4: 100%
kubectl scale deployment llm-serving-canary --replicas=10 -n llm-serving
kubectl delete deployment llm-serving-stable -n llm-serving
```

### Exercise 3: Monitoring
**Solution:**
```promql
# Compare latency
histogram_quantile(0.95, 
  rate(llm_request_duration_seconds_bucket{version="canary"}[5m])
) / 
histogram_quantile(0.95, 
  rate(llm_request_duration_seconds_bucket{version="stable"}[5m])
)
```

### Exercise 4: Rollback
**Solution:**
```bash
# Scale canary to 0
kubectl scale deployment llm-serving-canary --replicas=0 -n llm-serving

# Or update service to only route to stable
```

---

## Topic 8: Monitoring

### Exercise 1: Set Up Monitoring
**Solution:**
```bash
cd 08_monitoring
docker-compose up -d
# Access Grafana at http://localhost:3000
```

### Exercise 2: Create Dashboard
**Solution:**
1. Open Grafana
2. Add Prometheus data source
3. Create new dashboard
4. Add panels with queries from [`08_monitoring/README.md`](08_monitoring/README.md)

### Exercise 3: Set Up Alerts
**Solution:**
```yaml
# In Prometheus alerts.yml
groups:
- name: llm_alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    annotations:
      summary: "P95 latency > 1s"
```

### Exercise 4: GPU Monitoring
**Solution:**
```python
# Use prometheus_exporter.py
# GPU metrics are automatically collected if pynvml is installed
```

---

## Topic 9: Canary Deployments

(Same as Topic 7 - see above)

---

## Topic 10: Model Versioning

### Exercise 1: Create Registry
**Solution:**
```python
# models/registry.py
class ModelRegistry:
    def __init__(self, base_path="models"):
        self.base_path = base_path
    
    def register(self, version, model_path, metadata):
        version_dir = f"{self.base_path}/{version}"
        os.makedirs(version_dir, exist_ok=True)
        # Copy model files
        # Save metadata
```

### Exercise 2: Version Model
**Solution:**
```bash
# Tag model version
mkdir -p models/v1.0.0
cp model.bin models/v1.0.0/
cp tokenizer.json models/v1.0.0/
echo '{"version": "v1.0.0", "created_at": "2024-01-15"}' > models/v1.0.0/metadata.json
```

### Exercise 3: Deploy Version
**Solution:**
```yaml
# Update deployment
env:
- name: MODEL_VERSION
  value: "v1.0.0"
- name: MODEL_PATH
  value: "/models/v1.0.0"
```

### Exercise 4: Rollback
**Solution:**
```bash
# Update to previous version
kubectl set env deployment/llm-serving \
  MODEL_VERSION=v0.9.0 \
  -n llm-serving
```

---

## Topic 11: Drift Detection

### Exercise 1: Set Up Evidently
**Solution:**
```bash
pip install evidently pandas
```

### Exercise 2: Detect Data Drift
**Solution:**
```python
from drift_detector import DriftDetector
import pandas as pd

# Load reference data
reference = pd.read_csv("train_data.csv")

# Initialize detector
detector = DriftDetector(reference_data=reference)

# Load current data
current = pd.read_csv("production_data.csv")

# Detect drift
result = detector.detect_data_drift(current_data=current)
print(result)
```

### Exercise 3: Create Dashboard
**Solution:**
```python
from evidently.ui.dashboards import Dashboard

report = detector.generate_report(current_data)
dashboard = Dashboard("Drift Monitoring")
dashboard.add_report(report)
dashboard.show()
```

### Exercise 4: Set Up Alerts
**Solution:**
```python
if result["drift_detected"]:
    send_email_alert("Data drift detected!")
    send_slack_alert("Drift score: {result['drift_score']}")
```

---

## Topic 12: Triton

### Exercise 1: Deploy Model
**Solution:**
```bash
# Create model repository
mkdir -p model_repository/gpt2/1
# Copy model files
cp model.pt model_repository/gpt2/1/

# Start Triton
docker run --gpus all \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### Exercise 2: Multiple Models
**Solution:**
```
model_repository/
  gpt2/
    config.pbtxt
    1/model.pt
  distilgpt2/
    config.pbtxt
    1/model.pt
```

### Exercise 3: Dynamic Batching
**Solution:**
```protobuf
dynamic_batching {
  max_queue_delay_microseconds: 100000
  preferred_batch_size: [ 4, 8, 16 ]
  max_batch_size: 32
}
```

### Exercise 4: Model Ensemble
**Solution:** See `12_triton/README.md` for ensemble configuration

---

## General Tips

1. **Test locally first**: Always test changes locally before deploying
2. **Monitor metrics**: Watch metrics during exercises
3. **Read errors**: Error messages often contain solutions
4. **Experiment**: Try different values and see what happens
5. **Document**: Note what works and what doesn't

