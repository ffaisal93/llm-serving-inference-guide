# Quick Reference: vLLM, Latency, and Grafana

## 🚀 Quick Start Guide

### 1. vLLM Serving

#### Install vLLM
```bash
# Requires CUDA/GPU
pip install vllm
```

#### Start vLLM Server
```bash
cd 05_vllm_serving
pip install -r requirements.txt
python vllm_server.py
```

#### Or use vLLM's built-in server
```bash
python -m vllm.entrypoints.openai.api_server \
    --model gpt2 \
    --port 8000
```

#### Test vLLM API
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="gpt2",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)

print(response.choices[0].message.content)
```

---

### 2. Measure Latency

#### Option A: Using Locust (Interactive)
```bash
cd 04_load_testing
pip install -r requirements.txt

# Start your server first
# Then run Locust
locust -f locust_test.py --host=http://localhost:8000

# Open http://localhost:8089 in browser
```

#### Option B: Using Python Script (Headless)
```bash
cd 04_load_testing
python measure_latency.py --url http://localhost:8000 --requests 100
```

#### What You Get
- **P50 (Median)**: 50% of requests faster
- **P95**: 95% of requests faster  
- **P99**: 99% of requests faster
- **Throughput**: Requests per second

#### Compare Endpoints
```bash
python measure_latency.py --compare
# Compares basic serving vs vLLM
```

---

### 3. Grafana Monitoring

#### Start Monitoring Stack
```bash
cd 08_monitoring
docker-compose up -d
```

This starts:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

#### Add Metrics to Your App
```python
from prometheus_exporter import add_prometheus_middleware

app = FastAPI()
add_prometheus_middleware(app)  # That's it!
```

Your app now exposes `/metrics` endpoint.

#### Configure Prometheus
Edit `prometheus.yml` to scrape your app:
```yaml
scrape_configs:
  - job_name: 'llm-serving'
    static_configs:
      - targets: ['localhost:8000']
```

#### Create Grafana Dashboard

1. Open Grafana: http://localhost:3000
2. Login: admin/admin
3. Add data source: Prometheus (http://prometheus:9090)
4. Create dashboard with these queries:

**Request Rate:**
```promql
rate(llm_requests_total[5m])
```

**Latency (P95):**
```promql
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))
```

**Error Rate:**
```promql
rate(llm_requests_total{status="500"}[5m]) / rate(llm_requests_total[5m])
```

**GPU Utilization:**
```promql
llm_gpu_utilization_percent
```

---

## 📊 Complete Workflow Example

### Step 1: Start vLLM Server
```bash
cd 05_vllm_serving
python vllm_server.py
# Server running on http://localhost:8000
```

### Step 2: Measure Latency
```bash
cd 04_load_testing
python measure_latency.py --url http://localhost:8000 --requests 100
```

Output:
```
P50:  150ms
P95:  450ms
P99:  800ms
Throughput: 25.3 req/s
```

### Step 3: Start Monitoring
```bash
cd 08_monitoring
docker-compose up -d
```

### Step 4: View Metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Step 5: Load Test with Monitoring
```bash
# Terminal 1: Watch Grafana
open http://localhost:3000

# Terminal 2: Run load test
cd 04_load_testing
locust -f locust_test.py --host=http://localhost:8000 --headless --users 10 --spawn-rate 2
```

---

## 🔍 Key Metrics to Monitor

### Application Metrics
- `llm_requests_total` - Total requests
- `llm_request_duration_seconds` - Request latency
- `llm_active_requests` - Currently processing
- `llm_tokens_generated_total` - Tokens generated

### System Metrics
- `llm_gpu_utilization_percent` - GPU usage
- `llm_gpu_memory_used_bytes` - GPU memory
- `llm_gpu_temperature_celsius` - GPU temperature

### Important Queries

**Requests per second:**
```promql
rate(llm_requests_total[5m])
```

**95th percentile latency:**
```promql
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))
```

**Error percentage:**
```promql
rate(llm_requests_total{status="500"}[5m]) / rate(llm_requests_total[5m]) * 100
```

**GPU memory usage:**
```promql
llm_gpu_memory_used_bytes / llm_gpu_memory_total_bytes * 100
```

---

## 🎯 Performance Targets

### Good Performance
- **P50 latency**: < 200ms
- **P95 latency**: < 500ms
- **P99 latency**: < 1s
- **Error rate**: < 0.1%
- **GPU utilization**: 70-90%

### Warning Signs
- **P99 > 2s**: System struggling
- **Error rate > 1%**: Problems occurring
- **GPU utilization < 50%**: Underutilized
- **GPU utilization > 95%**: Overloaded

---

## 🐛 Troubleshooting

### vLLM won't start
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Install correct CUDA version
- Use smaller model for testing

### High latency
- Check GPU utilization
- Reduce batch size
- Use smaller model
- Check for bottlenecks (CPU, memory, I/O)

### No metrics in Grafana
- Check Prometheus is scraping: http://localhost:9090/targets
- Verify `/metrics` endpoint works: `curl http://localhost:8000/metrics`
- Check Prometheus config points to correct host

### GPU metrics missing
- Install pynvml: `pip install pynvml`
- Check GPU is accessible
- Verify NVIDIA drivers installed

---

## 📚 Learn More

- **vLLM**: `05_vllm_serving/README.md`
- **Latency**: `04_load_testing/README.md`
- **Monitoring**: `08_monitoring/README.md`

---

**Quick tip**: Start with basic serving, measure latency, then optimize with vLLM, and monitor with Grafana!

