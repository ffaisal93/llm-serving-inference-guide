# Topic 8: Monitoring & Observability with Grafana

## What You'll Learn

This topic teaches you how to:
- Set up Prometheus for metrics collection
- Create Grafana dashboards for visualization
- Monitor LLM serving performance
- Track GPU utilization
- Set up alerts for anomalies

## Why We Need This

### Business Need
- **SLA compliance**: Meet 99.9% uptime SLAs
- **Cost optimization**: Identify expensive operations
- **User experience**: Detect issues before users complain
- **Compliance**: Audit trails for regulated industries

### Technical Need
- **Debugging**: Understand why system is slow/failing
- **Capacity planning**: Know when to scale
- **Performance optimization**: Identify bottlenecks
- **Incident response**: Quick detection and resolution

### Real-World Impact
Without monitoring:
- ❌ Issues discovered by users (too late!)
- ❌ Can't debug production problems
- ❌ Don't know when to scale
- ❌ No visibility into system health

## Industry Use Cases

### 1. **Production ML Platforms**
**Company**: OpenAI, Anthropic, HuggingFace
**Use Case**:
- Monitor API performance 24/7
- Alert on latency spikes, errors
- Track cost per request

**Example:**
```promql
# Alert if P95 latency > 1s
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m])) > 1
```

### 2. **Enterprise ML Infrastructure**
**Company**: Banks, healthcare, finance
**Use Case**:
- Compliance monitoring
- Audit trails
- Performance SLAs

**Example:**
```promql
# Track all requests for audit
count(llm_requests_total)
```

### 3. **Cost Management**
**Company**: All companies with ML infrastructure
**Use Case**:
- Track GPU costs
- Identify expensive operations
- Optimize resource usage

**Example:**
```promql
# Cost per request
gpu_cost_per_hour / rate(llm_requests_total[1h])
```

### 4. **Incident Response**
**Company**: All production systems
**Use Case**:
- Detect issues immediately
- Alert on-call engineers
- Quick root cause analysis

**Example:**
```yaml
# Alert on high error rate
alert: HighErrorRate
expr: rate(llm_requests_total{status="500"}[5m]) > 0.05
```

### 5. **Performance Optimization**
**Company**: All companies
**Use Case**:
- Identify slow endpoints
- Find bottlenecks
- Optimize based on data

**Example:**
```promql
# Find slowest endpoints
topk(10, histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m])))
```

## Industry-Standard Boilerplate Code

### Complete Monitoring Setup (Industry Standard)

```python
"""
Production monitoring setup
Used by: All production ML systems
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import time

# Metrics (industry standard names)
REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status', 'model_version']
)

REQUEST_DURATION = Histogram(
    'llm_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ACTIVE_REQUESTS = Gauge(
    'llm_active_requests',
    'Active requests',
    ['endpoint']
)

TOKENS_GENERATED = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated',
    ['model_name']
)

GPU_UTILIZATION = Gauge(
    'llm_gpu_utilization_percent',
    'GPU utilization',
    ['gpu_id']
)

app = FastAPI()

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect metrics for every request"""
    endpoint = request.url.path
    method = request.method
    
    # Skip metrics endpoint
    if endpoint == "/metrics":
        return await call_next(request)
    
    # Track active requests
    ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()
    
    # Measure duration
    start = time.time()
    status_code = 200
    
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.time() - start
        
        # Record metrics
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code),
            model_version="v1.0.0"
        ).inc()
        
        ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

### Prometheus Alerts (Industry Standard)

```yaml
# alerts.yml
# Used by: Production monitoring systems
groups:
- name: llm_serving_alerts
  interval: 30s
  rules:
  # High latency alert
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "P95 latency > 1s"
      description: "Latency is {{ $value }}s"
  
  # High error rate
  - alert: HighErrorRate
    expr: rate(llm_requests_total{status="500"}[5m]) / rate(llm_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Error rate > 5%"
  
  # GPU high utilization
  - alert: GPUHighUtilization
    expr: llm_gpu_utilization_percent > 95
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "GPU utilization very high"
  
  # Service down
  - alert: ServiceDown
    expr: up{job="llm-serving"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "LLM serving is down"
```

### Grafana Dashboard JSON (Industry Standard)

```json
{
  "dashboard": {
    "title": "LLM Serving Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(llm_requests_total[5m])",
          "legendFormat": "{{endpoint}}"
        }]
      },
      {
        "title": "Latency (P95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))",
          "legendFormat": "P95"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(llm_requests_total{status=\"500\"}[5m]) / rate(llm_requests_total[5m])",
          "legendFormat": "Error Rate"
        }]
      },
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "llm_gpu_utilization_percent",
          "legendFormat": "GPU {{gpu_id}}"
        }]
      }
    ]
  }
}
```

## Key Concepts

### Monitoring Stack

1. **Prometheus**: Metrics collection and storage
2. **Grafana**: Visualization and dashboards
3. **Exporters**: Collect metrics from applications
4. **Alertmanager**: Handle alerts

### Key Metrics to Monitor

#### Application Metrics
- **Request rate**: Requests per second
- **Latency**: P50, P95, P99 response times
- **Error rate**: Failed requests percentage
- **Queue size**: Pending requests

#### System Metrics
- **GPU utilization**: How much GPU is being used
- **GPU memory**: Memory usage
- **CPU usage**: CPU utilization
- **Memory usage**: RAM usage

#### Model Metrics
- **Tokens generated**: Tokens per second
- **Batch size**: Current batch size
- **Model version**: Which model is running

## Architecture

```
[LLM Server] → [Prometheus Exporter] → [Prometheus] → [Grafana]
     ↓
[GPU Metrics] → [Node Exporter] → [Prometheus]
```

## Setup

### 1. Install Prometheus

```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*

# Or use Docker
docker run -d -p 9090:9090 prom/prometheus
```

### 2. Install Grafana

```bash
# Docker
docker run -d -p 3000:3000 grafana/grafana

# Or install locally
# See: https://grafana.com/docs/grafana/latest/setup-grafana/installation/
```

### 3. Set Up Metrics Export

Add Prometheus metrics to your serving application (see `prometheus_exporter.py`).

## Running the Stack

### Option 1: Docker Compose

```bash
cd 08_monitoring
docker-compose up -d
```

This starts:
- Prometheus on http://localhost:9090
- Grafana on http://localhost:3000
- Node Exporter (system metrics)

### Option 2: Manual Setup

1. Start Prometheus: `./prometheus --config.file=prometheus.yml`
2. Start Grafana: `./grafana-server`
3. Configure data source in Grafana

## Grafana Dashboards

### Pre-built Dashboards

1. **LLM Serving Overview**: Request rate, latency, errors
2. **GPU Monitoring**: GPU utilization, memory, temperature
3. **System Metrics**: CPU, memory, disk, network
4. **Model Performance**: Throughput, tokens/second

### Creating Custom Dashboards

1. Open Grafana (http://localhost:3000)
2. Login (default: admin/admin)
3. Add Prometheus data source
4. Create new dashboard
5. Add panels for metrics you care about

## Key Metrics Queries

### Request Rate
```promql
rate(http_requests_total[5m])
```

### Latency (P95)
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Error Rate
```promql
rate(http_requests_total{status="error"}[5m]) / rate(http_requests_total[5m])
```

### GPU Utilization
```promql
nvidia_gpu_utilization_gpu
```

### GPU Memory
```promql
nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes
```

## Alerts

### Example Alert Rules

```yaml
groups:
  - name: llm_serving
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        annotations:
          summary: "High latency detected"
      
      - alert: HighErrorRate
        expr: rate(http_requests_total{status="error"}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate"
      
      - alert: GPUHighUtilization
        expr: nvidia_gpu_utilization_gpu > 95
        for: 10m
        annotations:
          summary: "GPU utilization very high"
```

## Exercises

1. **Set up monitoring**: Deploy Prometheus and Grafana
2. **Create dashboard**: Build a dashboard for your metrics
3. **Set up alerts**: Configure alerts for high latency
4. **Monitor GPU**: Add GPU metrics to dashboard
5. **Compare performance**: Monitor before/after optimization

## Next Steps

- **Topic 6**: Use metrics for autoscaling
- **Topic 10**: Set up drift detection alerts
- **Topic 9**: Monitor model version performance

## Further Reading

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Guide](https://prometheus.io/docs/prometheus/latest/querying/basics/)

