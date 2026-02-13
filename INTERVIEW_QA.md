# LLM Serving & Inference Interview Q&A

Comprehensive interview questions and answers for LLM serving, inference, and MLOps roles.

## Table of Contents
1. [LLM Inference Fundamentals](#llm-inference-fundamentals)
2. [Model Serving](#model-serving)
3. [Performance Optimization](#performance-optimization)
4. [Kubernetes & Deployment](#kubernetes--deployment)
5. [Monitoring & Observability](#monitoring--observability)
6. [Production Best Practices](#production-best-practices)
7. [System Design](#system-design)

---

## LLM Inference Fundamentals

### Q1: Explain how LLM inference works step-by-step.

**Answer:**
1. **Tokenization**: Input text is converted to token IDs using the model's tokenizer
2. **Embedding**: Token IDs are converted to dense vectors (embeddings)
3. **Forward Pass**: 
   - Input passes through transformer layers
   - Each layer applies self-attention and feed-forward networks
   - Attention mechanism computes relationships between tokens
4. **Output Projection**: Final layer projects to vocabulary size
5. **Sampling**: Next token is sampled from probability distribution
6. **Autoregressive Generation**: Process repeats with new token until stop condition

**Key Points:**
- Inference is autoregressive (one token at a time)
- KV cache stores attention key-values to avoid recomputation
- Each token generation requires a full forward pass

---

### Q2: What is KV caching and why is it important?

**Answer:**
KV (Key-Value) caching stores the attention key and value matrices for previously processed tokens.

**How it works:**
- First token: Compute full Q, K, V for all tokens
- Subsequent tokens: Only compute Q for new token, reuse cached K, V

**Benefits:**
- **Speed**: Avoids recomputing attention for previous tokens
- **Memory trade-off**: Uses more memory but much faster
- **Critical for performance**: Without it, each token would recompute all previous tokens

**Example:**
- Without cache: Generate 100 tokens = 100 forward passes, each processing all 100 tokens
- With cache: Generate 100 tokens = 100 forward passes, but each only processes 1 new token

---

### Q3: What is the difference between training and inference?

**Answer:**

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Mode** | Training mode (gradients computed) | Evaluation mode (no gradients) |
| **Batch** | Large batches (32-128) | Small batches or single requests |
| **Memory** | Stores activations for backprop | Only forward pass needed |
| **Speed** | Slower (backprop overhead) | Faster (forward only) |
| **Optimization** | Gradient descent | Sampling/decoding strategies |
| **Hardware** | Multiple GPUs common | Single GPU often sufficient |

**Key Differences:**
- Training: Updates weights, needs gradients
- Inference: Uses fixed weights, generates predictions

---

### Q4: Explain attention mechanism in the context of inference.

**Answer:**
Attention determines which tokens to focus on when generating the next token.

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Components:**
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What information do I have?"
- **V (Value)**: "What is the actual information?"

**In Inference:**
- Computes relationships between current token and all previous tokens
- Allows model to "attend" to relevant context
- KV cache stores K and V to avoid recomputation

**Complexity:**
- O(n²) where n is sequence length
- This is why longer sequences are slower and more memory-intensive

---

## Model Serving

### Q5: How would you design an LLM serving API?

**Answer:**

**API Design:**
```python
POST /v1/completions
{
  "prompt": "The future of AI is",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Key Components:**
1. **Request Validation**: Validate inputs (prompt length, parameters)
2. **Model Loading**: Load model once at startup (not per request)
3. **Tokenization**: Convert text to tokens
4. **Generation**: Run inference with parameters
5. **Response**: Return generated text + metadata (latency, tokens)

**Considerations:**
- **Async support**: Handle concurrent requests
- **Streaming**: Support streaming responses
- **Error handling**: Graceful error responses
- **Rate limiting**: Prevent abuse
- **Health checks**: For Kubernetes probes

---

### Q6: What are the differences between HuggingFace Transformers and vLLM?

**Answer:**

| Feature | HuggingFace | vLLM |
|---------|-------------|------|
| **Batching** | Static batching | Continuous batching |
| **Throughput** | 1-5 req/s | 50-200 req/s |
| **Memory** | Standard | PagedAttention (efficient) |
| **GPU Utilization** | 20-40% | 80-95% |
| **Ease of Use** | Very easy | Moderate |
| **Flexibility** | High | Moderate |

**When to use HuggingFace:**
- Development and prototyping
- Small-scale deployments
- Need maximum flexibility

**When to use vLLM:**
- Production high-throughput
- Need maximum GPU utilization
- Many concurrent requests

---

### Q7: How does continuous batching work in vLLM?

**Answer:**
Continuous batching allows adding/removing requests dynamically during batch processing.

**Traditional Batching:**
1. Wait for batch to fill (e.g., 8 requests)
2. Process entire batch
3. Wait for all to complete
4. Start next batch

**Continuous Batching:**
1. Start processing batch
2. Add new requests as they arrive
3. Remove completed requests
4. Continue processing remaining requests
5. GPU always busy

**Benefits:**
- Higher GPU utilization
- Lower latency (no waiting for batch to fill)
- Better throughput

**Example:**
```
Time 0: [Req1, Req2, Req3] → Processing
Time 1: [Req1, Req2, Req3, Req4] → Req4 added
Time 2: [Req2, Req3, Req4] → Req1 completed, removed
```

---

## Performance Optimization

### Q8: How would you optimize LLM inference latency?

**Answer:**

**1. Model Optimization:**
- **Quantization**: FP16, INT8, INT4 (trade accuracy for speed)
- **Model pruning**: Remove unnecessary weights
- **Knowledge distillation**: Use smaller model

**2. Inference Optimization:**
- **KV caching**: Cache attention key-values
- **Batching**: Process multiple requests together
- **Continuous batching**: vLLM's approach

**3. Hardware:**
- **GPU**: Use GPU instead of CPU (10-100x faster)
- **Tensor cores**: Use specialized hardware
- **Model parallelism**: Split across multiple GPUs

**4. System:**
- **Pre-warming**: Load model before first request
- **Connection pooling**: Reuse connections
- **CDN**: Cache responses when appropriate

**5. Architecture:**
- **Async processing**: Don't block on I/O
- **Request queuing**: Handle bursts gracefully
- **Load balancing**: Distribute requests

---

### Q9: What is PagedAttention and why does it matter?

**Answer:**
PagedAttention is vLLM's memory management technique for KV cache.

**Problem it solves:**
- Traditional KV cache: Fixed-size blocks, memory fragmentation
- Wastes memory when sequences have different lengths
- Can't support very long sequences efficiently

**How it works:**
1. Divide KV cache into fixed-size "pages" (like OS memory pages)
2. Allocate pages on-demand as tokens are generated
3. Free pages when sequences complete
4. Reuse freed pages for new sequences

**Benefits:**
- **Efficient memory**: No fragmentation
- **Longer sequences**: Support sequences up to model max
- **Higher throughput**: More sequences fit in memory

**Analogy:**
Like virtual memory in operating systems - pages allocated as needed.

---

### Q10: Explain the trade-offs between latency and throughput.

**Answer:**

**Latency**: Time for one request (ms)
**Throughput**: Requests processed per second

**Trade-offs:**

1. **Batching:**
   - Larger batches → Higher throughput, higher latency
   - Smaller batches → Lower latency, lower throughput

2. **Model Size:**
   - Larger model → Higher quality, higher latency
   - Smaller model → Lower latency, potentially lower quality

3. **Quantization:**
   - Lower precision → Faster, lower memory, potential accuracy loss
   - Higher precision → Slower, more memory, better accuracy

4. **Hardware:**
   - More GPUs → Higher throughput, higher cost
   - Fewer GPUs → Lower cost, lower throughput

**Optimization Strategy:**
- **Low latency**: Small batches, optimized model, fast hardware
- **High throughput**: Large batches, continuous batching, multiple GPUs

---

## Kubernetes & Deployment

### Q11: How would you deploy an LLM model to Kubernetes?

**Answer:**

**1. Containerize:**
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

**2. Create Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: llm-serving
        image: llm-serving:v1.0
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
```

**3. Create Service:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-serving
spec:
  selector:
    app: llm-serving
  ports:
  - port: 80
    targetPort: 8000
```

**4. Deploy:**
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

**Key Considerations:**
- Resource limits (GPU, memory)
- Health checks
- Rolling updates
- ConfigMaps for configuration
- Secrets for API keys

---

### Q12: How does Horizontal Pod Autoscaling (HPA) work?

**Answer:**
HPA automatically scales the number of pod replicas based on metrics.

**How it works:**
1. HPA checks metrics every 15 seconds (default)
2. Compares current metric value to target
3. Calculates desired number of replicas
4. Updates deployment replica count
5. Kubernetes creates/destroys pods

**Example:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Metrics:**
- **CPU/Memory**: Built-in resource metrics
- **Custom metrics**: Requests per second, latency (requires Prometheus Adapter)

**Scaling Behavior:**
- **Scale up**: When metric > target (add pods)
- **Scale down**: When metric < target (remove pods)
- **Stabilization window**: Prevents flapping

---

### Q13: Explain canary deployments for model updates.

**Answer:**
Canary deployment gradually rolls out new model version to a small percentage of traffic.

**Process:**
1. **Deploy canary**: Deploy new version alongside stable
2. **Route traffic**: Split traffic (e.g., 90% stable, 10% canary)
3. **Monitor**: Compare metrics (latency, errors, quality)
4. **Gradual increase**: If good, increase canary traffic (25%, 50%, 100%)
5. **Rollback**: If issues, route all traffic back to stable

**Implementation:**
- **Kubernetes**: Multiple deployments with different replica counts
- **Service Mesh (Istio)**: Fine-grained traffic control
- **Application-level**: Route based on headers/parameters

**Benefits:**
- **Risk reduction**: Test on small traffic first
- **Quick rollback**: Revert if issues detected
- **A/B testing**: Compare model versions

**Monitoring:**
- Compare latency (P50, P95, P99)
- Error rates
- Business metrics (user satisfaction)

---

## Monitoring & Observability

### Q14: What metrics should you monitor for LLM serving?

**Answer:**

**Application Metrics:**
- **Request rate**: Requests per second
- **Latency**: P50, P95, P99 response times
- **Error rate**: Failed requests percentage
- **Queue size**: Pending requests
- **Throughput**: Tokens per second

**Model Metrics:**
- **Generation time**: Time to generate tokens
- **Tokens generated**: Average tokens per request
- **Model version**: Which model is running

**System Metrics:**
- **GPU utilization**: GPU usage percentage
- **GPU memory**: Used/total memory
- **CPU usage**: CPU utilization
- **Memory usage**: RAM usage

**Business Metrics:**
- **Cost per request**: Compute cost
- **User satisfaction**: Quality metrics
- **API usage**: Requests by endpoint

**Key Dashboards:**
1. **Performance**: Latency, throughput, errors
2. **Resource**: GPU, CPU, memory usage
3. **Model**: Version performance comparison

---

### Q15: How would you detect model drift in production?

**Answer:**

**Types of Drift:**
1. **Data drift**: Input distribution changes
2. **Concept drift**: Input-output relationship changes
3. **Prediction drift**: Output distribution changes

**Detection Methods:**

**1. Statistical Tests:**
- **PSI (Population Stability Index)**: Compare distributions
- **Kolmogorov-Smirnov**: Test distribution differences
- **Chi-square**: Test categorical distributions

**2. Monitoring Tools:**
- **Evidently AI**: Open-source drift detection
- **Prometheus**: Custom metrics
- **Custom scripts**: Compare reference vs current

**3. Implementation:**
```python
from evidently import Report, DataDriftTable

report = Report(metrics=[DataDriftTable()])
report.run(
    reference_data=train_data,
    current_data=production_data
)

if report.get_metric(DataDriftTable()).drift_detected:
    alert("Data drift detected!")
```

**4. Alerting:**
- Set thresholds (e.g., PSI > 0.2)
- Monitor continuously
- Alert on drift detection
- Investigate causes

**Actions:**
- **Retrain model**: If drift significant
- **Investigate**: Understand why drift occurred
- **Update baseline**: Update reference data if appropriate

---

## Production Best Practices

### Q16: What are the key considerations for production LLM serving?

**Answer:**

**1. Performance:**
- **Latency**: P95 < 500ms for most use cases
- **Throughput**: Handle expected load
- **Scalability**: Auto-scale based on demand

**2. Reliability:**
- **Health checks**: Liveness and readiness probes
- **Error handling**: Graceful degradation
- **Circuit breakers**: Prevent cascade failures
- **Retries**: With exponential backoff

**3. Monitoring:**
- **Metrics**: Comprehensive observability
- **Logging**: Structured logging
- **Alerting**: Proactive issue detection
- **Tracing**: Request tracing

**4. Security:**
- **Authentication**: API keys, OAuth
- **Rate limiting**: Prevent abuse
- **Input validation**: Sanitize inputs
- **Secrets management**: Secure API keys

**5. Cost:**
- **Resource optimization**: Right-size instances
- **Auto-scaling**: Scale down when not needed
- **Model optimization**: Use efficient models

**6. Model Management:**
- **Versioning**: Track model versions
- **A/B testing**: Compare model performance
- **Rollback**: Quick revert capability
- **Drift detection**: Monitor model degradation

---

### Q17: How would you handle a sudden spike in traffic?

**Answer:**

**Immediate Actions:**
1. **Auto-scaling**: HPA should scale up automatically
2. **Load balancing**: Distribute across pods
3. **Queue management**: Queue requests if needed
4. **Rate limiting**: Protect backend from overload

**Prevention:**
1. **Capacity planning**: Understand max capacity
2. **Load testing**: Test under expected load
3. **Auto-scaling**: Configure HPA properly
4. **Circuit breakers**: Prevent cascade failures

**Monitoring:**
- Watch pod count
- Monitor latency (P95, P99)
- Check error rates
- GPU utilization

**If Overwhelmed:**
1. **Degrade gracefully**: Return cached responses
2. **Rate limit**: Reject excess requests
3. **Scale manually**: If auto-scaling insufficient
4. **Add capacity**: More nodes/GPUs

**Post-Incident:**
- Analyze what happened
- Improve auto-scaling config
- Increase baseline capacity if needed
- Document learnings

---

## System Design

### Q18: Design a system to serve LLMs at scale.

**Answer:**

**Architecture:**
```
[Load Balancer] 
    ↓
[API Gateway] (Rate limiting, Auth)
    ↓
[Kubernetes Cluster]
    ├── [LLM Serving Pods] (vLLM)
    ├── [Monitoring] (Prometheus, Grafana)
    └── [Model Registry] (S3/GCS)
```

**Components:**

**1. Load Balancer:**
- Distribute traffic
- Health checks
- SSL termination

**2. API Gateway:**
- Authentication/Authorization
- Rate limiting
- Request routing
- API versioning

**3. Serving Layer:**
- **vLLM servers**: High-performance inference
- **Auto-scaling**: HPA based on metrics
- **GPU nodes**: Dedicated GPU instances

**4. Model Storage:**
- **Model registry**: S3/GCS for model files
- **Versioning**: Track model versions
- **Caching**: Cache models on nodes

**5. Monitoring:**
- **Metrics**: Prometheus
- **Dashboards**: Grafana
- **Logging**: Centralized logging
- **Alerting**: PagerDuty/Slack

**6. Data Pipeline:**
- **Request logging**: Store inputs/outputs
- **Drift detection**: Monitor data drift
- **A/B testing**: Compare model versions

**Scaling Strategy:**
- **Horizontal**: Add more pods (HPA)
- **Vertical**: Larger GPUs for bigger models
- **Multi-region**: Geographic distribution

**Key Metrics:**
- Latency (P50, P95, P99)
- Throughput (req/s)
- Error rate
- GPU utilization
- Cost per request

---

### Q19: How would you implement model versioning and rollback?

**Answer:**

**Versioning Strategy:**
1. **Semantic versioning**: v1.0.0, v1.1.0, v2.0.0
2. **Model registry**: Store models with metadata
3. **Metadata tracking**: Training date, metrics, dataset

**Implementation:**

**1. Model Registry:**
```
models/
  v1.0.0/
    model.bin
    tokenizer.json
    metadata.json
  v1.1.0/
    model.bin
    tokenizer.json
    metadata.json
```

**2. Deployment:**
```yaml
env:
- name: MODEL_VERSION
  value: "v1.0.0"
- name: MODEL_PATH
  value: "/models/v1.0.0"
```

**3. Rollback:**
```bash
# Update to previous version
kubectl set env deployment/llm-serving \
  MODEL_VERSION=v0.9.0

# Or use canary deployment
# Route traffic back to stable version
```

**4. API:**
```python
GET /api/v1/models/versions
POST /api/v1/models/rollback
{
  "target_version": "v1.0.0"
}
```

**Best Practices:**
- **Test before deploy**: Validate new version
- **Gradual rollout**: Use canary deployment
- **Monitor**: Track version performance
- **Document**: Changelog for each version

---

### Q20: Explain how you would optimize costs for LLM serving.

**Answer:**

**Cost Components:**
1. **Compute**: GPU/CPU instances
2. **Storage**: Model storage
3. **Network**: Data transfer
4. **Monitoring**: Observability tools

**Optimization Strategies:**

**1. Right-sizing:**
- Use appropriate instance types
- Don't over-provision
- Match workload to instance

**2. Auto-scaling:**
- Scale down during low traffic
- Scale up only when needed
- Use spot instances for non-critical workloads

**3. Model Optimization:**
- **Quantization**: INT8/INT4 (smaller, faster)
- **Pruning**: Remove unnecessary weights
- **Distillation**: Use smaller models

**4. Caching:**
- Cache model weights
- Cache common responses (if applicable)
- Use CDN for static content

**5. Batch Processing:**
- Batch requests when possible
- Use continuous batching (vLLM)
- Higher GPU utilization = lower cost per request

**6. Monitoring:**
- Track cost per request
- Identify expensive operations
- Optimize based on data

**7. Reserved Instances:**
- Commit to usage for discounts
- Use for predictable workloads

**Example:**
- Before: 10 GPUs, 50% utilization = $5000/month
- After: 5 GPUs, 90% utilization (with batching) = $2500/month
- **Savings: 50%**

---

## Additional Quick Questions

### Q21: What is the difference between batch size and sequence length?

**Answer:**
- **Batch size**: Number of requests processed together
- **Sequence length**: Number of tokens in a single request

**Example:**
- Batch size = 8: Process 8 requests simultaneously
- Sequence length = 512: Each request has up to 512 tokens

**Impact:**
- Larger batch size → Higher throughput, more memory
- Longer sequence length → More computation, more memory (KV cache)

---

### Q22: How does quantization affect model performance?

**Answer:**
Quantization reduces model precision to save memory and speed up inference.

**Types:**
- **FP32 → FP16**: 2x smaller, 2x faster, minimal accuracy loss
- **FP16 → INT8**: 2x smaller, 2x faster, small accuracy loss
- **INT8 → INT4**: 2x smaller, 2x faster, larger accuracy loss

**Trade-offs:**
- **Pros**: Faster inference, less memory, lower cost
- **Cons**: Potential accuracy loss, may need calibration

**When to use:**
- Production when speed/cost matters
- After validating accuracy is acceptable
- For edge deployment (limited resources)

---

### Q23: What is the difference between model parallelism and data parallelism?

**Answer:**

**Data Parallelism:**
- Same model on multiple GPUs
- Different data on each GPU
- Used in training (gradient sync)
- **Inference**: Not commonly used (each request needs full model)

**Model Parallelism:**
- Split model across multiple GPUs
- Each GPU holds part of model
- Used for large models that don't fit on one GPU
- **Inference**: Common for very large models (70B+)

**Example:**
- **Data parallel**: 8 GPUs, each running full GPT-2
- **Model parallel**: 8 GPUs, each holding 1/8 of GPT-3

---

## Tips for Interviews

1. **Be specific**: Use numbers and examples
2. **Show trade-offs**: Understand pros/cons
3. **Think system-wide**: Consider all components
4. **Ask clarifying questions**: Understand requirements
5. **Draw diagrams**: Visualize architecture
6. **Discuss monitoring**: Always mention observability
7. **Talk about failures**: How to handle edge cases

---

## Resources

- This repository's documentation
- vLLM documentation
- Kubernetes documentation
- Prometheus/Grafana guides
- Evidently AI documentation

Good luck with your interviews! 🚀

