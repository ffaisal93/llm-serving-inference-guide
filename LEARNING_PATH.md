# LLM Serving & Inference: Complete Learning Guide

## 🎯 How to Use This Guide

This guide is organized by **learning topics**, not time periods. Work through each topic at your own pace. Each topic builds on the previous one, but you can also jump to specific areas you want to learn.

## 📚 Learning Topics (In Order)

### 1. **LLM Inference Fundamentals** 
**What you'll learn**: How LLMs actually work under the hood
- Tokenization and vocabulary
- Forward pass through transformer layers
- Autoregressive generation
- Attention mechanism and KV caching
- Memory and computation requirements

**Start here**: `docs/llm_inference_fundamentals.md`

### 2. **Basic Model Serving**
**What you'll learn**: Serve a model with a simple API
- Loading models from HuggingFace
- Building FastAPI endpoints
- Request/response handling
- Basic error handling

**Practice**: `01_basic_serving/`

### 3. **Containerization**
**What you'll learn**: Package your serving application
- Docker basics for ML models
- Multi-stage builds
- Handling large model files
- Environment configuration

**Practice**: `02_docker/`

### 4. **Kubernetes Deployment**
**What you'll learn**: Deploy to production infrastructure
- K8s manifests for ML workloads
- Health checks and probes
- Resource limits and requests
- ConfigMaps and Secrets

**Practice**: `03_kubernetes/`

### 5. **Load Testing & Performance**
**What you'll learn**: Measure and understand performance
- Latency vs throughput
- Load testing with Locust
- Performance profiling
- Identifying bottlenecks

**Practice**: `04_load_testing/`

### 6. **High-Performance Serving (vLLM)**
**What you'll learn**: Optimize for production throughput
- vLLM architecture
- Continuous batching
- PagedAttention
- GPU optimization

**Practice**: `05_vllm_serving/`

### 7. **Autoscaling**
**What you'll learn**: Scale based on demand
- Horizontal Pod Autoscaling (HPA)
- Request-based scaling
- Concurrency metrics
- Scaling strategies

**Practice**: `06_autoscaling/`

### 8. **Monitoring & Observability**
**What you'll learn**: Track your serving system
- Prometheus metrics
- Grafana dashboards
- GPU monitoring
- Logging and tracing

**Practice**: `08_monitoring/`

### 9. **Canary Deployments**
**What you'll learn**: Safely roll out model updates
- Traffic splitting
- A/B testing models
- Gradual rollouts
- Rollback procedures

**Practice**: `09_canary_deployments/`

### 10. **Model Versioning**
**What you'll learn**: Manage multiple model versions
- Version management strategies
- Model registry
- Rollback procedures
- A/B testing infrastructure

**Practice**: `10_model_versioning/`

### 11. **Drift Detection**
**What you'll learn**: Detect when models degrade
- Data drift detection
- Concept drift
- Evidently AI integration
- Alerting on anomalies

**Practice**: `11_drift_detection/`

### 12. **Multi-Model Serving (Triton)**
**What you'll learn**: Serve multiple models efficiently
- NVIDIA Triton Inference Server
- Dynamic batching
- Model ensembles
- Multi-framework support

**Practice**: `12_triton/`

## 🚀 Quick Start Guide

### Step 1: Understand the Basics
Read `docs/llm_inference_fundamentals.md` to understand how LLM inference works.

### Step 2: Set Up Your Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install base dependencies
pip install -r requirements.txt
```

### Step 3: Start with Basic Serving
```bash
cd 01_basic_serving
python app.py
# Test it: curl http://localhost:8000/health
```

### Step 4: Progress Through Topics
Work through each numbered topic in order. Each includes:
- **README.md**: Explanation of concepts
- **Code examples**: Working implementations
- **Exercises**: Hands-on practice

## 🎓 Learning Approach

### For Each Topic:
1. **Read the documentation** - Understand the concepts
2. **Study the code** - See how it's implemented
3. **Run the examples** - Get hands-on experience
4. **Modify and experiment** - Break things, fix them, learn
5. **Move to next topic** - Build on what you learned

### Tips:
- **Don't rush**: Understanding > Speed
- **Experiment**: Change parameters, break things, learn why
- **Read error messages**: They teach you a lot
- **Use the docs**: Each topic has detailed explanations

## 📖 Prerequisites

### Required:
- Python 3.9+
- Basic Python knowledge
- Understanding of REST APIs
- Docker basics

### Helpful but not required:
- Kubernetes experience
- ML/AI background
- GPU access (CPU works for learning)

## 🔧 Technology Stack

You'll learn these tools:
- **FastAPI**: Web framework
- **HuggingFace Transformers**: Model loading
- **vLLM**: High-performance inference
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Locust**: Load testing
- **Prometheus/Grafana**: Monitoring
- **Evidently**: Drift detection
- **Triton**: Multi-model serving

## ❓ Common Questions

**Q: Do I need a GPU?**
A: Not for the basics. GPU helps with vLLM and production workloads, but you can learn on CPU.

**Q: How long will this take?**
A: Depends on your pace. Each topic can take a few hours to a few days. Focus on understanding, not speed.

**Q: Can I skip topics?**
A: The basics (1-5) should be done in order. Advanced topics (6-12) can be done based on interest.

**Q: What if I get stuck?**
A: Check the docs, read error messages carefully, experiment with simpler examples first.

## 🎯 Learning Goals

By the end, you'll be able to:
- ✅ Serve LLMs in production
- ✅ Optimize for performance
- ✅ Deploy to Kubernetes
- ✅ Monitor and debug serving systems
- ✅ Handle model updates safely
- ✅ Detect and handle model drift

Let's start learning! 🚀

