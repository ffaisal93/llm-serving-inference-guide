# LLM Serving & Inference: Complete Learning Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.20+-blue.svg)](https://kubernetes.io/)

A comprehensive, hands-on guide to learning LLM serving and inference from fundamentals to production deployment.

## 🎯 What You'll Learn

This repository teaches you everything about serving Large Language Models in production:

- **How LLM inference works** (tokenization, attention, generation)
- **Basic serving** (FastAPI, model loading, endpoints)
- **Containerization** (Docker for ML workloads)
- **Kubernetes deployment** (production infrastructure)
- **Performance optimization** (vLLM, batching, GPU utilization)
- **Scaling** (autoscaling, load balancing)
- **Production practices** (monitoring, versioning, drift detection)

## 📁 Repository Structure

```
mlops_serving/
├── LEARNING_PATH.md          # Start here! Complete learning guide
├── docs/                     # Detailed concept explanations
│   ├── llm_inference_fundamentals.md
│   ├── serving_architectures.md
│   └── optimization_techniques.md
├── 01_basic_serving/         # FastAPI + HuggingFace serving
├── 02_docker/                # Containerization
├── 03_kubernetes/            # K8s deployment
├── 04_load_testing/          # Performance testing
├── 05_vllm_serving/          # High-performance serving
├── 06_autoscaling/           # K8s autoscaling
├── 07_canary_deployments/    # Gradual rollouts
├── 08_monitoring/            # Prometheus + Grafana
├── 09_model_versioning/      # Version management
├── 10_drift_detection/       # Evidently integration
└── 11_triton/                # Multi-model serving
```

## 🚀 Quick Start

**👉 START HERE**: Read `HOW_TO_START.md` for a complete step-by-step guide!

### Quick Commands

```bash
# 1. Set up environment
cd 01_basic_serving
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Test it (in another terminal)
python test_api.py
# OR visit http://localhost:8000/docs for interactive API docs
```

## 📚 Learning Approach

This is a **hands-on, topic-based** learning path:

1. **Each topic is self-contained** with code, docs, and exercises
2. **Build incrementally** - each topic builds on previous concepts
3. **Learn by doing** - run code, modify it, break it, fix it
4. **Focus on understanding** - not just copying code

## 🎓 Prerequisites

- Python 3.9+
- Basic Python knowledge
- Understanding of REST APIs
- Docker (for containerization topics)
- Kubernetes cluster (minikube/kind for K8s topics)
- GPU (optional, helpful for vLLM)

## 📖 Topics Covered

### Core Concepts
- LLM architecture and inference pipeline
- Tokenization and vocabulary
- Attention mechanisms and KV caching
- Memory and computation requirements

### Serving Basics
- Model loading and initialization
- API endpoint design
- Request/response handling
- Error handling and validation

### Infrastructure
- Docker containerization
- Kubernetes deployment
- Health checks and probes
- Resource management

### Performance
- Load testing and benchmarking
- Latency vs throughput
- vLLM optimization
- GPU utilization

### Production
- Autoscaling strategies
- Canary deployments
- Monitoring and observability
- Model versioning
- Drift detection

## 🔧 Technologies

- **FastAPI**: Web framework
- **HuggingFace Transformers**: Model loading
- **vLLM**: High-performance inference
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Locust**: Load testing
- **Prometheus**: Metrics
- **Grafana**: Visualization
- **Evidently**: Drift detection
- **Triton**: Multi-model serving

## 📝 How to Use This Repository

1. **Start with LEARNING_PATH.md** - Understand the structure
2. **Read the fundamentals** - docs/llm_inference_fundamentals.md
3. **Work through topics in order** - Each numbered topic builds on the last
4. **Experiment and modify** - Don't just run code, change it and learn
5. **Read the code comments** - They explain the "why" not just the "what"
6. **Check solutions** - See EXERCISE_SOLUTIONS.md for exercise answers
7. **Prepare for interviews** - Review INTERVIEW_QA.md for common questions

## 🎯 Learning Goals

By completing this repository, you'll be able to:

- Understand how LLM inference works internally
- Build production-ready serving APIs
- Deploy models to Kubernetes
- Optimize for performance and cost
- Monitor and debug serving systems
- Handle model updates and drift

## 🤝 Contributing

This is a learning repository. Feel free to:
- Add more examples
- Improve documentation
- Fix bugs
- Share your learnings

## 📄 License

MIT License - Feel free to use this for learning and teaching.

---

## 📖 Getting Started

1. **New to this?** → Read `HOW_TO_START.md` for step-by-step instructions
2. **Want overview?** → Read `LEARNING_PATH.md` to see all topics
3. **Ready to code?** → Start with `01_basic_serving/`
4. **Need help?** → Check `EXERCISE_SOLUTIONS.md` for exercise answers
5. **Interview prep?** → Review `INTERVIEW_QA.md` for common questions

**Ready to start?** Open `HOW_TO_START.md` and begin your journey! 🚀

## 📚 Additional Resources

- **Exercise Solutions**: `EXERCISE_SOLUTIONS.md` - Answers to all exercises
- **Interview Q&A**: `INTERVIEW_QA.md` - 23+ interview questions with detailed answers
- **Quick Reference**: `QUICK_REFERENCE.md` - Quick commands and tips
- **Complete Topics**: `COMPLETE_TOPICS.md` - Overview of all 12 topics
- **Industry Boilerplate**: Each topic README includes industry-standard code and use cases

## 🏭 Industry Focus

Each topic now includes:
- **Why We Need This**: Business and technical justification
- **Industry Use Cases**: Real-world applications (5+ examples per topic)
- **Boilerplate Code**: Production-ready, industry-standard implementations
- **Company Examples**: How major companies use these technologies

Topics updated with industry content:
- ✅ Topic 1: Basic Serving (Customer support, content generation)
- ✅ Topic 2: Docker (Multi-cloud, CI/CD, edge deployment)
- ✅ Topic 3: Kubernetes (Large-scale platforms, enterprise ML)
- ✅ Topic 5: vLLM (High-throughput APIs, cost optimization)
- ✅ Topic 8: Monitoring (Production observability, incident response)
