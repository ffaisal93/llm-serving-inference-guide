# 👋 Welcome! Start Here

This repository teaches you **everything about LLM serving and inference** through hands-on, practical examples.

## 🎯 What You'll Learn

You'll learn how to:
- Serve LLMs in production
- Optimize for performance
- Deploy to Kubernetes
- Monitor and debug systems
- Handle model updates safely
- Detect and handle drift

## 📚 How This Repository is Organized

### Learning Structure
- **Topics are numbered** (01, 02, 03...) - work through them in order
- **Each topic is self-contained** - has its own code, docs, and examples
- **Builds incrementally** - each topic builds on previous concepts

### Key Files

| File | Purpose |
|------|---------|
| [`HOW_TO_START.md`](HOW_TO_START.md) | **👉 START HERE** - Step-by-step guide to begin learning |
| [`LEARNING_PATH.md`](LEARNING_PATH.md) | Overview of all topics and learning approach |
| [`README.md`](README.md) | Repository overview and quick reference |
| `docs/` | Detailed concept explanations |
| `01_basic_serving/` | Your first serving example |
| `02_docker/` | Containerization |
| `03_kubernetes/` | K8s deployment |
| ... | More topics as you progress |

## 🚀 Your First Steps

### 1. Read the Start Guide
```bash
cat HOW_TO_START.md
```
This has everything you need to begin.

### 2. Understand the Fundamentals
```bash
cat docs/llm_inference_fundamentals.md
```
Learn how LLM inference works internally.

### 3. Run Your First Server
```bash
cd 01_basic_serving
pip install -r requirements.txt
python app.py
```

### 4. Test It
```bash
# In another terminal
python test_api.py
```

## 📖 Learning Topics

1. **LLM Inference Fundamentals** - How LLMs work
2. **Basic Serving** - FastAPI + HuggingFace
3. **Containerization** - Docker
4. **Kubernetes** - K8s deployment
5. **Load Testing** - Performance measurement
6. **vLLM Serving** - High-performance serving
7. **Autoscaling** - Scale automatically
8. **Canary Deployments** - Safe rollouts
9. **Monitoring** - Metrics and dashboards
10. **Model Versioning** - Version management
11. **Drift Detection** - Detect issues
12. **Triton** - Multi-model serving

## 💡 Learning Approach

1. **Read** the documentation
2. **Study** the code
3. **Run** the examples
4. **Modify** and experiment
5. **Move** to the next topic

## ✅ Prerequisites

- Python 3.9+
- Basic Python knowledge
- Understanding of REST APIs
- (Optional) Docker and Kubernetes

**Don't have Docker/K8s?** No problem - you can learn the basics without them!

## 🎓 Ready to Start?

**👉 Open [`HOW_TO_START.md`](HOW_TO_START.md) and follow the step-by-step guide!**

---

**Questions?** Check the README.md in each topic directory for detailed explanations.

**Stuck?** Read error messages carefully, check the docs, and experiment with simpler examples first.

**Let's learn!** 🚀

