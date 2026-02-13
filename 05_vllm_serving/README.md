# Topic 5: High-Performance Serving with vLLM

## What You'll Learn

This topic teaches you how to use vLLM for high-performance LLM serving:
- What vLLM is and why it's fast
- Setting up vLLM server
- Understanding continuous batching
- PagedAttention memory optimization
- GPU utilization and throughput optimization

## Why We Need This

### Business Need
- **Cost reduction**: 10x higher throughput = 10x lower cost per request
- **User experience**: Lower latency = better user satisfaction
- **Scalability**: Handle more users with same infrastructure
- **Competitive advantage**: Faster responses than competitors

### Technical Need
- **GPU efficiency**: 80-95% utilization vs 20-40% with basic serving
- **Memory efficiency**: Support longer sequences with same memory
- **Throughput**: 50-200 req/s vs 1-5 req/s with basic serving
- **Production-ready**: Used by major companies in production

### Real-World Impact
Without vLLM:
- ❌ 10x higher infrastructure costs
- ❌ Poor user experience (slow responses)
- ❌ Can't scale to handle traffic
- ❌ Wasted GPU resources (expensive!)

## Industry Use Cases

### 1. **High-Volume API Services**
**Company**: OpenAI, Anthropic, Cohere
**Use Case**:
- Serve millions of requests per day
- Need maximum GPU utilization
- Cost-sensitive at scale

**Example:**
```python
# vLLM handles 1000 req/s vs 10 req/s with basic serving
# Cost: $10,000/month vs $100,000/month
```

### 2. **Real-Time Applications**
**Company**: Chatbots, code completion tools
**Use Case**:
- Sub-second response times required
- Many concurrent users
- Low latency critical

**Example:**
```python
# GitHub Copilot, ChatGPT use continuous batching
# User types → immediate suggestions
```

### 3. **Cost-Optimized ML Platforms**
**Company**: ML infrastructure companies
**Use Case**:
- Serve multiple customers on shared infrastructure
- Maximize GPU utilization = lower costs
- Pass savings to customers

**Example:**
```python
# Shared GPU cluster
# vLLM allows 10x more customers per GPU
```

### 4. **Long Context Windows**
**Company**: Document processing, code analysis
**Use Case**:
- Process long documents (10K+ tokens)
- PagedAttention enables longer sequences
- Memory-efficient

**Example:**
```python
# Process entire codebase (100K tokens)
# Traditional: Out of memory
# vLLM: Works efficiently
```

### 5. **Multi-Tenant Serving**
**Company**: SaaS platforms, ML platforms
**Use Case**:
- Serve multiple models/users simultaneously
- Efficient resource sharing
- Fair resource allocation

**Example:**
```python
# 100 customers, each with different model
# vLLM manages memory efficiently
```

## Industry-Standard Boilerplate Code

### Production vLLM Server (Industry Standard)

```python
"""
Production vLLM server
Used by: OpenAI-compatible APIs, high-throughput serving
"""
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# Configuration from environment (12-factor app)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "2048"))

app = FastAPI(title="vLLM Serving API")

# Initialize vLLM engine
llm_engine = None

@app.on_event("startup")
async def startup():
    global llm_engine
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        dtype="float16",  # FP16 for speed
        trust_remote_code=True,
    )
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible endpoint"""
    from vllm.utils import random_uuid
    
    request_id = random_uuid()
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )
    
    llm_engine.add_request(
        request_id=request_id,
        prompt=request.prompt,
        sampling_params=sampling_params,
    )
    
    final_output = None
    async for request_output in llm_engine.generate(
        request_id=request_id,
        prompt=request.prompt,
        sampling_params=sampling_params,
    ):
        final_output = request_output
    
    return {
        "choices": [{
            "text": final_output.outputs[0].text,
            "finish_reason": final_output.outputs[0].finish_reason
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### vLLM with OpenAI Client (Industry Standard)

```python
"""
Client code using vLLM server
Used by: Applications integrating with LLM APIs
"""
from openai import OpenAI

# vLLM provides OpenAI-compatible API
client = OpenAI(
    base_url="http://vllm-server:8000/v1",
    api_key="dummy"  # vLLM doesn't require real key
)

def generate_text(prompt: str, max_tokens: int = 100) -> str:
    """Generate text using vLLM server"""
    response = client.completions.create(
        model="gpt2",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].text

# Example: High-throughput batch processing
async def process_batch(prompts: list[str]):
    """Process multiple prompts concurrently"""
    import asyncio
    
    tasks = [
        generate_text(prompt) 
        for prompt in prompts
    ]
    return await asyncio.gather(*tasks)
```

## Key Concepts

### Continuous Batching
Traditional batching:
```
Request 1: [████████████████] (waiting)
Request 2: [████████████████] (waiting)
Request 3: [████████████████] (waiting)
→ Process all together
→ Wait for all to finish
```

vLLM continuous batching:
```
Request 1: [████████████] (done, remove)
Request 2: [████████████████] (processing)
Request 3: [████████] (processing)
Request 4: [██] (just added)
→ Process together, remove completed, add new
→ GPU always busy
```

### PagedAttention
- Divides KV cache into fixed-size pages
- Allocates pages on-demand
- Reuses freed pages
- **Result**: Support for longer sequences, less memory waste

## Installation

```bash
# vLLM requires CUDA (GPU)
pip install vllm

# Or with specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

## Running the Examples

### Option 1: Using vLLM's Built-in Server

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model gpt2 \
    --port 8000 \
    --tensor-parallel-size 1
```

### Option 2: Using Our Custom Server

```bash
cd 05_vllm_serving
pip install -r requirements.txt
python vllm_server.py
```

## API Usage

vLLM provides OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require real API key
)

# Chat completion
response = client.chat.completions.create(
    model="gpt2",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Performance Comparison

### Basic Serving (HuggingFace)
- **Throughput**: ~1-5 requests/second
- **Latency**: 100-500ms per request
- **GPU Utilization**: 20-40%

### vLLM Serving
- **Throughput**: ~50-200 requests/second
- **Latency**: 50-200ms per request
- **GPU Utilization**: 80-95%

**Why the difference?**
- Continuous batching keeps GPU busy
- PagedAttention uses memory efficiently
- Optimized CUDA kernels

## Configuration Options

### Model Loading
```python
from vllm import LLM

llm = LLM(
    model="gpt2",
    tensor_parallel_size=1,  # Number of GPUs
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    max_model_len=2048,  # Maximum sequence length
    dtype="float16",  # Use FP16 for speed
)
```

### Generation Parameters
```python
outputs = llm.generate(
    prompts=["Hello"],
    sampling_params={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 100,
    }
)
```

## Monitoring vLLM

vLLM exposes metrics you can monitor:
- Request queue size
- GPU utilization
- Throughput (tokens/second)
- Latency (P50, P95, P99)

See `08_monitoring/` for Grafana dashboards.

## Common Issues

### Out of Memory
- **Problem**: Model too large for GPU
- **Solutions**:
  - Use smaller model
  - Reduce `gpu_memory_utilization`
  - Use quantization (INT8, INT4)
  - Use tensor parallelism (split across GPUs)

### Slow Performance
- **Problem**: Not using GPU
- **Solution**: Make sure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

### Import Errors
- **Problem**: vLLM not installed correctly
- **Solution**: Install with correct CUDA version

## Exercises

1. **Compare Performance**: Run same model with HuggingFace vs vLLM, measure throughput
2. **Tune Parameters**: Experiment with `gpu_memory_utilization`, `max_model_len`
3. **Test Continuous Batching**: Send requests at different rates, observe GPU utilization
4. **Monitor Metrics**: Set up Prometheus to scrape vLLM metrics

## Next Steps

- **Topic 6**: Autoscaling with vLLM
- **Topic 8**: Monitoring vLLM with Grafana
- **Topic 11**: Multi-model serving with Triton

## Further Reading

- [vLLM Documentation](https://docs.vllm.ai/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Continuous Batching Explained](https://www.anyscale.com/blog/continuous-batching-llm-inference)

