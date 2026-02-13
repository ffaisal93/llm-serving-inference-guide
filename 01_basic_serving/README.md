# Topic 1: Basic LLM Serving

## What You'll Learn

This topic teaches you the fundamentals of serving an LLM:
- How to load a pre-trained model from HuggingFace
- Building a FastAPI endpoint for inference
- Understanding the inference pipeline
- Handling requests and responses
- Basic error handling

## Why We Need This

### Business Need
Companies need to expose LLM capabilities as APIs to:
- **Integrate AI into applications**: Chatbots, content generation, code completion
- **Serve multiple clients**: Web apps, mobile apps, internal tools
- **Scale independently**: Separate model serving from application logic
- **Enable monetization**: API-based business models

### Technical Need
- **Separation of concerns**: Model serving separate from application code
- **Resource management**: Dedicated servers for compute-intensive inference
- **Standardization**: REST APIs are universal, language-agnostic
- **Testing**: Easy to test model independently

### Real-World Impact
Without proper serving infrastructure:
- ❌ Models can't be used in production applications
- ❌ No way to integrate AI into existing systems
- ❌ Difficult to scale and maintain
- ❌ Hard to version and update models

## Industry Use Cases

### 1. **Customer Support Chatbots**
**Company**: E-commerce, SaaS platforms
**Use Case**: 
- Customer asks question → API call → LLM generates response
- Handles 24/7 support, reduces human agent workload

**Example Request:**
```python
POST /generate
{
  "prompt": "Customer: How do I return an item?\nAssistant:",
  "max_length": 100
}
```

### 2. **Content Generation**
**Company**: Marketing agencies, content platforms
**Use Case**:
- Generate blog posts, social media content, product descriptions
- API called from CMS or marketing tools

**Example Request:**
```python
POST /generate
{
  "prompt": "Write a product description for a wireless headphone:",
  "temperature": 0.8,  # More creative
  "max_length": 200
}
```

### 3. **Code Completion & Assistance**
**Company**: GitHub Copilot, IDEs
**Use Case**:
- Developer types code → API suggests completions
- Real-time code generation in editor

**Example Request:**
```python
POST /generate
{
  "prompt": "def calculate_total(items):\n    ",
  "max_length": 50,
  "temperature": 0.2  # More deterministic
}
```

### 4. **Translation Services**
**Company**: Google Translate, DeepL
**Use Case**:
- Translate text between languages
- API integrated into websites, apps

**Example Request:**
```python
POST /generate
{
  "prompt": "Translate to French: Hello, how are you?",
  "max_length": 50
}
```

### 5. **Sentiment Analysis**
**Company**: Social media platforms, review sites
**Use Case**:
- Analyze customer reviews, social media posts
- Real-time sentiment detection

**Example Request:**
```python
POST /generate
{
  "prompt": "Sentiment: This product is amazing!",
  "max_length": 10
}
```

## Industry-Standard Boilerplate Code

### Complete FastAPI Serving Application

```python
"""
Industry-standard LLM serving application
Used by: OpenAI API, Anthropic Claude, HuggingFace Inference API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import logging
from contextlib import asynccontextmanager

# Configure logging (industry standard)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model (loaded once at startup)
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown"""
    global model, tokenizer
    
    logger.info(f"Loading model on {device}...")
    model_name = "gpt2"  # In production: from environment variable
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield  # Application runs here
    
    # Cleanup
    model = None
    tokenizer = None
    logger.info("Model unloaded")


app = FastAPI(
    title="LLM Serving API",
    description="Production-ready LLM inference endpoint",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response Models (Industry standard structure)
class GenerationRequest(BaseModel):
    """Standard request format (similar to OpenAI API)"""
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(50, ge=1, le=500)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    stop: list[str] = Field(default_factory=list)


class GenerationResponse(BaseModel):
    """Standard response format"""
    text: str
    model: str
    usage: dict
    latency_ms: float


@app.get("/health")
async def health_check():
    """Health check for load balancers and K8s probes"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.post("/v1/completions", response_model=GenerationResponse)
async def generate_completion(request: GenerationRequest):
    """
    Main generation endpoint
    Industry standard: /v1/completions (OpenAI-compatible)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Tokenize
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        num_tokens = len(outputs[0]) - inputs['input_ids'].shape[1]
        
        logger.info(
            f"Generated {num_tokens} tokens in {latency_ms:.2f}ms"
        )
        
        return GenerationResponse(
            text=generated_text,
            model="gpt2",
            usage={
                "prompt_tokens": inputs['input_ids'].shape[1],
                "completion_tokens": num_tokens,
                "total_tokens": len(outputs[0])
            },
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

### Usage Example (Industry Standard)

```python
# Client code (how applications use the API)
import requests

def generate_text(prompt: str, max_tokens: int = 50) -> str:
    """
    Call LLM serving API
    Used by: Web applications, mobile apps, microservices
    """
    response = requests.post(
        "http://llm-api:8000/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["text"]


# Example: Customer support chatbot
def handle_customer_query(query: str) -> str:
    prompt = f"Customer: {query}\nAssistant:"
    return generate_text(prompt, max_tokens=100)


# Example: Content generation
def generate_product_description(product_name: str) -> str:
    prompt = f"Write a compelling product description for {product_name}:"
    return generate_text(prompt, max_tokens=200, temperature=0.8)
```

## Concepts Explained

### Model Loading
When you load a model, you're:
1. **Downloading weights** (if not cached) - These are the learned parameters
2. **Loading into memory** - CPU RAM or GPU VRAM
3. **Initializing tokenizer** - Converts text ↔ tokens
4. **Setting up device** - CPU or CUDA (GPU)

**Why this matters**: Model loading is expensive. You do it once at startup, not per request.

### Tokenization
- **Text → Tokens**: "Hello" → [15496]
- **Tokens → Text**: [15496] → "Hello"
- **Special tokens**: `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`

### Inference Process
1. Tokenize input prompt
2. Run forward pass through model (autoregressive)
3. Sample next token
4. Repeat until max_length or stop token
5. Decode tokens back to text

### Autoregressive Generation
LLMs generate **one token at a time**:
- Input: "The weather is"
- Step 1: Generate "nice"
- Step 2: Generate "today"
- Step 3: Generate "."
- Output: "The weather is nice today."

Each step uses the previous tokens as context.

## Code Structure

```
01_basic_serving/
├── README.md           # This file
├── app.py              # FastAPI application
├── model_loader.py     # Model loading logic
├── requirements.txt    # Dependencies
└── test_api.py        # Simple test script
```

## Running the Code

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python app.py
```

### 3. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_length": 50,
    "temperature": 0.7
  }'
```

### 4. View API Docs
Open http://localhost:8000/docs in your browser

## Key Code Sections

### Model Loading (`model_loader.py`)
```python
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Move to device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

### Generation (`app.py`)
```python
# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
outputs = model.generate(
    **inputs,
    max_length=max_length,
    temperature=temperature
)

# Decode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Understanding the Output

When you call the API, you get:
- `generated_text`: The complete generated text
- `prompt`: Your input (echoed back)
- `num_tokens`: Number of tokens generated
- `latency_ms`: Time taken in milliseconds
- `model_name`: Which model was used

## Common Issues

### Out of Memory
- **Problem**: Model too large for available RAM/VRAM
- **Solution**: Use a smaller model (gpt2-small) or reduce batch size

### Slow Inference
- **Problem**: Running on CPU
- **Solution**: Use GPU if available, or use a smaller model

### Import Errors
- **Problem**: Missing dependencies
- **Solution**: `pip install -r requirements.txt`

## Exercises

1. **Change the model**: Try different HuggingFace models (gpt2, distilgpt2, etc.)
2. **Modify parameters**: Experiment with temperature, top_p, max_length
3. **Add logging**: Log every request with timing information
4. **Error handling**: Add try/except for different error cases
5. **Batch requests**: Modify to handle multiple prompts at once

## Next Steps

Once you understand this, move to:
- **Topic 2**: Containerization with Docker
- **Topic 3**: Kubernetes deployment

## Further Reading

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

