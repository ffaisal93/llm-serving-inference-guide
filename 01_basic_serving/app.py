"""
Basic LLM Serving with FastAPI

This demonstrates fundamental LLM serving concepts:
- Model loading and initialization
- Tokenization and text generation
- Request/response handling
- Basic error handling
- Health checks for Kubernetes
"""

import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_loader import LLMModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance (loaded once at startup)
model: Optional[LLMModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads the model at startup and cleans up at shutdown.
    
    Model loading is expensive (loading weights into GPU memory),
    so we do it once at startup, not on every request.
    """
    global model
    
    logger.info("🚀 Starting up: Loading LLM model...")
    try:
        # Initialize model (this loads weights into memory/GPU)
        # Start with small model for learning - change to larger models later
        model = LLMModel(
            model_name="gpt2",  # Small, fast model for learning
            device="cpu"  # Change to "cuda" if you have GPU
        )
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down: Cleaning up model...")
    model = None


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="LLM Serving API - Basic Serving",
    description="Basic LLM serving endpoint for learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class TextGenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input text prompt", min_length=1, max_length=1000)
    max_length: int = Field(50, description="Maximum number of tokens to generate", ge=1, le=200)
    temperature: float = Field(1.0, description="Sampling temperature (higher = more random)", ge=0.1, le=2.0)
    top_p: float = Field(0.9, description="Nucleus sampling parameter", ge=0.1, le=1.0)
    num_return_sequences: int = Field(1, description="Number of sequences to generate", ge=1, le=5)


class TextGenerationResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str
    prompt: str
    num_tokens: int
    latency_ms: float
    model_name: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for Kubernetes liveness/readiness probes.
    
    This endpoint:
    - Checks if the model is loaded
    - Returns quickly (no model inference)
    - Used by K8s to determine if pod is healthy
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=model.model_name if model else None
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LLM Serving API - Basic Serving",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "docs": "/docs"
        },
        "model_loaded": model is not None
    }


@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """
    Main text generation endpoint.
    
    This endpoint demonstrates:
    1. Request validation (via Pydantic)
    2. Model inference
    3. Latency measurement
    4. Error handling
    
    The inference process:
    1. Tokenize input prompt
    2. Run model forward pass (autoregressive generation)
    3. Decode tokens back to text
    4. Return response
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Measure latency
    start_time = time.time()
    
    try:
        # Generate text using the model
        # This is where the actual LLM inference happens
        generated_text = model.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=request.num_return_sequences
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Count tokens (approximate)
        num_tokens = len(model.tokenizer.encode(generated_text))
        
        logger.info(
            f"Generated {num_tokens} tokens in {latency_ms:.2f}ms "
            f"(prompt: {request.prompt[:50]}...)"
        )
        
        return TextGenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            num_tokens=num_tokens,
            latency_ms=latency_ms,
            model_name=model.model_name
        )
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """
    Basic metrics endpoint (for Prometheus scraping later).
    
    In production, you'd use a proper metrics library like prometheus_client.
    This is a simplified version for learning.
    """
    if model is None:
        return {"error": "Model not loaded"}
    
    info = model.get_model_info()
    return {
        "model_name": info["model_name"],
        "model_loaded": True,
        "device": model.device,
        "num_parameters": info["num_parameters"],
        "vocab_size": info["vocab_size"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

