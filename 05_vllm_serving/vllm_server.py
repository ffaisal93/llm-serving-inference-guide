"""
vLLM High-Performance Serving Server

This demonstrates vLLM serving with:
- Continuous batching
- PagedAttention memory optimization
- OpenAI-compatible API
- Performance monitoring
"""

import argparse
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    import asyncio
except ImportError:
    print("vLLM not installed. Install with: pip install vllm")
    print("Note: vLLM requires CUDA (GPU)")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="vLLM Serving API",
    description="High-performance LLM serving with vLLM",
    version="1.0.0"
)


# Request/Response Models
class CompletionRequest(BaseModel):
    """Request for text completion"""
    prompt: str = Field(..., description="Input prompt", min_length=1)
    max_tokens: int = Field(100, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Nucleus sampling", ge=0.0, le=1.0)
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Stream responses")


class CompletionResponse(BaseModel):
    """Response for text completion"""
    text: str
    prompt: str
    num_tokens: int
    finish_reason: str


# Global vLLM engine
llm_engine: Optional[AsyncLLMEngine] = None
llm: Optional[LLM] = None  # For synchronous API


def initialize_vllm(
    model_name: str = "gpt2",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    dtype: str = "float16"
):
    """
    Initialize vLLM engine.
    
    Args:
        model_name: HuggingFace model identifier
        tensor_parallel_size: Number of GPUs to use (1 = single GPU)
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        max_model_len: Maximum sequence length (None = auto)
        dtype: Model precision (float16, float32, etc.)
    """
    global llm_engine, llm
    
    logger.info(f"Initializing vLLM with model: {model_name}")
    
    try:
        # For async API (OpenAI-compatible)
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
        )
        llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # For sync API (simpler, for learning)
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
        )
        
        logger.info("✅ vLLM engine initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize vLLM: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize vLLM on server startup"""
    initialize_vllm(
        model_name="gpt2",  # Start with small model
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if llm_engine is not None else "unhealthy",
        "engine_initialized": llm_engine is not None
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "vLLM Serving API",
        "endpoints": {
            "health": "/health",
            "complete": "/complete",
            "complete_async": "/complete_async",
            "docs": "/docs"
        }
    }


@app.post("/complete", response_model=CompletionResponse)
async def complete_text(request: CompletionRequest):
    """
    Synchronous text completion endpoint.
    
    This uses vLLM's synchronous API which is simpler to understand.
    For production, use the async endpoint for better performance.
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="vLLM engine not initialized")
    
    try:
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )
        
        # Generate (synchronous)
        # vLLM handles batching internally
        outputs = llm.generate(
            prompts=[request.prompt],
            sampling_params=sampling_params,
        )
        
        # Extract result
        output = outputs[0]
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason
        
        # Count tokens
        num_tokens = len(output.outputs[0].token_ids)
        
        return CompletionResponse(
            text=generated_text,
            prompt=request.prompt,
            num_tokens=num_tokens,
            finish_reason=finish_reason
        )
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/complete_async")
async def complete_text_async(request: CompletionRequest):
    """
    Asynchronous text completion endpoint.
    
    This uses vLLM's async API which is more efficient for production.
    It supports continuous batching and better resource utilization.
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM engine not initialized")
    
    try:
        from vllm.utils import random_uuid
        
        # Create request ID
        request_id = random_uuid()
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )
        
        # Add request to engine (non-blocking)
        llm_engine.add_request(
            request_id=request_id,
            prompt=request.prompt,
            sampling_params=sampling_params,
        )
        
        # Wait for result (this is where continuous batching happens)
        final_output = None
        async for request_output in llm_engine.generate(
            request_id=request_id,
            prompt=request.prompt,
            sampling_params=sampling_params,
        ):
            final_output = request_output
        
        if final_output is None:
            raise HTTPException(status_code=500, detail="No output generated")
        
        # Extract result
        generated_text = final_output.outputs[0].text
        finish_reason = final_output.outputs[0].finish_reason
        num_tokens = len(final_output.outputs[0].token_ids)
        
        return CompletionResponse(
            text=generated_text,
            prompt=request.prompt,
            num_tokens=num_tokens,
            finish_reason=finish_reason
        )
        
    except Exception as e:
        logger.error(f"Error during async generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """
    Basic metrics endpoint.
    
    In production, vLLM exposes more detailed metrics via Prometheus.
    This is a simplified version for learning.
    """
    if llm_engine is None:
        return {"error": "Engine not initialized"}
    
    # Get engine stats
    stats = llm_engine.get_stats()
    
    return {
        "engine_stats": stats,
        "model_loaded": llm_engine is not None
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)

