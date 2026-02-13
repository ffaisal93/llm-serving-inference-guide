"""
Prometheus Metrics Exporter for LLM Serving

This module adds Prometheus metrics to your FastAPI application.
It tracks:
- Request count and rate
- Latency (histogram)
- Error rate
- Active requests
- GPU metrics (if available)
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute
import time
import logging

logger = logging.getLogger(__name__)

# Define metrics
REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'llm_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]  # Buckets for latency
)

ACTIVE_REQUESTS = Gauge(
    'llm_active_requests',
    'Number of active requests',
    ['endpoint']
)

GENERATION_TOKENS = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated',
    ['model_name']
)

GENERATION_DURATION = Histogram(
    'llm_generation_duration_seconds',
    'Text generation duration in seconds',
    ['model_name'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# GPU metrics (if available)
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    
    GPU_UTILIZATION = Gauge(
        'llm_gpu_utilization_percent',
        'GPU utilization percentage',
        ['gpu_id']
    )
    
    GPU_MEMORY_USED = Gauge(
        'llm_gpu_memory_used_bytes',
        'GPU memory used in bytes',
        ['gpu_id']
    )
    
    GPU_MEMORY_TOTAL = Gauge(
        'llm_gpu_memory_total_bytes',
        'GPU memory total in bytes',
        ['gpu_id']
    )
    
    GPU_TEMPERATURE = Gauge(
        'llm_gpu_temperature_celsius',
        'GPU temperature in Celsius',
        ['gpu_id']
    )
    
except ImportError:
    logger.warning("pynvml not available. GPU metrics will not be collected.")
    GPU_AVAILABLE = False
except Exception as e:
    logger.warning(f"GPU metrics unavailable: {e}")
    GPU_AVAILABLE = False


def update_gpu_metrics():
    """Update GPU metrics if available"""
    if not GPU_AVAILABLE:
        return
    
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            GPU_UTILIZATION.labels(gpu_id=str(i)).set(util.gpu)
            
            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            GPU_MEMORY_USED.labels(gpu_id=str(i)).set(mem_info.used)
            GPU_MEMORY_TOTAL.labels(gpu_id=str(i)).set(mem_info.total)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            GPU_TEMPERATURE.labels(gpu_id=str(i)).set(temp)
    except Exception as e:
        logger.error(f"Error updating GPU metrics: {e}")


def add_prometheus_middleware(app: FastAPI):
    """
    Add Prometheus metrics middleware to FastAPI app.
    
    Usage:
        app = FastAPI()
        add_prometheus_middleware(app)
    """
    
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Middleware to collect metrics"""
        # Get endpoint name
        endpoint = request.url.path
        
        # Skip metrics endpoint itself
        if endpoint == "/metrics":
            return await call_next(request)
        
        # Track active requests
        ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()
        
        # Measure duration
        start_time = time.time()
        status_code = 200
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            status_code = 500
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status=str(status_code)
            ).inc()
            
            ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()
            
            # Update GPU metrics
            update_gpu_metrics()
    
    @app.get("/metrics")
    async def metrics():
        """
        Prometheus metrics endpoint.
        
        Prometheus will scrape this endpoint to collect metrics.
        """
        # Update GPU metrics before returning
        update_gpu_metrics()
        
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


def record_generation_metrics(model_name: str, num_tokens: int, duration: float):
    """
    Record metrics for text generation.
    
    Call this after generating text to record:
    - Number of tokens generated
    - Generation duration
    """
    GENERATION_TOKENS.labels(model_name=model_name).inc(num_tokens)
    GENERATION_DURATION.labels(model_name=model_name).observe(duration)


# Example usage in your FastAPI app:
"""
from fastapi import FastAPI
from prometheus_exporter import add_prometheus_middleware, record_generation_metrics

app = FastAPI()
add_prometheus_middleware(app)

@app.post("/generate")
async def generate(request: GenerationRequest):
    start = time.time()
    result = model.generate(request.prompt)
    duration = time.time() - start
    
    # Record metrics
    record_generation_metrics(
        model_name="gpt2",
        num_tokens=len(result.tokens),
        duration=duration
    )
    
    return result
"""

