"""
Example: Integrating Prometheus Metrics with FastAPI

This shows how to add Prometheus metrics to your LLM serving application.
"""

from fastapi import FastAPI
from prometheus_exporter import add_prometheus_middleware, record_generation_metrics
import time

app = FastAPI()

# Add Prometheus middleware
# This automatically tracks:
# - Request count
# - Request duration
# - Active requests
add_prometheus_middleware(app)


@app.post("/generate")
async def generate(request: dict):
    """
    Example endpoint that records generation metrics.
    """
    prompt = request.get("prompt", "")
    model_name = "gpt2"  # Your model name
    
    # Simulate generation
    start_time = time.time()
    
    # Your actual generation code here
    # result = model.generate(prompt)
    # For demo, just simulate
    time.sleep(0.1)  # Simulate generation time
    generated_text = "Generated text here"
    num_tokens = 50  # Actual token count
    
    duration = time.time() - start_time
    
    # Record metrics
    record_generation_metrics(
        model_name=model_name,
        num_tokens=num_tokens,
        duration=duration
    )
    
    return {
        "generated_text": generated_text,
        "tokens": num_tokens,
        "duration": duration
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

