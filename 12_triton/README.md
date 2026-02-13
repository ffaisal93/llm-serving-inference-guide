# Topic 12: Multi-Model Serving with Triton

## What You'll Learn

This topic teaches you how to:
- Deploy models with NVIDIA Triton Inference Server
- Serve multiple models simultaneously
- Use dynamic batching
- Optimize for different frameworks (PyTorch, TensorFlow, ONNX)
- Create model ensembles
- Monitor Triton performance

## Why Triton?

### Benefits
- **Multi-framework**: PyTorch, TensorFlow, ONNX, TensorRT
- **Dynamic batching**: Automatic request batching
- **Model ensembles**: Chain multiple models
- **High performance**: Optimized inference
- **Production-ready**: Used by major companies

### Use Cases
- **Multiple models**: Serve different models on same server
- **Model pipelines**: Chain models together
- **A/B testing**: Easy model switching
- **Resource efficiency**: Share GPU across models

## Triton Architecture

### Components
- **Triton Server**: Main inference server
- **Model Repository**: Storage for models
- **Model Config**: Configuration per model
- **Scheduler**: Request scheduling and batching

### Request Flow
```
Client → Triton Server → Model Backend → GPU → Response
```

## Installation

### Docker (Recommended)
```bash
docker pull nvcr.io/nvidia/tritonserver:23.10-py3
```

### Local Installation
```bash
# See Triton documentation for your platform
```

## Model Repository Structure

```
model_repository/
  model1/
    config.pbtxt
    1/
      model.pt
  model2/
    config.pbtxt
    1/
      model.onnx
```

## Model Configuration

### Basic Config
```protobuf
name: "gpt2"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ -1, 50257 ]
  }
]
```

### Dynamic Batching
```protobuf
dynamic_batching {
  max_queue_delay_microseconds: 100
  preferred_batch_size: [ 4, 8 ]
  max_batch_size: 16
}
```

## Starting Triton Server

### Basic
```bash
tritonserver --model-repository=/path/to/models
```

### With GPU
```bash
docker run --gpus all \
  -v /path/to/models:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

## Client API

### Python Client
```python
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("localhost:8000")

# Prepare input
inputs = [httpclient.InferInput("input_ids", [1, 10], "INT64")]
inputs[0].set_data_from_numpy(input_ids)

# Infer
result = client.infer("gpt2", inputs)
output = result.as_numpy("output")
```

### REST API
```bash
curl -X POST http://localhost:8000/v2/models/gpt2/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input_ids",
      "shape": [1, 10],
      "datatype": "INT64",
      "data": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    }]
  }'
```

## Dynamic Batching

### How It Works
1. Requests arrive at Triton
2. Triton queues requests
3. When batch ready (size or timeout), process together
4. Return responses

### Configuration
```protobuf
dynamic_batching {
  max_queue_delay_microseconds: 100000  # 100ms
  preferred_batch_size: [ 4, 8, 16 ]
  max_batch_size: 32
}
```

## Model Ensembles

### Chain Models
```protobuf
name: "pipeline"
platform: "ensemble"
input [
  { name: "input", data_type: TYPE_STRING }
]
output [
  { name: "output", data_type: TYPE_STRING }
]
ensemble_scheduling {
  step [
    {
      model_name: "tokenizer"
      model_version: -1
      input_map { key: "input" value: "input" }
      output_map { key: "output" value: "tokens" }
    },
    {
      model_name: "llm"
      model_version: -1
      input_map { key: "tokens" value: "input_ids" }
      output_map { key: "output" value: "generated" }
    }
  ]
}
```

## Monitoring

### Metrics Endpoint
```bash
curl http://localhost:8000/metrics
```

### Key Metrics
- Request count
- Inference latency
- Queue size
- GPU utilization
- Batch size

## Best Practices

1. **Organize models**: Clear repository structure
2. **Optimize configs**: Tune batching parameters
3. **Monitor performance**: Track metrics
4. **Version models**: Use versioning in repository
5. **Test thoroughly**: Validate before production

## Common Issues

### Model Not Loading
- Check model repository path
- Verify model format
- Check config.pbtxt syntax

### Low Throughput
- Increase batch size
- Tune dynamic batching
- Check GPU utilization

### Out of Memory
- Reduce batch size
- Use smaller models
- Add more GPUs

## Exercises

1. **Deploy model**: Set up Triton with one model
2. **Multiple models**: Serve multiple models
3. **Dynamic batching**: Configure and test batching
4. **Model ensemble**: Chain models together
5. **Monitor**: Set up monitoring

## Next Steps

- **Topic 8**: Monitor Triton with Grafana
- **Topic 6**: Autoscale Triton deployments
- **Topic 9**: Version models in Triton

## Further Reading

- [Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md)
- [Performance Tuning](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md)

