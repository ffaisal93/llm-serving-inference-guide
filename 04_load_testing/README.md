# Topic 4: Load Testing & Latency Measurement

## What You'll Learn

This topic teaches you how to:
- Measure inference latency (P50, P95, P99)
- Test throughput (requests/second)
- Identify performance bottlenecks
- Use Locust for load testing
- Analyze performance metrics

## Key Concepts

### Latency Metrics

**Latency** = Time from request sent to response received

Common percentiles:
- **P50 (Median)**: 50% of requests faster than this
- **P95**: 95% of requests faster than this
- **P99**: 99% of requests faster than this

**Why P95/P99 matter**: P50 might be 100ms, but P99 could be 5s. Users notice the slow requests!

### Throughput

**Throughput** = Requests processed per second

- **Single request**: Measure latency
- **Multiple requests**: Measure throughput
- **Trade-off**: Higher throughput often means higher latency

### Load Testing Types

1. **Baseline**: Single request, measure latency
2. **Ramp-up**: Gradually increase load
3. **Sustained**: Constant load for extended period
4. **Spike**: Sudden increase in load
5. **Stress**: Keep increasing until system breaks

## Tools

### Locust
- Python-based load testing
- Write tests in Python
- Web UI for monitoring
- Real-time statistics

### Alternatives
- **Apache Bench (ab)**: Simple, command-line
- **wrk**: High-performance, Lua scripting
- **k6**: JavaScript-based, modern

## Installation

```bash
pip install locust requests
```

## Running Load Tests

### Option 1: Basic Locust Test

```bash
cd 04_load_testing
locust -f locust_test.py --host=http://localhost:8000
```

Then open http://localhost:8089 in your browser.

### Option 2: Headless Mode (No UI)

```bash
locust -f locust_test.py \
    --host=http://localhost:8000 \
    --headless \
    --users 10 \
    --spawn-rate 2 \
    --run-time 60s
```

### Option 3: Custom Python Script

```bash
python measure_latency.py
```

## Understanding Results

### Latency Distribution
```
P50:  150ms  (median)
P95:  450ms  (95% of requests faster)
P99:  800ms  (99% of requests faster)
Max:  2000ms (worst case)
```

### Throughput
```
Requests/sec: 25.3
Total requests: 1518
Failures: 2 (0.13%)
```

### What to Look For
- **High P99**: System struggling under load
- **Increasing latency**: Resource exhaustion
- **Failures**: System overloaded or errors
- **Low throughput**: Bottleneck somewhere

## Performance Bottlenecks

### Common Issues

1. **CPU-bound**: Model too large for CPU
   - **Solution**: Use GPU, smaller model, or quantization

2. **Memory-bound**: Out of memory
   - **Solution**: Reduce batch size, use smaller model

3. **I/O-bound**: Slow tokenization or network
   - **Solution**: Optimize tokenization, use faster network

4. **GPU underutilized**: Not batching efficiently
   - **Solution**: Use continuous batching (vLLM)

## Exercises

1. **Baseline Test**: Measure single-request latency
2. **Ramp-up Test**: Gradually increase from 1 to 50 users
3. **Compare Models**: Test gpt2 vs distilgpt2 performance
4. **Find Bottleneck**: Identify what limits throughput
5. **Stress Test**: Find maximum capacity

## Next Steps

- **Topic 5**: Use vLLM for better performance
- **Topic 8**: Set up monitoring dashboards
- **Topic 6**: Configure autoscaling based on load

