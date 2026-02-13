"""
Latency Measurement Script

This script measures latency for LLM serving endpoints.
It calculates:
- Mean latency
- Median (P50)
- P95, P99 percentiles
- Min/Max latency
- Throughput

Usage:
    python measure_latency.py
"""

import requests
import time
import statistics
from typing import List, Dict
import json


def measure_request(url: str, payload: dict) -> float:
    """
    Measure latency for a single request.
    
    Returns:
        Latency in milliseconds
    """
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        elapsed = (time.time() - start) * 1000  # Convert to ms
        return elapsed
    except Exception as e:
        print(f"Error: {e}")
        return None


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculate percentile (P50, P95, P99, etc.)"""
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


def measure_latency(
    base_url: str = "http://localhost:8000",
    num_requests: int = 100,
    prompt: str = "The future of AI is",
    max_length: int = 50
) -> Dict:
    """
    Measure latency for multiple requests.
    
    Args:
        base_url: Base URL of the serving API
        num_requests: Number of requests to make
        prompt: Test prompt
        max_length: Maximum tokens to generate
        
    Returns:
        Dictionary with latency statistics
    """
    url = f"{base_url}/generate"
    payload = {
        "prompt": prompt,
        "max_length": max_length,
        "temperature": 0.7,
    }
    
    print(f"Measuring latency for {num_requests} requests...")
    print(f"URL: {url}")
    print(f"Prompt: {prompt}")
    print()
    
    latencies = []
    errors = 0
    
    for i in range(num_requests):
        latency = measure_request(url, payload)
        if latency is not None:
            latencies.append(latency)
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_requests} requests...")
        else:
            errors += 1
    
    if not latencies:
        return {"error": "All requests failed"}
    
    # Calculate statistics
    stats = {
        "total_requests": num_requests,
        "successful_requests": len(latencies),
        "failed_requests": errors,
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": calculate_percentile(latencies, 95),
        "p99_latency_ms": calculate_percentile(latencies, 99),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "throughput_rps": len(latencies) / (sum(latencies) / 1000),  # Requests per second
    }
    
    return stats


def print_results(stats: Dict):
    """Print latency statistics in a readable format"""
    print("=" * 60)
    print("LATENCY MEASUREMENT RESULTS")
    print("=" * 60)
    print()
    
    if "error" in stats:
        print(f"Error: {stats['error']}")
        return
    
    print(f"Total Requests:     {stats['total_requests']}")
    print(f"Successful:         {stats['successful_requests']}")
    print(f"Failed:             {stats['failed_requests']}")
    print()
    print("Latency Statistics:")
    print(f"  Mean:             {stats['mean_latency_ms']:.2f} ms")
    print(f"  Median (P50):     {stats['median_latency_ms']:.2f} ms")
    print(f"  P95:              {stats['p95_latency_ms']:.2f} ms")
    print(f"  P99:              {stats['p99_latency_ms']:.2f} ms")
    print(f"  Min:              {stats['min_latency_ms']:.2f} ms")
    print(f"  Max:              {stats['max_latency_ms']:.2f} ms")
    print(f"  Std Dev:          {stats['std_dev_ms']:.2f} ms")
    print()
    print(f"Throughput:         {stats['throughput_rps']:.2f} requests/second")
    print()
    
    # Interpretation
    print("Interpretation:")
    if stats['p99_latency_ms'] < 200:
        print("  ✅ Excellent: P99 < 200ms")
    elif stats['p99_latency_ms'] < 500:
        print("  ✅ Good: P99 < 500ms")
    elif stats['p99_latency_ms'] < 1000:
        print("  ⚠️  Acceptable: P99 < 1s")
    else:
        print("  ❌ Poor: P99 > 1s - consider optimization")
    
    if stats['p99_latency_ms'] / stats['median_latency_ms'] > 5:
        print("  ⚠️  High variance: P99 is much higher than median")
        print("     This indicates inconsistent performance")
    
    print("=" * 60)


def compare_endpoints():
    """Compare latency between different endpoints"""
    endpoints = [
        ("Basic Serving", "http://localhost:8000"),
        ("vLLM Serving", "http://localhost:8001"),
    ]
    
    results = {}
    for name, url in endpoints:
        print(f"\nTesting {name}...")
        try:
            # Quick health check
            health = requests.get(f"{url}/health", timeout=5)
            if health.status_code == 200:
                stats = measure_latency(url, num_requests=50)
                results[name] = stats
            else:
                print(f"  ⚠️  {name} not available (health check failed)")
        except Exception as e:
            print(f"  ⚠️  {name} not available: {e}")
    
    # Print comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        for name, stats in results.items():
            if "error" not in stats:
                print(f"\n{name}:")
                print(f"  P50: {stats['median_latency_ms']:.2f} ms")
                print(f"  P95: {stats['p95_latency_ms']:.2f} ms")
                print(f"  P99: {stats['p99_latency_ms']:.2f} ms")
                print(f"  Throughput: {stats['throughput_rps']:.2f} req/s")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Measure LLM serving latency")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                       help="Base URL of serving API")
    parser.add_argument("--requests", type=int, default=100,
                       help="Number of requests to make")
    parser.add_argument("--prompt", type=str, default="The future of AI is",
                       help="Test prompt")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple endpoints")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_endpoints()
    else:
        stats = measure_latency(
            base_url=args.url,
            num_requests=args.requests,
            prompt=args.prompt
        )
        print_results(stats)

