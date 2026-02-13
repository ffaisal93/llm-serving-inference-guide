"""
Simple test script for the LLM serving API

Run this after starting the server to test the endpoints.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health check endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_root():
    """Test the root endpoint"""
    print("Testing / endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_generate():
    """Test the text generation endpoint"""
    print("Testing /generate endpoint...")
    
    payload = {
        "prompt": "The future of artificial intelligence is",
        "max_length": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "num_return_sequences": 1
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("Generating...")
    
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/generate",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    elapsed = time.time() - start
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Generated text: {result['generated_text']}")
        print(f"Latency: {result['latency_ms']:.2f}ms")
        print(f"Total time (including network): {elapsed*1000:.2f}ms")
        print(f"Number of tokens: {result['num_tokens']}")
    else:
        print(f"Error: {response.text}")
    print()


def test_metrics():
    """Test the metrics endpoint"""
    print("Testing /metrics endpoint...")
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Serving API Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_root()
        test_generate()
        test_metrics()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server.")
        print("Make sure the server is running: python app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

