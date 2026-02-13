"""
Locust Load Testing Script

This script tests LLM serving endpoints under load.
It measures:
- Latency (P50, P95, P99)
- Throughput (requests/second)
- Error rates
- Response times

Usage:
    locust -f locust_test.py --host=http://localhost:8000
    Then open http://localhost:8089 in browser
"""

from locust import HttpUser, task, between
import random
import json


class LLMServingUser(HttpUser):
    """
    Simulates a user making requests to the LLM serving API.
    
    Each "user" represents a concurrent connection.
    """
    
    # Wait between 1-3 seconds between requests
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts. Can be used for setup."""
        self.prompts = [
            "The future of artificial intelligence is",
            "Once upon a time",
            "The weather today is",
            "In a world where",
            "The key to success is",
            "Technology has changed",
            "The best way to learn is",
            "Artificial intelligence will",
        ]
    
    @task(3)
    def generate_text(self):
        """
        Main task: Generate text.
        
        @task(3) means this task is 3x more likely than other tasks.
        """
        prompt = random.choice(self.prompts)
        
        payload = {
            "prompt": prompt,
            "max_length": 50,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        # Make POST request
        # Locust automatically measures latency
        with self.client.post(
            "/generate",
            json=payload,
            catch_response=True,
            name="generate_text"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Log custom metrics
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Health check endpoint (lighter load)"""
        self.client.get("/health", name="health_check")
    
    @task(1)
    def metrics(self):
        """Metrics endpoint"""
        self.client.get("/metrics", name="metrics")


class HighLoadUser(HttpUser):
    """
    High-load scenario: Many requests, short wait time.
    
    Use this to stress test the system.
    """
    wait_time = between(0.1, 0.5)  # Very short wait
    
    @task
    def rapid_generate(self):
        """Rapid fire requests"""
        payload = {
            "prompt": "Test prompt",
            "max_length": 30,
        }
        self.client.post("/generate", json=payload, name="rapid_generate")

