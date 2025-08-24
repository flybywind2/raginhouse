#!/usr/bin/env python3
"""
Test script for model selection functionality
"""
import json
import requests
from typing import Dict, Any

def test_model_selection():
    """Test the model selection API endpoint"""
    base_url = "http://localhost:8080"
    
    # Test data with different models
    test_cases = [
        {
            "model": "llama4 maverick",
            "query_text": "What is artificial intelligence?",
            "index_name": "test_index"
        },
        {
            "model": "llama4 scout", 
            "query_text": "What is machine learning?",
            "index_name": "test_index"
        },
        {
            "model": "gemma3",
            "query_text": "What is deep learning?", 
            "index_name": "test_index"
        }
    ]
    
    print("Testing Model Selection Functionality")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['model']}")
        print("-" * 30)
        
        # Prepare request payload
        payload = {
            "query_text": test_case["query_text"],
            "index_name": test_case["index_name"],
            "permission_groups": ["user"],
            "retriever": "rrf",
            "num_result_doc": 3,
            "model": test_case["model"],
            "answer_format": "markdown"
        }
        
        try:
            # Make API request
            response = requests.post(
                f"{base_url}/api/v1/ask",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Model Used: {test_case['model']}")
                print(f"Query: {test_case['query_text']}")
                print(f"Response Preview: {result.get('answer', '')[:100]}...")
                print(f"Latency: {result.get('latency_ms', 0):.2f}ms")
                print("✅ SUCCESS")
            else:
                print(f"❌ FAILED: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ REQUEST FAILED: {e}")
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {e}")

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8080/api/v1/health", timeout=10)
        print(f"\nHealth Check: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"Status: {health_data.get('status', 'unknown')}")
            print("✅ Server is healthy")
        else:
            print("❌ Server health check failed")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

if __name__ == "__main__":
    # First check if server is healthy
    test_health_endpoint()
    
    # Then test model selection
    test_model_selection()
    
    print("\n" + "=" * 50)
    print("Test completed. Check the web UI at http://localhost:8080")
    print("The model dropdown should now be available in the Query tab.")