#!/usr/bin/env python3
"""
Performance comparison test between standard and optimized RAG workflows
"""
import json
import time
import requests
import statistics
from typing import Dict, List, Any

def test_workflow_performance():
    """Test and compare performance between standard and optimized workflows"""
    
    base_url = "http://localhost:8080"
    
    # Test queries
    test_queries = [
        {
            "query_text": "What is artificial intelligence?",
            "index_name": "test_index",
            "permission_groups": ["user"],
            "retriever": "rrf",
            "num_result_doc": 5,
            "model": "llama4 maverick",
            "answer_format": "markdown"
        },
        {
            "query_text": "How does machine learning work?",
            "index_name": "test_index",
            "permission_groups": ["user"],
            "retriever": "rrf", 
            "num_result_doc": 5,
            "model": "llama4 scout",
            "answer_format": "markdown"
        },
        {
            "query_text": "Explain deep learning concepts",
            "index_name": "test_index",
            "permission_groups": ["user"],
            "retriever": "rrf",
            "num_result_doc": 5,
            "model": "gemma3",
            "answer_format": "markdown"
        }
    ]
    
    print("🧪 RAG Workflow Performance Comparison Test")
    print("=" * 60)
    
    results = {
        "standard": {"latencies": [], "errors": []},
        "optimized": {"latencies": [], "errors": []}
    }
    
    # Test Standard Workflow
    print("\n📊 Testing Standard Workflow (Sequential)")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: {query['query_text'][:50]}...")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/v1/ask",
                json=query,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                server_latency = result.get('latency_ms', 0)
                client_latency = (end_time - start_time) * 1000
                
                results["standard"]["latencies"].append({
                    "server_ms": server_latency,
                    "client_ms": client_latency,
                    "query": query['query_text'][:30]
                })
                
                print(f"  ✅ Server: {server_latency:.1f}ms, Client: {client_latency:.1f}ms")
                
                # Show debug info
                debug = result.get('debug', {})
                if debug:
                    print(f"     Model: {debug.get('model_used', 'unknown')}")
                    print(f"     Context: {debug.get('context_length', 0)} chars")
                    
            else:
                error_msg = f"HTTP {response.status_code}"
                results["standard"]["errors"].append(error_msg)
                print(f"  ❌ Error: {error_msg}")
                
        except Exception as e:
            results["standard"]["errors"].append(str(e))
            print(f"  ❌ Exception: {e}")
        
        time.sleep(1)  # Brief pause between requests
    
    # Test Optimized Workflow
    print("\n🚀 Testing Optimized Workflow (Parallel + Caching)")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: {query['query_text'][:50]}...")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/v1/ask/optimized",
                json=query,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                server_latency = result.get('latency_ms', 0)
                client_latency = (end_time - start_time) * 1000
                
                results["optimized"]["latencies"].append({
                    "server_ms": server_latency,
                    "client_ms": client_latency,
                    "query": query['query_text'][:30]
                })
                
                print(f"  ✅ Server: {server_latency:.1f}ms, Client: {client_latency:.1f}ms")
                
                # Show debug info
                debug = result.get('debug', {})
                if debug:
                    print(f"     Model: {debug.get('model_used', 'unknown')}")
                    print(f"     Workflow: {debug.get('workflow_type', 'unknown')}")
                    print(f"     Context: {debug.get('context_length', 0)} chars")
                    
            else:
                error_msg = f"HTTP {response.status_code}"
                results["optimized"]["errors"].append(error_msg)
                print(f"  ❌ Error: {error_msg}")
                
        except Exception as e:
            results["optimized"]["errors"].append(str(e))
            print(f"  ❌ Exception: {e}")
        
        time.sleep(1)  # Brief pause between requests
    
    # Test Cache Performance (Second Run)
    print("\n🔄 Testing Cache Performance (Second Run)")
    print("-" * 40)
    
    cache_test_query = test_queries[0]  # Use first query for cache test
    
    for run in range(3):
        print(f"Cache test run {run + 1}...")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/v1/ask/optimized",
                json=cache_test_query,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                server_latency = result.get('latency_ms', 0)
                client_latency = (end_time - start_time) * 1000
                
                print(f"  Run {run + 1}: Server {server_latency:.1f}ms, Client {client_latency:.1f}ms")
            else:
                print(f"  Run {run + 1}: Error {response.status_code}")
                
        except Exception as e:
            print(f"  Run {run + 1}: Exception {e}")
    
    # Performance Summary
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Standard workflow stats
    if results["standard"]["latencies"]:
        std_server_times = [r["server_ms"] for r in results["standard"]["latencies"]]
        std_client_times = [r["client_ms"] for r in results["standard"]["latencies"]]
        
        print(f"\n📊 Standard Workflow:")
        print(f"  Server Latency - Avg: {statistics.mean(std_server_times):.1f}ms, "
              f"Min: {min(std_server_times):.1f}ms, Max: {max(std_server_times):.1f}ms")
        print(f"  Client Latency - Avg: {statistics.mean(std_client_times):.1f}ms, "
              f"Min: {min(std_client_times):.1f}ms, Max: {max(std_client_times):.1f}ms")
        print(f"  Success Rate: {len(std_server_times)}/{len(test_queries)} ({len(std_server_times)/len(test_queries)*100:.1f}%)")
    
    # Optimized workflow stats  
    if results["optimized"]["latencies"]:
        opt_server_times = [r["server_ms"] for r in results["optimized"]["latencies"]]
        opt_client_times = [r["client_ms"] for r in results["optimized"]["latencies"]]
        
        print(f"\n🚀 Optimized Workflow:")
        print(f"  Server Latency - Avg: {statistics.mean(opt_server_times):.1f}ms, "
              f"Min: {min(opt_server_times):.1f}ms, Max: {max(opt_server_times):.1f}ms")
        print(f"  Client Latency - Avg: {statistics.mean(opt_client_times):.1f}ms, "
              f"Min: {min(opt_client_times):.1f}ms, Max: {max(opt_client_times):.1f}ms")
        print(f"  Success Rate: {len(opt_server_times)}/{len(test_queries)} ({len(opt_server_times)/len(test_queries)*100:.1f}%)")
    
    # Performance improvement calculation
    if (results["standard"]["latencies"] and results["optimized"]["latencies"]):
        std_avg = statistics.mean([r["server_ms"] for r in results["standard"]["latencies"]])
        opt_avg = statistics.mean([r["server_ms"] for r in results["optimized"]["latencies"]])
        
        if std_avg > 0:
            improvement = ((std_avg - opt_avg) / std_avg) * 100
            print(f"\n💡 Performance Improvement:")
            print(f"  Server Latency: {improvement:.1f}% {'improvement' if improvement > 0 else 'degradation'}")
            print(f"  Absolute Difference: {abs(std_avg - opt_avg):.1f}ms")
    
    # Error summary
    total_std_errors = len(results["standard"]["errors"])
    total_opt_errors = len(results["optimized"]["errors"])
    
    if total_std_errors > 0 or total_opt_errors > 0:
        print(f"\n⚠️  Error Summary:")
        print(f"  Standard Errors: {total_std_errors}")
        print(f"  Optimized Errors: {total_opt_errors}")
    
    # Recommendations
    print(f"\n🎯 Optimization Features Applied:")
    print(f"  ✅ Parallel retrieval execution with @task decorators")
    print(f"  ✅ Intelligent caching with TTL policies")
    print(f"  ✅ Retry policies for network resilience") 
    print(f"  ✅ Async checkpointing for better performance")
    print(f"  ✅ Context7 LangGraph best practices")
    
    print(f"\n" + "=" * 60)
    print("Test completed! Check results above for performance comparison.")

def test_server_health():
    """Check if server is running and healthy"""
    try:
        response = requests.get("http://localhost:8080/api/v1/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server is healthy: {health_data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach server: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking server health...")
    if test_server_health():
        print("\n🚀 Starting performance comparison tests...")
        test_workflow_performance()
    else:
        print("\n❌ Server is not available. Please start the server first:")
        print("   python run_dev.py")