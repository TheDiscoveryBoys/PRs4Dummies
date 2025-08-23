"""
Simple test script for the RAG API endpoints.
Run this after starting the server with: python main.py
"""

import requests
import json
import time

def test_api():
    """Test the RAG API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing PRs4Dummies RAG API...")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   📊 Status: {data['status']}")
            print(f"   📚 Vector store: {data['vector_store_info']['total_documents']} documents")
            print(f"   ⏱️  Uptime: {data['uptime_seconds']:.1f} seconds")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test 2: API info
    print("\n2. Testing info endpoint...")
    try:
        response = requests.get(f"{base_url}/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Info endpoint working")
            print(f"   🤖 LLM Type: {data['llm_type']}")
            print(f"   🔧 API Version: {data['api_version']}")
        else:
            print(f"   ❌ Info endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Info endpoint failed: {e}")
    
    # Test 3: Ask a question
    print("\n3. Testing question endpoint...")
    try:
        question_data = {
            "question": "What is Ansible?",
            "include_sources": True
        }
        
        response = requests.post(
            f"{base_url}/ask",
            json=question_data,
            timeout=30  # Longer timeout for LLM processing
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Question answered successfully")
            print(f"   🤔 Question: {data['question']}")
            print(f"   💡 Answer: {data['answer'][:100]}...")
            print(f"   📚 Sources used: {data['total_sources']}")
            print(f"   ⏱️  Processing time: {data['processing_time_ms']}ms")
        else:
            print(f"   ❌ Question endpoint failed: {response.status_code}")
            print(f"   📝 Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Question endpoint failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print("\n📖 Access the interactive API docs at:")
    print(f"   {base_url}/docs")
    
    return True

if __name__ == "__main__":
    print("🚀 PRs4Dummies RAG API Test")
    print("Make sure the server is running with: python main.py")
    print()
    
    # Wait a moment for user to read
    input("Press Enter to start testing...")
    
    test_api()
