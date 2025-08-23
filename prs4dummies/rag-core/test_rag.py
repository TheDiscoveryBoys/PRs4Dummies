"""
Test script for the RAG core functionality.

This script tests the RAG core without starting the full FastAPI server.
Useful for debugging and testing the core functionality.
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_rag_core():
    """Test the RAG core functionality."""
    try:
        print("🧪 Testing RAG Core...")
        
        # Import after setting up the path
        from rag_core import create_rag_core
        
        print("✅ Successfully imported RAG core")
        
        # Test vector store path
        vector_store_path = "../indexing/vector_store"
        if not Path(vector_store_path).exists():
            print(f"❌ Vector store not found at: {vector_store_path}")
            print("   Make sure you have run the indexing process first!")
            return False
        
        print(f"✅ Vector store found at: {vector_store_path}")
        
        # Test RAG core initialization
        print("🔄 Initializing RAG core...")
        rag = create_rag_core(vector_store_path=vector_store_path)
        print("✅ RAG core initialized successfully!")
        
        # Test vector store info
        print("📊 Getting vector store info...")
        info = rag.get_vector_store_info()
        print(f"   Total documents: {info.get('total_documents', 'Unknown')}")
        print(f"   Embedding dimension: {info.get('embedding_dimension', 'Unknown')}")
        print(f"   Model: {info.get('embedding_model', 'Unknown')}")
        
        # Test with a simple question
        print("\n🤔 Testing question answering...")
        test_question = "What is Ansible?"
        print(f"   Question: {test_question}")
        
        result = rag.answer_question(test_question)
        
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Sources used: {result['total_sources']}")
        
        if result.get('sources'):
            print("   Source details:")
            for i, source in enumerate(result['sources'][:3]):  # Show first 3 sources
                print(f"     {i+1}. {source.get('source', 'Unknown')}")
        
        print("\n🎉 RAG Core test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements-rag.txt")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    required_packages = [
        'langchain',
        'langchain_community', 
        'sentence_transformers',
        'faiss',
        'torch',
        'transformers',
        'fastapi',
        'uvicorn',
        'pydantic'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("   Install missing packages with: pip install -r requirements-rag.txt")
        return False
    else:
        print("✅ All required packages imported successfully!")
        return True

if __name__ == "__main__":
    print("🚀 PRs4Dummies RAG Core Test")
    print("=" * 40)
    
    # Test imports first
    if not test_imports():
        sys.exit(1)
    
    print("\n" + "=" * 40)
    
    # Test RAG core
    if test_rag_core():
        print("\n🎯 Ready to start the RAG API server!")
        print("   Run: python main.py")
    else:
        print("\n💥 RAG Core test failed. Check the errors above.")
        sys.exit(1)
