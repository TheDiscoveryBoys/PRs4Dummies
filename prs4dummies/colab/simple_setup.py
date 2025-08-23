#!/usr/bin/env python3
"""
Simple and reliable Google Colab setup for PRs4Dummies
Use this if the main setup script has issues
"""

import os
import sys

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            print("⚠️ No GPU detected")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed")
        return False

def main():
    """Simple main setup."""
    print("🔧 PRs4Dummies - Simple Colab Setup")
    print("=" * 40)
    
    # Check GPU first
    has_gpu = check_gpu()
    
    print("\n📦 Installing packages...")
    
    # Install FAISS first (special handling for GPU)
    if has_gpu:
        print("\n🚀 Installing FAISS-GPU (Vector database)...")
        print("   Trying CUDA 12.1 compatible version...")
        result = os.system("pip install faiss-gpu-cu121")
        if result == 0:
            print("✅ FAISS-GPU (CUDA 12.1) installed successfully")
        else:
            print("   Trying CUDA 11.8 compatible version...")
            result = os.system("pip install faiss-gpu-cu118")
            if result == 0:
                print("✅ FAISS-GPU (CUDA 11.8) installed successfully")
            else:
                print("⚠️ FAISS-GPU failed, falling back to CPU version...")
                os.system("pip install faiss-cpu")
    else:
        print("\n📦 Installing FAISS-CPU (Vector database)...")
        result = os.system("pip install faiss-cpu")
        if result == 0:
            print("✅ FAISS-CPU installed successfully")
        else:
            print("⚠️ FAISS-CPU installation had issues")
    
    # Install other packages
    packages = [
        ("langchain", "LangChain framework"),
        ("langchain-community", "LangChain community tools"),
        ("sentence-transformers", "Embedding models"),
        ("transformers", "Hugging Face transformers"),
        ("einops", "Tensor operations (required for nomic)"),
        ("tqdm", "Progress bars")
    ]
    
    for package, description in packages:
        print(f"\n📦 Installing {package} ({description})...")
        result = os.system(f"pip install {package}")
        if result == 0:
            print(f"✅ {package} installed successfully")
        else:
            print(f"⚠️ {package} installation had issues (might still work)")
    
    print("\n🔍 Testing imports...")
    
    # Test critical imports
    try:
        import faiss
        print("✅ FAISS available")
    except ImportError:
        print("❌ FAISS not available")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✅ LangChain embeddings available")
    except ImportError:
        print("❌ LangChain embeddings not available")
    
    try:
        import sentence_transformers
        print("✅ Sentence transformers available")
    except ImportError:
        print("❌ Sentence transformers not available")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Download the indexer:")
    print("   !wget https://github.com/TheDiscoveryBoys/PRs4Dummies/raw/colab-attempt/colab/colab_indexer.py")
    print("2. Run the indexer:")
    print("   %run colab_indexer.py")

if __name__ == "__main__":
    main()
