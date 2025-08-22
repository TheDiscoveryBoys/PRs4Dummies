#!/usr/bin/env python3
"""
Google Colab Setup Script for Ansible PR Indexer

This script sets up the indexing environment in Google Colab with GPU acceleration.
Run this at the top of your Colab notebook.

Usage in Colab:
    !wget https://raw.githubusercontent.com/your-repo/PRs4Dummies/main/indexing/setup_colab.py
    !python setup_colab.py
"""

import subprocess
import sys
import os

def install_requirements():
    """Install GPU-optimized requirements for Colab."""
    print("🚀 Setting up GPU-accelerated indexing environment...")
    
    # Install GPU version of FAISS
    print("📦 Installing faiss-gpu...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
    
    # Install other requirements
    requirements = [
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "sentence-transformers>=2.2.0", 
        "tqdm>=4.64.0",
        "python-dotenv>=1.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0"
    ]
    
    for req in requirements:
        print(f"📦 Installing {req}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name} ({gpu_count} GPU(s))")
            print(f"🔥 CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️ No GPU detected. Will use CPU.")
            return False
    except ImportError:
        print("⚠️ PyTorch not available. Will install and use CPU.")
        return False

def download_files():
    """Download the indexing script if not present."""
    if not os.path.exists("indexing.py"):
        print("📥 Downloading indexing script...")
        # In a real scenario, you'd download from your repository
        print("ℹ️ Please upload indexing.py to your Colab environment")
    else:
        print("✅ indexing.py found")

def print_usage_examples():
    """Print usage examples for Colab."""
    print("\n" + "="*60)
    print("🎯 READY TO INDEX! Usage examples:")
    print("="*60)
    
    print("\n💻 For small datasets (< 50 PRs):")
    print("!python indexing.py --device cuda --batch-size 32 --model all-MiniLM-L6-v2")
    
    print("\n🚀 For larger datasets (50+ PRs):")
    print("!python indexing.py --device cuda --batch-size 64 --model all-mpnet-base-v2")
    
    print("\n🐌 If GPU runs out of memory:")
    print("!python indexing.py --device cuda --batch-size 16 --chunk-size 1000")
    
    print("\n📊 To check device status:")
    print("!python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"")
    
    print("\n" + "="*60)

def main():
    """Main setup function."""
    print("🔧 Google Colab Indexer Setup")
    print("="*40)
    
    # Check current environment
    gpu_available = check_gpu()
    
    # Install requirements
    install_requirements()
    
    # Check if files are present
    download_files()
    
    # Final GPU check after installation
    print("\n🔍 Final environment check...")
    gpu_available = check_gpu()
    
    if gpu_available:
        print("✅ GPU setup complete!")
    else:
        print("✅ CPU setup complete!")
    
    # Print usage examples
    print_usage_examples()

if __name__ == "__main__":
    main()
