#!/usr/bin/env python3
"""
Google Colab Setup Script for PRs4Dummies Indexer
Run this first in your Colab notebook to set up the environment.
"""

import subprocess
import sys
import os

def check_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name}")
            print(f"🔥 CUDA version: {torch.version.cuda}")
            print(f"📊 GPU memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️ No GPU detected. Will use CPU (much slower).")
            return False
    except ImportError:
        print("⚠️ PyTorch not available. Installing...")
        return False

def install_requirements(use_gpu=True):
    """Install required packages for Colab."""
    print("📦 Installing dependencies...")
    
    # Core packages that are usually available in Colab
    requirements = [
        "langchain>=0.1.0",
        "langchain-community>=0.0.10", 
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "tqdm",
        "einops",  # Required for nomic models
    ]
    
    # Install FAISS
    if use_gpu:
        print("🚀 Installing GPU-accelerated FAISS...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
    else:
        print("🐌 Installing CPU FAISS...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    
    # Install other requirements
    for req in requirements:
        print(f"📦 Installing {req}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {req}: {e}")

def download_indexer():
    """Download the indexer script."""
    indexer_url = "https://raw.githubusercontent.com/yourusername/PRs4Dummies/main/colab/colab_indexer.py"
    
    # For now, just inform user to upload manually
    print("📥 Please upload the colab_indexer.py file to your Colab environment")
    print("   You can find it in the PRs4Dummies/colab/ directory")

def print_usage():
    """Print usage instructions."""
    print("\\n" + "="*60)
    print("🎯 SETUP COMPLETE! Usage Instructions:")
    print("="*60)
    
    print("\\n1️⃣ Upload your data:")
    print("   - Upload JSON files or a ZIP containing JSON files")
    print("   - Files will be processed automatically")
    
    print("\\n2️⃣ Run the indexer:")
    print("   %run colab_indexer.py")
    
    print("\\n3️⃣ For manual control:")
    print("   from colab_indexer import ColabPRIndexer, ColabIndexingConfig")
    print("   config = ColabIndexingConfig(embedding_model='nomic-ai/nomic-embed-text-v1.5')")
    print("   indexer = ColabPRIndexer(config)")
    print("   vector_store = indexer.run_indexing()")
    
    print("\\n4️⃣ Download results:")
    print("   - Vector store will be automatically packaged for download")
    print("   - Extract in your local project and use with RAG system")
    
    print("\\n💡 Performance Tips:")
    print("   - Use GPU runtime for 5-10x speedup")
    print("   - Larger batch sizes for bigger datasets")
    print("   - Use 'nomic-ai/nomic-embed-text-v1.5' for best quality")
    
    print("\\n" + "="*60)

def main():
    """Main setup function."""
    print("🔧 PRs4Dummies - Google Colab Setup")
    print("="*40)
    
    # Check if in Colab
    in_colab = check_colab()
    if in_colab:
        print("✅ Running in Google Colab")
    else:
        print("ℹ️ Not in Colab - some features may not work")
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Install requirements
    install_requirements(has_gpu)
    
    # Download indexer script
    if in_colab:
        download_indexer()
    
    # Final check
    print("\\n🔍 Final environment check...")
    try:
        import torch
        import faiss
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✅ All core dependencies available!")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please restart runtime and try again")
        return
    
    # Print usage
    print_usage()

if __name__ == "__main__":
    main()
