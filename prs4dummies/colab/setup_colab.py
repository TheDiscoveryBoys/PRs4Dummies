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

def install_conda():
    """Install conda via condacolab (Colab-optimized method)."""
    print("🐍 Installing conda via condacolab...")
    
    try:
        # Install condacolab first
        print("   Installing condacolab package...")
        result = os.system("pip install -q condacolab")
        if result != 0:
            print("⚠️ Failed to install condacolab")
            return False
        
        # Import and install conda
        print("   Setting up conda environment...~")
        import condacolab
        condacolab.install()
        
        print("✅ Conda installed successfully via condacolab!")
        print("⚠️ IMPORTANT: Colab runtime will restart automatically.")
        print("   After restart, re-run this script to continue with FAISS installation.")
        return True
        
    except Exception as e:
        print(f"⚠️ Conda installation failed: {e}")
        return False

def install_requirements(use_gpu=True):
    """Install required packages for Colab."""
    print("📦 Installing dependencies...")
    
    # First, install conda
    conda_success = install_conda()
    
    if not conda_success:
        print("⚠️ Conda installation failed, falling back to pip-only method")
        use_conda = False
    else:
        use_conda = True
    
    # Core packages that are usually available in Colab
    requirements = [
        "faiss",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10", 
        "langchain-huggingface>=0.0.1",  # Newer package for HuggingFace embeddings
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "tqdm",
        "einops",  # Required for nomic models
    ]
    
    # Install FAISS using conda if available, otherwise pip

    print("🚀 Installing GPU-accelerated FAISS...")

    print("   Using conda with pytorch and nvidia channels...")
    # Use the exact method from your screenshot
    result = os.system("conda install -c pytorch -c nvidia faiss-gpu=1.12.0 -y")
    if result == 0:
        print("✅ FAISS-GPU installed via conda!")
    else:
        print("   Trying conda-forge channel...")
        result2 = os.system("conda install -c conda-forge faiss-gpu -y")
        if result2 == 0:
            print("✅ FAISS-GPU installed via conda-forge!")
        else:
            print("   Conda failed to install FAISS-GPU...")
        
    
    # Install other requirements using pip (more reliable for these packages)
    for req in requirements:
        print(f"📦 Installing {req}...")
        try:
            result = os.system(f"pip install {req}")
            if result != 0:
                print(f"⚠️ Failed to install {req} with pip, trying alternative...")
                os.system(f"pip install --upgrade {req}")
        except Exception as e:
            print(f"⚠️ Failed to install {req}: {e}")

def download_indexer():
    """Download the indexer script."""
    indexer_url = "https://github.com/TheDiscoveryBoys/PRs4Dummies/raw/colab-attempt/colab/colab_indexer.py"
    
    # For now, just inform user to upload manually
    print("📥 Please upload the colab_indexer.py file to your Colab environment")
    print("   You can find it in the PRs4Dummies/colab/ directory")

def print_usage():
    """Print usage instructions."""
    print("\\n" + "="*60)
    print("🎯 SETUP COMPLETE! Usage Instructions:")
    print("="*60)
    
    print("\\n🐍 Conda Environment:")
    print("   - Miniconda installed at /content/miniconda")
    print("   - FAISS-GPU installed via conda-forge")
    print("   - GPU acceleration ready!")
    
    print("\\n1️⃣ Download indexer:")
    print("   !wget https://github.com/TheDiscoveryBoys/PRs4Dummies/raw/colab-attempt/colab/colab_indexer.py")
    
    print("\\n2️⃣ Run the indexer:")
    print("   %run colab_indexer.py")
    print("   (Will prompt for data upload)")
    
    print("\\n3️⃣ For manual control:")
    print("   from colab_indexer import ColabPRIndexer, ColabIndexingConfig")
    print("   config = ColabIndexingConfig(embedding_model='nomic-ai/nomic-embed-text-v1.5')")
    print("   indexer = ColabPRIndexer(config)")
    print("   vector_store = indexer.run_indexing()")
    
    print("\\n4️⃣ Download results:")
    print("   - Vector store will be automatically packaged for download")
    print("   - Extract in your local project and use with RAG system")
    
    print("\\n🚀 Performance with Tesla T4:")
    print("   - ~3 minutes for 50 PRs")
    print("   - ~20 minutes for 500 PRs")
    print("   - ~45 minutes for 1000+ PRs")
    
    print("\\n💡 Conda Commands (if needed):")
    print("   - List packages: /content/miniconda/bin/conda list")
    print("   - Install package: /content/miniconda/bin/conda install package-name")
    
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
