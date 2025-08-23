#!/usr/bin/env python3
"""
Example usage of PRs4Dummies Colab Indexer
Copy and paste these code blocks into Google Colab cells
"""

# ============================================================================
# CELL 1: Setup and GPU Check
# ============================================================================

# Check if we're in Colab and enable GPU
print("🔍 Checking environment...")

try:
    from google.colab import files
    print("✅ Running in Google Colab")
    IN_COLAB = True
except ImportError:
    print("ℹ️ Not in Colab")
    IN_COLAB = False

# Check GPU
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
    USE_GPU = True
else:
    print("⚠️ No GPU detected - Go to Runtime > Change runtime type > GPU")
    USE_GPU = False

# ============================================================================
# CELL 2: Install Dependencies
# ============================================================================

print("📦 Installing dependencies...")

# Install GPU-optimized FAISS
if USE_GPU:
    !pip install faiss-gpu
else:
    !pip install faiss-cpu

# Install other requirements
!pip install langchain langchain-community sentence-transformers
!pip install transformers einops tqdm

print("✅ Installation complete!")

# ============================================================================  
# CELL 3: Download Indexer Script
# ============================================================================

# Download the indexer script
print("📥 Downloading indexer script...")

# Method 1: Download from GitHub (if public)
# !wget https://github.com/TheDiscoveryBoys/PRs4Dummies/raw/colab-attempt/colab/colab_indexer.py

# Method 2: Upload manually (for now)
if IN_COLAB:
    print("📤 Please upload the colab_indexer.py file:")
    uploaded = files.upload()
    if 'colab_indexer.py' not in uploaded:
        print("❌ Please upload colab_indexer.py file")
else:
    print("ℹ️ Ensure colab_indexer.py is in the current directory")

# ============================================================================
# CELL 4: Upload Your PR Data  
# ============================================================================

if IN_COLAB:
    print("📤 Upload your scraped PR data files:")
    print("You can upload individual JSON files or a ZIP archive")
    
    uploaded = files.upload()
    
    # Process uploaded files
    import zipfile
    from pathlib import Path
    
    # Create data directory
    data_dir = Path("scraped_data")
    data_dir.mkdir(exist_ok=True)
    
    for filename, content in uploaded.items():
        filepath = Path(filename)
        
        if filepath.suffix.lower() == '.zip':
            print(f"📦 Extracting: {filename}")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        elif filepath.suffix.lower() == '.json':
            print(f"📄 Moving: {filename}")
            filepath.rename(data_dir / filename)
        else:
            print(f"⚠️ Unsupported file: {filename}")
    
    # Check results
    json_files = list(data_dir.glob("*.json"))
    print(f"✅ Found {len(json_files)} JSON files ready for processing")
else:
    print("ℹ️ Ensure your JSON files are in the 'scraped_data' directory")

# ============================================================================
# CELL 5: Quick Run (Automatic Settings)
# ============================================================================

# Import the indexer
from colab_indexer import ColabPRIndexer, ColabIndexingConfig
import time

# Auto-configure based on dataset size  
from pathlib import Path
json_files = list(Path("scraped_data").glob("*.json"))
num_files = len(json_files)

print(f"📊 Dataset: {num_files} PR files")

# Recommend settings
if num_files <= 10:
    model = "all-MiniLM-L6-v2"
    batch_size = 32
    time_est = "1-2 minutes"
elif num_files <= 50:
    model = "nomic-ai/nomic-embed-text-v1.5" 
    batch_size = 64
    time_est = "3-5 minutes"
else:
    model = "nomic-ai/nomic-embed-text-v1.5"
    batch_size = 128 if USE_GPU else 16
    time_est = "10-30 minutes"

print(f"🤖 Using model: {model}")
print(f"⚡ Batch size: {batch_size}")
print(f"⏱️ Estimated time: {time_est}")

# Create configuration
config = ColabIndexingConfig(
    embedding_model=model,
    batch_size=batch_size,
    device="cuda" if USE_GPU else "cpu"
)

# Run indexing
print("\\n🎬 Starting indexing...")
start_time = time.time()

indexer = ColabPRIndexer(config)
vector_store = indexer.run_indexing()

total_time = time.time() - start_time
print(f"\\n🎉 Completed in {total_time:.1f} seconds!")
print(f"📊 Created {vector_store.index.ntotal} embeddings")

# ============================================================================
# CELL 6: Custom Configuration (Advanced Users)
# ============================================================================

# For advanced users who want custom settings
print("🔧 Custom configuration example:")

custom_config = ColabIndexingConfig(
    # Choose your model
    embedding_model="nomic-ai/nomic-embed-text-v1.5",  # Best quality
    # embedding_model="all-mpnet-base-v2",              # Good balance  
    # embedding_model="all-MiniLM-L6-v2",               # Fastest
    
    # Performance settings
    batch_size=64,              # Adjust based on GPU memory
    device="cuda",              # or "cpu" 
    
    # Chunking settings
    chunk_size=1500,            # Larger = more context
    chunk_overlap=300,          # Overlap between chunks
    
    # Data paths
    data_dir="scraped_data",
    output_dir="vector_store"
)

# Uncomment to run with custom config:
# indexer_custom = ColabPRIndexer(custom_config) 
# vector_store_custom = indexer_custom.run_indexing()

# ============================================================================
# CELL 7: Test the Vector Store
# ============================================================================

print("🧪 Testing the vector store...")

# Test search functionality
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embeddings (use same model as indexing)
embeddings = HuggingFaceEmbeddings(
    model_name=config.embedding_model,
    model_kwargs={'device': config.device, 'trust_remote_code': True}
)

# Load vector store
vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

# Test queries
test_queries = [
    "bug fix authentication",
    "deprecation warning", 
    "performance improvement",
    "security vulnerability"
]

for query in test_queries:
    print(f"\\n🔍 Searching for: '{query}'")
    results = vector_store.similarity_search(query, k=2)
    
    for i, result in enumerate(results):
        pr_num = result.metadata.get('pr_number', 'Unknown')
        title = result.metadata.get('title', 'No title')
        print(f"   {i+1}. PR #{pr_num}: {title[:50]}...")

print("\\n✅ Vector store is working correctly!")

# ============================================================================
# CELL 8: Download Results  
# ============================================================================

if IN_COLAB:
    import shutil
    
    print("📦 Creating downloadable archive...")
    
    # Create ZIP archive
    shutil.make_archive("vector_store", 'zip', 'vector_store')
    
    # Show contents
    vector_files = list(Path("vector_store").glob("*"))
    print(f"\\n📁 Archive contains:")
    for file in vector_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   {file.name} ({size_mb:.1f} MB)")
    
    # Download
    print("\\n⬇️ Downloading vector_store.zip...")
    files.download("vector_store.zip")
    
    print("✅ Download complete!")
    print("\\n🎯 Next steps:")
    print("1. Extract vector_store.zip in your local project")
    print("2. Update your RAG system to use the new embeddings")
    print("3. Test with questions about your PRs")

else:
    print("ℹ️ Vector store saved locally in 'vector_store' directory")

# ============================================================================
# CELL 9: Performance Summary
# ============================================================================

# Show final performance stats
print("📊 INDEXING PERFORMANCE SUMMARY")
print("=" * 40)
print(f"📁 Dataset size: {num_files} PRs")
print(f"🤖 Model used: {config.embedding_model}")
print(f"🖥️ Device: {config.device}")
print(f"⚡ Batch size: {config.batch_size}")
if 'total_time' in locals():
    print(f"⏱️ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    chunks_per_sec = vector_store.index.ntotal / total_time
    print(f"🚀 Speed: {chunks_per_sec:.1f} chunks/second")
print(f"📊 Total embeddings: {vector_store.index.ntotal}")
print(f"🔢 Vector dimension: {vector_store.index.d}")

print("\\n🎉 All done! Your PR data is now ready for RAG queries!")
