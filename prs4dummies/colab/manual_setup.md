# 🛠️ Manual Colab Setup (If Automated Setup Fails)

If you're getting pip errors with the automated setup, use these manual commands instead:

## 🚀 **Method 1: Direct pip commands**

Copy and paste each block into separate Colab cells:

### **Cell 1: Check GPU**
```python
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    USE_GPU = True
else:
    print("⚠️ No GPU - enable in Runtime > Change runtime type")
    USE_GPU = False
```

### **Cell 2: Install FAISS**
```python
# Install FAISS (Colab-compatible method)
if USE_GPU:
    print("🚀 Installing FAISS-GPU for CUDA 12.6...")
    # Try CUDA 12.1 first (most compatible with Colab)
    result = !pip install faiss-gpu-cu121
    if result.returncode != 0:
        print("   Trying CUDA 11.8 version...")
        !pip install faiss-gpu-cu118
    print("✅ Installed FAISS-GPU")
else:
    !pip install faiss-cpu
    print("✅ Installed FAISS-CPU")
```

### **Cell 3: Install Core Dependencies**
```python
# Install LangChain and related packages
!pip install langchain langchain-community
!pip install sentence-transformers transformers
!pip install einops tqdm

print("✅ Core dependencies installed")
```

### **Cell 4: Test Installation**
```python
# Test that everything works
try:
    import faiss
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import sentence_transformers
    import torch
    print("✅ All packages imported successfully!")
    print("🚀 Ready to proceed with indexing")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Try restarting runtime and running setup again")
```

## 🔄 **Method 2: Alternative Setup Script**

If you prefer a script approach:

```python
# Download the simple setup script
!wget https://github.com/TheDiscoveryBoys/PRs4Dummies/raw/colab-attempt/colab/simple_setup.py
!python simple_setup.py
```

## 📥 **Download Indexer**

Once setup is complete:

```python
# Download the indexer script
!wget https://github.com/TheDiscoveryBoys/PRs4Dummies/raw/colab-attempt/colab/colab_indexer.py

# Run it
%run colab_indexer.py
```

## 🔧 **Troubleshooting**

| Error | Solution |
|-------|----------|
| `No module named pip3` | Use `!pip install` instead of complex subprocess calls |
| `CUDA out of memory` | Restart runtime, ensure GPU is enabled |
| `Import errors` | Run `!pip install package-name` for specific missing packages |
| `404 errors` | Check that files are in the `colab-attempt` branch |

## 🎯 **Why This Works Better**

- **Direct pip commands**: More reliable than subprocess calls
- **Cell-by-cell**: Easy to debug which step fails
- **Clear feedback**: See exactly what's happening
- **Colab-native**: Uses `!pip` which Colab handles best

Try Method 1 first - it's the most reliable approach for Colab!
