# Ansible PR Data Indexer

This component creates a searchable FAISS vector store from the LLM-optimized PR data scraped by the scraper component. It's specifically designed for RAG (Retrieval-Augmented Generation) systems.

## Key Improvements Over Basic Implementations

### 🎯 **LLM-Optimized Processing**
- Works directly with our scraper's clean JSON format
- Preserves all meaningful content while excluding noise
- Smart document structuring for better retrieval

### 🧩 **Advanced Chunking Strategy**
- Respects natural content boundaries (sections, code blocks, paragraphs)
- Larger chunks (1500 chars) with smart overlap (300 chars) for better context
- Validates chunk quality and provides statistics

### 🔍 **Better Embedding Model**
- Uses `all-mpnet-base-v2` instead of `all-MiniLM-L6-v2`
- Better performance on technical content and code
- Normalized embeddings for optimal FAISS performance

### 🛡️ **Production-Ready Features**
- Comprehensive error handling and validation
- Batch processing for memory efficiency
- Rich metadata for filtering and ranking
- Detailed logging and progress tracking

## Installation

### For Local Laptop (CPU Testing)
```bash
cd indexing
pip install -r requirements.txt  # Uses CPU version
```

### For Google Colab (GPU Acceleration)
```bash
# Option 1: Use the setup script
!python setup_colab.py

# Option 2: Manual installation
!pip install -r requirements-gpu.txt
```

### For GPU Machines
```bash
# Install GPU version manually
pip uninstall faiss-cpu
pip install faiss-gpu
pip install -r requirements-gpu.txt
```

## Usage

### 💻 Local Laptop (CPU) Testing
```bash
# Auto-detect device (will use CPU)
python indexing.py --data-dir ../scraper/scraped_data

# Force CPU explicitly (recommended for ≤50 PRs)
python indexing.py --device cpu --batch-size 16

# Use faster model for testing
python indexing.py --device cpu --model all-MiniLM-L6-v2 --batch-size 8
```

### 🚀 Google Colab (GPU) Usage
```bash
# Auto-detect GPU (recommended)
!python indexing.py --device auto --batch-size 64

# Force GPU usage
!python indexing.py --device cuda --batch-size 64 --model all-mpnet-base-v2

# Conservative GPU settings (if memory issues)
!python indexing.py --device cuda --batch-size 32 --chunk-size 1000
```

### 🔧 Device-Specific Options
```bash
# Force CPU (even if GPU available)
python indexing.py --force-cpu

# Apple Silicon Mac (MPS)
python indexing.py --device mps --batch-size 32

# Custom device selection
python indexing.py --device cuda --batch-size 128  # High-end GPU
```

### ⚙️ Advanced Configuration
```bash
# Custom model and chunking
python indexing.py --model sentence-transformers/all-mpnet-base-v2 \
                   --chunk-size 2000 --chunk-overlap 400

# Memory-efficient processing
python indexing.py --batch-size 16 --chunk-size 1000 --device cpu
```

## Document Structure

The indexer transforms each PR into a well-structured document:

```markdown
# Pull Request #12345: Fix authentication bug

## Pull Request Information
**Author:** username
**Merged by:** maintainer
**Labels:** bugfix, authentication, high-priority
**Changes:** +45 -12 lines, 3 files

## Description
This PR fixes the issue where users couldn't log in...

## Discussion
**reviewer commented:**
Looks good! Just one small suggestion...

**maintainer reviewed (APPROVED):**
Great work on this fix!

### Code Review Comments
**reviewer** on `auth/login.py`:
Perfect fix, thanks!

## Code Changes
```diff
diff --git a/auth/login.py b/auth/login.py
...
```

## Rich Metadata

Each document chunk includes comprehensive metadata for filtering and ranking:

```json
{
  "pr_number": 12345,
  "title": "Fix authentication bug",
  "author_login": "username",
  "merged_by_login": "maintainer", 
  "labels": ["bugfix", "authentication"],
  "labels_str": "bugfix, authentication",
  "additions": 45,
  "deletions": 12,
  "changed_files": 3,
  "has_diff": true,
  "comment_count": 3,
  "review_count": 2,
  "content_length": 1247,
  "chunk_id": 42,
  "chunk_hash": "a1b2c3d4e5f6",
  "source_type": "github_pr",
  "repository": "ansible/ansible"
}
```

## Output Structure

The indexer creates:

```
vector_store/
├── index.faiss          # FAISS vector index
├── index.pkl            # Document metadata and configuration
└── index_metadata.json  # Indexing statistics and settings
```

## Performance Characteristics

### Device Performance Comparison

| Device | Speed | Memory Usage | Best For |
|--------|-------|-------------|----------|
| **CPU (Laptop)** | Slow | Low | Testing ≤50 PRs |
| **GPU (Colab)** | **Fast** | Medium | Production, 50+ PRs |
| **Apple MPS** | Medium | Medium | Mac development |

### Processing Speed Examples
- **CPU (50 PRs)**: ~10-15 minutes
- **GPU (50 PRs)**: ~2-3 minutes  
- **GPU (500 PRs)**: ~15-20 minutes
- **CPU (500 PRs)**: ~2-3 hours

### Embedding Model Comparison

| Model | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| `all-MiniLM-L6-v2` | Good | Fast | Prototyping, CPU testing |
| `all-mpnet-base-v2` | **Better** | Medium | **Production (recommended)** |
| `all-mpnet-base-v1` | Better | Medium | Alternative technical model |

### Chunking Strategy

**Why larger chunks (1500 chars)?**
- Preserves more context for technical discussions
- Better for understanding code changes and their rationale
- Reduces fragmentation of related concepts

**Why smart separators?**
- Respects markdown structure (headers, code blocks)
- Preserves logical content boundaries
- Maintains readability and coherence

## Memory and Performance

### Memory Usage
- **Small dataset** (50 PRs): ~500MB RAM
- **Medium dataset** (500 PRs): ~2GB RAM  
- **Large dataset** (5000 PRs): ~8GB RAM

### Processing Time
- **Embedding creation**: ~1-2 seconds per document
- **FAISS index building**: ~5-10 seconds per 1000 chunks
- **Total time**: Roughly 1-2 minutes per 100 PRs

### Optimization Tips

**For limited memory:**
```bash
python indexing.py --batch-size 25 --chunk-size 1000
```

**For GPU acceleration:**
```bash
# Install faiss-gpu instead of faiss-cpu
pip uninstall faiss-cpu
pip install faiss-gpu
```

**For faster processing:**
```bash
# Use smaller, faster model
python indexing.py --model all-MiniLM-L6-v2
```

## Integration with RAG

The indexed vector store is designed for optimal RAG performance:

### Query Types Supported
- **Semantic search**: "How does Ansible handle authentication?"
- **Code-specific**: "Show me PRs that fix memory leaks"
- **Author-based**: "What changes has username made?"
- **Label filtering**: "Find all bugfix PRs"

### Retrieval Quality
- **High precision**: Relevant chunks due to smart chunking
- **Good recall**: Overlap ensures important content isn't lost
- **Rich context**: Metadata enables sophisticated filtering

### Example RAG Integration
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load the indexed data
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vector_store = FAISS.load_local("vector_store", embeddings)

# Search for relevant PRs
docs = vector_store.similarity_search(
    "authentication security fixes", 
    k=5,
    filter={"labels_str": {"$regex": "security|auth"}}
)
```

## Validation and Quality Control

The indexer includes comprehensive validation:

### Data Validation
- Checks JSON structure matches scraper output
- Validates required fields are present
- Handles corrupted or incomplete files gracefully

### Chunk Quality Analysis
- Reports chunk size distribution
- Identifies very small or oversized chunks
- Logs processing statistics

### Example Output
```
Chunk statistics:
  Average length: 1,347.2 characters
  Min length: 234 characters  
  Max length: 1,899 characters
Found 2 very small chunks (< 100 chars)
Found 0 oversized chunks (> 2250 chars)
```

## Troubleshooting

### Common Issues

**"No PR JSON files found"**
- Check that scraper has run successfully
- Verify `--data-dir` points to correct directory
- Look for files matching pattern `pr_*.json`

**Memory errors during embedding**
- Reduce `--batch-size` (try 10-25)
- Use smaller `--chunk-size` (try 1000)
- Consider using a smaller embedding model

**Slow processing**
- Increase `--batch-size` if you have more RAM
- Use GPU-accelerated FAISS if available
- Consider using `all-MiniLM-L6-v2` for faster processing

### Logs and Debugging
- Check `indexing.log` for detailed execution logs
- Use `--no-validate` to skip chunk validation for faster processing
- Increase logging verbosity by editing the logging level in the script

This indexer is designed to create high-quality vector stores that power effective RAG systems for Ansible PR data!
