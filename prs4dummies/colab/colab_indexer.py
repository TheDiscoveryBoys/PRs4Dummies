#!/usr/bin/env python3
"""
Google Colab Optimized PR Indexer
Adapted for Colab environment with GPU acceleration
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

# Check if we're in Colab
try:
    from google.colab import files
    IN_COLAB = True
    print("✅ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("ℹ️ Not in Colab - using local mode")

# Core dependencies
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    # Try the newer package first
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to the older package
    from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm.auto import tqdm
import torch

@dataclass
class ColabIndexingConfig:
    """Configuration optimized for Google Colab."""
    
    # Data paths (Colab-specific)
    data_dir: str = "scraped_data"
    output_dir: str = "vector_store"
    
    # Embedding model (optimized for Colab GPU)
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    
    # Device settings (auto-detect GPU)
    device: str = "auto"  # Will auto-detect cuda/cpu
    
    # Chunk settings (optimized for technical content)
    chunk_size: int = 1500
    chunk_overlap: int = 300
    
    # Performance settings (GPU-optimized)
    batch_size: int = 64  # Larger batch for GPU
    max_workers: int = 4
    
    # Validation
    validate_chunks: bool = True


class ColabPRIndexer:
    """Colab-optimized indexer for processing PR data."""
    
    def __init__(self, config: ColabIndexingConfig = None):
        self.config = config or ColabIndexingConfig()
        self._setup_logging()
        
        # Device detection and configuration
        self.device = self._detect_device()
        print(f"🖥️ Using device: {self.device}")
        
        # Adjust batch size based on device
        if self.device == "cuda":
            # GPU can handle larger batches
            self.batch_size = min(self.config.batch_size, 128)
            print(f"🚀 GPU detected! Using batch size: {self.batch_size}")
        else:
            # CPU needs smaller batches
            self.batch_size = min(self.config.batch_size, 16)
            print(f"🐌 Using CPU. Batch size: {self.batch_size}")
        
        # Initialize components
        self.embeddings = None
        self.text_splitter = None
        self._init_components()
        
    def _setup_logging(self):
        """Setup logging for Colab."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _detect_device(self) -> str:
        """Detect the best device for Colab."""
        if self.config.device != "auto":
            return self.config.device
            
        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _init_components(self):
        """Initialize embedding model and text splitter."""
        try:
            print(f"🤖 Initializing embedding model: {self.config.embedding_model}")
            
            # Configure model for the detected device
            model_kwargs = {
                'device': self.device,
                'trust_remote_code': True  # Required for nomic models
            }
            encode_kwargs = {'normalize_embeddings': True}  # Important for FAISS
            
            # Add device-specific optimizations
            if self.device == "cuda":
                # GPU optimizations
                encode_kwargs['batch_size'] = self.batch_size
                # Note: torch_dtype is handled by the model itself, not passed to SentenceTransformer
                print(f"⚡ GPU optimizations: batch size {self.batch_size}")
            else:
                # CPU optimizations
                encode_kwargs['batch_size'] = self.batch_size
                print(f"🔧 CPU optimizations: batch size {self.batch_size}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # Smart text splitter for technical content
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=[
                    "\\n\\n",      # Paragraph breaks
                    "\\n###",     # Markdown subheadings
                    "\\n##",      # Markdown headings
                    "\\n```",     # Code blocks
                    "\\n",        # Line breaks
                    ". ",        # Sentences
                    " ",         # Words
                    "",          # Characters
                ],
                length_function=len,
            )
            
            print("✅ Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def load_pr_data(self, data_dir: str) -> List[Dict[str, Any]]:
        """Load PR data from JSON files."""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        json_files = list(data_path.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {data_path}")
            
        print(f"📁 Found {len(json_files)} PR files")
        
        pr_data_list = []
        failed_files = []
        
        for json_file in tqdm(json_files, desc="Loading PR files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    pr_data = json.load(f)
                    
                # Validate the PR data structure
                if self._validate_pr_data(pr_data):
                    pr_data_list.append(pr_data)
                else:
                    failed_files.append(json_file)
                    
            except json.JSONDecodeError as e:
                failed_files.append(json_file)
                self.logger.error(f"JSON decode error in {json_file}: {e}")
            except Exception as e:
                failed_files.append(json_file)
                self.logger.error(f"Error loading {json_file}: {e}")
                
        if failed_files:
            print(f"⚠️ Failed to load {len(failed_files)} files")
            
        print(f"✅ Successfully loaded {len(pr_data_list)} PRs")
        return pr_data_list
        
    def _validate_pr_data(self, pr_data: Dict[str, Any]) -> bool:
        """Validate PR data structure."""
        required_fields = ['pr_number', 'title', 'body']
        
        for field in required_fields:
            if field not in pr_data:
                return False
                
        # Check for expected structure
        if not isinstance(pr_data.get('comments', []), list):
            return False
        if not isinstance(pr_data.get('reviews', []), list):
            return False
            
        return True

    def format_pr_to_document(self, pr_data: Dict[str, Any]) -> Document:
        """Transform PR data into a clean, structured document optimized for RAG."""
        pr_number = pr_data.get('pr_number', 'unknown')
        title = pr_data.get('title', 'No title')
        
        # Build structured content with clear sections
        content_parts = []
        
        # Header section
        content_parts.append(f"# Pull Request #{pr_number}: {title}")
        content_parts.append("")
        
        # Metadata section
        author = pr_data.get('author_login', 'Unknown')
        merged_by = pr_data.get('merged_by_login')
        labels = pr_data.get('labels', [])
        
        content_parts.append("## Pull Request Information")
        content_parts.append(f"**Author:** {author}")
        if merged_by:
            content_parts.append(f"**Merged by:** {merged_by}")
        if labels:
            content_parts.append(f"**Labels:** {', '.join(labels)}")
            
        # Code statistics
        additions = pr_data.get('additions', 0)
        deletions = pr_data.get('deletions', 0)
        changed_files = pr_data.get('changed_files', 0)
        content_parts.append(f"**Changes:** +{additions} -{deletions} lines, {changed_files} files")
        content_parts.append("")
        
        # Description section
        body = pr_data.get('body', '').strip()
        if body:
            content_parts.append("## Description")
            content_parts.append(body)
            content_parts.append("")
            
        # Comments and reviews section
        comments = pr_data.get('comments', [])
        reviews = pr_data.get('reviews', [])
        
        if comments or reviews:
            content_parts.append("## Discussion")
            
            # Regular comments
            regular_comments = [c for c in comments if c.get('comment_type') != 'review_comment']
            for comment in regular_comments:
                author = comment.get('author_login', 'Unknown')
                body = comment.get('body', '').strip()
                if body:
                    content_parts.append(f"**{author} commented:**")
                    content_parts.append(body)
                    content_parts.append("")
            
            # Review comments
            for review in reviews:
                author = review.get('author_login', 'Unknown')
                state = review.get('state', 'commented')
                body = review.get('body', '').strip()
                if body:
                    content_parts.append(f"**{author} {state}:**")
                    content_parts.append(body)
                    content_parts.append("")
            
            # Inline review comments
            inline_comments = [c for c in comments if c.get('comment_type') == 'review_comment']
            if inline_comments:
                content_parts.append("### Code Review Comments")
                for comment in inline_comments:
                    author = comment.get('author_login', 'Unknown')
                    body = comment.get('body', '').strip()
                    file_path = comment.get('file_path', '')
                    if body:
                        content_parts.append(f"**{author}** on `{file_path}`:")
                        content_parts.append(body)
                        content_parts.append("")
        
        # Code diff section
        diff = pr_data.get('diff', '').strip()
        if diff and diff != 'null':
            content_parts.append("## Code Changes")
            content_parts.append("```diff")
            content_parts.append(diff)
            content_parts.append("```")
            
        # Combine all parts
        content = "\\n".join(content_parts)
        
        # Create rich metadata for filtering and ranking
        metadata = {
            "pr_number": pr_number,
            "title": title,
            "author_login": author,
            "merged_by_login": merged_by or "",
            "labels": labels,
            "labels_str": ", ".join(labels),
            "additions": additions,
            "deletions": deletions,
            "changed_files": changed_files,
            "has_diff": bool(diff and diff != 'null'),
            "comment_count": len(comments),
            "review_count": len(reviews),
            "content_length": len(content),
            "source_type": "github_pr",
            "repository": "ansible/ansible",
            "pr_url": f"https://github.com/ansible/ansible/pull/{pr_number}",
            "source": f"https://github.com/ansible/ansible/pull/{pr_number}"
        }
        
        return Document(page_content=content, metadata=metadata)

    def process_documents(self, pr_data_list: List[Dict[str, Any]]) -> List[Document]:
        """Convert PR data to documents and split into chunks."""
        print(f"🔄 Processing {len(pr_data_list)} PRs into documents")
        
        # Convert to documents
        documents = []
        for pr_data in tqdm(pr_data_list, desc="Creating documents"):
            try:
                doc = self.format_pr_to_document(pr_data)
                documents.append(doc)
            except Exception as e:
                self.logger.error(f"Error processing PR {pr_data.get('pr_number', 'unknown')}: {e}")
        
        print(f"✅ Created {len(documents)} documents")
        
        # Split into chunks
        print("✂️ Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_hash"] = hashlib.md5(
                chunk.page_content.encode()
            ).hexdigest()[:12]
        
        print(f"✅ Created {len(chunks)} document chunks")
        
        # Chunk statistics
        if chunks:
            lengths = [len(chunk.page_content) for chunk in chunks]
            avg_length = sum(lengths) / len(lengths)
            print(f"📊 Chunk statistics:")
            print(f"   Average length: {avg_length:.1f} characters")
            print(f"   Min length: {min(lengths)} characters")
            print(f"   Max length: {max(lengths)} characters")
        
        return chunks

    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """Create FAISS vector store from document chunks."""
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")
        
        print(f"🚀 Creating FAISS vector store from {len(chunks)} chunks...")
        start_time = time.time()
        
        try:
            # Create vector store
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Vector store creation completed in {elapsed:.2f} seconds")
            
            return vector_store
            
        except Exception as e:
            self.logger.error(f"Error creating vector store: {e}")
            raise

    def save_vector_store(self, vector_store: FAISS, output_dir: str):
        """Save vector store and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        vector_store.save_local(str(output_path))
        print(f"💾 Vector store saved to {output_path}")
        
        # Save metadata
        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "total_chunks": vector_store.index.ntotal,
            "vector_dimension": vector_store.index.d,
            "device_used": self.device,
            "batch_size": self.batch_size
        }
        
        metadata_path = output_path / "index_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📄 Metadata saved to {metadata_path}")

    def run_indexing(self):
        """Main indexing process."""
        print("🎯 === Starting Ansible PR Indexing Process ===")
        
        try:
            # Load PR data
            print(f"📁 Loading PR data from {self.config.data_dir}")
            pr_data_list = self.load_pr_data(self.config.data_dir)
            
            # Process documents
            chunks = self.process_documents(pr_data_list)
            
            if not chunks:
                raise ValueError("No valid document chunks created")
            
            # Create vector store
            vector_store = self.create_vector_store(chunks)
            
            # Save results
            self.save_vector_store(vector_store, self.config.output_dir)
            
            print("🎉 === Indexing Process Completed Successfully ===")
            print(f"📊 Total PRs processed: {len(pr_data_list)}")
            print(f"📄 Document chunks created: {len(chunks)}")
            print(f"💾 Vector store saved to: {self.config.output_dir}")
            
            return vector_store
            
        except Exception as e:
            self.logger.error(f"Indexing process failed: {e}")
            raise


def check_gpu():
    """Check GPU availability."""
    print("🔍 Checking GPU availability...")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name}")
        print(f"🔥 CUDA version: {torch.version.cuda}")
        print(f"📊 GPU memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("⚠️ No GPU detected. Using CPU (much slower).")
        if IN_COLAB:
            print("💡 Go to Runtime > Change runtime type > Hardware accelerator > GPU")
        return False


def setup_colab_environment():
    """Setup the Colab environment."""
    print("🔧 Setting up Colab environment...")
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Install dependencies if needed
    try:
        import faiss
        if has_gpu:
            print("✅ FAISS-GPU available")
        else:
            print("✅ FAISS-CPU available")
    except ImportError:
        print("📦 Installing FAISS...")
        if has_gpu:
            os.system("pip install faiss-gpu")
        else:
            os.system("pip install faiss-cpu")
    
    # Check other dependencies
    dependencies = [
        "langchain", "langchain-community", "langchain-huggingface", "sentence-transformers", 
        "transformers", "einops", "tqdm"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep} available")
        except ImportError:
            print(f"📦 Installing {dep}...")
            os.system(f"pip install {dep}")
    
    return has_gpu


def recommend_settings(num_files: int, has_gpu: bool):
    """Recommend optimal settings based on data size and hardware."""
    if num_files <= 10:
        model = "all-MiniLM-L6-v2"  # Faster for small datasets
        batch_size = 32 if has_gpu else 8
        time_estimate = "1-2 minutes"
    elif num_files <= 50:
        model = "nomic-ai/nomic-embed-text-v1.5"  # Better quality
        batch_size = 64 if has_gpu else 16
        time_estimate = "2-5 minutes" if has_gpu else "10-20 minutes"
    else:
        model = "nomic-ai/nomic-embed-text-v1.5"  # Best for large datasets
        batch_size = 128 if has_gpu else 16
        time_estimate = "10-30 minutes" if has_gpu else "1-3 hours"
    
    print(f"📊 Dataset size: {num_files} PRs")
    print(f"🤖 Recommended model: {model}")
    print(f"⚡ Recommended batch size: {batch_size}")
    print(f"⏱️ Estimated time: {time_estimate}")
    
    return model, batch_size


def upload_data():
    """Handle data upload for Colab."""
    if not IN_COLAB:
        print("ℹ️ Not in Colab - please ensure your data is in the 'scraped_data' directory")
        return
    
    print("📤 Upload your scraped PR data:")
    print("Option 1: Upload individual JSON files")
    print("Option 2: Upload a ZIP file containing JSON files")
    
    uploaded = files.upload()
    
    # Create data directory
    data_dir = Path("scraped_data")
    data_dir.mkdir(exist_ok=True)
    
    # Process uploaded files
    import zipfile
    for filename, content in uploaded.items():
        filepath = Path(filename)
        
        if filepath.suffix.lower() == '.zip':
            print(f"📦 Extracting ZIP file: {filename}")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        elif filepath.suffix.lower() == '.json':
            print(f"📄 Processing JSON file: {filename}")
            # Move to data directory
            filepath.rename(data_dir / filename)
        else:
            print(f"⚠️ Unsupported file type: {filename}")


def download_results():
    """Create downloadable archive for Colab."""
    if not IN_COLAB:
        print("ℹ️ Not in Colab - vector store saved locally")
        return
    
    import shutil
    
    print("📦 Creating downloadable archive...")
    
    # Create ZIP archive
    archive_name = "vector_store"
    shutil.make_archive(archive_name, 'zip', 'vector_store')
    
    # Show what's included
    vector_store_files = list(Path("vector_store").glob("*"))
    print(f"\\n📁 Archive contains:")
    for file in vector_store_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   {file.name} ({size_mb:.1f} MB)")

    archive_size = Path(f"{archive_name}.zip").stat().st_size / (1024 * 1024)
    print(f"\\n📦 Total archive size: {archive_size:.1f} MB")

    # Download the archive
    print("\\n⬇️ Downloading vector store...")
    files.download(f"{archive_name}.zip")
    print("✅ Download complete!")


def main():
    """Main function for running in Colab or standalone."""
    print("🚀 PRs4Dummies - Colab Indexer")
    print("=" * 40)
    
    # Setup environment
    has_gpu = setup_colab_environment()
    
    # Handle data upload
    if IN_COLAB:
        upload_data()
    
    # Check data directory
    data_dir = Path("scraped_data")
    if not data_dir.exists() or not list(data_dir.glob("*.json")):
        print("❌ No JSON files found. Please upload your scraped PR data.")
        return
    
    json_files = list(data_dir.glob("*.json"))
    model, batch_size = recommend_settings(len(json_files), has_gpu)
    
    # Create configuration
    config = ColabIndexingConfig(
        embedding_model=model,
        batch_size=batch_size,
        device="cuda" if has_gpu else "cpu"
    )
    
    # Run indexing
    print("\\n🎬 Starting indexing process...")
    print("=" * 50)
    
    total_start_time = time.time()
    
    try:
        # Create indexer and run
        indexer = ColabPRIndexer(config)
        vector_store = indexer.run_indexing()
        
        # Calculate total time
        total_time = time.time() - total_start_time
        
        print("=" * 50)
        print(f"🎉 INDEXING COMPLETED SUCCESSFULLY!")
        print(f"⏱️ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Vector store contains {vector_store.index.ntotal} chunks")
        print(f"🔢 Vector dimension: {vector_store.index.d}")
        
        # Performance stats
        chunks_per_second = vector_store.index.ntotal / total_time
        print(f"⚡ Processing speed: {chunks_per_second:.1f} chunks/second")
        
        # Download results if in Colab
        if IN_COLAB:
            download_results()
        
        print("\\n🎯 Next steps:")
        print("1. Extract the vector store in your local project")
        print("2. Update your RAG system to use the new embeddings")
        print("3. Test with specific questions about your PRs")
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        print("💡 Try reducing batch size or using a smaller model")
        raise


if __name__ == "__main__":
    main()
