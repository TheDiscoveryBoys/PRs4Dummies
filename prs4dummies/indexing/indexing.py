#!/usr/bin/env python3
"""
Ansible PR Data Indexer for RAG System

This script loads the LLM-optimized JSON data scraped by scraper.py, processes it into
clean, searchable documents, chunks them strategically, and creates a FAISS vector
store optimized for RAG retrieval.

Key improvements over basic implementations:
- Handles the LLM-optimized JSON format from our scraper
- Smart chunking that preserves context boundaries
- Better embedding model selection
- Comprehensive error handling and validation
- Metadata optimization for filtering and ranking

Dependencies:
- langchain
- sentence-transformers
- faiss-cpu (or faiss-gpu if available)
- tqdm
- python-dotenv

Usage:
    python indexing.py --data-dir ../scraper/scraped_data --output-dir vector_store
"""

import os
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

try:
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from tqdm import tqdm
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install langchain langchain-community sentence-transformers faiss-cpu tqdm python-dotenv")
    exit(1)


@dataclass
class IndexingConfig:
    """Configuration for the indexing process."""
    # Embedding model - optimized for code and technical content
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"  # Better than MiniLM for technical content
    
    # Chunking configuration
    chunk_size: int = 1500  # Larger chunks for better context
    chunk_overlap: int = 300  # More overlap to preserve relationships
    
    # Processing options
    batch_size: int = 50  # Process embeddings in batches for memory efficiency
    max_retries: int = 3  # Retry failed operations
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda", or "mps" (for Apple Silicon)
    force_cpu: bool = False  # Force CPU even if GPU is available
    
    # Output options
    save_metadata: bool = True
    validate_chunks: bool = True


class PRIndexer:
    """Main indexer class for processing and indexing PR data."""
    
    def __init__(self, config: IndexingConfig = None):
        self.config = config or IndexingConfig()
        self._setup_logging()
        load_dotenv()  # Load environment variables
        
        # Device detection and configuration
        self.device = self._detect_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.embeddings = None
        self.text_splitter = None
        self._init_components()
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('indexing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _detect_device(self) -> str:
        """
        Detect the best available device for processing.
        
        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        if self.config.force_cpu:
            self.logger.info("Forcing CPU usage as requested")
            return "cpu"
            
        if self.config.device != "auto":
            # User specified a device, validate and use it
            requested_device = self.config.device.lower()
            if requested_device in ["cpu", "cuda", "mps"]:
                self.logger.info(f"Using user-specified device: {requested_device}")
                return requested_device
            else:
                self.logger.warning(f"Invalid device '{self.config.device}', falling back to auto-detection")
        
        # Auto-detect best available device
        try:
            import torch
            
            # Check for CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                self.logger.info(f"CUDA available: {gpu_count} GPU(s) detected - {gpu_name}")
                
                # Special handling for Google Colab
                if self._is_google_colab():
                    self.logger.info("Google Colab environment detected - optimizing for Colab GPU")
                    # Colab often has memory constraints, so we'll adjust batch size
                    if self.config.batch_size > 32:
                        self.config.batch_size = 32
                        self.logger.info(f"Adjusted batch size to {self.config.batch_size} for Colab")
                
                return "cuda"
            
            # Check for MPS (Apple Silicon GPU)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.logger.info("Apple Silicon MPS available")
                return "mps"
            
            else:
                self.logger.info("No GPU acceleration available, using CPU")
                return "cpu"
                
        except ImportError:
            self.logger.warning("PyTorch not available for device detection, using CPU")
            return "cpu"
        except Exception as e:
            self.logger.warning(f"Device detection failed: {e}, using CPU")
            return "cpu"
    
    def _is_google_colab(self) -> bool:
        """Check if running in Google Colab environment."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
        
    def _init_components(self):
        """Initialize embedding model and text splitter."""
        try:
            self.logger.info(f"Initializing embedding model: {self.config.embedding_model}")
            
            # Configure model for the detected device
            model_kwargs = {
                'device': self.device,
                'trust_remote_code': True  # Required for some models like nomic-embed
            }
            encode_kwargs = {'normalize_embeddings': True}  # Important for FAISS
            
            # Add device-specific optimizations
            if self.device == "cuda":
                # GPU optimizations
                encode_kwargs['batch_size'] = min(self.config.batch_size, 64)  # Prevent GPU memory issues
                model_kwargs['torch_dtype'] = 'float16'  # Use half precision for GPU efficiency
                self.logger.info("Using GPU optimizations: float16 precision, controlled batch size")
            elif self.device == "mps":
                # Apple Silicon optimizations
                encode_kwargs['batch_size'] = min(self.config.batch_size, 32)  # Conservative for MPS
                self.logger.info("Using Apple Silicon MPS optimizations")
            else:
                # CPU optimizations
                encode_kwargs['batch_size'] = min(self.config.batch_size, 16)  # Smaller batches for CPU
                self.logger.info("Using CPU optimizations: smaller batch sizes")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # Smart text splitter that respects code boundaries
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=[
                    "\n## ",  # Section breaks
                    "\n### ",  # Subsection breaks
                    "\n```\n",  # Code block boundaries
                    "\n\n",  # Paragraph breaks
                    "\n",  # Line breaks
                    ". ",  # Sentence breaks
                    " "  # Word breaks
                ],
                length_function=len,
            )
            
            self.logger.info("Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
            
    def load_pr_data(self, data_dir: Path) -> List[Dict[str, Any]]:
        """
        Load and validate PR JSON files.
        
        Args:
            data_dir: Directory containing PR JSON files
            
        Returns:
            List of validated PR data dictionaries
        """
        self.logger.info(f"Loading PR data from {data_dir}")
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        # Find PR JSON files (our scraper creates pr_*.json files)
        json_files = list(data_dir.glob("pr_*.json"))
        if not json_files:
            raise ValueError(f"No PR JSON files found in {data_dir}")
            
        self.logger.info(f"Found {len(json_files)} PR files")
        
        pr_data_list = []
        failed_files = []
        
        for json_file in tqdm(json_files, desc="Loading PR files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    pr_data = json.load(f)
                    
                # Validate the PR data structure (our LLM-optimized format)
                if self._validate_pr_data(pr_data):
                    pr_data_list.append(pr_data)
                    self.logger.debug(f"Loaded PR #{pr_data.get('pr_number', 'unknown')}")
                else:
                    failed_files.append(json_file)
                    self.logger.warning(f"Invalid PR data structure in {json_file}")
                    
            except json.JSONDecodeError as e:
                failed_files.append(json_file)
                self.logger.error(f"JSON decode error in {json_file}: {e}")
            except Exception as e:
                failed_files.append(json_file)
                self.logger.error(f"Error loading {json_file}: {e}")
                
        if failed_files:
            self.logger.warning(f"Failed to load {len(failed_files)} files: {[f.name for f in failed_files]}")
            
        self.logger.info(f"Successfully loaded {len(pr_data_list)} PRs")
        return pr_data_list
        
    def _validate_pr_data(self, pr_data: Dict[str, Any]) -> bool:
        """
        Validate PR data structure matches our LLM-optimized format.
        
        Args:
            pr_data: PR data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['pr_number', 'title', 'body']
        
        for field in required_fields:
            if field not in pr_data:
                self.logger.debug(f"Missing required field: {field}")
                return False
                
        # Check for expected structure
        if not isinstance(pr_data.get('comments', []), list):
            return False
        if not isinstance(pr_data.get('reviews', []), list):
            return False
            
        return True
        
    def format_pr_to_document(self, pr_data: Dict[str, Any]) -> Document:
        """
        Transform PR data into a clean, structured document optimized for RAG.
        
        Args:
            pr_data: PR data in our LLM-optimized format
            
        Returns:
            LangChain Document with structured content and rich metadata
        """
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
            for comment in comments:
                if comment.get('comment_type') != 'review_comment':  # Skip inline comments for now
                    author = comment.get('author_login', 'Unknown')
                    body = comment.get('body', '').strip()
                    if body:
                        content_parts.append(f"**{author} commented:**")
                        content_parts.append(body)
                        content_parts.append("")
            
            # Reviews
            for review in reviews:
                author = review.get('author_login', 'Unknown')
                state = review.get('state', 'COMMENTED')
                body = review.get('body', '').strip()
                if body:
                    content_parts.append(f"**{author} reviewed ({state}):**")
                    content_parts.append(body)
                    content_parts.append("")
                    
            # Inline review comments (code-specific)
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
        content = "\n".join(content_parts)
        
        # Create rich metadata for filtering and ranking
        metadata = {
            "pr_number": pr_number,
            "title": title,
            "author_login": author,
            "merged_by_login": merged_by or "",
            "labels": labels,
            "labels_str": ", ".join(labels),  # For easy filtering
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
            "source": f"https://github.com/ansible/ansible/pull/{pr_number}"  # For RAG source tracking
        }
        
        return Document(page_content=content, metadata=metadata)
        
    def process_documents(self, pr_data_list: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert PR data to documents and split into chunks.
        
        Args:
            pr_data_list: List of PR data dictionaries
            
        Returns:
            List of document chunks ready for embedding
        """
        self.logger.info(f"Processing {len(pr_data_list)} PRs into documents")
        
        # Convert to documents
        documents = []
        for pr_data in tqdm(pr_data_list, desc="Creating documents"):
            try:
                doc = self.format_pr_to_document(pr_data)
                documents.append(doc)
            except Exception as e:
                pr_num = pr_data.get('pr_number', 'unknown')
                self.logger.error(f"Failed to process PR #{pr_num}: {e}")
                
        self.logger.info(f"Created {len(documents)} documents")
        
        # Split into chunks
        self.logger.info("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_hash'] = hashlib.md5(chunk.page_content.encode()).hexdigest()[:12]
            
        self.logger.info(f"Created {len(chunks)} document chunks")
        
        if self.config.validate_chunks:
            self._validate_chunks(chunks)
            
        return chunks
        
    def _validate_chunks(self, chunks: List[Document]):
        """Validate chunk quality and log statistics."""
        if not chunks:
            self.logger.warning("No chunks created!")
            return
            
        # Calculate statistics
        lengths = [len(chunk.page_content) for chunk in chunks]
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
        
        self.logger.info(f"Chunk statistics:")
        self.logger.info(f"  Average length: {avg_length:.1f} characters")
        self.logger.info(f"  Min length: {min_length} characters")
        self.logger.info(f"  Max length: {max_length} characters")
        
        # Check for very small or very large chunks
        tiny_chunks = sum(1 for length in lengths if length < 100)
        huge_chunks = sum(1 for length in lengths if length > self.config.chunk_size * 1.5)
        
        if tiny_chunks > 0:
            self.logger.warning(f"Found {tiny_chunks} very small chunks (< 100 chars)")
        if huge_chunks > 0:
            self.logger.warning(f"Found {huge_chunks} oversized chunks (> {self.config.chunk_size * 1.5} chars)")
            
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """
        Create FAISS vector store from document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            FAISS vector store
        """
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")
            
        self.logger.info(f"Creating FAISS vector store from {len(chunks)} chunks...")
        start_time = time.time()
        
        try:
            # Create vector store in batches for memory efficiency
            if len(chunks) <= self.config.batch_size:
                vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                # Process in batches
                self.logger.info(f"Processing in batches of {self.config.batch_size}")
                first_batch = chunks[:self.config.batch_size]
                vector_store = FAISS.from_documents(first_batch, self.embeddings)
                
                # Add remaining chunks in batches
                for i in range(self.config.batch_size, len(chunks), self.config.batch_size):
                    batch = chunks[i:i + self.config.batch_size]
                    batch_store = FAISS.from_documents(batch, self.embeddings)
                    vector_store.merge_from(batch_store)
                    
                    progress = min(i + self.config.batch_size, len(chunks))
                    self.logger.info(f"Processed {progress}/{len(chunks)} chunks")
                    
            end_time = time.time()
            self.logger.info(f"Vector store creation completed in {end_time - start_time:.2f} seconds")
            
            return vector_store
            
        except Exception as e:
            self.logger.error(f"Failed to create vector store: {e}")
            raise
            
    def save_vector_store(self, vector_store: FAISS, output_dir: Path):
        """
        Save FAISS vector store and metadata.
        
        Args:
            vector_store: FAISS vector store to save
            output_dir: Directory to save the vector store
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the FAISS index
            vector_store.save_local(str(output_dir))
            self.logger.info(f"Vector store saved to {output_dir}")
            
            # Save indexing metadata
            if self.config.save_metadata:
                metadata = {
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding_model": self.config.embedding_model,
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                    "total_chunks": vector_store.index.ntotal,
                    "vector_dimension": vector_store.index.d
                }
                
                metadata_file = output_dir / "index_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                    
                self.logger.info(f"Metadata saved to {metadata_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to save vector store: {e}")
            raise
            
    def run_indexing(self, data_dir: Path, output_dir: Path) -> None:
        """
        Main indexing workflow.
        
        Args:
            data_dir: Directory containing scraped PR JSON files
            output_dir: Directory to save the vector store
        """
        try:
            self.logger.info("=== Starting Ansible PR Indexing Process ===")
            
            # Load and validate data
            pr_data_list = self.load_pr_data(data_dir)
            if not pr_data_list:
                raise ValueError("No valid PR data found")
                
            # Process into documents and chunks
            chunks = self.process_documents(pr_data_list)
            if not chunks:
                raise ValueError("No document chunks created")
                
            # Create vector store
            vector_store = self.create_vector_store(chunks)
            
            # Save results
            self.save_vector_store(vector_store, output_dir)
            
            self.logger.info("=== Indexing Process Completed Successfully ===")
            self.logger.info(f" Total PRs processed: {len(pr_data_list)}")
            self.logger.info(f" Document chunks created: {len(chunks)}")
            self.logger.info(f" Vector store saved to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Indexing process failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index Ansible PR data for RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detects best device)
  python indexing.py --data-dir ../scraper/scraped_data
  
  # Force CPU usage (for testing)
  python indexing.py --device cpu --batch-size 16
  
  # Use GPU explicitly
  python indexing.py --device cuda --batch-size 64
  
  # Google Colab optimization
  python indexing.py --device cuda --model all-MiniLM-L6-v2 --batch-size 32
        """
    )
    
    parser.add_argument("--data-dir", default="../scraper/scraped_data",
                       help="Directory containing scraped PR JSON files")
    parser.add_argument("--output-dir", default="vector_store",
                       help="Directory to save the FAISS vector store")
    parser.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5",
                       help="Hugging Face embedding model name")
    parser.add_argument("--chunk-size", type=int, default=1500,
                       help="Maximum chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=300,
                       help="Overlap between chunks in characters")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for processing embeddings")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use: auto (detect), cpu, cuda (GPU), or mps (Apple Silicon)")
    parser.add_argument("--force-cpu", action="store_true",
                       help="Force CPU usage even if GPU is available")
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip chunk validation")
    
    args = parser.parse_args()
    
    # Create configuration
    config = IndexingConfig(
        embedding_model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        device=args.device,
        force_cpu=args.force_cpu,
        validate_chunks=not args.no_validate
    )
    
    # Initialize indexer and run
    indexer = PRIndexer(config)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    indexer.run_indexing(data_dir, output_dir)


if __name__ == "__main__":
    main()
