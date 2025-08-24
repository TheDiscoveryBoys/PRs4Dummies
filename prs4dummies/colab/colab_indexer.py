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
    data_dir: str = "scraped_data_converted"
    output_dir: str = "vector_store"
    
    # Embedding model (optimized for Colab GPU)
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    
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
        """Setup logging for Colab with file output."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create a unique log file name with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"colab_indexer_{timestamp}.log"
        
        # Configure logging to write to file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()  # Keep console output for important messages
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log the log file location
        print(f"📝 Logging to file: {log_file}")
        self.logger.info(f"Logging started - output file: {log_file}")
        
        # Store log file path for later use
        self.log_file = log_file
        
    def _log_file_details(self, file_metadata: Dict[str, Any], file_content: str):
        """Log detailed file information to the text file."""
        try:
            # Create a detailed log entry
            log_entry = f"""
{'='*80}
FILE DOCUMENT ADDED
{'='*80}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
PR Number: {file_metadata.get('pr_number', 'Unknown')}
File Path: {file_metadata.get('file_path', 'Unknown')}
File Status: {file_metadata.get('file_status', 'Unknown')}
Additions: {file_metadata.get('file_additions', 0)}
Deletions: {file_metadata.get('file_deletions', 0)}
Content Length: {file_metadata.get('content_length', 0)} characters
Document Type: {file_metadata.get('document_type', 'Unknown')}

METADATA:
{json.dumps(file_metadata, indent=2, default=str)}

CONTENT:
{file_content}
{'='*80}

"""
            # Write to the log file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
            # Also log a summary to the console
            self.logger.info(f"Added file document for {file_metadata.get('file_path', 'Unknown')} (PR #{file_metadata.get('pr_number', 'Unknown')})")
            
        except Exception as e:
            # Fallback to regular logging if file writing fails
            self.logger.error(f"Failed to write detailed file log: {e}")
            self.logger.info(f"Added file document for {file_metadata}")
    
    def _log_pr_summary(self, pr_metadata: Dict[str, Any], pr_content: str):
        """Log detailed PR summary information to the text file."""
        try:
            # Create a detailed log entry for PR summary
            log_entry = f"""
{'='*80}
PR SUMMARY DOCUMENT ADDED
{'='*80}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
PR Number: {pr_metadata.get('pr_number', 'Unknown')}
Title: {pr_metadata.get('title', 'No title')}
Author: {pr_metadata.get('author_login', 'Unknown')}
Merged By: {pr_metadata.get('merged_by_login', 'Not merged')}
Labels: {pr_metadata.get('labels_str', 'No labels')}
Additions: {pr_metadata.get('additions', 0)}
Deletions: {pr_metadata.get('deletions', 0)}
Changed Files: {pr_metadata.get('changed_files_count', 0)}
Comments: {pr_metadata.get('comment_count', 0)}
Reviews: {pr_metadata.get('review_count', 0)}
Content Length: {pr_metadata.get('content_length', 0)} characters
Document Type: {pr_metadata.get('document_type', 'Unknown')}

METADATA:
{json.dumps(pr_metadata, indent=2, default=str)}

CONTENT:
{pr_content}
{'='*80}

"""
            # Write to the log file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
            # Also log a summary to the console
            self.logger.info(f"Added PR summary document for PR #{pr_metadata.get('pr_number', 'Unknown')}: {pr_metadata.get('title', 'No title')}")
            
        except Exception as e:
            # Fallback to regular logging if file writing fails
            self.logger.error(f"Failed to write detailed PR log: {e}")
            self.logger.info(f"Added PR summary document for PR #{pr_metadata.get('pr_number', 'Unknown')}")
    
    def _log_indexing_progress(self, message: str, level: str = "INFO"):
        """Log indexing progress to the text file."""
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {level}: {message}\n"
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            # Fallback to regular logging if file writing fails
            self.logger.error(f"Failed to write progress log: {e}")
    
    def _log_indexing_summary(self, pr_data_list: List[Dict[str, Any]], chunks: List[Document], vector_store: FAISS):
        """Log final indexing summary to the text file."""
        try:
            summary_entry = f"""
{'='*80}
INDEXING PROCESS COMPLETED
{'='*80}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total PRs Processed: {len(pr_data_list)}
Total Document Chunks Created: {len(chunks)}
Vector Store Dimensions: {vector_store.index.d}
Vector Store Total Documents: {vector_store.index.ntotal}
Device Used: {self.device}
Batch Size: {self.batch_size}
Embedding Model: {self.config.embedding_model}
Chunk Size: {self.config.chunk_size}
Chunk Overlap: {self.config.chunk_overlap}

PR STATISTICS:
{'-'*40}
"""
            # Add PR-level statistics
            pr_numbers = [pr.get('pr_number', 'Unknown') for pr in pr_data_list]
            total_additions = sum(pr.get('additions', 0) for pr in pr_data_list)
            total_deletions = sum(pr.get('deletions', 0) for pr in pr_data_list)
            total_comments = sum(len(pr.get('comments', [])) for pr in pr_data_list)
            total_reviews = sum(len(pr.get('reviews', [])) for pr in pr_data_list)
            
            summary_entry += f"""
Total Additions: {total_additions}
Total Deletions: {total_deletions}
Total Comments: {total_comments}
Total Reviews: {total_reviews}
PR Range: {min(pr_numbers) if pr_numbers else 'N/A'} to {max(pr_numbers) if pr_numbers else 'N/A'}

CHUNK STATISTICS:
{'-'*40}
"""
            # Add chunk-level statistics
            if chunks:
                chunk_lengths = [len(chunk.page_content) for chunk in chunks]
                avg_length = sum(chunk_lengths) / len(chunk_lengths)
                summary_entry += f"""
Average Chunk Length: {avg_length:.1f} characters
Min Chunk Length: {min(chunk_lengths)} characters
Max Chunk Length: {max(chunk_lengths)} characters
"""
            
            summary_entry += f"{'='*80}\n\n"
            
            # Write to the log file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(summary_entry)
                
            # Also log to console
            self.logger.info(f"Indexing completed successfully. Log saved to: {self.log_file}")
            
        except Exception as e:
            # Fallback to regular logging if file writing fails
            self.logger.error(f"Failed to write indexing summary: {e}")
    
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
                    "\n\n",      # Paragraph breaks
                    "\n###",     # Markdown subheadings
                    "\n##",      # Markdown headings
                    "\n```",     # Code blocks
                    "\n",        # Line breaks
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

    def format_pr_to_document(self, pr_data: Dict[str, Any]) -> List[Document]:
        """
        Transforms PR data into multiple, granular documents:
        1. A main document for the summary, metadata, and discussion.
        2. A separate document for EACH changed file's code patch.
        """
        pr_number = pr_data.get('pr_number', 'unknown')
        title = pr_data.get('title', 'No title')
        
        output_documents = []

        # --- Create rich base metadata for all related documents ---
        author = pr_data.get('author_login', 'Unknown')
        merged_by = pr_data.get('merged_by_login')
        labels = pr_data.get('labels', [])
        
        base_metadata = {
            "pr_number": pr_number,
            "title": title,
            "author_login": author,
            "merged_by_login": merged_by or "",
            "labels": labels,
            "labels_str": ", ".join(labels),
            "additions": pr_data.get('additions', 0), # Total additions
            "deletions": pr_data.get('deletions', 0), # Total deletions
            
            # --- CHANGED: Use the new 'changed_files_count' key ---
            "changed_files_count": pr_data.get('changed_files_count', 0),
            
            "comment_count": len(pr_data.get('comments', [])),
            "review_count": len(pr_data.get('reviews', [])),
            "source_type": "github_pr",
            "repository": "ansible/ansible",
            
            # --- CHANGED: Use the new 'url' key from the converted data ---
            "pr_url": pr_data.get('url', f"https://github.com/ansible/ansible/pull/{pr_number}"),
            "source": pr_data.get('url', f"https://github.com/ansible/ansible/pull/{pr_number}")
        }

        # --- Document 1: Summary and Discussion ---
        # This part remains largely the same, building a text block from the PR body and comments.
        content_parts = [f"# Pull Request #{pr_number}: {title}\n"]
        
        # Metadata section
        content_parts.append("## Pull Request Information")
        content_parts.append(f"**Author:** {author}")
        if merged_by:
            content_parts.append(f"**Merged by:** {merged_by}")
        if labels:
            content_parts.append(f"**Labels:** {', '.join(labels)}")
        content_parts.append("")
        
        # Description section
        body = pr_data.get('body', '').strip()
        if body:
            content_parts.append("## Description")
            content_parts.append(body)
            content_parts.append("")
            
        # Comments and reviews section (this logic can remain as is)
        comments = pr_data.get('comments', [])
        reviews = pr_data.get('reviews', [])
        if comments or reviews:
            content_parts.append("## Discussion")
            # (Your existing logic for formatting comments and reviews is fine here)
            for comment in comments:
                comment_author = comment.get('author_login', 'Unknown')
                comment_body = comment.get('body', '').strip()
                if comment_body:
                    content_parts.append(f"**{comment_author} commented:**\n{comment_body}\n")
            for review in reviews:
                review_author = review.get('author_login', 'Unknown')
                review_body = review.get('body', '').strip()
                if review_body:
                    content_parts.append(f"**{review_author} reviewed ({review.get('state')}):**\n{review_body}\n")

        main_content = "\n".join(content_parts)
        
        summary_metadata = base_metadata.copy()
        summary_metadata["document_type"] = "summary_discussion"
        summary_metadata["content_length"] = len(main_content)
        
        output_documents.append(Document(page_content=main_content, metadata=summary_metadata))
        
        # Log the PR summary document
        self._log_pr_summary(summary_metadata, main_content)
        
        # --- CHANGED: Create a separate document for EACH file change ---
        changed_files = pr_data.get('files', [])
        if changed_files:
            self._log_indexing_progress(f"Processing {len(changed_files)} changed files for PR #{pr_number}", "INFO")
            
            # Debug: Log the structure of the first file to understand the format
            if changed_files:
                first_file = changed_files[0]
                self._log_indexing_progress(
                    f"Sample file structure for PR #{pr_number}: keys={list(first_file.keys())}, "
                    f"has_file_diff={'file_diff' in first_file}, "
                    f"has_patch={'patch' in first_file}", 
                    "DEBUG"
                )
            
            for file_change in changed_files:
                file_path = file_change.get('file_path')
                # FIXED: Use 'file_diff' instead of 'patch' to match the actual JSON structure
                file_diff = file_change.get('file_diff', '').strip()
                
                if not file_path:
                    self._log_indexing_progress(f"Skipping file change with no path in PR #{pr_number}", "WARNING")
                    continue
                    
                if not file_diff:
                    self._log_indexing_progress(f"Skipping file change with no diff for {file_path} in PR #{pr_number}", "WARNING")
                    continue
                    
                # Create content specifically for this file's changes
                file_content = (
                    f"# File Change in PR #{pr_number}: `{file_path}`\n\n"
                    f"**Status:** {file_change.get('status', 'modified')}\n\n"
                    f"```diff\n"
                    f"{file_diff}\n"
                    f"```"
                )
                
                # Create specific metadata for this file document
                file_metadata = base_metadata.copy()
                file_metadata.update({
                    "document_type": "file_change",
                    "file_path": file_path,
                    "file_status": file_change.get('status'),
                    "file_additions": file_change.get('additions'),
                    "file_deletions": file_change.get('deletions'),
                    "content_length": len(file_content)
                })
                
                output_documents.append(Document(page_content=file_content, metadata=file_metadata))
                self._log_file_details(file_metadata, file_content)
                
                self._log_indexing_progress(f"Created file document for {file_path} in PR #{pr_number}", "INFO")
        else:
            self._log_indexing_progress(f"No changed files found for PR #{pr_number}", "INFO")
                
        return output_documents

    def process_documents(self, pr_data_list: List[Dict[str, Any]]) -> List[Document]:
        """Convert PR data to documents and split into chunks."""
        print(f"🔄 Processing {len(pr_data_list)} PRs into documents")
        self._log_indexing_progress(f"Processing {len(pr_data_list)} PRs into documents", "INFO")
        
        documents = []
        for pr_data in tqdm(pr_data_list, desc="Creating documents"):
            try:
                # --- CHANGED: Use extend to handle the list of documents from the updated function ---
                docs = self.format_pr_to_document(pr_data)
                documents.extend(docs)
                
                # Log detailed information about what was created for this PR
                pr_num = pr_data.get('pr_number', 'unknown')
                summary_docs = [d for d in docs if d.metadata.get('document_type') == 'summary_discussion']
                file_docs = [d for d in docs if d.metadata.get('document_type') == 'file_change']
                
                self._log_indexing_progress(
                    f"PR #{pr_num}: Created {len(summary_docs)} summary + {len(file_docs)} file documents = {len(docs)} total", 
                    "INFO"
                )
                
                # Log progress every 10 PRs
                if len(documents) % 10 == 0:
                    self._log_indexing_progress(f"Processed {len(documents)} documents so far", "INFO")
                    
            except Exception as e:
                error_msg = f"Error processing PR {pr_data.get('pr_number', 'unknown')}: {e}"
                self._log_indexing_progress(error_msg, "ERROR")
                self.logger.error(error_msg)
        
        print(f"✅ Created {len(documents)} documents (including separate diffs)")
        self._log_indexing_progress(f"Created {len(documents)} documents (including separate diffs)", "INFO")
        
        # Split into chunks
        print("✂️ Splitting documents into chunks...")
        self._log_indexing_progress("Splitting documents into chunks", "INFO")
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_hash"] = hashlib.md5(
                chunk.page_content.encode()
            ).hexdigest()[:12]
        
        print(f"✅ Created {len(chunks)} document chunks")
        self._log_indexing_progress(f"Created {len(chunks)} document chunks", "INFO")
        
        # Chunk statistics
        if chunks:
            lengths = [len(chunk.page_content) for chunk in chunks]
            avg_length = sum(lengths) / len(lengths)
            print(f"📊 Chunk statistics:")
            print(f"   Average length: {avg_length:.1f} characters")
            print(f"   Min length: {min(lengths)} characters")
            print(f"   Max length: {max(lengths)} characters")
            
            # Log chunk statistics
            self._log_indexing_progress(f"Chunk statistics - Avg: {avg_length:.1f}, Min: {min(lengths)}, Max: {max(lengths)}", "INFO")
        
        return chunks

    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """Create FAISS vector store from document chunks."""
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")
        
        print(f"🚀 Creating FAISS vector store from {len(chunks)} chunks...")
        start_time = time.time()
        
        try:
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
        
        vector_store.save_local(str(output_path))
        print(f"💾 Vector store saved to {output_path}")
        
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
        self._log_indexing_progress("=== Starting Ansible PR Indexing Process ===", "INFO")
        
        try:
            print(f"📁 Loading PR data from {self.config.data_dir}")
            self._log_indexing_progress(f"Loading PR data from {self.config.data_dir}", "INFO")
            pr_data_list = self.load_pr_data(self.config.data_dir)
            
            self._log_indexing_progress(f"Loaded {len(pr_data_list)} PR files", "INFO")
            chunks = self.process_documents(pr_data_list)
            
            if not chunks:
                raise ValueError("No valid document chunks created")
            
            self._log_indexing_progress(f"Created {len(chunks)} document chunks", "INFO")
            vector_store = self.create_vector_store(chunks)
            
            self._log_indexing_progress("Vector store created successfully", "INFO")
            self.save_vector_store(vector_store, self.config.output_dir)
            
            # Log final summary
            self._log_indexing_summary(pr_data_list, chunks, vector_store)
            
            print("🎉 === Indexing Process Completed Successfully ===")
            print(f"📊 Total PRs processed: {len(pr_data_list)}")
            print(f"📄 Document chunks created: {len(chunks)}")
            print(f"💾 Vector store saved to: {self.config.output_dir}")
            print(f"📝 Detailed log saved to: {self.log_file}")
            
            return vector_store
            
        except Exception as e:
            error_msg = f"Indexing process failed: {e}"
            self._log_indexing_progress(error_msg, "ERROR")
            self.logger.error(error_msg)
            raise


def check_gpu():
    """Check GPU availability."""
    print("🔍 Checking GPU availability...")
    if torch.cuda.is_available():
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
    has_gpu = check_gpu()
    
    try:
        import faiss
        print("✅ FAISS already installed")
    except ImportError:
        print("📦 Installing FAISS...")
        os.system("pip install faiss-gpu-cu121" if has_gpu else "pip install faiss-cpu")
    
    dependencies = ["langchain", "langchain-community", "langchain-huggingface", "sentence-transformers", "transformers", "einops", "tqdm"]
    os.system(f"pip install -q {' '.join(dependencies)}")
    
    return has_gpu


def recommend_settings(num_files: int, has_gpu: bool):
    """Recommend optimal settings based on data size and hardware."""
    if num_files <= 50:
        model = "all-MiniLM-L6-v2"
        batch_size = 32 if has_gpu else 8
        time_estimate = "1-2 minutes"
    elif num_files <= 500:
        model = "jinaai/jina-embeddings-v2-base-code"
        batch_size = 64 if has_gpu else 16
        time_estimate = "5-10 minutes" if has_gpu else "20-40 minutes"
    else:
        model = "jinaai/jina-embeddings-v2-base-code"
        batch_size = 128 if has_gpu else 16
        time_estimate = "15-30 minutes" if has_gpu else "1-2 hours"
    
    print(f"\n📊 Dataset size: {num_files} PRs")
    print(f"🤖 Recommended model: {model}")
    print(f"⚡ Recommended batch size: {batch_size}")
    print(f"⏱️ Estimated time: {time_estimate}")
    
    return model, batch_size


def upload_data():
    """Handle data upload for Colab."""
    if not IN_COLAB:
        print("ℹ️ Not in Colab - ensure your data is in the 'scraped_data' directory")
        return
    
    print("\n📤 Upload your scraped PR data (JSON files or a single ZIP file):")
    uploaded = files.upload()
    
    data_dir = Path("scraped_data_converted")
    data_dir.mkdir(exist_ok=True)
    
    import zipfile
    for filename in uploaded.keys():
        filepath = Path(filename)
        if filepath.suffix.lower() == '.zip':
            print(f"📦 Extracting ZIP file: {filename}")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            filepath.unlink() # Clean up the zip
        else:
            Path(filename).rename(data_dir / filename)


def download_results():
    """Create downloadable archive for Colab."""
    if not IN_COLAB:
        print("ℹ️ Vector store saved locally")
        return
    import shutil
    
    print("\n📦 Creating downloadable archive...")
    archive_name = "vector_store"
    shutil.make_archive(archive_name, 'zip', 'vector_store')
    
    archive_size = Path(f"{archive_name}.zip").stat().st_size / (1024 * 1024)
    print(f"📦 Total archive size: {archive_size:.1f} MB")

    print("\n⬇️ Downloading vector store...")
    files.download(f"{archive_name}.zip")
    print("✅ Download complete!")


def main():
    """Main function for running in Colab or standalone."""
    print("🚀 PRs4Dummies - Colab Indexer")
    print("=" * 40)
    
    has_gpu = setup_colab_environment()
    
    if IN_COLAB:
        upload_data()
    
    data_dir = Path("scraped_data_converted")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("❌ No JSON files found. Please upload your scraped PR data.")
        return
    
    json_files = list(data_dir.glob("*.json"))
    model, batch_size = recommend_settings(len(json_files), has_gpu)
    
    config = ColabIndexingConfig(embedding_model=model, batch_size=batch_size, device="cuda" if has_gpu else "cpu")
    
    print("\n🎬 Starting indexing process...")
    print("=" * 50)
    total_start_time = time.time()
    
    try:
        indexer = ColabPRIndexer(config)
        vector_store = indexer.run_indexing()
        
        total_time = time.time() - total_start_time
        print("=" * 50)
        print(f"🎉 INDEXING COMPLETED SUCCESSFULLY!")
        print(f"⏱️ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Vector store contains {vector_store.index.ntotal} chunks")
        
        if IN_COLAB:
            download_results()
            
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        raise

if __name__ == "__main__":
    main()