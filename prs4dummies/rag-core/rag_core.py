"""
RAG Core Module for PRs4Dummies

This module contains the core logic for the RAG (Retrieval-Augmented Generation) pipeline
that answers questions about pull requests using the pre-built FAISS index.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

# For local HuggingFace models
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalHuggingFaceLLM(LLM):
    """Local HuggingFace LLM wrapper for LangChain compatibility."""

    model_name: str = "google/flan-t5-base" # <-- Change the model name
    max_length: int = 512

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = None

    def _load_model(self):
        """Load the T5 model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            self._pipeline = pipeline(
                "text2text-generation", # <-- Change the pipeline task
                model=model,
                tokenizer=tokenizer,
                max_length=self.max_length,
                truncation=True,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "local_huggingface_t5"

    def _call(self, prompt: str, stop: List[str] = None, **kwargs) -> str:
        """Generate text using the local T5 model."""
        if self._pipeline is None:
            self._load_model()

        try:
            # Generate response. T5 models are much better at just returning the answer.
            result = self._pipeline(prompt)
            response = result[0]['generated_text']

            logger.info(f"Response preview: {response[:150]}...")
            return response if response else "The model returned an empty response."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"


class RAGCore:
    """Core RAG functionality for answering questions about pull requests."""
    
    def __init__(self, vector_store_path: str = None, embedding_model_name: str = None):
        """
        Initialize the RAG core.
        
        Args:
            vector_store_path: Path to the FAISS vector store
            embedding_model_name: Name of the embedding model to use
        """
        self.vector_store_path = vector_store_path or "vector_store"
        self.embedding_model_name = embedding_model_name or "nomic-ai/nomic-embed-text-v1.5"
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.rag_chain = None
        
        # Load components
        self._load_components()
    
    def _load_components(self):
        """Load all necessary components for the RAG pipeline."""
        try:
            # Load embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={
                    'device': 'cpu',  # Use CPU for compatibility
                    'trust_remote_code': True  # Required for some models like nomic-embed
                }
            )
            
            # Load FAISS vector store
            vector_store_dir = Path(self.vector_store_path)
            if not vector_store_dir.exists():
                raise FileNotFoundError(f"Vector store directory not found: {vector_store_dir}")
            
            logger.info("Loading FAISS vector store...")
            self.vector_store = FAISS.load_local(
                folder_path=str(vector_store_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True  # Safe since this is our own vector store
            )
            logger.info(f"Vector store loaded with {self.vector_store.index.ntotal} documents")
            
            # Initialize LLM (using local HuggingFace model for demo)
            logger.info("Initializing local LLM...")
            self.llm = LocalHuggingFaceLLM(
                model_name="google/flan-t5-base",  # Smaller, more efficient GPT-2 variant
                max_length=512  # Conservative limit
            )
            
            # Create prompt template
            self._create_prompt_template()
            
            # Create RAG chain
            self._create_rag_chain()
            
            logger.info("RAG core initialized successfully")
            
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise
    
    def _create_prompt_template(self):
        """Create the prompt template for the RAG system."""
#         template = """You are an expert software engineer analyzing pull requests from the Ansible repository.

# Your task is to answer questions mainly using the context provided from the pull request data - prioritise using this data in all scenarios.
# If you do not use that pull request data, provide an answer with a disclaimer that says "The following information has been sourced from outside the ansible repository".

# IMPORTANT RULES:
# 1. Base your answer ONLY on the provided context if possible. If not possible, refer to your knowledge with the disclaimer included in the response.
# 2. If you're not sure about something, say so
# 3. Be specific and reference the pull request details when possible
# 4. Focus on the technical aspects and code changes
# 5. If asked about code, explain the changes clearly
# 6. Always start your answer with a clear, direct response to the question

# CONTEXT (Pull Request Information):
# {context}

# QUESTION: {question}

# ANSWER:"""
        template = """You are an expert software engineer analyzing pull requests from the Ansible repository.

        Answer the questions.

        CONTEXT (Pull Request Information):
        {context}

        QUESTION: {question}

        ANSWER:"""
        
        self.prompt_template = PromptTemplate.from_template(template)
    
    def _create_rag_chain(self):
        """Create the RAG chain using LangChain Expression Language."""
        try:
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}  # Retrieve top 2 most relevant chunks to minimize prompt length
            )
            
            # Create the RAG chain
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": self.prompt_template,
                    "verbose": True
                },
                return_source_documents=True
            )
            
            logger.info("RAG chain created successfully")
            
        except Exception as e:
            logger.error(f"Error creating RAG chain: {e}")
            raise
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline
        
        Args:
            question: The user's question about pull requests
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            if not self.rag_chain:
                raise RuntimeError("RAG chain not initialized")
            
            logger.info(f"Processing question: {question}")
            
            # Get answer from RAG chain
            result = self.rag_chain({"query": question})
            
            # Extract answer and source documents
            answer = result.get("result", "No answer generated")
            source_documents = result.get("source_documents", [])
            
            # Format source information
            sources = []
            for doc in source_documents:
                if hasattr(doc, 'metadata'):
                    pr_number = doc.metadata.get("pr_number", "Unknown")
                    pr_url = doc.metadata.get("source", doc.metadata.get("pr_url", "Unknown"))
                    title = doc.metadata.get("title", "")
                    
                    # Create a meaningful source description
                    if pr_number != "Unknown" and title:
                        source_description = f"PR #{pr_number}: {title}"
                    elif pr_number != "Unknown":
                        source_description = f"PR #{pr_number}"
                    else:
                        source_description = "Unknown PR"
                    
                    sources.append({
                        "source": pr_url,
                        "source_description": source_description,
                        "pr_number": pr_number,
                        "title": title,
                        "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                        "relevance_score": doc.metadata.get("score", "N/A")
                    })
            
            response = {
                "answer": answer,
                "sources": sources,
                "question": question,
                "total_sources": len(sources)
            }
            
            logger.info(f"Generated answer with {len(sources)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "question": question,
                "total_sources": 0,
                "error": str(e)
            }
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the loaded vector store."""
        if not self.vector_store:
            return {"error": "Vector store not loaded"}
        
        try:
            return {
                "total_documents": self.vector_store.index.ntotal,
                "embedding_dimension": self.vector_store.index.d,
                "embedding_model": self.embedding_model_name,
                "vector_store_path": self.vector_store_path
            }
        except Exception as e:
            return {"error": f"Error getting vector store info: {str(e)}"}

# Convenience function for quick testing
def create_rag_core(vector_store_path: str = None, embedding_model_name: str = None) -> RAGCore:
    """Create and return a RAG core instance."""
    return RAGCore(vector_store_path, embedding_model_name)

if __name__ == "__main__":
    # Test the RAG core
    try:
        rag = create_rag_core()
        print("RAG Core initialized successfully!")
        
        # Test with a sample question
        test_question = "What are some common types of pull requests in the Ansible repository?"
        result = rag.answer_question(test_question)
        
        print(f"\nTest Question: {test_question}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        
    except Exception as e:
        print(f"Error testing RAG core: {e}")
