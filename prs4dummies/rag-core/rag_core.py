"""
RAG Core Module for PRs4Dummies (OpenAI Version)

This module contains the core logic for the RAG pipeline using OpenAI's API as the LLM.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

# --- CHANGED: Imports for OpenAI and environment variables ---
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# -----------------------------------------------------------

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- REMOVED: The entire LocalHuggingFaceLLM class and transformers/torch imports ---


class RAGCore:
    """Core RAG functionality for answering questions about pull requests."""
    
    def __init__(self, vector_store_path: str = None, embedding_model_name: str = None):
        """
        Initialize the RAG core.
        """
        self.vector_store_path = vector_store_path or "vector_store"
        self.embedding_model_name = embedding_model_name or "jinaai/jina-embeddings-v2-base-code"
        
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.rag_chain = None
        
        self._load_components()
    
    def _load_components(self):
        """Load all necessary components for the RAG pipeline."""
        try:
            # --- CHANGED: Load API key from .env file ---
            load_dotenv()
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")
            # -----------------------------------------------

            # Load embedding model (this part stays the same)
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                }
            )
            
            # Load FAISS vector store (this part stays the same)
            vector_store_dir = Path(self.vector_store_path)
            if not vector_store_dir.exists():
                raise FileNotFoundError(f"Vector store directory not found: {vector_store_dir}")
            
            logger.info("Loading FAISS vector store...")
            self.vector_store = FAISS.load_local(
                folder_path=str(vector_store_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded with {self.vector_store.index.ntotal} documents")
            
            # --- CHANGED: Initialize OpenAI LLM instead of local model ---
            logger.info("Initializing OpenAI LLM (gpt-3.5-turbo)...")
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0  # Set to 0 for more deterministic, fact-based answers
            )
            # -------------------------------------------------------------
            
            self._create_prompt_template()
            self._create_rag_chain()
            
            logger.info("RAG core initialized successfully")
            
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise
    
    def _create_prompt_template(self):
        """Create the prompt template for the RAG system."""
        # --- CHANGED: A stricter, more reliable prompt for RAG ---
        template = """You are an expert software engineer and AI assistant. Your task is to analyze pull requests from the Ansible repository.
Use the following pieces of retrieved context to answer the question. If you don't know the answer based on the context, just say that you don't know. Do not try to make up an answer.
Keep the answer concise and focus on the technical details.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        # ---------------------------------------------------------
        
        self.prompt_template = PromptTemplate.from_template(template)
    
    def _create_rag_chain(self):
        """Create the RAG chain."""
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 75}  # Can increase K for more powerful models like GPT
            )
            
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt_template},
                return_source_documents=True
            )
            
            logger.info("RAG chain created successfully")
            
        except Exception as e:
            logger.error(f"Error creating RAG chain: {e}")
            raise
    
    # --- No changes needed for the rest of the file (answer_question, get_vector_store_info, etc.) ---
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline."""
        try:
            if not self.rag_chain:
                raise RuntimeError("RAG chain not initialized")
            
            logger.info(f"Processing question: {question}")
            result = self.rag_chain.invoke({"query": question})
            
            answer = result.get("result", "No answer generated")
            source_documents = result.get("source_documents", [])
            
            sources = []
            for doc in source_documents:
                if hasattr(doc, 'metadata'):
                    sources.append({
                        "source": doc.metadata.get("source", "Unknown"),
                        "pr_number": doc.metadata.get("pr_number", "Unknown"),
                        "title": doc.metadata.get("title", "Unknown"),
                    })
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "total_sources": len(sources)
            }
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
        
        return {
            "total_documents": self.vector_store.index.ntotal,
            "embedding_model": self.embedding_model_name,
        }

if __name__ == "__main__":
    try:
        rag = RAGCore()
        print("RAG Core initialized successfully with OpenAI!")
        
        test_question = "Summarize the PR that limited bootstrap package install retries."
        result = rag.answer_question(test_question)
        
        print(f"\nTest Question: {test_question}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources: {result['sources']}")
        
    except Exception as e:
        print(f"Error testing RAG core: {e}")