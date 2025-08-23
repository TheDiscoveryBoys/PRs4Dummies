"""
RAG Core Module for PRs4Dummies (OpenAI Version)

This module contains the core logic for the RAG pipeline using OpenAI's API as the LLM.
"""

import os
import logging
import pickle
from typing import List, Dict, Any
from pathlib import Path

# --- CHANGED: Imports for OpenAI and environment variables ---
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
# -----------------------------------------------------------

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGCore:
    """Core RAG functionality for answering questions about pull requests."""
    
    def __init__(self, vector_store_path: str = None, embedding_model_name: str = None):
        """
        Initialize the RAG core.
        """
        self.vector_store_path = vector_store_path or "vector_store"
        self.embedding_model_name = embedding_model_name or "BAAI/bge-m3"
        
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.rag_chain = None
        self.all_chunks = None
        
        self._load_components()
    
    def _load_components(self):
        """Load all necessary components for the RAG pipeline."""
        try:
            load_dotenv()
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

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
            chunks_path = vector_store_dir / "chunks.pkl"
            if not chunks_path.exists():
                raise FileNotFoundError(f"chunks.pkl not found in {vector_store_dir}. Please re-run your indexer.")
            
            logger.info("Loading raw document chunks for keyword search...")
            with open(chunks_path, "rb") as f:
                self.all_chunks = pickle.load(f)
            logger.info(f"Loaded {len(self.all_chunks)} chunks successfully.")
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
        template = """You are an expert Principal Software Engineer and AI assistant. Your task is to provide an insightful analysis of a pull request based on the provided context.
        The importance of the instructions are ranked with 0 being the most important and 5 being the least important.
        **Instructions:**
        0.  **Simple Answer:** If the question is simple, provide a simple answer. It may not be related to the diff of the PR, for example, it may be a question about the PR title or description.
        1.  **Summarize:** Begin with a concise summary of the main purpose of the pull request.
        2.  **Analyze the "Why":** Based on the code diff and description, infer the developer's likely intent. Why did they make this change? What problem does it solve?
        3.  **Speculate on Impact:** Use your general software engineering knowledge to speculate on the potential impact of the changes. Consider aspects like performance, maintainability, potential bugs, or improvements to code quality.
        4.  **Grounding:** Base your analysis primarily on the provided context, but use your expert knowledge to interpret the code and infer intent.

        **CONTEXT FROM THE PULL REQUEST:**
        {context}

        **QUESTION:**
        {question}

        **EXPERT ANALYSIS:**
        """
        # ---------------------------------------------------------
        
        self.prompt_template = PromptTemplate.from_template(template)
    
    def _create_rag_chain(self):
        """Create the RAG chain using a hybrid EnsembleRetriever."""
        try:
            if not self.all_chunks:
                raise ValueError("Document chunks not loaded. Cannot create BM25 retriever.")

            # 1. Set up the Keyword Retriever (BM25)
            # logger.info("Initializing BM25 keyword retriever...")
            # bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
            # bm25_retriever.k = 10

            # 2. Set up the Semantic Retriever (FAISS)
            logger.info("Initializing FAISS semantic retriever...")
            faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})

            # 3. Initialize the EnsembleRetriever
            logger.info("Initializing EnsembleRetriever with weights...")
            self.ensemble_retriever = EnsembleRetriever(
                # retrievers=[bm25_retriever, faiss_retriever],
                retrievers=[faiss_retriever],
                weights=[1]  # Give 75% weight to keyword matches
            )

            # 4. Create the RAG chain
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.ensemble_retriever, # Use the ensemble retriever
                chain_type_kwargs={"prompt": self.prompt_template},
                return_source_documents=True
            )
            
            logger.info("RAG chain created successfully with EnsembleRetriever")
            
        except Exception as e:
            logger.error(f"Error creating RAG chain: {e}")
            raise
            
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