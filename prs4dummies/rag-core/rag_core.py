import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# LangChain and environment imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

class RAGCore:
    """Core RAG functionality for answering questions about pull requests."""
    
    def __init__(self, vector_store_path: str = None, embedding_model_name: str = None):
        """Initialize the RAG core."""
        # --- ADDED: Setup logging first ---
        self._setup_logging()

        self.vector_store_path = vector_store_path or "vector_store"
        self.embedding_model_name = embedding_model_name or "jinaai/jina-embeddings-v2-base-code"
        
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.prompt_template = None
        self.generic_rag_chain = None
        self.specific_qa_chain = None
        
        self._load_components()

    # --- ADDED: Logging setup method ---
    def _setup_logging(self):
        """Setup logging for the RAGCore class."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_components(self):
        """Load all necessary components for the RAG pipeline."""
        try:
            load_dotenv()
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in .env file.")

            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu', 'trust_remote_code': True}
            )
            
            self.logger.info("Loading FAISS vector store...")
            vector_store_dir = Path(self.vector_store_path)
            if not vector_store_dir.exists():
                raise FileNotFoundError(f"Vector store not found: {vector_store_dir}")
            self.vector_store = FAISS.load_local(
                folder_path=str(vector_store_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.logger.info(f"Vector store loaded with {self.vector_store.index.ntotal} documents.")
            
            self.logger.info("Initializing OpenAI LLM (gpt-3.5-turbo)...")
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
            self._create_prompt_template()
            self._create_chains()
            
            self.logger.info("RAG core initialized successfully.")
        except Exception as e:
            # --- FIXED: Use self.logger instead of global logger ---
            self.logger.error(f"Error loading components: {e}")
            raise

    def _create_prompt_template(self):
        """Create a more robust prompt template."""
        template = """You are an expert AI assistant specializing in software engineering and code analysis. 
        Your task is to answer questions about GitHub Pull Requests based *only* on the context provided.

        **Context from one or more Pull Requests:**
        {context}

        **Instructions:**
        - Analyze the context above to answer the following question.
        - If the question is about a specific PR number, ensure your answer is based *exclusively* on the documents for that PR.
        - If the provided context is insufficient to answer the question, state that you do not have enough information. Do not make up information.
        - Structure your answer clearly and concisely.

        **Question:**
        {question}

        **Answer:**
        """
        self.prompt_template = PromptTemplate.from_template(template)

    def _create_chains(self):
        """Create the RAG chains for both generic and specific queries."""
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10} 
        )
        self.generic_rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )

        self.specific_qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=self.prompt_template
        )
        self.logger.info("RAG chains created successfully.")
    
    def _get_pr_specific_docs(self, question: str) -> Tuple[Optional[int], List[Document]]:
        """
        Detects a PR number in a question and retrieves all associated documents.
        """
        # --- IMPROVED: Updated regex to handle "PR number 123" ---
        match = re.search(r'(?:pr|pull request|pr-)\s*(?:number\s*)?#?(\d+)', question, re.IGNORECASE)
        
        if not match:
            self.logger.info("No specific PR number detected in query. Using generic similarity search.")
            return None, []
        
        pr_number = int(match.group(1))
        self.logger.info(f"Detected specific query for PR #{pr_number}.")

        # Fetch a large number of docs and filter them by metadata in Python.
        # This is a robust way to ensure all chunks for a specific PR are found.
        all_docs = self.vector_store.similarity_search(question, k=200)
        
        pr_specific_docs = [
            doc for doc in all_docs 
            if doc.metadata.get("pr_number") == pr_number
        ]
        
        self.logger.info(f"Retrieved {len(pr_specific_docs)} document chunks for PR #{pr_number}.")
        return pr_number, pr_specific_docs

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the appropriate RAG strategy.
        """
        try:
            pr_number, pr_docs = self._get_pr_specific_docs(question)
            
            if pr_number is not None:
                if not pr_docs:
                    answer = f"I couldn't find any information for PR #{pr_number} in my database."
                    source_documents = []
                else:
                    result = self.specific_qa_chain.invoke(
                        {"input_documents": pr_docs, "question": question},
                        return_only_outputs=True
                    )
                    answer = result.get("output_text", "No answer generated.")
                    source_documents = pr_docs
            else:
                self.logger.info("Processing generic question with similarity search.")
                result = self.generic_rag_chain.invoke({"query": question})
                answer = result.get("result", "No answer generated.")
                source_documents = result.get("source_documents", [])

            sources = []
            seen_sources = set()
            for doc in source_documents:
                source_key = (doc.metadata.get("pr_number"), doc.metadata.get("title"))
                if source_key not in seen_sources:
                    sources.append({
                        "pr_number": doc.metadata.get("pr_number", "Unknown"),
                        "title": doc.metadata.get("title", "Unknown"),
                        "source": doc.metadata.get("source", "Unknown"),
                    })
                    seen_sources.add(source_key)

            # --- FIXED: Added 'total_sources' to the returned dictionary ---
            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "total_sources": len(sources)
            }
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            # --- FIXED: Added 'total_sources' to the error response for consistency ---
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": [],
                "question": question,
                "total_sources": 0
            }

    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the loaded vector store."""
        if not self.vector_store:
            return {"error": "Vector store not loaded."}
        return {
            "total_documents": self.vector_store.index.ntotal,
            "embedding_model": self.embedding_model_name,
        }
