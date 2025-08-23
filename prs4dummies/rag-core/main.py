"""
FastAPI Server for PRs4Dummies RAG API

This server exposes the RAG pipeline through a REST API, allowing users to ask questions
about pull requests and receive AI-generated answers based on the indexed data.
"""

import os
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import our RAG core
from rag_core import RAGCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., description="The question to ask about pull requests", min_length=1, max_length=1000)
    include_sources: bool = Field(default=True, description="Whether to include source information in the response")

class QuestionResponse(BaseModel):
    """Response model for question answers."""
    answer: str = Field(..., description="The AI-generated answer to the question")
    question: str = Field(..., description="The original question that was asked")
    sources: list = Field(default=[], description="List of source documents used to generate the answer")
    total_sources: int = Field(default=0, description="Total number of source documents used")
    processing_time_ms: float = Field(..., description="Time taken to process the question in milliseconds")
    timestamp: str = Field(..., description="ISO timestamp of when the response was generated")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status of the service")
    vector_store_info: Dict[str, Any] = Field(..., description="Information about the loaded vector store")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="ISO timestamp of the health check")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="ISO timestamp of the error")

# Global variables
app = FastAPI(
    title="PRs4Dummies RAG API",
    description="AI-powered API for answering questions about Ansible pull requests",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global RAG instance
rag_core: Optional[RAGCore] = None
startup_time = time.time()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG core when the server starts."""
    global rag_core
    
    try:
        logger.info("Starting PRs4Dummies RAG API...")
        
        # Get vector store path from environment or use default
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "../indexing/vector_store")
        embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        
        logger.info(f"Loading vector store from: {vector_store_path}")
        logger.info(f"Using embedding model: {embedding_model}")
        
        # Initialize RAG core
        rag_core = RAGCore(
            vector_store_path=vector_store_path,
            embedding_model_name=embedding_model
        )
        
        logger.info("RAG core initialized successfully!")
        logger.info("API is ready to receive requests")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG core: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when the server shuts down."""
    logger.info("Shutting down PRs4Dummies RAG API...")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Welcome to PRs4Dummies RAG API",
        "description": "AI-powered API for answering questions about Ansible pull requests",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        if not rag_core:
            raise HTTPException(status_code=503, detail="RAG core not initialized")
        
        # Get vector store information
        vector_store_info = rag_core.get_vector_store_info()
        
        # Calculate uptime
        uptime = time.time() - startup_time
        
        return HealthResponse(
            status="healthy",
            vector_store_info=vector_store_info,
            uptime_seconds=uptime,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, background_tasks: BackgroundTasks):
    """
    Ask a question about pull requests and get an AI-generated answer.
    
    This endpoint uses the RAG pipeline to:
    1. Retrieve relevant context from the vector store
    2. Generate an answer using the LLM
    3. Return the answer with source information
    """
    try:
        if not rag_core:
            raise HTTPException(status_code=503, detail="RAG core not initialized")
        
        start_time = time.time()
        
        logger.info(f"Processing question: {request.question}")
        
        # Get answer from RAG core
        result = rag_core.answer_question(request.question)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Prepare response
        response_data = {
            "answer": result["answer"],
            "question": request.question,
            "total_sources": result["total_sources"],
            "processing_time_ms": round(processing_time, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        # Include sources if requested and available
        if request.include_sources:
            response_data["sources"] = result.get("sources", [])
        else:
            response_data["sources"] = []
        
        # Log the response
        logger.info(f"Generated answer in {processing_time:.2f}ms with {result['total_sources']} sources")
        
        return QuestionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )

@app.get("/info", response_model=Dict[str, Any])
async def get_info():
    """Get information about the loaded vector store and system."""
    try:
        if not rag_core:
            raise HTTPException(status_code=503, detail="RAG core not initialized")
        
        vector_store_info = rag_core.get_vector_store_info()
        
        return {
            "vector_store": vector_store_info,
            "embedding_model": rag_core.embedding_model_name,
            "llm_type": "local_huggingface",
            "api_version": "1.0.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
    except Exception as e:
        logger.error(f"Error getting info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get info: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
