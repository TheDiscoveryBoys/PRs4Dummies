# PRs4Dummies RAG API

This is the Application Layer (Phase 3) of the PRs4Dummies project - a FastAPI-based web server that exposes the RAG (Retrieval-Augmented Generation) pipeline for answering questions about Ansible pull requests.

## 🚀 Features

- **RESTful API**: Clean HTTP endpoints for interacting with the RAG system
- **Question Answering**: Ask questions about pull requests and get AI-generated answers
- **Source Attribution**: See which documents were used to generate each answer
- **Health Monitoring**: Built-in health checks and system information endpoints
- **Interactive Documentation**: Auto-generated API docs with Swagger UI
- **Local LLM Support**: Uses local HuggingFace models for privacy and offline operation

## 📁 File Structure

```
prs4dummies/
├── indexing/
│   └── vector_store/    # Pre-built FAISS index
└── rag-core/
    ├── rag_core.py      # Core RAG logic and pipeline
    ├── main.py          # FastAPI server
    ├── test_rag.py      # Test script for RAG core
    ├── requirements-rag.txt  # Dependencies for RAG API
    └── README_RAG_API.md    # This file
```

## 🛠️ Installation

### 1. Install Dependencies

```bash
cd prs4dummies/rag-core
pip install -r requirements-rag.txt
```

### 2. Verify Setup

Run the test script to ensure everything is working:

```bash
python test_rag.py
```

This will:
- Check all required packages are installed
- Verify the vector store exists
- Test the RAG core initialization
- Run a sample question through the system

## 🚀 Quick Start

### 1. Start the Server

```bash
python main.py
```

The server will start on `http://localhost:8000` by default.

### 2. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **API Info**: http://localhost:8000/info

### 3. Ask Your First Question

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are common types of pull requests in Ansible?"}'
```

## 📚 API Endpoints

### POST /ask
Ask a question about pull requests.

**Request Body:**
```json
{
  "question": "What is Ansible?",
  "include_sources": true
}
```

**Response:**
```json
{
  "answer": "Ansible is an open-source automation tool...",
  "question": "What is Ansible?",
  "sources": [
    {
      "source": "pr_123.json",
      "chunk_id": "chunk_1",
      "relevance_score": 0.95
    }
  ],
  "total_sources": 1,
  "processing_time_ms": 1250.5,
  "timestamp": "2025-01-23T10:30:00Z"
}
```

### GET /health
Check the health status of the service.

**Response:**
```json
{
  "status": "healthy",
  "vector_store_info": {
    "total_documents": 280,
    "embedding_dimension": 768,
    "embedding_model": "all-mpnet-base-v2",
    "vector_store_path": "vector_store"
  },
  "uptime_seconds": 3600.5,
  "timestamp": "2025-01-23T10:30:00Z"
}
```

### GET /info
Get information about the loaded system.

### GET /
Basic information about the API.

## ⚙️ Configuration

### Environment Variables

You can configure the API using environment variables:

```bash
# Vector store location
export VECTOR_STORE_PATH="vector_store"

# Embedding model
export EMBEDDING_MODEL="all-mpnet-base-v2"

# Server configuration
export HOST="0.0.0.0"
export PORT="8000"
export RELOAD="false"
```

### Default Values

- **Vector Store Path**: `vector_store` (relative to current directory)
- **Embedding Model**: `all-mpnet-base-v2`
- **Host**: `0.0.0.0` (all interfaces)
- **Port**: `8000`
- **Reload**: `false` (disable auto-reload)

## 🔧 Customization

### Using Different LLM Models

The RAG core uses a local HuggingFace model by default. You can modify `rag_core.py` to use different models:

```python
# In rag_core.py, change the model name
self.llm = LocalHuggingFaceLLM(
    model_name="gpt2",  # Change to any HuggingFace model
    max_length=512
)
```

### Adjusting Retrieval Parameters

Modify the retrieval settings in `rag_core.py`:

```python
# Change number of retrieved documents
retriever = self.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}  # Retrieve top 10 instead of 5
)
```

### Customizing the Prompt Template

Edit the prompt template in `rag_core.py` to better suit your use case:

```python
template = """You are an expert software engineer...

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
```

## 🧪 Testing

### Test the RAG Core

```bash
python test_rag.py
```

### Test the API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Ansible?"}'
```

### Load Testing

For basic load testing, you can use tools like `ab` (Apache Bench) or `wrk`:

```bash
# Install wrk (on macOS: brew install wrk)
wrk -t12 -c400 -d30s --latency http://localhost:8000/health
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements-rag.txt`
   - Check Python version compatibility

2. **Vector Store Not Found**
   - Verify the `vector_store` directory exists
   - Check the `VECTOR_STORE_PATH` environment variable

3. **Model Loading Issues**
   - Ensure sufficient disk space for model downloads
   - Check internet connection for first-time model downloads
   - Verify PyTorch installation

4. **Memory Issues**
   - Reduce `max_length` in the LLM configuration
   - Use smaller embedding models
   - Reduce the number of retrieved documents (`k` parameter)

### Debug Mode

Enable debug logging by modifying the logging level in `main.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor the API performance using the built-in metrics:

- Processing time for each question
- Number of sources used
- Vector store statistics

## 🚀 Production Deployment

### Using Gunicorn

For production deployment, use Gunicorn with Uvicorn workers:

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements-rag.txt .
RUN pip install -r requirements-rag.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Environment Configuration

For production, set appropriate environment variables:

```bash
export VECTOR_STORE_PATH="/data/vector_store"
export EMBEDDING_MODEL="all-mpnet-base-v2"
export HOST="0.0.0.0"
export PORT="8000"
export RELOAD="false"
```

## 📊 Monitoring and Logging

The API includes comprehensive logging:

- Request/response logging
- Error tracking
- Performance metrics
- System health information

Logs are written to stdout/stderr and can be redirected to files or log aggregation services.

## 🔒 Security Considerations

- **CORS**: Currently allows all origins (`*`). Restrict this for production
- **Rate Limiting**: Consider adding rate limiting for production use
- **Authentication**: Add authentication if exposing to the internet
- **Input Validation**: All inputs are validated using Pydantic models

## 🤝 Contributing

To extend the RAG API:

1. Modify `rag_core.py` for core logic changes
2. Update `main.py` for new endpoints
3. Add tests in `test_rag.py`
4. Update this README with new features

## 📝 License

This project is part of PRs4Dummies and follows the same license terms.

---

**Next Steps**: After setting up the RAG API, you can:
- Build a web frontend to interact with the API
- Integrate with other systems via HTTP
- Scale the service for production use
- Add more sophisticated question processing
