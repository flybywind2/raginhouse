# RAG Agent

Enterprise RAG (Retrieval-Augmented Generation) Agent with LangGraph workflow orchestration.

## Features

🤖 **LangGraph Orchestration**: Advanced multi-step RAG workflows with state management
🔍 **Multi-Retrieval Fusion**: BM25, kNN, CC retrievers with RRF fusion
⚡ **Performance Optimized**: Semantic caching, MMR selection, cross-encoder reranking
📚 **Document Processing**: Docling-based parsing with structure awareness
🌐 **FastAPI Web UI**: Interactive interface with streaming responses
🔗 **Enterprise Integration**: Compatible with existing retrieval and LLM APIs

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

### 3. Run Development Server

```bash
python run_dev.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

### 4. Access the Application

- **Web UI**: http://localhost:8080/
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/api/v1/health

## Architecture

### Core Components

- **RAGWorkflow**: LangGraph-based workflow orchestrator
- **RetrieverClient**: Multi-retrieval API integration
- **LLMClient**: Language model client with header management
- **CacheService**: Redis-based semantic caching
- **DocumentProcessor**: Docling-based document parsing

### Workflow Stages

1. **Query Rewrite**: Multi-query expansion with LLM
2. **Parallel Retrieval**: BM25, kNN, CC retrievers
3. **Fusion & Reranking**: RRF fusion + cross-encoder reranking
4. **Context Assembly**: MMR-based context selection
5. **Answer Generation**: LLM-based answer synthesis
6. **Quality Control**: Self-critique and refinement

## Configuration

Key environment variables:

```bash
# External APIs
RAG_BASE_URL=http://localhost:8000
RAG_API_KEY=your_rag_api_key
DEP_TICKET=your_credential_key

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key
LLM_BASE_URL=https://model1.openai.com/v1
MODEL_NAME=llama4 maverick

# Performance Tuning
RRF_K=60
MMR_LAMBDA=0.7
TOKEN_BUDGET=4000
ENABLE_RERANKING=false
```

## API Usage

### Query Endpoint

```bash
curl -X POST "http://localhost:8080/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What is the main topic of the document?",
    "index_name": "default_index",
    "permission_groups": ["user"],
    "retriever": "rrf",
    "num_result_doc": 5,
    "answer_format": "markdown"
  }'
```

### Document Upload

```bash
curl -X POST "http://localhost:8080/api/v1/ingest/file" \
  -F "file=@document.pdf" \
  -F "index_name=default_index" \
  -F "permission_groups=user"
```

## Development

### Project Structure

```
ragagent/
├── src/
│   ├── agents/          # LangGraph workflow definitions
│   ├── api/             # FastAPI route handlers
│   ├── clients/         # External API clients
│   ├── core/            # Configuration and utilities
│   ├── models/          # Pydantic schemas
│   └── services/        # Business logic services
├── templates/           # Jinja2 templates
├── static/             # Static web assets
├── appendix/           # Reference implementations
├── main.py             # FastAPI application
└── run_dev.py         # Development server runner
```

### Testing

```bash
# Run tests
pytest tests/ -v

# Code formatting
black src/

# Type checking
mypy src/

# Linting
flake8 src/
```

### Performance Targets

- **Response Time**: P50 ≤ 2.5s, P95 ≤ 5s (6.5s with reranking)
- **Quality Metrics**: NDCG@5 ≥ 0.55, MRR@5 ≥ 0.45
- **Availability**: 99.5% monthly uptime

## Compatibility

This implementation is compatible with:

- **appendix/internal_llm.py**: LLM client patterns
- **appendix/rag_input.py**: Document insertion API
- **appendix/rag_retrieve.py**: Retrieval API responses

## License

This project is for internal enterprise use.