# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) Agent system that integrates with enterprise document search, summarization, and Q&A capabilities. The system is built using FastAPI, LangGraph for workflow orchestration, and integrates with existing retrieval APIs.

## Architecture

The system follows a modular architecture with these key components:

### Core Components
- **Query Preprocessor**: Language detection, normalization, optional query expansion
- **Retriever Client**: Calls `POST /retrieve-rrf|bm25|knn|cc` APIs with permission filtering
- **Evidence Ranker**: RRF/score-based reordering, deduplication
- **Context Assembler**: Token-limited context construction with highlights/metadata
- **Prompt Builder**: Standardizes role/instructions/context/questions/sources
- **LLM Client**: Based on `appendix/internal_llm.py` patterns with dynamic config
- **Postprocessor**: Source annotations, format conversion (JSON/Markdown), safety filters

### LangGraph Workflow Design
The system uses LangGraph for orchestrating complex multi-step workflows:

**State Schema**: `trace_id`, `user_id`, `index_name`, `retriever_cfg`, `query_raw`, `queries_expanded[]`, `retrieval_results[]`, `reranked[]`, `context`, `answer_draft`, `answer_final`, `citations[]`, `latency_map{node:ms}`

**Node Structure**:
- `QueryRewrite`: Multi-query expansion (MQE), stopword removal, language normalization
- `RetrieveBM25|RetrieveKNN|RetrieveCC`: Parallel execution with timeout fallback
- `FuseAndRerank`: RRF fusion → MMR deduplication → optional cross-encoder reranking
- `AssembleContext`: Token budget utility maximization with highlights
- `Generate`: LLM generation with forced citation
- `Critique`: Self-evaluation (evidence consistency/factuality/policy violations)
- `Refine`: Insufficient evidence re-retrieval or answer rewriting
- `End`: Response confirmation and logging

## API Integration Patterns

### External APIs (from appendix examples)

**Document Insertion** (`appendix/rag_input.py` pattern):
```python
# POST /insert-doc
headers = {
    "Content-Type": "application/json",
    "x-dep-ticket": credential_key,
    "api-key": rag_api_key
}
# Payload must be flat structure, no nested JSON
```

**Document Retrieval** (`appendix/rag_retrieve.py` pattern):
```python
# POST /retrieve-rrf|bm25|knn|cc
# Response format: Elasticsearch style with hits.hits[] array
```

**LLM Integration** (`appendix/internal_llm.py` pattern):
```python
# ChatOpenAI wrapper with required headers:
default_headers={
    "x-dep-ticket": credential_key,
    "Send-System-Name": "System_Name",
    "User-ID": "ID",
    "User-Type": "AD",
    "Prompt-Msg-Id": str(uuid.uuid4()),
    "Completion-Msg-Id": str(uuid.uuid4()),
}
```

## Configuration Management

### Environment Variables
```
RAG_BASE_URL=http://localhost:8000
RAG_API_KEY=your_rag_api_key
DEP_TICKET=your_credential_key
INDEX_NAME=your_index_name
RETRIEVER=rrf
OPENAI_API_KEY=your_openai_api_key
LLM_BASE_URL=https://model1.openai.com/v1
MODEL_NAME=llama4 maverick
DOCLING_ENABLED=true
CONFLUENCE_BASE_URL=your_confluence_url
CONFLUENCE_TOKEN=your_confluence_token
```

## Document Processing Pipeline

### Parsing with Docling
- Uses `docling` for PDF/DOCX/PPTX/XLSX parsing
- Preserves structure: sections/headers/tables/captions/page metadata
- LLM-based metadata enrichment via `additional_field` (flat string format)
- Structure-aware chunking: uses docling headers/sections + token limits

### Confluence Integration
- Page-based single collection (pageId-based)
- SSL verification disabled for MVP (verify=False)
- REST API integration for page/blog/attachment collection

## Web UI (FastAPI)

### Endpoints
- `GET /`: Query input form
- `POST /ask`: Query processing with streaming option
- `POST /feedback`: User feedback submission
- `POST /ingest/file`: File upload and processing
- `POST /ingest/confluence`: Confluence page ingestion

### Response Format
```json
{
    "answer": "string|object",
    "citations": [{"doc_id", "title", "score", "snippet"}],
    "latency_ms": "number",
    "trace_id": "string"
}
```

## Security and Permissions

### Permission Model
- `permission_groups` based access control
- Header-based authentication: `x-dep-ticket`, `api-key`
- No SSO/JWT in MVP (internal development only)

### Data Constraints
- Insert payloads must be flat structure (no nested JSON)
- `additional_field` serialized as single string
- PII masking deferred to post-MVP

## Performance Optimization

### Caching Strategy
- Semantic cache: `hash(normalize(query_text), permission_groups, index, retriever_cfg)` → TTL 30-120s
- Feedback hint cache for similar queries
- Cache invalidation on document updates/reindexing

### Retrieval Enhancement
- Hybrid search: BM25 + vector simultaneous calls → RRF
- Multi-query expansion (MQE): LLM-based paraphrasing/term expansion
- PRF (Pseudo-Relevance Feedback): top-k document keywords for re-querying
- MMR deduplication: similarity-diversity balance
- Cross-encoder reranking: top 50→10 precision reordering
- Context compression: LLM summarization/key sentence extraction

## Development Commands

Since this is an early stage project, establish these commands:

```bash
# Install dependencies (create requirements.txt)
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Run tests
pytest tests/ -v

# Lint code
flake8 src/
black src/

# Type checking
mypy src/
```

## Key Implementation Files Structure

```
/
├── src/
│   ├── agents/          # LangGraph workflow definitions
│   ├── api/             # FastAPI route handlers
│   ├── clients/         # External API clients (retriever, LLM)
│   ├── core/            # Core business logic
│   ├── models/          # Pydantic models and schemas
│   ├── services/        # Service layer implementations
│   └── utils/           # Utility functions
├── tests/              # Test files
├── appendix/           # Reference implementation examples
├── prd.md             # Product Requirements Document
├── trd.md             # Technical Requirements Document
└── requirements.txt    # Python dependencies
```

## Testing Strategy

- Unit tests: Prompt Builder, Context Assembler, score fusion logic
- Integration tests: Retriever mocking/local endpoints, permission filter cases
- Regression tests: Sample query snapshots, prohibited term/safety filters
- Load tests: 50-100 concurrent RPS with P95 latency verification

## Monitoring and Observability

### Structured Logging
- Include: `trace_id`, `user_id`, `index`, `retriever`, result count, latency
- Metrics: search/generation latency, token usage, error rates
- Tracing: Retriever/LLM call spans with `Prompt-Msg-Id`/`Completion-Msg-Id`
- LangGraph node-level latency/success rates, retry/branch ratios, cache hit rates

## Dependencies

Key Python packages to include:
- `fastapi` - Web framework
- `langchain-openai` - LLM integration
- `langgraph` - Workflow orchestration  
- `docling` - Document parsing
- `requests` - HTTP client
- `pydantic` - Data validation
- `uvicorn` - ASGI server
- `jinja2` - Template engine (for Web UI)
- `pytest` - Testing framework
- `httpx` - Async HTTP client

## Error Handling

- Retriever HTTP errors: 2xx status → retry (max 2), failure → empty evidence safety response
- LLM timeout/error: short summary mode fallback
- Empty search results: user re-query guidance
- Permission violations: filtered results with access denied message