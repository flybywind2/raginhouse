from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field
from enum import Enum
import operator


class RetrieverType(str, Enum):
    RRF = "rrf"
    BM25 = "bm25"
    KNN = "knn"
    CC = "cc"


class AnswerFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


# LangGraph State Schema
class RAGState(TypedDict):
    # Core identifiers
    trace_id: str
    user_id: str
    index_name: str
    
    # Query processing
    query_raw: str
    queries_expanded: List[str]
    
    # Retrieval results
    retrieval_results: Dict[str, List[Dict]]  # {retriever_type: results}
    reranked: List[Dict]
    
    # Context and generation
    context: str
    answer_draft: str
    answer_final: str
    citations: List[Dict]
    
    # Performance tracking
    latency_map: Dict[str, float]  # {node_name: milliseconds}
    
    # Configuration
    retriever_cfg: Dict[str, Any]
# permission_groups removed
    
    # Feedback
    feedback_hint: Optional[str]
    
    # Background tasks tracking
    completed_sections: Annotated[List[str], operator.add]


# FastAPI Request/Response Models
class QueryRequest(BaseModel):
    query_text: str = Field(..., description="User query text")
    index_name: str = Field(..., description="Target index name")
    retriever: RetrieverType = Field(default=RetrieverType.RRF, description="Retrieval method")
    num_result_doc: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    model: Optional[str] = Field(default=None, description="LLM model to use")
    answer_format: AnswerFormat = Field(default=AnswerFormat.MARKDOWN, description="Response format")


class Citation(BaseModel):
    doc_id: str
    title: str
    score: float
    snippet: str
    section_title: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    latency_ms: float
    trace_id: str
    debug: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., description="Trace ID from query response")
    rating: str = Field(..., pattern="^(up|down)$", description="User rating: up or down")
    reason: Optional[str] = Field(None, description="Reason for rating")
    proposed_answer: Optional[str] = Field(None, description="User's proposed better answer")
    selected_citations: Optional[List[str]] = Field(None, description="Citation IDs user found useful")
    tags: Optional[List[str]] = Field(None, description="Additional tags")


class DocumentIngestRequest(BaseModel):
    index_name: str = Field(..., description="Target index name")
# permission_groups removed


class ConfluenceIngestRequest(BaseModel):
    base_url: str = Field(..., description="Confluence base URL")
    page_id: str = Field(..., description="Page ID to ingest")
    index_name: str = Field(..., description="Target index name")


# External API Models (for compatibility with appendix)
class InsertDocumentPayload(BaseModel):
    index_name: str
    data: Dict[str, Any]  # Flat structure only
    chunk_factor: Dict[str, Any]


class RetrievalPayload(BaseModel):
    index_name: str
# permission_groups removed
    query_text: str
    num_result_doc: int = 5
    fields_exclude: Optional[List[str]] = None
    filter: Optional[Dict[str, List[str]]] = None


# Internal Models
class Section(BaseModel):
    name: str
    description: str
    level: int = 1


class DocumentMetadata(BaseModel):
    doc_id: str
    title: str
    summary: Optional[str] = None
    topics: List[str] = []
    language: str = "unknown"
    has_pii: bool = False
    doc_type: str = "unknown"
    created_time: str
# permission_groups removed


class ProcessedDocument(BaseModel):
    metadata: DocumentMetadata
    chunks: List[Dict[str, Any]]


class MetricsResponse(BaseModel):
    positive_rate: float
    counts_by_reason: Dict[str, int]
    ndcg_5: Optional[float] = None
    mrr_5: Optional[float] = None
    trending_queries: List[str] = []