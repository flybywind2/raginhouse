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
# 초보자용: LangGraph 워크플로우가 사용하는 상태(데이터 묶음)의 구조를 정의합니다.
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
    # 초보자용: 각 단계의 처리 시간을 ms 단위로 기록합니다.
    latency_map: Dict[str, float]  # {node_name: milliseconds}
    
    # Configuration
    retriever_cfg: Dict[str, Any]
# permission_groups removed
    
    # Feedback
    feedback_hint: Optional[str]
    
    # Background tasks tracking
    completed_sections: Annotated[List[str], operator.add]


# FastAPI Request/Response Models
# 초보자용: API 입출력 데이터의 모양을 정의해 자동 검증/문서화에 사용됩니다.
class QueryRequest(BaseModel):
    query_text: str = Field(..., description="User query text")
    index_name: str = Field(..., description="Target index name")
    retriever: RetrieverType = Field(default=RetrieverType.RRF, description="Retrieval method")
    num_result_doc: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    model: Optional[str] = Field(default=None, description="LLM model to use")
    answer_format: AnswerFormat = Field(default=AnswerFormat.MARKDOWN, description="Response format")


class Citation(BaseModel):
    # 초보자용: 답변에 사용된 문서의 인용 정보를 담습니다.
    doc_id: str
    title: str
    score: float
    snippet: str
    section_title: Optional[str] = None


class QueryResponse(BaseModel):
    # 초보자용: 클라이언트로 보내는 최종 응답 구조입니다.
    answer: str
    citations: List[Citation]
    latency_ms: float
    trace_id: str
    debug: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    # 초보자용: 사용자의 피드백(업/다운, 사유 등)을 받기 위한 모델입니다.
    trace_id: str = Field(..., description="Trace ID from query response")
    rating: str = Field(..., pattern="^(up|down)$", description="User rating: up or down")
    reason: Optional[str] = Field(None, description="Reason for rating")
    proposed_answer: Optional[str] = Field(None, description="User's proposed better answer")
    selected_citations: Optional[List[str]] = Field(None, description="Citation IDs user found useful")
    tags: Optional[List[str]] = Field(None, description="Additional tags")


class DocumentIngestRequest(BaseModel):
    # 초보자용: 파일 업로드 인덱싱 요청에 사용됩니다.
    index_name: str = Field(..., description="Target index name")
# permission_groups removed


class ConfluenceIngestRequest(BaseModel):
    # 초보자용: Confluence 페이지 인덱싱 요청에 사용됩니다.
    base_url: str = Field(..., description="Confluence base URL")
    page_id: str = Field(..., description="Page ID to ingest")
    index_name: str = Field(..., description="Target index name")


# External API Models (for compatibility with appendix)
class InsertDocumentPayload(BaseModel):
    # 초보자용: 외부 API에 문서를 넣을 때 사용하는 단순화된 페이로드 예시입니다.
    index_name: str
    data: Dict[str, Any]  # Flat structure only
    chunk_factor: Dict[str, Any]


class RetrievalPayload(BaseModel):
    # 초보자용: 외부 검색 API를 호출할 때 사용하는 페이로드 예시입니다.
    index_name: str
# permission_groups removed
    query_text: str
    num_result_doc: int = 5
    fields_exclude: Optional[List[str]] = None
    filter: Optional[Dict[str, List[str]]] = None


# Internal Models
class Section(BaseModel):
    # 초보자용: 문서의 섹션 정보를 표현하는 내부 모델입니다.
    name: str
    description: str
    level: int = 1


class DocumentMetadata(BaseModel):
    # 초보자용: 문서의 메타데이터를 표현합니다.
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
    # 초보자용: 파싱/청크 분할된 문서를 담습니다.
    metadata: DocumentMetadata
    chunks: List[Dict[str, Any]]


class MetricsResponse(BaseModel):
    # 초보자용: 피드백/성능과 관련된 지표를 담습니다.
    positive_rate: float
    counts_by_reason: Dict[str, int]
    ndcg_5: Optional[float] = None
    mrr_5: Optional[float] = None
    trending_queries: List[str] = []
