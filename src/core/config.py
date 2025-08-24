import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # External API Configuration (compatible with appendix)
    # 초보자용: 환경변수(.env)로 설정값을 주입합니다. 기본값이 있는 항목은 미설정 시 기본값 사용.
    RAG_BASE_URL: str = Field(default="http://localhost:8000", description="RAG API base URL")
    RAG_API_KEY: str = Field(..., description="RAG API key")
    DEP_TICKET: str = Field(..., description="Department ticket credential")
    
    # Default configurations
    INDEX_NAME: str = Field(default="default_index", description="Default index name")
    RETRIEVER: str = Field(default="rrf", description="Default retriever type")
    
    # LLM Configuration (compatible with internal_llm.py)
    # 초보자용: 모델 이름과 엔드포인트 매핑을 통해 동적으로 호출 대상을 바꿀 수 있습니다.
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    LLM_BASE_URL: str = Field(default="https://model1.openai.com/v1", description="LLM base URL")
    MODEL_NAME: str = Field(default="llama4 maverick", description="Default model name")
    
    # Available models mapping
    MODEL_ENDPOINTS: Dict[str, str] = Field(default={
        "llama4 maverick": "https://model1.openai.com/v1",
        "llama4 scout": "https://model2.openai.com/v1",
        "gemma3": "https://model3.openai.com/v1",
        "deepseek-r1": "https://model4.openai.com/v1",
        "gpt-oss": "https://model5.openai.com/v1"
    })
    
    # Document Processing
    MAX_CHUNK_SIZE: int = Field(default=1024, description="Maximum chunk size")
    CHUNK_OVERLAP: int = Field(default=128, description="Chunk overlap size")
    
    # Confluence Integration
    CONFLUENCE_BASE_URL: str = Field(default="", description="Confluence base URL")
    CONFLUENCE_TOKEN: str = Field(default="", description="Confluence API token")
    
    # Performance Configuration
    REQUEST_TIMEOUT: float = Field(default=15.0, description="HTTP request timeout in seconds")
    MAX_RETRIES: int = Field(default=2, description="Maximum number of retries")
    
    # Caching
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    CACHE_TTL: int = Field(default=300, description="Cache TTL in seconds")
    
    # RAG Optimization Parameters
    RRF_K: int = Field(default=60, description="RRF k parameter")
    MMR_LAMBDA: float = Field(default=0.7, description="MMR lambda parameter (relevance vs diversity)")
    TOKEN_BUDGET: int = Field(default=4000, description="Token budget for context")
    
    # Multi-query expansion
    NUM_QUERY_VARIANTS: int = Field(default=3, description="Number of query variants for expansion")
    QUERY_SIMILARITY_THRESHOLD: float = Field(default=0.85, description="Similarity threshold for query expansion")
    
    # Reranking
    ENABLE_RERANKING: bool = Field(default=False, description="Enable cross-encoder reranking")
    RERANKING_ENDPOINT: str = Field(default="", description="Reranking service endpoint")
    RERANKING_TIMEOUT: float = Field(default=2.0, description="Reranking timeout in seconds")
    MAX_RERANK_CANDIDATES: int = Field(default=50, description="Maximum candidates for reranking")
    RERANK_TOP_K: int = Field(default=10, description="Top K after reranking")
    
    # FastAPI Configuration
    APP_HOST: str = Field(default="0.0.0.0", description="FastAPI host")
    APP_PORT: int = Field(default=8080, description="FastAPI port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format: json or text")
    
# Security features removed
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Header configuration for external API calls (from appendix examples)
def get_rag_headers() -> Dict[str, str]:
    """Get headers for RAG API calls (compatible with appendix examples)

    초보자용 설명:
    - 외부 RAG API 호출 시 공통으로 포함할 헤더를 만듭니다.
    """
    return {
        "Content-Type": "application/json",
        "x-dep-ticket": settings.DEP_TICKET,
        "api-key": settings.RAG_API_KEY
    }


def get_llm_headers() -> Dict[str, str]:
    """Get headers for LLM API calls (compatible with internal_llm.py)

    초보자용 설명:
    - 내부 규약에 맞춘 공통 헤더(요청 식별자 등)를 생성합니다.
    """
    import uuid
    return {
        "x-dep-ticket": settings.DEP_TICKET,
        "Send-System-Name": "RAG_Agent",
        "User-ID": "system",
        "User-Type": "SERVICE",
        "Prompt-Msg-Id": str(uuid.uuid4()),
        "Completion-Msg-Id": str(uuid.uuid4()),
    }


def get_model_endpoint(model_name: str) -> str:
    """Get LLM endpoint for given model

    초보자용 설명:
    - 모델 이름으로 적절한 엔드포인트 URL을 돌려줍니다. 없으면 기본값 사용.
    """
    return settings.MODEL_ENDPOINTS.get(model_name, settings.LLM_BASE_URL)


# Retriever weights for RRF fusion
RETRIEVER_WEIGHTS = {
    "bm25": 0.4,
    "knn": 0.4,
    "cc": 0.2,
    "rrf": 1.0
}
