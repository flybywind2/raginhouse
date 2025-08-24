import httpx
import asyncio
import logging
from typing import List, Dict, Optional, Any
from src.core.config import settings, get_rag_headers
from src.models.schemas import RetrieverType, RetrievalPayload

logger = logging.getLogger(__name__)


class RetrieverClient:
    """Client for interacting with RAG retrieval APIs (compatible with appendix examples)

    초보자용 설명:
    - 외부 검색 API(BM25/kNN/CC 등)를 호출해 문서 목록을 가져옵니다.
    - 시간 제한/재시도 설정을 통해 네트워크 오류에 대비합니다.
    """
    
    def __init__(self):
        self.base_url = settings.RAG_BASE_URL
        self.timeout = settings.REQUEST_TIMEOUT
        self.max_retries = settings.MAX_RETRIES
    
    async def retrieve_documents(
        self,
        retriever_type: RetrieverType,
        query: str,
        index_name: str,
        num_docs: int = 5,
        fields_exclude: Optional[List[str]] = None,
        filters: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve documents using specified retriever

        초보자용 설명:
        - retriever_type에 따라 다른 엔드포인트로 요청을 보냅니다.
        - 응답 형식은 Elasticsearch 스타일의 `hits.hits` 배열을 포함합니다.
        """
        endpoint = f"{self.base_url}/retrieve-{retriever_type.value}"
        
        payload = RetrievalPayload(
            index_name=index_name,
            query_text=query,
            num_result_doc=num_docs,
            fields_exclude=fields_exclude or ["v_merge_title_content"],
            filter=filters
        )
        
        headers = get_rag_headers()
        
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        json=payload.model_dump()
                    )
                    
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Retrieved {len(result.get('hits', {}).get('hits', []))} documents from {retriever_type.value}")
                    return result
                else:
                    logger.error(f"Retriever {retriever_type.value} returned status {response.status_code}")
                    if attempt == self.max_retries:
                        raise Exception(f"Retriever {retriever_type.value} failed after {self.max_retries} attempts")
                        
            except Exception as e:
                logger.error(f"Retriever {retriever_type.value} attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    raise e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff (지수 백오프)
    
    async def parallel_retrieve(
        self,
        queries: List[str],
        index_name: str,
        retriever_types: List[RetrieverType],
        num_docs: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Parallel retrieval across multiple queries and retrievers

        초보자용 설명:
        - 여러 질의/여러 검색기를 동시에 실행해 시간을 절약합니다.
        - 네트워크/서버 에러는 각 태스크 단위로 처리하고, 가능한 결과만 모읍니다.
        """
        tasks = []
        task_metadata = []
        
        for retriever_type in retriever_types:
            for query in queries:
                task = self.retrieve_documents(
                    retriever_type=retriever_type,
                    query=query,
                    index_name=index_name,
                    permission_groups=permission_groups,
                    num_docs=num_docs
                )
                tasks.append(task)
                task_metadata.append((retriever_type.value, query))
        
        # Execute parallel retrieval with timeout
        # 초보자용: asyncio.gather로 비동기 작업을 한꺼번에 실행합니다.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        # 초보자용: 성공한 작업의 hits.hits만 모아서 유형별로 합칩니다.
        retrieval_results = {}
        for (retriever_type, query), result in zip(task_metadata, results):
            if isinstance(result, Exception):
                logger.warning(f"Retrieval failed for {retriever_type} with query '{query}': {result}")
                continue
            
            if retriever_type not in retrieval_results:
                retrieval_results[retriever_type] = []
            
            hits = result.get('hits', {}).get('hits', [])
            retrieval_results[retriever_type].extend(hits)
        
        return retrieval_results


class CircuitBreaker:
    """Circuit breaker pattern for retriever resilience"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        import time
        
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def on_failure(self):
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


# Global circuit breaker instances
retriever_circuit_breakers = {
    retriever_type.value: CircuitBreaker() 
    for retriever_type in RetrieverType
}
