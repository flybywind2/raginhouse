import httpx
import logging
from typing import List, Dict, Any, Optional
from src.core.config import settings, get_rag_headers
from src.models.schemas import InsertDocumentPayload

logger = logging.getLogger(__name__)


class DocumentClient:
    """Client for document insertion API (compatible with appendix/rag_input.py)

    초보자용 설명:
    - 외부 문서 입력 API에 청크 단위 문서를 전송해 색인합니다.
    - 실패 시 재시도하고, 여러 문서는 반복 호출로 처리합니다.
    """
    
    def __init__(self):
        self.base_url = settings.RAG_BASE_URL
        self.timeout = settings.REQUEST_TIMEOUT
        self.max_retries = settings.MAX_RETRIES
    
    async def insert_document(
        self,
        doc_data: Dict[str, Any],
        index_name: str,
        chunk_factor: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Insert single document (compatible with appendix/rag_input.py format)

        초보자용 설명:
        - 단일 문서를 외부 API로 전송합니다. chunk_factor로 분할 정책을 전달합니다.
        """
        endpoint = f"{self.base_url}/insert-doc"
        
        # Default chunk factor
        if chunk_factor is None:
            chunk_factor = {
                "logic": "fixed_size",
                "chunk_size": settings.MAX_CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "separator": " "
            }
        
        payload = InsertDocumentPayload(
            index_name=index_name,
            data=doc_data,
            chunk_factor=chunk_factor
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
                    logger.info(f"Successfully inserted document: {doc_data.get('doc_id', 'unknown')}")
                    return True
                else:
                    logger.warning(f"Document insertion failed: {response.status_code} - {response.text}")
                    if attempt == self.max_retries:
                        return False
                        
            except Exception as e:
                logger.error(f"Document insertion attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    return False
        
        return False
    
    async def insert_documents(
        self,
        documents: List[Dict[str, Any]],
        index_name: str,
        chunk_factor: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Insert multiple documents in batches

        초보자용 설명:
        - 문서 리스트를 순회하며 하나씩 전송합니다.
        - 성공률이 80% 이상이면 전체 성공으로 간주합니다.
        """
        if not documents:
            logger.warning("No documents to insert")
            return True
        
        success_count = 0
        
        for doc_data in documents:
            success = await self.insert_document(doc_data, index_name, chunk_factor)
            if success:
                success_count += 1
        
        success_rate = success_count / len(documents)
        logger.info(f"Document insertion: {success_count}/{len(documents)} successful ({success_rate:.2%})")
        
        # Consider it successful if at least 80% of documents were inserted
        return success_rate >= 0.8
