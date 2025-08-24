import logging
import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from src.models.schemas import (
    QueryRequest, QueryResponse, FeedbackRequest, 
    DocumentIngestRequest, ConfluenceIngestRequest, MetricsResponse
)
from src.services.rag_service import RAGService
from src.agents.rag_workflow import RAGWorkflow
from src.agents.optimized_rag_workflow import optimized_rag_workflow
from src.services.cache_service import CacheService
from src.services.document_processor import DocumentProcessor
from src.services.feedback_service import FeedbackService

logger = logging.getLogger(__name__)

# Services will be initialized on first use
rag_service = None
rag_workflow = None
cache_service = None
document_processor = None
feedback_service = None

def get_rag_service():
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service

def get_rag_workflow():
    global rag_workflow
    if rag_workflow is None:
        rag_workflow = RAGWorkflow()
    return rag_workflow

def get_cache_service():
    global cache_service
    if cache_service is None:
        cache_service = CacheService()
    return cache_service

def get_document_processor():
    global document_processor
    if document_processor is None:
        document_processor = DocumentProcessor()
    return document_processor

def get_feedback_service():
    global feedback_service
    if feedback_service is None:
        feedback_service = FeedbackService()
    return feedback_service

# Create router
router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest) -> QueryResponse:
    """
    Process a RAG query using LangGraph workflow
    """
    try:
        logger.info(f"Processing query: {request.query_text[:100]}")
        
        # Execute RAG workflow
        workflow = get_rag_workflow()
        result = await workflow.execute_workflow(
            query=request.query_text,
            index_name=request.index_name,
            retriever_type=request.retriever,
            num_docs=request.num_result_doc,
            model=request.model,
            answer_format=request.answer_format.value
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/optimized", response_model=QueryResponse)
async def ask_question_optimized(request: QueryRequest) -> QueryResponse:
    """
    Process a RAG query using optimized LangGraph workflow with parallel execution
    - Parallel retrieval tasks with @task decorators
    - Intelligent caching with TTL policies
    - Retry policies for resilience
    - Performance optimizations from Context7 best practices
    """
    try:
        logger.info(f"Processing optimized query: {request.query_text[:100]}")
        
        # Execute optimized RAG workflow with parallel processing
        result = await optimized_rag_workflow.execute_optimized_workflow(
            query=request.query_text,
            index_name=request.index_name,
            retriever_type=request.retriever,
            num_docs=request.num_result_doc,
            model=request.model,
            answer_format=request.answer_format.value
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Optimized query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ask/stream")
async def ask_question_stream(
    query_text: str,
    index_name: str,
    retriever: str = "rrf",
    num_result_doc: int = 5,
    model: Optional[str] = None,
    answer_format: str = "markdown"
):
    """
    Stream RAG query response using Server-Sent Events
    """
    async def generate_stream():
        try:
            yield "data: {\"status\": \"starting\", \"message\": \"Processing your query...\"}\n\n"
            
            # Note: This is a simplified streaming implementation
            # In practice, you'd want to stream intermediate results from LangGraph
            workflow = get_rag_workflow()
            result = await workflow.execute_workflow(
                query=query_text,
                index_name=index_name,
                num_docs=num_result_doc,
                model=model,
                answer_format=answer_format
            )
            
            yield f"data: {json.dumps(result)}\n\n"
            yield "data: {\"status\": \"completed\"}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"data: {{\"status\": \"error\", \"error\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a query response
    """
    try:
        feedback_svc = get_feedback_service()
        await feedback_svc.store_feedback(request)
        logger.info(f"Feedback stored for trace_id: {request.trace_id}")
        return {"status": "ok", "message": "Feedback submitted successfully"}
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/metrics", response_model=MetricsResponse)
async def get_feedback_metrics() -> MetricsResponse:
    """
    Get feedback and performance metrics
    """
    try:
        feedback_svc = get_feedback_service()
        metrics = await feedback_svc.get_metrics()
        return MetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    index_name: str = Form(...)
):
    """
    Ingest a document file for processing
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
            
        allowed_extensions = {'.pdf', '.docx', '.pptx', '.xlsx', '.txt'}
        file_ext = '.' + file.filename.split('.')[-1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Add background task for document processing
        doc_processor = get_document_processor()
        background_tasks.add_task(
            doc_processor.process_uploaded_file,
            file,
            index_name
        )
        
        logger.info(f"File {file.filename} queued for processing")
        return {
            "status": "accepted",
            "message": f"File {file.filename} is being processed in the background",
            "file_name": file.filename,
            "index_name": index_name
        }
        
    except Exception as e:
        logger.error(f"File ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/confluence")
async def ingest_confluence_page(
    background_tasks: BackgroundTasks,
    request: ConfluenceIngestRequest
):
    """
    Ingest a Confluence page
    """
    try:
        # Add background task for Confluence processing
        doc_processor = get_document_processor()
        background_tasks.add_task(
            doc_processor.process_confluence_page,
            request.base_url,
            request.page_id,
            request.index_name
        )
        
        logger.info(f"Confluence page {request.page_id} queued for processing")
        return {
            "status": "accepted",
            "message": f"Confluence page {request.page_id} is being processed in the background",
            "page_id": request.page_id,
            "index_name": request.index_name
        }
        
    except Exception as e:
        logger.error(f"Confluence ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache performance statistics
    """
    try:
        cache_svc = get_cache_service()
        stats = await cache_svc.get_cache_stats()
        return {"cache_stats": stats}
        
    except Exception as e:
        logger.error(f"Cache stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def clear_cache():
    """
    Clear all cache entries
    """
    try:
        cache_svc = get_cache_service()
        success = await cache_svc.clear_all_cache()
        if success:
            return {"status": "ok", "message": "Cache cleared successfully"}
        else:
            return {"status": "error", "message": "Failed to clear cache"}
            
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check service health
        cache_svc = get_cache_service()
        cache_stats = await cache_svc.get_cache_stats()
        
        from datetime import datetime, timezone
        return {
            "status": "healthy",
            "services": {
                "cache": cache_stats.get("status", "unknown"),
                "rag_workflow": "available",
                "document_processor": "available"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
