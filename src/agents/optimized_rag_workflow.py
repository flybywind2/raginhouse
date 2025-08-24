#!/usr/bin/env python3
"""
Optimized RAG Workflow using LangGraph StateGraph with Parallel Processing
Based on Context7 best practices for performance optimization
"""
import time
import uuid
import json
import hashlib
import logging
from typing import List, Dict, Optional, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from src.models.schemas import RAGState, RetrieverType, Citation
from src.clients.retriever_client import RetrieverClient
from src.clients.llm_client import LLMClient
from src.services.cache_service import CacheService
from src.services.rag_service import RRFFusionService, MMRContextSelector, CrossEncoderReranker
from src.core.config import settings

logger = logging.getLogger(__name__)


class OptimizedRAGWorkflow:
    """
    Optimized RAG Workflow using LangGraph StateGraph with Parallel Processing
    - Parallel retrieval execution using Send() for fan-out pattern
    - Intelligent caching with TTL policy  
    - Optimized node execution order
    - Advanced fusion and reranking
    
    초보자용 설명:
    - 검색 단계를 병렬로 실행하여 속도를 높입니다.
    - 중간 결과를 캐시에 저장해 반복 요청을 빠르게 처리합니다.
    """
    
    def __init__(self):
        # Initialize services
        self.retriever_client = RetrieverClient()
        self.llm_client = LLMClient()
        self.cache_service = CacheService()
        self.rrf_fusion = RRFFusionService(k=settings.RRF_K)
        self.mmr_selector = MMRContextSelector(lambda_param=settings.MMR_LAMBDA)
        self.reranker = CrossEncoderReranker()
        
        # Build the optimized workflow graph
        self.workflow = self._build_optimized_workflow()

    def _stable_hash(self, obj: Any) -> str:
        """Create a stable hash for cache keys from arbitrary objects.

        초보자용 설명:
        - 파이썬 내장 hash()는 실행마다 값이 달라질 수 있어 캐시에 부적합합니다.
        - json으로 직렬화 후 SHA-1 해시를 써서 항상 같은 키를 만듭니다.
        """
        try:
            data = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            data = str(obj)
        return hashlib.sha1(data.encode("utf-8")).hexdigest()
    
    def _build_optimized_workflow(self) -> StateGraph:
        """Build the optimized LangGraph workflow with parallel processing

        초보자용 설명:
        - 질의 확장 후 BM25/kNN/CC 검색 노드를 병렬로 실행합니다.
        - 세 검색이 모두 끝나면 융합/재랭킹 → 문맥 선택 → 답변 생성 → 평가/개선 순서로 진행합니다.
        """
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("query_expansion", self._query_expansion_node)
        workflow.add_node("retrieve_bm25", self._retrieve_bm25_node)
        workflow.add_node("retrieve_knn", self._retrieve_knn_node)
        workflow.add_node("retrieve_cc", self._retrieve_cc_node)
        workflow.add_node("fusion_and_rerank", self._fusion_and_rerank_node)
        workflow.add_node("context_selection", self._context_selection_node)
        workflow.add_node("answer_generation", self._answer_generation_node)
        workflow.add_node("answer_critique", self._answer_critique_node)
        workflow.add_node("answer_refinement", self._answer_refinement_node)
        
        # Define the optimized flow with parallel retrieval
        workflow.add_edge(START, "query_expansion")
        
        # Fan-out: Query expansion dispatches to parallel retrievers
        workflow.add_conditional_edges(
            "query_expansion",
            self._dispatch_retrieval_tasks,
            ["retrieve_bm25", "retrieve_knn", "retrieve_cc"]
        )
        
        # All retrievers go to fusion (LangGraph handles synchronization)
        workflow.add_edge("retrieve_bm25", "fusion_and_rerank")
        workflow.add_edge("retrieve_knn", "fusion_and_rerank") 
        workflow.add_edge("retrieve_cc", "fusion_and_rerank")
        
        # Continue the pipeline
        workflow.add_edge("fusion_and_rerank", "context_selection")
        workflow.add_edge("context_selection", "answer_generation")
        workflow.add_edge("answer_generation", "answer_critique")
        
        # Conditional refinement
        workflow.add_conditional_edges(
            "answer_critique",
            self._should_refine_answer,
            {
                "refine": "answer_refinement",
                "end": END
            }
        )
        
        workflow.add_edge("answer_refinement", END)
        
        return workflow.compile()
    
    def _dispatch_retrieval_tasks(self, state: RAGState) -> List[Send]:
        """Dispatch parallel retrieval tasks using Send()"""
        return [
            Send("retrieve_bm25", state),
            Send("retrieve_knn", state), 
            Send("retrieve_cc", state)
        ]
    
    async def _query_expansion_node(self, state: RAGState) -> RAGState:
        """Optimized query expansion with intelligent caching

        초보자용 설명:
        - 질문을 여러 표현으로 확장합니다. 동일 질문은 캐시에서 빠르게 가져옵니다.
        """
        start_time = time.time()
        logger.info(f"Expanding query: {state['query_raw'][:50]}")
        
        try:
            # Multi-query expansion using LLM with caching
            cache_key = f"query_expansion:{self._stable_hash(state['query_raw'])}"
            cached_queries = await self.cache_service.get_cached_result(cache_key)
            
            if cached_queries:
                expanded_queries = cached_queries
                logger.info(f"Using cached query expansion: {len(expanded_queries)} queries")
            else:
                expanded_queries = await self.llm_client.expand_queries(
                    query=state['query_raw'],
                    num_variants=settings.NUM_QUERY_VARIANTS,
                    similarity_threshold=settings.QUERY_SIMILARITY_THRESHOLD
                )
                # Cache for 5 minutes
                await self.cache_service.cache_result(cache_key, expanded_queries, ttl=300)
                logger.info(f"Generated {len(expanded_queries)} expanded queries")
            
            state['queries_expanded'] = expanded_queries
            state['latency_map']['query_expansion'] = (time.time() - start_time) * 1000
            
            return state
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            raise e
    
    async def _retrieve_bm25_node(self, state: RAGState) -> RAGState:
        """BM25 retrieval node with caching

        초보자용 설명:
        - 키워드 기반 전통 검색입니다. 질의별 결과를 캐시에 보관합니다.
        """
        start_time = time.time()
        logger.info("Starting BM25 retrieval")
        
        try:
            results = []
            num_docs = state['retriever_cfg'].get('num_docs', 5)
            
            for query in state['queries_expanded']:
                # Check cache first
                cache_key = f"bm25:{self._stable_hash(query)}:{state['index_name']}:{num_docs}"
                cached_result = await self.cache_service.get_cached_result(cache_key)
                
                if cached_result:
                    hits = cached_result
                    logger.info(f"Using cached BM25 results for query: {query[:30]}")
                else:
                    result = await self.retriever_client.retrieve_documents(
                        retriever_type=RetrieverType.BM25,
                        query=query,
                        index_name=state['index_name'],
                        num_docs=num_docs
                    )
                    hits = result.get('hits', {}).get('hits', [])
                    # Cache for 10 minutes
                    await self.cache_service.cache_result(cache_key, hits, ttl=600)
                
                results.extend(hits)
            
            # Update state
            if 'retrieval_results' not in state:
                state['retrieval_results'] = {}
            state['retrieval_results']['bm25'] = results
            state['latency_map']['retrieve_bm25'] = (time.time() - start_time) * 1000
            
            logger.info(f"BM25 retrieved {len(results)} documents")
            return state
            
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            raise e
    
    async def _retrieve_knn_node(self, state: RAGState) -> RAGState:
        """kNN retrieval node with caching

        초보자용 설명:
        - 임베딩 유사도 기반 검색입니다. 결과를 캐시에 보관합니다.
        """
        start_time = time.time()
        logger.info("Starting kNN retrieval")
        
        try:
            results = []
            num_docs = state['retriever_cfg'].get('num_docs', 5)
            
            for query in state['queries_expanded']:
                # Check cache first
                cache_key = f"knn:{self._stable_hash(query)}:{state['index_name']}:{num_docs}"
                cached_result = await self.cache_service.get_cached_result(cache_key)
                
                if cached_result:
                    hits = cached_result
                    logger.info(f"Using cached kNN results for query: {query[:30]}")
                else:
                    result = await self.retriever_client.retrieve_documents(
                        retriever_type=RetrieverType.KNN,
                        query=query,
                        index_name=state['index_name'],
                        num_docs=num_docs
                    )
                    hits = result.get('hits', {}).get('hits', [])
                    # Cache for 10 minutes
                    await self.cache_service.cache_result(cache_key, hits, ttl=600)
                
                results.extend(hits)
            
            # Update state
            if 'retrieval_results' not in state:
                state['retrieval_results'] = {}
            state['retrieval_results']['knn'] = results
            state['latency_map']['retrieve_knn'] = (time.time() - start_time) * 1000
            
            logger.info(f"kNN retrieved {len(results)} documents")
            return state
            
        except Exception as e:
            logger.error(f"kNN retrieval failed: {e}")
            raise e
    
    async def _retrieve_cc_node(self, state: RAGState) -> RAGState:
        """CC retrieval node with caching

        초보자용 설명:
        - 커스텀/결합형 검색(환경에 따라 달라질 수 있음)으로 가정합니다. 결과를 캐시에 보관합니다.
        """
        start_time = time.time()
        logger.info("Starting CC retrieval")
        
        try:
            results = []
            num_docs = state['retriever_cfg'].get('num_docs', 5)
            
            for query in state['queries_expanded']:
                # Check cache first
                cache_key = f"cc:{self._stable_hash(query)}:{state['index_name']}:{num_docs}"
                cached_result = await self.cache_service.get_cached_result(cache_key)
                
                if cached_result:
                    hits = cached_result
                    logger.info(f"Using cached CC results for query: {query[:30]}")
                else:
                    result = await self.retriever_client.retrieve_documents(
                        retriever_type=RetrieverType.CC,
                        query=query,
                        index_name=state['index_name'],
                        num_docs=num_docs
                    )
                    hits = result.get('hits', {}).get('hits', [])
                    # Cache for 10 minutes
                    await self.cache_service.cache_result(cache_key, hits, ttl=600)
                
                results.extend(hits)
            
            # Update state
            if 'retrieval_results' not in state:
                state['retrieval_results'] = {}
            state['retrieval_results']['cc'] = results
            state['latency_map']['retrieve_cc'] = (time.time() - start_time) * 1000
            
            logger.info(f"CC retrieved {len(results)} documents")
            return state
            
        except Exception as e:
            logger.error(f"CC retrieval failed: {e}")
            raise e
    
    async def _fusion_and_rerank_node(self, state: RAGState) -> RAGState:
        """RRF fusion and reranking node with caching

        초보자용 설명:
        - 서로 다른 검색기의 결과를 RRF로 합치고, 필요하면 교차 인코더로 재정렬합니다.
        - 이 중간 결과도 캐시하여 성능을 높입니다.
        """
        start_time = time.time()
        logger.info("Fusing and reranking results")
        
        try:
            retrieval_results = state['retrieval_results']
            
            # Check cache for fusion results
            cache_key = f"fusion:{self._stable_hash(sorted(retrieval_results.keys()))}:{self._stable_hash(state['queries_expanded'])}"
            cached_fused = await self.cache_service.get_cached_result(cache_key)
            
            if cached_fused:
                fused_results = cached_fused
                logger.info(f"Using cached fusion results: {len(fused_results)} docs")
            else:
                # Apply RRF fusion
                fused_results = self.rrf_fusion.fuse_results(retrieval_results)
                
                # Apply cross-encoder reranking if enabled
                if settings.ENABLE_RERANKING and len(fused_results) > 0:
                    reranked_results = await self.reranker.rerank_documents(
                        query=state['query_raw'],
                        documents=fused_results[:settings.MAX_RERANK_CANDIDATES],
                        top_k=settings.RERANK_TOP_K
                    )
                    fused_results = reranked_results
                
                # Cache for 3 minutes
                await self.cache_service.cache_result(cache_key, fused_results, ttl=180)
                logger.info(f"Generated fusion results: {len(fused_results)} docs")
            
            state['reranked'] = fused_results
            state['latency_map']['fusion_and_rerank'] = (time.time() - start_time) * 1000
            
            logger.info(f"Fused and reranked to {len(fused_results)} documents")
            return state
            
        except Exception as e:
            logger.error(f"Fusion and reranking failed: {e}")
            raise e
    
    async def _context_selection_node(self, state: RAGState) -> RAGState:
        """MMR context selection with caching

        초보자용 설명:
        - MMR를 사용해 중복을 줄이고, 다양한 문서를 선별해 문맥을 만듭니다.
        """
        start_time = time.time()
        logger.info("Selecting context")
        
        try:
            # Check cache for context selection
            doc_ids = [doc.get('_id', '') for doc in state['reranked']]
            docs_hash = self._stable_hash(doc_ids)
            cache_key = f"context:{docs_hash}:{self._stable_hash(state['query_raw'])}"
            cached_context = await self.cache_service.get_cached_result(cache_key)
            
            if cached_context:
                context, citations = cached_context
                logger.info(f"Using cached context: {len(context)} chars")
            else:
                # Select diverse documents within token budget
                selected_docs = self.mmr_selector.select_context(
                    documents=state['reranked'],
                    token_budget=settings.TOKEN_BUDGET
                )
                
                # Build context and citations
                context_parts = []
                citations = []
                
                for i, doc in enumerate(selected_docs):
                    doc_content = doc.get('_source', {}).get('content', '')
                    context_parts.append(f"Document {i+1}: {doc_content}")
                    
                    citations.append(Citation(
                        doc_id=doc.get('_id', f'doc_{i+1}'),
                        title=doc.get('_source', {}).get('title', f'Document {i+1}'),
                        score=doc.get('_score', 0.0),
                        snippet=doc_content[:200] + "..." if len(doc_content) > 200 else doc_content,
                        section_title=doc.get('_source', {}).get('section_title')
                    ))
                
                context = "\n\n".join(context_parts)
                
                # Cache for 2 minutes
                await self.cache_service.cache_result(cache_key, (context, citations), ttl=120)
                logger.info(f"Generated context: {len(context)} chars")
            
            state['context'] = context
            state['citations'] = [citation.dict() if hasattr(citation, 'dict') else citation for citation in citations]
            state['latency_map']['context_selection'] = (time.time() - start_time) * 1000
            
            logger.info(f"Selected context from {len(citations)} documents")
            return state
            
        except Exception as e:
            logger.error(f"Context selection failed: {e}")
            raise e
    
    async def _answer_generation_node(self, state: RAGState) -> RAGState:
        """LLM answer generation with caching

        초보자용 설명:
        - 문맥을 바탕으로 LLM이 답변 초안을 생성합니다. 동일 입력이면 캐시를 사용합니다.
        """
        start_time = time.time()
        logger.info("Generating answer")
        
        try:
            # Check cache for answer generation
            cache_key = f"answer:{self._stable_hash(state['query_raw'])}:{self._stable_hash(state['context'])}"
            cached_answer = await self.cache_service.get_cached_result(cache_key)
            
            if cached_answer:
                answer = cached_answer
                logger.info("Using cached answer")
            else:
                answer = await self.llm_client.generate_answer(
                    query=state['query_raw'],
                    context=state['context'],
                    citations=state['citations'],
                    answer_format=state['retriever_cfg'].get('answer_format', 'markdown')
                )
                
                # Cache for 1 minute
                await self.cache_service.cache_result(cache_key, answer, ttl=60)
                logger.info("Generated new answer")
            
            state['answer_draft'] = answer
            state['answer_final'] = answer  # Will be updated if refined
            state['latency_map']['answer_generation'] = (time.time() - start_time) * 1000
            
            logger.info("Answer generated successfully")
            return state
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise e
    
    async def _answer_critique_node(self, state: RAGState) -> RAGState:
        """Answer critique and quality assessment

        초보자용 설명:
        - 생성된 답변의 품질을 체크해 개선 필요 여부를 기록합니다.
        """
        start_time = time.time()
        logger.info("Critiquing answer")
        
        try:
            critique = await self.llm_client.critique_answer(
                query=state['query_raw'],
                answer=state['answer_draft'],
                context=state['context']
            )
            
            # Store critique in feedback_hint for decision making
            state['feedback_hint'] = str(critique.get('needs_refinement', False))
            state['latency_map']['answer_critique'] = (time.time() - start_time) * 1000
            
            logger.info(f"Answer critique completed, needs refinement: {critique.get('needs_refinement', False)}")
            return state
            
        except Exception as e:
            logger.error(f"Answer critique failed: {e}")
            raise e
    
    async def _answer_refinement_node(self, state: RAGState) -> RAGState:
        """Answer refinement node

        초보자용 설명:
        - 필요 시 평가 결과를 반영해 답변을 개선합니다.
        """
        start_time = time.time()
        logger.info("Refining answer")
        
        try:
            refined_answer = await self.llm_client.refine_answer(
                query=state['query_raw'],
                draft_answer=state['answer_draft'],
                critique="Please improve accuracy and completeness",
                context=state['context'],
                answer_format=state['retriever_cfg'].get('answer_format', 'markdown')
            )
            
            state['answer_final'] = refined_answer
            state['latency_map']['answer_refinement'] = (time.time() - start_time) * 1000
            
            logger.info("Answer refined successfully")
            return state
            
        except Exception as e:
            logger.error(f"Answer refinement failed: {e}")
            raise e
    
    def _should_refine_answer(self, state: RAGState) -> Literal["refine", "end"]:
        """Decide whether to refine the answer based on critique"""
        needs_refinement = state.get('feedback_hint', 'False').lower() in ['true', '1', 'yes']
        return "refine" if needs_refinement else "end"
    
    async def execute_optimized_workflow(
        self,
        query: str,
        index_name: str,
        retriever_type: RetrieverType = RetrieverType.RRF,
        num_docs: int = 5,
        model: Optional[str] = None,
        answer_format: str = "markdown"
    ) -> Dict[str, Any]:
        """Execute the optimized RAG workflow with LangGraph StateGraph

        초보자용 설명:
        - 표준 워크플로우와 같은 형태의 응답을 돌려주며, 캐시/병렬처리로 속도를 개선합니다.
        """
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        logger.info(f"Starting optimized workflow for query: {query[:50]}")
        
        try:
            # Create LLM client with selected model if provided
            if model and model != self.llm_client.model_name:
                self.llm_client = LLMClient(model_name=model)
                logger.info(f"Using selected model: {model}")
            
            # Initialize state
            initial_state = RAGState(
                trace_id=trace_id,
                user_id="user",
                index_name=index_name,
                query_raw=query,
                queries_expanded=[],
                retrieval_results={},
                reranked=[],
                context="",
                answer_draft="",
                answer_final="",
                citations=[],
                latency_map={},
                retriever_cfg={
                    "num_docs": num_docs,
                    "retriever_type": retriever_type.value,
                    "answer_format": answer_format
                },
                feedback_hint=None,
                completed_sections=[]
            )
            
            # Execute the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Calculate total latency
            total_latency = (time.time() - start_time) * 1000
            
            # Prepare response
            response = {
                "answer": final_state['answer_final'],
                "citations": final_state['citations'],
                "latency_ms": total_latency,
                "trace_id": trace_id,
                "debug": {
                    "expanded_queries": len(final_state.get('queries_expanded', [])),
                    "bm25_results": len(final_state.get('retrieval_results', {}).get('bm25', [])),
                    "knn_results": len(final_state.get('retrieval_results', {}).get('knn', [])),
                    "cc_results": len(final_state.get('retrieval_results', {}).get('cc', [])),
                    "fused_results": len(final_state.get('reranked', [])),
                    "context_length": len(final_state.get('context', '')),
                    "model_used": model or settings.MODEL_NAME,
                    "workflow_type": "optimized_langgraph",
                    "latency_breakdown": final_state.get('latency_map', {})
                }
            }
            
            logger.info(f"Optimized workflow completed in {total_latency:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Optimized workflow failed: {e}")
            total_latency = (time.time() - start_time) * 1000
            
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "citations": [],
                "latency_ms": total_latency,
                "trace_id": trace_id,
                "debug": {
                    "error": str(e),
                    "workflow_type": "optimized_langgraph"
                }
            }


# Global instance for use in routes
optimized_rag_workflow = OptimizedRAGWorkflow()
