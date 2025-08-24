import time
import uuid
import logging
import asyncio
from typing import List, Dict, Optional, Any
from collections import defaultdict
from src.models.schemas import RAGState, RetrieverType, Citation
from src.clients.retriever_client import RetrieverClient
from src.clients.llm_client import LLMClient
from src.services.cache_service import CacheService
from src.core.config import settings, RETRIEVER_WEIGHTS
import numpy as np

logger = logging.getLogger(__name__)


class RRFFusionService:
    """RRF (Reciprocal Rank Fusion) implementation with adaptive weights"""
    
    def __init__(self, k: int = 60):
        self.k = k
        self.weights = RETRIEVER_WEIGHTS.copy()
    
    def fuse_results(self, retrieval_results: Dict[str, List[Dict]]) -> List[Dict]:
        """RRF fusion with adaptive weighting based on result quality"""
        doc_scores = defaultdict(float)
        doc_metadata = {}
        
        for retriever_type, results in retrieval_results.items():
            if not results:
                continue
                
            weight = self.weights.get(retriever_type, 1.0)
            
            for rank, hit in enumerate(results, 1):
                doc_id = hit['_source']['doc_id']
                rrf_score = weight / (self.k + rank)
                doc_scores[doc_id] += rrf_score
                
                if doc_id not in doc_metadata:
                    doc_metadata[doc_id] = hit
        
        # Sort by combined RRF score
        ranked_docs = sorted(
            [(doc_id, score, doc_metadata[doc_id]) for doc_id, score in doc_scores.items()],
            key=lambda x: x[1], reverse=True
        )
        
        return [doc for _, _, doc in ranked_docs]


class MMRContextSelector:
    """MMR-based context selection for diversity"""
    
    def __init__(self, lambda_param: float = 0.7):
        self.lambda_param = lambda_param  # Balance relevance vs diversity
    
    def select_context(self, documents: List[Dict], token_budget: int = 4000) -> List[Dict]:
        """Select diverse, relevant documents within token budget"""
        if not documents:
            return []
        
        selected = []
        remaining = documents.copy()
        current_tokens = 0
        
        # Select first document (highest relevance)
        if remaining:
            first_doc = remaining.pop(0)
            selected.append(first_doc)
            current_tokens += self._estimate_tokens(first_doc['_source']['content'])
        
        while remaining and current_tokens < token_budget:
            best_doc = None
            best_score = float('-inf')
            
            for doc in remaining:
                content_tokens = self._estimate_tokens(doc['_source']['content'])
                if current_tokens + content_tokens > token_budget:
                    continue
                
                # Calculate MMR score
                relevance_score = doc.get('_score', 0)
                diversity_score = self._calculate_diversity(doc, selected)
                mmr_score = (self.lambda_param * relevance_score - 
                           (1 - self.lambda_param) * diversity_score)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = doc
            
            if best_doc:
                selected.append(best_doc)
                remaining.remove(best_doc)
                current_tokens += self._estimate_tokens(best_doc['_source']['content'])
            else:
                break
        
        return selected
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text) // 4
    
    def _calculate_diversity(self, doc: Dict, selected: List[Dict]) -> float:
        """Calculate diversity score (simplified implementation)"""
        if not selected:
            return 0.0
        
        doc_content = doc['_source']['content'].lower()
        similarities = []
        
        for selected_doc in selected:
            selected_content = selected_doc['_source']['content'].lower()
            # Simple word overlap similarity
            doc_words = set(doc_content.split())
            selected_words = set(selected_content.split())
            if doc_words and selected_words:
                similarity = len(doc_words & selected_words) / len(doc_words | selected_words)
                similarities.append(similarity)
        
        return max(similarities) if similarities else 0.0


class CrossEncoderReranker:
    """Cross-encoder reranking service"""
    
    def __init__(self):
        self.endpoint = settings.RERANKING_ENDPOINT
        self.max_candidates = settings.MAX_RERANK_CANDIDATES
        self.timeout = settings.RERANKING_TIMEOUT
        self.top_k = settings.RERANK_TOP_K
        self.enabled = settings.ENABLE_RERANKING and bool(self.endpoint)
    
    async def rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Rerank documents using cross-encoder"""
        if not self.enabled or len(documents) <= self.top_k:
            return documents[:self.top_k]
        
        # Take top candidates for reranking
        candidates = documents[:self.max_candidates]
        
        try:
            import httpx
            
            # Prepare reranking request
            pairs = [(query, doc['_source']['content'][:512]) for doc in candidates]
            
            # Call reranking service with timeout
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.endpoint,
                    json={'pairs': pairs},
                    headers={'Content-Type': 'application/json'}
                )
                scores = response.json()
            
            # Apply new scores and re-sort
            for doc, score in zip(candidates, scores):
                doc['_rerank_score'] = score
            
            reranked = sorted(candidates, key=lambda x: x.get('_rerank_score', 0), reverse=True)
            return reranked[:self.top_k]
            
        except asyncio.TimeoutError:
            logger.warning("Reranking timeout, falling back to original ranking")
            return documents[:self.top_k]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:self.top_k]


class RAGService:
    """Core RAG service orchestrator"""
    
    def __init__(self):
        self.retriever_client = RetrieverClient()
        self.llm_client = LLMClient()
        self.cache_service = CacheService()
        self.rrf_fusion = RRFFusionService(k=settings.RRF_K)
        self.mmr_selector = MMRContextSelector(lambda_param=settings.MMR_LAMBDA)
        self.reranker = CrossEncoderReranker()
    
    async def process_query(
        self,
        query: str,
        index_name: str,
        permission_groups: List[str],
        retriever_type: RetrieverType = RetrieverType.RRF,
        num_docs: int = 5,
        model: Optional[str] = None,
        answer_format: str = "markdown"
    ) -> Dict[str, Any]:
        """Process complete RAG query"""
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        # Initialize state
        state = RAGState(
            trace_id=trace_id,
            user_id="user",  # TODO: Extract from auth context
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
            retriever_cfg={"num_docs": num_docs, "retriever_type": retriever_type.value},
            permission_groups=permission_groups,
            feedback_hint=None,
            completed_sections=[]
        )
        
        try:
            # Check cache first
            cache_key = self.cache_service.generate_cache_key(
                query, permission_groups, index_name, state["retriever_cfg"]
            )
            cached_response = await self.cache_service.get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Cache hit for query: {query[:50]}")
                cached_response["trace_id"] = trace_id
                return cached_response
            
            # Step 1: Query Expansion
            step_start = time.time()
            if model:
                self.llm_client = LLMClient(model_name=model)
                
            expanded_queries = await self.llm_client.expand_queries(
                query, 
                num_variants=settings.NUM_QUERY_VARIANTS,
                similarity_threshold=settings.QUERY_SIMILARITY_THRESHOLD
            )
            state["queries_expanded"] = expanded_queries
            state["latency_map"]["query_expansion"] = (time.time() - step_start) * 1000
            
            # Step 2: Parallel Retrieval
            step_start = time.time()
            if retriever_type == RetrieverType.RRF:
                # Use multiple retriever types for RRF fusion
                retriever_types = [RetrieverType.BM25, RetrieverType.KNN, RetrieverType.CC]
            else:
                retriever_types = [retriever_type]
            
            retrieval_results = await self.retriever_client.parallel_retrieve(
                queries=expanded_queries,
                index_name=index_name,
                permission_groups=permission_groups,
                retriever_types=retriever_types,
                num_docs=num_docs
            )
            state["retrieval_results"] = retrieval_results
            state["latency_map"]["parallel_retrieval"] = (time.time() - step_start) * 1000
            
            # Step 3: Fusion and Reranking
            step_start = time.time()
            if retriever_type == RetrieverType.RRF:
                fused_results = self.rrf_fusion.fuse_results(retrieval_results)
            else:
                # Use single retriever results
                fused_results = retrieval_results.get(retriever_type.value, [])
            
            # Apply cross-encoder reranking if enabled
            reranked_results = await self.reranker.rerank_documents(query, fused_results)
            state["reranked"] = reranked_results
            state["latency_map"]["fusion_reranking"] = (time.time() - step_start) * 1000
            
            # Step 4: Context Assembly
            step_start = time.time()
            selected_docs = self.mmr_selector.select_context(
                reranked_results, 
                token_budget=settings.TOKEN_BUDGET
            )
            
            # Build context and citations
            context_parts = []
            citations = []
            
            for doc in selected_docs:
                source = doc['_source']
                doc_id = source.get('doc_id', 'unknown')
                title = source.get('title', 'Unknown Document')
                content = source.get('content', '')
                score = doc.get('_score', doc.get('_rerank_score', 0))
                
                context_parts.append(f"[{doc_id}] {content}")
                citations.append(Citation(
                    doc_id=doc_id,
                    title=title,
                    score=float(score),
                    snippet=content[:200] + "..." if len(content) > 200 else content,
                    section_title=source.get('section_title')
                ))
            
            state["context"] = "\n\n".join(context_parts)
            state["citations"] = [c.model_dump() for c in citations]
            state["latency_map"]["context_assembly"] = (time.time() - step_start) * 1000
            
            # Step 5: Answer Generation
            step_start = time.time()
            answer = await self.llm_client.generate_answer(
                query=query,
                context=state["context"],
                citations=state["citations"],
                answer_format=answer_format
            )
            state["answer_final"] = answer
            state["latency_map"]["answer_generation"] = (time.time() - step_start) * 1000
            
            # Prepare response
            total_latency = (time.time() - start_time) * 1000
            response = {
                "answer": answer,
                "citations": citations,
                "latency_ms": total_latency,
                "trace_id": trace_id,
                "debug": {
                    "expanded_queries": len(expanded_queries),
                    "retrieved_docs": sum(len(results) for results in retrieval_results.values()),
                    "selected_docs": len(selected_docs),
                    "latency_breakdown": state["latency_map"]
                }
            }
            
            # Cache the response
            await self.cache_service.cache_response(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"RAG processing failed for query '{query}': {e}")
            total_latency = (time.time() - start_time) * 1000
            
            return {
                "answer": "I apologize, but I encountered an error while processing your query. Please try again.",
                "citations": [],
                "latency_ms": total_latency,
                "trace_id": trace_id,
                "debug": {"error": str(e), "latency_breakdown": state["latency_map"]}
            }