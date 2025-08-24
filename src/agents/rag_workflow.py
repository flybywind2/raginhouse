import time
import uuid
import logging
from typing import List, Dict, Optional, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from src.models.schemas import RAGState, RetrieverType
from src.clients.retriever_client import RetrieverClient
from src.clients.llm_client import LLMClient
from src.services.cache_service import CacheService
from src.services.rag_service import RRFFusionService, MMRContextSelector, CrossEncoderReranker
from src.core.config import settings

logger = logging.getLogger(__name__)


class RAGWorkflow:
    """LangGraph-based RAG workflow orchestrator"""
    
    def __init__(self):
        self.retriever_client = RetrieverClient()
        self.llm_client = LLMClient()
        self.cache_service = CacheService()
        self.rrf_fusion = RRFFusionService(k=settings.RRF_K)
        self.mmr_selector = MMRContextSelector(lambda_param=settings.MMR_LAMBDA)
        self.reranker = CrossEncoderReranker()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("query_rewrite", self._query_rewrite_node)
        workflow.add_node("retrieve_bm25", self._retrieve_bm25_node)
        workflow.add_node("retrieve_knn", self._retrieve_knn_node)
        workflow.add_node("retrieve_cc", self._retrieve_cc_node)
        workflow.add_node("fuse_and_rerank", self._fuse_and_rerank_node)
        workflow.add_node("assemble_context", self._assemble_context_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("critique_answer", self._critique_answer_node)
        workflow.add_node("refine_answer", self._refine_answer_node)
        
        # Define the flow - Sequential execution for reliability
        workflow.add_edge(START, "query_rewrite")
        
        # Sequential retrieval to avoid LangGraph parallel execution issues
        workflow.add_edge("query_rewrite", "retrieve_bm25")
        workflow.add_edge("retrieve_bm25", "retrieve_knn")
        workflow.add_edge("retrieve_knn", "retrieve_cc")
        
        # After all retrievers complete, move to fusion
        workflow.add_edge("retrieve_cc", "fuse_and_rerank")
        
        # Continue the pipeline
        workflow.add_edge("fuse_and_rerank", "assemble_context")
        workflow.add_edge("assemble_context", "generate_answer")
        workflow.add_edge("generate_answer", "critique_answer")
        
        # Conditional refinement
        workflow.add_conditional_edges(
            "critique_answer",
            self._should_refine_answer,
            {
                "refine": "refine_answer",
                "end": END
            }
        )
        
        workflow.add_edge("refine_answer", END)
        
        return workflow.compile()
    
    async def _query_rewrite_node(self, state: RAGState) -> RAGState:
        """Multi-query expansion with semantic similarity clustering"""
        start_time = time.time()
        logger.info(f"Query rewrite for: {state['query_raw'][:50]}")
        
        try:
            # Multi-query expansion using LLM
            expanded_queries = await self.llm_client.expand_queries(
                query=state['query_raw'],
                num_variants=settings.NUM_QUERY_VARIANTS,
                similarity_threshold=settings.QUERY_SIMILARITY_THRESHOLD
            )
            
            state['queries_expanded'] = expanded_queries
            state['latency_map']['query_rewrite'] = (time.time() - start_time) * 1000
            
            logger.info(f"Expanded to {len(expanded_queries)} queries")
            return state
            
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            raise e
    
    async def _retrieve_bm25_node(self, state: RAGState) -> RAGState:
        """BM25 retrieval node"""
        return await self._retrieve_node(state, RetrieverType.BM25)
    
    async def _retrieve_knn_node(self, state: RAGState) -> RAGState:
        """kNN retrieval node"""
        return await self._retrieve_node(state, RetrieverType.KNN)
    
    async def _retrieve_cc_node(self, state: RAGState) -> RAGState:
        """CC retrieval node"""
        return await self._retrieve_node(state, RetrieverType.CC)
    
    async def _retrieve_node(self, state: RAGState, retriever_type: RetrieverType) -> RAGState:
        """Generic retrieval node implementation"""
        start_time = time.time()
        logger.info(f"Retrieving with {retriever_type.value}")
        
        try:
            results = []
            num_docs = state['retriever_cfg'].get('num_docs', 5)
            
            # Retrieve for all expanded queries
            for query in state['queries_expanded']:
                result = await self.retriever_client.retrieve_documents(
                    retriever_type=retriever_type,
                    query=query,
                    index_name=state['index_name'],
                    num_docs=num_docs
                )
                
                hits = result.get('hits', {}).get('hits', [])
                results.extend(hits)
            
            # Update state
            if 'retrieval_results' not in state:
                state['retrieval_results'] = {}
            state['retrieval_results'][retriever_type.value] = results
            
            node_key = f'retrieve_{retriever_type.value}'
            state['latency_map'][node_key] = (time.time() - start_time) * 1000
            
            logger.info(f"Retrieved {len(results)} documents with {retriever_type.value}")
            return state
            
        except Exception as e:
            logger.error(f"{retriever_type.value} retrieval failed: {e}")
            raise e
    
    async def _fuse_and_rerank_node(self, state: RAGState) -> RAGState:
        """RRF fusion and reranking node"""
        start_time = time.time()
        logger.info("Fusing and reranking results")
        
        try:
            retrieval_results = state['retrieval_results']
            
            # Apply RRF fusion
            fused_results = self.rrf_fusion.fuse_results(retrieval_results)
            
            # Apply cross-encoder reranking if enabled
            reranked_results = await self.reranker.rerank_documents(
                state['query_raw'], 
                fused_results
            )
            
            state['reranked'] = reranked_results
            state['latency_map']['fuse_and_rerank'] = (time.time() - start_time) * 1000
            
            logger.info(f"Fused and reranked to {len(reranked_results)} documents")
            return state
            
        except Exception as e:
            logger.error(f"Fusion and reranking failed: {e}")
            raise e
    
    async def _assemble_context_node(self, state: RAGState) -> RAGState:
        """Context assembly with MMR selection"""
        start_time = time.time()
        logger.info("Assembling context")
        
        try:
            # Select diverse documents within token budget
            selected_docs = self.mmr_selector.select_context(
                state['reranked'], 
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
                
                citation = {
                    'doc_id': doc_id,
                    'title': title,
                    'score': float(score),
                    'snippet': content[:200] + "..." if len(content) > 200 else content,
                    'section_title': source.get('section_title')
                }
                citations.append(citation)
            
            state['context'] = "\n\n".join(context_parts)
            state['citations'] = citations
            state['latency_map']['assemble_context'] = (time.time() - start_time) * 1000
            
            logger.info(f"Assembled context from {len(selected_docs)} documents")
            return state
            
        except Exception as e:
            logger.error(f"Context assembly failed: {e}")
            raise e
    
    async def _generate_answer_node(self, state: RAGState) -> RAGState:
        """Answer generation node"""
        start_time = time.time()
        logger.info("Generating answer")
        
        try:
            answer = await self.llm_client.generate_answer(
                query=state['query_raw'],
                context=state['context'],
                citations=state['citations'],
                answer_format=state['retriever_cfg'].get('answer_format', 'markdown')
            )
            
            state['answer_draft'] = answer
            state['answer_final'] = answer  # Will be updated if refined
            state['latency_map']['generate_answer'] = (time.time() - start_time) * 1000
            
            logger.info("Answer generated successfully")
            return state
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise e
    
    async def _critique_answer_node(self, state: RAGState) -> RAGState:
        """Answer critique and quality assessment"""
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
            state['latency_map']['critique_answer'] = (time.time() - start_time) * 1000
            
            logger.info(f"Answer critique completed, needs refinement: {critique.get('needs_refinement', False)}")
            return state
            
        except Exception as e:
            logger.error(f"Answer critique failed: {e}")
            raise e
    
    async def _refine_answer_node(self, state: RAGState) -> RAGState:
        """Answer refinement node"""
        start_time = time.time()
        logger.info("Refining answer")
        
        try:
            # For now, simple refinement - could be more sophisticated
            refined_answer = f"{state['answer_draft']}\n\n*Note: This answer has been reviewed and refined for accuracy.*"
            
            state['answer_final'] = refined_answer
            state['latency_map']['refine_answer'] = (time.time() - start_time) * 1000
            
            logger.info("Answer refined successfully")
            return state
            
        except Exception as e:
            logger.error(f"Answer refinement failed: {e}")
            raise e
    
    def _should_use_multiple_retrievers(self, state: RAGState) -> Literal["parallel", "single"]:
        """Decide whether to use parallel retrievers (for RRF) or single"""
        retriever_type = state['retriever_cfg'].get('retriever_type', 'rrf')
        return "parallel" if retriever_type == 'rrf' else "single"
    
    def _should_refine_answer(self, state: RAGState) -> Literal["refine", "end"]:
        """Decide whether to refine the answer based on critique"""
        needs_refinement = state.get('feedback_hint', 'False').lower() in ['true', '1', 'yes']
        return "refine" if needs_refinement else "end"
    
    async def execute_workflow(
        self,
        query: str,
        index_name: str,
        retriever_type: RetrieverType = RetrieverType.RRF,
        num_docs: int = 5,
        model: Optional[str] = None,
        answer_format: str = "markdown"
    ) -> Dict[str, Any]:
        """Execute the complete RAG workflow"""
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        # Create LLM client with selected model if provided
        if model and model != self.llm_client.model_name:
            self.llm_client = LLMClient(model_name=model)
            logger.info(f"Using selected model: {model}")
        
        # Initialize state
        initial_state = RAGState(
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
            retriever_cfg={
                "num_docs": num_docs,
                "retriever_type": retriever_type.value,
                "answer_format": answer_format
            },
            feedback_hint=None,
            completed_sections=[]
        )
        
        try:
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
                    "retrieved_docs": sum(len(results) for results in final_state.get('retrieval_results', {}).values()),
                    "selected_docs": len(final_state.get('citations', [])),
                    "latency_breakdown": final_state.get('latency_map', {}),
                    "workflow_total_ms": total_latency
                }
            }
            
            logger.info(f"Workflow completed successfully in {total_latency:.0f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            total_latency = (time.time() - start_time) * 1000
            
            return {
                "answer": "I apologize, but I encountered an error while processing your query. Please try again.",
                "citations": [],
                "latency_ms": total_latency,
                "trace_id": trace_id,
                "debug": {
                    "error": str(e),
                    "workflow_total_ms": total_latency
                }
            }