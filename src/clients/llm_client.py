import uuid
import asyncio
import logging
from typing import List, Dict, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from src.core.config import settings, get_llm_headers, get_model_endpoint

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM Client compatible with internal_llm.py pattern
    Supports dynamic model selection and proper header management
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.base_url = get_model_endpoint(self.model_name)
        self.timeout = settings.REQUEST_TIMEOUT
        self.max_retries = settings.MAX_RETRIES
        
        # Initialize ChatOpenAI with proper headers (compatible with internal_llm.py)
        self._llm = ChatOpenAI(
            base_url=self.base_url,
            model=self.model_name,
            default_headers=get_llm_headers(),
            timeout=self.timeout,
            max_retries=self.max_retries,
            temperature=0
        )
        
        self.parser = StrOutputParser()
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response from messages"""
        try:
            # Convert dict messages to LangChain message objects
            langchain_messages = []
            
            if system_prompt:
                langchain_messages.append(SystemMessage(content=system_prompt))
            
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "human" or msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant" or msg["role"] == "ai":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Generate response
            response = await self._llm.ainvoke(langchain_messages)
            return response.content
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    async def expand_queries(
        self,
        query: str,
        num_variants: int = 3,
        similarity_threshold: float = 0.85
    ) -> List[str]:
        """Multi-query expansion for improved retrieval"""
        system_prompt = """You are an expert at query expansion for information retrieval.
        Generate semantically diverse but related query variants that would help retrieve relevant documents.
        Each variant should maintain the core intent while exploring different phrasings, synonyms, and related concepts.
        Return only the query variants, one per line, without numbering or additional text."""
        
        human_prompt = f"""Original query: "{query}"
        
        Generate {num_variants} diverse query variants that would help retrieve the same type of information:"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "human", "content": human_prompt}
            ]
            
            response = await self.generate_response(messages)
            
            # Parse variants
            variants = [line.strip() for line in response.split('\n') if line.strip()]
            variants = [query] + variants[:num_variants]  # Include original query
            
            # Remove duplicates while preserving order
            seen = set()
            unique_variants = []
            for variant in variants:
                if variant.lower() not in seen:
                    seen.add(variant.lower())
                    unique_variants.append(variant)
            
            return unique_variants[:num_variants + 1]
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            raise e
    
    async def generate_metadata(
        self,
        title: str,
        content_sample: str,
        doc_type: str = "unknown"
    ) -> Dict[str, Any]:
        """Generate metadata for documents using LLM"""
        system_prompt = """You are a document analysis expert. Analyze the provided document and generate structured metadata.
        
        Return your analysis in the following format:
        SUMMARY: [2-3 sentence summary]
        TOPICS: [comma-separated list of 3-5 main topics/keywords]
        LANGUAGE: [detected language code: en, ko, etc.]
        PII: [true/false - contains personally identifiable information]
        TYPE: [document type: report, manual, presentation, etc.]"""
        
        human_prompt = f"""Document Title: {title}
        Document Type: {doc_type}
        Content Sample: {content_sample[:1000]}...
        
        Provide structured metadata analysis:"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "human", "content": human_prompt}
            ]
            
            response = await self.generate_response(messages)
            
            # Parse structured response
            metadata = {
                "summary": "",
                "topics": [],
                "language": "unknown",
                "has_pii": False,
                "doc_type": doc_type
            }
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('SUMMARY:'):
                    metadata["summary"] = line.replace('SUMMARY:', '').strip()
                elif line.startswith('TOPICS:'):
                    topics_str = line.replace('TOPICS:', '').strip()
                    metadata["topics"] = [t.strip() for t in topics_str.split(',')]
                elif line.startswith('LANGUAGE:'):
                    metadata["language"] = line.replace('LANGUAGE:', '').strip()
                elif line.startswith('PII:'):
                    pii_str = line.replace('PII:', '').strip().lower()
                    metadata["has_pii"] = pii_str in ['true', 'yes', '1']
                elif line.startswith('TYPE:'):
                    metadata["doc_type"] = line.replace('TYPE:', '').strip()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {
                "summary": "Summary not available",
                "topics": [],
                "language": "unknown",
                "has_pii": False,
                "doc_type": doc_type
            }
    
    async def generate_answer(
        self,
        query: str,
        context: str,
        citations: List[Dict[str, Any]],
        answer_format: str = "markdown"
    ) -> str:
        """Generate final answer based on context and citations"""
        system_prompt = f"""You are a helpful AI assistant that answers questions based solely on the provided context.

        IMPORTANT INSTRUCTIONS:
        1. Use ONLY the information provided in the context to answer the question
        2. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer that question based on the provided context."
        3. Include citation markers [doc_id] for each piece of information you use
        4. Format your response in {answer_format} format
        5. Be concise but comprehensive
        6. If you're uncertain about any information, explicitly state your uncertainty
        
        Context citations format: Each piece of context is marked with [doc_id] for reference."""
        
        # Build context with citations
        context_with_citations = ""
        for i, citation in enumerate(citations):
            doc_id = citation.get("doc_id", f"doc_{i}")
            content = citation.get("snippet", citation.get("content", ""))
            context_with_citations += f"[{doc_id}] {content}\n\n"
        
        human_prompt = f"""Question: {query}

        Context:
        {context_with_citations}
        
        Please provide a comprehensive answer based on the context above:"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "human", "content": human_prompt}
            ]
            
            response = await self.generate_response(messages)
            return response
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I apologize, but I encountered an error while generating an answer. Please try again."
    
    async def critique_answer(
        self,
        query: str,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """Self-critique the generated answer"""
        system_prompt = """You are a critical evaluator of AI-generated answers. Analyze the provided answer for:
        1. Factual accuracy based on the context
        2. Relevance to the question
        3. Completeness of the answer
        4. Proper use of citations
        
        Return your evaluation in this format:
        ACCURACY: [HIGH/MEDIUM/LOW] - brief explanation
        RELEVANCE: [HIGH/MEDIUM/LOW] - brief explanation  
        COMPLETENESS: [HIGH/MEDIUM/LOW] - brief explanation
        CITATIONS: [GOOD/POOR] - brief explanation
        NEEDS_REFINEMENT: [YES/NO]
        SUGGESTED_IMPROVEMENTS: [specific suggestions if any]"""
        
        human_prompt = f"""Question: {query}
        
        Generated Answer: {answer}
        
        Available Context: {context[:1000]}...
        
        Evaluate the answer:"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "human", "content": human_prompt}
            ]
            
            response = await self.generate_response(messages)
            
            # Parse critique
            critique = {
                "accuracy": "MEDIUM",
                "relevance": "MEDIUM", 
                "completeness": "MEDIUM",
                "citations": "GOOD",
                "needs_refinement": False,
                "suggestions": []
            }
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('NEEDS_REFINEMENT:'):
                    needs_ref = line.replace('NEEDS_REFINEMENT:', '').strip().upper()
                    critique["needs_refinement"] = needs_ref in ['YES', 'TRUE', '1']
                elif line.startswith('SUGGESTED_IMPROVEMENTS:'):
                    suggestions = line.replace('SUGGESTED_IMPROVEMENTS:', '').strip()
                    if suggestions and suggestions.lower() != 'none':
                        critique["suggestions"] = [suggestions]
            
            return critique
            
        except Exception as e:
            logger.error(f"Answer critique failed: {e}")
            return {
                "accuracy": "UNKNOWN",
                "relevance": "UNKNOWN",
                "completeness": "UNKNOWN", 
                "citations": "UNKNOWN",
                "needs_refinement": False,
                "suggestions": []
            }
    
    async def refine_answer(
        self,
        query: str,
        draft_answer: str,
        critique: str,
        context: str,
        answer_format: str = "markdown"
    ) -> str:
        """Refine answer based on critique"""
        system_prompt = f"""You are a helpful AI assistant that refines answers based on feedback.
        
        Your task is to improve the draft answer by addressing the critique points while maintaining the {answer_format} format.
        
        Guidelines:
        - Address specific issues mentioned in the critique
        - Maintain factual accuracy based on the provided context
        - Keep the same format and style as the original answer
        - Make the answer more complete and relevant
        - Ensure proper citation of sources"""
        
        human_prompt = f"""Original Query: {query}
        
        Draft Answer:
        {draft_answer}
        
        Critique and Improvement Suggestions:
        {critique}
        
        Available Context:
        {context}
        
        Please provide a refined answer that addresses the critique points:"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "human", "content": human_prompt}
            ]
            
            refined_answer = await self.generate_response(messages)
            logger.info("Answer refinement completed")
            return refined_answer
            
        except Exception as e:
            logger.error(f"Answer refinement failed: {e}")
            return draft_answer  # Return original if refinement fails