import os
import tempfile
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import requests
from fastapi import UploadFile
from src.clients.llm_client import LLMClient
from src.clients.document_client import DocumentClient
from src.models.schemas import ProcessedDocument, DocumentMetadata
from src.core.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Document processing service using docling and LLM metadata generation"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.document_client = DocumentClient()
        self.max_chunk_size = settings.MAX_CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    async def process_uploaded_file(
        self,
        file: UploadFile,
        index_name: str
    ) -> bool:
        """Process uploaded file with docling and ingest to RAG system"""
        temp_file_path = None
        
        try:
            # Save uploaded file temporarily
            suffix = Path(file.filename).suffix if file.filename else '.tmp'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            doc_id = self._generate_doc_id(file.filename or "unknown")
            
            # Process document
            processed_doc = await self._process_document_file(
                temp_file_path,
                doc_id,
                file.filename or "unknown"
            )
            
            # Ingest to RAG system
            success = await self.document_client.insert_documents(
                processed_doc.chunks,
                index_name
            )
            
            if success:
                logger.info(f"Successfully processed and ingested file: {file.filename}")
            else:
                logger.error(f"Failed to ingest processed file: {file.filename}")
                
            return success
            
        except Exception as e:
            logger.error(f"File processing failed for {file.filename}: {e}")
            return False
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    async def process_confluence_page(
        self,
        base_url: str,
        page_id: str,
        index_name: str
    ) -> bool:
        """Process Confluence page with single-page collection"""
        try:
            # Fetch page content from Confluence
            page_data = await self._fetch_confluence_page(base_url, page_id)
            if not page_data:
                return False
            
            doc_id = f"confluence_{page_id}"
            
            # Process the page content
            processed_doc = await self._process_confluence_content(
                page_data,
                doc_id
            )
            
            # Ingest to RAG system
            success = await self.document_client.insert_documents(
                processed_doc.chunks,
                index_name
            )
            
            if success:
                logger.info(f"Successfully processed and ingested Confluence page: {page_id}")
            else:
                logger.error(f"Failed to ingest Confluence page: {page_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Confluence processing failed for page {page_id}: {e}")
            return False
    
    async def _process_document_file(
        self,
        file_path: str,
        doc_id: str,
        filename: str
    ) -> ProcessedDocument:
        """Process document file using docling"""
        
        return await self._process_with_docling(file_path, doc_id, filename)
    
    async def _process_with_docling(
        self,
        file_path: str,
        doc_id: str,
        filename: str
    ) -> ProcessedDocument:
        """Process document with docling for structure awareness"""
        try:
            # Import docling dynamically to handle optional dependency
            from docling.document_converter import DocumentConverter
            
            converter = DocumentConverter()
            result = converter.convert(file_path)
            doc = result.document
            
            # Extract structured content
            sections = []
            for section in doc.sections:
                section_data = {
                    'title': section.title,
                    'content': section.text,
                    'level': section.level,
                    'tables': [self._serialize_table(table) for table in section.tables],
                    'figures': [self._serialize_figure(fig) for fig in section.figures]
                }
                sections.append(section_data)
            
            # Generate metadata using LLM
            title = doc.title or filename
            content_sample = " ".join([s['content'][:500] for s in sections[:3]])  # First 3 sections
            
            metadata_dict = await self.llm_client.generate_metadata(
                title=title,
                content_sample=content_sample,
                doc_type=self._detect_doc_type(filename)
            )
            
            metadata = DocumentMetadata(
                doc_id=doc_id,
                title=title,
                summary=metadata_dict.get('summary', ''),
                topics=metadata_dict.get('topics', []),
                language=metadata_dict.get('language', 'unknown'),
                has_pii=metadata_dict.get('has_pii', False),
                doc_type=metadata_dict.get('doc_type', 'unknown'),
                created_time=datetime.now().isoformat(),
                permission_groups=[]
            )
            
            # Structure-aware chunking
            chunks = await self._chunk_with_structure_awareness(sections, metadata)
            
            return ProcessedDocument(metadata=metadata, chunks=chunks)
            
        except ImportError as e:
            logger.error("Docling not available")
            raise e
        except Exception as e:
            logger.error(f"Docling processing failed: {e}")
            raise e
    
    
    async def _fetch_confluence_page(self, base_url: str, page_id: str) -> Optional[Dict[str, Any]]:
        """Fetch Confluence page content"""
        try:
            # Construct API URL
            api_url = f"{base_url.rstrip('/')}/rest/api/content/{page_id}?expand=body.storage,metadata.labels"
            
            headers = {}
            if settings.CONFLUENCE_TOKEN:
                headers["Authorization"] = f"Bearer {settings.CONFLUENCE_TOKEN}"
            
            # Make request with SSL verification disabled (MVP setting)
            response = requests.get(
                api_url,
                headers=headers,
                verify=settings.CONFLUENCE_VERIFY_SSL,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Confluence API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Confluence page fetch failed: {e}")
            return None
    
    async def _process_confluence_content(
        self,
        page_data: Dict[str, Any],
        doc_id: str
    ) -> ProcessedDocument:
        """Process Confluence page data"""
        try:
            title = page_data.get('title', 'Unknown Page')
            content = page_data.get('body', {}).get('storage', {}).get('value', '')
            
            # Simple HTML content extraction (could be enhanced)
            import re
            content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
            content = re.sub(r'\s+', ' ', content).strip()  # Normalize whitespace
            
            # Generate metadata
            metadata_dict = await self.llm_client.generate_metadata(
                title=title,
                content_sample=content[:1000],
                doc_type='confluence_page'
            )
            
            metadata = DocumentMetadata(
                doc_id=doc_id,
                title=title,
                summary=metadata_dict.get('summary', ''),
                topics=metadata_dict.get('topics', []),
                language=metadata_dict.get('language', 'unknown'),
                has_pii=metadata_dict.get('has_pii', False),
                doc_type='confluence_page',
                created_time=datetime.now().isoformat(),
                permission_groups=['user']  # Default permission
            )
            
            # Simple chunking
            chunks = await self._simple_chunk(content, metadata)
            
            return ProcessedDocument(metadata=metadata, chunks=chunks)
            
        except Exception as e:
            logger.error(f"Confluence content processing failed: {e}")
            raise
    
    async def _chunk_with_structure_awareness(
        self,
        sections: List[Dict[str, Any]],
        metadata: DocumentMetadata
    ) -> List[Dict[str, Any]]:
        """Structure-aware chunking using docling sections"""
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_title = section.get('title', '')
            section_content = section.get('content', '')
            
            if not section_content.strip():
                continue
            
            # If section is small enough, keep as single chunk
            if len(section_content) <= self.max_chunk_size:
                chunk_doc = self._create_chunk_document(
                    metadata, section_content, chunk_index, section_title
                )
                chunks.append(chunk_doc)
                chunk_index += 1
            else:
                # Split large sections
                section_chunks = self._split_text(section_content)
                for chunk_text in section_chunks:
                    chunk_doc = self._create_chunk_document(
                        metadata, chunk_text, chunk_index, section_title
                    )
                    chunks.append(chunk_doc)
                    chunk_index += 1
        
        return chunks
    
    async def _simple_chunk(
        self,
        content: str,
        metadata: DocumentMetadata
    ) -> List[Dict[str, Any]]:
        """Simple text chunking"""
        chunks = []
        text_chunks = self._split_text(content)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_doc = self._create_chunk_document(metadata, chunk_text, i)
            chunks.append(chunk_doc)
        
        return chunks
    
    def _create_chunk_document(
        self,
        metadata: DocumentMetadata,
        content: str,
        chunk_index: int,
        section_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create chunk document for insertion"""
        # Serialize metadata to flat string (required by API)
        additional_field = self._serialize_metadata(metadata, section_title)
        
        return {
            'doc_id': f"{metadata.doc_id}#{chunk_index:03d}",
            'title': metadata.title,
            'content': content.strip(),
            'section_title': section_title or '',
            'chunk_index': chunk_index,
            'permission_groups': metadata.permission_groups,
            'created_time': metadata.created_time,
            'additional_field': additional_field
        }
    
    def _serialize_metadata(
        self,
        metadata: DocumentMetadata,
        section_title: Optional[str] = None
    ) -> str:
        """Serialize metadata to flat string format (required by API)"""
        parts = [
            f"summary={metadata.summary or ''}",
            f"topics={','.join(metadata.topics)}",
            f"language={metadata.language}",
            f"pii={str(metadata.has_pii).lower()}",
            f"section={section_title or ''}",
            f"doc_type={metadata.doc_type}"
        ]
        return " | ".join(parts)
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            chunk = text[start:end]
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + self.max_chunk_size // 2:
                end = start + break_point + 1
            
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _serialize_table(self, table) -> str:
        """Serialize table to text representation"""
        # Simplified table serialization
        return f"Table: {getattr(table, 'caption', 'No caption')}"
    
    def _serialize_figure(self, figure) -> str:
        """Serialize figure to text representation"""
        # Simplified figure serialization
        return f"Figure: {getattr(figure, 'caption', 'No caption')}"
    
    def _detect_doc_type(self, filename: str) -> str:
        """Detect document type from filename"""
        ext = Path(filename).suffix.lower()
        type_mapping = {
            '.pdf': 'pdf',
            '.docx': 'document',
            '.doc': 'document',
            '.pptx': 'presentation',
            '.ppt': 'presentation',
            '.xlsx': 'spreadsheet',
            '.xls': 'spreadsheet',
            '.txt': 'text',
            '.md': 'markdown'
        }
        return type_mapping.get(ext, 'unknown')
    
    def _generate_doc_id(self, filename: str) -> str:
        """Generate unique document ID"""
        import hashlib
        timestamp = datetime.now().isoformat()
        content = f"{filename}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF (simplified implementation)"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except ImportError:
            logger.warning("PyPDF2 not available for PDF processing")
            return ""
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return ""
    
    async def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX (simplified implementation)"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            logger.warning("python-docx not available for DOCX processing")
            return ""
        except Exception as e:
            logger.error(f"DOCX text extraction failed: {e}")
            return ""