"""
RAG-Anything integration for DocsReview RAG system
Provides multimodal document processing capabilities
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)

try:
    from raganything import RAGAnything
    RAG_ANYTHING_AVAILABLE = True
except ImportError:
    RAG_ANYTHING_AVAILABLE = False
    logger.warning("RAG-Anything not available. Using fallback.")

class RAGAnythingIntegration:
    """Integration class for RAG-Anything multimodal processing"""
    
    def __init__(self, working_dir: str = "./rag_storage"):
        """
        Initialize RAG-Anything integration
        
        Args:
            working_dir: Directory for RAG-Anything storage
        """
        self.working_dir = working_dir
        self.rag_anything = None
        self.available = RAG_ANYTHING_AVAILABLE
        
        if self.available:
            try:
                # Initialize RAG-Anything with correct parameters
                self.rag_anything = RAGAnything()
                logger.info("RAG-Anything initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RAG-Anything: {e}")
                self.available = False
        else:
            logger.warning("RAG-Anything not available")
    
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a document with RAG-Anything
        
        Args:
            document_path: Path to the document to process
            
        Returns:
            Dictionary with processing results
        """
        if not self.available or not self.rag_anything:
            return {"error": "RAG-Anything not available"}
        
        try:
            # Process document with RAG-Anything
            result = self.rag_anything.process_document(document_path)
            logger.info(f"Document processed successfully: {document_path}")
            return result
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {e}")
            return {"error": str(e)}
    
    def query_document(self, query: str, document_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Query processed documents
        
        Args:
            query: Query string
            document_path: Optional specific document to query
            
        Returns:
            Dictionary with query results
        """
        if not self.available or not self.rag_anything:
            return {"error": "RAG-Anything not available"}
        
        try:
            # Query documents
            result = self.rag_anything.query(query, document_path)
            logger.info(f"Query processed successfully: {query[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return {"error": str(e)}
    
    def get_document_info(self, document_path: str) -> Dict[str, Any]:
        """
        Get information about a processed document
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dictionary with document information
        """
        if not self.available or not self.rag_anything:
            return {"error": "RAG-Anything not available"}
        
        try:
            # Get document information
            info = self.rag_anything.get_document_info(document_path)
            return info
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return {"error": str(e)}
    
    def list_processed_documents(self) -> List[str]:
        """
        List all processed documents
        
        Returns:
            List of processed document paths
        """
        if not self.available or not self.rag_anything:
            return []
        
        try:
            # List processed documents
            documents = self.rag_anything.list_documents()
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def clear_document(self, document_path: str) -> bool:
        """
        Clear a specific document from processing
        
        Args:
            document_path: Path to the document to clear
            
        Returns:
            True if successful, False otherwise
        """
        if not self.available or not self.rag_anything:
            return False
        
        try:
            # Clear document
            self.rag_anything.clear_document(document_path)
            logger.info(f"Document cleared: {document_path}")
            return True
        except Exception as e:
            logger.error(f"Error clearing document: {e}")
            return False
    
    def get_multimodal_content(self, document_path: str) -> Dict[str, Any]:
        """
        Get multimodal content (text, images, tables) from a document
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dictionary with multimodal content
        """
        if not self.available or not self.rag_anything:
            return {"error": "RAG-Anything not available"}
        
        try:
            # Get multimodal content
            content = self.rag_anything.get_multimodal_content(document_path)
            return content
        except Exception as e:
            logger.error(f"Error getting multimodal content: {e}")
            return {"error": str(e)}
    
    def is_available(self) -> bool:
        """Check if RAG-Anything is available"""
        return self.available and self.rag_anything is not None

# Global instance
rag_anything_integration = RAGAnythingIntegration()

def get_rag_anything() -> RAGAnythingIntegration:
    """Get the global RAG-Anything integration instance"""
    return rag_anything_integration