"""
MCP Server for DocsReview RAG System
Exposes RAG capabilities as MCP tools for Nanobot integration
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    CallToolRequest, CallToolResult, ListToolsRequest, ListToolsResult,
    ListResourcesRequest, ListResourcesResult, ReadResourceRequest, ReadResourceResult
)

# Import our existing RAG components
from rag_system import RAGSystem
from advanced_rag_system import AdvancedRAGSystem
from multimodal_rag_system import MultimodalRAGSystem
from rag_llm_integration import RAGLLMIntegration
from advanced_rag_integration import AdvancedRAGLLMIntegration
from docsreview import workflow, AgentState
from ocr_utils import extract_text_from_pdf, extract_text_from_url
from llm_fallback import get_llm
from database import init_db, save_report, delete_report

logger = logging.getLogger(__name__)

# Initialize database
init_db()

# Initialize LLM with Ollama
llm = get_llm()

@dataclass
class DocumentSession:
    """Manages document processing session state"""
    document_id: str
    document_text: str
    rag_system: Optional[RAGSystem] = None
    advanced_rag_system: Optional[AdvancedRAGSystem] = None
    multimodal_rag_system: Optional[MultimodalRAGSystem] = None
    rag_integration: Optional[RAGLLMIntegration] = None
    advanced_rag_integration: Optional[AdvancedRAGLLMIntegration] = None
    chat_history: List[Dict] = None
    
    def __post_init__(self):
        if self.chat_history is None:
            self.chat_history = []

# Global session storage
sessions: Dict[str, DocumentSession] = {}

# Initialize MCP server
server = Server("docsreview-rag")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available RAG tools"""
    return [
        Tool(
            name="extract_document_text",
            description="Extract text from PDF file or URL with OCR fallback",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["pdf", "url"],
                        "description": "Type of source: 'pdf' for file upload, 'url' for web content"
                    },
                    "source": {
                        "type": "string",
                        "description": "File path (for PDF) or URL (for web content)"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["es", "en"],
                        "default": "es",
                        "description": "Language for processing"
                    }
                },
                "required": ["source_type", "source"]
            }
        ),
        Tool(
            name="process_document_rag",
            description="Process document with RAG system (basic or advanced)",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Unique identifier for the document session"
                    },
                    "document_text": {
                        "type": "string",
                        "description": "Text content of the document"
                    },
                    "rag_type": {
                        "type": "string",
                        "enum": ["basic", "advanced", "multimodal"],
                        "default": "advanced",
                        "description": "Type of RAG system to use"
                    },
                    "chunk_size": {
                        "type": "integer",
                        "default": 800,
                        "description": "Size of document chunks"
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "default": 120,
                        "description": "Overlap between chunks"
                    }
                },
                "required": ["document_id", "document_text"]
            }
        ),
        Tool(
            name="generate_executive_summary",
            description="Generate executive summary using LangGraph workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document session ID"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search keywords or focus area for summary"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["es", "en"],
                        "default": "es",
                        "description": "Language for summary generation"
                    }
                },
                "required": ["document_id", "query"]
            }
        ),
        Tool(
            name="query_document",
            description="Query document using RAG system with chat context",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document session ID"
                    },
                    "query": {
                        "type": "string",
                        "description": "User question about the document"
                    },
                    "use_advanced": {
                        "type": "boolean",
                        "default": True,
                        "description": "Use advanced RAG with ReAct agent"
                    },
                    "conversation_context": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"}
                            }
                        },
                        "description": "Previous conversation context"
                    }
                },
                "required": ["document_id", "query"]
            }
        ),
        Tool(
            name="get_document_stats",
            description="Get statistics about processed document",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document session ID"
                    }
                },
                "required": ["document_id"]
            }
        ),
        Tool(
            name="list_document_sessions",
            description="List all active document processing sessions",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="clear_document_session",
            description="Clear a document processing session",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document session ID to clear"
                    }
                },
                "required": ["document_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    try:
        if name == "extract_document_text":
            return await handle_extract_document_text(arguments)
        elif name == "process_document_rag":
            return await handle_process_document_rag(arguments)
        elif name == "generate_executive_summary":
            return await handle_generate_executive_summary(arguments)
        elif name == "query_document":
            return await handle_query_document(arguments)
        elif name == "get_document_stats":
            return await handle_get_document_stats(arguments)
        elif name == "list_document_sessions":
            return await handle_list_document_sessions(arguments)
        elif name == "clear_document_session":
            return await handle_clear_document_session(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def handle_extract_document_text(arguments: Dict[str, Any]) -> List[TextContent]:
    """Extract text from PDF or URL"""
    source_type = arguments["source_type"]
    source = arguments["source"]
    language = arguments.get("language", "es")
    
    if source_type == "pdf":
        result = extract_text_from_pdf(source)
    elif source_type == "url":
        result = extract_text_from_url(source)
    else:
        return [TextContent(type="text", text="Error: Invalid source_type. Use 'pdf' or 'url'")]
    
    if "error" in result:
        return [TextContent(type="text", text=f"Error: {result['error']}")]
    
    # Create document session
    import uuid
    document_id = str(uuid.uuid4())
    sessions[document_id] = DocumentSession(
        document_id=document_id,
        document_text=result["doc_text"]
    )
    
    return [TextContent(
        type="text", 
        text=json.dumps({
            "document_id": document_id,
            "text_length": len(result["doc_text"]),
            "description": result.get("description", ""),
            "extracted_text_preview": result["doc_text"][:500] + "..." if len(result["doc_text"]) > 500 else result["doc_text"]
        }, indent=2)
    )]

async def handle_process_document_rag(arguments: Dict[str, Any]) -> List[TextContent]:
    """Process document with RAG system"""
    document_id = arguments["document_id"]
    document_text = arguments["document_text"]
    rag_type = arguments.get("rag_type", "advanced")
    chunk_size = arguments.get("chunk_size", 800)
    chunk_overlap = arguments.get("chunk_overlap", 120)
    
    if document_id not in sessions:
        return [TextContent(type="text", text="Error: Document session not found")]
    
    session = sessions[document_id]
    session.document_text = document_text
    
    try:
        if rag_type == "basic":
            session.rag_system = RAGSystem()
            session.rag_system.process_document(document_text, chunk_size=chunk_size, overlap=chunk_overlap)
            session.rag_integration = RAGLLMIntegration(llm, session.rag_system)
        elif rag_type == "advanced":
            session.advanced_rag_system = AdvancedRAGSystem()
            session.advanced_rag_system.process_document(document_text, chunk_size=chunk_size, overlap=chunk_overlap)
            session.advanced_rag_integration = AdvancedRAGLLMIntegration(llm, session.advanced_rag_system)
        elif rag_type == "multimodal":
            session.multimodal_rag_system = MultimodalRAGSystem()
            # For multimodal, we'd need file path, but for now use text content
            session.multimodal_rag_system._process_text_content(document_text)
        
        stats = session.rag_system.get_document_stats() if session.rag_system else session.advanced_rag_system.get_document_stats()
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "rag_type": rag_type,
                "document_stats": stats,
                "message": f"Document processed with {rag_type} RAG system"
            }, indent=2)
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error processing document: {str(e)}")]

async def handle_generate_executive_summary(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate executive summary using LangGraph workflow"""
    document_id = arguments["document_id"]
    query = arguments["query"]
    language = arguments.get("language", "es")
    
    if document_id not in sessions:
        return [TextContent(type="text", text="Error: Document session not found")]
    
    session = sessions[document_id]
    
    try:
        # Prepare initial state for LangGraph workflow
        initial_state = AgentState(
            query=query,
            source_type="upload",
            confirmed_source=f"Document {document_id}",
            language=language,
            references=[],
            file_path=None,
            doc_url=None,
            doc_text=session.document_text,
            summaries=[],
        )
        
        # Invoke workflow
        app = workflow.compile()
        result = app.invoke(initial_state, config={"recursion_limit": 50})
        
        # Save to database
        import uuid
        report_id = str(uuid.uuid4())
        save_report(
            report_id, query, "upload", f"Document {document_id}", 
            language, result['final_summary'], result.get('references', [])
        )
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "report_id": report_id,
                "executive_summary": result['final_summary'],
                "references": result.get('references', []),
                "language": language,
                "query": query
            }, indent=2)
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating summary: {str(e)}")]

async def handle_query_document(arguments: Dict[str, Any]) -> List[TextContent]:
    """Query document using RAG system"""
    document_id = arguments["document_id"]
    query = arguments["query"]
    use_advanced = arguments.get("use_advanced", True)
    conversation_context = arguments.get("conversation_context", [])
    
    if document_id not in sessions:
        return [TextContent(type="text", text="Error: Document session not found")]
    
    session = sessions[document_id]
    
    try:
        if use_advanced and session.advanced_rag_integration:
            # Use advanced RAG with ReAct agent
            response = session.advanced_rag_integration.generate_response_with_agent(query, conversation_context)
        elif session.rag_integration:
            # Use basic RAG
            response = session.rag_integration.generate_response(query, conversation_context)
        else:
            return [TextContent(type="text", text="Error: RAG system not initialized for this document")]
        
        # Update conversation history
        session.chat_history.append({"role": "user", "content": query})
        session.chat_history.append({"role": "assistant", "content": response.answer})
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "answer": response.answer,
                "confidence_score": response.confidence_score,
                "retrieval_info": response.retrieval_info,
                "relevant_chunks": response.relevant_chunks[:3],  # Limit for display
                "conversation_length": len(session.chat_history)
            }, indent=2)
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error querying document: {str(e)}")]

async def handle_get_document_stats(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get document statistics"""
    document_id = arguments["document_id"]
    
    if document_id not in sessions:
        return [TextContent(type="text", text="Error: Document session not found")]
    
    session = sessions[document_id]
    
    stats = {}
    if session.rag_system:
        stats["basic_rag"] = session.rag_system.get_document_stats()
    if session.advanced_rag_system:
        stats["advanced_rag"] = session.advanced_rag_system.get_document_stats()
    if session.multimodal_rag_system:
        stats["multimodal_rag"] = session.multimodal_rag_system.get_document_stats()
    
    stats["document_length"] = len(session.document_text)
    stats["chat_history_length"] = len(session.chat_history)
    
    return [TextContent(
        type="text",
        text=json.dumps(stats, indent=2)
    )]

async def handle_list_document_sessions(arguments: Dict[str, Any]) -> List[TextContent]:
    """List all document sessions"""
    session_list = []
    for doc_id, session in sessions.items():
        session_list.append({
            "document_id": doc_id,
            "document_length": len(session.document_text),
            "has_rag_system": session.rag_system is not None,
            "has_advanced_rag": session.advanced_rag_system is not None,
            "has_multimodal_rag": session.multimodal_rag_system is not None,
            "chat_history_length": len(session.chat_history)
        })
    
    return [TextContent(
        type="text",
        text=json.dumps({"sessions": session_list}, indent=2)
    )]

async def handle_clear_document_session(arguments: Dict[str, Any]) -> List[TextContent]:
    """Clear a document session"""
    document_id = arguments["document_id"]
    
    if document_id in sessions:
        del sessions[document_id]
        return [TextContent(type="text", text=f"Session {document_id} cleared successfully")]
    else:
        return [TextContent(type="text", text="Error: Session not found")]

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="docsreview-rag",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())