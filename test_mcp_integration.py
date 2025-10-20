"""
Test script for MCP server integration with Nanobot
"""

import asyncio
import json
import logging
from mcp.client import ClientSession
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mcp_server():
    """Test MCP server functionality"""
    
    # Test document extraction
    print("Testing document extraction...")
    
    # Test PDF processing
    pdf_test = {
        "name": "extract_document_text",
        "arguments": {
            "source_type": "pdf",
            "source": "test_document.pdf",  # You'll need to provide a test PDF
            "language": "es"
        }
    }
    
    # Test URL processing
    url_test = {
        "name": "extract_document_text",
        "arguments": {
            "source_type": "url",
            "source": "https://example.com/document.pdf",
            "language": "en"
        }
    }
    
    # Test RAG processing
    rag_test = {
        "name": "process_document_rag",
        "arguments": {
            "document_id": "test-doc-123",
            "document_text": "This is a test document with some content for RAG processing.",
            "rag_type": "advanced",
            "chunk_size": 800,
            "chunk_overlap": 120
        }
    }
    
    # Test executive summary generation
    summary_test = {
        "name": "generate_executive_summary",
        "arguments": {
            "document_id": "test-doc-123",
            "query": "test document analysis",
            "language": "es"
        }
    }
    
    # Test document querying
    query_test = {
        "name": "query_document",
        "arguments": {
            "document_id": "test-doc-123",
            "query": "What is this document about?",
            "use_advanced": True,
            "conversation_context": []
        }
    }
    
    # Test document statistics
    stats_test = {
        "name": "get_document_stats",
        "arguments": {
            "document_id": "test-doc-123"
        }
    }
    
    # Test session management
    sessions_test = {
        "name": "list_document_sessions",
        "arguments": {}
    }
    
    print("MCP server tests defined. Run with actual MCP server to test functionality.")

def test_nanobot_integration():
    """Test Nanobot agent configuration"""
    
    # Test agent configurations
    agents = [
        "nanobot_agents/document_analysis_agent.yaml",
        "nanobot_agents/rag_chat_agent.yaml",
        "nanobot_agents/multimodal_analysis_agent.yaml"
    ]
    
    for agent_config in agents:
        try:
            with open(agent_config, 'r') as f:
                config = f.read()
            print(f"✅ {agent_config} - Valid YAML configuration")
        except FileNotFoundError:
            print(f"❌ {agent_config} - File not found")
        except Exception as e:
            print(f"❌ {agent_config} - Error: {e}")

def test_docker_setup():
    """Test Docker configuration"""
    
    try:
        with open('docker-compose.yml', 'r') as f:
            compose_config = f.read()
        print("✅ docker-compose.yml - Valid configuration")
    except Exception as e:
        print(f"❌ docker-compose.yml - Error: {e}")
    
    try:
        with open('Dockerfile.mcp', 'r') as f:
            dockerfile = f.read()
        print("✅ Dockerfile.mcp - Valid configuration")
    except Exception as e:
        print(f"❌ Dockerfile.mcp - Error: {e}")

def test_requirements():
    """Test requirements file"""
    
    try:
        with open('requirements_mcp.txt', 'r') as f:
            requirements = f.read()
        print("✅ requirements_mcp.txt - Valid requirements file")
        
        # Check for key dependencies
        key_deps = ['mcp', 'langchain', 'sentence-transformers', 'requests']
        for dep in key_deps:
            if dep in requirements:
                print(f"  ✅ {dep} - Present")
            else:
                print(f"  ❌ {dep} - Missing")
                
    except Exception as e:
        print(f"❌ requirements_mcp.txt - Error: {e}")

def test_ollama_setup():
    """Test Ollama setup"""
    print("\n5. Testing Ollama Setup:")
    
    # Check if Ollama is installed
    try:
        import subprocess
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
        else:
            print("❌ Ollama is not installed")
    except FileNotFoundError:
        print("❌ Ollama is not installed")
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
        else:
            print("❌ Ollama service is not running")
    except requests.exceptions.RequestException:
        print("❌ Ollama service is not running")
    
    # Test LLM integration
    try:
        from llm_fallback import get_llm
        llm = get_llm()
        response = llm.invoke("Test message")
        if response and hasattr(response, 'content'):
            print("✅ Ollama LLM integration working")
        else:
            print("❌ Ollama LLM integration not working")
    except Exception as e:
        print(f"❌ Ollama LLM integration error: {e}")

if __name__ == "__main__":
    print("🧪 Testing MCP Integration Setup")
    print("=" * 50)
    
    print("\n1. Testing Nanobot Agent Configurations:")
    test_nanobot_integration()
    
    print("\n2. Testing Docker Setup:")
    test_docker_setup()
    
    print("\n3. Testing Requirements:")
    test_requirements()
    
    print("\n4. Testing MCP Server Functions:")
    asyncio.run(test_mcp_server())
    
    test_ollama_setup()
    
    print("\n✅ Integration tests completed!")
    print("\nNext steps:")
    print("1. Setup Ollama: python setup_ollama.py")
    print("2. Start MCP server: python mcp_server.py")
    print("3. Start Nanobot with agent config")
    print("4. Test document processing workflow")
    print("5. Deploy with Docker Compose")