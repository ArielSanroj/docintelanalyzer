# RAG-Anything Integration with DocsReview RAG System

This document describes the integration of RAG-Anything with the DocsReview RAG system, providing advanced multimodal document processing capabilities.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run the automated setup script
./setup_rag_anything_env.sh

# Or activate existing environment
source activate_rag_anything.sh
```

### 2. Test Installation

```bash
# Test basic functionality
python test_simple_rag_anything.py

# Test complete integration
python test_rag_anything_complete.py
```

### 3. Run the Application

```bash
# Run Streamlit app
streamlit run app.py

# Run MCP server
python mcp_server.py
```

## 📋 Requirements

- **Python**: 3.13+ (required for RAG-Anything)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ free space for models and dependencies

## 🔧 Installation

### Automated Installation

The `setup_rag_anything_env.sh` script handles everything:

```bash
chmod +x setup_rag_anything_env.sh
./setup_rag_anything_env.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv_rag_anything
source venv_rag_anything/bin/activate

# Install RAG-Anything
pip install raganything

# Install additional dependencies
pip install -r requirements_rag_anything.txt
```

## 🏗️ Architecture

### Components

1. **RAG-Anything Core**: Multimodal document processing
2. **DocsReview RAG System**: Document chunking and retrieval
3. **Ollama LLM**: Local language model
4. **MCP Server**: Model Context Protocol integration
5. **Streamlit UI**: Web interface

### Integration Flow

```
Document → RAG-Anything → DocsReview RAG → Ollama LLM → Response
    ↓           ↓              ↓            ↓
Multimodal → Chunking → Embeddings → Generation
Processing   (Text/Images)  (Vector)    (Local)
```

## 🧪 Testing

### Basic Tests

```bash
# Test RAG-Anything functionality
python test_simple_rag_anything.py

# Test complete integration
python test_rag_anything_complete.py

# Test VALIO report processing
python test_valio_report.py
```

### Test Coverage

- ✅ RAG-Anything initialization
- ✅ Document processing
- ✅ Multimodal content extraction
- ✅ Query processing
- ✅ Ollama LLM integration
- ✅ DocsReview RAG integration
- ✅ MCP server functionality

## 📊 Features

### RAG-Anything Capabilities

- **Multimodal Processing**: Text, images, tables, equations
- **Document Parsing**: PDF, HTML, Markdown support
- **Advanced Chunking**: Adaptive text segmentation
- **Vector Storage**: Efficient similarity search
- **Query Processing**: Natural language queries

### DocsReview Integration

- **Hybrid Search**: BM25 + semantic similarity
- **Adaptive Chunking**: Content-aware segmentation
- **Local Embeddings**: Sentence Transformers
- **Fallback System**: Graceful degradation
- **Session Management**: Document state tracking

### Ollama LLM Features

- **Local Processing**: No API costs
- **Privacy**: Data stays local
- **Customizable**: Model parameters
- **Fast**: No network latency
- **Offline**: No internet required

## 🔧 Configuration

### Environment Variables

```bash
# Ollama API key
export OLLAMA_API_KEY="your_ollama_api_key"

# RAG-Anything working directory
export RAG_ANYTHING_WORK_DIR="./rag_storage"

# Optional: Custom model paths
export SENTENCE_TRANSFORMERS_HOME="./models"
```

### Model Configuration

```python
# RAG-Anything configuration
rag_anything = RAGAnything(
    working_dir="./rag_storage",
    parser="mineru",
    parse_method="auto"
)

# DocsReview RAG configuration
rag_system = RAGSystem(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=1500,
    chunk_overlap=300
)

# Ollama LLM configuration
llm = OllamaLLM(
    model="llama3.1:8b",
    base_url="http://localhost:11434"
)
```

## 📁 File Structure

```
docsreview/
├── venv_rag_anything/          # Virtual environment
├── rag_storage/                # RAG-Anything storage
├── requirements_rag_anything.txt
├── setup_rag_anything_env.sh
├── activate_rag_anything.sh
├── test_simple_rag_anything.py
├── test_rag_anything_complete.py
├── rag_anything_integration.py
├── rag_llm_integration.py      # Updated with RAG-Anything
├── mcp_server.py              # MCP server
├── app.py                     # Streamlit app
└── RAG_ANYTHING_README.md     # This file
```

## 🚀 Usage Examples

### Basic Document Processing

```python
from rag_anything_integration import get_rag_anything

# Get RAG-Anything instance
rag_anything = get_rag_anything()

# Process document
result = rag_anything.process_document("document.pdf")

# Query document
response = rag_anything.query_document("What is the main topic?")
```

### DocsReview Integration

```python
from rag_llm_integration import RAGLLMIntegration
from rag_system import RAGSystem
from llm_fallback import get_llm

# Initialize components
llm = get_llm()
rag_system = RAGSystem()
rag_integration = RAGLLMIntegration(llm, rag_system, use_rag_anything=True)

# Generate response
response = rag_integration.generate_response(
    "What are the key findings?",
    document_path="document.pdf"
)
```

### MCP Server Usage

```python
# Start MCP server
python mcp_server.py

# Available tools:
# - extract_document_text
# - process_document_rag
# - generate_executive_summary
# - query_document
# - get_document_stats
```

## 🔍 Troubleshooting

### Common Issues

1. **RAG-Anything not available**
   ```bash
   # Check Python version
   python --version  # Should be 3.13+
   
   # Reinstall RAG-Anything
   pip install --upgrade raganything
   ```

2. **Ollama not running**
   ```bash
   # Start Ollama service
   ollama serve
   
   # Pull model
   ollama pull llama3.1:8b
   ```

3. **Memory issues**
   ```bash
   # Reduce batch size in sentence_transformers
   export SENTENCE_TRANSFORMERS_BATCH_SIZE=16
   ```

4. **Import errors**
   ```bash
   # Activate virtual environment
   source venv_rag_anything/bin/activate
   
   # Reinstall dependencies
   pip install -r requirements_rag_anything.txt
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python test_simple_rag_anything.py
```

## 📈 Performance

### Benchmarks

- **Document Processing**: ~2-5 seconds per page
- **Query Response**: ~1-3 seconds
- **Memory Usage**: ~2-4GB for full setup
- **Storage**: ~1-2GB for models

### Optimization Tips

1. **Use GPU**: Install CUDA for faster processing
2. **Batch Processing**: Process multiple documents together
3. **Model Caching**: Keep models in memory
4. **Chunk Optimization**: Tune chunk size for your documents

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd docsreview

# Setup development environment
./setup_rag_anything_env.sh
source activate_rag_anything.sh

# Install development dependencies
pip install pytest black flake8 mypy
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
python test_simple_rag_anything.py

# Run with coverage
pytest --cov=.
```

## 📚 Documentation

- [RAG-Anything Documentation](https://github.com/HKUDS/RAGAnything)
- [DocsReview RAG System](README.md)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [Ollama Documentation](https://ollama.ai/docs)

## 🆘 Support

### Getting Help

1. Check the troubleshooting section above
2. Review the test outputs for error messages
3. Check the logs in `rag_storage/` directory
4. Open an issue with detailed error information

### Reporting Issues

When reporting issues, please include:

1. Python version (`python --version`)
2. Operating system
3. Error messages and stack traces
4. Steps to reproduce
5. Test output (`python test_simple_rag_anything.py`)

## 🎯 Next Steps

### Planned Features

1. **Advanced Multimodal Processing**: Better image and table understanding
2. **Custom Model Support**: Integration with other LLMs
3. **Batch Processing**: Process multiple documents efficiently
4. **API Endpoints**: REST API for integration
5. **Cloud Deployment**: Docker and Kubernetes support

### Roadmap

- **Phase 1**: Basic RAG-Anything integration ✅
- **Phase 2**: Advanced multimodal processing
- **Phase 3**: Performance optimization
- **Phase 4**: Cloud deployment
- **Phase 5**: Enterprise features

---

**Happy RAG-ing! 🚀**