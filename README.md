# DocsReview RAG LLM Project

A sophisticated document analysis application with RAG (Retrieval-Augmented Generation) capabilities.

## Features

- **Document Analysis**: Upload PDFs or provide URLs for automatic text extraction
- **Executive Summary Generation**: AI-powered executive summaries using NVIDIA's Llama models
- **RAG System**: Advanced retrieval-augmented generation for document Q&A
- **Chat Interface**: Interactive chat with documents using RAG
- **Multi-language Support**: Spanish and English language support
- **OCR Fallback**: Automatic OCR processing for image-based PDFs


## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Ollama**:
   ```bash
   # Install and setup Ollama
   python setup_ollama.py
   
   # Or manually:
   # Install Ollama: https://ollama.ai
   # Pull model: ollama pull llama3.1:8b
   # Set API key: export OLLAMA_API_KEY="c6f1e109560b4b098ff80b99c5942d42.DdN4aonYSge8plew0dvp3XO_"
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```


## Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Python with LangChain and LangGraph
- **AI Models**: Ollama Llama 3.1 8B (Local)
- **RAG System**: Sentence transformers with hybrid search
- **Database**: SQLite for local storage

## Components

- `app.py` - Main Streamlit application
- `docsreview.py` - Document analysis workflow
- `rag_system.py` - RAG implementation
- `rag_llm_integration.py` - RAG + LLM integration
- `database.py` - Local database management
- `ocr_utils.py` - OCR and text extraction utilities

## Testing

Run the application and test document analysis:
```bash
streamlit run app.py
```

## Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment instructions
- [STREAMLIT_DEPLOYMENT_GUIDE.md](STREAMLIT_DEPLOYMENT_GUIDE.md) - Streamlit deployment guide
