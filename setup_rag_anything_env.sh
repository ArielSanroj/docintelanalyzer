#!/bin/bash
"""
Setup script for RAG-Anything environment with DocsReview RAG system
"""

set -e  # Exit on any error

echo "🚀 Setting up RAG-Anything Environment for DocsReview RAG System"
echo "=================================================================="

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.13"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is compatible (requires 3.13+)"
else
    echo "❌ Python $python_version is not compatible (requires 3.13+)"
    echo "Please install Python 3.13 or higher"
    exit 1
fi

# Create virtual environment
echo ""
echo "2. Creating virtual environment..."
if [ -d "venv_rag_anything" ]; then
    echo "   Virtual environment already exists. Removing old one..."
    rm -rf venv_rag_anything
fi

python3 -m venv venv_rag_anything
echo "✅ Virtual environment created: venv_rag_anything"

# Activate virtual environment
echo ""
echo "3. Activating virtual environment..."
source venv_rag_anything/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "4. Upgrading pip..."
pip install --upgrade pip
echo "✅ Pip upgraded"

# Install RAG-Anything first (has complex dependencies)
echo ""
echo "5. Installing RAG-Anything..."
pip install raganything
echo "✅ RAG-Anything installed"

# Install core dependencies
echo ""
echo "6. Installing core dependencies..."
pip install langchain langchain-community langchain-core langgraph
echo "✅ Core LangChain dependencies installed"

# Install embeddings and vector stores
echo ""
echo "7. Installing embeddings and vector stores..."
pip install sentence-transformers faiss-cpu
echo "✅ Embeddings and vector stores installed"

# Install LLM integrations
echo ""
echo "8. Installing LLM integrations..."
pip install ollama openai anthropic
echo "✅ LLM integrations installed"

# Install document processing
echo ""
echo "9. Installing document processing..."
pip install PyMuPDF pytesseract Pillow beautifulsoup4 lxml requests
echo "✅ Document processing installed"

# Install data processing
echo ""
echo "10. Installing data processing..."
pip install numpy pandas scikit-learn scipy
echo "✅ Data processing installed"

# Install multimodal processing
echo ""
echo "11. Installing multimodal processing..."
pip install opencv-python matplotlib seaborn
echo "✅ Multimodal processing installed"

# Install web framework
echo ""
echo "12. Installing web framework..."
pip install streamlit fastapi uvicorn
echo "✅ Web framework installed"

# Install MCP integration
echo ""
echo "13. Installing MCP integration..."
pip install mcp mcp-server mcp-server-stdio
echo "✅ MCP integration installed"

# Install utilities
echo ""
echo "14. Installing utilities..."
pip install python-dotenv pydantic typing-extensions loguru
echo "✅ Utilities installed"

# Install additional dependencies
echo ""
echo "15. Installing additional dependencies..."
pip install rank-bm25 nltk spacy
echo "✅ Additional dependencies installed"

# Test installation
echo ""
echo "16. Testing installation..."
python -c "
import raganything
import langchain
import sentence_transformers
import ollama
import streamlit
print('✅ All core packages imported successfully')
"

# Create activation script
echo ""
echo "17. Creating activation script..."
cat > activate_rag_anything.sh << 'EOF'
#!/bin/bash
echo "🚀 Activating RAG-Anything Environment"
echo "======================================"
source venv_rag_anything/bin/activate
echo "✅ Environment activated!"
echo ""
echo "Available commands:"
echo "  python test_simple_rag_anything.py  # Test basic functionality"
echo "  python test_rag_anything_complete.py  # Test complete integration"
echo "  streamlit run app.py  # Run Streamlit app"
echo "  python mcp_server.py  # Run MCP server"
echo ""
EOF

chmod +x activate_rag_anything.sh
echo "✅ Activation script created: activate_rag_anything.sh"

# Create test script
echo ""
echo "18. Creating test script..."
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify RAG-Anything installation
"""

def test_imports():
    """Test all critical imports"""
    try:
        import raganything
        print("✅ RAG-Anything imported successfully")
    except ImportError as e:
        print(f"❌ RAG-Anything import failed: {e}")
        return False
    
    try:
        import langchain
        print("✅ LangChain imported successfully")
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import ollama
        print("✅ Ollama imported successfully")
    except ImportError as e:
        print(f"❌ Ollama import failed: {e}")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    return True

def test_rag_anything():
    """Test RAG-Anything basic functionality"""
    try:
        from raganything import RAGAnything
        rag = RAGAnything()
        print("✅ RAG-Anything initialized successfully")
        return True
    except Exception as e:
        print(f"❌ RAG-Anything initialization failed: {e}")
        return False

def main():
    print("🧪 Testing RAG-Anything Installation")
    print("=" * 40)
    
    success1 = test_imports()
    success2 = test_rag_anything()
    
    if success1 and success2:
        print("\n🎉 All tests passed! RAG-Anything is ready to use.")
        return True
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
EOF

chmod +x test_installation.py
echo "✅ Test script created: test_installation.py"

# Run test
echo ""
echo "19. Running installation test..."
python test_installation.py

echo ""
echo "🎉 RAG-Anything Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source activate_rag_anything.sh"
echo "2. Test installation: python test_installation.py"
echo "3. Run tests: python test_simple_rag_anything.py"
echo "4. Start development: streamlit run app.py"
echo ""
echo "Environment location: $(pwd)/venv_rag_anything"
echo "Activation script: $(pwd)/activate_rag_anything.sh"
echo ""