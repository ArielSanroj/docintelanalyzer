#!/bin/bash

# Script para ver logs del sistema RAG en tiempo real

echo "🔍 Monitor de Logs del Sistema RAG"
echo "=================================="

# Verificar si Streamlit está ejecutándose
if pgrep -f "streamlit run app.py" > /dev/null; then
    echo "✅ Streamlit está ejecutándose"
    
    # Mostrar información del proceso
    echo "📊 Proceso de Streamlit:"
    ps aux | grep "streamlit run app.py" | grep -v grep
    
    echo ""
    echo "📝 Para ver logs en tiempo real:"
    echo "1. Abre otra terminal"
    echo "2. Ejecuta: streamlit run app.py (sin --server.headless)"
    echo "3. O ejecuta: tail -f ~/.streamlit/logs/streamlit.log"
    
else
    echo "❌ Streamlit no está ejecutándose"
    echo "💡 Ejecuta: source .venv/bin/activate && streamlit run app.py"
fi

echo ""
echo "🔧 Información del sistema:"
echo "Python: $(python --version 2>&1)"
echo "Streamlit: $(python -c 'import streamlit; print(streamlit.__version__)' 2>/dev/null || echo 'No disponible')"
echo "LangChain: $(python -c 'import langchain; print(langchain.__version__)' 2>/dev/null || echo 'No disponible')"
echo "ChromaDB: $(python -c 'import chromadb; print(chromadb.__version__)' 2>/dev/null || echo 'No disponible')"

echo ""
echo "📋 Comandos útiles:"
echo "- Ver logs: streamlit run app.py"
echo "- Reiniciar: pkill -f streamlit && source .venv/bin/activate && streamlit run app.py"
echo "- Debug: python -c 'import app; print(\"App importado correctamente\")'"