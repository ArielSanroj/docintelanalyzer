#!/usr/bin/env python3
"""
Test script for Ollama integration
"""

import os
import sys
from llm_fallback import get_llm

def test_ollama_connection():
    """Test Ollama connection and response"""
    print("🧪 Testing Ollama Integration")
    print("=" * 40)
    
    try:
        # Initialize LLM
        print("1. Initializing Ollama LLM...")
        llm = get_llm()
        print("✅ LLM initialized successfully")
        
        # Test simple response
        print("\n2. Testing simple response...")
        response = llm.invoke("Hello, this is a test. Please respond with 'Ollama test successful'.")
        print(f"Response: {response.content}")
        
        # Test document analysis
        print("\n3. Testing document analysis...")
        doc_prompt = """
        Generate a brief executive summary of this document:
        
        This is a test document about artificial intelligence and machine learning.
        It covers topics such as neural networks, deep learning, and natural language processing.
        The document discusses various applications in healthcare, finance, and technology.
        """
        
        response = llm.invoke(doc_prompt)
        print(f"Document Analysis Response: {response.content}")
        
        # Test RAG-style response
        print("\n4. Testing RAG-style response...")
        rag_prompt = """
        Based on the following document content, answer the question:
        
        Document: "The company reported a 15% increase in revenue for Q3 2024. 
        This growth was driven by strong performance in the technology sector and 
        expansion into new markets. The CEO announced plans for further investment 
        in research and development."
        
        Question: What was the revenue growth for Q3 2024?
        """
        
        response = llm.invoke(rag_prompt)
        print(f"RAG Response: {response.content}")
        
        print("\n🎉 All tests passed! Ollama integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is installed: https://ollama.ai")
        print("2. Start Ollama service: ollama serve")
        print("3. Pull the model: ollama pull llama3.1:8b")
        print("4. Check API key in .env file")
        return False

if __name__ == "__main__":
    success = test_ollama_connection()
    sys.exit(0 if success else 1)