#!/usr/bin/env python3
"""
Test script for LLMHelper integration with DocsReview RAG system.
Demonstrates the enhanced LLM capabilities and provider switching.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_helper import create_llm_helper, LLMConfig
from rag_system import RAGSystem
from rag_llm_integration import RAGLLMIntegration

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm_helper_basic():
    """Test basic LLMHelper functionality"""
    print("=" * 60)
    print("Testing LLMHelper Basic Functionality")
    print("=" * 60)
    
    # Create LLMHelper
    helper = create_llm_helper()
    
    # Test provider info
    info = helper.get_provider_info()
    print(f"Provider: {info['provider']}")
    print(f"Model: {info['model']}")
    print(f"Available providers: {info['available_providers']}")
    
    # Test basic text generation
    test_text = "Microsoft hired Sarah Johnson as Chief Technology Officer on January 15, 2024 in Seattle, Washington for $500,000 annually."
    
    print("\n1. Testing basic text generation:")
    simple_response = helper.invoke_llm("Hello, how are you?")
    print(f"Simple response: {simple_response}")
    
    print("\n2. Testing sentiment analysis:")
    sentiment = helper.analyze_sentiment("This is a great product with excellent features!")
    print(f"Sentiment: {sentiment}")
    
    print("\n3. Testing text summarization:")
    summary = helper.summarize_text(test_text, max_length=50)
    print(f"Summary: {summary}")
    
    print("\n4. Testing question answering:")
    context = "The company was founded in 1975 by Bill Gates and Paul Allen. It is headquartered in Redmond, Washington."
    answer = helper.answer_question("When was the company founded?", context)
    print(f"Answer: {answer}")
    
    print("\n5. Testing entity extraction (simplified):")
    try:
        entities = helper.extract_entities(test_text)
        print(f"Entities: {entities}")
    except Exception as e:
        print(f"Entity extraction failed: {e}")

def test_rag_integration():
    """Test RAG integration with LLMHelper"""
    print("\n" + "=" * 60)
    print("Testing RAG Integration with LLMHelper")
    print("=" * 60)
    
    # Sample document text
    doc_text = """
    DECRETO 1072 DE 2015
    Por el cual se expide el Decreto Único Reglamentario del Sector Trabajo.
    
    El Presidente de la República de Colombia, en ejercicio de las facultades constitucionales y legales,
    especialmente las conferidas en el artículo 189 numeral 11 de la Constitución Política,
    y en desarrollo de lo dispuesto en la Ley 1562 de 2012,
    
    DECRETA:
    
    TÍTULO I
    DISPOSICIONES GENERALES
    
    ARTÍCULO 1. Objeto. El presente decreto tiene por objeto compilar, racionalizar y actualizar
    las disposiciones reglamentarias del sector trabajo, con el fin de facilitar su consulta y aplicación.
    
    ARTÍCULO 2. Ámbito de aplicación. Las disposiciones contenidas en el presente decreto
    se aplicarán a todas las entidades del sector trabajo y a los particulares que desarrollen
    actividades relacionadas con el trabajo.
    
    TÍTULO II
    ESTRUCTURA ORGANIZACIONAL
    
    ARTÍCULO 10. Ministerio del Trabajo. El Ministerio del Trabajo es la entidad rectora
    del sector trabajo y tiene como objeto formular, adoptar, dirigir, coordinar y ejecutar
    la política pública en materia de trabajo.
    """
    
    # Create RAG system
    rag_system = RAGSystem()
    rag_system.process_document(doc_text, chunk_size=500, overlap=100)
    
    # Create RAG integration with LLMHelper
    rag_integration = RAGLLMIntegration(
        llm=None,  # Will use LLMHelper
        rag_system=rag_system,
        use_llm_helper=True,
        llm_config={"provider": "auto"}
    )
    
    # Test queries
    queries = [
        "¿Cuál es el objeto del decreto?",
        "¿Qué entidades están bajo el ámbito de aplicación?",
        "¿Cuál es la función del Ministerio del Trabajo?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        # Generate response using LLMHelper
        response = rag_integration.generate_response_with_llm_helper(query)
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Chunks found: {response.retrieval_info['chunks_found']}")

def test_provider_switching():
    """Test provider switching capabilities"""
    print("\n" + "=" * 60)
    print("Testing Provider Switching")
    print("=" * 60)
    
    helper = create_llm_helper()
    
    # Get current provider info
    info = helper.get_provider_info()
    print(f"Current provider: {info['provider']}")
    print(f"Available providers: {info['available_providers']}")
    
    # Test switching (if multiple providers available)
    available_providers = info['available_providers']
    if len(available_providers) > 1:
        for provider in available_providers[:2]:  # Test first 2 providers
            print(f"\nTesting switch to: {provider}")
            success = helper.switch_provider(provider)
            if success:
                new_info = helper.get_provider_info()
                print(f"Successfully switched to: {new_info['provider']}")
                
                # Test basic functionality with new provider
                test_response = helper.invoke_llm("Hello, how are you?")
                print(f"Test response: {test_response[:100]}...")
            else:
                print(f"Failed to switch to: {provider}")

def test_advanced_capabilities():
    """Test advanced LLMHelper capabilities"""
    print("\n" + "=" * 60)
    print("Testing Advanced Capabilities")
    print("=" * 60)
    
    helper = create_llm_helper()
    
    # Test conversation history
    print("1. Testing conversation history:")
    helper.update_conversation_history("What is artificial intelligence?", "AI is a field of computer science...")
    helper.update_conversation_history("How does machine learning work?", "Machine learning uses algorithms...")
    
    summary = helper.get_conversation_summary()
    print(f"Conversation summary: {summary}")
    
    # Test RAG response generation
    print("\n2. Testing RAG response generation:")
    chunks = [
        "The company was founded in 1975 by Bill Gates and Paul Allen.",
        "Microsoft is headquartered in Redmond, Washington.",
        "The company develops software, hardware, and cloud services."
    ]
    
    rag_response = helper.generate_rag_response(
        query="When was Microsoft founded?",
        retrieved_chunks=chunks
    )
    print(f"RAG Response: {rag_response}")

def main():
    """Run all tests"""
    print("LLMHelper Integration Test Suite")
    print("=" * 60)
    
    try:
        # Test basic functionality
        test_llm_helper_basic()
        
        # Test RAG integration
        test_rag_integration()
        
        # Test provider switching
        test_provider_switching()
        
        # Test advanced capabilities
        test_advanced_capabilities()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()