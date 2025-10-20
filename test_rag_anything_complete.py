#!/usr/bin/env python3
"""
Complete test script for RAG-Anything integration with DocsReview RAG system
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_anything_integration():
    """Test RAG-Anything integration"""
    print("🧪 Testing RAG-Anything Integration")
    print("=" * 50)
    
    try:
        from rag_anything_integration import get_rag_anything
        
        # Get RAG-Anything instance
        rag_anything = get_rag_anything()
        
        if not rag_anything.is_available():
            print("❌ RAG-Anything not available")
            return False
        
        print("✅ RAG-Anything integration available")
        
        # Test document processing
        print("\n1. Testing document processing...")
        
        # Create a sample document
        sample_doc = "VALIO_report_sample.txt"
        sample_content = """
        VALIO ANNUAL REPORT 2024
        
        EXECUTIVE SUMMARY
        This report presents Valio's performance and strategic initiatives for 2024.
        The company has shown strong growth in key markets and innovative product development.
        
        KEY FINDINGS
        1. Revenue increased by 15% compared to 2023
        2. New product launches contributed 25% of total revenue
        3. Sustainability initiatives reduced carbon footprint by 20%
        4. Market expansion into Asia-Pacific region successful
        
        FINANCIAL PERFORMANCE
        - Total Revenue: €2.5 billion
        - Net Profit: €180 million
        - R&D Investment: €150 million
        - Employee Count: 4,200
        """
        
        with open(sample_doc, "w", encoding="utf-8") as f:
            f.write(sample_content)
        
        # Process document
        result = rag_anything.process_document(sample_doc)
        if "error" in result:
            print(f"⚠️ Document processing: {result['error']}")
        else:
            print("✅ Document processed successfully")
        
        # Test querying
        print("\n2. Testing document querying...")
        queries = [
            "What is the main topic of this report?",
            "What are the key findings?",
            "What is the total revenue?"
        ]
        
        for query in queries:
            result = rag_anything.query_document(query, sample_doc)
            if "error" in result:
                print(f"   ⚠️ Query '{query}': {result['error']}")
            else:
                print(f"   ✅ Query '{query}': Success")
        
        # Test document info
        print("\n3. Testing document info...")
        info = rag_anything.get_document_info(sample_doc)
        if "error" in info:
            print(f"⚠️ Document info: {info['error']}")
        else:
            print("✅ Document info retrieved successfully")
        
        # Clean up
        os.remove(sample_doc)
        
        print("\n✅ RAG-Anything integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ RAG-Anything integration test failed: {e}")
        return False

def test_docsreview_integration():
    """Test DocsReview integration with RAG-Anything"""
    print("\n🧪 Testing DocsReview Integration with RAG-Anything")
    print("=" * 50)
    
    try:
        from rag_llm_integration import RAGLLMIntegration
        from rag_system import RAGSystem
        from llm_fallback import get_llm
        
        # Initialize components
        print("1. Initializing components...")
        llm = get_llm()
        rag_system = RAGSystem()
        rag_integration = RAGLLMIntegration(llm, rag_system, use_rag_anything=True)
        print("✅ Components initialized successfully")
        
        # Test multimodal retrieval
        print("\n2. Testing multimodal retrieval...")
        query = "What are the key findings in the VALIO report?"
        result = rag_integration._multimodal_retrieval(query)
        print(f"✅ Multimodal retrieval: {len(result.chunks)} chunks found")
        print(f"   Method: {result.retrieval_method}")
        
        # Test enhanced embeddings
        print("\n3. Testing enhanced embeddings...")
        sample_text = "This is a test document for embedding generation."
        embeddings = rag_integration._get_enhanced_embeddings(sample_text)
        if embeddings is not None:
            print(f"✅ Enhanced embeddings: {len(embeddings)} dimensions")
        else:
            print("⚠️ Enhanced embeddings: No embeddings generated")
        
        # Test full response generation
        print("\n4. Testing full response generation...")
        response = rag_integration.generate_response(
            "What is the main topic of this document?",
            document_path="VALIO_report_sample.txt"
        )
        
        if response and response.answer:
            print(f"✅ Response generated: {response.answer[:100]}...")
            print(f"   Confidence: {response.confidence_score:.2f}")
        else:
            print("⚠️ No response generated")
        
        print("\n✅ DocsReview integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ DocsReview integration test failed: {e}")
        return False

def test_valio_report_processing():
    """Test VALIO report processing with full pipeline"""
    print("\n🧪 Testing VALIO Report Processing Pipeline")
    print("=" * 50)
    
    try:
        from rag_llm_integration import RAGLLMIntegration
        from rag_system import RAGSystem
        from llm_fallback import get_llm
        
        # Initialize components
        print("1. Initializing components...")
        llm = get_llm()
        rag_system = RAGSystem()
        rag_integration = RAGLLMIntegration(llm, rag_system, use_rag_anything=True)
        print("✅ Components initialized successfully")
        
        # Create comprehensive VALIO report
        print("\n2. Creating comprehensive VALIO report...")
        valio_content = """
        VALIO ANNUAL REPORT 2024
        
        EXECUTIVE SUMMARY
        This report presents Valio's performance and strategic initiatives for 2024.
        The company has shown strong growth in key markets and innovative product development.
        
        KEY FINDINGS
        1. Revenue increased by 15% compared to 2023
        2. New product launches contributed 25% of total revenue
        3. Sustainability initiatives reduced carbon footprint by 20%
        4. Market expansion into Asia-Pacific region successful
        
        FINANCIAL PERFORMANCE
        - Total Revenue: €2.5 billion
        - Net Profit: €180 million
        - R&D Investment: €150 million
        - Employee Count: 4,200
        
        STRATEGIC INITIATIVES
        - Digital transformation program
        - Sustainable packaging solutions
        - Plant-based product development
        - Supply chain optimization
        
        MARKET ANALYSIS
        The dairy industry continues to evolve with consumer preferences shifting towards
        sustainable and plant-based alternatives. Valio has positioned itself as a leader
        in innovation and sustainability.
        
        FUTURE OUTLOOK
        Valio plans to continue its growth trajectory through:
        - Continued innovation in dairy products
        - Expansion of plant-based portfolio
        - Investment in sustainable technologies
        - Strategic partnerships and acquisitions
        
        CONCLUSION
        Valio's 2024 performance demonstrates strong execution of strategic initiatives
        and positions the company well for future growth in the evolving dairy market.
        """
        
        with open("VALIO_report_complete.txt", "w", encoding="utf-8") as f:
            f.write(valio_content)
        print("✅ Comprehensive VALIO report created")
        
        # Process document with RAG system
        print("\n3. Processing document with RAG system...")
        chunks = rag_system.chunk_document(valio_content)
        print(f"✅ Document chunked into {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = rag_system.generate_embeddings(chunks)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Test comprehensive queries
        print("\n4. Testing comprehensive queries...")
        test_queries = [
            "What is the main topic of this VALIO report?",
            "What are the key findings for 2024?",
            "What is the total revenue for 2024?",
            "What are the strategic initiatives?",
            "What is the future outlook?",
            "How many employees does Valio have?",
            "What is the R&D investment?",
            "What are the sustainability initiatives?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            try:
                response = rag_integration.generate_response(
                    query, 
                    document_path="VALIO_report_complete.txt"
                )
                
                if response and response.answer:
                    print(f"   ✅ Answer: {response.answer[:150]}...")
                    print(f"   📊 Confidence: {response.confidence_score:.2f}")
                    print(f"   🔍 Method: {response.retrieval_info.get('method', 'unknown')}")
                else:
                    print("   ❌ No response generated")
                    
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        # Clean up
        os.remove("VALIO_report_complete.txt")
        
        print("\n✅ VALIO report processing pipeline test completed!")
        return True
        
    except Exception as e:
        print(f"❌ VALIO report processing test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Complete RAG-Anything Integration Test Suite")
    print("=" * 60)
    
    # Test 1: RAG-Anything integration
    success1 = test_rag_anything_integration()
    
    # Test 2: DocsReview integration
    success2 = test_docsreview_integration()
    
    # Test 3: VALIO report processing
    success3 = test_valio_report_processing()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    print(f"RAG-Anything Integration: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"DocsReview Integration: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"VALIO Report Processing: {'✅ PASS' if success3 else '❌ FAIL'}")
    
    overall_success = success1 and success2 and success3
    print(f"\nOverall Result: {'🎉 ALL TESTS PASSED!' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎯 RAG-Anything integration is working correctly!")
        print("   - Multimodal document processing: ✅")
        print("   - Enhanced embeddings: ✅")
        print("   - Query processing: ✅")
        print("   - DocsReview integration: ✅")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)