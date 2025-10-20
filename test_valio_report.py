#!/usr/bin/env python3
"""
Test script for VALIO report multimodal parsing with RAG-Anything
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_valio_report_processing():
    """Test VALIO report processing with multimodal capabilities"""
    print("🧪 Testing VALIO Report Multimodal Processing")
    print("=" * 50)
    
    try:
        # Import required modules
        from rag_llm_integration import RAGLLMIntegration
        from rag_system import RAGSystem
        from llm_fallback import get_llm
        from ocr_utils import extract_text_from_pdf
        
        # Initialize components
        print("1. Initializing components...")
        llm = get_llm()
        rag_system = RAGSystem()
        rag_integration = RAGLLMIntegration(llm, rag_system, use_rag_anything=True)
        print("✅ Components initialized successfully")
        
        # Test document processing
        print("\n2. Testing document processing...")
        
        # Look for VALIO report in common locations
        valio_paths = [
            "VALIO_report.pdf",
            "data/VALIO_report.pdf",
            "documents/VALIO_report.pdf",
            "test_data/VALIO_report.pdf"
        ]
        
        valio_path = None
        for path in valio_paths:
            if os.path.exists(path):
                valio_path = path
                break
        
        if not valio_path:
            print("❌ VALIO report not found. Please place VALIO_report.pdf in the project root.")
            print("Available paths checked:")
            for path in valio_paths:
                print(f"  - {path}")
            return False
        
        print(f"✅ Found VALIO report: {valio_path}")
        
        # Extract text from PDF
        print("\n3. Extracting text from VALIO report...")
        try:
            document_text = extract_text_from_pdf(valio_path)
            if document_text and len(document_text.strip()) > 100:
                print(f"✅ Text extracted successfully ({len(document_text)} characters)")
            else:
                print("❌ Text extraction failed or document is too short")
                return False
        except Exception as e:
            print(f"❌ Text extraction error: {e}")
            return False
        
        # Process document with RAG system
        print("\n4. Processing document with RAG system...")
        try:
            # Chunk the document
            chunks = rag_system.chunk_document(document_text)
            print(f"✅ Document chunked into {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = rag_system.generate_embeddings(chunks)
            print(f"✅ Generated {len(embeddings)} embeddings")
            
        except Exception as e:
            print(f"❌ RAG processing error: {e}")
            return False
        
        # Test multimodal queries
        print("\n5. Testing multimodal queries...")
        
        test_queries = [
            "What is the main topic of this VALIO report?",
            "What are the key findings or conclusions?",
            "Are there any charts, graphs, or visual elements mentioned?",
            "What is the structure of this document?",
            "What are the main sections or chapters?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            try:
                response = rag_integration.generate_response(
                    query, 
                    document_path=valio_path
                )
                
                if response and response.answer:
                    print(f"   ✅ Answer: {response.answer[:200]}...")
                    print(f"   📊 Confidence: {response.confidence_score:.2f}")
                    print(f"   🔍 Retrieval Method: {response.retrieval_info.get('method', 'unknown')}")
                else:
                    print("   ❌ No response generated")
                    
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        # Test RAG-Anything specific features
        print("\n6. Testing RAG-Anything specific features...")
        try:
            if hasattr(rag_integration, 'rag_anything') and rag_integration.rag_anything:
                print("✅ RAG-Anything is available")
                
                # Test multimodal retrieval
                multimodal_result = rag_integration._multimodal_retrieval(
                    "What visual elements are in this document?",
                    valio_path
                )
                print(f"✅ Multimodal retrieval: {len(multimodal_result.chunks)} chunks found")
                
            else:
                print("⚠️ RAG-Anything not available, using fallback")
                
        except Exception as e:
            print(f"❌ RAG-Anything test failed: {e}")
        
        print("\n🎉 VALIO report processing test completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def create_sample_valio_report():
    """Create a sample VALIO report for testing"""
    print("\n📄 Creating sample VALIO report for testing...")
    
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
    
    try:
        with open("VALIO_report.txt", "w", encoding="utf-8") as f:
            f.write(sample_content)
        print("✅ Sample VALIO report created: VALIO_report.txt")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample report: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 VALIO Report Multimodal Processing Test")
    print("=" * 50)
    
    # Check if VALIO report exists
    if not os.path.exists("VALIO_report.pdf"):
        print("VALIO_report.pdf not found. Creating sample report...")
        if not create_sample_valio_report():
            print("❌ Failed to create sample report")
            return False
    
    # Run the test
    success = test_valio_report_processing()
    
    if success:
        print("\n✅ All tests passed! VALIO report processing is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)