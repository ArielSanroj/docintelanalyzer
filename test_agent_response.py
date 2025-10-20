#!/usr/bin/env python3
"""
Script para probar las respuestas del agente RAG.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_rag_system import AdvancedRAGSystem
from advanced_rag_integration import AdvancedRAGLLMIntegration
from llm_fallback import get_llm

def test_agent_response():
    """Prueba la respuesta del agente con un documento de ejemplo."""
    print("🧪 Probando respuestas del agente RAG...")
    
    try:
        # Inicializar LLM
        print("1. Inicializando LLM...")
        llm = get_llm()
        print(f"✅ LLM inicializado: {type(llm).__name__}")
        
        # Crear documento de prueba
        test_document = """
        BENEFICIOS PARA SOCIOS
        
        Los socios de nuestra empresa tienen acceso a los siguientes beneficios:
        
        1. Descuentos especiales en productos y servicios
        2. Acceso prioritario a nuevos lanzamientos
        3. Programa de puntos que se pueden canjear por premios
        4. Soporte técnico preferencial
        5. Invitaciones a eventos exclusivos
        
        Para más información sobre beneficios, contactar al departamento de socios.
        """
        
        print("2. Inicializando sistema RAG avanzado...")
        rag_system = AdvancedRAGSystem()
        
        # Procesar documento
        print("3. Procesando documento de prueba...")
        rag_system.process_document(test_document)
        print(f"✅ Documento procesado: {rag_system.get_document_stats()}")
        
        # Inicializar integración
        print("4. Inicializando integración RAG-LLM...")
        rag_integration = AdvancedRAGLLMIntegration(llm, rag_system)
        
        # Probar consultas
        test_queries = [
            "¿Qué beneficios tienen los socios?",
            "¿Cómo puedo obtener descuentos como socio?",
            "¿Qué es el programa de puntos?"
        ]
        
        print("\n5. Probando consultas:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Consulta {i}: {query} ---")
            try:
                response = rag_integration.generate_response_with_agent(query)
                print(f"✅ Respuesta: {response.answer[:100]}...")
                print(f"   Confianza: {response.confidence_score}")
                print(f"   Herramientas usadas: {response.retrieval_info.get('tool_calls', 0)}")
                if response.agent_reasoning:
                    print(f"   Razonamiento: {response.agent_reasoning}")
            except Exception as e:
                print(f"❌ Error en consulta {i}: {e}")
        
        print("\n✅ Prueba completada!")
        
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent_response()