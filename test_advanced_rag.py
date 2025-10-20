#!/usr/bin/env python3
"""
Script de prueba para demostrar las capacidades del sistema RAG avanzado.
"""

import os
import time
from dotenv import load_dotenv
from advanced_rag_system import AdvancedRAGSystem
from advanced_rag_integration import AdvancedRAGLLMIntegration
from llm_fallback import get_llm

# Cargar variables de entorno
load_dotenv()

def test_advanced_rag():
    """Prueba el sistema RAG avanzado con un documento de ejemplo."""
    
    print("🚀 Iniciando prueba del Sistema RAG Avanzado")
    print("=" * 50)
    
    # Documento de ejemplo
    sample_document = """
    Contrato de Servicios de Consultoría
    
    Este contrato establece los términos y condiciones para la prestación de servicios de consultoría 
    en tecnología por parte de TechConsult LLC a la empresa cliente.
    
    Términos Principales:
    1. Duración: El contrato tendrá una duración de 12 meses a partir de la fecha de firma.
    2. Honorarios: Los honorarios se establecerán en $150 por hora de consultoría.
    3. Pagos: Los pagos se realizarán mensualmente dentro de los 30 días posteriores a la facturación.
    4. Confidencialidad: Ambas partes se comprometen a mantener la confidencialidad de la información.
    5. Terminación: Cualquiera de las partes puede terminar el contrato con 30 días de anticipación.
    
    Responsabilidades del Consultor:
    - Proporcionar asesoramiento técnico especializado
    - Desarrollar estrategias de implementación tecnológica
    - Capacitar al personal del cliente
    - Entregar reportes mensuales de progreso
    
    Responsabilidades del Cliente:
    - Proporcionar acceso a sistemas y personal necesario
    - Pagar los honorarios según lo acordado
    - Colaborar en la implementación de recomendaciones
    - Mantener la confidencialidad de metodologías propietarias
    
    Este contrato se rige por las leyes del estado de California y cualquier disputa será resuelta 
    mediante arbitraje vinculante.
    """
    
    try:
        # Inicializar LLM
        print("📥 Inicializando LLM...")
        llm = get_llm()
        
        # Inicializar sistema RAG avanzado
        print("🤖 Inicializando Sistema RAG Avanzado...")
        start_time = time.time()
        
        rag_system = AdvancedRAGSystem()
        
        # Procesar documento
        print("📄 Procesando documento...")
        rag_system.process_document(sample_document, chunk_size=800, chunk_overlap=120)
        
        # Crear integración avanzada
        print("🔗 Configurando integración avanzada...")
        rag_integration = AdvancedRAGLLMIntegration(llm, rag_system)
        
        init_time = time.time() - start_time
        print(f"✅ Sistema inicializado en {init_time:.2f} segundos")
        
        # Mostrar estadísticas
        stats = rag_system.get_document_stats()
        print("\n📊 Estadísticas del Sistema:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
        # Pruebas de consultas
        test_queries = [
            "¿Cuáles son los honorarios por hora?",
            "¿Cuánto tiempo dura el contrato?",
            "¿Cuáles son las responsabilidades del consultor?",
            "¿Cómo se pueden terminar los pagos?",
            "¿Qué leyes rigen este contrato?"
        ]
        
        print("\n🔍 Probando consultas:")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Consulta {i}: {query}")
            print("-" * 30)
            
            start_query = time.time()
            
            # Usar agente ReAct si está disponible
            if rag_integration.agent:
                response = rag_integration.generate_response_with_agent(query)
                print(f"🤖 Respuesta (Agente ReAct): {response.answer}")
                if response.agent_reasoning:
                    print(f"🧠 Razonamiento: {response.agent_reasoning}")
            else:
                response = rag_integration.generate_response_traditional(query)
                print(f"💬 Respuesta (Tradicional): {response.answer}")
            
            query_time = time.time() - start_query
            print(f"⏱️ Tiempo de respuesta: {query_time:.2f} segundos")
            print(f"🎯 Score de confianza: {response.confidence_score:.2f}")
            
            if response.relevant_chunks:
                print(f"📚 Chunks relevantes encontrados: {len(response.relevant_chunks)}")
        
        # Prueba de búsqueda híbrida
        print("\n🔍 Probando búsqueda híbrida:")
        print("=" * 50)
        
        hybrid_query = "confidencialidad y terminación"
        print(f"📝 Consulta híbrida: {hybrid_query}")
        
        retrieval_result = rag_system.retrieve_documents(hybrid_query, search_type="hybrid", top_k=3)
        print(f"🔍 Tipo de búsqueda: {retrieval_result.retrieval_type}")
        print(f"📊 Documentos encontrados: {len(retrieval_result.chunks)}")
        
        for i, chunk in enumerate(retrieval_result.chunks):
            print(f"  {i+1}. Score: {retrieval_result.scores[i]:.3f}")
            print(f"     Texto: {chunk.text[:100]}...")
        
        print("\n✅ Prueba completada exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_advanced_rag()