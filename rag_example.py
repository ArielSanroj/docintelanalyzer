"""
Ejemplo de uso del sistema RAG + LLM integrado.
Este archivo demuestra cómo usar las nuevas funcionalidades.
"""

import logging
from docsreview import llm
from rag_system import RAGSystem
from rag_llm_integration import RAGLLMIntegration

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_rag_llm_integration():
    """Demuestra el uso del sistema RAG + LLM integrado."""
    
    # Texto de ejemplo (simulando un documento)
    sample_document = """
    CONTRATO DE PRESTACIÓN DE SERVICIOS
    
    Entre los suscritos, por una parte, JUAN PÉREZ GARCÍA, identificado con cédula de ciudadanía 
    número 12345678, domiciliado en la ciudad de Bogotá, quien en adelante se denominará EL PRESTADOR, 
    y por la otra parte, EMPRESA ABC S.A.S., identificada con NIT 900123456-1, domiciliada en 
    Medellín, quien en adelante se denominará EL CONTRATANTE, se ha convenido celebrar el presente 
    contrato de prestación de servicios bajo las siguientes cláusulas:
    
    PRIMERA. OBJETO: El objeto del presente contrato es la prestación de servicios de consultoría 
    en tecnología de la información por parte de EL PRESTADOR a favor de EL CONTRATANTE.
    
    SEGUNDA. OBLIGACIONES DEL PRESTADOR: EL PRESTADOR se compromete a:
    a) Desarrollar un sistema de gestión empresarial
    b) Capacitar al personal en el uso del sistema
    c) Proporcionar soporte técnico durante 6 meses
    d) Entregar la documentación técnica completa
    
    TERCERA. OBLIGACIONES DEL CONTRATANTE: EL CONTRATANTE se compromete a:
    a) Pagar la suma de $50,000,000 pesos colombianos
    b) Proporcionar acceso a los sistemas necesarios
    c) Designar un representante técnico
    d) Realizar los pagos según cronograma establecido
    
    CUARTA. PLAZO: El presente contrato tendrá una duración de 12 meses contados a partir de la 
    fecha de suscripción.
    
    QUINTA. FORMA DE PAGO: Los pagos se realizarán mensualmente por valor de $4,166,667 pesos 
    colombianos durante los primeros 11 meses, y el saldo restante en el mes 12.
    
    SEXTA. TERMINACIÓN: El contrato podrá terminarse por mutuo acuerdo o por incumplimiento 
    de cualquiera de las partes con previo aviso de 30 días.
    
    En constancia de lo anterior, se firma el presente contrato en Bogotá, el 15 de marzo de 2024.
    """
    
    print("=== DEMO: Sistema RAG + LLM Integrado ===\n")
    
    # 1. Inicializar sistema RAG
    print("1. Inicializando sistema RAG...")
    rag_system = RAGSystem()
    
    # 2. Procesar documento
    print("2. Procesando documento...")
    rag_system.process_document(sample_document)
    
    # 3. Mostrar estadísticas
    stats = rag_system.get_document_stats()
    print(f"3. Estadísticas del documento:")
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # 4. Inicializar integración RAG + LLM
    print("\n4. Inicializando integración RAG + LLM...")
    rag_llm = RAGLLMIntegration(llm, rag_system)
    
    # 5. Ejemplos de consultas
    queries = [
        "¿Cuál es el objeto del contrato?",
        "¿Cuánto tiempo dura el contrato?",
        "¿Cuáles son las obligaciones del prestador?",
        "¿Cómo se realizan los pagos?",
        "¿Qué pasa si hay incumplimiento?"
    ]
    
    print("\n5. Generando respuestas con RAG + LLM:")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\nConsulta {i}: {query}")
        print("-" * 40)
        
        # Generar respuesta
        response = rag_llm.generate_response(query)
        
        print(f"Respuesta: {response.answer}")
        print(f"Confianza: {response.confidence_score:.3f}")
        print(f"Chunks encontrados: {response.retrieval_info['chunks_found']}")
        print(f"Tipo de consulta: {response.retrieval_info['query_type']}")
        
        # Mostrar chunks relevantes
        if response.relevant_chunks:
            print("Chunks relevantes:")
            for j, chunk in enumerate(response.relevant_chunks, 1):
                print(f"  {j}. {chunk[:100]}...")
        
        print("=" * 60)
    
    # 6. Demostrar análisis de intención
    print("\n6. Análisis de intenciones de consulta:")
    print("-" * 40)
    
    test_queries = [
        "¿Qué es un contrato?",
        "¿Cómo se desarrolla un sistema?",
        "¿Cuándo se firma el contrato?",
        "¿Dónde se domicilia la empresa?",
        "¿Quién es el prestador?",
        "¿Por qué se termina el contrato?"
    ]
    
    for query in test_queries:
        intent = rag_llm.analyze_query_intent(query)
        print(f"'{query}' -> Tipo: {intent['query_type']}, Palabras clave: {intent['important_words']}")
    
    # 7. Resumen de conversación
    print("\n7. Resumen de conversación:")
    print("-" * 40)
    conv_summary = rag_llm.get_conversation_summary()
    for key, value in conv_summary.items():
        print(f"{key}: {value}")
    
    print("\n=== Demo completado ===")

if __name__ == "__main__":
    demo_rag_llm_integration()