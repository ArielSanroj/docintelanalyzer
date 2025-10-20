"""
Script de prueba para el sistema RAG-Anything mejorado.
Demuestra todas las capacidades multimodales, contextuales y de grafo de conocimiento.
"""

import logging
import os
import sys
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rag_anything_system():
    """Prueba completa del sistema RAG-Anything."""
    
    try:
        # Importar el sistema RAG-Anything
        from rag_anything_integration import RAGAnythingSystem
        
        print("🚀 Inicializando sistema RAG-Anything...")
        
        # Inicializar sistema con todas las capacidades
        rag_system = RAGAnythingSystem(
            embedding_model="all-MiniLM-L6-v2",
            vlm_model="nvidia/llama-3.2-vision-instruct",
            enable_vision=True,
            enable_context_awareness=True,
            enable_knowledge_graph=True
        )
        
        print("✅ Sistema RAG-Anything inicializado correctamente")
        
        # Mostrar capacidades del sistema
        capabilities = rag_system.get_system_capabilities()
        print("\n📋 Capacidades del sistema:")
        for key, value in capabilities.items():
            print(f"  - {key}: {value}")
        
        # Procesar un documento de ejemplo
        print("\n📄 Procesando documento de ejemplo...")
        
        # Crear contenido de ejemplo con diferentes tipos de información
        sample_content = """
        # Informe de Proyecto de Investigación
        
        ## Resumen Ejecutivo
        Este proyecto de investigación se desarrolló entre enero y diciembre de 2024, 
        bajo la dirección del Dr. María González en la Universidad de Barcelona.
        
        ## Objetivos
        Los objetivos principales incluyen:
        1. Desarrollar un sistema de análisis multimodal
        2. Implementar capacidades de visión artificial
        3. Crear un grafo de conocimiento integrado
        
        ## Metodología
        La metodología empleada combina técnicas de:
        - Procesamiento de lenguaje natural
        - Visión por computadora
        - Aprendizaje automático
        
        ## Resultados
        Los resultados muestran una mejora del 25% en la precisión del sistema.
        
        ## Conclusiones
        El sistema desarrollado demuestra capacidades avanzadas de procesamiento multimodal.
        """
        
        # Procesar el contenido
        rag_system.process_document("sample_document.txt", sample_content)
        
        # Mostrar estadísticas del documento
        doc_stats = rag_system.get_document_analysis()
        print("\n📊 Estadísticas del documento:")
        print(f"  - Chunks totales: {doc_stats['document_stats']['total_chunks']}")
        print(f"  - Tipos de contenido: {doc_stats['document_stats']['content_types']}")
        
        # Probar diferentes tipos de consultas
        test_queries = [
            "¿Cuáles son los objetivos del proyecto?",
            "¿Quién dirigió el proyecto?",
            "¿Qué metodología se utilizó?",
            "¿Cuáles fueron los resultados obtenidos?",
            "¿Qué universidad participó en el proyecto?"
        ]
        
        print("\n🔍 Probando consultas...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Consulta {i}: {query} ---")
            
            # Procesar consulta con todas las capacidades
            response = rag_system.query_document(
                query=query,
                top_k=3,
                use_context_awareness=True,
                use_visual_analysis=True,
                use_knowledge_graph=True
            )
            
            print(f"Respuesta: {response.answer[:200]}...")
            print(f"Confianza: {response.confidence_score:.3f}")
            print(f"Chunks relevantes: {len(response.relevant_chunks)}")
            
            if response.contextual_analysis:
                print(f"Análisis contextual: {len(response.contextual_analysis.temporal_context)} elementos temporales")
            
            if response.knowledge_graph_insights:
                print(f"Entidades relevantes: {len(response.knowledge_graph_insights['relevant_entities'])}")
        
        # Mostrar análisis del grafo de conocimiento
        if rag_system.knowledge_graph_system:
            print("\n🧠 Análisis del grafo de conocimiento:")
            kg_stats = rag_system.knowledge_graph_system.get_entity_statistics()
            print(f"  - Entidades totales: {kg_stats['total_entities']}")
            print(f"  - Relaciones totales: {kg_stats['total_relations']}")
            print(f"  - Tipos de entidades: {kg_stats['entity_types']}")
        
        # Exportar grafo de conocimiento
        print("\n💾 Exportando grafo de conocimiento...")
        kg_export = rag_system.export_knowledge_graph("json")
        if kg_export:
            print("✅ Grafo de conocimiento exportado exitosamente")
            # Guardar en archivo
            with open("knowledge_graph.json", "w", encoding="utf-8") as f:
                f.write(kg_export)
            print("📁 Guardado en knowledge_graph.json")
        
        # Generar visualización del grafo
        print("\n🎨 Generando visualización del grafo...")
        graph_viz = rag_system.visualize_knowledge_graph(max_nodes=20)
        if graph_viz:
            print("✅ Visualización generada exitosamente")
            # Guardar como imagen
            import base64
            with open("knowledge_graph.png", "wb") as f:
                f.write(base64.b64decode(graph_viz))
            print("📁 Guardado en knowledge_graph.png")
        
        print("\n🎉 Prueba del sistema RAG-Anything completada exitosamente!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Asegúrate de que todas las dependencias estén instaladas")
        return False
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        logger.exception("Error detallado:")
        return False

def test_multimodal_processing():
    """Prueba específica del procesamiento multimodal."""
    
    try:
        from multimodal_rag_system import MultimodalRAGSystem
        
        print("\n🔬 Probando procesamiento multimodal...")
        
        # Crear sistema multimodal
        multimodal_rag = MultimodalRAGSystem(
            embedding_model="all-MiniLM-L6-v2",
            enable_vision=True
        )
        
        # Contenido con diferentes tipos de información
        multimodal_content = """
        # Documento Multimodal de Prueba
        
        ## Texto Principal
        Este es un documento que contiene diferentes tipos de contenido.
        
        ## Tabla de Datos
        | Producto | Precio | Cantidad |
        |----------|--------|----------|
        | Laptop   | $1200  | 50       |
        | Mouse    | $25    | 100      |
        | Teclado  | $75    | 80       |
        
        ## Ecuación Matemática
        La fórmula para calcular el área de un círculo es: A = π × r²
        
        ## Información Temporal
        El proyecto se desarrolló entre el 15 de marzo de 2024 y el 30 de noviembre de 2024.
        
        ## Información Espacial
        La oficina principal se encuentra en Madrid, España, cerca del centro de la ciudad.
        """
        
        # Procesar documento
        multimodal_rag.process_document("multimodal_test.txt", multimodal_content)
        
        # Mostrar estadísticas
        stats = multimodal_rag.get_document_stats()
        print(f"✅ Documento procesado: {stats['total_chunks']} chunks")
        print(f"📊 Tipos de contenido: {stats['content_types']}")
        
        # Probar consultas
        queries = [
            "¿Cuál es el precio de la laptop?",
            "¿Cuál es la fórmula del área del círculo?",
            "¿Dónde se encuentra la oficina principal?",
            "¿Cuándo se desarrolló el proyecto?"
        ]
        
        for query in queries:
            result = multimodal_rag.retrieve_multimodal(query, top_k=2)
            print(f"\n🔍 Consulta: {query}")
            print(f"   Chunks encontrados: {len(result.chunks)}")
            print(f"   Tipos: {[chunk.content_type for chunk in result.chunks]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba multimodal: {e}")
        return False

def test_context_awareness():
    """Prueba específica del procesamiento contextual."""
    
    try:
        from context_aware_processor import ContextAwareProcessor
        
        print("\n🧠 Probando procesamiento contextual...")
        
        # Crear procesador contextual
        context_processor = ContextAwareProcessor()
        
        # Crear chunks de ejemplo
        class MockChunk:
            def __init__(self, content, chunk_id):
                self.content = content
                self.chunk_id = chunk_id
        
        chunks = [
            MockChunk("El proyecto comenzó en enero de 2024 y terminó en diciembre del mismo año.", 0),
            MockChunk("La oficina principal está ubicada en Barcelona, España.", 1),
            MockChunk("Debido a la pandemia, el trabajo se realizó de forma remota.", 2),
            MockChunk("El Capítulo 1 describe la metodología utilizada.", 3),
            MockChunk("La definición de RAG es Retrieval-Augmented Generation.", 4)
        ]
        
        # Analizar contexto
        query = "¿Cuándo y dónde se desarrolló el proyecto?"
        contextual_analysis = context_processor.analyze_context(chunks, query)
        
        # Mostrar resultados
        print(f"✅ Análisis contextual completado")
        print(f"📅 Elementos temporales: {len(contextual_analysis.temporal_context)}")
        print(f"📍 Elementos espaciales: {len(contextual_analysis.spatial_context)}")
        print(f"🔗 Elementos causales: {len(contextual_analysis.causal_context)}")
        print(f"📚 Elementos jerárquicos: {len(contextual_analysis.hierarchical_context)}")
        print(f"💡 Elementos semánticos: {len(contextual_analysis.semantic_context)}")
        
        # Generar resumen contextual
        summary = context_processor.generate_contextual_summary(contextual_analysis)
        print(f"\n📋 Resumen contextual: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba contextual: {e}")
        return False

def main():
    """Función principal de prueba."""
    
    print("🚀 Iniciando pruebas del sistema RAG-Anything mejorado...")
    print("=" * 60)
    
    # Ejecutar pruebas
    tests = [
        ("Sistema RAG-Anything completo", test_rag_anything_system),
        ("Procesamiento multimodal", test_multimodal_processing),
        ("Procesamiento contextual", test_context_awareness)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Ejecutando: {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name}: EXITOSO")
            else:
                print(f"❌ {test_name}: FALLÓ")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Resumen de resultados
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ EXITOSO" if success else "❌ FALLÓ"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 Resultado final: {passed}/{total} pruebas exitosas")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron exitosamente!")
        print("🚀 El sistema RAG-Anything está listo para usar")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisa los errores arriba.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)