"""
Aplicación Streamlit mejorada con sistema RAG-Anything.
Integra capacidades multimodales, contextuales y de grafo de conocimiento.
"""

import streamlit as st
import logging
import os
from pathlib import Path
import tempfile
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar página
st.set_page_config(
    page_title="RAG-Anything Enhanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_rag_system():
    """Inicializa el sistema RAG-Anything."""
    if 'rag_system' not in st.session_state:
        try:
            from rag_anything_integration import RAGAnythingSystem
            
            st.session_state.rag_system = RAGAnythingSystem(
                embedding_model="all-MiniLM-L6-v2",
                vlm_model="nvidia/llama-3.2-vision-instruct",
                enable_vision=True,
                enable_context_awareness=True,
                enable_knowledge_graph=True
            )
            st.session_state.rag_initialized = True
            logger.info("Sistema RAG-Anything inicializado")
        except Exception as e:
            st.error(f"Error inicializando sistema RAG: {e}")
            st.session_state.rag_initialized = False
    else:
        st.session_state.rag_initialized = True

def main():
    """Función principal de la aplicación."""
    
    # Título principal
    st.title("🚀 RAG-Anything Enhanced")
    st.markdown("Sistema RAG Multimodal Avanzado con Capacidades de Visión, Contexto y Grafo de Conocimiento")
    
    # Inicializar sistema
    initialize_rag_system()
    
    if not st.session_state.rag_initialized:
        st.error("No se pudo inicializar el sistema RAG. Verifica las dependencias.")
        return
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Configuración de capacidades
        st.subheader("Capacidades del Sistema")
        use_vision = st.checkbox("👁️ Análisis Visual", value=True, help="Habilitar análisis de imágenes y gráficos")
        use_context = st.checkbox("🧠 Procesamiento Contextual", value=True, help="Habilitar análisis contextual avanzado")
        use_kg = st.checkbox("🔗 Grafo de Conocimiento", value=True, help="Habilitar construcción de grafo de conocimiento")
        
        # Configuración de recuperación
        st.subheader("Parámetros de Recuperación")
        top_k = st.slider("Número de chunks a recuperar", 1, 10, 5)
        min_confidence = st.slider("Confianza mínima", 0.0, 1.0, 0.3, 0.1)
        
        # Mostrar estadísticas del sistema
        if st.session_state.rag_system.multimodal_rag.multimodal_chunks:
            st.subheader("📊 Estadísticas del Documento")
            stats = st.session_state.rag_system.get_document_analysis()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chunks Totales", stats['document_stats']['total_chunks'])
                st.metric("Entidades", stats.get('knowledge_graph', {}).get('total_entities', 0))
            
            with col2:
                content_types = stats['document_stats']['content_types']
                st.metric("Tipos de Contenido", len(content_types))
                for content_type, count in content_types.items():
                    st.caption(f"{content_type}: {count}")
    
    # Pestañas principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Documentos", 
        "🔍 Consultas", 
        "🧠 Grafo de Conocimiento", 
        "📊 Análisis", 
        "⚙️ Configuración"
    ])
    
    with tab1:
        document_processing_tab()
    
    with tab2:
        query_processing_tab(top_k, use_vision, use_context, use_kg)
    
    with tab3:
        knowledge_graph_tab()
    
    with tab4:
        analysis_tab()
    
    with tab5:
        configuration_tab()

def document_processing_tab():
    """Pestaña de procesamiento de documentos."""
    st.header("📄 Procesamiento de Documentos")
    
    # Opciones de carga
    upload_option = st.radio(
        "Selecciona método de carga:",
        ["Subir archivo", "Usar archivo de ejemplo", "Pegar texto"]
    )
    
    if upload_option == "Subir archivo":
        uploaded_file = st.file_uploader(
            "Selecciona un documento",
            type=['pdf', 'docx', 'pptx', 'xlsx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'txt'],
            help="Formatos soportados: PDF, DOCX, PPTX, XLSX, imágenes, TXT"
        )
        
        if uploaded_file is not None:
            # Guardar archivo temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner("Procesando documento..."):
                    st.session_state.rag_system.process_document(tmp_path)
                
                st.success("✅ Documento procesado exitosamente!")
                
                # Mostrar estadísticas
                stats = st.session_state.rag_system.get_document_analysis()
                display_document_stats(stats)
                
            except Exception as e:
                st.error(f"Error procesando documento: {e}")
            finally:
                # Limpiar archivo temporal
                os.unlink(tmp_path)
    
    elif upload_option == "Usar archivo de ejemplo":
        if st.button("Cargar documento de ejemplo"):
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
            
            try:
                with st.spinner("Procesando documento de ejemplo..."):
                    st.session_state.rag_system.process_document("sample_document.txt", sample_content)
                
                st.success("✅ Documento de ejemplo procesado!")
                display_document_stats(st.session_state.rag_system.get_document_analysis())
                
            except Exception as e:
                st.error(f"Error procesando ejemplo: {e}")
    
    elif upload_option == "Pegar texto":
        text_content = st.text_area(
            "Pega el contenido del documento aquí:",
            height=300,
            help="Pega el texto que quieres analizar"
        )
        
        if st.button("Procesar texto") and text_content.strip():
            try:
                with st.spinner("Procesando texto..."):
                    st.session_state.rag_system.process_document("text_input.txt", text_content)
                
                st.success("✅ Texto procesado exitosamente!")
                display_document_stats(st.session_state.rag_system.get_document_analysis())
                
            except Exception as e:
                st.error(f"Error procesando texto: {e}")

def query_processing_tab(top_k, use_vision, use_context, use_kg):
    """Pestaña de procesamiento de consultas."""
    st.header("🔍 Consultas Inteligentes")
    
    if not st.session_state.rag_system.multimodal_rag.multimodal_chunks:
        st.warning("⚠️ No hay documento cargado. Ve a la pestaña 'Documentos' para cargar uno.")
        return
    
    # Consulta del usuario
    query = st.text_input(
        "Haz tu consulta:",
        placeholder="Ej: ¿Cuáles son los objetivos del proyecto?",
        help="Puedes hacer preguntas sobre el contenido del documento"
    )
    
    # Botones de consultas de ejemplo
    st.subheader("💡 Consultas de Ejemplo")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Resumen general"):
            query = "¿De qué trata este documento?"
    
    with col2:
        if st.button("🎯 Objetivos"):
            query = "¿Cuáles son los objetivos principales?"
    
    with col3:
        if st.button("📊 Resultados"):
            query = "¿Cuáles fueron los resultados obtenidos?"
    
    # Procesar consulta
    if query:
        try:
            with st.spinner("Procesando consulta..."):
                response = st.session_state.rag_system.query_document(
                    query=query,
                    top_k=top_k,
                    use_context_awareness=use_context,
                    use_visual_analysis=use_vision,
                    use_knowledge_graph=use_kg
                )
            
            # Mostrar respuesta
            st.subheader("💬 Respuesta")
            st.write(response.answer)
            
            # Mostrar información adicional
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confianza", f"{response.confidence_score:.3f}")
            with col2:
                st.metric("Chunks Relevantes", len(response.relevant_chunks))
            with col3:
                st.metric("Análisis Visual", "✅" if response.visual_analysis else "❌")
            
            # Mostrar chunks relevantes
            if response.relevant_chunks:
                with st.expander("📄 Chunks Relevantes"):
                    for i, chunk in enumerate(response.relevant_chunks):
                        st.write(f"**Chunk {i+1}** ({chunk.content_type}):")
                        st.write(chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content)
                        st.divider()
            
            # Mostrar análisis contextual
            if response.contextual_analysis:
                with st.expander("🧠 Análisis Contextual"):
                    context_summary = st.session_state.rag_system.context_processor.generate_contextual_summary(
                        response.contextual_analysis
                    )
                    st.write(context_summary)
            
            # Mostrar insights del grafo de conocimiento
            if response.knowledge_graph_insights:
                with st.expander("🔗 Insights del Grafo de Conocimiento"):
                    insights = response.knowledge_graph_insights
                    if insights['relevant_entities']:
                        st.write("**Entidades Relevantes:**")
                        for entity in insights['relevant_entities']:
                            st.write(f"- {entity['name']} ({entity['type']}) - Confianza: {entity['confidence']:.3f}")
                    
                    if insights['entity_relationships']:
                        st.write("**Relaciones Identificadas:**")
                        for rel in insights['entity_relationships']:
                            st.write(f"- {rel['source']} → {rel['target']} ({rel['relation_type']})")
            
        except Exception as e:
            st.error(f"Error procesando consulta: {e}")

def knowledge_graph_tab():
    """Pestaña del grafo de conocimiento."""
    st.header("🧠 Grafo de Conocimiento")
    
    if not st.session_state.rag_system.knowledge_graph_system or not st.session_state.rag_system.knowledge_graph_system.knowledge_graph.entities:
        st.warning("⚠️ No hay grafo de conocimiento disponible. Procesa un documento primero.")
        return
    
    # Estadísticas del grafo
    kg_stats = st.session_state.rag_system.knowledge_graph_system.get_entity_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Entidades", kg_stats['total_entities'])
    with col2:
        st.metric("Relaciones", kg_stats['total_relations'])
    with col3:
        st.metric("Componentes", kg_stats['connected_components'])
    with col4:
        st.metric("Densidad", f"{kg_stats['graph_density']:.3f}")
    
    # Tipos de entidades
    st.subheader("📊 Tipos de Entidades")
    entity_types = kg_stats['entity_types']
    for entity_type, count in entity_types.items():
        st.write(f"**{entity_type.title()}**: {count}")
    
    # Búsqueda de entidades
    st.subheader("🔍 Explorar Entidades")
    entity_search = st.text_input("Buscar entidad:", placeholder="Ej: Universidad de Barcelona")
    
    if entity_search:
        # Buscar entidades relacionadas
        related_entities = st.session_state.rag_system.knowledge_graph_system.find_related_entities(
            entity_search, max_depth=2
        )
        
        if related_entities:
            st.write("**Entidades Relacionadas:**")
            for entity in related_entities:
                st.write(f"- {entity['entity_name']} ({entity['entity_type']}) - Profundidad: {entity['depth']}")
        else:
            st.write("No se encontraron entidades relacionadas.")
    
    # Exportar grafo
    st.subheader("💾 Exportar Grafo")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Exportar JSON"):
            kg_json = st.session_state.rag_system.export_knowledge_graph("json")
            if kg_json:
                st.download_button(
                    "Descargar JSON",
                    kg_json,
                    "knowledge_graph.json",
                    "application/json"
                )
    
    with col2:
        if st.button("Generar Visualización"):
            graph_viz = st.session_state.rag_system.visualize_knowledge_graph(max_nodes=30)
            if graph_viz:
                st.image(graph_viz, caption="Visualización del Grafo de Conocimiento")
            else:
                st.warning("No se pudo generar la visualización. Instala matplotlib para habilitar esta función.")

def analysis_tab():
    """Pestaña de análisis."""
    st.header("📊 Análisis del Sistema")
    
    if not st.session_state.rag_system.multimodal_rag.multimodal_chunks:
        st.warning("⚠️ No hay documento cargado.")
        return
    
    # Análisis del documento
    analysis = st.session_state.rag_system.get_document_analysis()
    
    # Estadísticas generales
    st.subheader("📈 Estadísticas Generales")
    doc_stats = analysis['document_stats']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunks Totales", doc_stats['total_chunks'])
        st.metric("Caracteres Totales", doc_stats['total_characters'])
    with col2:
        st.metric("Palabras Totales", doc_stats['total_words'])
        st.metric("Longitud Promedio", f"{doc_stats['avg_chunk_length']:.0f}")
    
    # Tipos de contenido
    st.subheader("📋 Distribución de Contenido")
    content_types = doc_stats['content_types']
    
    if content_types:
        import plotly.express as px
        import pandas as pd
        
        df = pd.DataFrame(list(content_types.items()), columns=['Tipo', 'Cantidad'])
        fig = px.pie(df, values='Cantidad', names='Tipo', title="Distribución de Tipos de Contenido")
        st.plotly_chart(fig, use_container_width=True)
    
    # Capacidades del sistema
    st.subheader("⚙️ Capacidades del Sistema")
    capabilities = analysis['system_capabilities']
    
    for capability, enabled in capabilities.items():
        if isinstance(enabled, bool):
            st.write(f"**{capability}**: {'✅' if enabled else '❌'}")
        else:
            st.write(f"**{capability}**: {enabled}")
    
    # Estadísticas de procesamiento
    if 'processing_stats' in analysis:
        st.subheader("🔧 Estadísticas de Procesamiento")
        proc_stats = analysis['processing_stats']
        
        for key, value in proc_stats.items():
            if isinstance(value, dict):
                st.write(f"**{key}**:")
                for sub_key, sub_value in value.items():
                    st.write(f"  - {sub_key}: {sub_value}")
            else:
                st.write(f"**{key}**: {value}")

def configuration_tab():
    """Pestaña de configuración."""
    st.header("⚙️ Configuración del Sistema")
    
    # Información del sistema
    st.subheader("ℹ️ Información del Sistema")
    
    if st.session_state.rag_initialized:
        capabilities = st.session_state.rag_system.get_system_capabilities()
        
        st.write("**Modelos Configurados:**")
        st.write(f"- Embedding: {capabilities['embedding_model']}")
        if capabilities['vlm_model']:
            st.write(f"- VLM: {capabilities['vlm_model']}")
        
        st.write("**Capacidades Habilitadas:**")
        for capability, enabled in capabilities.items():
            if isinstance(enabled, bool):
                st.write(f"- {capability}: {'✅' if enabled else '❌'}")
    
    # Configuración avanzada
    st.subheader("🔧 Configuración Avanzada")
    
    if st.button("🔄 Reiniciar Sistema"):
        st.session_state.rag_system.reset_system()
        st.success("Sistema reiniciado exitosamente!")
    
    if st.button("🧹 Limpiar Memoria"):
        if 'rag_system' in st.session_state:
            del st.session_state.rag_system
        st.session_state.rag_initialized = False
        st.success("Memoria limpiada!")
    
    # Información de dependencias
    st.subheader("📦 Dependencias")
    st.write("Para instalar todas las dependencias necesarias:")
    st.code("pip install -r requirements_rag_anything.txt", language="bash")
    
    # Enlaces útiles
    st.subheader("🔗 Enlaces Útiles")
    st.write("- [Documentación RAG-Anything](https://github.com/HKUDS/RAG-Anything)")
    st.write("- [LangChain Documentation](https://python.langchain.com/)")
    st.write("- [Sentence Transformers](https://www.sbert.net/)")

def display_document_stats(stats):
    """Muestra estadísticas del documento."""
    st.subheader("📊 Estadísticas del Documento")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Chunks Totales", stats['total_chunks'])
        st.metric("Caracteres", stats['total_characters'])
    
    with col2:
        st.metric("Palabras", stats['total_words'])
        st.metric("Longitud Promedio", f"{stats['avg_chunk_length']:.0f}")
    
    with col3:
        st.metric("Embeddings", "✅" if stats['has_embeddings'] else "❌")
        st.metric("Visión", "✅" if stats['vision_enabled'] else "❌")
    
    # Tipos de contenido
    if 'content_types' in stats and stats['content_types']:
        st.write("**Tipos de Contenido:**")
        for content_type, count in stats['content_types'].items():
            st.write(f"- {content_type}: {count}")

if __name__ == "__main__":
    main()