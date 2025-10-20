import streamlit as st
from dotenv import load_dotenv
import os
import uuid
import json
import warnings
import logging
from datetime import datetime

# Import our common modules
from database import init_db, save_report, delete_report
from docsreview import workflow, AgentState
from rag_system import RAGSystem
from rag_llm_integration import RAGLLMIntegration
from advanced_rag_system import AdvancedRAGSystem
from advanced_rag_integration import AdvancedRAGLLMIntegration
from ingredient_renderer import process_ingredient_response
from llm_fallback import get_llm
from llm_helper import create_llm_helper

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Debug: Confirm Ollama API key
ollama_key = os.getenv('OLLAMA_API_KEY') or st.secrets.get('OLLAMA_API_KEY')
print(f"OLLAMA_API_KEY loaded: {'Yes' if ollama_key else 'No'}")

# Initialize database
init_db()

# Initialize LLM
llm = get_llm()


# All workflow logic is now imported from docsreview.py

# Streamlit UI
if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(layout="wide")
    st.markdown("# Bienvenido a DocIntelAnnalyzer")
    st.markdown("## Generador de Resúmenes Ejecutivos")
    st.markdown("### Tu Propia Base de Datos")
    st.write("Sube un PDF o ingresa una URL para comenzar")
    

    # Input fields
    language = st.radio("Idioma", ["es", "en"], index=0)
    source_type = st.radio("Tipo de Fuente", ["Subir archivo", "URL"], index=0)

    if source_type == "Subir archivo":
        uploaded_file = st.file_uploader("Seleccionar Archivo PDF", type="pdf")
        confirmed_source = uploaded_file.name if uploaded_file else None
        file_path = f"/tmp/{uploaded_file.name}" if uploaded_file else None
        if uploaded_file:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        source_type = "upload"  # Convert to internal format
    else:
        confirmed_source = st.text_input("URL del Documento", placeholder="e.g., https://example.com/doc.pdf")
        file_path = None
        source_type = "url"  # Convert to internal format

    # Generate button
    st.markdown("#### Generar Resumen Ejecutivo")
    if st.button("🔍 Generar Resumen Ejecutivo"):
        if not confirmed_source:
            st.error("Por favor, suba un archivo o ingrese una URL.")
        else:
            try:
                # Use filename as query
                query = os.path.splitext(confirmed_source)[0] if confirmed_source else "Documento sin título"
                
                # Prepare initial state
                initial_state = AgentState(
                    query=query,
                    source_type=source_type,
                    confirmed_source=confirmed_source,
                    language=language,
                    references=[],
                    file_path=file_path,
                    doc_url=confirmed_source if source_type == "url" else None,
                    doc_text="",  # Will be populated by fetch_node
                    summaries=[],  # Initialize summaries to avoid NoneType error
                )

                # Invoke workflow with recursion limit
                app = workflow.compile()
                result = app.invoke(initial_state, config={"recursion_limit": 50})

                # Display results
                st.subheader("Resumen Ejecutivo")
                st.write(result['final_summary'])

                st.subheader("Referencias")
                references = result.get('references', [])
                if references:
                    for ref in references:
                        st.write(f"- {ref['code']}: {ref['description']} [{'Archivo' if 'file' in ref else 'URL'}: {ref.get('file', ref.get('url', ''))}]")
                else:
                    st.write("No hay referencias disponibles.")

                # Save to database
                report_id = str(uuid.uuid4())
                if save_report(report_id, query, source_type, confirmed_source, language, result['final_summary'], result.get('references', [])):
                    st.success(f"Resumen ejecutivo guardado con ID: {report_id}")
                else:
                    st.error("Error al guardar el informe en la base de datos")
                

                # Store in session state for display
                if 'reports' not in st.session_state:
                    st.session_state.reports = []
                st.session_state.reports.append({
                    'id': report_id,
                    'query': query,
                    'source_type': source_type,
                    'confirmed_source': confirmed_source,
                    'language': language,
                    'final_summary': result['final_summary'],
                    'references': references
                })
                
                # Store document text for chat
                st.session_state.current_doc_text = result.get('doc_text', '')
                st.session_state.current_summary = result['final_summary']
                
                # Reiniciar sistema RAG para el nuevo documento
                if 'rag_system' in st.session_state:
                    del st.session_state.rag_system
                if 'rag_llm_integration' in st.session_state:
                    del st.session_state.rag_llm_integration
                
                # Limpiar historial de chat para el nuevo documento
                if 'chat_history' in st.session_state:
                    st.session_state.chat_history = []
                
                # Debug: Log that document text is being stored
                logger.debug(f"Storing document text for chat. Length: {len(result.get('doc_text', ''))}")
                logger.debug(f"Document text preview: {result.get('doc_text', '')[:200]}...")
                logger.info("Sistema RAG reiniciado para nuevo documento")

            except Exception as e:
                st.error(f"Error al generar el resumen ejecutivo: {str(e)}")

    # Display stored reports
    if 'reports' in st.session_state and st.session_state.reports:
        st.subheader("Resúmenes Ejecutivos Almacenados")
        for i, report in enumerate(st.session_state.reports):
            with st.expander(f"Resumen {i + 1} - {report['query']} (ID: {report['id']})"):
                st.write(f"**Consulta:** {report['query']}")
                st.write(f"**Fuente:** {report['confirmed_source']}")
                st.write(f"**Idioma:** {report['language']}")
                st.write(f"**Resumen:** {report['final_summary']}")
                st.write("**Referencias:**")
                for ref in report['references']:
                    st.write(f"- {ref['code']}: {ref['description']} [{'Archivo' if 'file' in ref else 'URL'}: {ref.get('file', ref.get('url', ''))}]")
                if st.button("Eliminar", key=f"delete_{i}"):
                    if delete_report(report['id']):
                        st.session_state.reports.pop(i)
                        st.rerun()
                    else:
                        st.error("Error al eliminar el informe")

    # Chat interface
    if 'current_summary' in st.session_state and st.session_state.current_summary:
        st.markdown("#### Chatea con tu propia informacion")
        st.write("Puede hacer preguntas sobre el documento analizado:")
        
        # Initialize chat history and RAG system first
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Debug info (can be removed in production)
        with st.expander("🔍 Información de Debug (Click para ver)"):
            doc_text_length = len(st.session_state.get('current_doc_text', ''))
            st.write(f"**Longitud del texto del documento:** {doc_text_length} caracteres")
            if doc_text_length > 0:
                st.write(f"**Vista previa del documento:** {st.session_state.get('current_doc_text', '')[:500]}...")
                st.write("**¿El texto contiene 'GMF'?**", "Sí" if "GMF" in st.session_state.get('current_doc_text', '').upper() else "No")
                st.write("**¿El texto contiene 'ahorro'?**", "Sí" if "ahorro" in st.session_state.get('current_doc_text', '').lower() else "No")
                
                # Generic debug: Check if first word of last prompt exists in document
                if st.session_state.chat_history:
                    last_prompt = st.session_state.chat_history[-1]["content"] if st.session_state.chat_history[-1]["role"] == "user" else ""
                    if last_prompt:
                        first_word = last_prompt.split()[0] if last_prompt.split() else ""
                        if first_word:
                            st.write(f"**Contiene '{first_word}':**", "Sí" if first_word.upper() in st.session_state.get('current_doc_text', '').upper() else "No")
            else:
                st.warning("⚠️ No se encontró texto del documento en la sesión")
        
        # Mostrar información del documento actual
        current_doc = st.session_state.get('current_doc_text', '')
        if current_doc:
            doc_source = st.session_state.get('confirmed_source', 'Documento actual')
            st.info(f"📄 **Documento activo:** {doc_source} ({len(current_doc)} caracteres)")
        else:
            st.warning("⚠️ No hay documento cargado. Genere un resumen ejecutivo para poder hacer preguntas.")
        
        # Initialize RAG system if document is available
        doc_text = st.session_state.get('current_doc_text', '')
        
        if doc_text and 'rag_system' not in st.session_state:
            with st.spinner("🚀 Inicializando Sistema RAG (esto puede tomar unos segundos la primera vez)..."):
                try:
                    # Crear nuevo sistema RAG
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📥 Descargando modelos de embeddings...")
                    progress_bar.progress(15)
                    
                    st.session_state.rag_system = RAGSystem()
                    progress_bar.progress(30)
                    
                    status_text.text("📄 Procesando documento con chunking optimizado...")
                    progress_bar.progress(60)
                    
                    st.session_state.rag_system.process_document(doc_text, chunk_size=1500)
                    status_text.text("🔗 Configurando integración LLM...")
                    progress_bar.progress(80)
                    st.session_state.rag_llm_integration = RAGLLMIntegration(llm, st.session_state.rag_system)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Sistema RAG inicializado correctamente")
                    
                    # Limpiar indicadores de progreso
                    import time
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Mostrar estadísticas del documento procesado
                    stats = st.session_state.rag_system.get_document_stats()
                    st.success("✅ Sistema RAG inicializado correctamente")
                    st.info(f"📄 Documento procesado: {stats['total_chunks']} chunks, {stats['total_words']} palabras")
                    st.info("🤖 Modelo usado: all-MiniLM-L6-v2 (optimizado para velocidad)")
                    
                    logger.info(f"RAG system initialized for document with {stats['total_chunks']} chunks")
                    
                except Exception as e:
                    st.error(f"❌ Error inicializando RAG: {str(e)}")
                    logger.error(f"RAG initialization error: {e}")
                    # Limpiar estado en caso de error
                    if 'rag_system' in st.session_state:
                        del st.session_state.rag_system
                    if 'rag_llm_integration' in st.session_state:
                        del st.session_state.rag_llm_integration
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show additional info for assistant messages if available
                if message["role"] == "assistant" and "rag_info" in message:
                    with st.expander("📊 Información de recuperación"):
                        rag_info = message["rag_info"]
                        st.write(f"**Confianza:** {rag_info['confidence_score']:.2f}")
                        st.write(f"**Chunks encontrados:** {rag_info['retrieval_info']['chunks_found']}")
                        st.write(f"**Tipo de consulta:** {rag_info['retrieval_info']['query_type']}")
                        st.write(f"**Relevancia promedio:** {rag_info['retrieval_info']['avg_relevance_score']:.3f}")
        
        # Chat input
        if prompt := st.chat_input("Haga una pregunta sobre el documento..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate AI response using RAG + LLM
            with st.chat_message("assistant"):
                with st.spinner("Analizando documento y generando respuesta..."):
                    try:
                        # Check if RAG system is available
                        if 'rag_llm_integration' not in st.session_state:
                            st.error("⚠️ Sistema RAG no inicializado. Por favor, genere un resumen ejecutivo primero.")
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": "❌ Error: Sistema RAG no disponible. Genere un resumen ejecutivo primero para poder hacer preguntas sobre el documento."
                            })
                        else:
                            # Verificar que el documento actual esté disponible
                            current_doc = st.session_state.get('current_doc_text', '')
                            if not current_doc:
                                st.error("⚠️ No hay documento cargado. Por favor, genere un resumen ejecutivo primero.")
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": "❌ Error: No hay documento cargado. Genere un resumen ejecutivo primero."
                                })
                            else:
                                # Use RAG + LLM integration
                                rag_integration = st.session_state.rag_llm_integration
                                conversation_context = st.session_state.chat_history[-6:] if len(st.session_state.chat_history) > 6 else st.session_state.chat_history
                                
                                # Generate response using RAG integration
                                rag_response = rag_integration.generate_response(prompt, conversation_context)
                                
                                # Display the answer
                                if rag_response.answer and rag_response.answer != "No se pudo generar respuesta":
                                    # Procesar respuesta para ingredientes
                                    processed_answer = process_ingredient_response(rag_response.answer)
                                    if processed_answer.strip():
                                        st.write(processed_answer)
                                else:
                                    # Si el agente ejecutó herramientas pero no generó respuesta útil
                                    if hasattr(rag_response, 'retrieval_info') and rag_response.retrieval_info.get('tool_calls', 0) > 0:
                                        st.info("🤖 El agente encontró información en los documentos pero necesita una pregunta más específica. Intenta reformular tu consulta.")
                                        st.write("**Información encontrada:** El agente ejecutó herramientas de búsqueda y encontró contenido relevante.")
                                    else:
                                        st.write(rag_response.answer)
                                
                                # Show confidence warning if low
                                if rag_response.confidence_score < 0.5:
                                    st.warning("⚠️ Respuesta basada en chunks parciales; doc grande, considera query más específica.")
                                
                                # Show relevant chunks in expandable section
                                if rag_response.relevant_chunks:
                                    with st.expander("🔍 Información relevante encontrada"):
                                        for i, chunk in enumerate(rag_response.relevant_chunks):
                                            st.write(f"**Fragmento {i+1}:**")
                                            st.write(chunk)
                                            st.write("---")
                                
                                
                                # Update conversation history with RAG info
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": rag_response.answer,
                                    "rag_info": {
                                        "confidence_score": rag_response.confidence_score,
                                        "retrieval_info": rag_response.retrieval_info,
                                        "relevant_chunks": rag_response.relevant_chunks
                                    }
                                })
                                
                                # Update RAG integration conversation history
                                rag_integration.update_conversation_history(prompt, rag_response)
                                
                            
                    except Exception as e:
                        error_msg = f"Error al generar respuesta: {str(e)}"
                        st.write(error_msg)
                        
                        # Mostrar información de debug útil
                        with st.expander("🔍 Información de Debug del Error"):
                            st.write(f"**Error:** {str(e)}")
                            st.write(f"**Tipo de error:** {type(e).__name__}")
                            st.write(f"**Documento disponible:** {len(st.session_state.get('current_doc_text', ''))} caracteres")
                            st.write(f"**Sistema RAG disponible:** {'rag_system' in st.session_state}")
                            st.write(f"**Integración RAG disponible:** {'rag_llm_integration' in st.session_state}")
                            
                            if 'rag_system' in st.session_state:
                                try:
                                    stats = st.session_state.rag_system.get_document_stats()
                                    st.write(f"**Estadísticas RAG:** {stats}")
                                except Exception as stats_error:
                                    st.write(f"**Error obteniendo stats:** {stats_error}")
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                        
                        # Debug: Log the error details
                        logger.error(f"RAG Chat error: {str(e)}")
                        logger.error(f"Document text available: {len(st.session_state.get('current_doc_text', ''))} chars")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Debug and clear buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🗑️ Limpiar Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Reiniciar RAG"):
                if 'rag_system' in st.session_state:
                    del st.session_state.rag_system
                if 'rag_llm_integration' in st.session_state:
                    del st.session_state.rag_llm_integration
                st.success("Sistema RAG reiniciado")
                st.rerun()
        
        with col3:
            if st.button("📊 Info RAG"):
                if 'rag_system' in st.session_state:
                    stats = st.session_state.rag_system.get_document_stats()
                    st.write("**Estadísticas del documento:**")
                    for key, value in stats.items():
                        st.write(f"- {key}: {value}")
                    
                    if 'rag_llm_integration' in st.session_state:
                        conv_summary = st.session_state.rag_llm_integration.get_conversation_summary()
                        st.write("**Resumen de conversación:**")
                        for key, value in conv_summary.items():
                            st.write(f"- {key}: {value}")
                else:
                    st.write("Sistema RAG no inicializado")
        
        with col4:
            if st.button("📊 Ver Estadísticas"):
                if 'rag_system' in st.session_state:
                    stats = st.session_state.rag_system.get_document_stats()
                    st.write("**Estadísticas del documento:**")
                    for key, value in stats.items():
                        st.write(f"- {key}: {value}")
                else:
                    st.write("Sistema RAG no inicializado")
        
        
        # Legacy debug button
        if st.button("🔍 Ver Prompt Completo (Legacy)"):
            doc_text = st.session_state.get('current_doc_text', '')
            doc_text_sample = doc_text[:5000] if len(doc_text) > 5000 else doc_text
            
            debug_prompt = f"""Eres un asistente de análisis de documentos. Tu única función es extraer y presentar información del documento proporcionado.

CONTENIDO DEL DOCUMENTO:
{doc_text_sample}

PREGUNTA: como ahorramos el gmf

TAREA: Busca en el contenido del documento cualquier información relacionada con la pregunta y responde basándote ÚNICAMENTE en esa información.

REGLAS:
- Si encuentras información relevante en el documento, responde con esa información específica
- Si NO encuentras información relevante, di "No encontré información sobre esto en el documento"
- NO digas que no puedes dar asesoramiento - tu trabajo es solo presentar la información del documento
- Responde en español

Información encontrada en el documento:"""
            
            st.text_area("Prompt que se envía al LLM:", debug_prompt, height=300)
