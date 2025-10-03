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
from docsreview import workflow
from rag_system import RAGSystem
from rag_llm_integration import RAGLLMIntegration

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Debug: Confirm NVIDIA API key
try:
    import streamlit as st
    nvidia_key = os.getenv('NVIDIA_API_KEY') or st.secrets.get('NVIDIA_API_KEY')
except:
    nvidia_key = os.getenv('NVIDIA_API_KEY')
print(f"NVIDIA_API_KEY loaded: {'Yes' if nvidia_key else 'No'}")

# Initialize database
init_db()

# Lovable integration removed from this deployment. All related UI and env var handling has been disabled.
lovable_integration = None

# All workflow logic is now imported from docsreview.py

# Streamlit UI
if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(layout="wide")
    st.title("Generador de Resúmenes Ejecutivos")
    st.write("Suba un PDF o ingrese una URL para generar un resumen ejecutivo profesional. Fecha: 26 de septiembre de 2025, 06:16 AM -05.")
    
    # Lovable Integration Status
    if lovable_integration:
        st.success("🔗 Integración con Lovable activa")
    else:
        st.info("ℹ️ Integración con Lovable no disponible (falta API key)")

    # Lovable Project Management
    if lovable_integration:
        with st.expander("🔗 Gestión de Proyectos Lovable", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                project_name = st.text_input("Nombre del Proyecto", value="Análisis de Documentos", key="lovable_project")
                if st.button("Crear/Usar Proyecto"):
                    try:
                        project = lovable_integration.setup_project(project_name, "Proyecto de análisis de documentos")
                        st.success(f"Proyecto configurado: {project.name}")
                        st.session_state.lovable_project = project
                    except Exception as e:
                        st.error(f"Error configurando proyecto: {str(e)}")
            
            with col2:
                if st.button("Ver Dashboard"):
                    try:
                        if hasattr(st.session_state, 'lovable_project') and st.session_state.lovable_project:
                            dashboard_data = lovable_integration.get_project_dashboard_data()
                            st.write("**Dashboard del Proyecto:**")
                            st.write(f"- Total documentos: {dashboard_data['total_documents']}")
                            st.write(f"- Análisis: {dashboard_data['analysis_documents']}")
                            st.write(f"- Historiales de chat: {dashboard_data['chat_histories']}")
                        else:
                            st.warning("Primero configure un proyecto")
                    except Exception as e:
                        st.error(f"Error obteniendo dashboard: {str(e)}")

    # Input fields
    query = st.text_input("Keywords de búsqueda (opcional)", placeholder="Ej: contrato, términos, cláusulas, etc.")
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
    if st.button("🔍 Generar Resumen Ejecutivo"):
        if not confirmed_source:
            st.error("Por favor, suba un archivo o ingrese una URL.")
        else:
            try:
                # Use filename as query if no query provided
                if not query.strip():
                    query = os.path.splitext(confirmed_source)[0] if confirmed_source else "Documento sin título"
                
                # Prepare initial state
                import database  # Import the module here to avoid undefined name error
                initial_state = database.AgentState(
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
                
                # Upload to Lovable if integration is available
                if lovable_integration and hasattr(st.session_state, 'lovable_project') and st.session_state.lovable_project:
                    try:
                        # Format data for Lovable
                        import database  # Import the module here to avoid undefined name error
                        lovable_data = format_document_for_lovable(
                            document_name=confirmed_source,
                            summary=result['final_summary'],
                            references=result.get('references', []),
                            chat_history=[]  # Will be updated when chat is used
                        )
                        
                        # Upload analysis result
                        lovable_doc = lovable_integration.upload_analysis_result(
                            document_name=confirmed_source,
                            summary=result['final_summary'],
                            references=result.get('references', []),
                            metadata={
                                "report_id": report_id,
                                "query": query,
                                "language": language,
                                "source_type": source_type,
                                "analysis_timestamp": datetime.now().isoformat()
                            }
                        )
                        
                        st.success(f"📤 Análisis subido a Lovable: {lovable_doc.name}")
                        
                        # Store Lovable document ID for later chat sync
                        st.session_state.lovable_document_id = lovable_doc.id
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error subiendo a Lovable: {str(e)}")
                        logger.error(f"Lovable upload error: {e}")
                elif lovable_integration:
                    st.info("💡 Configure un proyecto Lovable para subir automáticamente los análisis")

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
        st.subheader("💬 Chat sobre el Documento")
        st.write("Puede hacer preguntas sobre el documento analizado:")
        
        # Debug info (can be removed in production)
        with st.expander("🔍 Información de Debug (Click para ver)"):
            doc_text_length = len(st.session_state.get('current_doc_text', ''))
            st.write(f"**Longitud del texto del documento:** {doc_text_length} caracteres")
            if doc_text_length > 0:
                st.write(f"**Vista previa del documento:** {st_session_state.get('current_doc_text', '')[:500]}...")
                st.write("**¿El texto contiene 'GMF'?**", "Sí" if "GMF" in st_session_state.get('current_doc_text', '').upper() else "No")
                st.write("**¿El texto contiene 'ahorro'?**", "Sí" if "ahorro" in st_session_state.get('current_doc_text', '').lower() else "No")
                
                # Generic debug: Check if first word of last prompt exists in document
                if st_session_state.chat_history:
                    last_prompt = st_session_state.chat_history[-1]["content"] if st_session_state.chat_history[-1]["role"] == "user" else ""
                    if last_prompt:
                        first_word = last_prompt.split()[0] if last_prompt.split() else ""
                        if first_word:
                            st.write(f"**Contiene '{first_word}':**", "Sí" if first_word.upper() in st_session_state.get('current_doc_text', '').upper() else "No")
            else:
                st.warning("⚠️ No se encontró texto del documento en la sesión")
        
        # Initialize chat history and RAG system
        if 'chat_history' not in st_session_state:
            st_session_state.chat_history = []
        
        # Mostrar información del documento actual
        current_doc = st_session_state.get('current_doc_text', '')
        if current_doc:
            doc_source = st_session_state.get('confirmed_source', 'Documento actual')
            st.info(f"📄 **Documento activo:** {doc_source} ({len(current_doc)} caracteres)")
        else:
            st.warning("⚠️ No hay documento cargado. Genere un resumen ejecutivo para poder hacer preguntas.")
        
        # Initialize RAG system if document is available
        doc_text = st_session_state.get('current_doc_text', '')
        if doc_text and 'rag_system' not in st_session_state:
            with st.spinner("Inicializando sistema RAG para el documento actual..."):
                try:
                    # Crear nuevo sistema RAG
                    st_session_state.rag_system = RAGSystem()
                    
                    # Procesar el documento actual con nuevo default
                    st_session_state.rag_system.process_document(doc_text, chunk_size=1500)
                    
                    # Crear integración RAG + LLM
                    st_session_state.rag_llm_integration = RAGLLMIntegration(llm, st_session_state.rag_system)
                    
                    # Mostrar estadísticas del documento procesado
                    stats = st_session_state.rag_system.get_document_stats()
                    st.success(f"✅ Sistema RAG inicializado correctamente")
                    st.info(f"📄 Documento procesado: {stats['total_chunks']} chunks, {stats['total_words']} palabras")
                    
                    logger.info(f"RAG system initialized for document with {stats['total_chunks']} chunks")
                    
                except Exception as e:
                    st.error(f"❌ Error inicializando RAG: {str(e)}")
                    logger.error(f"RAG initialization error: {e}")
                    # Limpiar estado en caso de error
                    if 'rag_system' in st_session_state:
                        del st_session_state.rag_system
                    if 'rag_llm_integration' in st_session_state:
                        del st_session_state.rag_llm_integration
        
        # Display chat history
        for message in st_session_state.chat_history:
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
            st_session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate AI response using RAG + LLM
            with st.chat_message("assistant"):
                with st.spinner("Analizando documento y generando respuesta..."):
                    try:
                        # Check if RAG system is available
                        if 'rag_llm_integration' not in st_session_state:
                            st.error("⚠️ Sistema RAG no inicializado. Por favor, genere un resumen ejecutivo primero.")
                            st_session_state.chat_history.append({
                                "role": "assistant", 
                                "content": "❌ Error: Sistema RAG no disponible. Genere un resumen ejecutivo primero para poder hacer preguntas sobre el documento."
                            })
                        else:
                            # Verificar que el documento actual esté disponible
                            current_doc = st_session_state.get('current_doc_text', '')
                            if not current_doc:
                                st.error("⚠️ No hay documento cargado. Por favor, genere un resumen ejecutivo primero.")
                                st_session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": "❌ Error: No hay documento cargado. Genere un resumen ejecutivo primero."
                                })
                            else:
                                # Use RAG + LLM integration
                                rag_integration = st_session_state.rag_llm_integration
                                conversation_context = st_session_state.chat_history[-6:] if len(st_session_state.chat_history) > 6 else st_session_state.chat_history
                                
                                # Generate enhanced response
                                rag_response = rag_integration.generate_response(prompt, conversation_context)
                                
                                # Display the answer
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
                                st_session_state.chat_history.append({
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
                                
                                # Sync chat history to Lovable if available
                                if lovable_integration and hasattr(st_session_state, 'lovable_project') and st_session_state.lovable_project:
                                    try:
                                        # Sync current chat history
                                        lovable_integration.sync_chat_history(
                                            chat_history=st_session_state.chat_history,
                                            document_name=st_session_state.get('confirmed_source', 'Documento actual')
                                        )
                                    except Exception as e:
                                        logger.warning(f"Failed to sync chat to Lovable: {e}")
                            
                    except Exception as e:
                        error_msg = f"Error al generar respuesta: {str(e)}"
                        st.write(error_msg)
                        st_session_state.chat_history.append({"role": "assistant", "content": error_msg})
                        
                        # Debug: Log the error details
                        logger.error(f"RAG Chat error: {str(e)}")
                        logger.error(f"Document text available: {len(st_session_state.get('current_doc_text', ''))} chars")
        
        # Debug and clear buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🗑️ Limpiar Chat"):
                st_session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st_button("🔄 Reiniciar RAG"):
                if 'rag_system' in st_session_state:
                    del st_session_state.rag_system
                if 'rag_llm_integration' in st_session_state:
                    del st_session_state.rag_llm_integration
                st.success("Sistema RAG reiniciado")
                st.rerun()
        
        with col3:
            if st_button("📊 Info RAG"):
                if 'rag_system' in st_session_state:
                    stats = st_session_state.rag_system.get_document_stats()
                    st.write("**Estadísticas del documento:**")
                    for key, value in stats.items():
                        st.write(f"- {key}: {value}")
                    
                    if 'rag_llm_integration' in st_session_state:
                        conv_summary = st_session_state.rag_llm_integration.get_conversation_summary()
                        st.write("**Resumen de conversación:**")
                        for key, value in conv_summary.items():
                            st.write(f"- {key}: {value}")
                else:
                    st.write("Sistema RAG no inicializado")
        
        with col4:
            if lovable_integration and hasattr(st_session_state, 'lovable_project') and st_session_state.lovable_project:
                if st_button("📤 Sincronizar Chat"):
                    try:
                        lovable_integration.sync_chat_history(
                            chat_history=st_session_state.chat_history,
                            document_name=st_session_state.get('confirmed_source', 'Documento actual')
                        )
                        st.success("Chat sincronizado con Lovable")
                    except Exception as e:
                        st.error(f"Error sincronizando: {str(e)}")
            else:
                st_button("📤 Sincronizar Chat", disabled=True, help="Configure Lovable primero")
        
        # Legacy debug button
        if st_button("🔍 Ver Prompt Completo (Legacy)"):
            doc_text = st_session_state.get('current_doc_text', '')
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
