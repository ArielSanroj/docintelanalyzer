from typing import TypedDict, List
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import warnings
import logging
import uuid
import json

# Import our common modules
from database import init_db
from ocr_utils import extract_text_from_pdf, extract_text_from_url

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

class AgentState(TypedDict):
    query: str
    source_type: str  # "upload" or "url"
    confirmed_source: str  # File name or URL
    file_path: str  # For uploads
    doc_url: str  # For URLs
    doc_text: str
    language: str
    summaries: List[str]
    final_summary: str
    references: List[dict]

# Database initialization is now handled by the database module

llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0.1)

@tool
def scan_uploaded_pdf(file_path: str) -> dict:
    """Fetch and extract text from an uploaded PDF with OCR fallback."""
    return extract_text_from_pdf(file_path)

@tool
def fetch_regulation_text(url: str) -> dict:
    """Fetch and extract text from a URL (HTML or PDF), with OCR fallback for image-based PDFs."""
    return extract_text_from_url(url)

tools = [scan_uploaded_pdf, fetch_regulation_text]
# Note: agent_llm is defined but unused; consider using it in nodes or removing it
agent_llm = llm.bind_tools(tools)

# Nodes
def fetch_node(state: AgentState) -> AgentState:
    logger.debug(f"Entering fetch_node with state: {state}")
    new_state = state.copy()  # Create a new state to avoid modifying the original
    if state['source_type'] == "upload" and state.get("file_path"):
        try:
            file_path = state["file_path"]
            logger.debug(f"Opening uploaded file: {file_path}")
            result = scan_uploaded_pdf.invoke({"file_path": file_path})
            if "error" in result:
                new_state['doc_text'] = result['error']
                new_state['references'] = []
            else:
                new_state['doc_text'] = result['doc_text']
                new_state['references'] = [{
                    "code": "Ref1",
                    "file": result['file_path'],
                    "description": result['description']
                }]
        except Exception as e:
            error_msg = str(e)
            if "tesseract is not installed" in error_msg.lower() or "cannot open resource" in error_msg.lower():
                error_msg = "Tesseract OCR no está instalado o no se encuentran los paquetes de idioma 'spa' o 'eng'. Instale con 'brew install tesseract tesseract-lang' en PyCharm's Terminal."
            new_state['doc_text'] = f"Error: Error procesando archivo: {error_msg}"
            new_state['references'] = []
    elif state['source_type'] == "url":
        result = fetch_regulation_text.invoke({"url": state['confirmed_source']})
        if "error" in result:
            new_state['doc_text'] = result['error']
            new_state['references'] = []
        else:
            new_state['doc_url'] = result['doc_url']
            new_state['doc_text'] = result['doc_text']
            new_state['references'] = [{
                "code": "Ref1",
                "url": result['doc_url'],
                "description": result['description']
            }]
    else:
        new_state['doc_text'] = f"Error: Source type {state['source_type']} no soportado"
        new_state['references'] = []
    logger.debug(f"Exiting fetch_node with new_state: {new_state}")
    return new_state

def analyze_node(state: AgentState) -> AgentState:
    logger.debug(f"Entering analyze_node with state: {state}")
    new_state = state.copy()
    if not state.get('doc_text') or state['doc_text'].lower().startswith("error"):
        new_state['summaries'] = ["No se pudo analizar: documento no disponible o inválido para generar resumen ejecutivo."]
    else:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Genera un resumen ejecutivo profesional del documento. El resumen debe:

- Ser proporcional al tamaño del documento (corto si es corto, largo si es largo)
- Máximo 300 palabras
- Explicar de manera sucinta y ejecutiva qué dice el documento
- Usar un tono profesional y formal
- Ser claro y directo, sin secciones adicionales

Solo genera el resumen ejecutivo, sin puntos clave, riesgos, recomendaciones u otras secciones."""),
            HumanMessage(content=f"Documento: {state['doc_text'][:4000]}\nConsulta: {state['query']}\nIdioma: {state['language']}")
        ])
        response = (prompt | llm).invoke({})
        if response.content is None:
            logger.error("LLM returned None for analyze_node")
            new_state['summaries'] = ["Error: No se pudo generar el resumen ejecutivo - LLM no devolvió contenido"]
        else:
            new_state['summaries'] = [response.content]
    logger.debug(f"Exiting analyze_node with new_state: {new_state}")
    return new_state

def summarize_node(state: AgentState) -> AgentState:
    logger.debug(f"Entering summarize_node with state: {state}")
    new_state = state.copy()
    logger.debug(f"summaries before initialization: {new_state.get('summaries')}")
    if not new_state.get('summaries'):
        new_state['summaries'] = []
    logger.debug(f"summaries after initialization: {new_state['summaries']}")
    # Solo generar un resumen ejecutivo
    summary_type = "Resumen ejecutivo" if state['language'] == "es" else "Executive summary"
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""Genera un {summary_type.lower()} profesional en {state['language']} que:

- Sea proporcional al tamaño del documento (corto si es corto, largo si es largo)
- Máximo 300 palabras
- Explique de manera sucinta y ejecutiva qué dice el documento
- Use un tono profesional y formal
- Sea claro y directo

Solo genera el resumen ejecutivo, sin secciones adicionales."""),
        HumanMessage(content=f"Documento: {state['doc_text'][:4000]}\nConsulta: {state['query']}")
    ])
    response = (prompt | llm).invoke({})
    if response.content is None:
        logger.error(f"LLM returned None for {summary_type}")
        new_state['summaries'].append("Error: No se pudo generar el resumen ejecutivo - LLM no devolvió contenido")
    else:
        logger.debug(f"Appending response.content: {response.content}")
        new_state['summaries'].append(response.content)
    logger.debug(f"Exiting summarize_node with new_state: {new_state}")
    return new_state

def compiler_node(state: AgentState) -> AgentState:
    logger.debug(f"Entering compiler_node with state: {state}")
    new_state = state.copy()
    summaries_text = '\n'.join(state['summaries']) if state.get('summaries') else "No hay resúmenes disponibles debido a un error en el procesamiento del documento."
    references_text = "\n".join([
        f"{ref['code']}: {ref['description']} [{'Archivo' if 'file' in ref else 'URL'}: {ref.get('file', ref.get('url', ''))}]"
        for ref in state.get('references', [])
    ]) or "No hay referencias disponibles."
    # Include error message in report if doc_text indicates failure
    error_message = state['doc_text'] if state.get('doc_text', '').lower().startswith("error") else ""
    prompt_content = f"Resúmenes: {summaries_text}\nIdioma: {state['language']}\nReferencias:\n{references_text}"
    if error_message:
        prompt_content += f"\nMensaje de Error: {error_message}"
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Compila el resumen ejecutivo en un informe final profesional. El informe debe:

- Ser claro y directo
- Usar un tono profesional y formal
- Máximo 300 palabras
- Explicar de manera sucinta qué dice el documento
- Si hay errores, inclúyelos con explicación clara
- Añadir: 'Resumen generado por IA; consulte a un experto financiero o legal para validez.'

Solo genera el resumen ejecutivo final, sin secciones adicionales."""),
        HumanMessage(content=prompt_content)
    ])
    response = (prompt | llm).invoke({})
    if response.content is None:
        logger.error("LLM returned None for compiler_node")
        new_state['final_summary'] = f"Error: No se pudo generar el resumen ejecutivo final. Mensaje de error: {error_message or 'LLM no devolvió contenido.'}\nResumen generado por IA; consulte a un experto financiero o legal para validez."
    else:
        new_state['final_summary'] = response.content
    logger.debug(f"Exiting compiler_node with new_state: {new_state}")
    return new_state

def router_node(state: AgentState) -> str:
    logger.debug(f"Entering router_node with state: {state}")
    logger.debug(f"doc_text: {state.get('doc_text')}")
    if not state.get('doc_text'):
        logger.debug("Routing to fetch due to missing doc_text")
        return "fetch"
    if isinstance(state['doc_text'], str) and state['doc_text'].lower().startswith("error"):
        logger.debug("Error in doc_text; routing to compiler to produce graceful report")
        return "compiler"
    if not state.get('summaries') or len(state['summaries']) == 0:
        logger.debug("Routing to analyze due to empty summaries")
        return "analyze"
    if len(state.get('summaries', [])) < 3:
        logger.debug("Routing to summarize due to insufficient summaries")
        return "summarize"
    logger.debug("Routing to compiler as all conditions met")
    return "compiler"

# Build the workflow
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("fetch", fetch_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("compiler", compiler_node)
workflow.set_entry_point("fetch")
workflow.add_conditional_edges("fetch", router_node, {"fetch": "fetch", "analyze": "analyze", "summarize": "summarize", "compiler": "compiler", END: END})
workflow.add_conditional_edges("analyze", router_node, {"fetch": "fetch", "analyze": "analyze", "summarize": "summarize", "compiler": "compiler", END: END})
workflow.add_conditional_edges("summarize", router_node, {"fetch": "fetch", "analyze": "analyze", "summarize": "summarize", "compiler": "compiler", END: END})
workflow.add_conditional_edges("compiler", lambda s: END)
app = workflow.compile()

# This module only contains the workflow logic
# The Streamlit UI is in app.py