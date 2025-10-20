"""
Sistema RAG Avanzado basado en las mejores prácticas de NVIDIA Nemotron.
Implementa ReAct Agent, reranking, y búsqueda híbrida.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from langgraph.prebuilt import create_react_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
# Ollama integration - using local embeddings and LLM
from llm_fallback import get_llm
from langchain_core.documents import Document
from langchain_core.tools import Tool

# Optional NVIDIA imports
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    print("NVIDIA AI endpoints not available. Install with: pip install langchain-nvidia-ai-endpoints")

logger = logging.getLogger(__name__)

# BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not available. Install with: pip install rank_bm25 for hybrid search")

@dataclass
class DocumentChunk:
    """Representa un fragmento del documento con metadatos."""
    text: str
    start_pos: int
    end_pos: int
    chunk_id: int
    metadata: Dict = None
    embedding: Optional[np.ndarray] = None

@dataclass
class RetrievalResult:
    """Resultado de la recuperación de información."""
    chunks: List[DocumentChunk]
    scores: List[float]
    query: str
    total_chunks_searched: int
    retrieval_type: str  # "semantic", "keyword", "hybrid"

class AdvancedRAGSystem:
    """Sistema RAG Avanzado con ReAct Agent y reranking."""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 rerank_model: str = "local-reranker",
                 llm_model: str = "llama3.1:8b"):
        """
        Inicializa el sistema RAG avanzado.
        
        Args:
            embedding_model: Modelo de embeddings
            rerank_model: Modelo de reranking (NVIDIA si está disponible)
            llm_model: Modelo LLM principal (usando Ollama)
        """
        self.embedding_model_name = embedding_model
        self.rerank_model_name = rerank_model
        self.llm_model_name = llm_model
        
        # Componentes del sistema
        self.embedding_model = None
        self.embedding_function = None
        self.nvidia_embeddings = None
        self.reranker = None
        self.llm = None
        self.vector_store = None
        self.bm25_index = None
        self.document_chunks: List[DocumentChunk] = []
        self.chunk_embeddings: np.ndarray = None
        
        # Configuración optimizada basada en NVIDIA
        self.chunk_size = 800
        self.chunk_overlap = 120
        self.top_k = 6  # Número de documentos a recuperar inicialmente
        self.vector_store_type = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa todos los modelos necesarios."""
        try:
            logger.info("Inicializando modelos avanzados...")
            
            # Modelo de embeddings con caché
            cache_folder = os.path.expanduser("~/.cache/sentence_transformers")
            os.makedirs(cache_folder, exist_ok=True)
            
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                cache_folder=cache_folder
            )
            try:
                self.embedding_function = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)
            except Exception as embedding_err:
                logger.error(f"Error inicializando embeddings de LangChain: {embedding_err}")
                raise

            if self.embedding_model_name.startswith("nvidia/") and NVIDIA_AVAILABLE:
                try:
                    self.nvidia_embeddings = NVIDIAEmbeddings(model=self.embedding_model_name)
                    logger.info("Embeddings NVIDIA inicializados")
                except Exception as e:
                    logger.warning(f"No se pudo inicializar NVIDIA embeddings: {e}")
                    self.nvidia_embeddings = None
            elif self.embedding_model_name.startswith("nvidia/") and not NVIDIA_AVAILABLE:
                logger.warning("NVIDIA embeddings no disponibles, usando embeddings locales")
                self.nvidia_embeddings = None
            
            # Modelo de reranking de NVIDIA (si está disponible)
            if NVIDIA_AVAILABLE:
                try:
                    self.reranker = NVIDIARerank(model=self.rerank_model_name)
                    logger.info("Modelo de reranking NVIDIA inicializado")
                except Exception as e:
                    logger.warning(f"No se pudo inicializar el reranker NVIDIA: {e}")
                    self.reranker = None
            else:
                logger.info("NVIDIA reranker no disponible, usando solo búsqueda semántica")
                self.reranker = None
            
            # LLM principal - usar Ollama por defecto
            try:
                self.llm = get_llm()
                logger.info("LLM Ollama inicializado correctamente")
            except Exception as e:
                logger.error(f"No se pudo inicializar el LLM Ollama: {e}")
                raise
            
            logger.info("Modelos avanzados inicializados correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando modelos: {e}")
            raise
    
    def process_document(self, text: str, chunk_size: int = None, chunk_overlap: int = None):
        """
        Procesa un documento con chunking optimizado.
        
        Args:
            text: Texto del documento
            chunk_size: Tamaño de los chunks (opcional)
            chunk_overlap: Solapamiento entre chunks (opcional)
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap

        estimated_length = len(text)
        if estimated_length < 1_500:
            chunk_size = max(400, estimated_length)
            chunk_overlap = min(int(chunk_size * 0.2), chunk_overlap)
        elif estimated_length < 8_000:
            chunk_size = min(max(chunk_size, 700), 900)
            chunk_overlap = min(int(chunk_size * 0.25), max(chunk_overlap, 150))
        elif estimated_length > 20_000:
            chunk_size = max(chunk_size, 1_000)
            chunk_overlap = max(int(chunk_size * 0.2), chunk_overlap)
        
        logger.info(f"Procesando documento con chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # Usar RecursiveCharacterTextSplitter como recomienda NVIDIA
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Dividir el texto en chunks
        text_chunks = splitter.split_text(text)
        
        # Crear objetos DocumentChunk
        self.document_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                text=chunk_text,
                start_pos=i * chunk_size,
                end_pos=(i + 1) * chunk_size,
                chunk_id=i,
                metadata={"chunk_id": i, "length": len(chunk_text)}
            )
            self.document_chunks.append(chunk)
        
        # Generar embeddings
        self._generate_embeddings()
        
        # Crear índice BM25 para búsqueda híbrida
        self._create_bm25_index()
        
        # Crear vector store
        self._create_vector_store()
        
        logger.info(f"Documento procesado: {len(self.document_chunks)} chunks")
    
    def _generate_embeddings(self):
        """Genera embeddings para todos los chunks."""
        logger.info("Generando embeddings...")
        
        texts = [chunk.text for chunk in self.document_chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        self.chunk_embeddings = embeddings
        
        # Asignar embeddings a los chunks
        for i, chunk in enumerate(self.document_chunks):
            chunk.embedding = embeddings[i]
        
        logger.info(f"Embeddings generados: {embeddings.shape}")
    
    def _create_bm25_index(self):
        """Crea índice BM25 para búsqueda por palabras clave."""
        if not BM25_AVAILABLE:
            logger.warning("BM25 no disponible, saltando índice BM25")
            return
        
        logger.info("Creando índice BM25...")
        
        # Tokenizar textos para BM25
        tokenized_texts = []
        for chunk in self.document_chunks:
            # Tokenización simple (se puede mejorar)
            tokens = re.findall(r'\b\w+\b', chunk.text.lower())
            tokenized_texts.append(tokens)
        
        self.bm25_index = BM25Okapi(tokenized_texts)
        logger.info("Índice BM25 creado")
    
    def _create_vector_store(self):
        """Crea el vector store usando FAISS o Chroma como fallback."""
        logger.info("Creando vector store...")
        
        documents = [
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in self.document_chunks
        ]

        if not documents:
            logger.warning("No hay documentos para indexar, omitiendo vector store")
            self.vector_store = None
            self.vector_store_type = None
            return

        if self.embedding_function is None:
            logger.error("Función de embeddings de LangChain no inicializada")
            self.vector_store = None
            self.vector_store_type = None
            return

        # Intentar usar FAISS con embeddings NVIDIA si están disponibles
        if self.nvidia_embeddings is not None:
            try:
                self.vector_store = FAISS.from_documents(documents, self.nvidia_embeddings)
                self.vector_store_type = "FAISS_NVIDIA"
                logger.info("Vector store FAISS creado con NVIDIA embeddings")
                return
            except Exception as e:
                logger.warning(f"No se pudo crear FAISS con NVIDIA embeddings: {e}")

        # Fallback a FAISS con embeddings locales
        try:
            self.vector_store = FAISS.from_documents(documents, self.embedding_function)
            self.vector_store_type = "FAISS_LOCAL"
            logger.info("Vector store FAISS creado con embeddings locales")
            return
        except Exception as faiss_error:
            logger.warning(f"No se pudo crear FAISS local: {faiss_error}")

        # Último fallback: usar Chroma
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embedding_function,
                metadatas=metadatas,
                collection_name="documents"
            )
            self.vector_store_type = "Chroma"
            logger.info("Vector store Chroma creado exitosamente")
        except Exception as chroma_error:
            logger.error(f"No se pudo crear ningún vector store: {chroma_error}")
            self.vector_store = None
            self.vector_store_type = None
            logger.warning("Vector store no disponible, usando búsqueda en memoria")

        logger.info("Vector store inicialización completada")
    
    def retrieve_documents(self, query: str, search_type: str = "hybrid", top_k: int = None) -> RetrievalResult:
        """
        Recupera documentos relevantes usando búsqueda híbrida.
        
        Args:
            query: Consulta del usuario
            search_type: Tipo de búsqueda ("semantic", "keyword", "hybrid")
            top_k: Número de documentos a recuperar
            
        Returns:
            RetrievalResult con documentos recuperados
        """
        if top_k is None:
            top_k = self.top_k
        
        logger.info(f"Recuperando documentos para query: '{query[:50]}...' (tipo: {search_type})")
        
        if search_type == "semantic":
            return self._semantic_search(query, top_k)
        elif search_type == "keyword":
            return self._keyword_search(query, top_k)
        elif search_type == "hybrid":
            return self._hybrid_search(query, top_k)
        else:
            raise ValueError(f"Tipo de búsqueda no soportado: {search_type}")
    
    def _semantic_search(self, query: str, top_k: int) -> RetrievalResult:
        """Búsqueda semántica usando embeddings."""
        if self.vector_store is None:
            logger.error("Vector store no inicializado, usando búsqueda en memoria")
            # Fallback a búsqueda en memoria usando embeddings
            return self._memory_search(query, top_k)
        
        try:
            # Usar el retriever del vector store
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            
            docs = retriever.get_relevant_documents(query)
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            logger.info("Fallback a búsqueda en memoria")
            return self._memory_search(query, top_k)
        
        # Convertir a DocumentChunk
        chunks = []
        scores = []
        query_embedding = self.embedding_model.encode([query])
        
        for doc in docs:
            # Buscar el chunk correspondiente
            for chunk in self.document_chunks:
                if chunk.text == doc.page_content:
                    chunks.append(chunk)
                    # Calcular score de similitud
                    query_embedding = self.embedding_model.encode([query])
                    similarity = cosine_similarity(query_embedding, [chunk.embedding])[0][0]
                    scores.append(float(similarity))
                    break
        
        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            query=query,
            total_chunks_searched=len(self.document_chunks),
            retrieval_type="semantic"
        )
    
    def _memory_search(self, query: str, top_k: int) -> RetrievalResult:
        """Búsqueda en memoria usando embeddings directamente."""
        logger.info("Ejecutando búsqueda en memoria")
        
        if not self.document_chunks or self.chunk_embeddings is None:
            logger.warning("No hay chunks o embeddings disponibles")
            return RetrievalResult(
                chunks=[],
                scores=[],
                query=query,
                total_chunks_searched=0,
                retrieval_type="memory_fallback"
            )
        
        try:
            # Generar embedding para la query
            query_embedding = self.embedding_model.encode([query])
            
            # Calcular similitudes
            similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
            
            # Obtener top_k documentos
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            chunks = []
            scores = []
            
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Solo documentos con similitud > 0.1
                    chunks.append(self.document_chunks[idx])
                    scores.append(float(similarities[idx]))
            
            logger.info(f"Búsqueda en memoria completada: {len(chunks)} documentos encontrados")
            
            return RetrievalResult(
                chunks=chunks,
                scores=scores,
                query=query,
                total_chunks_searched=len(self.document_chunks),
                retrieval_type="memory_fallback"
            )
            
        except Exception as e:
            logger.error(f"Error en búsqueda en memoria: {e}")
            return RetrievalResult(
                chunks=[],
                scores=[],
                query=query,
                total_chunks_searched=len(self.document_chunks),
                retrieval_type="memory_error"
            )
    
    def _keyword_search(self, query: str, top_k: int) -> RetrievalResult:
        """Búsqueda por palabras clave usando BM25."""
        if self.bm25_index is None:
            logger.warning("BM25 no disponible, usando búsqueda semántica")
            return self._semantic_search(query, top_k)
        
        # Tokenizar query
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Obtener scores BM25
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Obtener top_k documentos
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        chunks = []
        chunk_scores = []
        
        for idx in top_indices:
            if scores[idx] > 0:  # Solo documentos con score > 0
                chunks.append(self.document_chunks[idx])
                chunk_scores.append(float(scores[idx]))
        
        return RetrievalResult(
            chunks=chunks,
            scores=chunk_scores,
            query=query,
            total_chunks_searched=len(self.document_chunks),
            retrieval_type="keyword"
        )
    
    def _hybrid_search(self, query: str, top_k: int) -> RetrievalResult:
        """Búsqueda híbrida combinando semántica y palabras clave."""
        logger.info("Ejecutando búsqueda híbrida...")
        
        try:
            # Obtener resultados semánticos
            semantic_result = self._semantic_search(query, top_k * 2)
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            semantic_result = RetrievalResult(
                chunks=[], scores=[], query=query, 
                total_chunks_searched=len(self.document_chunks), 
                retrieval_type="semantic_error"
            )
        
        try:
            # Obtener resultados de palabras clave
            keyword_result = self._keyword_search(query, top_k * 2)
        except Exception as e:
            logger.error(f"Error en búsqueda de palabras clave: {e}")
            keyword_result = RetrievalResult(
                chunks=[], scores=[], query=query, 
                total_chunks_searched=len(self.document_chunks), 
                retrieval_type="keyword_error"
            )
        
        # Combinar y rerankear resultados
        combined_chunks = {}
        
        # Agregar resultados semánticos con peso 0.7
        for i, chunk in enumerate(semantic_result.chunks):
            if chunk.chunk_id not in combined_chunks:
                combined_chunks[chunk.chunk_id] = {
                    'chunk': chunk,
                    'semantic_score': semantic_result.scores[i] * 0.7,
                    'keyword_score': 0.0
                }
        
        # Agregar resultados de palabras clave con peso 0.3
        for i, chunk in enumerate(keyword_result.chunks):
            if chunk.chunk_id not in combined_chunks:
                combined_chunks[chunk.chunk_id] = {
                    'chunk': chunk,
                    'semantic_score': 0.0,
                    'keyword_score': keyword_result.scores[i] * 0.3
                }
            else:
                combined_chunks[chunk.chunk_id]['keyword_score'] = keyword_result.scores[i] * 0.3
        
        # Calcular scores combinados
        final_chunks = []
        final_scores = []
        
        for chunk_data in combined_chunks.values():
            combined_score = chunk_data['semantic_score'] + chunk_data['keyword_score']
            final_chunks.append(chunk_data['chunk'])
            final_scores.append(combined_score)
        
        # Ordenar por score combinado
        sorted_indices = np.argsort(final_scores)[::-1]
        
        # Tomar top_k
        top_chunks = [final_chunks[i] for i in sorted_indices[:top_k]]
        top_scores = [final_scores[i] for i in sorted_indices[:top_k]]

        # Fallback: ensure at least one chunk is available for downstream RAG steps
        if not top_chunks and final_chunks:
            fallback_idx = sorted_indices[0]
            logger.warning(
                "No chunks met hybrid thresholds; falling back to highest-scoring chunk (score=%.3f)",
                float(final_scores[fallback_idx])
            )
            top_chunks = [final_chunks[fallback_idx]]
            top_scores = [float(final_scores[fallback_idx])]
        
        return RetrievalResult(
            chunks=top_chunks,
            scores=top_scores,
            query=query,
            total_chunks_searched=len(self.document_chunks),
            retrieval_type="hybrid"
        )
    
    def rerank_documents(self, query: str, retrieval_result: RetrievalResult) -> RetrievalResult:
        """
        Rerankea documentos usando el modelo de reranking de NVIDIA.
        
        Args:
            query: Consulta original
            retrieval_result: Resultado de la recuperación inicial
            
        Returns:
            RetrievalResult rerankeado
        """
        if self.reranker is None:
            logger.warning("Reranker no disponible, devolviendo resultados originales")
            return retrieval_result
        
        logger.info("Rerankeando documentos...")
        
        # Preparar documentos para reranking
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    **(chunk.metadata or {}),
                    "chunk_id": chunk.chunk_id,
                    "score": retrieval_result.scores[i] if i < len(retrieval_result.scores) else None,
                }
            )
            for i, chunk in enumerate(retrieval_result.chunks)
        ]
        
        try:
            # Usar el reranker de NVIDIA
            reranked_docs = self.reranker.compress_documents(documents, query)
            
            # Reorganizar chunks y scores según el reranking
            reranked_chunks = []
            reranked_scores = []
            
            chunk_map = {
                chunk.chunk_id: (chunk, retrieval_result.scores[i] if i < len(retrieval_result.scores) else None)
                for i, chunk in enumerate(retrieval_result.chunks)
            }

            for doc in reranked_docs:
                chunk_id = doc.metadata.get("chunk_id") if isinstance(doc.metadata, dict) else None
                if chunk_id in chunk_map:
                    chunk, score = chunk_map[chunk_id]
                    reranked_chunks.append(chunk)
                    reranked_scores.append(score)
                else:
                    # Fallback por coincidencia de texto
                    for original_chunk, original_score in chunk_map.values():
                        if original_chunk.text == doc.page_content:
                            reranked_chunks.append(original_chunk)
                            reranked_scores.append(original_score)
                            break
            
            return RetrievalResult(
                chunks=reranked_chunks,
                scores=reranked_scores,
                query=query,
                total_chunks_searched=retrieval_result.total_chunks_searched,
                retrieval_type=f"{retrieval_result.retrieval_type}_reranked"
            )
            
        except Exception as e:
            logger.error(f"Error en reranking: {e}")
            return retrieval_result
    
    def create_react_agent(self, system_prompt: str = None) -> Any:
        """
        Crea un agente ReAct usando LangGraph.
        
        Args:
            system_prompt: Prompt del sistema (opcional)
            
        Returns:
            Agente ReAct configurado
        """
        if self.vector_store is None:
            raise ValueError("Vector store no inicializado")
        
        # Prompt del sistema optimizado basado en NVIDIA
        if system_prompt is None:
            system_prompt = (
                "Eres un asistente experto en análisis de documentos en español.\n"
                "- Usa la herramienta 'document_search' para buscar información relevante en los documentos.\n"
                "- SIEMPRE proporciona respuestas completas y útiles basadas en los documentos encontrados.\n"
                "- Si encuentras información relevante, explica claramente lo que encontraste.\n"
                "- Si no estás seguro, di que no sabes.\n"
                "- Cita las fuentes usando [Doc] para fragmentos de documentos.\n"
                "- Si los documentos no contienen información suficiente, explica claramente qué información falta.\n"
                "- Mantén las respuestas concisas, precisas y conversacionales.\n"
                "- Responde SIEMPRE en español.\n"
                "- IMPORTANTE: Si usas herramientas de búsqueda, asegúrate de proporcionar una respuesta útil basada en los resultados."
            )
        
        # Crear retriever con reranking si está disponible
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        if self.reranker is not None:
            retriever = ContextualCompressionRetriever(
                base_retriever=base_retriever,
                base_compressor=self.reranker
            )
        else:
            retriever = base_retriever

        def document_search(query: str) -> str:
            """Herramienta de búsqueda para el agente ReAct."""
            try:
                docs = retriever.invoke(query)
            except Exception as search_error:
                logger.error(f"Error en herramienta document_search: {search_error}")
                return "No se pudo completar la búsqueda en los documentos."

            if not docs:
                return "No se encontraron fragmentos relevantes en los documentos."

            formatted_results = []
            for idx, doc in enumerate(docs, start=1):
                metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
                chunk_id = metadata.get("chunk_id", idx)
                source = metadata.get("type", "fragmento")
                snippet = doc.page_content.strip()
                if len(snippet) > 400:
                    snippet = snippet[:400] + "..."
                formatted_results.append(
                    f"[Doc {idx} | chunk {chunk_id} | tipo {source}] {snippet}"
                )

            return "\n".join(formatted_results)

        retriever_tool = Tool(
            name="document_search",
            func=document_search,
            description=(
                "Busca y devuelve fragmentos relevantes del documento proporcionado. "
                "Úsala cuando necesites evidencia antes de responder."
            ),
        )
        
        # Crear agente ReAct
        agent = create_react_agent(
            model=self.llm,
            tools=[retriever_tool],
            prompt=system_prompt
        )
        
        logger.info("Agente ReAct creado exitosamente")
        return agent
    
    def get_document_stats(self) -> Dict:
        """Obtiene estadísticas del documento procesado."""
        chunk_lengths = [len(chunk.text) for chunk in self.document_chunks]
        total_words = sum(len(chunk.text.split()) for chunk in self.document_chunks)

        return {
            "total_chunks": len(self.document_chunks),
            "total_words": total_words,
            "avg_chunk_size": float(np.mean(chunk_lengths)) if chunk_lengths else 0.0,
            "embedding_model": self.embedding_model_name,
            "reranker_available": self.reranker is not None,
            "bm25_available": self.bm25_index is not None,
            "vector_store_type": self.vector_store_type,
            "agent_available": self.vector_store is not None
        }