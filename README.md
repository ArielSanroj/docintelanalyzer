# DocsReview - Generador de Informes Financieros/Legales con RAG + LLM

Sistema avanzado de análisis automatizado de documentos financieros y legales usando **RAG (Retrieval-Augmented Generation)** integrado con **LLM** para respuestas más precisas y contextualizadas.

## Características Principales

### 🔍 Sistema RAG Avanzado
- **Recuperación inteligente**: Usa embeddings para encontrar información específica
- **Chunking inteligente**: Divide documentos en fragmentos optimizados
- **Scoring de relevancia**: Filtra información por pertinencia
- **Análisis de intención**: Detecta el tipo de consulta automáticamente

### 🤖 Integración LLM Mejorada
- **Contexto específico**: El LLM recibe solo información relevante recuperada por RAG
- **Prompts optimizados**: Genera respuestas más precisas y contextualizadas
- **Confianza medible**: Sistema de scoring de confianza para cada respuesta
- **Conversación contextual**: Mantiene contexto de interacciones previas

### 📄 Procesamiento de Documentos
- **PDFs subidos o desde URL**: Múltiples fuentes de entrada
- **OCR automático**: Extracción de texto de documentos escaneados
- **Análisis con IA**: Usando Llama 3.1 de NVIDIA
- **Resúmenes estructurados**: Resumen ejecutivo profesional

### 💬 Chat Inteligente
- **Preguntas específicas**: Responde consultas puntuales sobre el documento
- **Información recuperada**: Muestra los fragmentos relevantes encontrados
- **Métricas de calidad**: Confianza, relevancia y estadísticas de recuperación
- **Historial contextual**: Mantiene contexto de la conversación

### 🗄️ Almacenamiento y Gestión
- **Base de datos SQLite**: Almacenamiento persistente
- **Interfaz web**: Streamlit para fácil uso
- **Gestión de sesiones**: Estado persistente durante la sesión

## Instalación

1. **Instalar dependencias del sistema**:
```bash
# Para macOS (Apple Silicon)
brew install tesseract tesseract-lang

# Para Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng
```

2. **Instalar dependencias Python**:
```bash
pip install -r requirements.txt
```

3. **Configurar variables de entorno**:
```bash
# Crear archivo .env
echo "NVIDIA_API_KEY=tu_api_key_aqui" > .env
```

## Uso

### Aplicación Web
Ejecutar la aplicación:
```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

### Demo del Sistema RAG + LLM
Para probar las funcionalidades RAG directamente:
```bash
python rag_example.py
```

## Flujo de Trabajo RAG + LLM

1. **Carga del documento**: El usuario sube un PDF o proporciona una URL
2. **Procesamiento RAG**: 
   - El documento se divide en chunks inteligentes
   - Se generan embeddings para cada chunk
   - Se almacena la información estructurada
3. **Consulta del usuario**: El usuario hace una pregunta específica
4. **Recuperación inteligente**: 
   - Se analiza la intención de la consulta
   - Se recuperan los chunks más relevantes
   - Se optimizan los parámetros según el tipo de consulta
5. **Generación mejorada**: 
   - El LLM recibe solo información relevante
   - Se genera una respuesta contextualizada
   - Se calcula un score de confianza
6. **Presentación**: Se muestra la respuesta con información de recuperación

## Estructura del Proyecto

### Archivos Principales
- `app.py` - Aplicación principal con Streamlit y chat RAG
- `docsreview.py` - Motor de procesamiento con LangGraph
- `rag_system.py` - Sistema RAG con embeddings y recuperación
- `rag_llm_integration.py` - Integración avanzada RAG + LLM
- `rag_example.py` - Demo del sistema RAG + LLM

### Archivos de Soporte
- `database.py` - Funciones de base de datos
- `ocr_utils.py` - Utilidades de OCR
- `requirements.txt` - Dependencias Python (incluye RAG)
- `regulations.db` - Base de datos SQLite (se crea automáticamente)

## Requisitos

- Python 3.8+
- NVIDIA API Key
- Tesseract OCR instalado
- Conexión a internet para modelos de IA
- Memoria RAM: Mínimo 4GB (recomendado 8GB+ para embeddings)

## Características Técnicas del RAG

### Modelo de Embeddings
- **Modelo**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensiones**: 384 dimensiones por embedding
- **Idiomas**: Soporte multilingüe (español e inglés)

### Parámetros de Chunking
- **Tamaño de chunk**: 500 caracteres (configurable)
- **Solapamiento**: 50 caracteres entre chunks
- **Segmentación**: Por límites de oración cuando es posible

### Tipos de Consulta Detectados
- **Definición**: "¿Qué es...?", "Definir..."
- **Proceso**: "¿Cómo...?", "Pasos para..."
- **Temporal**: "¿Cuándo...?", "Fecha..."
- **Ubicación**: "¿Dónde...?", "Lugar..."
- **Persona**: "¿Quién...?", "Responsable..."
- **Causal**: "¿Por qué...?", "Razón..."

### Métricas de Calidad
- **Score de confianza**: 0.0 - 1.0
- **Relevancia promedio**: Similitud coseno promedio
- **Cobertura**: Número de chunks relevantes encontrados
- **Especificidad**: Bonus por términos específicos en la consulta

## Ventajas del Sistema RAG + LLM

1. **Precisión mejorada**: Solo información relevante llega al LLM
2. **Eficiencia**: Menos tokens procesados por el LLM
3. **Transparencia**: Se puede ver qué información se usó
4. **Escalabilidad**: Funciona con documentos grandes
5. **Contextualización**: Respuestas específicas al documento
6. **Confianza medible**: Score de confianza para cada respuesta