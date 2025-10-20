import os
import pymupdf as fitz  # pymupdf
from PIL import Image
import pytesseract
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re  # Para regex genérico

logger = logging.getLogger(__name__)

# Set Tesseract path for Apple Silicon Mac
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def extract_text_from_pdf(file_path: str) -> dict:
    """Extract text from PDF with OCR fallback for image-based pages"""
    logger.debug(f"Procesando archivo: {file_path}")
    try:
        if not os.path.exists(file_path):
            return {"error": f"Error: Archivo no encontrado: {file_path}"}
        
        doc = fitz.open(file_path)
        text = ""
        
        # Check available Tesseract languages
        available_langs = pytesseract.get_languages(config='')
        logger.debug(f"Idiomas disponibles para Tesseract: {available_langs}")
        use_spa = 'spa' in available_langs
        use_eng = 'eng' in available_langs
        
        if not (use_spa or use_eng):
            return {"error": f"Error: No se encontraron los paquetes de idioma 'spa' ni 'eng' para Tesseract. Instale con 'brew install tesseract-lang' en PyCharm's Terminal."}
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text").strip()
            
            if len(page_text) < 50:
                logger.debug(f"Página {page_num + 1} parece basada en imagen; aplicando OCR...")
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Try Spanish OCR if available
                if use_spa:
                    page_text = pytesseract.image_to_string(img, lang='spa')
                    # Fallback to English if result is too short and English is available
                    if len(page_text.strip()) < 20 and use_eng:
                        logger.debug(f"OCR en español dio texto corto; intentando en inglés...")
                        page_text = pytesseract.image_to_string(img, lang='eng')
                elif use_eng:
                    logger.debug(f"Paquete 'spa' no disponible; usando inglés...")
                    page_text = pytesseract.image_to_string(img, lang='eng')
                
                if len(page_text.strip()) < 20:
                    logger.warning(f"OCR falló en página {page_num + 1}; texto muy corto.")
            
            text += f"\n--- Página {page_num + 1} ---\n{page_text}\n"
        
        doc.close()
        
        if text.strip():
            return {
                "file_path": file_path,
                "doc_text": text,
                "description": f"Archivo subido: {os.path.basename(file_path)}"
            }
        return {"error": f"Error: No se pudo extraer texto de: {file_path}"}
        
    except Exception as e:
        error_msg = str(e)
        if "tesseract is not installed" in error_msg.lower() or "cannot open resource" in error_msg.lower():
            error_msg = "Tesseract OCR no está instalado o no se encuentran los paquetes de idioma 'spa' o 'eng'. Instale con 'brew install tesseract tesseract-lang' en PyCharm's Terminal."
        return {"error": f"Error: Error leyendo {file_path}: {error_msg}"}

def _process_pdf_stream(content: bytes, original_url: str) -> dict:
    """Helper genérico para procesar PDF stream (de URL o upload)"""
    doc = fitz.open(stream=content, filetype="pdf")
    text = ""
    
    # Check available Tesseract languages
    available_langs = pytesseract.get_languages(config='')
    logger.debug(f"Idiomas disponibles para Tesseract: {available_langs}")
    use_spa = 'spa' in available_langs
    use_eng = 'eng' in available_langs
    
    if not (use_spa or use_eng):
        return {"error": f"Error: No se encontraron los paquetes de idioma 'spa' ni 'eng' para Tesseract. Instale con 'brew install tesseract-lang' en PyCharm's Terminal."}
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text").strip()
        
        if len(page_text) < 50:
            logger.debug(f"Página {page_num + 1} parece basada en imagen; aplicando OCR...")
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Try Spanish OCR if available
            if use_spa:
                page_text = pytesseract.image_to_string(img, lang='spa')
                # Fallback to English if result is too short and English is available
                if len(page_text.strip()) < 20 and use_eng:
                    logger.debug(f"OCR en español dio texto corto; intentando en inglés...")
                    page_text = pytesseract.image_to_string(img, lang='eng')
            elif use_eng:
                logger.debug(f"Paquete 'spa' no disponible; usando inglés...")
                page_text = pytesseract.image_to_string(img, lang='eng')
            
            if len(page_text.strip()) < 20:
                logger.warning(f"OCR falló en página {page_num + 1}; texto muy corto.")
        
        text += f"\n--- Página {page_num + 1} ---\n{page_text}\n"
    
    page_count = len(doc)
    doc.close()
    
    if len(text) > 500000:
        text = text[:500000] + "\n[Texto PDF truncado por límite de procesamiento]"
        logger.warning("Texto PDF truncado a 500k chars para evitar sobrecarga")
    
    return {
        "doc_url": original_url,
        "doc_text": text,
        "description": f"Documento PDF: {original_url} ({page_count} páginas, {len(text)} chars)"
    }

def extract_text_from_url(url: str) -> dict:
    """Extract text from URL (HTML or PDF) with OCR fallback for image-based PDFs"""
    logger.debug(f"Obteniendo URL: {url}")
    try:
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        response.raise_for_status()
        
        if url.lower().endswith('.pdf'):
            return _process_pdf_stream(response.content, url)
        
        # Detecta enlaces PDF genéricos en HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_links = []
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            text = a.get_text().lower()
            if re.search(r'\.pdf$', href) or ('pdf' in href and 'download' in text) or any(word in text for word in ['descargar', 'download', 'pdf', 'documento']):
                full_href = urljoin(url, a['href'])
                pdf_links.append(full_href)
        
        if pdf_links:
            pdf_url = pdf_links[0]  # Toma primero; log si múltiples
            logger.debug(f"PDF detectado: {pdf_url} (de {len(pdf_links)})")
            pdf_response = requests.get(pdf_url, timeout=60, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
            pdf_response.raise_for_status()
            result = _process_pdf_stream(pdf_response.content, url)
            result['description'] += " (PDF full de enlace)"
            return result
        
        # Fallback HTML: Limpia elementos ruido
        for elem in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            elem.decompose()
        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)  # Compacta líneas
        
        # Límite genérico
        if len(text) > 500000:
            text = text[:500000] + "\n[Truncado por límite]"
            logger.warning("Texto truncado a 500k chars")
        
        if not text.strip():
            return {"error": "No se pudo extraer texto (página vacía o solo ruido)."}
        
        logger.debug(f"Texto HTML extraído: {len(text)} chars")
        return {
            "doc_url": url,
            "doc_text": text,
            "description": f"URL: {url} (HTML, {len(text)} chars)"
        }
    except Exception as e:
        logger.error(f"Error URL {url}: {e}")
        return {"error": f"Error: {str(e)}"}