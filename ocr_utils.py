import os
import fitz  # pymupdf
from PIL import Image
import pytesseract
import logging

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

def extract_text_from_url(url: str) -> dict:
    """Extract text from URL (HTML or PDF) with OCR fallback for image-based PDFs"""
    import requests
    from bs4 import BeautifulSoup
    
    logger.debug(f"Obteniendo URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        if url.lower().endswith('.pdf'):
            doc = fitz.open(stream=response.content, filetype="pdf")
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
        else:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
        
        if not text.strip():
            return {"error": "Error: No se pudo extraer texto del documento."}
        
        return {
            "doc_url": url,
            "doc_text": text,
            "description": f"Documento desde URL: {url}"
        }
        
    except Exception as e:
        error_msg = str(e)
        if "tesseract is not installed" in error_msg.lower() or "cannot open resource" in error_msg.lower():
            error_msg = "Tesseract OCR no está instalado o no se encuentran los paquetes de idioma 'spa' o 'eng'. Instale con 'brew install tesseract tesseract-lang' en PyCharm's Terminal."
        return {"error": f"Error: Error obteniendo URL: {error_msg}"}