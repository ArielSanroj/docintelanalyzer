#!/usr/bin/env python3
"""
Script para ver logs del sistema RAG en tiempo real.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def view_streamlit_logs():
    """Muestra los logs de Streamlit en tiempo real."""
    print("🔍 Monitoreando logs de Streamlit...")
    print("Presiona Ctrl+C para salir")
    print("=" * 50)
    
    try:
        # Buscar el proceso de Streamlit
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        streamlit_processes = []
        
        for line in result.stdout.split('\n'):
            if 'streamlit run app.py' in line and 'grep' not in line:
                streamlit_processes.append(line)
        
        if not streamlit_processes:
            print("❌ No se encontró ningún proceso de Streamlit ejecutándose")
            print("💡 Ejecuta: source .venv/bin/activate && streamlit run app.py")
            return
        
        print(f"✅ Encontrados {len(streamlit_processes)} proceso(s) de Streamlit:")
        for i, proc in enumerate(streamlit_processes):
            print(f"  {i+1}. {proc}")
        
        print("\n📊 Para ver logs en tiempo real:")
        print("1. Abre otra terminal")
        print("2. Ejecuta: tail -f ~/.streamlit/logs/streamlit.log")
        print("3. O ejecuta: streamlit run app.py (sin --server.headless)")
        
    except KeyboardInterrupt:
        print("\n👋 Saliendo del monitor de logs...")
    except Exception as e:
        print(f"❌ Error: {e}")

def view_python_logs():
    """Muestra logs de Python si están configurados."""
    print("🐍 Verificando configuración de logs de Python...")
    
    # Verificar si hay archivos de log
    log_files = [
        "app.log",
        "rag_system.log", 
        "advanced_rag_system.log",
        "logs/app.log",
        "logs/rag.log"
    ]
    
    found_logs = []
    for log_file in log_files:
        if os.path.exists(log_file):
            found_logs.append(log_file)
    
    if found_logs:
        print(f"✅ Archivos de log encontrados: {found_logs}")
        for log_file in found_logs:
            print(f"📄 {log_file}: {os.path.getsize(log_file)} bytes")
            print(f"   Última modificación: {time.ctime(os.path.getmtime(log_file))}")
    else:
        print("ℹ️ No se encontraron archivos de log específicos")
        print("💡 Los logs se muestran en la consola de Streamlit")

def show_debug_info():
    """Muestra información de debug del sistema."""
    print("🔧 Información de Debug del Sistema:")
    print("=" * 50)
    
    # Verificar entorno virtual
    venv_path = Path(".venv")
    if venv_path.exists():
        print(f"✅ Entorno virtual: {venv_path.absolute()}")
        
        # Verificar Python
        python_path = venv_path / "bin" / "python"
        if python_path.exists():
            try:
                result = subprocess.run([str(python_path), "--version"], 
                                      capture_output=True, text=True)
                print(f"✅ Python: {result.stdout.strip()}")
            except:
                print("❌ Error verificando versión de Python")
    else:
        print("❌ Entorno virtual no encontrado")
    
    # Verificar dependencias clave
    key_packages = [
        "streamlit", "langchain", "langgraph", 
        "chromadb", "sentence_transformers"
    ]
    
    print("\n📦 Dependencias clave:")
    for package in key_packages:
        try:
            result = subprocess.run([
                str(python_path), "-c", f"import {package}; print({package}.__version__)"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print(f"✅ {package}: {result.stdout.strip()}")
            else:
                print(f"❌ {package}: No disponible")
        except:
            print(f"❌ {package}: Error verificando")

def main():
    """Función principal."""
    print("🚀 Monitor de Logs del Sistema RAG")
    print("=" * 50)
    
    show_debug_info()
    print("\n")
    view_python_logs()
    print("\n")
    view_streamlit_logs()

if __name__ == "__main__":
    main()