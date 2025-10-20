#!/usr/bin/env python3
"""
Demo del renderizador de ingredientes en Streamlit.
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingredient_renderer import parse_ingredient_html, render_ingredient_badges, render_ingredient_summary

def main():
    st.set_page_config(
        page_title="Demo Ingredientes",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Demo del Renderizador de Ingredientes")
    st.markdown("---")
    
    # HTML de ejemplo
    sample_html = '''
    <div class="ingredients-container">
        <span class="ingredient-badge safe">
            helianthus annuus seed oil - 85.0/100
        </span>

        <span class="ingredient-badge safe">
            aloe barbadensis leaf extract - 90.0/100
        </span>

        <span class="ingredient-badge safe">
            aqua - 95.0/100
        </span>

        <span class="ingredient-badge warning">
            acrylates/c10-30 alkyl acrylate crosspolymer - 50.0/100
        </span>

        <span class="ingredient-badge warning">
            ethylhexylglycerin - 70.0/100
        </span>

        <span class="ingredient-badge warning">
            isopropyl - 45.0/100
        </span>

        <span class="ingredient-badge warning">
            phenoxyethanol - 40.0/100
        </span>
    </div>
    '''
    
    st.subheader("📝 HTML Original")
    st.code(sample_html, language="html")
    
    st.markdown("---")
    
    # Parsear y renderizar
    ingredients = parse_ingredient_html(sample_html)
    
    if ingredients:
        st.subheader("📊 Resumen de Ingredientes")
        render_ingredient_summary(ingredients)
        
        st.markdown("---")
        
        st.subheader("🏷️ Ingredientes Renderizados")
        render_ingredient_badges(ingredients, columns_per_row=2)
        
        st.markdown("---")
        
        st.subheader("📋 Datos Parseados")
        st.json(ingredients)
    else:
        st.error("No se pudieron parsear los ingredientes")

if __name__ == "__main__":
    main()