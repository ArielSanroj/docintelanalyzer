"""
Módulo para renderizar ingredientes con puntuaciones de seguridad en Streamlit.
"""

import re
import streamlit as st
from typing import List, Dict, Tuple

def parse_ingredient_html(html_content: str) -> List[Dict[str, any]]:
    """
    Parsea el HTML de ingredientes y extrae la información.
    
    Args:
        html_content: Contenido HTML con ingredientes
        
    Returns:
        Lista de diccionarios con información de ingredientes
    """
    ingredients = []
    
    # Patrón para extraer ingredientes con puntuaciones
    pattern = r'<span class="ingredient-badge (safe|warning|danger)">\s*([^<]+?)\s*-\s*(\d+\.?\d*)/100\s*</span>'
    
    matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        safety_level, ingredient_name, score = match
        ingredients.append({
            'name': ingredient_name.strip(),
            'score': float(score),
            'safety_level': safety_level.strip(),
            'percentage': float(score)
        })
    
    return ingredients

def render_ingredient_badges(ingredients: List[Dict[str, any]], columns_per_row: int = 3):
    """
    Renderiza los ingredientes como badges en Streamlit.
    
    Args:
        ingredients: Lista de ingredientes parseados
        columns_per_row: Número de columnas por fila
    """
    if not ingredients:
        st.warning("No se encontraron ingredientes para mostrar.")
        return
    
    st.subheader("🧪 Ingredientes Detectados")
    
    # Agrupar ingredientes por nivel de seguridad
    safe_ingredients = [ing for ing in ingredients if ing['safety_level'] == 'safe']
    warning_ingredients = [ing for ing in ingredients if ing['safety_level'] == 'warning']
    danger_ingredients = [ing for ing in ingredients if ing['safety_level'] == 'danger']
    
    # Mostrar ingredientes seguros
    if safe_ingredients:
        st.markdown("### ✅ Ingredientes Seguros")
        render_ingredient_group(safe_ingredients, columns_per_row, "success")
    
    # Mostrar ingredientes con advertencia
    if warning_ingredients:
        st.markdown("### ⚠️ Ingredientes con Advertencia")
        render_ingredient_group(warning_ingredients, columns_per_row, "warning")
    
    # Mostrar ingredientes peligrosos
    if danger_ingredients:
        st.markdown("### 🚨 Ingredientes Peligrosos")
        render_ingredient_group(danger_ingredients, columns_per_row, "error")

def render_ingredient_group(ingredients: List[Dict[str, any]], columns_per_row: int, color_theme: str):
    """
    Renderiza un grupo de ingredientes con el mismo nivel de seguridad.
    
    Args:
        ingredients: Lista de ingredientes
        columns_per_row: Número de columnas por fila
        color_theme: Tema de color (success, warning, error)
    """
    # Crear columnas
    cols = st.columns(columns_per_row)
    
    for i, ingredient in enumerate(ingredients):
        col_index = i % columns_per_row
        with cols[col_index]:
            # Crear badge personalizado
            badge_html = create_ingredient_badge(
                ingredient['name'],
                ingredient['score'],
                ingredient['safety_level']
            )
            st.markdown(badge_html, unsafe_allow_html=True)

def create_ingredient_badge(name: str, score: float, safety_level: str) -> str:
    """
    Crea HTML para un badge de ingrediente.
    
    Args:
        name: Nombre del ingrediente
        score: Puntuación de seguridad (0-100)
        safety_level: Nivel de seguridad (safe, warning, danger)
        
    Returns:
        HTML del badge
    """
    # Definir colores según el nivel de seguridad
    color_map = {
        'safe': {
            'bg': '#d4edda',
            'text': '#155724',
            'border': '#c3e6cb'
        },
        'warning': {
            'bg': '#fff3cd',
            'text': '#856404',
            'border': '#ffeaa7'
        },
        'danger': {
            'bg': '#f8d7da',
            'text': '#721c24',
            'border': '#f5c6cb'
        }
    }
    
    colors = color_map.get(safety_level, color_map['warning'])
    
    # Crear emoji según el nivel
    emoji_map = {
        'safe': '✅',
        'warning': '⚠️',
        'danger': '🚨'
    }
    emoji = emoji_map.get(safety_level, '⚠️')
    
    # Crear HTML del badge
    badge_html = f"""
    <div style="
        background-color: {colors['bg']};
        color: {colors['text']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 14px;
        font-weight: 500;
        display: inline-block;
        width: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600;">{emoji} {name}</span>
            <span style="font-size: 12px; background-color: rgba(0,0,0,0.1); padding: 2px 6px; border-radius: 4px;">
                {score:.1f}/100
            </span>
        </div>
    </div>
    """
    
    return badge_html

def render_ingredient_summary(ingredients: List[Dict[str, any]]):
    """
    Renderiza un resumen de los ingredientes.
    
    Args:
        ingredients: Lista de ingredientes parseados
    """
    if not ingredients:
        return
    
    # Calcular estadísticas
    total_ingredients = len(ingredients)
    safe_count = len([ing for ing in ingredients if ing['safety_level'] == 'safe'])
    warning_count = len([ing for ing in ingredients if ing['safety_level'] == 'warning'])
    danger_count = len([ing for ing in ingredients if ing['safety_level'] == 'danger'])
    
    avg_score = sum(ing['score'] for ing in ingredients) / total_ingredients
    
    # Mostrar resumen
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Ingredientes", total_ingredients)
    
    with col2:
        st.metric("Seguros", safe_count, delta=f"{safe_count/total_ingredients*100:.1f}%")
    
    with col3:
        st.metric("Con Advertencia", warning_count, delta=f"{warning_count/total_ingredients*100:.1f}%")
    
    with col4:
        st.metric("Puntuación Promedio", f"{avg_score:.1f}/100")

def process_ingredient_response(response_text: str) -> str:
    """
    Procesa una respuesta que contiene HTML de ingredientes y la convierte a formato Streamlit.
    
    Args:
        response_text: Texto de respuesta que puede contener HTML de ingredientes
        
    Returns:
        Texto procesado con ingredientes renderizados
    """
    # Buscar y extraer HTML de ingredientes
    ingredient_pattern = r'<div[^>]*>.*?<span class="ingredient-badge[^>]*>.*?</span>.*?</div>'
    ingredient_html = re.search(ingredient_pattern, response_text, re.DOTALL)
    
    if ingredient_html:
        # Extraer el HTML de ingredientes
        html_content = ingredient_html.group(0)
        
        # Parsear ingredientes
        ingredients = parse_ingredient_html(html_content)
        
        if ingredients:
            # Renderizar ingredientes
            render_ingredient_badges(ingredients)
            render_ingredient_summary(ingredients)
            
            # Remover HTML de ingredientes del texto original
            processed_text = re.sub(ingredient_pattern, '', response_text, flags=re.DOTALL).strip()
            return processed_text
    
    return response_text