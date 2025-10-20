#!/usr/bin/env python3
"""
Script para probar el renderizador de ingredientes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingredient_renderer import parse_ingredient_html, render_ingredient_badges, render_ingredient_summary

def test_ingredient_parsing():
    """Prueba el parsing de ingredientes HTML."""
    print("🧪 Probando parser de ingredientes...")
    
    # HTML de ejemplo
    test_html = '''
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
    
    # Parsear ingredientes
    ingredients = parse_ingredient_html(test_html)
    
    print(f"✅ Ingredientes parseados: {len(ingredients)}")
    
    for ingredient in ingredients:
        print(f"  - {ingredient['name']}: {ingredient['score']}/100 ({ingredient['safety_level']})")
    
    return ingredients

def test_ingredient_rendering():
    """Prueba el rendering de ingredientes."""
    print("\n🎨 Probando renderizado de ingredientes...")
    
    # Crear datos de prueba
    test_ingredients = [
        {'name': 'helianthus annuus seed oil', 'score': 85.0, 'safety_level': 'safe'},
        {'name': 'aloe barbadensis leaf extract', 'score': 90.0, 'safety_level': 'safe'},
        {'name': 'aqua', 'score': 95.0, 'safety_level': 'safe'},
        {'name': 'acrylates/c10-30 alkyl acrylate crosspolymer', 'score': 50.0, 'safety_level': 'warning'},
        {'name': 'ethylhexylglycerin', 'score': 70.0, 'safety_level': 'warning'},
        {'name': 'isopropyl', 'score': 45.0, 'safety_level': 'warning'},
        {'name': 'phenoxyethanol', 'score': 40.0, 'safety_level': 'warning'}
    ]
    
    print("✅ Datos de prueba creados")
    
    # Simular renderizado (sin Streamlit)
    print("\n📊 Resumen de ingredientes:")
    render_ingredient_summary(test_ingredients)
    
    print("\n🏷️ Badges de ingredientes:")
    for ingredient in test_ingredients:
        from ingredient_renderer import create_ingredient_badge
        badge_html = create_ingredient_badge(
            ingredient['name'],
            ingredient['score'],
            ingredient['safety_level']
        )
        print(f"  {ingredient['name']}: {ingredient['safety_level']} - {ingredient['score']}/100")

def main():
    """Función principal de prueba."""
    print("🚀 Prueba del Renderizador de Ingredientes")
    print("=" * 50)
    
    try:
        # Probar parsing
        ingredients = test_ingredient_parsing()
        
        # Probar rendering
        test_ingredient_rendering()
        
        print("\n✅ Todas las pruebas completadas exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()