"""
Sistema de grafo de conocimiento mejorado basado en RAG-Anything.
Implementa extracción de entidades y descubrimiento de relaciones avanzado.
"""

import logging
import re
import json
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict, Counter
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Entidad en el grafo de conocimiento."""
    id: str
    name: str
    entity_type: str  # "person", "organization", "location", "concept", "date", "number"
    mentions: List[int] = field(default_factory=list)  # Chunk IDs donde aparece
    properties: Dict = field(default_factory=dict)
    confidence: float = 0.0
    embedding: Optional[np.ndarray] = None

@dataclass
class Relation:
    """Relación entre entidades."""
    id: str
    source_entity: str
    target_entity: str
    relation_type: str  # "is_a", "part_of", "located_in", "works_for", "related_to"
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)  # Texto que evidencia la relación
    metadata: Dict = field(default_factory=dict)

@dataclass
class KnowledgeGraph:
    """Grafo de conocimiento completo."""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    entity_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    graph: Optional[nx.Graph] = None

class EnhancedKnowledgeGraph:
    """Sistema de grafo de conocimiento mejorado."""
    
    def __init__(self, embedding_model=None):
        """
        Inicializa el sistema de grafo de conocimiento.
        
        Args:
            embedding_model: Modelo de embeddings para entidades
        """
        self.embedding_model = embedding_model
        self.knowledge_graph = KnowledgeGraph()
        
        # Patrones de extracción de entidades
        self.entity_patterns = {
            "person": [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Nombres completos
                r'\b(?:Dr|Prof|Sr|Sra|Srta)\.?\s+[A-Z][a-z]+\b',  # Títulos + nombres
                r'\b[A-Z][a-z]+\s+(?:Jr|Sr|III|IV)\.?\b'  # Nombres con sufijos
            ],
            "organization": [
                r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation|University|Institute|Foundation|Association)\b',
                r'\b(?:La|El|Los|Las)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Organizaciones en español
                r'\b[A-Z]{2,}\b'  # Acrónimos
            ],
            "location": [
                r'\b(?:en|de|del|de la|de las|de los)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                r'\b(?:país|ciudad|región|estado|provincia|municipio|zona|área)\s+[A-Z][a-z]+\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b'  # Ciudad, Estado
            ],
            "date": [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Fechas numéricas
                r'\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+\d{1,2},?\s+\d{4}\b',
                r'\b(?:lunes|martes|miércoles|jueves|viernes|sábado|domingo)\b',
                r'\b(?:hoy|ayer|mañana|esta semana|la semana pasada|el mes pasado|el año pasado)\b'
            ],
            "number": [
                r'\b\d+(?:\.\d+)?\b',  # Números enteros y decimales
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',  # Números con separadores de miles
                r'\b(?:uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|veinte|treinta|cuarenta|cincuenta|cien|mil|millón)\b'
            ],
            "concept": [
                r'\b(?:definición|concepto|término|significado|explicación|descripción)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:es|son|define|significa|representa)\b',
                r'\b(?:método|técnica|proceso|sistema|modelo|enfoque|estrategia)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            ]
        }
        
        # Patrones de relaciones
        self.relation_patterns = {
            "is_a": [
                r'(\w+)\s+(?:es|son)\s+(?:un|una|el|la|los|las)\s+(\w+)',
                r'(\w+)\s+(?:es|son)\s+(\w+)',
                r'(\w+)\s+(?:define|significa|representa)\s+(\w+)'
            ],
            "part_of": [
                r'(\w+)\s+(?:es|son)\s+(?:parte de|componente de|elemento de)\s+(\w+)',
                r'(\w+)\s+(?:pertenece a|está en|se encuentra en)\s+(\w+)'
            ],
            "located_in": [
                r'(\w+)\s+(?:está|se encuentra|se ubica)\s+(?:en|dentro de)\s+(\w+)',
                r'(\w+)\s+(?:situado|ubicado)\s+(?:en|en la|en el)\s+(\w+)'
            ],
            "works_for": [
                r'(\w+)\s+(?:trabaja|trabajó|trabajará)\s+(?:en|para|con)\s+(\w+)',
                r'(\w+)\s+(?:empleado|empleada|miembro)\s+(?:de|del|de la)\s+(\w+)'
            ],
            "related_to": [
                r'(\w+)\s+(?:relacionado|conectado|asociado)\s+(?:con|a|al|a la)\s+(\w+)',
                r'(\w+)\s+(?:y|e)\s+(\w+)\s+(?:están|son)\s+(?:relacionados|conectados)'
            ]
        }
    
    def build_knowledge_graph(self, chunks: List[Any]) -> KnowledgeGraph:
        """
        Construye el grafo de conocimiento a partir de los chunks.
        
        Args:
            chunks: Lista de chunks multimodales
            
        Returns:
            Grafo de conocimiento construido
        """
        logger.info("Construyendo grafo de conocimiento mejorado...")
        
        # Extraer entidades
        entities = self._extract_entities(chunks)
        
        # Extraer relaciones
        relations = self._extract_relations(chunks, entities)
        
        # Generar embeddings para entidades
        if self.embedding_model:
            self._generate_entity_embeddings(entities)
        
        # Construir grafo de NetworkX
        self._build_networkx_graph(entities, relations)
        
        # Actualizar grafo de conocimiento
        self.knowledge_graph.entities = entities
        self.knowledge_graph.relations = relations
        
        logger.info(f"Grafo de conocimiento construido: {len(entities)} entidades, {len(relations)} relaciones")
        
        return self.knowledge_graph
    
    def _extract_entities(self, chunks: List[Any]) -> Dict[str, Entity]:
        """Extrae entidades de los chunks."""
        entities = {}
        entity_counter = Counter()
        
        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            chunk_id = getattr(chunk, 'chunk_id', 0)
            
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Normalizar nombre de entidad
                        entity_name = self._normalize_entity_name(match, entity_type)
                        
                        if entity_name not in entities:
                            entities[entity_name] = Entity(
                                id=f"{entity_type}_{len(entities)}",
                                name=entity_name,
                                entity_type=entity_type,
                                mentions=[],
                                properties={},
                                confidence=0.0
                            )
                        
                        # Agregar mención
                        entities[entity_name].mentions.append(chunk_id)
                        entity_counter[entity_name] += 1
        
        # Calcular confianza basada en frecuencia
        for entity_name, entity in entities.items():
            frequency = entity_counter[entity_name]
            entity.confidence = min(frequency / 5.0, 1.0)  # Normalizar a [0,1]
            entity.properties["frequency"] = frequency
            entity.properties["unique_mentions"] = len(set(entity.mentions))
        
        return entities
    
    def _normalize_entity_name(self, name: str, entity_type: str) -> str:
        """Normaliza el nombre de una entidad."""
        # Limpiar y normalizar
        normalized = re.sub(r'[^\w\s]', '', name.strip())
        normalized = ' '.join(normalized.split())  # Eliminar espacios múltiples
        
        # Normalizaciones específicas por tipo
        if entity_type == "person":
            # Capitalizar nombres de personas
            normalized = ' '.join(word.capitalize() for word in normalized.split())
        elif entity_type == "organization":
            # Mantener mayúsculas para organizaciones
            normalized = normalized.title()
        elif entity_type == "location":
            # Capitalizar ubicaciones
            normalized = ' '.join(word.capitalize() for word in normalized.split())
        
        return normalized
    
    def _extract_relations(self, chunks: List[Any], entities: Dict[str, Entity]) -> List[Relation]:
        """Extrae relaciones entre entidades."""
        relations = []
        relation_id = 0
        
        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            chunk_id = getattr(chunk, 'chunk_id', 0)
            
            for relation_type, patterns in self.relation_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        source_text = match.group(1).strip()
                        target_text = match.group(2).strip()
                        
                        # Buscar entidades correspondientes
                        source_entity = self._find_entity_by_name(source_text, entities)
                        target_entity = self._find_entity_by_name(target_text, entities)
                        
                        if source_entity and target_entity and source_entity != target_entity:
                            # Verificar si la relación ya existe
                            existing_relation = self._find_existing_relation(
                                source_entity, target_entity, relation_type, relations
                            )
                            
                            if existing_relation:
                                # Incrementar confianza de relación existente
                                existing_relation.confidence = min(existing_relation.confidence + 0.1, 1.0)
                                existing_relation.evidence.append(match.group(0))
                            else:
                                # Crear nueva relación
                                relation = Relation(
                                    id=f"rel_{relation_id}",
                                    source_entity=source_entity,
                                    target_entity=target_entity,
                                    relation_type=relation_type,
                                    confidence=0.5,  # Confianza inicial
                                    evidence=[match.group(0)],
                                    metadata={"chunk_id": chunk_id}
                                )
                                relations.append(relation)
                                relation_id += 1
        
        return relations
    
    def _find_entity_by_name(self, name: str, entities: Dict[str, Entity]) -> Optional[str]:
        """Encuentra una entidad por nombre (búsqueda aproximada)."""
        normalized_name = self._normalize_entity_name(name, "unknown")
        
        # Búsqueda exacta
        if normalized_name in entities:
            return normalized_name
        
        # Búsqueda aproximada
        for entity_name, entity in entities.items():
            if self._names_similar(normalized_name, entity_name):
                return entity_name
        
        return None
    
    def _names_similar(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """Verifica si dos nombres son similares."""
        # Normalizar nombres
        n1 = re.sub(r'[^\w\s]', '', name1.lower())
        n2 = re.sub(r'[^\w\s]', '', name2.lower())
        
        # Calcular similitud de Jaccard
        words1 = set(n1.split())
        words2 = set(n2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        return jaccard_similarity >= threshold
    
    def _find_existing_relation(self, source: str, target: str, relation_type: str, relations: List[Relation]) -> Optional[Relation]:
        """Encuentra una relación existente."""
        for relation in relations:
            if (relation.source_entity == source and 
                relation.target_entity == target and 
                relation.relation_type == relation_type):
                return relation
        return None
    
    def _generate_entity_embeddings(self, entities: Dict[str, Entity]):
        """Genera embeddings para las entidades."""
        if not self.embedding_model:
            return
        
        logger.info("Generando embeddings para entidades...")
        
        entity_names = list(entities.keys())
        embeddings = self.embedding_model.encode(entity_names)
        
        for i, entity_name in enumerate(entity_names):
            entities[entity_name].embedding = embeddings[i]
            self.knowledge_graph.entity_embeddings[entity_name] = embeddings[i]
    
    def _build_networkx_graph(self, entities: Dict[str, Entity], relations: List[Relation]):
        """Construye el grafo de NetworkX."""
        self.knowledge_graph.graph = nx.Graph()
        
        # Agregar nodos (entidades)
        for entity_name, entity in entities.items():
            self.knowledge_graph.graph.add_node(
                entity_name,
                entity_type=entity.entity_type,
                confidence=entity.confidence,
                mentions=entity.mentions,
                properties=entity.properties
            )
        
        # Agregar aristas (relaciones)
        for relation in relations:
            self.knowledge_graph.graph.add_edge(
                relation.source_entity,
                relation.target_entity,
                relation_type=relation.relation_type,
                confidence=relation.confidence,
                evidence=relation.evidence,
                metadata=relation.metadata
            )
    
    def find_related_entities(self, entity_name: str, max_depth: int = 2) -> List[Dict]:
        """
        Encuentra entidades relacionadas a una entidad dada.
        
        Args:
            entity_name: Nombre de la entidad
            max_depth: Profundidad máxima de búsqueda
            
        Returns:
            Lista de entidades relacionadas
        """
        if not self.knowledge_graph.graph or entity_name not in self.knowledge_graph.graph:
            return []
        
        related_entities = []
        
        # Búsqueda en anchura limitada por profundidad
        visited = set()
        queue = [(entity_name, 0)]  # (entity, depth)
        
        while queue:
            current_entity, depth = queue.pop(0)
            
            if current_entity in visited or depth > max_depth:
                continue
            
            visited.add(current_entity)
            
            if depth > 0:  # No incluir la entidad original
                entity_data = self.knowledge_graph.graph.nodes[current_entity]
                related_entities.append({
                    "entity_name": current_entity,
                    "entity_type": entity_data.get("entity_type", "unknown"),
                    "confidence": entity_data.get("confidence", 0.0),
                    "depth": depth,
                    "properties": entity_data.get("properties", {})
                })
            
            # Agregar vecinos al queue
            for neighbor in self.knowledge_graph.graph.neighbors(current_entity):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return related_entities
    
    def get_entity_statistics(self) -> Dict:
        """Obtiene estadísticas del grafo de conocimiento."""
        if not self.knowledge_graph.graph:
            return {"status": "No graph built"}
        
        entity_types = Counter()
        relation_types = Counter()
        
        for entity in self.knowledge_graph.entities.values():
            entity_types[entity.entity_type] += 1
        
        for relation in self.knowledge_graph.relations:
            relation_types[relation.relation_type] += 1
        
        return {
            "total_entities": len(self.knowledge_graph.entities),
            "total_relations": len(self.knowledge_graph.relations),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
            "graph_density": nx.density(self.knowledge_graph.graph),
            "connected_components": nx.number_connected_components(self.knowledge_graph.graph),
            "avg_clustering": nx.average_clustering(self.knowledge_graph.graph)
        }
    
    def export_graph(self, format: str = "json") -> str:
        """Exporta el grafo de conocimiento en el formato especificado."""
        if format == "json":
            graph_data = {
                "entities": {
                    name: {
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "mentions": entity.mentions,
                        "properties": entity.properties,
                        "confidence": entity.confidence
                    }
                    for name, entity in self.knowledge_graph.entities.items()
                },
                "relations": [
                    {
                        "id": rel.id,
                        "source_entity": rel.source_entity,
                        "target_entity": rel.target_entity,
                        "relation_type": rel.relation_type,
                        "confidence": rel.confidence,
                        "evidence": rel.evidence,
                        "metadata": rel.metadata
                    }
                    for rel in self.knowledge_graph.relations
                ]
            }
            return json.dumps(graph_data, indent=2, ensure_ascii=False)
        
        elif format == "gml":
            if self.knowledge_graph.graph:
                return "\n".join(nx.generate_gml(self.knowledge_graph.graph))
        
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def visualize_graph(self, max_nodes: int = 50) -> Optional[str]:
        """Genera una visualización del grafo (requiere matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            if not self.knowledge_graph.graph:
                return None
            
            # Limitar número de nodos para visualización
            if len(self.knowledge_graph.graph.nodes) > max_nodes:
                # Seleccionar nodos con mayor grado
                degrees = dict(self.knowledge_graph.graph.degree())
                top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                subgraph = self.knowledge_graph.graph.subgraph([node for node, _ in top_nodes])
            else:
                subgraph = self.knowledge_graph.graph
            
            # Crear visualización
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
            
            # Colores por tipo de entidad
            entity_type_colors = {
                "person": "red",
                "organization": "blue", 
                "location": "green",
                "date": "orange",
                "number": "purple",
                "concept": "brown"
            }
            
            # Dibujar nodos
            for entity_type, color in entity_type_colors.items():
                nodes = [node for node, data in subgraph.nodes(data=True) 
                        if data.get("entity_type") == entity_type]
                if nodes:
                    nx.draw_networkx_nodes(subgraph, pos, nodelist=nodes, 
                                         node_color=color, node_size=300, alpha=0.7)
            
            # Dibujar aristas
            nx.draw_networkx_edges(subgraph, pos, alpha=0.5, edge_color="gray")
            
            # Dibujar etiquetas
            nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight="bold")
            
            # Crear leyenda
            patches = [mpatches.Patch(color=color, label=entity_type.title()) 
                      for entity_type, color in entity_type_colors.items()]
            plt.legend(handles=patches, loc="upper right")
            
            plt.title("Grafo de Conocimiento")
            plt.axis("off")
            
            # Guardar como string base64
            import io
            import base64
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except ImportError:
            logger.warning("matplotlib no disponible para visualización")
            return None
        except Exception as e:
            logger.error(f"Error generando visualización: {e}")
            return None