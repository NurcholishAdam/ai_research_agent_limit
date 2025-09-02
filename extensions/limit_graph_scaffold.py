#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH Scaffold
Implements the graph construction component that adapts LIMIT's qrels into semantic graph format.
This component handles the transformation from LIMIT dataset to semantic graph representation.
"""

import spacy
import networkx as nx
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Import core components
from limit_graph_core import (
    LimitGraphNode, LimitGraphEdge, LimitQuery, BaseLimitGraphComponent,
    LIMIT_GRAPH_REGISTRY
)

class LimitGraphScaffold(BaseLimitGraphComponent):
    """
    LIMIT-GRAPH Scaffold: Converts LIMIT corpus to semantic graph
    
    Architecture:
    - Nodes: documents, entities, predicates  
    - Edges: semantic relations (likes, dislikes, owns)
    - Uses spaCy + custom rules to extract triples from LIMIT corpus
    - Stores in NetworkX for fast traversal
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Initialize spaCy for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Using fallback processing.")
            self.nlp = None
        
        # Graph storage (NetworkX for fast traversal)
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, LimitGraphNode] = {}
        self.edges: List[LimitGraphEdge] = []
        
        # Custom extraction rules for semantic relations
        self.relation_patterns = {
            "likes": ["likes", "enjoys", "prefers", "loves"],
            "dislikes": ["dislikes", "hates", "avoids"],
            "owns": ["owns", "has", "possesses"],
            "contains": ["contains", "includes"],
            "located_in": ["in", "at", "located"],
            "part_of": ["part of", "belongs to"]
        }
        
        # Register with global registry
        LIMIT_GRAPH_REGISTRY.register_component(self)
        
        print("🏗️ LIMIT-GRAPH Scaffold initialized")
    
    def process_limit_corpus(self, corpus_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process LIMIT corpus and build semantic graph"""
        
        stats = {"documents": 0, "entities": 0, "relations": 0, "queries": 0}
        
        for item in corpus_data:
            # Process query
            query_node = LimitGraphNode(
                node_id=f"query_{item['query_id']}",
                node_type="query",
                content=item["query"]
            )
            self._add_node(query_node)
            stats["queries"] += 1
            
            # Process relevant documents
            for doc_id in item["relevant_docs"]:
                doc_node = LimitGraphNode(
                    node_id=doc_id,
                    node_type="document", 
                    content=f"Document {doc_id}"
                )
                self._add_node(doc_node)
                stats["documents"] += 1
            
            # Process graph edges (explicit relations)
            for edge_data in item["graph_edges"]:
                self._add_explicit_edge(edge_data)
                stats["relations"] += 1
            
            # Extract entities using spaCy + custom rules
            if self.nlp:
                entities = self._extract_entities(item["query"])
                stats["entities"] += len(entities)
        
        print(f"📊 Corpus processed: {stats}")
        return stats
    
    def _add_node(self, node: LimitGraphNode):
        """Add node to graph"""
        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node
            self.graph.add_node(node.node_id, **node.metadata)
    
    def _add_explicit_edge(self, edge_data: Dict[str, str]):
        """Add explicit edge from LIMIT data"""
        source = edge_data["source"]
        target = edge_data["target"] 
        relation = edge_data["relation"]
        
        # Ensure target entity exists
        if target not in self.nodes:
            entity_node = LimitGraphNode(
                node_id=target,
                node_type="entity",
                content=target
            )
            self._add_node(entity_node)
        
        # Add edge
        edge = LimitGraphEdge(
            source=source,
            target=target,
            relation=relation,
            confidence=1.0,  # High confidence for explicit edges
            provenance=["limit_qrels"]
        )
        
        self.edges.append(edge)
        self.graph.add_edge(source, target, relation=relation)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities using spaCy + custom rules"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_id = f"entity_{ent.text.lower()}"
            if entity_id not in self.nodes:
                entity_node = LimitGraphNode(
                    node_id=entity_id,
                    node_type="entity",
                    content=ent.text
                )
                self._add_node(entity_node)
                entities.append(entity_id)
        
        return entities
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        
        node_types = defaultdict(int)
        for node in self.nodes.values():
            node_types[node.node_type] += 1
        
        relation_types = defaultdict(int)
        for edge in self.edges:
            relation_types[edge.relation] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": dict(node_types),
            "relation_types": dict(relation_types),
            "graph_density": nx.density(self.graph) if len(self.graph) > 1 else 0.0,
            "connected_components": nx.number_weakly_connected_components(self.graph)
        }
    
    def export_to_neo4j_format(self) -> Dict[str, Any]:
        """Export graph to Neo4j-compatible format"""
        
        neo4j_data = {
            "nodes": [],
            "relationships": []
        }
        
        # Export nodes
        for node in self.nodes.values():
            neo4j_node = {
                "id": node.node_id,
                "labels": [node.node_type.capitalize()],
                "properties": {
                    "content": node.content,
                    "relevance_score": node.relevance_score,
                    "created_at": node.created_at.isoformat(),
                    **node.metadata
                }
            }
            neo4j_data["nodes"].append(neo4j_node)
        
        # Export relationships
        for edge in self.edges:
            neo4j_rel = {
                "startNode": edge.source,
                "endNode": edge.target,
                "type": edge.relation.upper(),
                "properties": {
                    "confidence": edge.confidence,
                    "provenance": edge.provenance,
                    "created_at": edge.created_at.isoformat(),
                    **edge.metadata
                }
            }
            neo4j_data["relationships"].append(neo4j_rel)
        
        return neo4j_data