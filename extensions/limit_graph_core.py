#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH Core Components
Core data structures and base classes for the LIMIT-GRAPH system.
This serves as the foundation for all other LIMIT-GRAPH modules.
"""

import json
import uuid
import networkx as nx
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

@dataclass
class LimitGraphNode:
    """
    Core node structure for LIMIT-GRAPH
    Represents documents, entities, or predicates in the semantic graph
    """
    node_id: str
    node_type: str  # "document", "entity", "predicate", "query"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    relevance_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class LimitGraphEdge:
    """
    Core edge structure for LIMIT-GRAPH
    Represents semantic relations (likes, dislikes, owns, etc.)
    """
    source: str
    target: str
    relation: str  # "likes", "dislikes", "owns", "contains", "located_in", etc.
    confidence: float = 0.8
    provenance: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class LimitQuery:
    """
    LIMIT query structure matching the JSON format from the images
    {
      "query_id": "q42",
      "query": "Who likes apples?",
      "relevant_docs": ["d12", "d27"],
      "graph_edges": [
        {"source": "d12", "target": "apples", "relation": "likes"},
        {"source": "d27", "target": "apples", "relation": "likes"}
      ]
    }
    """
    query_id: str      # "q42"
    query: str         # "Who likes apples?"
    relevant_docs: List[str]  # ["d12", "d27"]
    graph_edges: List[Dict[str, str]]  # [{"source": "d12", "target": "apples", "relation": "likes"}]
    expected_relations: List[str] = field(default_factory=list)
    complexity_level: str = "medium"

@dataclass
class RetrievalResult:
    """Result from hybrid retrieval system"""
    query_id: str
    query: str
    retrieved_docs: List[str]
    component_scores: Dict[str, Dict[str, float]]  # {"sparse": {"d12": 0.8}, "dense": {...}}
    fusion_scores: Dict[str, float]  # Final fused scores
    graph_coverage: float
    provenance_integrity: float
    trace_replay_accuracy: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for LIMIT-GRAPH"""
    # Standard LIMIT metrics
    recall_at_k: Dict[int, float] = field(default_factory=dict)  # {1: 0.8, 5: 0.9, 10: 0.95}
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    
    # Extended metrics for agentic memory
    graph_coverage: float = 0.0  # % of relevant edges traversed
    provenance_integrity: float = 0.0  # % of answers with correct source lineage
    trace_replay_accuracy: float = 0.0  # ability to reconstruct memory evolution
    
    # Component-wise performance
    component_recall: Dict[str, Dict[int, float]] = field(default_factory=dict)  # {"sparse": {10: 0.7}}
    fusion_effectiveness: float = 0.0  # improvement over best individual component

class BaseLimitGraphComponent:
    """Base class for all LIMIT-GRAPH components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.component_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.metadata = {}
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "component_id": self.component_id,
            "component_type": self.__class__.__name__,
            "created_at": self.created_at.isoformat(),
            "config": self.config,
            "metadata": self.metadata
        }

class LimitGraphRegistry:
    """Registry for managing LIMIT-GRAPH components and their relationships"""
    
    def __init__(self):
        self.components: Dict[str, BaseLimitGraphComponent] = {}
        self.component_dependencies: Dict[str, List[str]] = {}
        self.integration_status: Dict[str, str] = {}
    
    def register_component(self, component: BaseLimitGraphComponent, dependencies: List[str] = None):
        """Register a component with optional dependencies"""
        self.components[component.component_id] = component
        self.component_dependencies[component.component_id] = dependencies or []
        self.integration_status[component.component_id] = "registered"
        
        print(f"📝 Registered component: {component.__class__.__name__} ({component.component_id})")
    
    def get_component(self, component_id: str) -> Optional[BaseLimitGraphComponent]:
        """Get component by ID"""
        return self.components.get(component_id)
    
    def get_components_by_type(self, component_type: str) -> List[BaseLimitGraphComponent]:
        """Get all components of a specific type"""
        return [comp for comp in self.components.values() 
                if comp.__class__.__name__ == component_type]
    
    def check_dependencies(self, component_id: str) -> bool:
        """Check if all dependencies for a component are satisfied"""
        dependencies = self.component_dependencies.get(component_id, [])
        return all(dep_id in self.components for dep_id in dependencies)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        return {
            "total_components": len(self.components),
            "component_types": list(set(comp.__class__.__name__ for comp in self.components.values())),
            "integration_status": self.integration_status,
            "dependency_graph": self.component_dependencies
        }

# Global registry instance
LIMIT_GRAPH_REGISTRY = LimitGraphRegistry()

# Utility functions
def create_sample_limit_dataset() -> List[Dict[str, Any]]:
    """Create sample LIMIT dataset for testing and demos"""
    return [
        {
            "query_id": "q42",
            "query": "Who likes apples?",
            "relevant_docs": ["d12", "d27"],
            "graph_edges": [
                {"source": "d12", "target": "apples", "relation": "likes"},
                {"source": "d27", "target": "apples", "relation": "likes"}
            ]
        },
        {
            "query_id": "q43",
            "query": "What does John own?",
            "relevant_docs": ["d15", "d33"],
            "graph_edges": [
                {"source": "John", "target": "car", "relation": "owns"},
                {"source": "John", "target": "house", "relation": "owns"}
            ]
        },
        {
            "query_id": "q44",
            "query": "Where is the library located?",
            "relevant_docs": ["d8", "d19"],
            "graph_edges": [
                {"source": "library", "target": "downtown", "relation": "located_in"},
                {"source": "library", "target": "Main Street", "relation": "located_in"}
            ]
        }
    ]

def validate_limit_query(query_data: Dict[str, Any]) -> bool:
    """Validate LIMIT query structure"""
    required_fields = ["query_id", "query", "relevant_docs", "graph_edges"]
    return all(field in query_data for field in required_fields)

def convert_to_limit_query(query_data: Dict[str, Any]) -> LimitQuery:
    """Convert dictionary to LimitQuery object"""
    if not validate_limit_query(query_data):
        raise ValueError(f"Invalid LIMIT query structure: {query_data}")
    
    return LimitQuery(
        query_id=query_data["query_id"],
        query=query_data["query"],
        relevant_docs=query_data["relevant_docs"],
        graph_edges=query_data["graph_edges"],
        expected_relations=query_data.get("expected_relations", []),
        complexity_level=query_data.get("complexity_level", "medium")
    )