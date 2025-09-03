#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Reasoner Module
Traverses semantic relevance graphs to discover indirect document-query matches that embeddings miss.
This module implements sophisticated graph traversal algorithms for finding relevant documents
through multi-hop reasoning over semantic relationships.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import spacy
from datetime import datetime

# Import core components
from limit_graph_core import (
    LimitGraphNode, LimitGraphEdge, BaseLimitGraphComponent, 
    LIMIT_GRAPH_REGISTRY
)

@dataclass
class EntityLinkingResult:
    """Result from entity linking process"""
    query: str
    entities: List[str]
    entity_scores: Dict[str, float]
    linking_method: str
    confidence: float

@dataclass
class GraphTraversalResult:
    """Result from graph traversal"""
    query_entities: List[str]
    traversal_paths: List[List[str]]
    relevant_docs: List[str]
    relevance_scores: Dict[str, float]
    traversal_depth: int
    coverage_metrics: Dict[str, Any]

@dataclass
class ReasoningPath:
    """Represents a reasoning path through the graph"""
    path_id: str
    nodes: List[str]
    edges: List[str]
    path_score: float
    reasoning_type: str  # "direct", "indirect", "multi_hop"
    evidence: List[str]

class EntityLinker(BaseLimitGraphComponent):
    """
    Entity linking component to extract query entities
    Uses spaCy NER + custom rules + graph-based disambiguation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Initialize spaCy for entity recognition
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Using fallback entity linking.")
            self.nlp = None
        
        # Entity linking configuration
        self.min_entity_confidence = config.get("min_entity_confidence", 0.5)
        self.max_entities_per_query = config.get("max_entities_per_query", 10)
        
        # Entity patterns for custom recognition
        self.entity_patterns = {
            "PERSON": ["who", "person", "people", "someone"],
            "LOCATION": ["where", "place", "location", "city", "country"],
            "OBJECT": ["what", "thing", "object", "item"],
            "ACTION": ["does", "do", "action", "activity"]
        }
        
        # Register with global registry
        LIMIT_GRAPH_REGISTRY.register_component(self)
        
        print("🔗 Entity Linker initialized")
    
    def extract_query_entities(self, query: str, graph_nodes: Dict[str, LimitGraphNode] = None) -> EntityLinkingResult:
        """
        Extract entities from query using multiple methods:
        1. spaCy NER
        2. Custom pattern matching
        3. Graph-based disambiguation
        """
        
        entities = []
        entity_scores = {}
        
        # Method 1: spaCy NER
        if self.nlp:
            spacy_entities = self._extract_spacy_entities(query)
            entities.extend(spacy_entities)
            for entity in spacy_entities:
                entity_scores[entity] = 0.8  # High confidence for NER
        
        # Method 2: Custom pattern matching
        pattern_entities = self._extract_pattern_entities(query)
        for entity, score in pattern_entities.items():
            if entity not in entities:
                entities.append(entity)
            entity_scores[entity] = max(entity_scores.get(entity, 0), score)
        
        # Method 3: Graph-based disambiguation
        if graph_nodes:
            graph_entities = self._extract_graph_entities(query, graph_nodes)
            for entity, score in graph_entities.items():
                if entity not in entities:
                    entities.append(entity)
                entity_scores[entity] = max(entity_scores.get(entity, 0), score)
        
        # Filter by confidence and limit count
        filtered_entities = [
            entity for entity in entities 
            if entity_scores.get(entity, 0) >= self.min_entity_confidence
        ]
        
        # Sort by score and limit
        filtered_entities.sort(key=lambda x: entity_scores.get(x, 0), reverse=True)
        filtered_entities = filtered_entities[:self.max_entities_per_query]
        
        # Calculate overall confidence
        overall_confidence = np.mean([entity_scores.get(e, 0) for e in filtered_entities]) if filtered_entities else 0.0
        
        return EntityLinkingResult(
            query=query,
            entities=filtered_entities,
            entity_scores={e: entity_scores.get(e, 0) for e in filtered_entities},
            linking_method="hybrid",
            confidence=overall_confidence
        )
    
    def _extract_spacy_entities(self, query: str) -> List[str]:
        """Extract entities using spaCy NER"""
        if not self.nlp:
            return []
        
        doc = self.nlp(query)
        entities = []
        
        for ent in doc.ents:
            # Normalize entity text
            entity_text = ent.text.lower().strip()
            if len(entity_text) > 2:  # Filter out very short entities
                entities.append(entity_text)
        
        return entities
    
    def _extract_pattern_entities(self, query: str) -> Dict[str, float]:
        """Extract entities using custom patterns"""
        query_lower = query.lower()
        entities = {}
        
        # Look for question patterns
        if "who" in query_lower:
            # Extract potential person names (capitalized words)
            words = query.split()
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    entities[word.lower()] = 0.7
        
        if "what" in query_lower:
            # Extract potential objects (nouns after "what")
            words = query_lower.split()
            what_idx = words.index("what") if "what" in words else -1
            if what_idx >= 0 and what_idx < len(words) - 1:
                potential_object = words[what_idx + 1]
                if len(potential_object) > 2:
                    entities[potential_object] = 0.6
        
        if "where" in query_lower:
            # Extract potential locations
            words = query.split()
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    entities[word.lower()] = 0.7
        
        return entities
    
    def _extract_graph_entities(self, query: str, graph_nodes: Dict[str, LimitGraphNode]) -> Dict[str, float]:
        """Extract entities using graph-based disambiguation"""
        query_words = set(query.lower().split())
        entities = {}
        
        # Find nodes that match query words
        for node_id, node in graph_nodes.items():
            if node.node_type == "entity":
                node_words = set(node.content.lower().split())
                
                # Calculate word overlap
                overlap = len(query_words.intersection(node_words))
                if overlap > 0:
                    # Score based on overlap and node relevance
                    score = (overlap / len(query_words)) * node.relevance_score
                    if score > 0.3:  # Minimum threshold
                        entities[node.content.lower()] = score
        
        return entities

class GraphTraverser(BaseLimitGraphComponent):
    """
    Graph traversal component for finding relevant documents
    Implements multiple traversal strategies for different reasoning types
    """
    
    def __init__(self, graph: nx.MultiDiGraph, config: Dict[str, Any] = None):
        super().__init__(config)
        self.graph = graph
        
        # Traversal configuration
        self.max_depth = config.get("max_depth", 3)
        self.max_paths = config.get("max_paths", 100)
        self.min_path_score = config.get("min_path_score", 0.1)
        
        # Traversal strategies
        self.traversal_strategies = {
            "bfs": self._breadth_first_traversal,
            "dfs": self._depth_first_traversal,
            "weighted": self._weighted_traversal,
            "semantic": self._semantic_traversal
        }
        
        # Register with global registry
        LIMIT_GRAPH_REGISTRY.register_component(self)
        
        print("🕸️ Graph Traverser initialized")
    
    def find_relevant_docs(self, query_entities: List[str], strategy: str = "weighted") -> GraphTraversalResult:
        """
        Find relevant documents through graph traversal
        
        Args:
            query_entities: List of entities extracted from query
            strategy: Traversal strategy ("bfs", "dfs", "weighted", "semantic")
        
        Returns:
            GraphTraversalResult with paths and relevant documents
        """
        
        if strategy not in self.traversal_strategies:
            strategy = "weighted"
        
        # Find starting nodes (entities in graph)
        start_nodes = self._find_entity_nodes(query_entities)
        
        if not start_nodes:
            return GraphTraversalResult(
                query_entities=query_entities,
                traversal_paths=[],
                relevant_docs=[],
                relevance_scores={},
                traversal_depth=0,
                coverage_metrics={"coverage": 0.0, "paths_explored": 0}
            )
        
        # Execute traversal strategy
        traversal_func = self.traversal_strategies[strategy]
        paths, doc_scores = traversal_func(start_nodes)
        
        # Extract relevant documents
        relevant_docs = list(doc_scores.keys())
        
        # Calculate coverage metrics
        coverage_metrics = self._calculate_coverage_metrics(paths, relevant_docs)
        
        return GraphTraversalResult(
            query_entities=query_entities,
            traversal_paths=paths,
            relevant_docs=relevant_docs,
            relevance_scores=doc_scores,
            traversal_depth=max(len(path) for path in paths) if paths else 0,
            coverage_metrics=coverage_metrics
        )
    
    def _find_entity_nodes(self, query_entities: List[str]) -> List[str]:
        """Find graph nodes corresponding to query entities"""
        entity_nodes = []
        
        for entity in query_entities:
            # Direct match
            if entity in self.graph.nodes():
                entity_nodes.append(entity)
                continue
            
            # Fuzzy match
            for node_id in self.graph.nodes():
                if entity.lower() in node_id.lower() or node_id.lower() in entity.lower():
                    entity_nodes.append(node_id)
                    break
        
        return entity_nodes
    
    def _breadth_first_traversal(self, start_nodes: List[str]) -> Tuple[List[List[str]], Dict[str, float]]:
        """Breadth-first traversal to find documents"""
        paths = []
        doc_scores = defaultdict(float)
        visited = set()
        
        # BFS queue: (node, path, depth, score)
        queue = deque([(node, [node], 0, 1.0) for node in start_nodes])
        
        while queue and len(paths) < self.max_paths:
            current_node, path, depth, score = queue.popleft()
            
            if current_node in visited or depth >= self.max_depth:
                continue
            
            visited.add(current_node)
            
            # If this is a document node, record it
            if current_node.startswith("d"):  # Document nodes start with 'd'
                paths.append(path)
                doc_scores[current_node] = max(doc_scores[current_node], score)
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_score = score * 0.8  # Decay score with distance
                    
                    if new_score >= self.min_path_score:
                        queue.append((neighbor, new_path, depth + 1, new_score))
        
        return paths, dict(doc_scores)
    
    def _depth_first_traversal(self, start_nodes: List[str]) -> Tuple[List[List[str]], Dict[str, float]]:
        """Depth-first traversal to find documents"""
        paths = []
        doc_scores = defaultdict(float)
        
        def dfs(node: str, path: List[str], depth: int, score: float, visited: Set[str]):
            if len(paths) >= self.max_paths or depth >= self.max_depth or node in visited:
                return
            
            visited.add(node)
            
            # If this is a document node, record it
            if node.startswith("d"):
                paths.append(path + [node])
                doc_scores[node] = max(doc_scores[node], score)
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    new_score = score * 0.8
                    if new_score >= self.min_path_score:
                        dfs(neighbor, path + [node], depth + 1, new_score, visited.copy())
        
        # Start DFS from each starting node
        for start_node in start_nodes:
            dfs(start_node, [], 0, 1.0, set())
        
        return paths, dict(doc_scores)
    
    def _weighted_traversal(self, start_nodes: List[str]) -> Tuple[List[List[str]], Dict[str, float]]:
        """Weighted traversal considering edge weights and node importance"""
        paths = []
        doc_scores = defaultdict(float)
        
        # Use Dijkstra-like algorithm with custom weights
        # Priority queue: (negative_score, node, path, depth)
        import heapq
        pq = [(-1.0, node, [node], 0) for node in start_nodes]
        visited = set()
        
        while pq and len(paths) < self.max_paths:
            neg_score, current_node, path, depth = heapq.heappop(pq)
            score = -neg_score
            
            if current_node in visited or depth >= self.max_depth:
                continue
            
            visited.add(current_node)
            
            # If this is a document node, record it
            if current_node.startswith("d"):
                paths.append(path)
                doc_scores[current_node] = max(doc_scores[current_node], score)
            
            # Explore neighbors with edge weights
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    # Calculate edge weight (relation strength)
                    edge_weight = self._calculate_edge_weight(current_node, neighbor)
                    new_score = score * edge_weight
                    
                    if new_score >= self.min_path_score:
                        new_path = path + [neighbor]
                        heapq.heappush(pq, (-new_score, neighbor, new_path, depth + 1))
        
        return paths, dict(doc_scores)
    
    def _semantic_traversal(self, start_nodes: List[str]) -> Tuple[List[List[str]], Dict[str, float]]:
        """Semantic traversal considering relation types and semantic similarity"""
        paths = []
        doc_scores = defaultdict(float)
        
        # Semantic relation weights
        relation_weights = {
            "likes": 0.9,
            "owns": 0.8,
            "located_in": 0.7,
            "contains": 0.6,
            "part_of": 0.5,
            "related_to": 0.4
        }
        
        # BFS with semantic scoring
        queue = deque([(node, [node], 0, 1.0) for node in start_nodes])
        visited = set()
        
        while queue and len(paths) < self.max_paths:
            current_node, path, depth, score = queue.popleft()
            
            if current_node in visited or depth >= self.max_depth:
                continue
            
            visited.add(current_node)
            
            # If this is a document node, record it
            if current_node.startswith("d"):
                paths.append(path)
                doc_scores[current_node] = max(doc_scores[current_node], score)
            
            # Explore neighbors with semantic weights
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    # Get relation type
                    relation = self._get_edge_relation(current_node, neighbor)
                    semantic_weight = relation_weights.get(relation, 0.3)
                    
                    new_score = score * semantic_weight
                    if new_score >= self.min_path_score:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path, depth + 1, new_score))
        
        return paths, dict(doc_scores)
    
    def _calculate_edge_weight(self, source: str, target: str) -> float:
        """Calculate edge weight based on relation strength"""
        # Get edge data
        edge_data = self.graph.get_edge_data(source, target)
        if not edge_data:
            return 0.5  # Default weight
        
        # Use confidence if available
        if isinstance(edge_data, dict):
            return edge_data.get("confidence", 0.5)
        elif hasattr(edge_data, "confidence"):
            return edge_data.confidence
        
        return 0.5
    
    def _get_edge_relation(self, source: str, target: str) -> str:
        """Get relation type for edge"""
        edge_data = self.graph.get_edge_data(source, target)
        if not edge_data:
            return "related_to"
        
        if isinstance(edge_data, dict):
            return edge_data.get("relation", "related_to")
        elif hasattr(edge_data, "relation"):
            return edge_data.relation
        
        return "related_to"
    
    def _calculate_coverage_metrics(self, paths: List[List[str]], relevant_docs: List[str]) -> Dict[str, Any]:
        """Calculate coverage metrics for traversal"""
        total_nodes = len(self.graph.nodes())
        total_edges = len(self.graph.edges())
        
        # Nodes covered
        covered_nodes = set()
        for path in paths:
            covered_nodes.update(path)
        
        # Edges covered
        covered_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                covered_edges.add((path[i], path[i + 1]))
        
        return {
            "node_coverage": len(covered_nodes) / total_nodes if total_nodes > 0 else 0,
            "edge_coverage": len(covered_edges) / total_edges if total_edges > 0 else 0,
            "paths_explored": len(paths),
            "documents_found": len(relevant_docs),
            "avg_path_length": np.mean([len(path) for path in paths]) if paths else 0
        }

class GraphReasoner(BaseLimitGraphComponent):
    """
    Main Graph Reasoner Module
    Orchestrates entity linking and graph traversal to find relevant documents
    that embeddings might miss through indirect reasoning paths
    """
    
    def __init__(self, graph_scaffold, config: Dict[str, Any] = None):
        super().__init__(config)
        self.graph_scaffold = graph_scaffold
        
        # Initialize components
        self.entity_linker = EntityLinker(config.get("entity_linking", {}))
        self.graph_traverser = GraphTraverser(
            graph_scaffold.graph if hasattr(graph_scaffold, 'graph') else nx.MultiDiGraph(),
            config.get("graph_traversal", {})
        )
        
        # Reasoning configuration
        self.fusion_strategy = config.get("fusion_strategy", "weighted_combination")
        self.indirect_weight = config.get("indirect_weight", 0.7)
        self.direct_weight = config.get("direct_weight", 1.0)
        
        # Register with global registry
        LIMIT_GRAPH_REGISTRY.register_component(
            self,
            dependencies=[
                graph_scaffold.component_id if hasattr(graph_scaffold, 'component_id') else None,
                self.entity_linker.component_id,
                self.graph_traverser.component_id
            ]
        )
        
        print("🧠 Graph Reasoner initialized")
    
    def reason_and_retrieve(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Main reasoning and retrieval method
        
        Process:
        1. Extract query entities using entity linking
        2. Traverse graph to find relevant documents
        3. Score and rank documents
        4. Return results for fusion with other retrievers
        """
        
        # Step 1: Entity linking
        entity_result = self.entity_linker.extract_query_entities(
            query, 
            self.graph_scaffold.nodes if hasattr(self.graph_scaffold, 'nodes') else None
        )
        
        if not entity_result.entities:
            return {
                "query": query,
                "retrieved_docs": [],
                "reasoning_paths": [],
                "entity_linking": entity_result,
                "traversal_result": None,
                "scores": {},
                "reasoning_type": "no_entities"
            }
        
        # Step 2: Graph traversal
        traversal_result = self.graph_traverser.find_relevant_docs(
            entity_result.entities,
            strategy="weighted"
        )
        
        # Step 3: Score and rank documents
        scored_docs = self._score_documents(entity_result, traversal_result)
        
        # Step 4: Select top-k documents
        top_docs = sorted(scored_docs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Step 5: Generate reasoning paths
        reasoning_paths = self._generate_reasoning_paths(traversal_result, top_docs)
        
        return {
            "query": query,
            "retrieved_docs": [doc_id for doc_id, _ in top_docs],
            "reasoning_paths": reasoning_paths,
            "entity_linking": entity_result,
            "traversal_result": traversal_result,
            "scores": dict(top_docs),
            "reasoning_type": "graph_reasoning"
        }
    
    def _score_documents(self, entity_result: EntityLinkingResult, 
                        traversal_result: GraphTraversalResult) -> Dict[str, float]:
        """Score documents based on entity linking and graph traversal"""
        
        doc_scores = {}
        
        # Base scores from traversal
        for doc_id, score in traversal_result.relevance_scores.items():
            doc_scores[doc_id] = score
        
        # Boost scores based on entity confidence
        entity_boost = entity_result.confidence * 0.2
        for doc_id in doc_scores:
            doc_scores[doc_id] += entity_boost
        
        # Boost scores based on path quality
        for path in traversal_result.traversal_paths:
            if path and path[-1] in doc_scores:
                # Shorter paths get higher boost
                path_boost = 1.0 / len(path) * 0.3
                doc_scores[path[-1]] += path_boost
        
        # Normalize scores
        if doc_scores:
            max_score = max(doc_scores.values())
            if max_score > 0:
                doc_scores = {doc_id: score / max_score for doc_id, score in doc_scores.items()}
        
        return doc_scores
    
    def _generate_reasoning_paths(self, traversal_result: GraphTraversalResult, 
                                 top_docs: List[Tuple[str, float]]) -> List[ReasoningPath]:
        """Generate reasoning paths for top documents"""
        
        reasoning_paths = []
        
        for doc_id, score in top_docs:
            # Find paths that lead to this document
            doc_paths = [path for path in traversal_result.traversal_paths if path and path[-1] == doc_id]
            
            for i, path in enumerate(doc_paths[:3]):  # Limit to top 3 paths per document
                reasoning_type = "direct" if len(path) <= 2 else "indirect" if len(path) <= 4 else "multi_hop"
                
                reasoning_path = ReasoningPath(
                    path_id=f"{doc_id}_path_{i}",
                    nodes=path,
                    edges=[f"{path[j]}->{path[j+1]}" for j in range(len(path)-1)],
                    path_score=score,
                    reasoning_type=reasoning_type,
                    evidence=[f"Entity: {traversal_result.query_entities}"] + 
                            [f"Relation: {edge}" for edge in [f"{path[j]}->{path[j+1]}" for j in range(len(path)-1)]]
                )
                
                reasoning_paths.append(reasoning_path)
        
        return reasoning_paths
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning performance"""
        
        return {
            "component_id": self.component_id,
            "entity_linker_stats": {
                "min_confidence": self.entity_linker.min_entity_confidence,
                "max_entities": self.entity_linker.max_entities_per_query
            },
            "traverser_stats": {
                "max_depth": self.graph_traverser.max_depth,
                "max_paths": self.graph_traverser.max_paths,
                "strategies": list(self.graph_traverser.traversal_strategies.keys())
            },
            "fusion_config": {
                "strategy": self.fusion_strategy,
                "indirect_weight": self.indirect_weight,
                "direct_weight": self.direct_weight
            }
        }

# Integration functions for use in memory agent
def integrate_graph_reasoner_with_memory_agent(memory_agent, graph_reasoner: GraphReasoner):
    """
    Integration function to connect Graph Reasoner with Memory Agent
    This allows the Memory Agent to use graph reasoning results in fusion
    """
    
    # Add graph reasoner to memory agent
    memory_agent.graph_reasoner = graph_reasoner
    
    # Enhance memory agent's fusion method
    original_fuse_method = memory_agent.fuse_and_retrieve
    
    def enhanced_fuse_and_retrieve(query: str, sparse_results: List[str], 
                                  dense_results: List[str], graph_results: List[str], 
                                  top_k: int) -> Dict[str, Any]:
        
        # Get graph reasoning results
        reasoning_result = graph_reasoner.reason_and_retrieve(query, top_k)
        
        # Use reasoning results as graph_results
        enhanced_graph_results = reasoning_result["retrieved_docs"]
        
        # Call original fusion with enhanced graph results
        fusion_result = original_fuse_method(
            query, sparse_results, dense_results, enhanced_graph_results, top_k
        )
        
        # Add reasoning information to result
        fusion_result["graph_reasoning"] = {
            "entity_linking": reasoning_result["entity_linking"],
            "reasoning_paths": reasoning_result["reasoning_paths"],
            "reasoning_type": reasoning_result["reasoning_type"]
        }
        
        return fusion_result
    
    # Replace the method
    memory_agent.fuse_and_retrieve = enhanced_fuse_and_retrieve
    
    print("🔗 Graph Reasoner integrated with Memory Agent")

# Utility functions
def create_graph_reasoner(graph_scaffold, config: Dict[str, Any] = None) -> GraphReasoner:
    """Create and configure a Graph Reasoner"""
    return GraphReasoner(graph_scaffold, config)

def demo_graph_reasoning():
    """Demo function to show graph reasoning capabilities"""
    
    print("🧠 Graph Reasoning Demo")
    
    # This would typically use actual graph scaffold
    # For demo, we'll create a simple mock
    class MockGraphScaffold:
        def __init__(self):
            self.graph = nx.MultiDiGraph()
            self.nodes = {}
            
            # Add sample nodes and edges
            self.graph.add_node("apples", type="entity")
            self.graph.add_node("d12", type="document")
            self.graph.add_node("Alice", type="entity")
            self.graph.add_edge("Alice", "apples", relation="likes")
            self.graph.add_edge("d12", "apples", relation="contains")
    
    # Create reasoner
    scaffold = MockGraphScaffold()
    reasoner = create_graph_reasoner(scaffold)
    
    # Test reasoning
    result = reasoner.reason_and_retrieve("Who likes apples?", top_k=5)
    
    print(f"Query: {result['query']}")
    print(f"Retrieved docs: {result['retrieved_docs']}")
    print(f"Reasoning type: {result['reasoning_type']}")
    print(f"Entity linking: {result['entity_linking'].entities}")
    
    return result

if __name__ == "__main__":
    demo_graph_reasoning()