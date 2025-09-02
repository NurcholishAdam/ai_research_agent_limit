#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH Benchmark Implementation
Implements the complete LIMIT-GRAPH benchmark system with graph construction,
hybrid retrieval, and evaluation components. This is the main benchmark interface
that integrates with the core LIMIT-GRAPH architecture.
"""

import json
import spacy
import networkx as nx
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

# Import core components
from limit_graph_core import (
    LimitGraphNode, LimitGraphEdge, LimitQuery, RetrievalResult, EvaluationMetrics,
    BaseLimitGraphComponent, LIMIT_GRAPH_REGISTRY, create_sample_limit_dataset,
    convert_to_limit_query
)

# Import existing components
try:
    from memory_r1_modular import MemoryR1Enhanced, GraphTriple, GraphFragment
    from ci_hooks_integration import CIHooksValidator, CITestResult
    MEMORY_R1_AVAILABLE = True
except ImportError:
    MEMORY_R1_AVAILABLE = False
    print("⚠️ Memory-R1 system not available")

class LimitGraphBenchmark(BaseLimitGraphComponent):
    """
    Main LIMIT-GRAPH Benchmark class that orchestrates all components
    This serves as the primary interface for the benchmark system
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Initialize components
        self.graph_scaffold = None
        self.hybrid_retriever = None
        self.evaluator = None
        self.stress_tester = None
        
        # Register with global registry
        LIMIT_GRAPH_REGISTRY.register_component(self)
        
        print("🏗️ LIMIT-GRAPH Benchmark initialized")
    
    def initialize_components(self):
        """Initialize all benchmark components"""
        
        # Import and initialize components
        from limit_graph_scaffold import LimitGraphScaffold
        from limit_graph_system import HybridAgentRetriever, LimitGraphEvaluator, LimitGraphStressTests
        
        # Initialize scaffold
        self.graph_scaffold = LimitGraphScaffold(self.config.get("scaffold", {}))
        LIMIT_GRAPH_REGISTRY.register_component(self.graph_scaffold)
        
        # Initialize retriever
        self.hybrid_retriever = HybridAgentRetriever(
            self.graph_scaffold, 
            self.config.get("retriever", {})
        )
        LIMIT_GRAPH_REGISTRY.register_component(
            self.hybrid_retriever, 
            dependencies=[self.graph_scaffold.component_id]
        )
        
        # Initialize evaluator
        self.evaluator = LimitGraphEvaluator(
            self.graph_scaffold, 
            self.hybrid_retriever
        )
        LIMIT_GRAPH_REGISTRY.register_component(
            self.evaluator,
            dependencies=[self.graph_scaffold.component_id, self.hybrid_retriever.component_id]
        )
        
        # Initialize stress tester
        self.stress_tester = LimitGraphStressTests(self.evaluator)
        LIMIT_GRAPH_REGISTRY.register_component(
            self.stress_tester,
            dependencies=[self.evaluator.component_id]
        )
        
        print("✅ All LIMIT-GRAPH components initialized")
    
    def run_benchmark(self, corpus_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run complete benchmark evaluation"""
        
        if not self.graph_scaffold:
            self.initialize_components()
        
        # Use sample data if none provided
        if corpus_data is None:
            corpus_data = create_sample_limit_dataset()
        
        # Process corpus
        corpus_stats = self.graph_scaffold.process_limit_corpus(corpus_data)
        
        # Convert to LimitQuery objects
        test_queries = [convert_to_limit_query(item) for item in corpus_data]
        
        # Run evaluation
        eval_results = self.evaluator.evaluate_retrieval(test_queries)
        
        # Run stress tests
        stress_results = self.stress_tester.run_stress_tests()
        
        # Compile benchmark results
        benchmark_results = {
            "benchmark_id": self.component_id,
            "timestamp": datetime.now().isoformat(),
            "corpus_stats": corpus_stats,
            "evaluation_results": eval_results,
            "stress_test_results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "message": r.message,
                    "execution_time": r.execution_time
                }
                for r in stress_results
            ],
            "component_status": LIMIT_GRAPH_REGISTRY.get_integration_status()
        }
        
        return benchmark_results

class LimitGraphConstructor:
    """Constructs semantic graph from LIMIT corpus using spaCy + custom rules"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Graph storage
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, LimitGraphNode] = {}
        self.edges: List[LimitGraphEdge] = []
        
        # Extraction rules
        self.relation_patterns = {
            "likes": ["likes", "enjoys", "prefers", "loves", "favors"],
            "dislikes": ["dislikes", "hates", "avoids", "despises"],
            "owns": ["owns", "has", "possesses", "holds"],
            "contains": ["contains", "includes", "comprises", "has"],
            "located_in": ["in", "at", "located", "situated"],
            "part_of": ["part of", "belongs to", "member of"],
            "causes": ["causes", "leads to", "results in", "triggers"],
            "similar_to": ["similar to", "like", "resembles", "comparable to"]
        }
        
        print("🏗️ LIMIT-GRAPH Constructor initialized")
    
    def process_limit_corpus(self, corpus_path: str) -> Dict[str, Any]:
        """Process LIMIT corpus and extract semantic graph"""
        
        print(f"📚 Processing LIMIT corpus from {corpus_path}")
        
        # Load LIMIT data
        limit_data = self._load_limit_data(corpus_path)
        
        # Extract nodes and edges
        extraction_stats = {
            "documents_processed": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "graph_nodes": 0,
            "graph_edges": 0
        }
        
        for query_data in limit_data:
            # Process query
            self._process_query(query_data)
            
            # Process relevant documents
            for doc_id in query_data.get("relevant_docs", []):
                doc_content = self._get_document_content(doc_id, corpus_path)
                if doc_content:
                    self._extract_from_document(doc_id, doc_content)
                    extraction_stats["documents_processed"] += 1
            
            # Process graph edges if provided
            for edge_data in query_data.get("graph_edges", []):
                self._add_graph_edge(edge_data)
                extraction_stats["relations_extracted"] += 1
        
        # Update statistics
        extraction_stats["graph_nodes"] = len(self.nodes)
        extraction_stats["graph_edges"] = len(self.edges)
        extraction_stats["entities_extracted"] = len([n for n in self.nodes.values() if n.node_type == "entity"])
        
        print(f"✅ Corpus processing completed: {extraction_stats}")
        return extraction_stats
    
    def _load_limit_data(self, corpus_path: str) -> List[Dict[str, Any]]:
        """Load LIMIT dataset"""
        
        # Sample LIMIT-style data structure
        sample_data = [
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
        
        # Try to load actual data if available
        corpus_file = Path(corpus_path)
        if corpus_file.exists():
            try:
                with open(corpus_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading corpus, using sample data: {e}")
        
        return sample_data
    
    def _get_document_content(self, doc_id: str, corpus_path: str) -> Optional[str]:
        """Get document content by ID"""
        
        # Sample document contents
        sample_docs = {
            "d12": "Alice likes apples and enjoys eating them daily.",
            "d27": "Bob really likes apples, especially green ones.",
            "d15": "John owns a red car and a beautiful house.",
            "d33": "John has many possessions including a car.",
            "d8": "The library is located in downtown area.",
            "d19": "You can find the library on Main Street downtown."
        }
        
        return sample_docs.get(doc_id)
    
    def _process_query(self, query_data: Dict[str, Any]):
        """Process query and add to graph"""
        
        query_id = query_data["query_id"]
        query_text = query_data["query"]
        
        # Add query as a node
        query_node = LimitGraphNode(
            node_id=f"query_{query_id}",
            node_type="query",
            content=query_text,
            metadata={"original_id": query_id}
        )
        
        self.nodes[query_node.node_id] = query_node
        self.graph.add_node(query_node.node_id, **query_node.metadata)
    
    def _extract_from_document(self, doc_id: str, content: str):
        """Extract entities and relations from document"""
        
        if not self.nlp:
            return
        
        # Add document node
        doc_node = LimitGraphNode(
            node_id=doc_id,
            node_type="document",
            content=content,
            metadata={"length": len(content)}
        )
        
        self.nodes[doc_id] = doc_node
        self.graph.add_node(doc_id, **doc_node.metadata)
        
        # Process with spaCy
        doc = self.nlp(content)
        
        # Extract entities
        for ent in doc.ents:
            entity_id = f"entity_{ent.text.lower().replace(' ', '_')}"
            
            if entity_id not in self.nodes:
                entity_node = LimitGraphNode(
                    node_id=entity_id,
                    node_type="entity",
                    content=ent.text,
                    metadata={"label": ent.label_, "start": ent.start, "end": ent.end}
                )
                
                self.nodes[entity_id] = entity_node
                self.graph.add_node(entity_id, **entity_node.metadata)
            
            # Add document-entity edge
            doc_entity_edge = LimitGraphEdge(
                source=doc_id,
                target=entity_id,
                relation="contains",
                confidence=0.9,
                provenance=[doc_id]
            )
            
            self.edges.append(doc_entity_edge)
            self.graph.add_edge(doc_id, entity_id, relation="contains")
        
        # Extract relations using patterns
        self._extract_relations_from_text(doc_id, content)
    
    def _extract_relations_from_text(self, doc_id: str, text: str):
        """Extract relations using pattern matching"""
        
        text_lower = text.lower()
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Simple extraction - find entities around the pattern
                    parts = text_lower.split(pattern)
                    if len(parts) >= 2:
                        # Extract potential subject and object
                        subject_part = parts[0].strip().split()[-3:]  # Last 3 words before pattern
                        object_part = parts[1].strip().split()[:3]    # First 3 words after pattern
                        
                        if subject_part and object_part:
                            subject = "_".join(subject_part).strip(".,!?")
                            object_text = "_".join(object_part).strip(".,!?")
                            
                            # Create entity nodes if they don't exist
                            subject_id = f"entity_{subject}"
                            object_id = f"entity_{object_text}"
                            
                            for entity_id, entity_text in [(subject_id, " ".join(subject_part)), 
                                                         (object_id, " ".join(object_part))]:
                                if entity_id not in self.nodes:
                                    entity_node = LimitGraphNode(
                                        node_id=entity_id,
                                        node_type="entity",
                                        content=entity_text,
                                        metadata={"extracted_from": doc_id}
                                    )
                                    
                                    self.nodes[entity_id] = entity_node
                                    self.graph.add_node(entity_id, **entity_node.metadata)
                            
                            # Add relation edge
                            relation_edge = LimitGraphEdge(
                                source=subject_id,
                                target=object_id,
                                relation=relation_type,
                                confidence=0.7,
                                provenance=[doc_id],
                                metadata={"pattern": pattern}
                            )
                            
                            self.edges.append(relation_edge)
                            self.graph.add_edge(subject_id, object_id, relation=relation_type)
    
    def _add_graph_edge(self, edge_data: Dict[str, str]):
        """Add explicit graph edge from LIMIT data"""
        
        source = edge_data["source"]
        target = edge_data["target"]
        relation = edge_data["relation"]
        
        # Ensure nodes exist
        for node_id, node_content in [(source, source), (target, target)]:
            if node_id not in self.nodes:
                # Determine node type
                node_type = "document" if node_id.startswith("d") else "entity"
                
                node = LimitGraphNode(
                    node_id=node_id,
                    node_type=node_type,
                    content=node_content,
                    metadata={"from_graph_edges": True}
                )
                
                self.nodes[node_id] = node
                self.graph.add_node(node_id, **node.metadata)
        
        # Add edge
        graph_edge = LimitGraphEdge(
            source=source,
            target=target,
            relation=relation,
            confidence=1.0,  # High confidence for explicit edges
            provenance=["limit_qrels"],
            metadata={"explicit": True}
        )
        
        self.edges.append(graph_edge)
        self.graph.add_edge(source, target, relation=relation)
    
    def export_to_networkx(self) -> nx.MultiDiGraph:
        """Export graph to NetworkX format"""
        return self.graph.copy()
    
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
                    **edge.metadata
                }
            }
            neo4j_data["relationships"].append(neo4j_rel)
        
        return neo4j_data
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        
        return {
            "nodes": {
                "total": len(self.nodes),
                "by_type": {
                    node_type: len([n for n in self.nodes.values() if n.node_type == node_type])
                    for node_type in set(n.node_type for n in self.nodes.values())
                }
            },
            "edges": {
                "total": len(self.edges),
                "by_relation": {
                    relation: len([e for e in self.edges if e.relation == relation])
                    for relation in set(e.relation for e in self.edges)
                }
            },
            "graph_metrics": {
                "density": nx.density(self.graph),
                "connected_components": nx.number_weakly_connected_components(self.graph),
                "average_clustering": nx.average_clustering(self.graph.to_undirected()) if len(self.graph) > 0 else 0
            }
        }

class HybridAgentRetriever:
    """
    Hybrid retrieval system implementing the architecture from the images:
    
    Retrieval Flow (Mermaid diagram):
    graph TD
    Q[Query] --> S[BM25]
    Q --> D[Dense Retriever] 
    Q --> G[Graph Reasoner]
    S --> M[Memory Agent]
    D --> M
    G --> M
    M --> A[Answer Generator]
    
    Components Table:
    - Sparse Retriever: BM25 baseline for lexical grounding
    - Dense Retriever: Multi-vector (e.g., ColBERT) for semantic matching  
    - Graph Reasoner: Traverses relevance graph to find indirect matches
    - Memory Agent: Uses RL to select, update, and distill memory entries
    """
    
    def __init__(self, graph_constructor: LimitGraphConstructor, config: Dict[str, Any] = None):
        self.graph_constructor = graph_constructor
        self.config = config or {}
        
        # Initialize retrieval components as per architecture
        self.sparse_retriever = SparseRetriever(config.get("sparse", {}))  # BM25 baseline
        self.dense_retriever = DenseRetriever(config.get("dense", {}))     # ColBERT-style
        self.graph_reasoner = GraphReasoner(graph_constructor.graph, config.get("graph", {}))  # Graph traversal
        self.memory_agent = MemoryAgent(config.get("memory", {}))          # RL-based memory
        
        # Fusion strategy - Memory Agent learns to fuse signals
        self.fusion_strategy = config.get("fusion_strategy", "memory_agent_fusion")
        self.fusion_weights = config.get("fusion_weights", {
            "sparse": 0.25,    # BM25 weight
            "dense": 0.35,     # Dense retriever weight  
            "graph": 0.25,     # Graph reasoner weight
            "memory": 0.15     # Memory agent weight
        })
        
        # Provenance and trace buffer for memory operations
        self.provenance_tracker = ProvenanceTracker()
        self.trace_buffer = TraceBuffer(max_size=config.get("trace_buffer_size", 1000))
        
        print("🔍 Hybrid Agent Retriever initialized with 4-component architecture")
    
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Perform hybrid retrieval using all components"""
        
        query_id = f"query_{uuid.uuid4().hex[:8]}"
        
        # Get results from each retriever
        sparse_results = self.sparse_retriever.retrieve(query, top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k)
        graph_results = self.graph_reasoner.retrieve(query, top_k)
        memory_results = self.memory_agent.retrieve(query, top_k)
        
        # Fuse results
        fused_results = self._fuse_results({
            "sparse": sparse_results,
            "dense": dense_results,
            "graph": graph_results,
            "memory": memory_results
        }, top_k)
        
        # Calculate metrics
        graph_coverage = self._calculate_graph_coverage(query, fused_results["retrieved_docs"])
        provenance_integrity = self._calculate_provenance_integrity(fused_results)
        
        return RetrievalResult(
            query_id=query_id,
            retrieved_docs=fused_results["retrieved_docs"],
            retrieval_scores=fused_results["scores"],
            graph_coverage=graph_coverage,
            provenance_integrity=provenance_integrity,
            fusion_strategy=self.fusion_strategy,
            metadata={
                "component_results": {
                    "sparse": len(sparse_results),
                    "dense": len(dense_results),
                    "graph": len(graph_results),
                    "memory": len(memory_results)
                }
            }
        )
    
    def _fuse_results(self, component_results: Dict[str, List[str]], top_k: int) -> Dict[str, Any]:
        """Fuse results from different retrieval components"""
        
        # Collect all documents with scores
        doc_scores = defaultdict(lambda: {"total": 0.0, "components": {}})
        
        for component, results in component_results.items():
            weight = self.fusion_weights.get(component, 0.25)
            
            for i, doc_id in enumerate(results):
                # Score based on rank (higher rank = lower score)
                rank_score = (len(results) - i) / len(results)
                weighted_score = rank_score * weight
                
                doc_scores[doc_id]["total"] += weighted_score
                doc_scores[doc_id]["components"][component] = rank_score
        
        # Sort by total score and take top_k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["total"], reverse=True)
        
        retrieved_docs = [doc_id for doc_id, _ in sorted_docs[:top_k]]
        scores = {doc_id: score_data["total"] for doc_id, score_data in sorted_docs[:top_k]}
        
        return {
            "retrieved_docs": retrieved_docs,
            "scores": scores,
            "fusion_details": {doc_id: score_data for doc_id, score_data in sorted_docs[:top_k]}
        }
    
    def _calculate_graph_coverage(self, query: str, retrieved_docs: List[str]) -> float:
        """Calculate percentage of relevant graph edges traversed"""
        
        # Find query-relevant edges
        relevant_edges = []
        for edge in self.graph_constructor.edges:
            if any(doc_id in [edge.source, edge.target] for doc_id in retrieved_docs):
                relevant_edges.append(edge)
        
        # Calculate coverage
        if not relevant_edges:
            return 0.0
        
        traversed_edges = len(relevant_edges)
        total_possible_edges = len(self.graph_constructor.edges)
        
        return traversed_edges / max(total_possible_edges, 1)
    
    def _calculate_provenance_integrity(self, fused_results: Dict[str, Any]) -> float:
        """Calculate percentage of answers with correct source lineage"""
        
        # Check if retrieved documents have proper provenance
        docs_with_provenance = 0
        total_docs = len(fused_results["retrieved_docs"])
        
        for doc_id in fused_results["retrieved_docs"]:
            # Check if document has provenance in graph
            doc_edges = [e for e in self.graph_constructor.edges if e.source == doc_id or e.target == doc_id]
            if doc_edges and any(e.provenance for e in doc_edges):
                docs_with_provenance += 1
        
        return docs_with_provenance / max(total_docs, 1)

class SparseRetriever:
    """BM25 baseline for lexical grounding"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k1 = config.get("k1", 1.2)
        self.b = config.get("b", 0.75)
        
        # Simple document collection for demo
        self.documents = {
            "d12": "Alice likes apples and enjoys eating them daily.",
            "d27": "Bob really likes apples, especially green ones.",
            "d15": "John owns a red car and a beautiful house.",
            "d33": "John has many possessions including a car.",
            "d8": "The library is located in downtown area.",
            "d19": "You can find the library on Main Street downtown."
        }
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve using BM25 scoring"""
        
        query_terms = query.lower().split()
        doc_scores = {}
        
        for doc_id, doc_content in self.documents.items():
            doc_terms = doc_content.lower().split()
            score = 0.0
            
            for term in query_terms:
                if term in doc_terms:
                    tf = doc_terms.count(term)
                    # Simplified BM25 calculation
                    score += tf / (tf + self.k1 * (1 - self.b + self.b * len(doc_terms) / 10))
            
            doc_scores[doc_id] = score
        
        # Sort by score and return top_k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k] if _ > 0]

class DenseRetriever:
    """Multi-vector (e.g., ColBERT) for semantic matching"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_dim = config.get("embedding_dim", 768)
        
        # Mock embeddings for demo
        self.doc_embeddings = {
            "d12": np.random.random(self.embedding_dim),
            "d27": np.random.random(self.embedding_dim),
            "d15": np.random.random(self.embedding_dim),
            "d33": np.random.random(self.embedding_dim),
            "d8": np.random.random(self.embedding_dim),
            "d19": np.random.random(self.embedding_dim)
        }
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve using dense embeddings"""
        
        # Mock query embedding
        query_embedding = np.random.random(self.embedding_dim)
        
        # Calculate cosine similarity
        doc_scores = {}
        for doc_id, doc_embedding in self.doc_embeddings.items():
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            doc_scores[doc_id] = similarity
        
        # Sort by similarity and return top_k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

class GraphReasoner:
    """Traverses relevance graph to find indirect matches"""
    
    def __init__(self, graph: nx.MultiDiGraph, config: Dict[str, Any]):
        self.graph = graph
        self.config = config
        self.max_hops = config.get("max_hops", 3)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve using graph traversal"""
        
        query_terms = query.lower().split()
        relevant_nodes = []
        
        # Find nodes matching query terms
        for node_id, node_data in self.graph.nodes(data=True):
            node_content = node_data.get("content", "").lower()
            if any(term in node_content for term in query_terms):
                relevant_nodes.append(node_id)
        
        # Expand through graph traversal
        expanded_nodes = set(relevant_nodes)
        for start_node in relevant_nodes:
            # BFS traversal up to max_hops
            visited = set()
            queue = [(start_node, 0)]
            
            while queue:
                node, depth = queue.pop(0)
                if depth >= self.max_hops or node in visited:
                    continue
                
                visited.add(node)
                expanded_nodes.add(node)
                
                # Add neighbors
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        
        # Filter for document nodes and return top_k
        doc_nodes = [node for node in expanded_nodes if node.startswith("d")]
        return doc_nodes[:top_k]

class MemoryAgent:
    """Uses RL to select, update, and distill memory entries"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_entries = {}
        self.access_counts = defaultdict(int)
        
        # Initialize with some memory entries
        self.memory_entries = {
            "mem_1": {"content": "People often like fruits", "relevance": 0.8},
            "mem_2": {"content": "Ownership relations are important", "relevance": 0.7},
            "mem_3": {"content": "Location information is frequently queried", "relevance": 0.9}
        }
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve using memory-based reasoning"""
        
        query_lower = query.lower()
        memory_scores = {}
        
        for mem_id, mem_data in self.memory_entries.items():
            content = mem_data["content"].lower()
            relevance = mem_data["relevance"]
            
            # Simple relevance scoring
            score = 0.0
            for word in query_lower.split():
                if word in content:
                    score += relevance
            
            if score > 0:
                memory_scores[mem_id] = score
                self.access_counts[mem_id] += 1
        
        # Sort by score and return top_k
        sorted_memories = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return [mem_id for mem_id, _ in sorted_memories[:top_k]]
    
    def update_memory(self, query: str, results: List[str], feedback: float):
        """Update memory based on retrieval feedback"""
        
        # Simple memory update based on feedback
        new_mem_id = f"mem_{len(self.memory_entries) + 1}"
        self.memory_entries[new_mem_id] = {
            "content": f"Query pattern: {query}",
            "relevance": max(0.1, min(1.0, feedback))
        }

class LimitGraphEvaluator:
    """Evaluation harness for LIMIT-GRAPH with agentic memory extensions"""
    
    def __init__(self, graph_constructor: LimitGraphConstructor, retriever: HybridAgentRetriever):
        self.graph_constructor = graph_constructor
        self.retriever = retriever
        
        # Evaluation metrics
        self.metrics = {
            "recall_at_k": {},
            "graph_coverage": {},
            "provenance_integrity": {},
            "trace_replay_accuracy": {}
        }
        
        print("📊 LIMIT-GRAPH Evaluator initialized")
    
    def evaluate_retrieval(self, test_queries: List[LimitQuery], k_values: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """Evaluate retrieval performance"""
        
        print(f"🧪 Evaluating retrieval on {len(test_queries)} queries")
        
        results = {
            "per_query": {},
            "aggregate": {
                "recall_at_k": {k: [] for k in k_values},
                "graph_coverage": [],
                "provenance_integrity": [],
                "component_performance": {
                    "sparse": [],
                    "dense": [],
                    "graph": [],
                    "memory": []
                }
            }
        }
        
        for query in test_queries:
            query_results = self._evaluate_single_query(query, k_values)
            results["per_query"][query.query_id] = query_results
            
            # Aggregate metrics
            for k in k_values:
                results["aggregate"]["recall_at_k"][k].append(query_results[f"recall_at_{k}"])
            
            results["aggregate"]["graph_coverage"].append(query_results["graph_coverage"])
            results["aggregate"]["provenance_integrity"].append(query_results["provenance_integrity"])
        
        # Calculate averages
        for k in k_values:
            results["aggregate"][f"avg_recall_at_{k}"] = np.mean(results["aggregate"]["recall_at_k"][k])
        
        results["aggregate"]["avg_graph_coverage"] = np.mean(results["aggregate"]["graph_coverage"])
        results["aggregate"]["avg_provenance_integrity"] = np.mean(results["aggregate"]["provenance_integrity"])
        
        print(f"✅ Evaluation completed")
        return results
    
    def _evaluate_single_query(self, query: LimitQuery, k_values: List[int]) -> Dict[str, Any]:
        """Evaluate single query"""
        
        # Perform retrieval
        retrieval_result = self.retriever.retrieve(query.query, max(k_values))
        
        # Calculate recall@k for each k
        query_results = {}
        relevant_docs = set(query.relevant_docs)
        
        for k in k_values:
            retrieved_at_k = set(retrieval_result.retrieved_docs[:k])
            recall = len(retrieved_at_k.intersection(relevant_docs)) / len(relevant_docs) if relevant_docs else 0
            query_results[f"recall_at_{k}"] = recall
        
        # Add other metrics
        query_results["graph_coverage"] = retrieval_result.graph_coverage
        query_results["provenance_integrity"] = retrieval_result.provenance_integrity
        query_results["fusion_strategy"] = retrieval_result.fusion_strategy
        
        return query_results
    
    def evaluate_trace_replay_accuracy(self, memory_system=None) -> Dict[str, Any]:
        """Evaluate ability to reconstruct memory evolution"""
        
        if not memory_system or not MEMORY_R1_AVAILABLE:
            return {"error": "Memory system not available for trace replay evaluation"}
        
        try:
            # Use CI hooks for trace replay evaluation
            validator = CIHooksValidator(memory_system)
            
            # Test trace replay
            replay_result = validator.replay_trace(0, 5)
            
            return {
                "trace_replay_status": replay_result.status,
                "trace_replay_accuracy": 1.0 if replay_result.status == "pass" else 0.0,
                "details": replay_result.details,
                "message": replay_result.message
            }
        
        except Exception as e:
            return {"error": str(e), "trace_replay_accuracy": 0.0}
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report"""
        
        report = f"""
# LIMIT-GRAPH Evaluation Report

## Overview
- Queries evaluated: {len(results.get('per_query', {}))}
- Evaluation timestamp: {datetime.now().isoformat()}

## Retrieval Performance

### Recall@K
"""
        
        for k in [1, 5, 10]:
            if f"avg_recall_at_{k}" in results.get("aggregate", {}):
                avg_recall = results["aggregate"][f"avg_recall_at_{k}"]
                report += f"- Recall@{k}: {avg_recall:.3f}\n"
        
        report += f"""
### Graph-based Metrics
- Average Graph Coverage: {results.get('aggregate', {}).get('avg_graph_coverage', 0):.3f}
- Average Provenance Integrity: {results.get('aggregate', {}).get('avg_provenance_integrity', 0):.3f}

### Component Analysis
"""
        
        # Add component performance if available
        component_perf = results.get("aggregate", {}).get("component_performance", {})
        for component, scores in component_perf.items():
            if scores:
                report += f"- {component.capitalize()}: {np.mean(scores):.3f}\n"
        
        return report

# CI-evaluable stress tests
class LimitGraphStressTests:
    """CI-evaluable stress tests for memory-aware agents"""
    
    def __init__(self, graph_constructor: LimitGraphConstructor, retriever: HybridAgentRetriever):
        self.graph_constructor = graph_constructor
        self.retriever = retriever
    
    def run_stress_tests(self) -> List[CITestResult]:
        """Run comprehensive stress tests"""
        
        test_results = []
        
        # Test 1: Graph construction stress test
        test_results.append(self._test_graph_construction_stress())
        
        # Test 2: Retrieval scalability test
        test_results.append(self._test_retrieval_scalability())
        
        # Test 3: Memory consistency test
        test_results.append(self._test_memory_consistency())
        
        # Test 4: Provenance integrity stress test
        test_results.append(self._test_provenance_integrity_stress())
        
        return test_results
    
    def _test_graph_construction_stress(self) -> CITestResult:
        """Test graph construction under stress"""
        
        start_time = datetime.now()
        
        try:
            # Simulate large corpus processing
            large_corpus_size = 1000
            nodes_created = 0
            edges_created = 0
            
            for i in range(large_corpus_size):
                # Simulate document processing
                doc_id = f"stress_doc_{i}"
                content = f"This is stress test document {i} with various entities and relations."
                
                # Add to graph (simplified)
                if doc_id not in self.graph_constructor.nodes:
                    node = LimitGraphNode(
                        node_id=doc_id,
                        node_type="document",
                        content=content
                    )
                    self.graph_constructor.nodes[doc_id] = node
                    nodes_created += 1
                
                # Add some edges
                if i > 0:
                    edge = LimitGraphEdge(
                        source=f"stress_doc_{i-1}",
                        target=doc_id,
                        relation="follows"
                    )
                    self.graph_constructor.edges.append(edge)
                    edges_created += 1
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Check if performance is acceptable
            if execution_time < 10.0 and nodes_created > 0:  # Should complete in under 10 seconds
                status = "pass"
                message = f"Graph construction stress test passed: {nodes_created} nodes, {edges_created} edges in {execution_time:.2f}s"
            else:
                status = "fail"
                message = f"Graph construction too slow: {execution_time:.2f}s"
            
            return CITestResult(
                test_name="graph_construction_stress",
                status=status,
                message=message,
                details={
                    "nodes_created": nodes_created,
                    "edges_created": edges_created,
                    "execution_time": execution_time
                },
                execution_time=execution_time,
                timestamp=start_time
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return CITestResult(
                test_name="graph_construction_stress",
                status="fail",
                message=f"Graph construction stress test failed: {e}",
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=start_time
            )
    
    def _test_retrieval_scalability(self) -> CITestResult:
        """Test retrieval scalability"""
        
        start_time = datetime.now()
        
        try:
            # Test with multiple concurrent queries
            test_queries = [
                f"Test query {i} with various terms and complexity"
                for i in range(100)
            ]
            
            retrieval_times = []
            successful_retrievals = 0
            
            for query in test_queries:
                query_start = datetime.now()
                try:
                    result = self.retriever.retrieve(query, top_k=10)
                    query_time = (datetime.now() - query_start).total_seconds()
                    retrieval_times.append(query_time)
                    successful_retrievals += 1
                except Exception:
                    pass
            
            execution_time = (datetime.now() - start_time).total_seconds()
            avg_retrieval_time = np.mean(retrieval_times) if retrieval_times else float('inf')
            
            # Check performance criteria
            if avg_retrieval_time < 1.0 and successful_retrievals > 90:  # Average < 1s, >90% success
                status = "pass"
                message = f"Retrieval scalability test passed: {successful_retrievals}/100 queries, avg {avg_retrieval_time:.3f}s"
            else:
                status = "fail"
                message = f"Retrieval scalability issues: {successful_retrievals}/100 queries, avg {avg_retrieval_time:.3f}s"
            
            return CITestResult(
                test_name="retrieval_scalability",
                status=status,
                message=message,
                details={
                    "total_queries": len(test_queries),
                    "successful_retrievals": successful_retrievals,
                    "avg_retrieval_time": avg_retrieval_time,
                    "total_execution_time": execution_time
                },
                execution_time=execution_time,
                timestamp=start_time
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return CITestResult(
                test_name="retrieval_scalability",
                status="fail",
                message=f"Retrieval scalability test failed: {e}",
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=start_time
            )
    
    def _test_memory_consistency(self) -> CITestResult:
        """Test memory consistency under stress"""
        
        start_time = datetime.now()
        
        try:
            # Test memory agent consistency
            memory_agent = self.retriever.memory_agent
            initial_memory_count = len(memory_agent.memory_entries)
            
            # Perform multiple memory operations
            for i in range(50):
                query = f"Memory test query {i}"
                results = memory_agent.retrieve(query, top_k=5)
                memory_agent.update_memory(query, results, 0.8)
            
            final_memory_count = len(memory_agent.memory_entries)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Check consistency
            if final_memory_count > initial_memory_count and execution_time < 5.0:
                status = "pass"
                message = f"Memory consistency test passed: {initial_memory_count} -> {final_memory_count} entries"
            else:
                status = "fail"
                message = f"Memory consistency issues: {initial_memory_count} -> {final_memory_count} entries"
            
            return CITestResult(
                test_name="memory_consistency",
                status=status,
                message=message,
                details={
                    "initial_memory_count": initial_memory_count,
                    "final_memory_count": final_memory_count,
                    "memory_growth": final_memory_count - initial_memory_count
                },
                execution_time=execution_time,
                timestamp=start_time
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return CITestResult(
                test_name="memory_consistency",
                status="fail",
                message=f"Memory consistency test failed: {e}",
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=start_time
            )
    
    def _test_provenance_integrity_stress(self) -> CITestResult:
        """Test provenance integrity under stress"""
        
        start_time = datetime.now()
        
        try:
            # Test provenance tracking with many operations
            provenance_violations = 0
            total_operations = 100
            
            for i in range(total_operations):
                # Simulate retrieval operation
                query = f"Provenance test {i}"
                result = self.retriever.retrieve(query, top_k=5)
                
                # Check provenance integrity
                if result.provenance_integrity < 0.8:  # Threshold for acceptable integrity
                    provenance_violations += 1
            
            execution_time = (datetime.now() - start_time).total_seconds()
            violation_rate = provenance_violations / total_operations
            
            # Check if provenance integrity is maintained
            if violation_rate < 0.1:  # Less than 10% violations acceptable
                status = "pass"
                message = f"Provenance integrity stress test passed: {violation_rate:.1%} violation rate"
            else:
                status = "fail"
                message = f"Provenance integrity issues: {violation_rate:.1%} violation rate"
            
            return CITestResult(
                test_name="provenance_integrity_stress",
                status=status,
                message=message,
                details={
                    "total_operations": total_operations,
                    "provenance_violations": provenance_violations,
                    "violation_rate": violation_rate
                },
                execution_time=execution_time,
                timestamp=start_time
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return CITestResult(
                test_name="provenance_integrity_stress",
                status="fail",
                message=f"Provenance integrity stress test failed: {e}",
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=start_time
            )

# Utility functions
def create_limit_graph_system(config: Dict[str, Any] = None) -> Tuple[LimitGraphConstructor, HybridAgentRetriever, LimitGraphEvaluator]:
    """Create complete LIMIT-GRAPH system"""
    
    config = config or {}
    
    # Create graph constructor
    graph_constructor = LimitGraphConstructor(config.get("graph_constructor", {}))
    
    # Create hybrid retriever
    retriever = HybridAgentRetriever(graph_constructor, config.get("retriever", {}))
    
    # Create evaluator
    evaluator = LimitGraphEvaluator(graph_constructor, retriever)
    
    return graph_constructor, retriever, evaluator

def run_limit_graph_demo():
    """Run LIMIT-GRAPH demonstration"""
    
    print("🚀 LIMIT-GRAPH Benchmark Demo")
    
    # Create system
    graph_constructor, retriever, evaluator = create_limit_graph_system()
    
    # Process corpus
    corpus_stats = graph_constructor.process_limit_corpus("sample_limit_corpus.json")
    print(f"📊 Corpus processing: {corpus_stats}")
    
    # Test retrieval
    test_queries = [
        LimitQuery(
            query_id="test_1",
            query="Who likes apples?",
            relevant_docs=["d12", "d27"],
            graph_edges=[
                {"source": "d12", "target": "apples", "relation": "likes"},
                {"source": "d27", "target": "apples", "relation": "likes"}
            ]
        )
    ]
    
    # Evaluate
    eval_results = evaluator.evaluate_retrieval(test_queries)
    print(f"📈 Evaluation results: {eval_results['aggregate']}")
    
    # Run stress tests
    stress_tester = LimitGraphStressTests(graph_constructor, retriever)
    stress_results = stress_tester.run_stress_tests()
    
    print(f"🧪 Stress test results:")
    for result in stress_results:
        print(f"   {result.test_name}: {result.status} - {result.message}")
    
    # Generate report
    report = evaluator.generate_evaluation_report(eval_results)
    print(f"\n📄 Evaluation Report:\n{report}")

if __name__ == "__main__":
    run_limit_graph_demo()
