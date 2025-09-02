#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH System Implementation
Implements the hybrid retrieval system and evaluation components for LIMIT-GRAPH.
This module contains the core retrieval and evaluation logic.
"""

import json
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Import core components
from limit_graph_core import (
    LimitGraphNode, LimitGraphEdge, LimitQuery, RetrievalResult, EvaluationMetrics,
    BaseLimitGraphComponent, LIMIT_GRAPH_REGISTRY
)

# Import existing components
try:
    from memory_r1_modular import MemoryR1Enhanced, GraphTriple, GraphFragment
    from ci_hooks_integration import CIHooksValidator, CITestResult
    MEMORY_R1_AVAILABLE = True
except ImportError:
    MEMORY_R1_AVAILABLE = False

class HybridAgentRetriever(BaseLimitGraphComponent):
    """
    Retrieval Prototype: Hybrid Agent Retriever
    
    Components (matching the table structure from images):
    - Sparse Retriever: BM25 baseline for lexical grounding
    - Dense Retriever: Multi-vector (e.g., ColBERT) for semantic matching
    - Graph Reasoner: Traverses relevance graph to find indirect matches  
    - Memory Agent: Uses RL to select, update, and distill memory entries
    
    Retrieval Flow (from Mermaid diagram):
    Query → [BM25, Dense, Graph] → Memory Agent → Answer Generator
    """
    
    def __init__(self, graph_scaffold, config: Dict[str, Any] = None):
        super().__init__(config)
        self.graph_scaffold = graph_scaffold
        
        # Initialize retrieval components
        self.sparse_retriever = SparseRetriever(config.get("sparse", {}))
        self.dense_retriever = DenseRetriever(config.get("dense", {}))
        self.graph_reasoner = GraphReasoner(graph_scaffold.graph, config.get("graph", {}))
        self.memory_agent = MemoryAgent(config.get("memory", {}))
        
        # Register with global registry
        LIMIT_GRAPH_REGISTRY.register_component(
            self, 
            dependencies=[graph_scaffold.component_id] if hasattr(graph_scaffold, 'component_id') else []
        )
        
        print("🔍 Hybrid Agent Retriever initialized")
    
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """
        Retrieval Flow:
        Memory Agent learns to fuse signals from all retrievers and uses 
        provenance and trace buffer to define memory ops
        """
        
        query_id = f"query_{uuid.uuid4().hex[:8]}"
        
        # Get results from each component
        sparse_results = self.sparse_retriever.retrieve(query, top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k)
        graph_results = self.graph_reasoner.retrieve(query, top_k)
        
        # Memory Agent fuses signals and manages memory operations
        fused_results = self.memory_agent.fuse_and_retrieve(
            query=query,
            sparse_results=sparse_results,
            dense_results=dense_results,
            graph_results=graph_results,
            top_k=top_k
        )
        
        # Calculate extended metrics
        graph_coverage = self._calculate_graph_coverage(query, fused_results["retrieved_docs"])
        provenance_integrity = self._calculate_provenance_integrity(fused_results)
        trace_replay_accuracy = self._calculate_trace_replay_accuracy(fused_results)
        
        return RetrievalResult(
            query_id=query_id,
            query=query,
            retrieved_docs=fused_results["retrieved_docs"],
            component_scores={
                "sparse": {doc: 1.0 for doc in sparse_results},
                "dense": {doc: 1.0 for doc in dense_results},
                "graph": {doc: 1.0 for doc in graph_results}
            },
            fusion_scores=fused_results["fusion_scores"],
            graph_coverage=graph_coverage,
            provenance_integrity=provenance_integrity,
            trace_replay_accuracy=trace_replay_accuracy,
            metadata=fused_results.get("metadata", {})
        )
    
    def _calculate_graph_coverage(self, query: str, retrieved_docs: List[str]) -> float:
        """Calculate % of relevant edges traversed"""
        if not hasattr(self.graph_scaffold, 'edges') or not self.graph_scaffold.edges:
            return 0.0
        
        # Find edges involving retrieved documents
        relevant_edges = [
            edge for edge in self.graph_scaffold.edges
            if edge.source in retrieved_docs or edge.target in retrieved_docs
        ]
        
        return len(relevant_edges) / len(self.graph_scaffold.edges) if self.graph_scaffold.edges else 0.0
    
    def _calculate_provenance_integrity(self, fused_results: Dict[str, Any]) -> float:
        """Calculate % of answers with correct source lineage"""
        provenance_data = fused_results.get("provenance", {})
        retrieved_docs = fused_results.get("retrieved_docs", [])
        
        if not retrieved_docs:
            return 0.0
        
        docs_with_provenance = sum(
            1 for doc_id in retrieved_docs 
            if doc_id in provenance_data and provenance_data[doc_id]
        )
        
        return docs_with_provenance / len(retrieved_docs)
    
    def _calculate_trace_replay_accuracy(self, fused_results: Dict[str, Any]) -> float:
        """Calculate ability to reconstruct memory evolution"""
        trace_id = fused_results.get("trace_id")
        if trace_id is None:
            return 0.0
        
        # Try to replay the trace
        replayed_trace = self.memory_agent.trace_buffer.replay_trace(trace_id)
        
        if replayed_trace:
            original_results = set(fused_results["retrieved_docs"])
            replayed_results = set(replayed_trace.get("final_results", []))
            
            if original_results and replayed_results:
                intersection = len(original_results.intersection(replayed_results))
                union = len(original_results.union(replayed_results))
                return intersection / union if union > 0 else 0.0
        
        return 0.0

class HybridAgentRetriever:
    """
    Retrieval Prototype: Hybrid Agent Retriever
    
    Components (matching the table structure):
    - Sparse Retriever: BM25 baseline for lexical grounding
    - Dense Retriever: Multi-vector (e.g., ColBERT) for semantic matching
    - Graph Reasoner: Traverses relevance graph to find indirect matches  
    - Memory Agent: Uses RL to select, update, and distill memory entries
    """
    
    def __init__(self, graph_scaffold: LimitGraphScaffold, config: Dict[str, Any] = None):
        self.graph_scaffold = graph_scaffold
        self.config = config or {}
        
        # Initialize retrieval components
        self.sparse_retriever = SparseRetriever(config.get("sparse", {}))
        self.dense_retriever = DenseRetriever(config.get("dense", {}))
        self.graph_reasoner = GraphReasoner(graph_scaffold.graph, config.get("graph", {}))
        self.memory_agent = MemoryAgent(config.get("memory", {}))
        
        print("🔍 Hybrid Agent Retriever initialized")
    
    def retrieve(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Retrieval Flow:
        Memory Agent learns to fuse signals from all retrievers and uses 
        provenance and trace buffer to define memory ops
        """
        
        # Get results from each component
        sparse_results = self.sparse_retriever.retrieve(query, top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k)
        graph_results = self.graph_reasoner.retrieve(query, top_k)
        
        # Memory Agent fuses signals and manages memory operations
        fused_results = self.memory_agent.fuse_and_retrieve(
            query=query,
            sparse_results=sparse_results,
            dense_results=dense_results,
            graph_results=graph_results,
            top_k=top_k
        )
        
        return fused_results

class SparseRetriever:
    """BM25 baseline for lexical grounding"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Mock document collection
        self.documents = {
            "d12": "Alice likes apples and enjoys eating them daily",
            "d27": "Bob really likes apples, especially green ones"
        }
    
    def retrieve(self, query: str, top_k: int) -> List[str]:
        """BM25 retrieval"""
        # Simplified BM25 implementation
        query_terms = query.lower().split()
        scores = {}
        
        for doc_id, content in self.documents.items():
            score = sum(1 for term in query_terms if term in content.lower())
            if score > 0:
                scores[doc_id] = score
        
        # Return top_k documents
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

class DenseRetriever:
    """Multi-vector (e.g., ColBERT) for semantic matching"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Mock embeddings
        self.doc_embeddings = {
            "d12": np.random.random(768),
            "d27": np.random.random(768)
        }
    
    def retrieve(self, query: str, top_k: int) -> List[str]:
        """Dense retrieval using embeddings"""
        query_embedding = np.random.random(768)  # Mock query embedding
        
        similarities = {}
        for doc_id, doc_emb in self.doc_embeddings.items():
            sim = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities[doc_id] = sim
        
        sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

class GraphReasoner:
    """Traverses relevance graph to find indirect matches"""
    
    def __init__(self, graph: nx.MultiDiGraph, config: Dict[str, Any]):
        self.graph = graph
        self.config = config
        self.max_hops = config.get("max_hops", 2)
    
    def retrieve(self, query: str, top_k: int) -> List[str]:
        """Graph-based retrieval through traversal"""
        query_terms = query.lower().split()
        relevant_nodes = []
        
        # Find nodes matching query terms
        for node_id in self.graph.nodes():
            if any(term in node_id.lower() for term in query_terms):
                relevant_nodes.append(node_id)
        
        # Expand through graph traversal
        expanded_nodes = set(relevant_nodes)
        for node in relevant_nodes:
            # BFS traversal
            neighbors = list(self.graph.neighbors(node))
            expanded_nodes.update(neighbors[:self.max_hops])
        
        # Filter for document nodes
        doc_nodes = [n for n in expanded_nodes if n.startswith("d")]
        return doc_nodes[:top_k]

class MemoryAgent:
    """
    Uses RL to select, update, and distill memory entries
    Implements the Memory Agent role from the component table
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_entries = {}
        self.provenance_tracker = ProvenanceTracker()
        self.trace_buffer = TraceBuffer()
        
        # Initialize with some memory entries
        self.memory_entries = {
            "mem_1": {"content": "People like fruits", "confidence": 0.8},
            "mem_2": {"content": "Ownership relations", "confidence": 0.7}
        }
    
    def fuse_and_retrieve(self, query: str, sparse_results: List[str], 
                         dense_results: List[str], graph_results: List[str], 
                         top_k: int) -> Dict[str, Any]:
        """
        Memory Agent learns to fuse signals from all retrievers and uses
        provenance and trace buffer to define memory operations
        """
        
        # Collect all candidate documents
        all_candidates = set(sparse_results + dense_results + graph_results)
        
        # Score fusion using learned weights (simplified RL approach)
        fusion_scores = {}
        for doc_id in all_candidates:
            score = 0.0
            if doc_id in sparse_results:
                score += 0.3  # Learned weight for sparse
            if doc_id in dense_results:
                score += 0.4  # Learned weight for dense
            if doc_id in graph_results:
                score += 0.3  # Learned weight for graph
            
            fusion_scores[doc_id] = score
        
        # Select top_k documents
        sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        final_results = [doc_id for doc_id, _ in sorted_docs[:top_k]]
        
        # Update memory and trace buffer
        self._update_memory(query, final_results)
        self._log_trace(query, sparse_results, dense_results, graph_results, final_results)
        
        return {
            "retrieved_docs": final_results,
            "fusion_scores": dict(sorted_docs[:top_k]),
            "component_results": {
                "sparse": sparse_results,
                "dense": dense_results, 
                "graph": graph_results
            },
            "provenance": self.provenance_tracker.get_provenance(final_results),
            "trace_id": self.trace_buffer.get_latest_trace_id()
        }
    
    def _update_memory(self, query: str, results: List[str]):
        """Update memory entries based on retrieval results"""
        memory_id = f"mem_{len(self.memory_entries) + 1}"
        self.memory_entries[memory_id] = {
            "content": f"Query pattern: {query}",
            "confidence": 0.8,
            "results": results
        }
    
    def _log_trace(self, query: str, sparse: List[str], dense: List[str], 
                   graph: List[str], final: List[str]):
        """Log trace for replay and analysis"""
        trace_entry = {
            "query": query,
            "components": {"sparse": sparse, "dense": dense, "graph": graph},
            "final_results": final,
            "timestamp": datetime.now().isoformat()
        }
        self.trace_buffer.add_trace(trace_entry)

class ProvenanceTracker:
    """Tracks provenance for source lineage"""
    
    def __init__(self):
        self.provenance_data = {}
    
    def get_provenance(self, doc_ids: List[str]) -> Dict[str, List[str]]:
        """Get provenance information for documents"""
        return {doc_id: ["limit_corpus", "retrieval_fusion"] for doc_id in doc_ids}

class TraceBuffer:
    """Buffer for storing and replaying traces"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.traces = []
        self.trace_counter = 0
    
    def add_trace(self, trace_entry: Dict[str, Any]):
        """Add trace entry to buffer"""
        trace_entry["trace_id"] = self.trace_counter
        self.traces.append(trace_entry)
        self.trace_counter += 1
        
        # Maintain buffer size
        if len(self.traces) > self.max_size:
            self.traces.pop(0)
    
    def get_latest_trace_id(self) -> int:
        """Get ID of latest trace"""
        return self.trace_counter - 1 if self.traces else 0
    
    def replay_trace(self, trace_id: int) -> Optional[Dict[str, Any]]:
        """Replay specific trace for analysis"""
        for trace in self.traces:
            if trace.get("trace_id") == trace_id:
                return trace
        return None

class LimitGraphEvaluator:
    """
    Evaluation Harness inspired by LIMIT's structure, extended for agentic memory:
    - Recall@k: per retriever and fused output
    - Graph Coverage: % of relevant edges traversed  
    - Provenance Integrity: % of answers with correct source lineage
    - Trace Replay Accuracy: ability to reconstruct memory evolution
    """
    
    def __init__(self, graph_scaffold: LimitGraphScaffold, retriever: HybridAgentRetriever):
        self.graph_scaffold = graph_scaffold
        self.retriever = retriever
        
        print("📊 LIMIT-Graph Evaluator initialized")
    
    def evaluate_retrieval(self, test_queries: List[LimitQuery]) -> Dict[str, Any]:
        """Evaluate retrieval performance with extended metrics"""
        
        results = {
            "recall_at_k": {},
            "graph_coverage": [],
            "provenance_integrity": [],
            "trace_replay_accuracy": [],
            "per_query_results": {}
        }
        
        for query in test_queries:
            # Perform retrieval
            retrieval_result = self.retriever.retrieve(query.query, top_k=10)
            
            # Calculate Recall@k per retriever and fused output
            relevant_docs = set(query.relevant_docs)
            retrieved_docs = set(retrieval_result["retrieved_docs"])
            
            for k in [1, 5, 10]:
                if k not in results["recall_at_k"]:
                    results["recall_at_k"][k] = {"fused": [], "sparse": [], "dense": [], "graph": []}
                
                # Fused recall@k
                fused_at_k = set(retrieval_result["retrieved_docs"][:k])
                fused_recall = len(fused_at_k.intersection(relevant_docs)) / len(relevant_docs)
                results["recall_at_k"][k]["fused"].append(fused_recall)
                
                # Component recall@k
                for component in ["sparse", "dense", "graph"]:
                    comp_results = retrieval_result["component_results"][component][:k]
                    comp_recall = len(set(comp_results).intersection(relevant_docs)) / len(relevant_docs)
                    results["recall_at_k"][k][component].append(comp_recall)
            
            # Graph Coverage: % of relevant edges traversed
            graph_coverage = self._calculate_graph_coverage(query, retrieval_result)
            results["graph_coverage"].append(graph_coverage)
            
            # Provenance Integrity: % of answers with correct source lineage
            provenance_integrity = self._calculate_provenance_integrity(retrieval_result)
            results["provenance_integrity"].append(provenance_integrity)
            
            # Trace Replay Accuracy: ability to reconstruct memory evolution
            trace_accuracy = self._calculate_trace_replay_accuracy(retrieval_result)
            results["trace_replay_accuracy"].append(trace_accuracy)
            
            # Store per-query results
            results["per_query_results"][query.query_id] = {
                "recall_at_10": fused_recall,
                "graph_coverage": graph_coverage,
                "provenance_integrity": provenance_integrity,
                "trace_replay_accuracy": trace_accuracy
            }
        
        # Calculate averages
        for k in results["recall_at_k"]:
            for component in results["recall_at_k"][k]:
                results["recall_at_k"][k][component] = np.mean(results["recall_at_k"][k][component])
        
        results["avg_graph_coverage"] = np.mean(results["graph_coverage"])
        results["avg_provenance_integrity"] = np.mean(results["provenance_integrity"])
        results["avg_trace_replay_accuracy"] = np.mean(results["trace_replay_accuracy"])
        
        return results
    
    def _calculate_graph_coverage(self, query: LimitQuery, result: Dict[str, Any]) -> float:
        """Calculate % of relevant edges traversed"""
        relevant_edges = len(query.graph_edges)
        if relevant_edges == 0:
            return 1.0
        
        # Check how many relevant edges were considered in graph reasoning
        traversed_edges = len([e for e in query.graph_edges 
                             if e["source"] in result["retrieved_docs"] or 
                                e["target"] in result["retrieved_docs"]])
        
        return traversed_edges / relevant_edges
    
    def _calculate_provenance_integrity(self, result: Dict[str, Any]) -> float:
        """Calculate % of answers with correct source lineage"""
        provenance_data = result.get("provenance", {})
        if not provenance_data:
            return 0.0
        
        # Check if all retrieved documents have provenance
        docs_with_provenance = sum(1 for doc_id in result["retrieved_docs"] 
                                 if doc_id in provenance_data and provenance_data[doc_id])
        
        return docs_with_provenance / len(result["retrieved_docs"]) if result["retrieved_docs"] else 0.0
    
    def _calculate_trace_replay_accuracy(self, result: Dict[str, Any]) -> float:
        """Calculate ability to reconstruct memory evolution"""
        trace_id = result.get("trace_id")
        if trace_id is None:
            return 0.0
        
        # Try to replay the trace
        memory_agent = self.retriever.memory_agent
        replayed_trace = memory_agent.trace_buffer.replay_trace(trace_id)
        
        if replayed_trace:
            # Check if replayed results match original results
            original_results = set(result["retrieved_docs"])
            replayed_results = set(replayed_trace.get("final_results", []))
            
            if original_results and replayed_results:
                intersection = len(original_results.intersection(replayed_results))
                union = len(original_results.union(replayed_results))
                return intersection / union if union > 0 else 0.0
        
        return 0.0

class LimitGraphStressTests:
    """CI-evaluable stress tests for memory-aware agents"""
    
    def __init__(self, evaluator: LimitGraphEvaluator):
        self.evaluator = evaluator
    
    def run_stress_tests(self) -> List[CITestResult]:
        """Run comprehensive stress tests exposed via CI hooks"""
        
        test_results = []
        
        # Test 1: Graph construction stress
        test_results.append(self._test_graph_construction_stress())
        
        # Test 2: Retrieval scalability
        test_results.append(self._test_retrieval_scalability())
        
        # Test 3: Memory consistency
        test_results.append(self._test_memory_consistency())
        
        # Test 4: Trace replay accuracy
        test_results.append(self._test_trace_replay_stress())
        
        return test_results
    
    def _test_graph_construction_stress(self) -> CITestResult:
        """Test graph construction under load"""
        start_time = datetime.now()
        
        try:
            # Simulate processing large corpus
            large_corpus = [
                {
                    "query_id": f"stress_q{i}",
                    "query": f"Stress test query {i}",
                    "relevant_docs": [f"d{i}", f"d{i+1}"],
                    "graph_edges": [{"source": f"d{i}", "target": f"entity{i}", "relation": "relates"}]
                }
                for i in range(100)
            ]
            
            scaffold = LimitGraphScaffold()
            stats = scaffold.process_limit_corpus(large_corpus)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Check performance criteria
            if execution_time < 5.0 and stats["documents"] > 0:
                status = "pass"
                message = f"Graph construction stress test passed: {stats} in {execution_time:.2f}s"
            else:
                status = "fail"
                message = f"Graph construction too slow: {execution_time:.2f}s"
            
            return CITestResult(
                test_name="graph_construction_stress",
                status=status,
                message=message,
                details=stats,
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
            # Test with many concurrent queries
            test_queries = [f"Scalability test query {i}" for i in range(50)]
            
            scaffold = LimitGraphScaffold()
            retriever = HybridAgentRetriever(scaffold)
            
            successful_retrievals = 0
            for query in test_queries:
                try:
                    result = retriever.retrieve(query, top_k=5)
                    if result["retrieved_docs"]:
                        successful_retrievals += 1
                except Exception:
                    pass
            
            execution_time = (datetime.now() - start_time).total_seconds()
            success_rate = successful_retrievals / len(test_queries)
            
            if success_rate > 0.8 and execution_time < 10.0:
                status = "pass"
                message = f"Retrieval scalability test passed: {success_rate:.1%} success rate"
            else:
                status = "fail"
                message = f"Retrieval scalability issues: {success_rate:.1%} success rate"
            
            return CITestResult(
                test_name="retrieval_scalability",
                status=status,
                message=message,
                details={"success_rate": success_rate, "total_queries": len(test_queries)},
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
            scaffold = LimitGraphScaffold()
            retriever = HybridAgentRetriever(scaffold)
            memory_agent = retriever.memory_agent
            
            initial_memory_count = len(memory_agent.memory_entries)
            
            # Perform multiple memory operations
            for i in range(20):
                query = f"Memory consistency test {i}"
                retriever.retrieve(query, top_k=3)
            
            final_memory_count = len(memory_agent.memory_entries)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if final_memory_count > initial_memory_count:
                status = "pass"
                message = f"Memory consistency test passed: {initial_memory_count} -> {final_memory_count} entries"
            else:
                status = "fail"
                message = f"Memory not updating: {initial_memory_count} -> {final_memory_count} entries"
            
            return CITestResult(
                test_name="memory_consistency",
                status=status,
                message=message,
                details={"initial": initial_memory_count, "final": final_memory_count},
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
    
    def _test_trace_replay_stress(self) -> CITestResult:
        """Test trace replay accuracy under stress"""
        start_time = datetime.now()
        
        try:
            scaffold = LimitGraphScaffold()
            retriever = HybridAgentRetriever(scaffold)
            
            # Generate traces
            trace_ids = []
            for i in range(10):
                query = f"Trace test {i}"
                result = retriever.retrieve(query, top_k=3)
                trace_ids.append(result["trace_id"])
            
            # Test replay accuracy
            successful_replays = 0
            for trace_id in trace_ids:
                replayed = retriever.memory_agent.trace_buffer.replay_trace(trace_id)
                if replayed:
                    successful_replays += 1
            
            execution_time = (datetime.now() - start_time).total_seconds()
            replay_accuracy = successful_replays / len(trace_ids) if trace_ids else 0
            
            if replay_accuracy > 0.8:
                status = "pass"
                message = f"Trace replay stress test passed: {replay_accuracy:.1%} accuracy"
            else:
                status = "fail"
                message = f"Trace replay issues: {replay_accuracy:.1%} accuracy"
            
            return CITestResult(
                test_name="trace_replay_stress",
                status=status,
                message=message,
                details={"replay_accuracy": replay_accuracy, "total_traces": len(trace_ids)},
                execution_time=execution_time,
                timestamp=start_time
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return CITestResult(
                test_name="trace_replay_stress",
                status="fail",
                message=f"Trace replay stress test failed: {e}",
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=start_time
            )

# CLI interface matching the bash command structure
def run_evaluation_cli():
    """CLI interface: python eval.py --benchmark limit-graph --model hybrid-agent --memory"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="LIMIT-GRAPH Evaluation")
    parser.add_argument("--benchmark", default="limit-graph", help="Benchmark name")
    parser.add_argument("--model", default="hybrid-agent", help="Model type")
    parser.add_argument("--memory", action="store_true", help="Enable memory-aware evaluation")
    
    args = parser.parse_args()
    
    print(f"🚀 Running evaluation: {args.benchmark} with {args.model}")
    
    # Create sample LIMIT data
    sample_data = [
        {
            "query_id": "q42",
            "query": "Who likes apples?",
            "relevant_docs": ["d12", "d27"],
            "graph_edges": [
                {"source": "d12", "target": "apples", "relation": "likes"},
                {"source": "d27", "target": "apples", "relation": "likes"}
            ]
        }
    ]
    
    # Initialize system
    scaffold = LimitGraphScaffold()
    scaffold.process_limit_corpus(sample_data)
    
    retriever = HybridAgentRetriever(scaffold)
    evaluator = LimitGraphEvaluator(scaffold, retriever)
    
    # Create test queries
    test_queries = [LimitQuery(**item) for item in sample_data]
    
    # Run evaluation
    results = evaluator.evaluate_retrieval(test_queries)
    
    # Print results
    print(f"\n📊 Evaluation Results:")
    print(f"   Recall@10 (fused): {results['recall_at_k'][10]['fused']:.3f}")
    print(f"   Graph Coverage: {results['avg_graph_coverage']:.3f}")
    print(f"   Provenance Integrity: {results['avg_provenance_integrity']:.3f}")
    print(f"   Trace Replay Accuracy: {results['avg_trace_replay_accuracy']:.3f}")
    
    # Run stress tests if memory enabled
    if args.memory:
        print(f"\n🧪 Running stress tests...")
        stress_tester = LimitGraphStressTests(evaluator)
        stress_results = stress_tester.run_stress_tests()
        
        passed_tests = sum(1 for r in stress_results if r.status == "pass")
        print(f"   Stress tests: {passed_tests}/{len(stress_results)} passed")
        
        # Return exit code for CI
        return 0 if passed_tests == len(stress_results) else 1
    
    return 0

if __name__ == "__main__":
    exit_code = run_evaluation_cli()
    exit(exit_code)