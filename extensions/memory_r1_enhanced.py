#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-R1 Enhanced System with Modular Extensions
Implements semantic graph reasoning, provenance validation, and trace buffer replay
with CI-evaluable hooks for comprehensive memory management.
"""

import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque, defaultdict
import networkx as nx
import numpy as np
from pathlib import Path

# Core data structures
@dataclass
class GraphTriple:
    """Semantic graph triple (subject, predicate, object)"""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_turn: int = 0
    extracted_at: datetime = field(default_factory=datetime.now)

@dataclass
class GraphFragment:
    """Graph fragment containing multiple triples"""
    fragment_id: str
    triples: List[GraphTriple]
    entities: Set[str]
    relations: Set[str]
    confidence_score: float
    source_content: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ProvenanceMetadata:
    """Provenance tracking metadata for memory entries"""
    entry_id: str
    content: str
    source_turn: int
    update_chain: List[str]
    confidence_score: float
    trustworthiness: float
    created_at: datetime
    last_updated: datetime
    transformation_history: List[Dict[str, Any]]
    validation_status: str

@dataclass
class TraceEntry:
    """Trace buffer entry for replay functionality"""
    trace_id: str
    turn_id: int
    input_text: str
    extracted_facts: List[str]
    memory_operations: List[str]
    output_response: str
    reward_signal: Optional[float]
    graph_state_hash: str
    provenance_updates: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class GraphOperation(Enum):
    """Graph memory operations"""
    ADD_NODE = "add_node"
    MERGE_EDGE = "merge_edge"
    DELETE_SUBGRAPH = "delete_subgraph"
    NOOP = "noop"

class ValidationResult(Enum):
    """Validation result status"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"

# Module 1: Semantic Graph Reasoning Module
class GraphBuilder:
    """Converts extracted facts into semantic graphs"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.relation_patterns = {
            "is_a": ["is", "are", "was", "were"],
            "has": ["has", "have", "contains"],
            "located_in": ["in", "at", "located"],
            "part_of": ["part of", "belongs to"],
            "causes": ["causes", "leads to", "results in"]
        }
    
    def extract_triples_from_text(self, text: str, turn_id: int = 0) -> List[GraphTriple]:
        """Extract semantic triples from text"""
        triples = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            words = sentence.split()
            entities = [word for word in words if word[0].isupper() and len(word) > 2]
            
            for relation_type, patterns in self.relation_patterns.items():
                for pattern in patterns:
                    if pattern in sentence.lower():
                        parts = sentence.lower().split(pattern)
                        if len(parts) >= 2 and entities:
                            subject = self._extract_nearest_entity(parts[0], entities)
                            object_part = self._extract_nearest_entity(parts[1], entities)
                            
                            if subject and object_part and subject != object_part:
                                triple = GraphTriple(
                                    subject=subject,
                                    predicate=relation_type,
                                    object=object_part,
                                    confidence=0.7,
                                    source_turn=turn_id
                                )
                                triples.append(triple)
        
        return triples
    
    def _extract_nearest_entity(self, text: str, entities: List[str]) -> Optional[str]:
        """Extract the nearest entity from text"""
        words = text.strip().split()
        for word in reversed(words):
            clean_word = word.strip('.,!?()[]{}')
            if clean_word in entities:
                return clean_word
        return None
    
    def build_graph_fragment(self, triples: List[GraphTriple], source_content: str) -> GraphFragment:
        """Build a graph fragment from extracted triples"""
        
        if not triples:
            return GraphFragment(
                fragment_id=str(uuid.uuid4()),
                triples=[],
                entities=set(),
                relations=set(),
                confidence_score=0.0,
                source_content=source_content
            )
        
        entities = set()
        relations = set()
        
        for triple in triples:
            entities.add(triple.subject)
            entities.add(triple.object)
            relations.add(triple.predicate)
        
        confidence_score = np.mean([t.confidence for t in triples]) if triples else 0.0
        
        return GraphFragment(
            fragment_id=str(uuid.uuid4()),
            triples=triples,
            entities=entities,
            relations=relations,
            confidence_score=confidence_score,
            source_content=source_content
        )

class GraphMemoryBank:
    """Stores evolving graph state with efficient operations"""
    
    def __init__(self, storage_path: str = "memory_r1_graph_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.graph = nx.MultiDiGraph()
        self.fragments: Dict[str, GraphFragment] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.relation_index: Dict[str, Set[str]] = defaultdict(set)
        self.operation_history: List[Dict[str, Any]] = []
        
        print("🧠 Graph Memory Bank initialized")
    
    def add_fragment(self, fragment: GraphFragment) -> List[GraphOperation]:
        """Add a graph fragment and return operations performed"""
        operations = []
        
        self.fragments[fragment.fragment_id] = fragment
        
        for triple in fragment.triples:
            if not self.graph.has_node(triple.subject):
                self.graph.add_node(triple.subject, 
                                  entity_type="entity",
                                  confidence=triple.confidence,
                                  first_seen=triple.extracted_at.isoformat())
                operations.append(GraphOperation.ADD_NODE)
                
            if not self.graph.has_node(triple.object):
                self.graph.add_node(triple.object,
                                  entity_type="entity", 
                                  confidence=triple.confidence,
                                  first_seen=triple.extracted_at.isoformat())
                operations.append(GraphOperation.ADD_NODE)
            
            existing_edges = self.graph.get_edge_data(triple.subject, triple.object)
            if existing_edges:
                for key, edge_data in existing_edges.items():
                    if edge_data.get('relation') == triple.predicate:
                        old_conf = edge_data.get('confidence', 0.5)
                        new_conf = (old_conf + triple.confidence) / 2
                        edge_data['confidence'] = new_conf
                        operations.append(GraphOperation.MERGE_EDGE)
                        break
                else:
                    self.graph.add_edge(triple.subject, triple.object,
                                      relation=triple.predicate,
                                      confidence=triple.confidence,
                                      source_turn=triple.source_turn)
                    operations.append(GraphOperation.ADD_NODE)
            else:
                self.graph.add_edge(triple.subject, triple.object,
                                  relation=triple.predicate,
                                  confidence=triple.confidence,
                                  source_turn=triple.source_turn)
                operations.append(GraphOperation.ADD_NODE)
        
        # Update indices
        for entity in fragment.entities:
            self.entity_index[entity].add(fragment.fragment_id)
        
        for relation in fragment.relations:
            self.relation_index[relation].add(fragment.fragment_id)
        
        self._record_operation("add_fragment", {
            "fragment_id": fragment.fragment_id,
            "operations": [op.value for op in operations],
            "entities_added": len(fragment.entities)
        })
        
        return operations
    
    def query_graph(self, query_entities: List[str], max_hops: int = 2) -> Dict[str, Any]:
        """Query graph for related entities and relations"""
        
        result = {
            "query_entities": query_entities,
            "related_entities": set(),
            "relations": [],
            "subgraph_nodes": set(),
            "confidence_scores": {}
        }
        
        for entity in query_entities:
            if entity in self.graph:
                result["subgraph_nodes"].add(entity)
                
                visited = set()
                queue = [(entity, 0)]
                
                while queue:
                    current_entity, hops = queue.pop(0)
                    
                    if hops >= max_hops or current_entity in visited:
                        continue
                    
                    visited.add(current_entity)
                    
                    for neighbor in self.graph.neighbors(current_entity):
                        result["related_entities"].add(neighbor)
                        result["subgraph_nodes"].add(neighbor)
                        
                        edge_data = self.graph.get_edge_data(current_entity, neighbor)
                        if edge_data:
                            for edge_key, edge_attrs in edge_data.items():
                                relation_info = {
                                    "subject": current_entity,
                                    "predicate": edge_attrs.get("relation", "unknown"),
                                    "object": neighbor,
                                    "confidence": edge_attrs.get("confidence", 0.5)
                                }
                                result["relations"].append(relation_info)
                        
                        if hops + 1 < max_hops:
                            queue.append((neighbor, hops + 1))
        
        for entity in result["subgraph_nodes"]:
            if entity in self.graph:
                node_data = self.graph.nodes[entity]
                result["confidence_scores"][entity] = node_data.get("confidence", 0.5)
        
        return result
    
    def get_graph_state_hash(self) -> str:
        """Get hash of current graph state"""
        nodes = sorted(self.graph.nodes(data=True))
        edges = sorted(self.graph.edges(data=True))
        
        state_repr = {
            "nodes": nodes,
            "edges": edges,
            "fragment_count": len(self.fragments)
        }
        
        state_str = json.dumps(state_repr, sort_keys=True, default=str)
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def _record_operation(self, operation_type: str, details: Dict[str, Any]):
        """Record operation in history"""
        operation_record = {
            "operation_id": str(uuid.uuid4()),
            "operation_type": operation_type,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "graph_state_hash": self.get_graph_state_hash()
        }
        
        self.operation_history.append(operation_record)
        
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-1000:]

class GraphRLPolicy:
    """RL policy for graph operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.policy_weights = {
            "add_node_weight": 0.8,
            "merge_edge_weight": 0.9,
            "confidence_threshold": 0.6
        }
    
    def select_operation(self, context: Dict[str, Any]) -> GraphOperation:
        """Select graph operation based on context"""
        confidence_score = context.get("confidence_score", 0.5)
        
        if confidence_score < self.policy_weights["confidence_threshold"]:
            return GraphOperation.NOOP
        
        return GraphOperation.ADD_NODE
    
    def update_policy(self, reward_signal: float, context: Dict[str, Any], action: GraphOperation):
        """Update policy weights based on reward"""
        learning_rate = 0.01
        
        if reward_signal > 0:
            if action == GraphOperation.ADD_NODE:
                self.policy_weights["add_node_weight"] += learning_rate * reward_signal
        
        # Clip weights
        for key in self.policy_weights:
            if "weight" in key:
                self.policy_weights[key] = np.clip(self.policy_weights[key], 0.1, 1.0)

# Module 2: Provenance Validator
class ProvenanceTracker:
    """Logs source and transformation history"""
    
    def __init__(self, storage_path: str = "memory_r1_provenance"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.provenance_records: Dict[str, ProvenanceMetadata] = {}
        self.update_chains: Dict[str, List[str]] = defaultdict(list)
        
        print("📋 Provenance Tracker initialized")
    
    def create_provenance_record(self, content: str, source_turn: int, 
                                confidence_score: float = 0.8) -> str:
        """Create new provenance record"""
        entry_id = str(uuid.uuid4())
        
        provenance = ProvenanceMetadata(
            entry_id=entry_id,
            content=content,
            source_turn=source_turn,
            update_chain=[entry_id],
            confidence_score=confidence_score,
            trustworthiness=confidence_score,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            transformation_history=[{
                "operation": "create",
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence_score
            }],
            validation_status="pending"
        )
        
        self.provenance_records[entry_id] = provenance
        self.update_chains[entry_id] = [entry_id]
        
        return entry_id
    
    def validate_provenance_integrity(self) -> Dict[str, Any]:
        """Validate integrity of all provenance records"""
        validation_results = {
            "total_records": len(self.provenance_records),
            "valid_records": 0,
            "invalid_records": 0,
            "warnings": [],
            "errors": []
        }
        
        for entry_id, provenance in self.provenance_records.items():
            try:
                if not provenance.entry_id or not provenance.content:
                    validation_results["errors"].append(f"Invalid record {entry_id}")
                    validation_results["invalid_records"] += 1
                    continue
                
                if not (0.0 <= provenance.confidence_score <= 1.0):
                    validation_results["warnings"].append(f"Record {entry_id}: confidence out of bounds")
                
                validation_results["valid_records"] += 1
                
            except Exception as e:
                validation_results["errors"].append(f"Record {entry_id}: {str(e)}")
                validation_results["invalid_records"] += 1
        
        return validation_results

class ConfidenceScorer:
    """Assigns trust scores using heuristics"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def calculate_confidence_score(self, content: str, source_turn: int, 
                                 context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for content"""
        context = context or {}
        
        # Simple heuristic based on content length and turn
        content_length = len(content.split())
        length_score = min(1.0, content_length / 20.0)
        
        # Recency score
        current_turn = context.get("current_turn", source_turn)
        turn_diff = current_turn - source_turn
        recency_score = max(0.1, 1.0 - (turn_diff / 10.0))
        
        return (length_score + recency_score) / 2

# Module 3: Trace Buffer Replay Module
class TraceBuffer:
    """Circular buffer storing recent agent interactions"""
    
    def __init__(self, max_size: int = 1000, storage_path: str = "memory_r1_traces"):
        self.max_size = max_size
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.traces: deque = deque(maxlen=max_size)
        self.trace_index: Dict[str, int] = {}
        self.turn_index: Dict[int, List[str]] = defaultdict(list)
        
        print(f"🔄 Trace Buffer initialized (max_size: {max_size})")
    
    def add_trace(self, turn_id: int, input_text: str, extracted_facts: List[str],
                 memory_operations: List[str], output_response: str, 
                 graph_state_hash: str, provenance_updates: List[str],
                 reward_signal: Optional[float] = None) -> str:
        """Add new trace entry to buffer"""
        
        trace_id = str(uuid.uuid4())
        
        trace_entry = TraceEntry(
            trace_id=trace_id,
            turn_id=turn_id,
            input_text=input_text,
            extracted_facts=extracted_facts,
            memory_operations=memory_operations,
            output_response=output_response,
            reward_signal=reward_signal,
            graph_state_hash=graph_state_hash,
            provenance_updates=provenance_updates
        )
        
        self.traces.append(trace_entry)
        
        position = len(self.traces) - 1
        self.trace_index[trace_id] = position
        self.turn_index[turn_id].append(trace_id)
        
        return trace_id
    
    def get_traces_by_turn_range(self, start_turn: int, end_turn: int) -> List[TraceEntry]:
        """Get traces within turn range"""
        traces = []
        for turn_id in range(start_turn, end_turn + 1):
            if turn_id in self.turn_index:
                for trace_id in self.turn_index[turn_id]:
                    if trace_id in self.trace_index:
                        position = self.trace_index[trace_id]
                        if position < len(self.traces):
                            traces.append(self.traces[position])
        
        return sorted(traces, key=lambda t: t.turn_id)

class ReplayEngine:
    """Reconstructs memory state and agent decisions over time"""
    
    def __init__(self, trace_buffer: TraceBuffer, graph_memory: GraphMemoryBank, 
                 provenance_tracker: ProvenanceTracker):
        self.trace_buffer = trace_buffer
        self.graph_memory = graph_memory
        self.provenance_tracker = provenance_tracker
        
        print("🎬 Replay Engine initialized")
    
    def replay_trace_sequence(self, start_turn: int, end_turn: int) -> Dict[str, Any]:
        """Replay trace sequence and reconstruct state evolution"""
        
        traces = self.trace_buffer.get_traces_by_turn_range(start_turn, end_turn)
        
        if not traces:
            return {"error": f"No traces found in range {start_turn}-{end_turn}"}
        
        replay_result = {
            "start_turn": start_turn,
            "end_turn": end_turn,
            "traces_replayed": len(traces),
            "decision_points": [],
            "reward_attribution": {}
        }
        
        for trace in traces:
            decision_point = {
                "turn_id": trace.turn_id,
                "input": trace.input_text,
                "extracted_facts": trace.extracted_facts,
                "memory_operations": trace.memory_operations,
                "output": trace.output_response,
                "reward": trace.reward_signal
            }
            replay_result["decision_points"].append(decision_point)
        
        return replay_result

# Main Memory-R1 Enhanced System
class MemoryR1Enhanced:
    """Main Memory-R1 system with all three modules integrated"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize modules
        self.graph_builder = GraphBuilder(config.get("graph_builder", {}))
        self.graph_memory = GraphMemoryBank(config.get("storage_path", "memory_r1_data"))
        self.graph_rl_policy = GraphRLPolicy(config.get("rl_policy", {}))
        
        self.provenance_tracker = ProvenanceTracker(config.get("provenance_path", "memory_r1_provenance"))
        self.confidence_scorer = ConfidenceScorer(config.get("confidence_scorer", {}))
        
        trace_buffer_config = config.get("trace_buffer", {})
        self.trace_buffer = TraceBuffer(
            max_size=trace_buffer_config.get("max_size", 1000),
            storage_path=trace_buffer_config.get("storage_path", "memory_r1_traces")
        )
        self.replay_engine = ReplayEngine(self.trace_buffer, self.graph_memory, self.provenance_tracker)
        
        self.current_turn = 0
        self.system_stats = {
            "total_extractions": 0,
            "total_operations": 0,
            "total_rewards": 0.0
        }
        
        print("🧠 Memory-R1 Enhanced System initialized")
    
    def process_input(self, input_text: str, reward_signal: Optional[float] = None) -> Dict[str, Any]:
        """Main processing pipeline"""
        
        self.current_turn += 1
        
        result = {
            "turn_id": self.current_turn,
            "input_text": input_text,
            "extracted_facts": [],
            "graph_operations": [],
            "provenance_entries": [],
            "memory_operations": [],
            "output_response": "",
            "success": True
        }
        
        try:
            # Extract triples
            triples = self.graph_builder.extract_triples_from_text(input_text, self.current_turn)
            result["extracted_facts"] = [f"{t.subject} {t.predicate} {t.object}" for t in triples]
            
            if triples:
                # Build fragment
                fragment = self.graph_builder.build_graph_fragment(triples, input_text)
                
                # Create provenance records
                provenance_entries = []
                for triple in triples:
                    fact_content = f"{triple.subject} {triple.predicate} {triple.object}"
                    confidence_score = self.confidence_scorer.calculate_confidence_score(
                        fact_content, self.current_turn, {"current_turn": self.current_turn}
                    )
                    entry_id = self.provenance_tracker.create_provenance_record(
                        fact_content, self.current_turn, confidence_score
                    )
                    provenance_entries.append(entry_id)
                
                result["provenance_entries"] = provenance_entries
                
                # Select operation
                operation_context = {
                    "confidence_score": fragment.confidence_score
                }
                selected_operation = self.graph_rl_policy.select_operation(operation_context)
                
                # Execute operations
                if selected_operation != GraphOperation.NOOP:
                    operations = self.graph_memory.add_fragment(fragment)
                    result["graph_operations"] = [op.value for op in operations]
                    result["memory_operations"] = [f"execute_{selected_operation.value}"]
                    
                    if reward_signal is not None:
                        self.graph_rl_policy.update_policy(reward_signal, operation_context, selected_operation)
                
                result["output_response"] = self._generate_response(fragment, operations if selected_operation != GraphOperation.NOOP else [])
            
            else:
                result["output_response"] = "I understand your input, but couldn't extract specific facts."
                result["memory_operations"] = ["noop"]
            
            # Record trace
            graph_state_hash = self.graph_memory.get_graph_state_hash()
            trace_id = self.trace_buffer.add_trace(
                turn_id=self.current_turn,
                input_text=input_text,
                extracted_facts=result["extracted_facts"],
                memory_operations=result["memory_operations"],
                output_response=result["output_response"],
                graph_state_hash=graph_state_hash,
                provenance_updates=result["provenance_entries"],
                reward_signal=reward_signal
            )
            
            result["trace_id"] = trace_id
            
            # Update stats
            self.system_stats["total_extractions"] += len(result["extracted_facts"])
            self.system_stats["total_operations"] += len(result["graph_operations"])
            if reward_signal:
                self.system_stats["total_rewards"] += reward_signal
        
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _generate_response(self, fragment: GraphFragment, operations: List[GraphOperation]) -> str:
        """Generate response based on memory updates"""
        
        if not fragment.triples:
            return "I processed your input but didn't find specific facts to remember."
        
        entities = list(fragment.entities)[:3]
        
        response_parts = [
            f"I've processed your input and extracted {len(fragment.triples)} factual relationships."
        ]
        
        if entities:
            response_parts.append(f"Key entities: {', '.join(entities)}.")
        
        if operations:
            response_parts.append(f"Performed {len(operations)} memory operations.")
        
        # Query related information
        if entities:
            query_result = self.graph_memory.query_graph(entities[:2], max_hops=1)
            if query_result["related_entities"]:
                related = list(query_result["related_entities"])[:3]
                response_parts.append(f"This connects to: {', '.join(related)}.")
        
        response_parts.append(f"Confidence: {fragment.confidence_score:.2f}")
        
        return " ".join(response_parts)
    
    def query_memory(self, query_text: str) -> Dict[str, Any]:
        """Query the memory system"""
        
        query_triples = self.graph_builder.extract_triples_from_text(query_text)
        query_entities = []
        
        for triple in query_triples:
            query_entities.extend([triple.subject, triple.object])
        
        query_entities = list(set(query_entities))[:3]
        
        if query_entities:
            graph_result = self.graph_memory.query_graph(query_entities, max_hops=2)
            return {
                "query": query_text,
                "query_entities": query_entities,
                "graph_result": graph_result,
                "total_results": len(graph_result["related_entities"])
            }
        else:
            return {
                "query": query_text,
                "message": "Could not extract entities from query",
                "total_results": 0
            }
    
    # CI-Evaluable Hooks Implementation
    
    def validate_graph_state(self) -> Dict[str, Any]:
        """Validate current graph state integrity"""
        
        validation_result = {
            "timestamp": datetime.now().isoformat(),
            "graph_validation": {
                "status": "valid",
                "node_count": len(self.graph_memory.graph.nodes),
                "edge_count": len(self.graph_memory.graph.edges),
                "fragment_count": len(self.graph_memory.fragments),
                "issues": []
            },
            "overall_status": "valid"
        }
        
        try:
            # Check for orphaned nodes
            isolated_nodes = list(nx.isolates(self.graph_memory.graph))
            if isolated_nodes:
                validation_result["graph_validation"]["issues"].append(
                    f"Found {len(isolated_nodes)} isolated nodes"
                )
            
            # Validate fragment consistency
            for fragment_id, fragment in self.graph_memory.fragments.items():
                for triple in fragment.triples:
                    if not self.graph_memory.graph.has_node(triple.subject):
                        validation_result["graph_validation"]["issues"].append(
                            f"Fragment {fragment_id}: subject '{triple.subject}' not in graph"
                        )
                        validation_result["graph_validation"]["status"] = "invalid"
                    
                    if not self.graph_memory.graph.has_node(triple.object):
                        validation_result["graph_validation"]["issues"].append(
                            f"Fragment {fragment_id}: object '{triple.object}' not in graph"
                        )
                        validation_result["graph_validation"]["status"] = "invalid"
            
            if validation_result["graph_validation"]["status"] == "invalid":
                validation_result["overall_status"] = "invalid"
            elif validation_result["graph_validation"]["issues"]:
                validation_result["overall_status"] = "warning"
        
        except Exception as e:
            validation_result["overall_status"] = "error"
            validation_result["error"] = str(e)
        
        return validation_result
    
    def check_provenance_integrity(self) -> Dict[str, Any]:
        """Check integrity of provenance tracking system"""
        
        base_validation = self.provenance_tracker.validate_provenance_integrity()
        
        enhanced_validation = {
            "timestamp": datetime.now().isoformat(),
            "base_validation": base_validation,
            "trace_provenance_consistency": {
                "status": "valid",
                "issues": []
            },
            "overall_status": "valid"
        }
        
        try:
            # Check consistency between traces and provenance
            for trace in self.trace_buffer.traces:
                for provenance_id in trace.provenance_updates:
                    if provenance_id not in self.provenance_tracker.provenance_records:
                        enhanced_validation["trace_provenance_consistency"]["issues"].append(
                            f"Trace {trace.trace_id} references non-existent provenance {provenance_id}"
                        )
                        enhanced_validation["trace_provenance_consistency"]["status"] = "invalid"
            
            if (base_validation["invalid_records"] > 0 or
                enhanced_validation["trace_provenance_consistency"]["status"] == "invalid"):
                enhanced_validation["overall_status"] = "invalid"
            elif (base_validation["warnings"] or 
                  enhanced_validation["trace_provenance_consistency"]["issues"]):
                enhanced_validation["overall_status"] = "warning"
        
        except Exception as e:
            enhanced_validation["overall_status"] = "error"
            enhanced_validation["error"] = str(e)
        
        return enhanced_validation
    
    def replay_trace(self, start_turn: int, end_turn: int) -> Dict[str, Any]:
        """Replay trace sequence for debugging and analysis"""
        
        try:
            replay_result = self.replay_engine.replay_trace_sequence(start_turn, end_turn)
            
            enhanced_replay = {
                "timestamp": datetime.now().isoformat(),
                "replay_parameters": {
                    "start_turn": start_turn,
                    "end_turn": end_turn,
                    "requested_range": end_turn - start_turn + 1
                },
                "base_replay": replay_result,
                "performance_metrics": {}
            }
            
            if "error" not in replay_result:
                # Calculate performance metrics
                rewards = [dp["reward"] for dp in replay_result["decision_points"] if dp["reward"] is not None]
                
                performance_metrics = {
                    "total_rewards": sum(rewards) if rewards else 0.0,
                    "average_reward": np.mean(rewards) if rewards else 0.0,
                    "positive_reward_ratio": sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0
                }
                
                enhanced_replay["performance_metrics"] = performance_metrics
            
            return enhanced_replay
        
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": f"Replay failed: {str(e)}",
                "start_turn": start_turn,
                "end_turn": end_turn
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_turn": self.current_turn,
            "system_stats": dict(self.system_stats),
            "module_status": {
                "graph_memory": {
                    "nodes": len(self.graph_memory.graph.nodes),
                    "edges": len(self.graph_memory.graph.edges),
                    "fragments": len(self.graph_memory.fragments)
                },
                "provenance_tracker": {
                    "records": len(self.provenance_tracker.provenance_records)
                },
                "trace_buffer": {
                    "traces": len(self.trace_buffer.traces),
                    "utilization": len(self.trace_buffer.traces) / self.trace_buffer.max_size
                }
            },
            "validation_status": {
                "graph_state": self.validate_graph_state()["overall_status"],
                "provenance_integrity": self.check_provenance_integrity()["overall_status"]
            }
        }

# Demo function
def run_memory_r1_demo():
    """Run a demonstration of the Memory-R1 Enhanced system"""
    
    print("🚀 Memory-R1 Enhanced System Demo")
    print("=" * 50)
    
    # Initialize system
    system = MemoryR1Enhanced()
    
    # Demo inputs
    demo_inputs = [
        ("Paris is the capital of France.", 0.8),
        ("France is located in Europe.", 0.9),
        ("The Eiffel Tower is in Paris.", 0.7),
        ("Europe has many countries.", 0.6),
        ("What do you know about Paris?", None)
    ]
    
    print(f"\n📝 Processing {len(demo_inputs)} demo inputs...")
    
    for i, (input_text, reward) in enumerate(demo_inputs, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Input: {input_text}")
        
        result = system.process_input(input_text, reward)
        
        print(f"Extracted facts: {len(result['extracted_facts'])}")
        print(f"Graph operations: {result['graph_operations']}")
        print(f"Response: {result['output_response']}")
        
        if reward:
            print(f"Reward: {reward}")
    
    # Demo query
    print(f"\n🔍 Querying memory...")
    query_result = system.query_memory("Tell me about Paris and France")
    print(f"Query results: {query_result['total_results']} entities found")
    
    # Demo CI hooks
    print(f"\n🔧 Testing CI-Evaluable Hooks...")
    
    # Validate graph state
    graph_validation = system.validate_graph_state()
    print(f"Graph validation: {graph_validation['overall_status']}")
    
    # Check provenance integrity
    provenance_validation = system.check_provenance_integrity()
    print(f"Provenance validation: {provenance_validation['overall_status']}")
    
    # Replay trace
    if system.current_turn >= 3:
        replay_result = system.replay_trace(1, 3)
        if "error" not in replay_result:
            print(f"Trace replay: {replay_result['base_replay']['traces_replayed']} traces replayed")
        else:
            print(f"Trace replay error: {replay_result['error']}")
    
    # System status
    status = system.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Current turn: {status['current_turn']}")
    print(f"   Graph nodes: {status['module_status']['graph_memory']['nodes']}")
    print(f"   Provenance records: {status['module_status']['provenance_tracker']['records']}")
    print(f"   Trace buffer utilization: {status['module_status']['trace_buffer']['utilization']:.1%}")
    
    print(f"\n✅ Memory-R1 Enhanced Demo Complete!")
    
    return system

if __name__ == "__main__":
    demo_system = run_memory_r1_demo()