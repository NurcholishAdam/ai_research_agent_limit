#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace Replay Script for CI
Replays agent memory traces to validate consistency and evolution.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class TraceEntry:
    """Represents a single trace entry"""
    timestamp: str
    query_id: str
    query: str
    action_type: str
    retrieved_docs: List[str]
    reasoning_path: List[str]
    confidence_score: float
    memory_state: Dict[str, Any]

@dataclass
class TraceReplayResult:
    """Result of trace replay validation"""
    trace_id: str
    total_entries: int
    valid_entries: int
    consistency_score: float
    memory_evolution_valid: bool
    errors: List[str]

class MockMemoryTrace:
    """Mock memory trace for demonstration"""
    
    def __init__(self):
        self.traces = self._generate_mock_traces()
    
    def _generate_mock_traces(self) -> Dict[str, List[TraceEntry]]:
        """Generate mock trace data"""
        
        traces = {}
        
        # Trace 1: Simple query sequence
        trace1 = [
            TraceEntry(
                timestamp="2024-01-01T10:00:00Z",
                query_id="q1",
                query="Who likes apples?",
                action_type="entity_linking",
                retrieved_docs=["d1", "d3", "d4"],
                reasoning_path=["Alice", "likes", "apples"],
                confidence_score=0.85,
                memory_state={"entities_seen": ["Alice", "apples"], "relations_learned": ["likes"]}
            ),
            TraceEntry(
                timestamp="2024-01-01T10:01:00Z",
                query_id="q2", 
                query="What contains vitamin C?",
                action_type="graph_traversal",
                retrieved_docs=["d2", "d5"],
                reasoning_path=["oranges", "contains", "vitamin_c"],
                confidence_score=0.92,
                memory_state={"entities_seen": ["Alice", "apples", "oranges", "vitamin_c"], "relations_learned": ["likes", "contains"]}
            ),
            TraceEntry(
                timestamp="2024-01-01T10:02:00Z",
                query_id="q6",
                query="Find people who like citrus fruits", 
                action_type="multi_hop_reasoning",
                retrieved_docs=["d4", "d2"],
                reasoning_path=["Bob", "likes", "oranges", "part_of", "citrus"],
                confidence_score=0.78,
                memory_state={"entities_seen": ["Alice", "apples", "oranges", "vitamin_c", "Bob", "citrus"], "relations_learned": ["likes", "contains", "part_of"]}
            )
        ]
        traces["session_1"] = trace1
        
        # Trace 2: Complex reasoning sequence
        trace2 = [
            TraceEntry(
                timestamp="2024-01-01T11:00:00Z",
                query_id="q8",
                query="Which farmers grow organic produce?",
                action_type="entity_linking",
                retrieved_docs=["d1", "d3", "d10"],
                reasoning_path=["Alice", "practices", "organic_farming"],
                confidence_score=0.88,
                memory_state={"entities_seen": ["Alice", "organic_farming"], "relations_learned": ["practices"]}
            ),
            TraceEntry(
                timestamp="2024-01-01T11:01:00Z",
                query_id="q10",
                query="Find documents about people who live near farms",
                action_type="spatial_reasoning",
                retrieved_docs=["d12", "d1"],
                reasoning_path=["Alice", "owns", "farm", "located_in", "area"],
                confidence_score=0.72,
                memory_state={"entities_seen": ["Alice", "organic_farming", "farm", "area"], "relations_learned": ["practices", "owns", "located_in"]}
            )
        ]
        traces["session_2"] = trace2
        
        return traces

class TraceValidator:
    """Validates trace consistency and memory evolution"""
    
    def __init__(self):
        self.validation_rules = {
            "memory_monotonic": self._validate_memory_monotonic,
            "confidence_reasonable": self._validate_confidence_reasonable,
            "reasoning_path_valid": self._validate_reasoning_path_valid,
            "temporal_consistency": self._validate_temporal_consistency
        }
    
    def validate_trace(self, trace_id: str, trace_entries: List[TraceEntry]) -> TraceReplayResult:
        """Validate a complete trace"""
        
        errors = []
        valid_entries = 0
        
        print(f"🔍 Validating trace: {trace_id}")
        
        # Run validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_errors = rule_func(trace_entries)
                if not rule_errors:
                    print(f"   ✅ {rule_name}: PASSED")
                else:
                    print(f"   ❌ {rule_name}: FAILED")
                    for error in rule_errors:
                        print(f"      • {error}")
                    errors.extend(rule_errors)
            except Exception as e:
                error_msg = f"{rule_name}: Validation error - {e}"
                print(f"   💥 {error_msg}")
                errors.append(error_msg)
        
        # Count valid entries
        valid_entries = len(trace_entries) - len([e for e in errors if "Entry" in e])
        
        # Calculate consistency score
        consistency_score = valid_entries / len(trace_entries) if trace_entries else 0.0
        
        # Check memory evolution
        memory_evolution_valid = self._check_memory_evolution(trace_entries)
        
        return TraceReplayResult(
            trace_id=trace_id,
            total_entries=len(trace_entries),
            valid_entries=valid_entries,
            consistency_score=consistency_score,
            memory_evolution_valid=memory_evolution_valid,
            errors=errors
        )
    
    def _validate_memory_monotonic(self, trace_entries: List[TraceEntry]) -> List[str]:
        """Validate that memory grows monotonically (no forgetting)"""
        
        errors = []
        
        for i in range(1, len(trace_entries)):
            prev_memory = trace_entries[i-1].memory_state
            curr_memory = trace_entries[i].memory_state
            
            # Check entities_seen grows monotonically
            if "entities_seen" in prev_memory and "entities_seen" in curr_memory:
                prev_entities = set(prev_memory["entities_seen"])
                curr_entities = set(curr_memory["entities_seen"])
                
                if not prev_entities.issubset(curr_entities):
                    forgotten = prev_entities - curr_entities
                    errors.append(f"Entry {i}: Entities forgotten: {forgotten}")
            
            # Check relations_learned grows monotonically
            if "relations_learned" in prev_memory and "relations_learned" in curr_memory:
                prev_relations = set(prev_memory["relations_learned"])
                curr_relations = set(curr_memory["relations_learned"])
                
                if not prev_relations.issubset(curr_relations):
                    forgotten = prev_relations - curr_relations
                    errors.append(f"Entry {i}: Relations forgotten: {forgotten}")
        
        return errors
    
    def _validate_confidence_reasonable(self, trace_entries: List[TraceEntry]) -> List[str]:
        """Validate that confidence scores are reasonable"""
        
        errors = []
        
        for i, entry in enumerate(trace_entries):
            confidence = entry.confidence_score
            
            # Check confidence range
            if not (0.0 <= confidence <= 1.0):
                errors.append(f"Entry {i}: Invalid confidence {confidence} (must be 0.0-1.0)")
            
            # Check confidence vs retrieved docs count
            if len(entry.retrieved_docs) == 0 and confidence > 0.5:
                errors.append(f"Entry {i}: High confidence {confidence} with no retrieved docs")
            
            # Check confidence vs reasoning path length
            if len(entry.reasoning_path) > 5 and confidence > 0.9:
                errors.append(f"Entry {i}: Very high confidence {confidence} with long reasoning path")
        
        return errors
    
    def _validate_reasoning_path_valid(self, trace_entries: List[TraceEntry]) -> List[str]:
        """Validate that reasoning paths are structurally valid"""
        
        errors = []
        
        for i, entry in enumerate(trace_entries):
            reasoning_path = entry.reasoning_path
            
            # Check path length
            if len(reasoning_path) == 0:
                errors.append(f"Entry {i}: Empty reasoning path")
                continue
            
            # Check path structure for multi-hop reasoning
            if entry.action_type == "multi_hop_reasoning":
                if len(reasoning_path) < 3:
                    errors.append(f"Entry {i}: Multi-hop reasoning path too short: {reasoning_path}")
                
                # Check alternating entity-relation pattern
                for j in range(1, len(reasoning_path), 2):
                    if j < len(reasoning_path):
                        # Relations should be at odd indices
                        relation = reasoning_path[j]
                        if relation not in ["likes", "contains", "part_of", "owns", "practices", "located_in"]:
                            errors.append(f"Entry {i}: Invalid relation '{relation}' in reasoning path")
        
        return errors
    
    def _validate_temporal_consistency(self, trace_entries: List[TraceEntry]) -> List[str]:
        """Validate temporal consistency of trace entries"""
        
        errors = []
        
        for i in range(1, len(trace_entries)):
            prev_time = datetime.fromisoformat(trace_entries[i-1].timestamp.replace('Z', '+00:00'))
            curr_time = datetime.fromisoformat(trace_entries[i].timestamp.replace('Z', '+00:00'))
            
            # Check timestamps are monotonic
            if curr_time <= prev_time:
                errors.append(f"Entry {i}: Timestamp not monotonic: {curr_time} <= {prev_time}")
            
            # Check reasonable time gaps
            time_diff = (curr_time - prev_time).total_seconds()
            if time_diff > 3600:  # More than 1 hour
                errors.append(f"Entry {i}: Large time gap: {time_diff}s")
        
        return errors
    
    def _check_memory_evolution(self, trace_entries: List[TraceEntry]) -> bool:
        """Check if memory evolution is valid"""
        
        if len(trace_entries) < 2:
            return True
        
        # Check that memory state evolves
        first_memory = trace_entries[0].memory_state
        last_memory = trace_entries[-1].memory_state
        
        # Memory should grow
        if "entities_seen" in first_memory and "entities_seen" in last_memory:
            if len(last_memory["entities_seen"]) <= len(first_memory["entities_seen"]):
                return False
        
        return True

def load_trace_data(epoch: str = "latest") -> Dict[str, List[TraceEntry]]:
    """Load trace data (mock implementation)"""
    
    print(f"📂 Loading trace data for epoch: {epoch}")
    
    # In a real implementation, this would load from files or database
    mock_trace = MockMemoryTrace()
    
    print(f"   ✅ Loaded {len(mock_trace.traces)} trace sessions")
    
    return mock_trace.traces

def main():
    """Main trace replay function"""
    
    print("📼 LIMIT-GRAPH Trace Replay Starting...")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Replay LIMIT-GRAPH agent traces")
    parser.add_argument("--epoch", default="latest", help="Trace epoch to replay")
    args = parser.parse_args()
    
    # Load trace data
    traces = load_trace_data(args.epoch)
    
    if not traces:
        print("❌ No trace data found")
        return False
    
    # Initialize validator
    validator = TraceValidator()
    
    # Validate each trace
    print(f"\n🔍 Validating {len(traces)} trace sessions...")
    
    all_results = []
    overall_success = True
    
    for trace_id, trace_entries in traces.items():
        result = validator.validate_trace(trace_id, trace_entries)
        all_results.append(result)
        
        print(f"\n📊 Trace {trace_id} Results:")
        print(f"   Total entries: {result.total_entries}")
        print(f"   Valid entries: {result.valid_entries}")
        print(f"   Consistency score: {result.consistency_score:.3f}")
        print(f"   Memory evolution: {'✅ Valid' if result.memory_evolution_valid else '❌ Invalid'}")
        
        if result.errors:
            print(f"   Errors: {len(result.errors)}")
            overall_success = False
    
    # Calculate overall statistics
    print(f"\n📈 Overall Trace Replay Statistics:")
    
    total_entries = sum(r.total_entries for r in all_results)
    total_valid = sum(r.valid_entries for r in all_results)
    avg_consistency = sum(r.consistency_score for r in all_results) / len(all_results)
    memory_evolution_success = sum(1 for r in all_results if r.memory_evolution_valid) / len(all_results)
    
    print(f"   Total trace entries: {total_entries}")
    print(f"   Valid entries: {total_valid}")
    print(f"   Overall consistency: {avg_consistency:.3f}")
    print(f"   Memory evolution success: {memory_evolution_success:.1%}")
    
    # CI thresholds
    thresholds = {
        "consistency": 0.8,
        "memory_evolution": 0.9
    }
    
    print(f"\n🎯 CI Trace Replay Check:")
    
    consistency_passed = avg_consistency >= thresholds["consistency"]
    memory_passed = memory_evolution_success >= thresholds["memory_evolution"]
    
    print(f"   Consistency: {avg_consistency:.3f} >= {thresholds['consistency']:.3f} {'✅ PASS' if consistency_passed else '❌ FAIL'}")
    print(f"   Memory Evolution: {memory_evolution_success:.1%} >= {thresholds['memory_evolution']:.1%} {'✅ PASS' if memory_passed else '❌ FAIL'}")
    
    ci_passed = consistency_passed and memory_passed and overall_success
    
    # Summary
    print(f"\n{'='*50}")
    print(f"🏁 LIMIT-GRAPH Trace Replay Summary")
    print(f"{'='*50}")
    
    if ci_passed:
        print("✅ All trace validations PASSED")
        print("🎉 Memory traces are consistent and valid!")
        return True
    else:
        print("❌ Some trace validations FAILED")
        print("🔧 Memory trace issues need investigation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)