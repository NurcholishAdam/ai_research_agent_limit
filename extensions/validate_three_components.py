#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Three Key LIMIT-GRAPH Components
Simple validation script that demonstrates:
1. Graph Reasoner Module - entity linking and graph traversal
2. RL Reward Function - reward calculation and learning
3. Integration - how they work together
"""

import sys
import traceback
from typing import Dict, Any

def validate_graph_reasoner():
    """Validate Graph Reasoner Module"""
    
    print("🧠 Validating Graph Reasoner Module...")
    
    try:
        from graph_reasoner import GraphReasoner, EntityLinker, GraphTraverser
        print("   ✅ Graph Reasoner imports successful")
        
        # Test entity linking
        entity_linker = EntityLinker()
        result = entity_linker.extract_query_entities("Who likes apples?")
        print(f"   ✅ Entity linking works: found {len(result.entities)} entities")
        
        # Test with mock graph
        import networkx as nx
        from limit_graph_core import LimitGraphNode
        
        class MockScaffold:
            def __init__(self):
                self.graph = nx.MultiDiGraph()
                self.nodes = {}
                
                # Add test nodes
                self.graph.add_node("Alice", type="entity")
                self.graph.add_node("apples", type="entity") 
                self.graph.add_node("d1", type="document")
                
                # Add test edges
                self.graph.add_edge("Alice", "apples", relation="likes")
                self.graph.add_edge("d1", "Alice", relation="mentions")
                
                # Add nodes dict
                self.nodes["Alice"] = LimitGraphNode("Alice", "entity", "Alice", 0.8)
                self.nodes["apples"] = LimitGraphNode("apples", "entity", "apples", 0.8)
                self.nodes["d1"] = LimitGraphNode("d1", "document", "Document about Alice", 0.9)
        
        # Test graph reasoner
        scaffold = MockScaffold()
        reasoner = GraphReasoner(scaffold)
        result = reasoner.reason_and_retrieve("Who likes apples?", top_k=3)
        
        print(f"   ✅ Graph reasoning works: found {len(result['retrieved_docs'])} documents")
        print(f"   ✅ Entity linking confidence: {result['entity_linking'].confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Graph Reasoner validation failed: {e}")
        traceback.print_exc()
        return False

def validate_rl_reward_function():
    """Validate RL Reward Function"""
    
    print("\n🎯 Validating RL Reward Function...")
    
    try:
        from rl_reward_function import (
            RLRewardFunction, RewardComponents, FusionAction, 
            RewardContext, RecallCalculator, ProvenanceScorer
        )
        from limit_graph_core import LimitQuery
        
        print("   ✅ RL Reward Function imports successful")
        
        # Test individual components
        recall_calc = RecallCalculator()
        recall_result = recall_calc.calculate_recall_reward(
            retrieved_docs=["d1", "d2", "d3"],
            ground_truth_docs=["d1", "d3", "d4"]
        )
        print(f"   ✅ Recall calculation works: recall@10 = {recall_result.get('recall_at_10', 0):.3f}")
        
        provenance_scorer = ProvenanceScorer()
        prov_result = provenance_scorer.calculate_provenance_reward(
            retrieved_docs=["d1", "d2"],
            provenance_data={"d1": ["source1"], "d2": ["source2"]}
        )
        print(f"   ✅ Provenance scoring works: score = {prov_result.get('provenance_score', 0):.3f}")
        
        # Test complete reward function
        reward_fn = RLRewardFunction()
        
        # Create test fusion action
        fusion_action = FusionAction(
            action_id="test_action",
            query="test query",
            component_weights={"sparse": 0.3, "dense": 0.4, "graph": 0.3},
            retrieved_docs=["d1", "d2", "d3"],
            fusion_strategy="test"
        )
        
        # Create reward context
        reward_context = RewardContext(
            query=LimitQuery("test_q", "test query", ["d1", "d3"], []),
            ground_truth_docs=["d1", "d3"],
            fusion_action=fusion_action,
            previous_actions=[],
            memory_state={},
            provenance_data={"d1": ["source1"], "d2": ["source2"], "d3": ["source3"]},
            trace_history=[]
        )
        
        # Calculate reward
        reward = reward_fn.calculate_reward(reward_context)
        print(f"   ✅ Reward calculation works: total reward = {reward.total_reward:.3f}")
        print(f"   ✅ Reward components: recall={reward.recall_reward:.3f}, provenance={reward.provenance_reward:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ RL Reward Function validation failed: {e}")
        traceback.print_exc()
        return False

def validate_integration():
    """Validate Integration of Components"""
    
    print("\n🔗 Validating Component Integration...")
    
    try:
        from graph_reasoner import GraphReasoner, integrate_graph_reasoner_with_memory_agent
        from rl_reward_function import RLRewardFunction, integrate_reward_function_with_memory_agent
        
        print("   ✅ Integration imports successful")
        
        # Create mock memory agent
        class MockMemoryAgent:
            def __init__(self):
                self.action_history = []
                self.graph_reasoner = None
                self.reward_function = None
            
            def fuse_and_retrieve(self, query, sparse_results, dense_results, graph_results, top_k):
                # Simple fusion for testing
                all_docs = list(set(sparse_results + dense_results + graph_results))
                return {
                    "retrieved_docs": all_docs[:top_k],
                    "fusion_scores": {doc: 1.0 for doc in all_docs[:top_k]}
                }
        
        # Create components
        import networkx as nx
        from limit_graph_core import LimitGraphNode
        
        class MockScaffold:
            def __init__(self):
                self.graph = nx.MultiDiGraph()
                self.nodes = {"test": LimitGraphNode("test", "entity", "test", 0.8)}
        
        memory_agent = MockMemoryAgent()
        graph_reasoner = GraphReasoner(MockScaffold())
        reward_function = RLRewardFunction()
        
        # Test integrations
        integrate_graph_reasoner_with_memory_agent(memory_agent, graph_reasoner)
        print("   ✅ Graph Reasoner integration successful")
        
        integrate_reward_function_with_memory_agent(memory_agent, reward_function)
        print("   ✅ RL Reward Function integration successful")
        
        # Test integrated functionality
        result = memory_agent.fuse_and_retrieve(
            "test query", ["d1"], ["d2"], ["d3"], top_k=3
        )
        print(f"   ✅ Integrated fusion works: retrieved {len(result['retrieved_docs'])} docs")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration validation failed: {e}")
        traceback.print_exc()
        return False

def validate_core_dependencies():
    """Validate core dependencies are available"""
    
    print("📦 Validating Core Dependencies...")
    
    dependencies = [
        ("networkx", "Graph processing"),
        ("numpy", "Numerical computations"),
        ("spacy", "Natural language processing"),
        ("datetime", "Time handling"),
        ("typing", "Type hints"),
        ("dataclasses", "Data structures"),
        ("collections", "Data collections")
    ]
    
    missing_deps = []
    
    for dep_name, description in dependencies:
        try:
            __import__(dep_name)
            print(f"   ✅ {dep_name}: {description}")
        except ImportError:
            print(f"   ❌ {dep_name}: {description} - MISSING")
            missing_deps.append(dep_name)
    
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def validate_limit_graph_core():
    """Validate LIMIT-GRAPH core components"""
    
    print("\n🏗️ Validating LIMIT-GRAPH Core...")
    
    try:
        from limit_graph_core import (
            LimitGraphNode, LimitGraphEdge, LimitQuery, 
            BaseLimitGraphComponent, LIMIT_GRAPH_REGISTRY,
            create_sample_limit_dataset
        )
        
        print("   ✅ Core data structures imported")
        
        # Test data structure creation
        node = LimitGraphNode("test_node", "entity", "test content", 0.8)
        edge = LimitGraphEdge("source", "target", "test_relation", 0.9)
        query = LimitQuery("q1", "test query", ["d1"], [])
        
        print("   ✅ Data structures creation works")
        
        # Test registry
        registry_status = LIMIT_GRAPH_REGISTRY.get_integration_status()
        print(f"   ✅ Registry works: {registry_status['total_components']} components")
        
        # Test sample dataset
        sample_data = create_sample_limit_dataset()
        print(f"   ✅ Sample dataset: {len(sample_data)} queries")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LIMIT-GRAPH core validation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    
    print("🧪 LIMIT-GRAPH Three Components Validation")
    print("=" * 50)
    
    validation_results = []
    
    # Run all validations
    validation_results.append(("Dependencies", validate_core_dependencies()))
    validation_results.append(("LIMIT-GRAPH Core", validate_limit_graph_core()))
    validation_results.append(("Graph Reasoner", validate_graph_reasoner()))
    validation_results.append(("RL Reward Function", validate_rl_reward_function()))
    validation_results.append(("Integration", validate_integration()))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Validation Summary")
    print("=" * 50)
    
    passed = 0
    total = len(validation_results)
    
    for component, result in validation_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {component}: {status}")
        if result:
            passed += 1
    
    success_rate = passed / total
    print(f"\nOverall: {passed}/{total} components validated ({success_rate:.1%})")
    
    if success_rate == 1.0:
        print("\n🎉 All components validated successfully!")
        print("\nNext steps:")
        print("   • Run: python extensions/demo_complete_limit_graph_system.py")
        print("   • Launch dashboard: python extensions/dashboardMemR1.py")
        print("   • Read guide: extensions/QUICK_START_GUIDE.md")
    else:
        print(f"\n⚠️ {total - passed} components failed validation")
        print("Check error messages above and install missing dependencies")
    
    return success_rate == 1.0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)