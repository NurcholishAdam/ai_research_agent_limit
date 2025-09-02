#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH Complete Demo
Demonstrates the complete LIMIT-GRAPH system with all components:
1. Graph construction from LIMIT corpus
2. Hybrid retrieval with 4 components
3. Evaluation harness with extended metrics
4. CI-evaluable stress tests
5. Integration with Memory-R1 system
"""

import json
import time
from datetime import datetime
from pathlib import Path

# Import components
try:
    from limit_graph_system import (
        LimitGraphScaffold, HybridAgentRetriever, LimitGraphEvaluator,
        LimitGraphStressTests, LimitQuery
    )
    from memory_r1_modular import MemoryR1Enhanced
    from ci_hooks_integration import CIHooksValidator
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    print(f"⚠️ Components not available: {e}")

def create_sample_limit_dataset():
    """Create sample LIMIT dataset matching the JSON structure"""
    
    dataset = [
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
    
    # Save dataset
    with open("sample_limit_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print("📄 Sample LIMIT dataset created")
    return dataset

def demo_graph_construction():
    """Demo 1: Benchmark Scaffold - LIMIT-GRAPH construction"""
    
    print("\n🏗️ === Demo 1: Graph Construction ===")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available")
        return None
    
    # Create sample dataset
    dataset = create_sample_limit_dataset()
    
    # Initialize LIMIT-GRAPH scaffold
    scaffold = LimitGraphScaffold()
    
    print("📚 Processing LIMIT corpus...")
    stats = scaffold.process_limit_corpus(dataset)
    
    print(f"✅ Graph construction completed:")
    print(f"   Documents: {stats['documents']}")
    print(f"   Entities: {stats['entities']}")
    print(f"   Relations: {stats['relations']}")
    print(f"   Graph nodes: {len(scaffold.nodes)}")
    print(f"   Graph edges: {len(scaffold.edges)}")
    
    # Show graph structure
    print(f"\n📊 Graph Structure:")
    for node_id, node in list(scaffold.nodes.items())[:5]:  # Show first 5 nodes
        print(f"   Node: {node_id} ({node.node_type}) - {node.content}")
    
    for edge in scaffold.edges[:3]:  # Show first 3 edges
        print(f"   Edge: {edge.source} --[{edge.relation}]--> {edge.target}")
    
    return scaffold

def demo_hybrid_retrieval(scaffold):
    """Demo 2: Retrieval Prototype - Hybrid Agent Retriever"""
    
    print("\n🔍 === Demo 2: Hybrid Retrieval ===")
    
    if not scaffold:
        return None
    
    # Initialize hybrid retriever
    retriever = HybridAgentRetriever(scaffold)
    
    # Test queries
    test_queries = [
        "Who likes apples?",
        "What does John own?",
        "Where is the library?"
    ]
    
    print("🧪 Testing hybrid retrieval components...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        
        # Perform retrieval
        result = retriever.retrieve(query, top_k=5)
        
        print(f"      Retrieved docs: {result['retrieved_docs']}")
        print(f"      Fusion scores: {result['fusion_scores']}")
        
        # Show component contributions
        comp_results = result["component_results"]
        print(f"      Component results:")
        print(f"        Sparse (BM25): {comp_results['sparse']}")
        print(f"        Dense (ColBERT): {comp_results['dense']}")
        print(f"        Graph Reasoner: {comp_results['graph']}")
        
        # Show provenance and trace
        print(f"      Provenance: {len(result['provenance'])} entries")
        print(f"      Trace ID: {result['trace_id']}")
    
    return retriever

def demo_evaluation_harness(scaffold, retriever):
    """Demo 3: Evaluation Harness with extended metrics"""
    
    print("\n📊 === Demo 3: Evaluation Harness ===")
    
    if not scaffold or not retriever:
        return None
    
    # Initialize evaluator
    evaluator = LimitGraphEvaluator(scaffold, retriever)
    
    # Create test queries
    dataset = create_sample_limit_dataset()
    test_queries = [LimitQuery(**item) for item in dataset]
    
    print(f"🧪 Evaluating {len(test_queries)} queries...")
    
    # Run evaluation
    results = evaluator.evaluate_retrieval(test_queries)
    
    print(f"✅ Evaluation completed:")
    print(f"\n📈 Recall@K Results:")
    for k in [1, 5, 10]:
        if k in results["recall_at_k"]:
            print(f"   Recall@{k}:")
            for component, score in results["recall_at_k"][k].items():
                print(f"     {component}: {score:.3f}")
    
    print(f"\n🕸️ Graph-based Metrics:")
    print(f"   Graph Coverage: {results['avg_graph_coverage']:.3f}")
    print(f"   Provenance Integrity: {results['avg_provenance_integrity']:.3f}")
    print(f"   Trace Replay Accuracy: {results['avg_trace_replay_accuracy']:.3f}")
    
    # Show per-query results
    print(f"\n📋 Per-Query Results:")
    for query_id, query_results in results["per_query_results"].items():
        print(f"   {query_id}: R@10={query_results['recall_at_10']:.3f}, "
              f"GC={query_results['graph_coverage']:.3f}, "
              f"PI={query_results['provenance_integrity']:.3f}")
    
    return evaluator

def demo_stress_tests(evaluator):
    """Demo 4: CI-evaluable stress tests"""
    
    print("\n🧪 === Demo 4: CI-Evaluable Stress Tests ===")
    
    if not evaluator:
        return []
    
    # Initialize stress tester
    stress_tester = LimitGraphStressTests(evaluator)
    
    print("🔧 Running comprehensive stress tests...")
    
    # Run all stress tests
    test_results = stress_tester.run_stress_tests()
    
    print(f"✅ Stress tests completed: {len(test_results)} tests")
    
    # Show results
    passed_tests = 0
    for result in test_results:
        status_emoji = "✅" if result.status == "pass" else "❌" if result.status == "fail" else "⚠️"
        print(f"   {status_emoji} {result.test_name}: {result.status}")
        print(f"      {result.message}")
        print(f"      Execution time: {result.execution_time:.3f}s")
        
        if result.status == "pass":
            passed_tests += 1
    
    print(f"\n📊 Stress Test Summary:")
    print(f"   Passed: {passed_tests}/{len(test_results)}")
    print(f"   Success rate: {passed_tests/len(test_results):.1%}")
    
    return test_results

def demo_memory_r1_integration():
    """Demo 5: Integration with Memory-R1 system"""
    
    print("\n🧠 === Demo 5: Memory-R1 Integration ===")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Memory-R1 components not available")
        return None
    
    try:
        # Initialize Memory-R1 system
        memory_system = MemoryR1Enhanced({
            "storage_path": "demo_limit_memory_r1"
        })
        
        # Initialize CI validator
        ci_validator = CIHooksValidator(memory_system)
        
        print("✅ Memory-R1 system initialized")
        
        # Process some queries through Memory-R1
        sample_queries = [
            "Alice likes apples and enjoys eating them daily.",
            "John owns a red car and a beautiful house.",
            "The library is located in the downtown area."
        ]
        
        print("📝 Processing queries through Memory-R1...")
        for query in sample_queries:
            result = memory_system.process_input(query)
            print(f"   Processed: {query[:40]}... -> {len(result.get('extracted_facts', []))} facts")
        
        # Run CI validation hooks
        print("\n🔧 Running Memory-R1 CI validation...")
        
        graph_result = ci_validator.validate_graph_state()
        print(f"   validate_graph_state(): {graph_result.status} - {graph_result.message}")
        
        provenance_result = ci_validator.check_provenance_integrity()
        print(f"   check_provenance_integrity(): {provenance_result.status} - {provenance_result.message}")
        
        replay_result = ci_validator.replay_trace(0, 2)
        print(f"   replay_trace(0, 2): {replay_result.status} - {replay_result.message}")
        
        # Show integration benefits
        print(f"\n🔗 Integration Benefits:")
        print(f"   ✅ Memory-R1 provides persistent semantic memory")
        print(f"   ✅ CI hooks enable automated validation")
        print(f"   ✅ Trace replay supports memory evolution analysis")
        print(f"   ✅ Provenance tracking ensures source lineage")
        
        return memory_system, ci_validator
        
    except Exception as e:
        print(f"❌ Memory-R1 integration failed: {e}")
        return None, None

def demo_cli_evaluation():
    """Demo 6: CLI evaluation matching the bash command"""
    
    print("\n💻 === Demo 6: CLI Evaluation ===")
    
    print("🔧 Simulating CLI command:")
    print("   python eval.py --benchmark limit-graph --model hybrid-agent --memory")
    
    # Import and run CLI function
    try:
        from limit_graph_system import run_evaluation_cli
        
        # Simulate CLI arguments
        import sys
        original_argv = sys.argv
        sys.argv = ["eval.py", "--benchmark", "limit-graph", "--model", "hybrid-agent", "--memory"]
        
        # Run evaluation
        exit_code = run_evaluation_cli()
        
        # Restore original argv
        sys.argv = original_argv
        
        print(f"\n🏁 CLI Evaluation completed with exit code: {exit_code}")
        
        if exit_code == 0:
            print("✅ All tests passed - CI would succeed")
        else:
            print("❌ Some tests failed - CI would fail")
        
        return exit_code
        
    except Exception as e:
        print(f"❌ CLI evaluation failed: {e}")
        return 1

def cleanup_demo_files():
    """Clean up demo files"""
    
    print("\n🧹 Cleaning up demo files...")
    
    demo_files = [
        "sample_limit_dataset.json"
    ]
    
    demo_dirs = [
        "demo_limit_memory_r1"
    ]
    
    # Remove files
    for file_path in demo_files:
        if Path(file_path).exists():
            Path(file_path).unlink()
            print(f"   Removed: {file_path}")
    
    # Remove directories
    import shutil
    for dir_path in demo_dirs:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
            print(f"   Removed: {dir_path}")
    
    print("✅ Cleanup completed")

def main():
    """Main demo function"""
    
    print("🎭 LIMIT-GRAPH Complete Demo")
    print("=" * 50)
    print("Demonstrating:")
    print("1. 🏗️ Graph construction from LIMIT corpus")
    print("2. 🔍 Hybrid retrieval (BM25 + ColBERT + Graph + Memory)")
    print("3. 📊 Evaluation harness with extended metrics")
    print("4. 🧪 CI-evaluable stress tests")
    print("5. 🧠 Memory-R1 system integration")
    print("6. 💻 CLI evaluation interface")
    
    if not COMPONENTS_AVAILABLE:
        print("\n❌ Required components not available")
        print("Please ensure all dependencies are installed")
        return
    
    try:
        # Run all demos
        scaffold = demo_graph_construction()
        retriever = demo_hybrid_retrieval(scaffold)
        evaluator = demo_evaluation_harness(scaffold, retriever)
        stress_results = demo_stress_tests(evaluator)
        memory_system, ci_validator = demo_memory_r1_integration()
        cli_exit_code = demo_cli_evaluation()
        
        # Summary
        print("\n🎉 === Demo Summary ===")
        print("✅ Graph construction: LIMIT corpus → semantic graph")
        print("✅ Hybrid retrieval: 4-component fusion architecture")
        print("✅ Evaluation metrics: Recall@K + Graph Coverage + Provenance + Trace Replay")
        print("✅ Stress tests: CI-evaluable performance validation")
        print("✅ Memory-R1 integration: Persistent memory + CI hooks")
        print("✅ CLI interface: Compatible with CI/CD pipelines")
        
        # Key metrics
        if evaluator and hasattr(evaluator, '_last_results'):
            results = evaluator._last_results
            print(f"\n📊 Key Performance Metrics:")
            print(f"   Recall@10 (fused): {results.get('recall_at_k', {}).get(10, {}).get('fused', 0):.3f}")
            print(f"   Graph Coverage: {results.get('avg_graph_coverage', 0):.3f}")
            print(f"   Provenance Integrity: {results.get('avg_provenance_integrity', 0):.3f}")
        
        if stress_results:
            passed_stress = sum(1 for r in stress_results if r.status == "pass")
            print(f"   Stress Tests: {passed_stress}/{len(stress_results)} passed")
        
        print(f"   CLI Exit Code: {cli_exit_code}")
        
        # Architecture summary
        print(f"\n🏗️ Architecture Implemented:")
        print(f"   📊 LIMIT-GRAPH Scaffold: Nodes (docs, entities, predicates) + Edges (relations)")
        print(f"   🔍 Hybrid Retriever: BM25 + ColBERT + Graph Reasoner + Memory Agent")
        print(f"   📈 Extended Evaluation: Recall@K + Graph Coverage + Provenance + Trace Replay")
        print(f"   🧪 CI Integration: Automated stress tests + validation hooks")
        print(f"   🧠 Memory Integration: Persistent memory + provenance tracking")
        
        # Ask about cleanup
        try:
            response = input("\n🧹 Clean up demo files? (Y/n): ").strip().lower()
            if response != 'n':
                cleanup_demo_files()
        except KeyboardInterrupt:
            print("\n🧹 Cleaning up...")
            cleanup_demo_files()
    
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ LIMIT-GRAPH Complete Demo finished!")

if __name__ == "__main__":
    main()