#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH Integration Demo
Demonstrates the complete integration of LIMIT-GRAPH benchmark with Memory-R1 system,
including graph-based relevance modeling, hybrid retrieval fusion, and CI-evaluable stress tests.
"""

import json
import time
from datetime import datetime
from pathlib import Path

# Import components
try:
    from limit_graph_benchmark import (
        create_limit_graph_system, LimitQuery, run_limit_graph_demo
    )
    from limit_graph_evaluation_harness import LimitGraphEvaluationHarness
    from memory_r1_complete_integration import create_complete_system
    from integration_orchestrator import AIResearchAgentExtensions
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    print(f"⚠️ Components not available: {e}")

def create_sample_limit_corpus():
    """Create sample LIMIT corpus for demonstration"""
    
    corpus_data = [
        {
            "query_id": "q1",
            "query": "Who likes apples?",
            "relevant_docs": ["d1", "d2"],
            "graph_edges": [
                {"source": "Alice", "target": "apples", "relation": "likes"},
                {"source": "Bob", "target": "apples", "relation": "likes"}
            ]
        },
        {
            "query_id": "q2", 
            "query": "What does John own?",
            "relevant_docs": ["d3", "d4"],
            "graph_edges": [
                {"source": "John", "target": "car", "relation": "owns"},
                {"source": "John", "target": "house", "relation": "owns"}
            ]
        },
        {
            "query_id": "q3",
            "query": "Where is the library located?",
            "relevant_docs": ["d5", "d6"],
            "graph_edges": [
                {"source": "library", "target": "downtown", "relation": "located_in"},
                {"source": "library", "target": "Main Street", "relation": "located_in"}
            ]
        },
        {
            "query_id": "q4",
            "query": "Find people who like fruits and what they own",
            "relevant_docs": ["d1", "d2", "d3", "d7"],
            "graph_edges": [
                {"source": "Alice", "target": "apples", "relation": "likes"},
                {"source": "Alice", "target": "bicycle", "relation": "owns"},
                {"source": "Bob", "target": "oranges", "relation": "likes"},
                {"source": "Bob", "target": "laptop", "relation": "owns"}
            ]
        }
    ]
    
    # Save corpus
    corpus_path = "demo_limit_corpus.json"
    with open(corpus_path, "w") as f:
        json.dump(corpus_data, f, indent=2)
    
    print(f"📄 Sample LIMIT corpus created: {corpus_path}")
    return corpus_path

def demo_graph_construction():
    """Demonstrate graph construction from LIMIT corpus"""
    
    print("\n🏗️ === Graph Construction Demo ===")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available")
        return None
    
    # Create sample corpus
    corpus_path = create_sample_limit_corpus()
    
    # Initialize LIMIT-GRAPH system
    graph_constructor, retriever, evaluator = create_limit_graph_system({
        "graph_constructor": {"corpus_path": corpus_path}
    })
    
    # Process corpus
    print("📚 Processing LIMIT corpus...")
    corpus_stats = graph_constructor.process_limit_corpus(corpus_path)
    
    print(f"✅ Graph construction completed:")
    print(f"   Documents processed: {corpus_stats['documents_processed']}")
    print(f"   Entities extracted: {corpus_stats['entities_extracted']}")
    print(f"   Relations extracted: {corpus_stats['relations_extracted']}")
    print(f"   Graph nodes: {corpus_stats['graph_nodes']}")
    print(f"   Graph edges: {corpus_stats['graph_edges']}")
    
    # Show graph statistics
    graph_stats = graph_constructor.get_graph_statistics()
    print(f"\n📊 Graph Statistics:")
    print(f"   Node types: {graph_stats['nodes']['by_type']}")
    print(f"   Relation types: {graph_stats['edges']['by_relation']}")
    print(f"   Graph density: {graph_stats['graph_metrics']['density']:.3f}")
    
    return graph_constructor, retriever, evaluator

def demo_hybrid_retrieval():
    """Demonstrate hybrid retrieval with all components"""
    
    print("\n🔍 === Hybrid Retrieval Demo ===")
    
    # Get system from previous demo
    graph_constructor, retriever, evaluator = demo_graph_construction()
    if not retriever:
        return
    
    # Test queries
    test_queries = [
        "Who likes apples?",
        "What does John own?", 
        "Where is the library?",
        "Find people who like fruits"
    ]
    
    print("🧪 Testing hybrid retrieval...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        
        # Perform retrieval
        result = retriever.retrieve(query, top_k=5)
        
        print(f"      Retrieved docs: {result.retrieved_docs}")
        print(f"      Graph coverage: {result.graph_coverage:.3f}")
        print(f"      Provenance integrity: {result.provenance_integrity:.3f}")
        print(f"      Fusion strategy: {result.fusion_strategy}")
        
        # Show component contributions
        component_results = result.metadata.get("component_results", {})
        print(f"      Component results: {component_results}")
    
    return retriever

def demo_evaluation_harness():
    """Demonstrate comprehensive evaluation harness"""
    
    print("\n📊 === Evaluation Harness Demo ===")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available")
        return
    
    # Initialize evaluation harness
    config = {
        "k_values": [1, 3, 5, 10],
        "output_dir": "demo_evaluation_results"
    }
    
    harness = LimitGraphEvaluationHarness(config)
    
    # Initialize system
    corpus_path = create_sample_limit_corpus()
    harness.initialize_system(corpus_path)
    
    # Load test datasets
    harness.load_test_dataset("basic_relations", "basic_relations.json")
    harness.load_test_dataset("complex_reasoning", "complex_reasoning.json")
    
    print(f"📂 Loaded {len(harness.test_datasets)} test datasets")
    
    # Run comprehensive evaluation
    print("🧪 Running comprehensive evaluation...")
    report = harness.run_comprehensive_evaluation()
    
    print(f"✅ Evaluation completed: {report.report_id}")
    print(f"   Queries evaluated: {report.dataset_info['total_queries']}")
    print(f"   Recall@5: {report.metrics.recall_at_k.get(5, 0.0):.3f}")
    print(f"   Recall@10: {report.metrics.recall_at_k.get(10, 0.0):.3f}")
    print(f"   Graph coverage: {report.metrics.graph_coverage:.3f}")
    print(f"   Provenance integrity: {report.metrics.provenance_integrity:.3f}")
    print(f"   Fusion effectiveness: {report.metrics.fusion_effectiveness:.3f}")
    
    # Show CI test results
    ci_passed = sum(1 for r in report.ci_test_results if r.status == "pass")
    ci_total = len(report.ci_test_results)
    print(f"   CI tests: {ci_passed}/{ci_total} passed")
    
    # Show recommendations
    if report.recommendations:
        print(f"   Recommendations:")
        for rec in report.recommendations[:3]:  # Show first 3
            print(f"      - {rec}")
    
    return harness, report

def demo_memory_r1_integration():
    """Demonstrate integration with Memory-R1 system"""
    
    print("\n🧠 === Memory-R1 Integration Demo ===")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available")
        return
    
    # Create Memory-R1 complete system
    memory_config = {
        "memory_r1": {"storage_path": "demo_memory_r1_limit"},
        "auto_validation": True,
        "auto_training": False,
        "dashboard_port": 8053
    }
    
    memory_system = create_complete_system(memory_config)
    
    # Create LIMIT-GRAPH system
    corpus_path = create_sample_limit_corpus()
    graph_constructor, retriever, evaluator = create_limit_graph_system()
    
    # Initialize evaluation harness with Memory-R1 integration
    harness = LimitGraphEvaluationHarness({
        "output_dir": "demo_integrated_results"
    })
    
    harness.initialize_system(corpus_path)
    harness.memory_system = memory_system.memory_system
    harness.ci_validator = memory_system.ci_validator
    
    print("✅ Memory-R1 integration established")
    
    # Process some queries through Memory-R1
    sample_queries = [
        "Alice likes apples and enjoys eating them.",
        "John owns a red car and a house.",
        "The library is located downtown on Main Street."
    ]
    
    print("📝 Processing queries through Memory-R1...")
    for query in sample_queries:
        result = memory_system.process_research_query(query)
        print(f"   Processed: {query[:50]}...")
    
    # Load test datasets and run evaluation
    harness.load_test_dataset("integrated_test", "integrated_test.json")
    
    # Run evaluation with Memory-R1 integration
    print("🧪 Running integrated evaluation...")
    report = harness.run_comprehensive_evaluation(["integrated_test"])
    
    print(f"✅ Integrated evaluation completed:")
    print(f"   Memory system status: {'✅' if harness.memory_system else '❌'}")
    print(f"   CI validator status: {'✅' if harness.ci_validator else '❌'}")
    print(f"   Trace replay accuracy: {report.metrics.trace_replay_accuracy:.3f}")
    
    return memory_system, harness

def demo_complete_integration():
    """Demonstrate complete integration with all extensions"""
    
    print("\n🚀 === Complete Integration Demo ===")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available")
        return
    
    # Create complete configuration
    config = {
        "enable_memory_r1_integration": True,
        "enable_limit_graph_benchmark": True,
        "memory_r1_integration": {
            "memory_r1": {"storage_path": "demo_complete_memory"},
            "auto_validation": True,
            "dashboard_port": 8054
        },
        "limit_graph_benchmark": {
            "evaluation": {"output_dir": "demo_complete_results"},
            "default_datasets": ["basic_relations", "complex_reasoning"]
        }
    }
    
    # Initialize complete extensions system
    print("🏗️ Initializing complete extensions system...")
    extensions = AIResearchAgentExtensions("demo_integration_config.json")
    extensions.config = config
    
    # Initialize all stages
    status = await extensions.initialize_all_stages()
    
    print(f"✅ Extensions initialized:")
    print(f"   Total stages: {status['total_stages']}")
    print(f"   Success rate: {status['success_rate']:.1%}")
    print(f"   Memory-R1 available: {status['memory_r1_integration']['available']}")
    
    # Test LIMIT-GRAPH integration
    if extensions.evaluation_harness:
        print("\n📊 Testing LIMIT-GRAPH integration...")
        
        # Create sample corpus
        corpus_path = create_sample_limit_corpus()
        extensions.evaluation_harness.initialize_system(corpus_path)
        
        # Run quick evaluation
        report = extensions.evaluation_harness.run_comprehensive_evaluation()
        
        print(f"   Evaluation report: {report.report_id}")
        print(f"   Overall performance: {report.metrics.recall_at_k.get(10, 0.0):.3f}")
    
    # Get performance dashboard
    dashboard = extensions.get_performance_dashboard()
    
    print(f"\n📈 Performance Dashboard:")
    print(f"   Integration overview: {dashboard['integration_overview']['success_rate']:.1%}")
    if "memory_r1_dashboard" in dashboard:
        print(f"   Memory-R1 dashboard: Available")
    
    return extensions

def demo_ci_evaluation():
    """Demonstrate CI-evaluable stress tests"""
    
    print("\n🧪 === CI-Evaluable Stress Tests Demo ===")
    
    # Run the evaluation as if it were a CI pipeline
    print("🔧 Running CI evaluation pipeline...")
    
    # Simulate CI command: python eval.py --benchmark limit-graph --model hybrid-agent --memory
    try:
        # Initialize system
        harness = LimitGraphEvaluationHarness({
            "output_dir": "ci_evaluation_results"
        })
        
        corpus_path = create_sample_limit_corpus()
        harness.initialize_system(corpus_path)
        
        # Load test datasets
        harness.load_test_dataset("ci_test", "ci_test.json")
        
        # Run evaluation
        report = harness.run_comprehensive_evaluation(["ci_test"])
        
        # Check CI criteria
        ci_passed = True
        ci_messages = []
        
        # Recall@10 threshold
        recall_10 = report.metrics.recall_at_k.get(10, 0.0)
        if recall_10 < 0.5:
            ci_passed = False
            ci_messages.append(f"Recall@10 below threshold: {recall_10:.3f} < 0.5")
        
        # Graph coverage threshold
        if report.metrics.graph_coverage < 0.3:
            ci_passed = False
            ci_messages.append(f"Graph coverage below threshold: {report.metrics.graph_coverage:.3f} < 0.3")
        
        # Provenance integrity threshold
        if report.metrics.provenance_integrity < 0.8:
            ci_passed = False
            ci_messages.append(f"Provenance integrity below threshold: {report.metrics.provenance_integrity:.3f} < 0.8")
        
        # CI test failures
        failed_ci_tests = [r for r in report.ci_test_results if r.status == "fail"]
        if failed_ci_tests:
            ci_passed = False
            ci_messages.append(f"{len(failed_ci_tests)} CI tests failed")
        
        # Print CI results
        exit_code = 0 if ci_passed else 1
        status = "PASS" if ci_passed else "FAIL"
        
        print(f"🏁 CI Evaluation Result: {status}")
        print(f"   Exit code: {exit_code}")
        print(f"   Recall@10: {recall_10:.3f}")
        print(f"   Graph coverage: {report.metrics.graph_coverage:.3f}")
        print(f"   Provenance integrity: {report.metrics.provenance_integrity:.3f}")
        print(f"   CI tests passed: {len(report.ci_test_results) - len(failed_ci_tests)}/{len(report.ci_test_results)}")
        
        if ci_messages:
            print(f"   Issues:")
            for msg in ci_messages:
                print(f"      - {msg}")
        
        return exit_code
        
    except Exception as e:
        print(f"❌ CI evaluation failed: {e}")
        return 1

def cleanup_demo_files():
    """Clean up demo files"""
    
    print("\n🧹 Cleaning up demo files...")
    
    demo_files = [
        "demo_limit_corpus.json",
        "basic_relations.json",
        "complex_reasoning.json",
        "integrated_test.json",
        "ci_test.json",
        "demo_integration_config.json"
    ]
    
    demo_dirs = [
        "demo_evaluation_results",
        "demo_memory_r1_limit",
        "demo_integrated_results",
        "demo_complete_memory",
        "demo_complete_results",
        "ci_evaluation_results"
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

async def main():
    """Main demo function"""
    
    print("🎭 LIMIT-GRAPH Integration Demo")
    print("=" * 50)
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Required components not available")
        print("   Please ensure all dependencies are installed")
        return
    
    try:
        # Demo 1: Graph Construction
        demo_graph_construction()
        
        # Demo 2: Hybrid Retrieval
        demo_hybrid_retrieval()
        
        # Demo 3: Evaluation Harness
        demo_evaluation_harness()
        
        # Demo 4: Memory-R1 Integration
        demo_memory_r1_integration()
        
        # Demo 5: Complete Integration
        await demo_complete_integration()
        
        # Demo 6: CI Evaluation
        ci_exit_code = demo_ci_evaluation()
        
        print("\n🎉 === Demo Summary ===")
        print("✅ Graph construction from LIMIT corpus")
        print("✅ Hybrid retrieval with sparse + dense + graph + memory")
        print("✅ Comprehensive evaluation harness")
        print("✅ Memory-R1 system integration")
        print("✅ Complete extensions integration")
        print(f"✅ CI evaluation pipeline (exit code: {ci_exit_code})")
        
        print(f"\n📊 Key Features Demonstrated:")
        print(f"   🏗️ LIMIT corpus → semantic graph construction")
        print(f"   🔍 4-component hybrid retrieval (BM25 + ColBERT + Graph + Memory)")
        print(f"   📈 Recall@K, graph coverage, provenance integrity metrics")
        print(f"   🧠 Memory-R1 integration with trace replay")
        print(f"   🔧 CI-evaluable stress tests and validation hooks")
        print(f"   📊 Comprehensive evaluation reports and visualizations")
        
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
    
    print("\n✅ LIMIT-GRAPH Integration Demo completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())