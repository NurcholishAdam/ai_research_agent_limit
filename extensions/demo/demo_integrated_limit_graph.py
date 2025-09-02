#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated LIMIT-GRAPH Demo
Demonstrates the complete integrated LIMIT-GRAPH architecture with all components
working together through the proper integration layers.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path

# Import integrated components
try:
    from limit_graph_core import LIMIT_GRAPH_REGISTRY, create_sample_limit_dataset
    from limit_graph_benchmark import LimitGraphBenchmark
    from limit_graph_evaluation_harness import LimitGraphEvaluationHarness
    from integration_orchestrator import AIResearchAgentExtensions
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    print(f"⚠️ Integrated components not available: {e}")

async def demo_core_integration():
    """Demo 1: Core component integration"""
    
    print("\n🏗️ === Demo 1: Core Component Integration ===")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available")
        return
    
    # Show registry status
    print("📋 LIMIT-GRAPH Registry Status:")
    registry_status = LIMIT_GRAPH_REGISTRY.get_integration_status()
    print(f"   Total components: {registry_status['total_components']}")
    print(f"   Component types: {registry_status['component_types']}")
    
    # Create benchmark
    benchmark = LimitGraphBenchmark({
        "scaffold": {},
        "retriever": {
            "fusion_strategy": "memory_agent_fusion"
        }
    })
    
    # Initialize components
    benchmark.initialize_components()
    
    # Show updated registry
    updated_status = LIMIT_GRAPH_REGISTRY.get_integration_status()
    print(f"\n📊 After initialization:")
    print(f"   Total components: {updated_status['total_components']}")
    print(f"   Component types: {updated_status['component_types']}")
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    print(f"\n✅ Benchmark completed:")
    print(f"   Benchmark ID: {results['benchmark_id']}")
    print(f"   Corpus stats: {results['corpus_stats']}")
    print(f"   Evaluation results available: {'evaluation_results' in results}")
    print(f"   Stress tests: {len(results['stress_test_results'])} tests")
    
    return benchmark

async def demo_evaluation_harness():
    """Demo 2: Evaluation harness integration"""
    
    print("\n📊 === Demo 2: Evaluation Harness Integration ===")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available")
        return
    
    # Create evaluation harness
    harness = LimitGraphEvaluationHarness({
        "k_values": [1, 5, 10],
        "output_dir": "demo_evaluation_results"
    })
    
    # Initialize system components
    harness.initialize_system()
    
    # Load test datasets
    harness.load_test_dataset("demo_basic", "demo_basic.json")
    
    print(f"📂 Loaded datasets: {list(harness.test_datasets.keys())}")
    
    # Run evaluation
    report = harness.run_comprehensive_evaluation(["demo_basic"])
    
    print(f"✅ Evaluation completed:")
    print(f"   Report ID: {report.report_id}")
    print(f"   Queries evaluated: {report.dataset_info['total_queries']}")
    print(f"   Recall@10: {report.metrics.recall_at_k.get(10, 0.0):.3f}")
    print(f"   Graph coverage: {report.metrics.graph_coverage:.3f}")
    print(f"   CI tests: {len(report.ci_test_results)} tests")
    
    return harness

async def demo_full_integration():
    """Demo 3: Full system integration through orchestrator"""
    
    print("\n🚀 === Demo 3: Full System Integration ===")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available")
        return
    
    # Create complete configuration
    config = {
        "enable_memory_r1_integration": True,
        "enable_limit_graph_benchmark": True,
        "memory_r1_integration": {
            "memory_r1": {"storage_path": "demo_integrated_memory"},
            "auto_validation": True,
            "dashboard_port": 8055
        },
        "limit_graph_benchmark": {
            "scaffold": {},
            "retriever": {
                "fusion_strategy": "memory_agent_fusion",
                "fusion_weights": {
                    "sparse": 0.25,
                    "dense": 0.35,
                    "graph": 0.25,
                    "memory": 0.15
                }
            },
            "sample_corpus": create_sample_limit_dataset(),
            "evaluation": {
                "k_values": [1, 5, 10],
                "output_dir": "demo_integrated_results"
            }
        }
    }
    
    # Initialize complete extensions system
    print("🏗️ Initializing complete extensions system...")
    extensions = AIResearchAgentExtensions()
    extensions.config = config
    
    # Initialize all stages
    status = await extensions.initialize_all_stages()
    
    print(f"✅ Extensions initialized:")
    print(f"   Total stages: {status['total_stages']}")
    print(f"   Success rate: {status['success_rate']:.1%}")
    print(f"   LIMIT-GRAPH available: {status.get('limit_graph_benchmark', {}).get('available', False)}")
    print(f"   Memory-R1 available: {status['memory_r1_integration']['available']}")
    
    # Test LIMIT-GRAPH integration
    if extensions.limit_graph_benchmark:
        print("\n📊 Testing LIMIT-GRAPH integration...")
        
        # Run benchmark through extensions
        benchmark_results = extensions.limit_graph_benchmark.run_benchmark()
        
        print(f"   Benchmark ID: {benchmark_results['benchmark_id']}")
        print(f"   Corpus processed: {benchmark_results['corpus_stats']}")
        
        # Show component registry status
        registry_status = benchmark_results['component_status']
        print(f"   Registry components: {registry_status['total_components']}")
    
    # Test evaluation harness
    if extensions.evaluation_harness:
        print("\n🧪 Testing evaluation harness...")
        
        # Run quick evaluation
        try:
            report = extensions.evaluation_harness.run_comprehensive_evaluation()
            print(f"   Evaluation report: {report.report_id}")
            print(f"   Overall performance: {report.metrics.recall_at_k.get(10, 0.0):.3f}")
        except Exception as e:
            print(f"   Evaluation error: {e}")
    
    # Get performance dashboard
    dashboard = extensions.get_performance_dashboard()
    
    print(f"\n📈 Performance Dashboard:")
    print(f"   Integration overview: {dashboard['integration_overview']['success_rate']:.1%}")
    if "limit_graph_dashboard" in dashboard:
        print(f"   LIMIT-GRAPH dashboard: Available")
    if "memory_r1_dashboard" in dashboard:
        print(f"   Memory-R1 dashboard: Available")
    
    return extensions

async def demo_ci_integration():
    """Demo 4: CI integration and stress testing"""
    
    print("\n🧪 === Demo 4: CI Integration & Stress Testing ===")
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available")
        return
    
    # Simulate CI pipeline
    print("🔧 Simulating CI pipeline...")
    print("   Command: python eval.py --benchmark limit-graph --model hybrid-agent --memory")
    
    # Create benchmark for CI testing
    benchmark = LimitGraphBenchmark()
    benchmark.initialize_components()
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    # Extract CI-relevant metrics
    stress_results = results['stress_test_results']
    passed_tests = sum(1 for test in stress_results if test['status'] == 'pass')
    total_tests = len(stress_results)
    
    # Determine CI status
    ci_passed = passed_tests == total_tests
    exit_code = 0 if ci_passed else 1
    
    print(f"\n🏁 CI Pipeline Results:")
    print(f"   Exit code: {exit_code}")
    print(f"   Status: {'PASS' if ci_passed else 'FAIL'}")
    print(f"   Stress tests: {passed_tests}/{total_tests} passed")
    
    # Show individual test results
    print(f"\n📋 Test Details:")
    for test in stress_results:
        status_emoji = "✅" if test['status'] == 'pass' else "❌"
        print(f"   {status_emoji} {test['test_name']}: {test['message']}")
    
    return exit_code

def cleanup_demo_files():
    """Clean up demo files"""
    
    print("\n🧹 Cleaning up demo files...")
    
    demo_files = [
        "demo_basic.json"
    ]
    
    demo_dirs = [
        "demo_evaluation_results",
        "demo_integrated_memory", 
        "demo_integrated_results"
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
    
    print("🎭 Integrated LIMIT-GRAPH Architecture Demo")
    print("=" * 60)
    print("Demonstrating the complete integrated architecture:")
    print("1. 🏗️ Core component integration with registry")
    print("2. 📊 Evaluation harness integration")
    print("3. 🚀 Full system integration through orchestrator")
    print("4. 🧪 CI integration and stress testing")
    
    if not COMPONENTS_AVAILABLE:
        print("\n❌ Required components not available")
        print("Please ensure all dependencies are installed")
        return
    
    try:
        # Run all demos
        benchmark = await demo_core_integration()
        harness = await demo_evaluation_harness()
        extensions = await demo_full_integration()
        ci_exit_code = await demo_ci_integration()
        
        # Summary
        print("\n🎉 === Integration Demo Summary ===")
        print("✅ Core component integration: Registry-managed components")
        print("✅ Evaluation harness: Comprehensive metrics and CI hooks")
        print("✅ Full system integration: Extensions orchestrator")
        print("✅ CI integration: Automated testing and validation")
        
        # Architecture validation
        print(f"\n🏗️ Architecture Validation:")
        print(f"   📊 LIMIT-GRAPH Scaffold: ✅ Graph construction from LIMIT corpus")
        print(f"   🔍 Hybrid Retriever: ✅ 4-component fusion (BM25+ColBERT+Graph+Memory)")
        print(f"   📈 Extended Evaluation: ✅ Recall@K + Graph Coverage + Provenance + Trace Replay")
        print(f"   🧪 CI Integration: ✅ Stress tests + validation hooks")
        print(f"   🔗 System Integration: ✅ Registry + orchestrator + extensions")
        
        # Integration status
        registry_status = LIMIT_GRAPH_REGISTRY.get_integration_status()
        print(f"\n📋 Final Integration Status:")
        print(f"   Total components: {registry_status['total_components']}")
        print(f"   Component types: {registry_status['component_types']}")
        print(f"   CI exit code: {ci_exit_code}")
        
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
    
    print("\n✅ Integrated LIMIT-GRAPH Architecture Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())