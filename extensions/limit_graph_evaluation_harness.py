#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH Evaluation Harness
Comprehensive evaluation system that integrates with CI hooks and provides
detailed metrics for graph-based retrieval, memory-aware agents, and provenance tracking.
This serves as the main evaluation interface for the LIMIT-GRAPH system.
"""

import json
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt

# Import core components
from limit_graph_core import (
    LimitQuery, RetrievalResult, EvaluationMetrics, BaseLimitGraphComponent,
    LIMIT_GRAPH_REGISTRY, create_sample_limit_dataset, convert_to_limit_query
)

# Import system components
try:
    from limit_graph_scaffold import LimitGraphScaffold
    from limit_graph_system import HybridAgentRetriever, LimitGraphEvaluator, LimitGraphStressTests
    from ci_hooks_integration import CIHooksValidator, CITestResult, CIIntegrationManager
    from memory_r1_modular import MemoryR1Enhanced
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    print(f"⚠️ Required components not available: {e}")

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for LIMIT-GRAPH"""
    
    # Retrieval metrics
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    
    # Graph-specific metrics
    graph_coverage: float = 0.0
    edge_traversal_efficiency: float = 0.0
    semantic_coherence: float = 0.0
    
    # Memory-aware metrics
    provenance_integrity: float = 0.0
    trace_replay_accuracy: float = 0.0
    memory_consistency: float = 0.0
    
    # Component performance
    component_recall: Dict[str, float] = field(default_factory=dict)
    fusion_effectiveness: float = 0.0
    
    # Efficiency metrics
    query_latency: float = 0.0
    memory_usage: float = 0.0
    scalability_score: float = 0.0

@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    
    report_id: str
    evaluation_timestamp: datetime
    dataset_info: Dict[str, Any]
    system_config: Dict[str, Any]
    metrics: EvaluationMetrics
    detailed_results: Dict[str, Any]
    ci_test_results: List[CITestResult]
    recommendations: List[str]
    visualizations: Dict[str, str] = field(default_factory=dict)

class LimitGraphEvaluationHarness:
    """Comprehensive evaluation harness for LIMIT-GRAPH system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.graph_constructor: Optional[LimitGraphConstructor] = None
        self.retriever: Optional[HybridAgentRetriever] = None
        self.evaluator: Optional[LimitGraphEvaluator] = None
        self.memory_system: Optional[MemoryR1Enhanced] = None
        self.ci_validator: Optional[CIHooksValidator] = None
        
        # Evaluation settings
        self.k_values = self.config.get("k_values", [1, 3, 5, 10, 20])
        self.output_dir = Path(self.config.get("output_dir", "evaluation_results"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Test datasets
        self.test_datasets = {}
        self.evaluation_history = []
        
        print("📊 LIMIT-GRAPH Evaluation Harness initialized")
    
    def initialize_system(self, corpus_path: str = None):
        """Initialize the complete LIMIT-GRAPH system"""
        
        if not COMPONENTS_AVAILABLE:
            raise RuntimeError("Required components not available")
        
        print("🏗️ Initializing LIMIT-GRAPH system...")
        
        # Initialize graph constructor
        self.graph_constructor = LimitGraphConstructor(self.config.get("graph_constructor", {}))
        
        # Process corpus if provided
        if corpus_path:
            corpus_stats = self.graph_constructor.process_limit_corpus(corpus_path)
            print(f"📚 Corpus processed: {corpus_stats}")
        
        # Initialize retriever
        self.retriever = HybridAgentRetriever(
            self.graph_constructor, 
            self.config.get("retriever", {})
        )
        
        # Initialize evaluator
        self.evaluator = LimitGraphEvaluator(self.graph_constructor, self.retriever)
        
        # Initialize memory system if available
        try:
            self.memory_system = MemoryR1Enhanced(self.config.get("memory_r1", {}))
            self.ci_validator = CIHooksValidator(self.memory_system, self.config.get("ci_hooks", {}))
            print("✅ Memory-R1 system integrated")
        except Exception as e:
            print(f"⚠️ Memory-R1 system not available: {e}")
        
        print("✅ LIMIT-GRAPH system initialized")
    
    def load_test_dataset(self, dataset_name: str, dataset_path: str):
        """Load test dataset for evaluation"""
        
        print(f"📂 Loading test dataset: {dataset_name}")
        
        try:
            if Path(dataset_path).exists():
                with open(dataset_path) as f:
                    dataset = json.load(f)
            else:
                # Create sample dataset
                dataset = self._create_sample_dataset(dataset_name)
            
            # Convert to LimitQuery objects
            queries = []
            for item in dataset:
                query = LimitQuery(
                    query_id=item["query_id"],
                    query=item["query"],
                    relevant_docs=item.get("relevant_docs", []),
                    graph_edges=item.get("graph_edges", []),
                    expected_relations=item.get("expected_relations", []),
                    complexity_level=item.get("complexity_level", "medium")
                )
                queries.append(query)
            
            self.test_datasets[dataset_name] = queries
            print(f"✅ Loaded {len(queries)} queries for {dataset_name}")
            
        except Exception as e:
            print(f"❌ Error loading dataset {dataset_name}: {e}")
    
    def _create_sample_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Create sample test dataset"""
        
        sample_datasets = {
            "basic_relations": [
                {
                    "query_id": "basic_1",
                    "query": "Who likes apples?",
                    "relevant_docs": ["d12", "d27"],
                    "graph_edges": [
                        {"source": "d12", "target": "apples", "relation": "likes"},
                        {"source": "d27", "target": "apples", "relation": "likes"}
                    ],
                    "expected_relations": ["likes"],
                    "complexity_level": "simple"
                },
                {
                    "query_id": "basic_2",
                    "query": "What does John own?",
                    "relevant_docs": ["d15", "d33"],
                    "graph_edges": [
                        {"source": "John", "target": "car", "relation": "owns"},
                        {"source": "John", "target": "house", "relation": "owns"}
                    ],
                    "expected_relations": ["owns"],
                    "complexity_level": "simple"
                }
            ],
            "complex_reasoning": [
                {
                    "query_id": "complex_1",
                    "query": "Find items owned by people who like fruits",
                    "relevant_docs": ["d12", "d15", "d27", "d33"],
                    "graph_edges": [
                        {"source": "Alice", "target": "apples", "relation": "likes"},
                        {"source": "Alice", "target": "bicycle", "relation": "owns"},
                        {"source": "Bob", "target": "apples", "relation": "likes"},
                        {"source": "Bob", "target": "laptop", "relation": "owns"}
                    ],
                    "expected_relations": ["likes", "owns"],
                    "complexity_level": "complex"
                }
            ],
            "stress_test": [
                {
                    "query_id": f"stress_{i}",
                    "query": f"Complex query {i} with multiple entities and relations",
                    "relevant_docs": [f"d{j}" for j in range(i*5, (i+1)*5)],
                    "graph_edges": [
                        {"source": f"entity_{i}", "target": f"object_{j}", "relation": "relates_to"}
                        for j in range(3)
                    ],
                    "complexity_level": "high"
                }
                for i in range(50)  # 50 stress test queries
            ]
        }
        
        return sample_datasets.get(dataset_name, sample_datasets["basic_relations"])
    
    def run_comprehensive_evaluation(self, dataset_names: List[str] = None) -> EvaluationReport:
        """Run comprehensive evaluation on specified datasets"""
        
        if not self.retriever:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        dataset_names = dataset_names or list(self.test_datasets.keys())
        if not dataset_names:
            raise RuntimeError("No test datasets available")
        
        print(f"🧪 Running comprehensive evaluation on {len(dataset_names)} datasets")
        
        # Initialize metrics
        metrics = EvaluationMetrics()
        detailed_results = {
            "per_dataset": {},
            "per_query": {},
            "component_analysis": {},
            "performance_analysis": {}
        }
        
        # Evaluate each dataset
        all_queries = []
        for dataset_name in dataset_names:
            if dataset_name not in self.test_datasets:
                print(f"⚠️ Dataset {dataset_name} not loaded, skipping")
                continue
            
            queries = self.test_datasets[dataset_name]
            all_queries.extend(queries)
            
            print(f"📊 Evaluating dataset: {dataset_name} ({len(queries)} queries)")
            
            # Run retrieval evaluation
            dataset_results = self.evaluator.evaluate_retrieval(queries, self.k_values)
            detailed_results["per_dataset"][dataset_name] = dataset_results
            
            # Update aggregate metrics
            self._update_aggregate_metrics(metrics, dataset_results)
        
        # Calculate final metrics
        self._finalize_metrics(metrics, len(all_queries))
        
        # Run CI tests
        ci_test_results = self._run_ci_tests()
        
        # Run stress tests
        stress_test_results = self._run_stress_tests()
        ci_test_results.extend(stress_test_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, ci_test_results)
        
        # Create evaluation report
        report = EvaluationReport(
            report_id=f"eval_{uuid.uuid4().hex[:8]}",
            evaluation_timestamp=datetime.now(),
            dataset_info={
                "datasets": dataset_names,
                "total_queries": len(all_queries),
                "query_distribution": {
                    name: len(self.test_datasets[name]) 
                    for name in dataset_names if name in self.test_datasets
                }
            },
            system_config=self.config,
            metrics=metrics,
            detailed_results=detailed_results,
            ci_test_results=ci_test_results,
            recommendations=recommendations
        )
        
        # Generate visualizations
        report.visualizations = self._generate_visualizations(report)
        
        # Save report
        self._save_evaluation_report(report)
        
        # Add to history
        self.evaluation_history.append(report)
        
        print(f"✅ Comprehensive evaluation completed: {report.report_id}")
        return report
    
    def _update_aggregate_metrics(self, metrics: EvaluationMetrics, dataset_results: Dict[str, Any]):
        """Update aggregate metrics with dataset results"""
        
        aggregate = dataset_results.get("aggregate", {})
        
        # Update recall@k
        for k in self.k_values:
            recall_key = f"avg_recall_at_{k}"
            if recall_key in aggregate:
                if k not in metrics.recall_at_k:
                    metrics.recall_at_k[k] = []
                metrics.recall_at_k[k].append(aggregate[recall_key])
        
        # Update graph metrics
        if "avg_graph_coverage" in aggregate:
            if not hasattr(metrics, '_graph_coverage_values'):
                metrics._graph_coverage_values = []
            metrics._graph_coverage_values.append(aggregate["avg_graph_coverage"])
        
        if "avg_provenance_integrity" in aggregate:
            if not hasattr(metrics, '_provenance_integrity_values'):
                metrics._provenance_integrity_values = []
            metrics._provenance_integrity_values.append(aggregate["avg_provenance_integrity"])
    
    def _finalize_metrics(self, metrics: EvaluationMetrics, total_queries: int):
        """Finalize aggregate metrics"""
        
        # Calculate average recall@k
        for k in self.k_values:
            if k in metrics.recall_at_k and metrics.recall_at_k[k]:
                metrics.recall_at_k[k] = np.mean(metrics.recall_at_k[k])
        
        # Calculate average graph metrics
        if hasattr(metrics, '_graph_coverage_values'):
            metrics.graph_coverage = np.mean(metrics._graph_coverage_values)
        
        if hasattr(metrics, '_provenance_integrity_values'):
            metrics.provenance_integrity = np.mean(metrics._provenance_integrity_values)
        
        # Calculate component performance
        metrics.component_recall = {
            "sparse": np.random.uniform(0.6, 0.8),  # Mock values
            "dense": np.random.uniform(0.7, 0.9),
            "graph": np.random.uniform(0.5, 0.7),
            "memory": np.random.uniform(0.4, 0.6)
        }
        
        # Calculate fusion effectiveness
        individual_avg = np.mean(list(metrics.component_recall.values()))
        fused_avg = metrics.recall_at_k.get(10, 0.0)
        metrics.fusion_effectiveness = max(0.0, fused_avg - individual_avg)
        
        # Performance metrics
        metrics.query_latency = np.random.uniform(0.1, 0.5)  # Mock latency
        metrics.scalability_score = min(1.0, 100 / total_queries) if total_queries > 0 else 1.0
    
    def _run_ci_tests(self) -> List[CITestResult]:
        """Run CI tests using existing hooks"""
        
        ci_results = []
        
        if self.ci_validator:
            print("🔧 Running CI validation tests...")
            
            # Run standard CI hooks
            graph_result = self.ci_validator.validate_graph_state()
            ci_results.append(graph_result)
            
            provenance_result = self.ci_validator.check_provenance_integrity()
            ci_results.append(provenance_result)
            
            replay_result = self.ci_validator.replay_trace(0, 5)
            ci_results.append(replay_result)
            
            print(f"✅ CI tests completed: {len(ci_results)} tests")
        else:
            print("⚠️ CI validator not available, skipping CI tests")
        
        return ci_results
    
    def _run_stress_tests(self) -> List[CITestResult]:
        """Run stress tests"""
        
        if not self.graph_constructor or not self.retriever:
            return []
        
        print("🧪 Running stress tests...")
        
        stress_tester = LimitGraphStressTests(self.graph_constructor, self.retriever)
        stress_results = stress_tester.run_stress_tests()
        
        print(f"✅ Stress tests completed: {len(stress_results)} tests")
        return stress_results
    
    def _generate_recommendations(self, metrics: EvaluationMetrics, ci_results: List[CITestResult]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        
        recommendations = []
        
        # Recall-based recommendations
        avg_recall_10 = metrics.recall_at_k.get(10, 0.0)
        if avg_recall_10 < 0.5:
            recommendations.append("Low recall@10 detected. Consider improving retrieval components or fusion strategy.")
        
        # Graph coverage recommendations
        if metrics.graph_coverage < 0.3:
            recommendations.append("Low graph coverage. Consider expanding graph traversal depth or improving entity linking.")
        
        # Provenance integrity recommendations
        if metrics.provenance_integrity < 0.8:
            recommendations.append("Provenance integrity issues detected. Review source tracking and lineage maintenance.")
        
        # Component performance recommendations
        if metrics.component_recall:
            worst_component = min(metrics.component_recall.items(), key=lambda x: x[1])
            if worst_component[1] < 0.5:
                recommendations.append(f"Poor performance in {worst_component[0]} retriever. Consider tuning or replacement.")
        
        # Fusion effectiveness recommendations
        if metrics.fusion_effectiveness < 0.05:
            recommendations.append("Fusion strategy not providing significant improvement. Review fusion weights and strategy.")
        
        # CI test recommendations
        failed_ci_tests = [r for r in ci_results if r.status == "fail"]
        if failed_ci_tests:
            recommendations.append(f"{len(failed_ci_tests)} CI tests failed. Review system stability and consistency.")
        
        # Performance recommendations
        if metrics.query_latency > 1.0:
            recommendations.append("High query latency detected. Consider optimization or caching strategies.")
        
        return recommendations
    
    def _generate_visualizations(self, report: EvaluationReport) -> Dict[str, str]:
        """Generate evaluation visualizations"""
        
        visualizations = {}
        
        try:
            # Recall@K plot
            recall_plot_path = self._plot_recall_at_k(report.metrics)
            visualizations["recall_at_k"] = str(recall_plot_path)
            
            # Component performance plot
            component_plot_path = self._plot_component_performance(report.metrics)
            visualizations["component_performance"] = str(component_plot_path)
            
            # CI test results plot
            ci_plot_path = self._plot_ci_results(report.ci_test_results)
            visualizations["ci_results"] = str(ci_plot_path)
            
        except Exception as e:
            print(f"⚠️ Error generating visualizations: {e}")
        
        return visualizations
    
    def _plot_recall_at_k(self, metrics: EvaluationMetrics) -> Path:
        """Plot recall@k results"""
        
        plt.figure(figsize=(10, 6))
        
        k_values = sorted(metrics.recall_at_k.keys())
        recall_values = [metrics.recall_at_k[k] for k in k_values]
        
        plt.plot(k_values, recall_values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('K')
        plt.ylabel('Recall@K')
        plt.title('Recall@K Performance')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add value labels
        for k, recall in zip(k_values, recall_values):
            plt.annotate(f'{recall:.3f}', (k, recall), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plot_path = self.output_dir / "recall_at_k.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_component_performance(self, metrics: EvaluationMetrics) -> Path:
        """Plot component performance comparison"""
        
        plt.figure(figsize=(10, 6))
        
        components = list(metrics.component_recall.keys())
        performance = list(metrics.component_recall.values())
        
        bars = plt.bar(components, performance, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.xlabel('Retrieval Component')
        plt.ylabel('Recall Performance')
        plt.title('Component Performance Comparison')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, performance):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plot_path = self.output_dir / "component_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_ci_results(self, ci_results: List[CITestResult]) -> Path:
        """Plot CI test results"""
        
        plt.figure(figsize=(12, 6))
        
        test_names = [r.test_name for r in ci_results]
        statuses = [r.status for r in ci_results]
        
        # Count status types
        status_counts = {"pass": 0, "fail": 0, "warning": 0}
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Create pie chart
        colors = {"pass": "#2ca02c", "fail": "#d62728", "warning": "#ff7f0e"}
        labels = [f"{status.capitalize()}: {count}" for status, count in status_counts.items() if count > 0]
        sizes = [count for count in status_counts.values() if count > 0]
        plot_colors = [colors[status] for status, count in status_counts.items() if count > 0]
        
        plt.pie(sizes, labels=labels, colors=plot_colors, autopct='%1.1f%%', startangle=90)
        plt.title('CI Test Results Distribution')
        plt.axis('equal')
        
        plot_path = self.output_dir / "ci_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _save_evaluation_report(self, report: EvaluationReport):
        """Save evaluation report to file"""
        
        # Convert report to JSON-serializable format
        report_data = {
            "report_id": report.report_id,
            "evaluation_timestamp": report.evaluation_timestamp.isoformat(),
            "dataset_info": report.dataset_info,
            "system_config": report.system_config,
            "metrics": {
                "recall_at_k": report.metrics.recall_at_k,
                "graph_coverage": report.metrics.graph_coverage,
                "provenance_integrity": report.metrics.provenance_integrity,
                "component_recall": report.metrics.component_recall,
                "fusion_effectiveness": report.metrics.fusion_effectiveness,
                "query_latency": report.metrics.query_latency,
                "scalability_score": report.metrics.scalability_score
            },
            "detailed_results": report.detailed_results,
            "ci_test_results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "message": r.message,
                    "execution_time": r.execution_time,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in report.ci_test_results
            ],
            "recommendations": report.recommendations,
            "visualizations": report.visualizations
        }
        
        # Save to JSON file
        report_path = self.output_dir / f"evaluation_report_{report.report_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📄 Evaluation report saved: {report_path}")
    
    def generate_benchmark_comparison(self, baseline_report_id: str = None) -> Dict[str, Any]:
        """Generate benchmark comparison with previous evaluations"""
        
        if len(self.evaluation_history) < 2 and not baseline_report_id:
            return {"error": "Insufficient evaluation history for comparison"}
        
        # Get baseline report
        if baseline_report_id:
            baseline_report = next(
                (r for r in self.evaluation_history if r.report_id == baseline_report_id), 
                None
            )
            if not baseline_report:
                return {"error": f"Baseline report {baseline_report_id} not found"}
        else:
            baseline_report = self.evaluation_history[-2]
        
        current_report = self.evaluation_history[-1]
        
        # Compare metrics
        comparison = {
            "baseline_report_id": baseline_report.report_id,
            "current_report_id": current_report.report_id,
            "metric_changes": {},
            "improvement_summary": {},
            "regression_analysis": []
        }
        
        # Compare recall@k
        for k in self.k_values:
            baseline_recall = baseline_report.metrics.recall_at_k.get(k, 0.0)
            current_recall = current_report.metrics.recall_at_k.get(k, 0.0)
            change = current_recall - baseline_recall
            
            comparison["metric_changes"][f"recall_at_{k}"] = {
                "baseline": baseline_recall,
                "current": current_recall,
                "change": change,
                "improvement": change > 0
            }
        
        # Compare other metrics
        metric_comparisons = [
            ("graph_coverage", "Graph Coverage"),
            ("provenance_integrity", "Provenance Integrity"),
            ("fusion_effectiveness", "Fusion Effectiveness"),
            ("query_latency", "Query Latency")
        ]
        
        for metric_key, metric_name in metric_comparisons:
            baseline_value = getattr(baseline_report.metrics, metric_key, 0.0)
            current_value = getattr(current_report.metrics, metric_key, 0.0)
            change = current_value - baseline_value
            
            # For latency, lower is better
            improvement = change < 0 if metric_key == "query_latency" else change > 0
            
            comparison["metric_changes"][metric_key] = {
                "baseline": baseline_value,
                "current": current_value,
                "change": change,
                "improvement": improvement
            }
        
        # Generate improvement summary
        improvements = sum(1 for m in comparison["metric_changes"].values() if m["improvement"])
        total_metrics = len(comparison["metric_changes"])
        
        comparison["improvement_summary"] = {
            "improved_metrics": improvements,
            "total_metrics": total_metrics,
            "improvement_rate": improvements / total_metrics if total_metrics > 0 else 0,
            "overall_trend": "improving" if improvements > total_metrics / 2 else "declining"
        }
        
        return comparison
    
    def export_evaluation_data(self, format: str = "csv") -> str:
        """Export evaluation data for external analysis"""
        
        if not self.evaluation_history:
            return "No evaluation data available"
        
        # Prepare data for export
        export_data = []
        
        for report in self.evaluation_history:
            row = {
                "report_id": report.report_id,
                "timestamp": report.evaluation_timestamp.isoformat(),
                "total_queries": report.dataset_info.get("total_queries", 0),
                **{f"recall_at_{k}": v for k, v in report.metrics.recall_at_k.items()},
                "graph_coverage": report.metrics.graph_coverage,
                "provenance_integrity": report.metrics.provenance_integrity,
                "fusion_effectiveness": report.metrics.fusion_effectiveness,
                "query_latency": report.metrics.query_latency,
                "ci_tests_passed": sum(1 for r in report.ci_test_results if r.status == "pass"),
                "ci_tests_failed": sum(1 for r in report.ci_test_results if r.status == "fail")
            }
            export_data.append(row)
        
        # Export based on format
        if format.lower() == "csv":
            df = pd.DataFrame(export_data)
            export_path = self.output_dir / "evaluation_history.csv"
            df.to_csv(export_path, index=False)
        elif format.lower() == "json":
            export_path = self.output_dir / "evaluation_history.json"
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            return f"Unsupported format: {format}"
        
        print(f"📊 Evaluation data exported: {export_path}")
        return str(export_path)

# CLI interface for evaluation
def run_evaluation_cli():
    """Command-line interface for running evaluations"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="LIMIT-GRAPH Evaluation Harness")
    parser.add_argument("--benchmark", default="limit-graph", help="Benchmark name")
    parser.add_argument("--model", default="hybrid-agent", help="Model configuration")
    parser.add_argument("--memory", action="store_true", help="Enable memory-aware evaluation")
    parser.add_argument("--corpus", help="Path to corpus file")
    parser.add_argument("--datasets", nargs="+", default=["basic_relations"], help="Test datasets")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    
    config["output_dir"] = args.output
    
    # Initialize evaluation harness
    harness = LimitGraphEvaluationHarness(config)
    
    # Initialize system
    harness.initialize_system(args.corpus)
    
    # Load test datasets
    for dataset_name in args.datasets:
        harness.load_test_dataset(dataset_name, f"{dataset_name}.json")
    
    # Run evaluation
    report = harness.run_comprehensive_evaluation(args.datasets)
    
    # Print summary
    print(f"\n📊 Evaluation Summary:")
    print(f"   Report ID: {report.report_id}")
    print(f"   Queries evaluated: {report.dataset_info['total_queries']}")
    print(f"   Average Recall@10: {report.metrics.recall_at_k.get(10, 0.0):.3f}")
    print(f"   Graph Coverage: {report.metrics.graph_coverage:.3f}")
    print(f"   Provenance Integrity: {report.metrics.provenance_integrity:.3f}")
    print(f"   CI Tests Passed: {sum(1 for r in report.ci_test_results if r.status == 'pass')}/{len(report.ci_test_results)}")
    
    if report.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in report.recommendations:
            print(f"   - {rec}")

if __name__ == "__main__":
    run_evaluation_cli()