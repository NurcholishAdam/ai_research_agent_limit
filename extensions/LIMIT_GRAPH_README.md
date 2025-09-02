# LIMIT-GRAPH System

Advanced retrieval system that goes beyond traditional embeddings using graph reasoning and reinforcement learning.

## About LIMIT Dataset

This system is built upon the **LIMIT dataset** - a comprehensive benchmark for evaluating retrieval systems beyond traditional embedding-based approaches. The LIMIT dataset provides:

- **Complex Query-Document Relationships**: Multi-hop reasoning requirements that challenge standard retrieval methods
- **Graph-Structured Knowledge**: Semantic relationships between entities, documents, and concepts
- **Evaluation Framework**: Comprehensive metrics for measuring retrieval performance across different reasoning types

**Dataset Source**: [LIMIT: Less Is More for Instruction Tuning Across Evaluation Paradigms](https://arxiv.org/abs/2311.13133)

The LIMIT-GRAPH system extends this dataset by:
- Converting LIMIT's qrels into semantic graph format
- Adding graph-based reasoning capabilities
- Implementing reinforcement learning for optimal retrieval fusion
- Providing comprehensive evaluation metrics for memory-aware agents

## Three Key Components ✨

### 1. 🧠 Graph Reasoner Module
Discovers documents through multi-hop reasoning that embeddings miss.

**Quick Test**:
```bash
python extensions/graph_reasoner.py
```

### 2. 🎯 RL Reward Function  
Guides Memory Agent to learn optimal fusion strategies.

**Quick Test**:
```bash
python extensions/rl_reward_function.py
```

### 3. 📚 Contributor Guide
Complete onboarding documentation for contributors.

**Read Guide**:
```bash
cat extensions/LIMIT_GRAPH_CONTRIBUTOR_GUIDE.md
```

## Quick Start 🚀

### 1. Validate System
```bash
python extensions/validate_three_components.py
```

### 2. Run Complete Demo
```bash
python extensions/demo_complete_limit_graph_system.py
```

### 3. Launch Dashboard
```bash
python extensions/dashboardMemR1.py
# Visit http://localhost:8050
```

### 4. Run Evaluation
```bash
python eval.py --benchmark limit-graph --model hybrid-agent --memory
```

## Architecture Flow

```
LIMIT Dataset → Graph Construction → Query Processing → Reasoning & Fusion → Results
     ↓              ↓                    ↓                    ↓
  Qrels &        Semantic           Entity Linking      Memory Agent
  Relations      Graph              Graph Traversal     (RL-guided)
```

**Data Flow**:
1. **LIMIT Dataset**: Source queries and relevance judgments
2. **Graph Construction**: Convert LIMIT data into semantic graph format
3. **Query Processing**: Extract entities and plan reasoning paths
4. **Reasoning & Fusion**: Multi-hop traversal + learned fusion strategies
5. **Results**: Ranked documents with provenance and reasoning paths

## Key Benefits

- **🔍 Beyond Embeddings**: Discovers documents through multi-hop reasoning that traditional embeddings miss
- **📊 LIMIT-Based**: Built on rigorous LIMIT dataset benchmarks for comprehensive evaluation
- **🧠 Adaptive Learning**: RL-guided fusion learns optimal strategies from retrieval history
- **� Graphy-Aware**: Leverages semantic relationships for indirect document discovery
- **🔗 Memory-Aware**: Maintains provenance and learns from past retrieval patterns
- **🧪 Research-Ready**: Comprehensive evaluation framework with extended metrics
- **🚀 Production-Ready**: CI/CD integration with automated testing and validation

## Files Overview

| File | Purpose |
|------|---------|
| `graph_reasoner.py` | Graph traversal and entity linking |
| `rl_reward_function.py` | Reward calculation and learning |
| `LIMIT_GRAPH_CONTRIBUTOR_GUIDE.md` | Contributor onboarding |
| `validate_three_components.py` | System validation |
| `demo_complete_limit_graph_system.py` | Complete demo |
| `dashboardMemR1.py` | Interactive dashboard |
| `QUICK_START_GUIDE.md` | Quick setup guide |

## Performance Metrics

### Standard LIMIT Metrics
- **Recall@K**: Proportion of relevant documents retrieved in top-K results
- **Precision@K**: Proportion of retrieved documents that are relevant
- **MRR**: Mean Reciprocal Rank of first relevant document
- **NDCG**: Normalized Discounted Cumulative Gain

### Extended Metrics for Graph Reasoning
- **Graph Coverage**: % of relevant graph edges traversed during reasoning
- **Provenance Integrity**: % of answers with correct source lineage tracking
- **Trace Replay Accuracy**: Ability to reconstruct memory evolution and reasoning paths
- **Multi-hop Success Rate**: Performance on queries requiring indirect reasoning

### LIMIT Dataset Advantages
- **Complex Reasoning**: Evaluates multi-hop inference capabilities
- **Entity-Aware**: Tests entity linking and relationship understanding
- **Memory Consistency**: Measures coherent reasoning across query sessions

## Extension Points

- **Graph Relations**: Add new semantic relations
- **Reward Components**: Improve reward shaping
- **Dashboard Views**: Add new visualizations
- **Evaluation Metrics**: Add new performance measures

## Documentation

- 📖 **Quick Start**: `QUICK_START_GUIDE.md`
- 🧩 **Contributor Guide**: `LIMIT_GRAPH_CONTRIBUTOR_GUIDE.md`  
- 🏗️ **Integration Summary**: `LIMIT_GRAPH_INTEGRATION_SUMMARY.md`
- 🧪 **Validation**: `validate_three_components.py`

## Support

- **Issues**: Report bugs and request features
- **Testing**: Run `validate_three_components.py`
- **Dashboard**: Monitor system at `http://localhost:8050`
- **Evaluation**: Benchmark with `eval.py --benchmark limit-graph`

## References

- **LIMIT Dataset**: [Less Is More for Instruction Tuning Across Evaluation Paradigms](https://arxiv.org/abs/2311.13133)
- **Graph-based Retrieval**: Semantic graph reasoning for information retrieval
- **Reinforcement Learning**: Multi-objective reward functions for retrieval fusion
- **Memory-Aware Agents**: Provenance tracking and trace replay for consistent reasoning

## Citation

If you use this LIMIT-GRAPH system in your research, please cite:

```bibtex
@article{limit2023,
  title={Less Is More for Instruction Tuning Across Evaluation Paradigms},
  author={[LIMIT Dataset Authors]},
  journal={arXiv preprint arXiv:2311.13133},
  year={2023}
}
```

---

**Get started in 30 seconds**:
```bash
python extensions/validate_three_components.py && python extensions/demo_complete_limit_graph_system.py
```