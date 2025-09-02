# LIMIT-GRAPH Contributor Onboarding Guide

Welcome to the LIMIT-GRAPH project! This guide will help you get started contributing to our advanced retrieval system that goes beyond traditional embeddings.

## 🧩 Structure

### 1. Overview

**Purpose**: Stress-test retrieval systems beyond embeddings
- Traditional embedding-based retrieval misses indirect relationships
- LIMIT-GRAPH uses semantic graphs to discover connections that embeddings can't find
- Combines multiple retrieval strategies with reinforcement learning for optimal fusion

**Architecture**: Hybrid retriever + graph memory + RL agent
```
Query → [BM25, Dense, Graph Reasoner, Memory Agent] → Fusion → Results
```

**Key Innovation**: Memory Agent learns to optimally fuse signals from all retrievers using:
- Graph-based reasoning for indirect matches
- Provenance tracking for source lineage
- RL-based fusion strategy learning

### 2. Setup

#### Prerequisites
- Python 3.8+
- Git
- Basic knowledge of information retrieval and graph algorithms

#### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd limit-graph
```

2. **Install dependencies**
```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install networkx dash plotly spacy numpy pandas matplotlib seaborn

# Install spaCy model for entity linking
python -m spacy download en_core_web_sm
```

3. **Load LIMIT-GRAPH dataset**
```bash
# Download sample dataset
python scripts/download_limit_dataset.py

# Or create sample data for testing
python -c "
from extensions.limit_graph_core import create_sample_limit_dataset
import json
data = create_sample_limit_dataset()
with open('sample_limit_data.json', 'w') as f:
    json.dump(data, f, indent=2)
"
```

4. **Verify installation**
```bash
python extensions/demo_integrated_limit_graph.py
```

### 3. Modules

The LIMIT-GRAPH system is organized into several key modules:

#### Core Foundation
- **`limit_graph_core.py`**: Core data structures and component registry
- **`limit_graph_scaffold.py`**: Graph construction from LIMIT corpus
- **`limit_graph_benchmark.py`**: Main benchmark orchestration

#### Retrieval Components
- **`graph_reasoner.py`**: Graph traversal logic for indirect matches
  - Entity linking to extract query entities
  - Multi-strategy graph traversal (BFS, DFS, weighted, semantic)
  - Reasoning path generation and scoring

- **`limit_graph_system.py`**: Hybrid retrieval system
  - Sparse retriever (BM25)
  - Dense retriever (ColBERT-style)
  - Graph reasoner integration
  - Memory agent coordination

#### Learning and Optimization
- **`rl_reward_function.py`**: RL-based fusion learning
  - Recall-based rewards
  - Provenance scoring
  - Trace penalty calculation
  - Adaptive weight learning

#### Visualization and Analysis
- **`dashboardMemR1.py`**: Provenance + trace visualization
  - Interactive semantic graph visualization
  - Provenance chain tracking
  - Agent trace replay
  - Performance metrics dashboard

#### Integration
- **`limit_graph_evaluation_harness.py`**: Comprehensive evaluation system
- **`integration_orchestrator.py`**: System-wide integration management

### 4. Evaluation

#### Running Evaluations

**Basic evaluation**:
```bash
python eval.py --benchmark limit-graph
```

**With memory-aware features**:
```bash
python eval.py --benchmark limit-graph --model hybrid-agent --memory
```

**Full evaluation suite**:
```bash
python extensions/demo_integrated_limit_graph.py
```

#### Metrics

The system evaluates performance using multiple metrics:

**Standard Retrieval Metrics**:
- **Recall@k**: Proportion of relevant documents retrieved in top-k results
- **Precision@k**: Proportion of retrieved documents that are relevant
- **MRR**: Mean Reciprocal Rank of first relevant document
- **NDCG**: Normalized Discounted Cumulative Gain

**Extended Metrics for Agentic Memory**:
- **Graph Coverage**: % of relevant edges traversed during reasoning
- **Provenance Integrity**: % of answers with correct source lineage
- **Trace Replay Accuracy**: Ability to reconstruct memory evolution

**Example Output**:
```
📊 Evaluation Results:
   Recall@10 (fused): 0.847
   Graph Coverage: 0.623
   Provenance Integrity: 0.891
   Trace Replay Accuracy: 0.756
```

### 5. Contributing

We welcome contributions in several areas:

#### 🔗 Add New Graph Relations

**Current Relations**: likes, dislikes, owns, contains, located_in, part_of

**To add new relations**:

1. **Update relation patterns** in `graph_reasoner.py`:
```python
# In EntityLinker.__init__()
self.relation_patterns = {
    # ... existing patterns ...
    "created_by": ["created by", "authored by", "made by"],
    "influences": ["influences", "affects", "impacts"]
}
```

2. **Add semantic weights** in `GraphTraverser._semantic_traversal()`:
```python
relation_weights = {
    # ... existing weights ...
    "created_by": 0.8,
    "influences": 0.6
}
```

3. **Test your changes**:
```bash
python extensions/graph_reasoner.py
```

#### 🎯 Improve RL Reward Shaping

**Current Components**: recall, provenance, trace penalty

**To improve reward functions**:

1. **Add new reward components** in `rl_reward_function.py`:
```python
class NoveltyScorer(BaseLimitGraphComponent):
    """Rewards discovering novel information"""
    
    def calculate_novelty_reward(self, retrieved_docs, previous_retrievals):
        # Implementation here
        pass
```

2. **Integrate with main reward function**:
```python
# In RLRewardFunction.__init__()
self.novelty_scorer = NoveltyScorer(config.get("novelty", {}))

# In calculate_reward()
novelty_reward = self.novelty_scorer.calculate_novelty_reward(...)
```

3. **Test reward function**:
```bash
python extensions/rl_reward_function.py
```

#### 📊 Extend Dashboard Views

**Current Views**: Semantic Graph, Provenance Explorer, Trace Replay, Validation & CI, RL Training, System Status

**To add new dashboard views**:

1. **Add new tab** in `dashboardMemR1.py`:
```python
# In _setup_layout()
dcc.Tab(label='Your New View', value='your-tab')

# In render_tab_content()
elif active_tab == 'your-tab':
    return self._render_your_tab()
```

2. **Implement visualization**:
```python
def _render_your_tab(self) -> html.Div:
    """Render your custom visualization"""
    # Create your Plotly figures and Dash components
    return html.Div([...])
```

3. **Test dashboard**:
```bash
python extensions/dashboardMemR1.py
```

#### 🧪 Add Evaluation Metrics

**To add new evaluation metrics**:

1. **Extend EvaluationMetrics** in `limit_graph_core.py`:
```python
@dataclass
class EvaluationMetrics:
    # ... existing metrics ...
    your_new_metric: float = 0.0
```

2. **Implement calculation** in evaluation harness:
```python
def _calculate_your_metric(self, query, result):
    # Your metric calculation
    return metric_value
```

3. **Integrate with evaluation pipeline**:
```python
# In evaluate_retrieval()
your_metric = self._calculate_your_metric(query, result)
metrics.your_new_metric = your_metric
```

## 🚀 Development Workflow

### 1. Setting Up Development Environment

```bash
# Create development branch
git checkout -b feature/your-feature-name

# Install development dependencies
uv pip install -r requirements-dev.txt

# Run tests to ensure everything works
python extensions/test_limit_graph_integration.py
```

### 2. Making Changes

1. **Follow the architecture**: Use the component registry system
2. **Inherit from base classes**: All components should inherit from `BaseLimitGraphComponent`
3. **Register components**: Use `LIMIT_GRAPH_REGISTRY.register_component()`
4. **Add tests**: Create tests for your new functionality
5. **Update documentation**: Update relevant documentation files

### 3. Testing Your Changes

```bash
# Run unit tests
python extensions/test_limit_graph_integration.py

# Run integration demo
python extensions/demo_integrated_limit_graph.py

# Run specific component tests
python extensions/graph_reasoner.py  # For graph reasoner
python extensions/rl_reward_function.py  # For reward function
```

### 4. Submitting Changes

1. **Ensure tests pass**:
```bash
python extensions/test_limit_graph_integration.py
```

2. **Run evaluation to check performance**:
```bash
python eval.py --benchmark limit-graph --model hybrid-agent --memory
```

3. **Create pull request** with:
   - Clear description of changes
   - Test results
   - Performance impact analysis
   - Updated documentation

## 📚 Key Concepts

### Graph-Based Reasoning
- **Entity Linking**: Extract entities from queries using spaCy + custom rules
- **Graph Traversal**: Multiple strategies (BFS, DFS, weighted, semantic) to find relevant documents
- **Indirect Matching**: Discover documents through multi-hop reasoning paths

### Memory Agent Fusion
- **Component Weights**: Learn optimal weights for sparse, dense, graph, and memory retrievers
- **Fusion Strategies**: Weighted combination, learned fusion, adaptive strategies
- **Memory Updates**: Track and learn from retrieval history

### Provenance Tracking
- **Source Lineage**: Track where information comes from
- **Citation Chains**: Maintain chains of evidence
- **Integrity Scoring**: Measure quality of provenance information

### Reinforcement Learning
- **Reward Components**: Recall, provenance, trace penalties
- **Action Space**: Fusion weight adjustments
- **Learning**: Adapt strategies based on performance feedback

## 🔧 Debugging and Troubleshooting

### Common Issues

1. **spaCy model not found**:
```bash
python -m spacy download en_core_web_sm
```

2. **Import errors**:
```bash
# Ensure you're in the right directory
cd extensions
python -c "from limit_graph_core import *"
```

3. **Dashboard not loading**:
```bash
# Check port availability
python extensions/dashboardMemR1.py
# Navigate to http://localhost:8050
```

4. **Evaluation failures**:
```bash
# Run with debug mode
python extensions/demo_integrated_limit_graph.py
```

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling

```python
# Profile graph traversal
python -m cProfile extensions/graph_reasoner.py

# Profile reward calculation
python -m cProfile extensions/rl_reward_function.py
```

## 📖 Additional Resources

### Documentation
- **Architecture Overview**: `LIMIT_GRAPH_INTEGRATION_SUMMARY.md`
- **API Reference**: Check docstrings in each module
- **Examples**: See `demo_*.py` files

### Research Papers
- Information retrieval with graph reasoning
- Reinforcement learning for information fusion
- Provenance tracking in knowledge systems

### Community
- **Issues**: Report bugs and request features via GitHub issues
- **Discussions**: Join technical discussions
- **Code Reviews**: Participate in pull request reviews

## 🎯 Contribution Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints for function signatures
- Add comprehensive docstrings
- Include error handling

### Testing
- Write unit tests for new functionality
- Include integration tests
- Test edge cases and error conditions
- Verify performance impact

### Documentation
- Update relevant documentation files
- Add inline comments for complex logic
- Include usage examples
- Update this contributor guide if needed

### Performance
- Profile performance-critical code
- Optimize graph traversal algorithms
- Monitor memory usage
- Benchmark against baselines

---

**Welcome to the LIMIT-GRAPH community!** 🚀

Your contributions help push the boundaries of information retrieval beyond traditional embeddings. Whether you're adding new graph relations, improving RL reward shaping, or extending dashboard views, every contribution makes the system more powerful and useful.

Happy coding! 🎉