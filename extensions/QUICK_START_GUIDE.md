# LIMIT-GRAPH Quick Start Guide

## Three Key Components Implementation

### 1. Graph Reasoner Module (`graph_reasoner.py`)

**Purpose**: Traverses semantic relevance graphs to discover indirect document-query matches that embeddings miss.

**Key Features**:
- **Entity Linking**: Extracts query entities using spaCy NER + custom patterns
- **Graph Traversal**: Multiple strategies (BFS, DFS, weighted, semantic) 
- **Reasoning Paths**: Generates explainable multi-hop reasoning

**Usage**:
```python
from graph_reasoner import GraphReasoner

# Initialize with graph scaffold
reasoner = GraphReasoner(graph_scaffold, config)

# Find documents through reasoning
result = reasoner.reason_and_retrieve("Who likes apples?", top_k=10)
print(f"Found docs: {result['retrieved_docs']}")
print(f"Reasoning paths: {result['reasoning_paths']}")
```

**Integration with Memory Agent**:
```python
from graph_reasoner import integrate_graph_reasoner_with_memory_agent
integrate_graph_reasoner_with_memory_agent(memory_agent, graph_reasoner)
```

### 2. RL Reward Function (`rl_reward_function.py`)

**Purpose**: Guides Memory Agent to learn optimal fusion strategies across retrievers.

**Components**:
- **Recall Calculator**: Measures retrieval accuracy (Recall@K, MRR, NDCG)
- **Provenance Scorer**: Rewards valid source lineage and traceability  
- **Trace Penalty Calculator**: Penalizes conflicting memory updates

**Usage**:
```python
from rl_reward_function import RLRewardFunction, RewardContext

# Initialize reward function
reward_fn = RLRewardFunction({
    "recall_weight": 0.5,
    "provenance_weight": 0.3, 
    "trace_penalty_weight": 0.2
})

# Calculate reward for fusion action
reward = reward_fn.calculate_reward(reward_context)
print(f"Total reward: {reward.total_reward}")
```

**Integration with Memory Agent**:
```python
from rl_reward_function import integrate_reward_function_with_memory_agent
integrate_reward_function_with_memory_agent(memory_agent, reward_function)
```

### 3. Complete System Integration

**Architecture Flow**:
```
Query → Entity Linking → Graph Traversal → Memory Fusion (RL-guided) → Results
```

**Demo Usage**:
```python
# Run complete system demo
python extensions/demo_complete_limit_graph_system.py

# Run integration tests
python extensions/test_limit_graph_integration.py

# Launch dashboard
python extensions/dashboardMemR1.py
```

## Quick Setup

### 1. Install Dependencies
```bash
pip install networkx spacy numpy pandas matplotlib plotly dash
python -m spacy download en_core_web_sm
```

### 2. Run Components
```bash
# Test Graph Reasoner
python extensions/graph_reasoner.py

# Test RL Reward Function  
python extensions/rl_reward_function.py

# Run Complete Demo
python extensions/demo_complete_limit_graph_system.py
```

### 3. Evaluation
```bash
# Run benchmark
python eval.py --benchmark limit-graph --model hybrid-agent --memory

# View results
python extensions/dashboardMemR1.py
# Navigate to http://localhost:8050
```

## Key Benefits

### Graph Reasoner Benefits
- **Indirect Discovery**: Finds documents through multi-hop reasoning
- **Entity-Aware**: Extracts and links query entities to graph nodes
- **Explainable**: Provides reasoning paths for transparency

### RL Reward Function Benefits  
- **Adaptive Learning**: Learns optimal fusion weights over time
- **Multi-Objective**: Balances recall, provenance, and consistency
- **Trace-Aware**: Prevents conflicting memory updates

### Integrated System Benefits
- **Synergistic Performance**: Components work together for better results
- **Memory-Aware**: Learns from retrieval history
- **Provenance Tracking**: Maintains source lineage throughout

## Extension Points

### Adding New Graph Relations
```python
# In graph_reasoner.py EntityLinker
self.relation_patterns = {
    # ... existing patterns ...
    "influences": ["influences", "affects", "impacts"],
    "created_by": ["created by", "authored by"]
}
```

### Adding New Reward Components
```python
# Create new reward component
class NoveltyScorer(BaseLimitGraphComponent):
    def calculate_novelty_reward(self, retrieved_docs, previous_retrievals):
        novel_docs = set(retrieved_docs) - set(previous_retrievals)
        return len(novel_docs) / len(retrieved_docs)
```

### Adding Dashboard Views
```python
# In dashboardMemR1.py
dcc.Tab(label='📈 New View', value='new-view')

def _render_new_view_tab(self):
    return html.Div([...])  # Your visualization
```

## Performance Metrics

### Standard Metrics
- **Recall@K**: Proportion of relevant documents retrieved
- **Precision@K**: Proportion of retrieved documents that are relevant
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

### Extended Metrics
- **Graph Coverage**: % of relevant graph edges traversed
- **Provenance Integrity**: % of answers with correct source lineage
- **Trace Replay Accuracy**: Ability to reconstruct memory evolution

## Troubleshooting

### Common Issues
1. **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
2. **Import errors**: Ensure you're in the extensions directory
3. **Dashboard not loading**: Check port 8050 is available

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run components with debug
python extensions/graph_reasoner.py  # Shows entity linking details
python extensions/rl_reward_function.py  # Shows reward calculations
```

## Next Steps

1. **Read Full Guide**: See `LIMIT_GRAPH_CONTRIBUTOR_GUIDE.md` for detailed contribution guidelines
2. **Explore Components**: Check individual component files for implementation details
3. **Run Tests**: Use `test_limit_graph_integration.py` for comprehensive testing
4. **Contribute**: Add new relations, reward components, or dashboard views

---

**Quick Commands Summary**:
```bash
# Setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Test Components
python extensions/graph_reasoner.py
python extensions/rl_reward_function.py

# Run System
python extensions/demo_complete_limit_graph_system.py
python extensions/dashboardMemR1.py

# Evaluate
python eval.py --benchmark limit-graph
```

This quick start guide gets you running with the three key LIMIT-GRAPH components in minutes!