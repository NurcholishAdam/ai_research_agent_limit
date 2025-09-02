#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL Reward Function for Memory Fusion
This function guides the Memory Agent to learn optimal fusion strategies across retrievers.
Components:
- Recall: Measures retrieval accuracy against ground truth
- Provenance Score: Rewards valid source lineage and traceability
- Trace Penalty: Penalizes conflicting memory updates and inconsistencies
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import math

# Import core components
from limit_graph_core import (
    LimitQuery, RetrievalResult, BaseLimitGraphComponent,
    LIMIT_GRAPH_REGISTRY
)

@dataclass
class RewardComponents:
    """Individual components of the reward function"""
    recall_reward: float = 0.0
    provenance_reward: float = 0.0
    trace_penalty: float = 0.0
    consistency_bonus: float = 0.0
    efficiency_bonus: float = 0.0
    total_reward: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "recall_reward": self.recall_reward,
            "provenance_reward": self.provenance_reward,
            "trace_penalty": self.trace_penalty,
            "consistency_bonus": self.consistency_bonus,
            "efficiency_bonus": self.efficiency_bonus,
            "total_reward": self.total_reward
        }

@dataclass
class FusionAction:
    """Represents a fusion action taken by the Memory Agent"""
    action_id: str
    query: str
    component_weights: Dict[str, float]  # {"sparse": 0.3, "dense": 0.4, "graph": 0.3}
    retrieved_docs: List[str]
    fusion_strategy: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RewardContext:
    """Context information for reward calculation"""
    query: LimitQuery
    ground_truth_docs: List[str]
    fusion_action: FusionAction
    previous_actions: List[FusionAction]
    memory_state: Dict[str, Any]
    provenance_data: Dict[str, List[str]]
    trace_history: List[Dict[str, Any]]

class RecallCalculator(BaseLimitGraphComponent):
    """
    Calculates recall-based rewards to measure retrieval accuracy
    Supports multiple recall metrics: Recall@K, MRR, NDCG
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Recall calculation configuration
        self.k_values = config.get("k_values", [1, 5, 10])
        self.recall_weight = config.get("recall_weight", 1.0)
        self.mrr_weight = config.get("mrr_weight", 0.5)
        self.ndcg_weight = config.get("ndcg_weight", 0.3)
        
        # Register with global registry
        LIMIT_GRAPH_REGISTRY.register_component(self)
        
        print("📊 Recall Calculator initialized")
    
    def calculate_recall_reward(self, retrieved_docs: List[str], 
                               ground_truth_docs: List[str]) -> Dict[str, float]:
        """
        Calculate recall-based reward components
        
        Returns:
            Dictionary with recall@k, MRR, and NDCG scores
        """
        
        if not ground_truth_docs or not retrieved_docs:
            return {f"recall_at_{k}": 0.0 for k in self.k_values}
        
        ground_truth_set = set(ground_truth_docs)
        rewards = {}
        
        # Calculate Recall@K for different K values
        for k in self.k_values:
            retrieved_at_k = set(retrieved_docs[:k])
            recall_at_k = len(retrieved_at_k.intersection(ground_truth_set)) / len(ground_truth_set)
            rewards[f"recall_at_{k}"] = recall_at_k * self.recall_weight
        
        # Calculate Mean Reciprocal Rank (MRR)
        mrr_score = self._calculate_mrr(retrieved_docs, ground_truth_docs)
        rewards["mrr"] = mrr_score * self.mrr_weight
        
        # Calculate Normalized Discounted Cumulative Gain (NDCG)
        ndcg_score = self._calculate_ndcg(retrieved_docs, ground_truth_docs)
        rewards["ndcg"] = ndcg_score * self.ndcg_weight
        
        # Overall recall reward (weighted combination)
        overall_recall = (
            rewards.get("recall_at_10", 0.0) * 0.5 +
            rewards.get("mrr", 0.0) * 0.3 +
            rewards.get("ndcg", 0.0) * 0.2
        )
        rewards["overall_recall"] = overall_recall
        
        return rewards
    
    def _calculate_mrr(self, retrieved_docs: List[str], ground_truth_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        ground_truth_set = set(ground_truth_docs)
        
        for i, doc in enumerate(retrieved_docs):
            if doc in ground_truth_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_ndcg(self, retrieved_docs: List[str], ground_truth_docs: List[str], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        ground_truth_set = set(ground_truth_docs)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            if doc in ground_truth_set:
                dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (perfect ranking)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth_docs), k)))
        
        return dcg / idcg if idcg > 0 else 0.0

class ProvenanceScorer(BaseLimitGraphComponent):
    """
    Calculates provenance-based rewards to encourage valid source lineage
    Rewards retrievals that maintain proper citation and traceability
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Provenance scoring configuration
        self.provenance_weight = config.get("provenance_weight", 0.8)
        self.lineage_depth_bonus = config.get("lineage_depth_bonus", 0.1)
        self.source_diversity_bonus = config.get("source_diversity_bonus", 0.2)
        self.citation_accuracy_weight = config.get("citation_accuracy_weight", 0.5)
        
        # Register with global registry
        LIMIT_GRAPH_REGISTRY.register_component(self)
        
        print("🔗 Provenance Scorer initialized")
    
    def calculate_provenance_reward(self, retrieved_docs: List[str], 
                                   provenance_data: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate provenance-based reward components
        
        Args:
            retrieved_docs: List of retrieved document IDs
            provenance_data: Dictionary mapping doc_id -> list of source citations
        
        Returns:
            Dictionary with provenance reward components
        """
        
        if not retrieved_docs:
            return {"provenance_score": 0.0, "lineage_completeness": 0.0, "source_diversity": 0.0}
        
        # Calculate lineage completeness
        docs_with_provenance = sum(1 for doc in retrieved_docs if doc in provenance_data and provenance_data[doc])
        lineage_completeness = docs_with_provenance / len(retrieved_docs)
        
        # Calculate source diversity
        all_sources = set()
        for doc in retrieved_docs:
            if doc in provenance_data:
                all_sources.update(provenance_data[doc])
        
        source_diversity = len(all_sources) / len(retrieved_docs) if retrieved_docs else 0.0
        
        # Calculate citation depth (average provenance chain length)
        citation_depths = []
        for doc in retrieved_docs:
            if doc in provenance_data and provenance_data[doc]:
                citation_depths.append(len(provenance_data[doc]))
            else:
                citation_depths.append(0)
        
        avg_citation_depth = np.mean(citation_depths) if citation_depths else 0.0
        
        # Calculate overall provenance score
        provenance_score = (
            lineage_completeness * self.provenance_weight +
            source_diversity * self.source_diversity_bonus +
            min(avg_citation_depth / 3.0, 1.0) * self.lineage_depth_bonus  # Normalize depth
        )
        
        return {
            "provenance_score": provenance_score,
            "lineage_completeness": lineage_completeness,
            "source_diversity": source_diversity,
            "avg_citation_depth": avg_citation_depth
        }

class TracePenaltyCalculator(BaseLimitGraphComponent):
    """
    Calculates trace-based penalties to discourage conflicting memory updates
    Penalizes actions that create inconsistencies in the memory trace
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Trace penalty configuration
        self.conflict_penalty_weight = config.get("conflict_penalty_weight", -0.5)
        self.inconsistency_penalty_weight = config.get("inconsistency_penalty_weight", -0.3)
        self.repetition_penalty_weight = config.get("repetition_penalty_weight", -0.2)
        self.max_penalty = config.get("max_penalty", -1.0)
        
        # Register with global registry
        LIMIT_GRAPH_REGISTRY.register_component(self)
        
        print("⚠️ Trace Penalty Calculator initialized")
    
    def calculate_trace_penalty(self, current_action: FusionAction, 
                               previous_actions: List[FusionAction],
                               trace_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate trace-based penalties
        
        Args:
            current_action: Current fusion action
            previous_actions: List of previous fusion actions
            trace_history: History of memory trace entries
        
        Returns:
            Dictionary with penalty components
        """
        
        penalties = {
            "conflict_penalty": 0.0,
            "inconsistency_penalty": 0.0,
            "repetition_penalty": 0.0,
            "total_penalty": 0.0
        }
        
        if not previous_actions:
            return penalties
        
        # Calculate conflict penalty
        conflict_penalty = self._calculate_conflict_penalty(current_action, previous_actions)
        penalties["conflict_penalty"] = conflict_penalty * self.conflict_penalty_weight
        
        # Calculate inconsistency penalty
        inconsistency_penalty = self._calculate_inconsistency_penalty(current_action, trace_history)
        penalties["inconsistency_penalty"] = inconsistency_penalty * self.inconsistency_penalty_weight
        
        # Calculate repetition penalty
        repetition_penalty = self._calculate_repetition_penalty(current_action, previous_actions)
        penalties["repetition_penalty"] = repetition_penalty * self.repetition_penalty_weight
        
        # Total penalty (capped at max_penalty)
        total_penalty = sum([
            penalties["conflict_penalty"],
            penalties["inconsistency_penalty"],
            penalties["repetition_penalty"]
        ])
        penalties["total_penalty"] = max(total_penalty, self.max_penalty)
        
        return penalties
    
    def _calculate_conflict_penalty(self, current_action: FusionAction, 
                                   previous_actions: List[FusionAction]) -> float:
        """Calculate penalty for conflicting fusion strategies"""
        
        # Look for conflicting weight assignments for similar queries
        similar_queries = [
            action for action in previous_actions[-10:]  # Last 10 actions
            if self._query_similarity(current_action.query, action.query) > 0.7
        ]
        
        if not similar_queries:
            return 0.0
        
        # Calculate weight divergence
        conflicts = 0
        for prev_action in similar_queries:
            for component in current_action.component_weights:
                if component in prev_action.component_weights:
                    weight_diff = abs(
                        current_action.component_weights[component] - 
                        prev_action.component_weights[component]
                    )
                    if weight_diff > 0.3:  # Significant weight change
                        conflicts += weight_diff
        
        return min(conflicts / len(similar_queries), 1.0)
    
    def _calculate_inconsistency_penalty(self, current_action: FusionAction,
                                        trace_history: List[Dict[str, Any]]) -> float:
        """Calculate penalty for memory trace inconsistencies"""
        
        if not trace_history:
            return 0.0
        
        # Check for contradictory information in retrieved documents
        current_docs = set(current_action.retrieved_docs)
        
        inconsistencies = 0
        for trace_entry in trace_history[-5:]:  # Last 5 trace entries
            if "retrieved_docs" in trace_entry:
                prev_docs = set(trace_entry["retrieved_docs"])
                
                # Check for documents that were previously rejected but now accepted
                # This is a simplified heuristic - in practice, would need semantic analysis
                overlap = len(current_docs.intersection(prev_docs))
                total_docs = len(current_docs.union(prev_docs))
                
                if total_docs > 0:
                    consistency_ratio = overlap / total_docs
                    if consistency_ratio < 0.3:  # Low consistency
                        inconsistencies += (0.3 - consistency_ratio)
        
        return min(inconsistencies / 5, 1.0)  # Normalize by number of trace entries checked
    
    def _calculate_repetition_penalty(self, current_action: FusionAction,
                                     previous_actions: List[FusionAction]) -> float:
        """Calculate penalty for repetitive fusion strategies"""
        
        if len(previous_actions) < 3:
            return 0.0
        
        # Check for repetitive weight patterns
        recent_actions = previous_actions[-5:]  # Last 5 actions
        
        # Calculate variance in component weights
        weight_variances = {}
        for component in current_action.component_weights:
            weights = [action.component_weights.get(component, 0) for action in recent_actions]
            weights.append(current_action.component_weights[component])
            
            if len(weights) > 1:
                weight_variances[component] = np.var(weights)
        
        # Low variance indicates repetitive behavior
        avg_variance = np.mean(list(weight_variances.values())) if weight_variances else 0
        repetition_score = max(0, 0.1 - avg_variance) / 0.1  # Normalize to [0, 1]
        
        return repetition_score
    
    def _query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries (simplified)"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

class RLRewardFunction(BaseLimitGraphComponent):
    """
    Main RL Reward Function that combines all reward components
    Guides the Memory Agent to learn optimal fusion strategies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Initialize component calculators
        self.recall_calculator = RecallCalculator(config.get("recall", {}))
        self.provenance_scorer = ProvenanceScorer(config.get("provenance", {}))
        self.trace_penalty_calculator = TracePenaltyCalculator(config.get("trace_penalty", {}))
        
        # Reward function weights
        self.recall_weight = config.get("recall_weight", 0.5)
        self.provenance_weight = config.get("provenance_weight", 0.3)
        self.trace_penalty_weight = config.get("trace_penalty_weight", 0.2)
        
        # Bonus weights
        self.consistency_bonus_weight = config.get("consistency_bonus_weight", 0.1)
        self.efficiency_bonus_weight = config.get("efficiency_bonus_weight", 0.1)
        
        # Reward history for learning
        self.reward_history: List[RewardComponents] = []
        
        # Register with global registry
        LIMIT_GRAPH_REGISTRY.register_component(
            self,
            dependencies=[
                self.recall_calculator.component_id,
                self.provenance_scorer.component_id,
                self.trace_penalty_calculator.component_id
            ]
        )
        
        print("🎯 RL Reward Function initialized")
    
    def calculate_reward(self, reward_context: RewardContext) -> RewardComponents:
        """
        Calculate comprehensive reward for a fusion action
        
        Args:
            reward_context: Context containing all necessary information
        
        Returns:
            RewardComponents with detailed reward breakdown
        """
        
        # Calculate recall reward
        recall_rewards = self.recall_calculator.calculate_recall_reward(
            reward_context.fusion_action.retrieved_docs,
            reward_context.ground_truth_docs
        )
        recall_reward = recall_rewards.get("overall_recall", 0.0) * self.recall_weight
        
        # Calculate provenance reward
        provenance_rewards = self.provenance_scorer.calculate_provenance_reward(
            reward_context.fusion_action.retrieved_docs,
            reward_context.provenance_data
        )
        provenance_reward = provenance_rewards.get("provenance_score", 0.0) * self.provenance_weight
        
        # Calculate trace penalty
        trace_penalties = self.trace_penalty_calculator.calculate_trace_penalty(
            reward_context.fusion_action,
            reward_context.previous_actions,
            reward_context.trace_history
        )
        trace_penalty = trace_penalties.get("total_penalty", 0.0) * self.trace_penalty_weight
        
        # Calculate consistency bonus
        consistency_bonus = self._calculate_consistency_bonus(reward_context) * self.consistency_bonus_weight
        
        # Calculate efficiency bonus
        efficiency_bonus = self._calculate_efficiency_bonus(reward_context) * self.efficiency_bonus_weight
        
        # Total reward
        total_reward = (
            recall_reward + 
            provenance_reward + 
            trace_penalty +  # Note: penalty is negative
            consistency_bonus + 
            efficiency_bonus
        )
        
        # Create reward components
        reward_components = RewardComponents(
            recall_reward=recall_reward,
            provenance_reward=provenance_reward,
            trace_penalty=trace_penalty,
            consistency_bonus=consistency_bonus,
            efficiency_bonus=efficiency_bonus,
            total_reward=total_reward
        )
        
        # Store in history
        self.reward_history.append(reward_components)
        
        return reward_components
    
    def _calculate_consistency_bonus(self, reward_context: RewardContext) -> float:
        """Calculate bonus for consistent fusion strategies"""
        
        if len(reward_context.previous_actions) < 2:
            return 0.0
        
        # Reward consistent performance across similar queries
        current_weights = reward_context.fusion_action.component_weights
        
        # Find similar previous actions
        similar_actions = []
        for prev_action in reward_context.previous_actions[-10:]:
            if self._query_similarity(
                reward_context.fusion_action.query, 
                prev_action.query
            ) > 0.6:
                similar_actions.append(prev_action)
        
        if not similar_actions:
            return 0.0
        
        # Calculate weight consistency
        consistency_scores = []
        for prev_action in similar_actions:
            weight_similarity = 0.0
            for component in current_weights:
                if component in prev_action.component_weights:
                    weight_diff = abs(
                        current_weights[component] - 
                        prev_action.component_weights[component]
                    )
                    weight_similarity += (1.0 - weight_diff)
            
            consistency_scores.append(weight_similarity / len(current_weights))
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_efficiency_bonus(self, reward_context: RewardContext) -> float:
        """Calculate bonus for efficient retrieval (fewer components, better results)"""
        
        # Reward strategies that achieve good results with fewer active components
        active_components = sum(1 for weight in reward_context.fusion_action.component_weights.values() if weight > 0.1)
        
        # Efficiency is inversely related to number of active components
        efficiency_score = 1.0 / active_components if active_components > 0 else 0.0
        
        # But only reward if results are still good (this would need recall info)
        # For now, use a simple heuristic
        num_retrieved = len(reward_context.fusion_action.retrieved_docs)
        if num_retrieved > 0:
            efficiency_score *= min(num_retrieved / 10.0, 1.0)  # Normalize by expected retrieval count
        
        return efficiency_score
    
    def _query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward function performance"""
        
        if not self.reward_history:
            return {"message": "No reward history available"}
        
        recent_rewards = self.reward_history[-100:]  # Last 100 rewards
        
        return {
            "total_rewards_calculated": len(self.reward_history),
            "recent_statistics": {
                "avg_total_reward": np.mean([r.total_reward for r in recent_rewards]),
                "avg_recall_reward": np.mean([r.recall_reward for r in recent_rewards]),
                "avg_provenance_reward": np.mean([r.provenance_reward for r in recent_rewards]),
                "avg_trace_penalty": np.mean([r.trace_penalty for r in recent_rewards]),
                "reward_variance": np.var([r.total_reward for r in recent_rewards])
            },
            "component_weights": {
                "recall_weight": self.recall_weight,
                "provenance_weight": self.provenance_weight,
                "trace_penalty_weight": self.trace_penalty_weight
            }
        }
    
    def adapt_weights(self, performance_feedback: Dict[str, float]):
        """Adapt reward weights based on performance feedback"""
        
        # Simple adaptive mechanism - in practice, would use more sophisticated methods
        learning_rate = 0.1
        
        if "recall_performance" in performance_feedback:
            recall_perf = performance_feedback["recall_performance"]
            if recall_perf < 0.5:  # Poor recall performance
                self.recall_weight = min(1.0, self.recall_weight + learning_rate)
        
        if "provenance_performance" in performance_feedback:
            prov_perf = performance_feedback["provenance_performance"]
            if prov_perf < 0.5:  # Poor provenance performance
                self.provenance_weight = min(1.0, self.provenance_weight + learning_rate)
        
        # Renormalize weights
        total_weight = self.recall_weight + self.provenance_weight + self.trace_penalty_weight
        if total_weight > 0:
            self.recall_weight /= total_weight
            self.provenance_weight /= total_weight
            self.trace_penalty_weight /= total_weight
        
        print(f"🔄 Reward weights adapted: recall={self.recall_weight:.3f}, "
              f"provenance={self.provenance_weight:.3f}, penalty={self.trace_penalty_weight:.3f}")

# Integration functions
def create_rl_reward_function(config: Dict[str, Any] = None) -> RLRewardFunction:
    """Create and configure RL reward function"""
    return RLRewardFunction(config)

def integrate_reward_function_with_memory_agent(memory_agent, reward_function: RLRewardFunction):
    """
    Integrate RL reward function with Memory Agent for learning
    """
    
    # Add reward function to memory agent
    memory_agent.reward_function = reward_function
    
    # Store action history for reward calculation
    if not hasattr(memory_agent, 'action_history'):
        memory_agent.action_history = []
    
    # Enhance memory agent's fusion method to use rewards
    original_fuse_method = memory_agent.fuse_and_retrieve
    
    def reward_guided_fuse_and_retrieve(query: str, sparse_results: List[str], 
                                       dense_results: List[str], graph_results: List[str], 
                                       top_k: int) -> Dict[str, Any]:
        
        # Call original fusion
        fusion_result = original_fuse_method(query, sparse_results, dense_results, graph_results, top_k)
        
        # Create fusion action
        fusion_action = FusionAction(
            action_id=f"action_{len(memory_agent.action_history)}",
            query=query,
            component_weights={"sparse": 0.25, "dense": 0.35, "graph": 0.4},  # Would be learned
            retrieved_docs=fusion_result["retrieved_docs"],
            fusion_strategy="weighted_combination"
        )
        
        # Store action
        memory_agent.action_history.append(fusion_action)
        
        # Add reward information to result
        fusion_result["fusion_action"] = fusion_action
        fusion_result["reward_ready"] = True
        
        return fusion_result
    
    # Replace the method
    memory_agent.fuse_and_retrieve = reward_guided_fuse_and_retrieve
    
    print("🎯 RL Reward Function integrated with Memory Agent")

def demo_reward_function():
    """Demo function to show reward function capabilities"""
    
    print("🎯 RL Reward Function Demo")
    
    # Create reward function
    reward_function = create_rl_reward_function({
        "recall_weight": 0.5,
        "provenance_weight": 0.3,
        "trace_penalty_weight": 0.2
    })
    
    # Create sample reward context
    from limit_graph_core import LimitQuery
    
    query = LimitQuery(
        query_id="demo_q1",
        query="Who likes apples?",
        relevant_docs=["d12", "d27"],
        graph_edges=[]
    )
    
    fusion_action = FusionAction(
        action_id="demo_action_1",
        query="Who likes apples?",
        component_weights={"sparse": 0.3, "dense": 0.4, "graph": 0.3},
        retrieved_docs=["d12", "d15", "d27"],
        fusion_strategy="weighted_combination"
    )
    
    reward_context = RewardContext(
        query=query,
        ground_truth_docs=["d12", "d27"],
        fusion_action=fusion_action,
        previous_actions=[],
        memory_state={},
        provenance_data={"d12": ["source1"], "d27": ["source2"], "d15": []},
        trace_history=[]
    )
    
    # Calculate reward
    reward_components = reward_function.calculate_reward(reward_context)
    
    print(f"Reward Components:")
    for key, value in reward_components.to_dict().items():
        print(f"  {key}: {value:.3f}")
    
    return reward_components

if __name__ == "__main__":
    demo_reward_function()
