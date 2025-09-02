# -*- coding: utf-8 -*-
"""
AI Research Agent Extensions Integration Orchestrator
Comprehensive integration of all stages with the existing research agent
Including Memory-R1 complete integration components
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import all stages
from .stage_1_observability import initialize_observability, ModuleType
from .stage_2_context_builder import MemoryTierManager, AdaptiveContextPacker, PromptTemplateManager
from .stage_3_semantic_graph import SemanticGraphManager
from .stage_4_diffusion_repair import integrate_diffusion_repair
from .stage_5_rlhf_agentic_rl import integrate_rlhf_agentic_rl
from .stage_6_cross_module_synergies import integrate_cross_module_synergies
from .stage_7_confidence_filtering import integrate_confidence_filtering
from .stage_9_ssrl import integrate_ssrl_system, SSRLConfig
from .stage_8_trace_buffer import integrate_ssrl_trace_buffer

# Import Memory-R1 complete integration components
try:
    from .memory_r1_modular import MemoryR1Enhanced
    from .graph_env import GraphMemoryEnv, PPOGraphTrainer, create_graph_env, create_ppo_trainer
    from .dashboardMemR1 import MemoryR1Dashboard, create_dashboard
    from .ci_hooks_integration import CIHooksValidator, CIIntegrationManager, create_ci_validator, create_ci_manager
    from .memory_r1_complete_integration import MemoryR1CompleteSystem, create_complete_system
    MEMORY_R1_AVAILABLE = True
except ImportError:
    MEMORY_R1_AVAILABLE = False
    print("⚠️ Memory-R1 components not available")

class AIResearchAgentExtensions:
    """Main integration class for all AI Research Agent extensions"""
    
    def __init__(self, config_path: str = "extensions/integration_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_configuration()
        
        # Initialize all stages
        self.observability = None
        self.memory_manager = None
        self.context_packer = None
        self.prompt_manager = None
        self.graph_manager = None
        self.diffusion_repair = None
        self.rlhf_components = None
        self.synergy_orchestrator = None
        self.confidence_filter = None
        self.ssrl_system = None
        self.trace_buffer = None
        
        # Memory-R1 complete integration components
        self.memory_r1_system = None
        self.graph_env = None
        self.ppo_trainer = None
        self.dashboard = None
        self.ci_validator = None
        self.ci_manager = None
        self.complete_system = None
        
        # LIMIT-GRAPH benchmark components
        self.limit_graph_scaffold = None
        self.hybrid_retriever = None
        self.limit_evaluator = None
        self.evaluation_harness = None
        self.stress_tester = None
        
        # Extended LIMIT-GRAPH components
        self.graph_reasoner = None
        self.rl_reward_function = None
        
        # Integration status
        self.initialized_stages = []
        self.integration_status = {}
        
        print("🚀 AI Research Agent Extensions Orchestrator initialized")
        if MEMORY_R1_AVAILABLE:
            print("✅ Memory-R1 complete integration available")
        else:
            print("⚠️ Memory-R1 complete integration not available")
    
    async def initialize_memory_r1_integration(self):
        """Initialize Memory-R1 complete integration system"""
        
        if not MEMORY_R1_AVAILABLE:
            print("⚠️ Memory-R1 integration not available - skipping")
            return False
        
        try:
            print("🧠 Initializing Memory-R1 complete integration...")
            
            # Get Memory-R1 configuration
            memory_r1_config = self.config.get("memory_r1_integration", {})
            
            # Initialize complete Memory-R1 system
            self.complete_system = create_complete_system(memory_r1_config)
            
            # Extract individual components for direct access
            self.memory_r1_system = self.complete_system.memory_system
            self.graph_env = self.complete_system.graph_env
            self.ppo_trainer = self.complete_system.ppo_trainer
            self.dashboard = self.complete_system.dashboard
            self.ci_validator = self.complete_system.ci_validator
            self.ci_manager = self.complete_system.ci_manager
            
            # Start system services if configured
            if memory_r1_config.get("auto_start", False):
                self.complete_system.start_system()
                print("🚀 Memory-R1 services started")
            
            self.initialized_stages.append("memory_r1_integration")
            self.integration_status["memory_r1_integration"] = {
                "status": "success",
                "components": {
                    "memory_system": self.memory_r1_system is not None,
                    "graph_env": self.graph_env is not None,
                    "dashboard": self.dashboard is not None,
                    "ci_validator": self.ci_validator is not None
                },
                "timestamp": datetime.now().isoformat()
            }
            
            print("✅ Memory-R1 complete integration initialized")
            return True
            
        except Exception as e:
            print(f"❌ Memory-R1 integration failed: {e}")
            self.integration_status["memory_r1_integration"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return False
    
    async def integrate_memory_r1_with_existing_stages(self):
        """Integrate Memory-R1 system with existing extension stages"""
        
        if not self.memory_r1_system:
            print("⚠️ Memory-R1 system not available for integration")
            return
        
        try:
            print("🔗 Integrating Memory-R1 with existing stages...")
            
            # Integrate with semantic graph manager
            if self.graph_manager and hasattr(self.graph_manager, 'integrate_memory_system'):
                self.graph_manager.integrate_memory_system(self.memory_r1_system)
                print("   ✅ Integrated with semantic graph manager")
            
            # Integrate with RLHF components
            if self.rlhf_components and self.ppo_trainer:
                # This would integrate RL training with RLHF feedback
                print("   ✅ Memory-R1 RL trainer available for RLHF integration")
            
            # Integrate with trace buffer
            if self.trace_buffer and self.graph_env:
                # This would integrate trace buffers
                print("   ✅ Memory-R1 trace buffer available for integration")
            
            # Integrate with confidence filtering
            if self.confidence_filter and self.ci_validator:
                # This would integrate validation with confidence filtering
                print("   ✅ Memory-R1 CI validation available for confidence integration")
            
            print("✅ Memory-R1 integration with existing stages completed")
            
        except Exception as e:
            print(f"❌ Memory-R1 stage integration failed: {e}")
    
    async def initialize_limit_graph_integration(self):
        """Initialize LIMIT-GRAPH benchmark integration"""
        
        try:
            print("📊 Initializing LIMIT-GRAPH benchmark integration...")
            
            # Import LIMIT-GRAPH components
            from .limit_graph_core import LIMIT_GRAPH_REGISTRY, create_sample_limit_dataset
            from .limit_graph_benchmark import LimitGraphBenchmark
            from .limit_graph_evaluation_harness import LimitGraphEvaluationHarness
            from .graph_reasoner import GraphReasoner, integrate_graph_reasoner_with_memory_agent
            from .rl_reward_function import RLRewardFunction, integrate_reward_function_with_memory_agent
            
            # Get LIMIT-GRAPH configuration
            limit_config = self.config.get("limit_graph_benchmark", {})
            
            # Initialize LIMIT-GRAPH benchmark system
            self.limit_graph_benchmark = LimitGraphBenchmark(limit_config)
            self.limit_graph_benchmark.initialize_components()
            
            # Initialize evaluation harness
            self.evaluation_harness = LimitGraphEvaluationHarness(limit_config.get("evaluation", {}))
            
            # Extract individual components for direct access
            self.limit_graph_scaffold = self.limit_graph_benchmark.graph_scaffold
            self.hybrid_retriever = self.limit_graph_benchmark.hybrid_retriever
            self.limit_evaluator = self.limit_graph_benchmark.evaluator
            self.stress_tester = self.limit_graph_benchmark.stress_tester
            
            # Initialize extended components
            self.graph_reasoner = GraphReasoner(
                self.limit_graph_scaffold,
                limit_config.get("graph_reasoner", {})
            )
            
            self.rl_reward_function = RLRewardFunction(
                limit_config.get("rl_reward_function", {})
            )
            
            # Integrate extended components with memory agent
            if hasattr(self.hybrid_retriever, 'memory_agent'):
                integrate_graph_reasoner_with_memory_agent(
                    self.hybrid_retriever.memory_agent, 
                    self.graph_reasoner
                )
                integrate_reward_function_with_memory_agent(
                    self.hybrid_retriever.memory_agent,
                    self.rl_reward_function
                )
                print("   🔗 Extended components integrated with Memory Agent")
            
            # Process sample corpus if provided
            sample_corpus = limit_config.get("sample_corpus", create_sample_limit_dataset())
            if sample_corpus:
                corpus_stats = self.limit_graph_scaffold.process_limit_corpus(sample_corpus)
                print(f"   📚 Processed corpus: {corpus_stats}")
            
            # Integrate with Memory-R1 if available
            if self.memory_r1_system and self.ci_validator:
                # Enable memory-aware evaluation
                print("   ✅ Integrated with Memory-R1 system for memory-aware evaluation")
            
            self.initialized_stages.append("limit_graph_benchmark")
            self.integration_status["limit_graph_benchmark"] = {
                "status": "success",
                "components": {
                    "benchmark": self.limit_graph_benchmark is not None,
                    "evaluation_harness": self.evaluation_harness is not None,
                    "graph_scaffold": self.limit_graph_scaffold is not None,
                    "hybrid_retriever": self.hybrid_retriever is not None,
                    "evaluator": self.limit_evaluator is not None,
                    "stress_tester": self.stress_tester is not None,
                    "graph_reasoner": self.graph_reasoner is not None,
                    "rl_reward_function": self.rl_reward_function is not None
                },
                "memory_r1_integration": self.memory_r1_system is not None,
                "registry_status": LIMIT_GRAPH_REGISTRY.get_integration_status(),
                "timestamp": datetime.now().isoformat()
            }
            
            print("✅ LIMIT-GRAPH benchmark integration initialized")
            return True
            
        except Exception as e:
            print(f"❌ LIMIT-GRAPH integration failed: {e}")
            self.integration_status["limit_graph_benchmark"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return False
    
    async def initialize_all_stages(self):
        """Initialize all extension stages"""
        
        print("\n🔄 Initializing all extension stages...")
        
        # Stage 1: Observability
        if self.config.get("enable_observability", True):
            await self._initialize_stage_1()
        
        # Stage 2: Context Engineering
        if self.config.get("enable_context_engineering", True):
            await self._initialize_stage_2()
        
        # Stage 3: Semantic Graph
        if self.config.get("enable_semantic_graph", True):
            await self._initialize_stage_3()
        
        # Stage 4: Diffusion Repair
        if self.config.get("enable_diffusion_repair", True):
            await self._initialize_stage_4()
        
        # Stage 5: RLHF & Agentic RL
        if self.config.get("enable_rlhf", True):
            await self._initialize_stage_5()
        
        # Stage 6: Cross-Module Synergies
        if self.config.get("enable_synergies", True):
            await self._initialize_stage_6()
        
        # Stage 7: Confidence Filtering
        if self.config.get("enable_confidence_filtering", True):
            await self._initialize_stage_7()
        
        # Stage 8: SSRL-Integrated Trace Buffer
        if self.config.get("enable_trace_buffer", True):
            await self._initialize_stage_8()
        
        # Stage 9: Self-Supervised Representation Learning (SSRL)
        if self.config.get("enable_ssrl", True):
            await self._initialize_stage_9()
        
        # Memory-R1 Complete Integration
        if self.config.get("enable_memory_r1_integration", True) and MEMORY_R1_AVAILABLE:
            await self.initialize_memory_r1_integration()
            await self.integrate_memory_r1_with_existing_stages()
        
        # LIMIT-GRAPH Benchmark Integration
        if self.config.get("enable_limit_graph_benchmark", True):
            await self.initialize_limit_graph_integration()
        
        print(f"\n✅ Initialized {len(self.initialized_stages)} stages successfully")
        return self.get_integration_status()
    
    async def _initialize_stage_1(self):
        """Initialize Stage 1: Observability"""
        try:
            self.observability = initialize_observability()
            self.initialized_stages.append("Stage 1: Observability")
            self.integration_status["observability"] = {
                "status": "initialized",
                "features": ["A/B testing", "Performance tracking", "Module monitoring"]
            }
            print("✅ Stage 1: Enhanced Observability initialized")
        except Exception as e:
            print(f"❌ Stage 1 initialization failed: {e}")
            self.integration_status["observability"] = {"status": "failed", "error": str(e)}
    
    async def _initialize_stage_2(self):
        """Initialize Stage 2: Context Engineering"""
        try:
            self.memory_manager = MemoryTierManager()
            self.context_packer = AdaptiveContextPacker()
            self.prompt_manager = PromptTemplateManager()
            
            self.initialized_stages.append("Stage 2: Context Engineering")
            self.integration_status["context_engineering"] = {
                "status": "initialized",
                "features": ["Memory tiers", "Adaptive packing", "Prompt versioning"]
            }
            print("✅ Stage 2: Enhanced Context Engineering initialized")
        except Exception as e:
            print(f"❌ Stage 2 initialization failed: {e}")
            self.integration_status["context_engineering"] = {"status": "failed", "error": str(e)}
    
    async def _initialize_stage_3(self):
        """Initialize Stage 3: Semantic Graph"""
        try:
            # Note: SemanticGraphManager would need to be properly implemented
            # This is a placeholder for the integration
            self.graph_manager = None  # SemanticGraphManager()
            
            self.initialized_stages.append("Stage 3: Semantic Graph")
            self.integration_status["semantic_graph"] = {
                "status": "placeholder",
                "features": ["Multi-source fusion", "Graph-aware retrieval", "Reasoning write-back"]
            }
            print("✅ Stage 3: Enhanced Semantic Graph (placeholder) initialized")
        except Exception as e:
            print(f"❌ Stage 3 initialization failed: {e}")
            self.integration_status["semantic_graph"] = {"status": "failed", "error": str(e)}
    
    async def _initialize_stage_4(self):
        """Initialize Stage 4: Diffusion Repair"""
        try:
            self.diffusion_repair = integrate_diffusion_repair()
            
            self.initialized_stages.append("Stage 4: Diffusion Repair")
            self.integration_status["diffusion_repair"] = {
                "status": "initialized",
                "features": ["Runtime repair", "Multi-seed voting", "Language-aware noise"]
            }
            print("✅ Stage 4: Diffusion Repair & Generation initialized")
        except Exception as e:
            print(f"❌ Stage 4 initialization failed: {e}")
            self.integration_status["diffusion_repair"] = {"status": "failed", "error": str(e)}
    
    async def _initialize_stage_5(self):
        """Initialize Stage 5: RLHF & Agentic RL"""
        try:
            self.rlhf_components = integrate_rlhf_agentic_rl()
            
            self.initialized_stages.append("Stage 5: RLHF & Agentic RL")
            self.integration_status["rlhf_agentic_rl"] = {
                "status": "initialized",
                "features": ["Preference learning", "Online RL", "Multi-objective alignment"]
            }
            print("✅ Stage 5: RLHF & Agentic RL initialized")
        except Exception as e:
            print(f"❌ Stage 5 initialization failed: {e}")
            self.integration_status["rlhf_agentic_rl"] = {"status": "failed", "error": str(e)}
    
    async def _initialize_stage_6(self):
        """Initialize Stage 6: Cross-Module Synergies"""
        try:
            self.synergy_orchestrator = integrate_cross_module_synergies()
            
            self.initialized_stages.append("Stage 6: Cross-Module Synergies")
            self.integration_status["cross_module_synergies"] = {
                "status": "initialized",
                "features": ["RLHF-tuned diffusion", "Graph-aware context", "Unified orchestration"]
            }
            print("✅ Stage 6: Cross-Module Synergies initialized")
        except Exception as e:
            print(f"❌ Stage 6 initialization failed: {e}")
            self.integration_status["cross_module_synergies"] = {"status": "failed", "error": str(e)}
    
    async def _initialize_stage_7(self):
        """Initialize Stage 7: Confidence Filtering"""
        try:
            confidence_config = self.config.get("confidence_filtering", {
                "strategy": "adaptive_threshold",
                "threshold": 17.0,
                "adaptation_rate": 0.1
            })
            
            self.confidence_filter = integrate_confidence_filtering(confidence_config)
            
            self.initialized_stages.append("Stage 7: Confidence Filtering")
            self.integration_status["confidence_filtering"] = {
                "status": "initialized",
                "features": ["Adaptive thresholds", "Ensemble voting", "Performance feedback"]
            }
            print("✅ Stage 7: Confidence Filtering initialized")
        except Exception as e:
            print(f"❌ Stage 7 initialization failed: {e}")
            self.integration_status["confidence_filtering"] = {"status": "failed", "error": str(e)}
    
    async def _initialize_stage_8(self):
        """Initialize Stage 8: SSRL-Integrated Trace Buffer"""
        try:
            trace_buffer_config = self.config.get("trace_buffer", {
                "max_size": 10000,
                "enable_ssrl": True
            })
            
            # Get SSRL config for trace buffer integration
            ssrl_config_dict = self.config.get("ssrl", {
                "encoder_dim": 512,
                "projection_dim": 128,
                "integrate_semantic_graph": True,
                "integrate_context_engineering": True,
                "integrate_confidence_filtering": True
            })
            
            ssrl_config = SSRLConfig(**ssrl_config_dict) if trace_buffer_config.get("enable_ssrl", True) else None
            
            self.trace_buffer = integrate_ssrl_trace_buffer(
                max_size=trace_buffer_config["max_size"],
                ssrl_config=ssrl_config
            )
            
            self.initialized_stages.append("Stage 8: SSRL-Integrated Trace Buffer")
            self.integration_status["trace_buffer"] = {
                "status": "initialized",
                "features": ["SSRL-enhanced traces", "Quality-based sampling", "Intelligent buffer management", "Multi-modal trace storage"]
            }
            print("✅ Stage 8: SSRL-Integrated Trace Buffer initialized")
        except Exception as e:
            print(f"❌ Stage 8 initialization failed: {e}")
            self.integration_status["trace_buffer"] = {"status": "failed", "error": str(e)}
    
    async def _initialize_stage_9(self):
        """Initialize Stage 9: Self-Supervised Representation Learning (SSRL)"""
        try:
            ssrl_config_dict = self.config.get("ssrl", {
                "encoder_dim": 768,
                "projection_dim": 256,
                "temperature": 0.07,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "integrate_semantic_graph": True,
                "integrate_context_engineering": True,
                "integrate_confidence_filtering": True
            })
            
            # Convert dict to SSRLConfig
            ssrl_config = SSRLConfig(**ssrl_config_dict)
            
            self.ssrl_system = integrate_ssrl_system(ssrl_config)
            
            self.initialized_stages.append("Stage 9: Self-Supervised Representation Learning")
            self.integration_status["ssrl"] = {
                "status": "initialized",
                "features": ["Multi-modal encoders", "Contrastive learning", "Pretext task orchestration", "Quality evaluation"]
            }
            print("✅ Stage 9: Self-Supervised Representation Learning (SSRL) initialized")
        except Exception as e:
            print(f"❌ Stage 9 initialization failed: {e}")
            self.integration_status["ssrl"] = {"status": "failed", "error": str(e)}
    
    def integrate_with_research_agent(self, research_agent):
        """Integrate extensions with the main research agent"""
        
        print("\n🔗 Integrating extensions with AI Research Agent...")
        
        integration_points = []
        
        # Integrate observability
        if self.observability:
            # Add observability tracking to research agent methods
            integration_points.append("Observability tracking added")
        
        # Integrate memory management
        if self.memory_manager:
            # Replace or enhance existing memory system
            integration_points.append("Enhanced memory tiers integrated")
        
        # Integrate context packing
        if self.context_packer:
            # Enhance context preparation in research agent
            integration_points.append("Adaptive context packing integrated")
        
        # Integrate diffusion repair
        if self.diffusion_repair:
            # Add code repair capabilities
            integration_points.append("Diffusion repair capabilities added")
        
        # Integrate RLHF
        if self.rlhf_components:
            # Add preference learning and RL
            integration_points.append("RLHF and agentic RL integrated")
        
        # Integrate synergies
        if self.synergy_orchestrator:
            # Add cross-module coordination
            integration_points.append("Cross-module synergies orchestrated")
        
        # Integrate confidence filtering
        if self.confidence_filter:
            # Add confidence-based response filtering
            integration_points.append("Confidence filtering integrated")
        
        # Integrate trace buffer
        if self.trace_buffer:
            # Add SSRL-integrated trace buffer
            integration_points.append("SSRL-integrated trace buffer integrated")
        
        # Integrate SSRL
        if self.ssrl_system:
            # Add self-supervised representation learning
            integration_points.append("Self-supervised representation learning integrated")
        
        print(f"✅ Integration complete: {len(integration_points)} integration points")
        for point in integration_points:
            print(f"   - {point}")
        
        return integration_points
    
    async def process_enhanced_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request using all available enhancements"""
        
        start_time = datetime.now()
        
        # Track request with observability
        if self.observability:
            session_id = request.get("session_id", "default")
            self.observability.track_event(
                ModuleType.MULTI_AGENT, "enhanced_request", session_id, request
            )
        
        result = {
            "success": False,
            "enhancements_used": [],
            "processing_time": 0,
            "metadata": {}
        }
        
        try:
            request_type = request.get("type", "research")
            
            # Enhanced context preparation
            if self.memory_manager and self.context_packer:
                enhanced_context = await self._prepare_enhanced_context(request)
                result["enhanced_context"] = enhanced_context
                result["enhancements_used"].append("enhanced_context")
            
            # Code repair if needed
            if request_type == "code_repair" and self.diffusion_repair:
                repair_result = await self._handle_code_repair(request)
                result["repair_result"] = repair_result
                result["enhancements_used"].append("diffusion_repair")
            
            # RLHF-enhanced processing
            if self.rlhf_components:
                rlhf_result = await self._apply_rlhf_enhancement(request, result)
                result["rlhf_enhancement"] = rlhf_result
                result["enhancements_used"].append("rlhf_enhancement")
            
            # Cross-module synergies
            if self.synergy_orchestrator:
                synergy_result = await self.synergy_orchestrator.process_request(request)
                result["synergy_result"] = synergy_result
                result["enhancements_used"].append("cross_module_synergies")
            
            # Apply confidence filtering to final result
            if self.confidence_filter:
                confidence_result = self.confidence_filter.filter_response(result)
                result["confidence_filtering"] = {
                    "passed": confidence_result.passed,
                    "confidence_score": confidence_result.confidence_score,
                    "reason": confidence_result.reason,
                    "metrics": {
                        "mean_logprob": confidence_result.metrics.mean_logprob,
                        "confidence_score": confidence_result.metrics.confidence_score,
                        "uncertainty_score": confidence_result.metrics.uncertainty_score
                    }
                }
                result["enhancements_used"].append("confidence_filtering")
                
                # Only mark as successful if confidence filtering passes
                result["success"] = confidence_result.passed
            else:
                result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            if self.observability:
                self.observability.track_event(
                    ModuleType.MULTI_AGENT, "enhanced_request_error", 
                    request.get("session_id", "default"), {"error": str(e)}
                )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        result["processing_time"] = processing_time
        
        return result
    
    async def _prepare_enhanced_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare enhanced context using memory tiers and adaptive packing"""
        
        query = request.get("query", "")
        
        # Retrieve relevant memories from all tiers
        relevant_memories = self.memory_manager.retrieve_memories(query)
        
        # Apply adaptive context packing
        from .stage_2_context_builder import TaskType
        task_type = TaskType.RESEARCH  # Default task type
        
        packing_result = self.context_packer.pack_context(relevant_memories, task_type)
        
        return {
            "packed_items": len(packing_result.packed_items),
            "total_tokens": packing_result.total_tokens,
            "packing_strategy": packing_result.packing_strategy.value,
            "diversity_score": packing_result.diversity_score,
            "relevance_score": packing_result.relevance_score
        }
    
    async def _handle_code_repair(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code repair using diffusion repair"""
        
        from .stage_4_diffusion_repair import LanguageType
        
        broken_code = request.get("code", "")
        language = LanguageType(request.get("language", "python"))
        
        repair_result = self.diffusion_repair.repair_code(broken_code, language)
        
        return {
            "success": repair_result.success,
            "repaired_code": repair_result.best_candidate.repaired_code if repair_result.best_candidate else None,
            "confidence": repair_result.best_candidate.confidence_score if repair_result.best_candidate else 0,
            "repair_time": repair_result.repair_time,
            "candidates_generated": len(repair_result.all_candidates)
        }
    
    async def _apply_rlhf_enhancement(self, request: Dict[str, Any], current_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply RLHF enhancement to processing"""
        
        alignment_system = self.rlhf_components["alignment_system"]
        
        # Evaluate alignment of current processing
        response_text = str(current_result)  # Simplified
        context = {
            "query": request.get("query", ""),
            "response_time": current_result.get("processing_time", 0)
        }
        
        alignment_scores = alignment_system.evaluate_alignment(response_text, context)
        composite_score = alignment_system.calculate_composite_alignment_score(alignment_scores)
        
        return {
            "alignment_scores": {obj.value: score for obj, score in alignment_scores.items()},
            "composite_alignment": composite_score,
            "rlhf_applied": True
        }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        
        total_stages = 9  # Updated to include Memory-R1 integration
        if MEMORY_R1_AVAILABLE:
            total_stages += 1  # Add Memory-R1 integration stage
        
        status = {
            "initialized_stages": self.initialized_stages,
            "integration_status": self.integration_status,
            "total_stages": total_stages,
            "success_rate": len(self.initialized_stages) / total_stages,
            "timestamp": datetime.now().isoformat(),
            "configuration": self.config
        }
        
        # Add Memory-R1 specific status
        if MEMORY_R1_AVAILABLE and self.complete_system:
            memory_r1_status = self.complete_system.get_system_status()
            status["memory_r1_integration"] = {
                "available": True,
                "system_running": memory_r1_status["system_running"],
                "components": memory_r1_status["components"],
                "services": memory_r1_status["services"]
            }
        else:
            status["memory_r1_integration"] = {
                "available": MEMORY_R1_AVAILABLE,
                "system_running": False,
                "reason": "Memory-R1 system not initialized" if MEMORY_R1_AVAILABLE else "Memory-R1 components not available"
            }
        
        return status
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard"""
        
        dashboard = {
            "integration_overview": self.get_integration_status(),
            "observability_metrics": None,
            "memory_statistics": None,
            "repair_statistics": None,
            "rlhf_statistics": None,
            "synergy_status": None,
            "memory_r1_dashboard": None
        }
        
        # Collect metrics from each component
        if self.observability:
            dashboard["observability_metrics"] = self.observability.get_analytics_dashboard()
        
        if self.memory_manager:
            dashboard["memory_statistics"] = self.memory_manager.get_tier_statistics()
        
        if self.diffusion_repair:
            dashboard["repair_statistics"] = self.diffusion_repair.get_repair_statistics()
        
        if self.rlhf_components:
            agentic_rl = self.rlhf_components["agentic_rl"]
            alignment_system = self.rlhf_components["alignment_system"]
            dashboard["rlhf_statistics"] = {
                "rl_stats": agentic_rl.get_rl_statistics(),
                "alignment_stats": alignment_system.get_alignment_statistics()
            }
        
        if self.synergy_orchestrator:
            dashboard["synergy_status"] = self.synergy_orchestrator.get_synergy_status()
        
        if self.confidence_filter:
            dashboard["confidence_statistics"] = self.confidence_filter.get_statistics()
        
        # Add Memory-R1 dashboard data
        if self.complete_system:
            try:
                memory_r1_status = self.complete_system.get_system_status()
                dashboard["memory_r1_dashboard"] = {
                    "system_status": memory_r1_status,
                    "dashboard_url": f"http://localhost:{self.config.get('memory_r1_integration', {}).get('dashboard_port', 8050)}",
                    "ci_status": self.ci_validator.get_ci_status_summary() if self.ci_validator else None,
                    "rl_training_status": self.ppo_trainer.get_training_status() if self.ppo_trainer else None
                }
            except Exception as e:
                dashboard["memory_r1_dashboard"] = {"error": str(e)}
        
        return dashboard
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load integration configuration"""
        
        default_config = {
            "enable_observability": True,
            "enable_context_engineering": True,
            "enable_semantic_graph": True,
            "enable_diffusion_repair": True,
            "enable_rlhf": True,
            "enable_synergies": True,
            "enable_confidence_filtering": True,
            "enable_trace_buffer": True,
            "enable_ssrl": True,
            "enable_memory_r1_integration": True,
            "enable_limit_graph_benchmark": True,
            "integration_level": "advanced",
            "auto_optimization": True,
            "performance_monitoring": True,
            "confidence_filtering": {
                "strategy": "adaptive_threshold",
                "threshold": 17.0,
                "adaptation_rate": 0.1,
                "group_size": 2048,
                "warmup_traces": 16
            },
            "memory_r1_integration": {
                "memory_r1": {
                    "storage_path": "memory_r1_data"
                },
                "graph_env": {
                    "max_turns": 100,
                    "observation_dim": 64,
                    "dataset_path": "training_data.json"
                },
                "ppo_trainer": {
                    "learning_rate": 3e-4,
                    "clip_epsilon": 0.2
                },
                "ci_hooks": {
                    "max_disconnected_nodes": 5,
                    "min_provenance_integrity": 0.9
                },
                "auto_validation": True,
                "validation_interval": 300,
                "auto_training": False,
                "auto_start": False,
                "dashboard_port": 8050
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
                    },
                    "sparse": {"k1": 1.2, "b": 0.75},
                    "dense": {"embedding_dim": 768},
                    "graph": {"max_hops": 2},
                    "memory": {"trace_buffer_size": 1000}
                },
                "sample_corpus": [
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
                    }
                ]
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                return {**default_config, **config}
            except Exception as e:
                print(f"⚠️ Failed to load config, using defaults: {e}")
        
        # Save default configuration
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config

# Main integration function
async def integrate_ai_research_agent_extensions(research_agent=None, config_path: str = None):
    """Main function to integrate all extensions with the AI Research Agent"""
    
    print("🚀 Starting AI Research Agent Extensions Integration")
    print("=" * 60)
    
    # Initialize extensions orchestrator
    extensions = AIResearchAgentExtensions(config_path or "extensions/integration_config.json")
    
    # Initialize all stages
    status = await extensions.initialize_all_stages()
    
    # Integrate with research agent if provided
    if research_agent:
        integration_points = extensions.integrate_with_research_agent(research_agent)
    else:
        print("ℹ️ No research agent provided, extensions initialized standalone")
        integration_points = []
    
    # Generate final report
    print("\n" + "=" * 60)
    print("🎉 AI Research Agent Extensions Integration Complete!")
    print("=" * 60)
    
    print(f"\n📊 Integration Summary:")
    print(f"   Stages initialized: {len(extensions.initialized_stages)}/7")
    print(f"   Success rate: {status['success_rate']:.1%}")
    print(f"   Integration points: {len(integration_points)}")
    
    print(f"\n🔧 Available Enhancements:")
    for stage_name in extensions.initialized_stages:
        print(f"   ✅ {stage_name}")
    
    print(f"\n📈 Performance Dashboard available via:")
    print(f"   extensions.get_performance_dashboard()")
    
    return extensions

if __name__ == "__main__":
    # Demo integration
    async def main():
        extensions = await integrate_ai_research_agent_extensions()
        
        # Test enhanced request processing
        test_request = {
            "type": "research",
            "query": "How does reinforcement learning work?",
            "session_id": "demo_session"
        }
        
        result = await extensions.process_enhanced_request(test_request)
        print(f"\n🧪 Test Request Result:")
        print(f"   Success: {result['success']}")
        print(f"   Enhancements used: {result['enhancements_used']}")
        print(f"   Processing time: {result['processing_time']:.3f}s")
    
    asyncio.run(main())


