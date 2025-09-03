#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entity Linker for Query Parsing
This module extracts entities from queries and links them to graph nodes for traversal.
Supports multiple entity extraction methods and graph-based disambiguation.
"""

import spacy
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import numpy as np

@dataclass
class EntityMention:
    """Represents an entity mention in text"""
    text: str
    start: int
    end: int
    entity_type: str
    confidence: float

@dataclass
class LinkedEntity:
    """Represents a linked entity with graph node information"""
    mention: EntityMention
    graph_node_id: str
    node_type: str
    linking_score: float
    disambiguation_method: str

@dataclass
class QueryEntityResult:
    """Result of entity linking for a query"""
    query: str
    entities: List[LinkedEntity]
    linking_confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class EntityExtractor:
    """Extracts entity mentions from text using multiple methods"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize spaCy model
        try:
            model_name = self.config.get("spacy_model", "en_core_web_sm")
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Entity type patterns
        self.entity_patterns = {
            "PERSON": [
                r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # First Last
                r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b"  # Title Name
            ],
            "LOCATION": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|Town|Village|County|State|Country))\b",
                r"\b(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
            ],
            "ORGANIZATION": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|Ltd|LLC|Company|University|School))\b"
            ],
            "PRODUCT": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Model|Version|Edition))\b"
            ]
        }
        
        # Question word patterns for entity type inference
        self.question_patterns = {
            "who": ["PERSON"],
            "where": ["LOCATION", "PLACE"],
            "what": ["OBJECT", "CONCEPT", "PRODUCT"],
            "which": ["OBJECT", "CONCEPT"],
            "when": ["TIME", "DATE"],
            "how": ["METHOD", "PROCESS"]
        }
        
        print("🔗 Entity Extractor initialized")
    
    def extract_entities(self, text: str) -> List[EntityMention]:
        """Extract entity mentions from text using multiple methods"""
        
        entities = []
        
        # Method 1: spaCy NER
        if self.nlp:
            spacy_entities = self._extract_spacy_entities(text)
            entities.extend(spacy_entities)
        
        # Method 2: Pattern-based extraction
        pattern_entities = self._extract_pattern_entities(text)
        entities.extend(pattern_entities)
        
        # Method 3: Question-based entity inference
        question_entities = self._extract_question_entities(text)
        entities.extend(question_entities)
        
        # Remove duplicates and merge overlapping entities
        entities = self._merge_overlapping_entities(entities)
        
        return entities
    
    def _extract_spacy_entities(self, text: str) -> List[EntityMention]:
        """Extract entities using spaCy NER"""
        
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            # Filter out very short or common entities
            if len(ent.text.strip()) < 2:
                continue
                
            # Map spaCy labels to our entity types
            entity_type = self._map_spacy_label(ent.label_)
            
            # Calculate confidence based on entity properties
            confidence = self._calculate_spacy_confidence(ent)
            
            entities.append(EntityMention(
                text=ent.text.strip(),
                start=ent.start_char,
                end=ent.end_char,
                entity_type=entity_type,
                confidence=confidence
            ))
        
        return entities
    
    def _map_spacy_label(self, spacy_label: str) -> str:
        """Map spaCy entity labels to our entity types"""
        
        label_mapping = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION", 
            "GPE": "LOCATION",  # Geopolitical entity
            "LOC": "LOCATION",
            "PRODUCT": "PRODUCT",
            "EVENT": "EVENT",
            "WORK_OF_ART": "WORK",
            "LAW": "LAW",
            "LANGUAGE": "LANGUAGE",
            "DATE": "DATE",
            "TIME": "TIME",
            "PERCENT": "PERCENT",
            "MONEY": "MONEY",
            "QUANTITY": "QUANTITY",
            "ORDINAL": "ORDINAL",
            "CARDINAL": "CARDINAL"
        }
        
        return label_mapping.get(spacy_label, "ENTITY")
    
    def _calculate_spacy_confidence(self, ent) -> float:
        """Calculate confidence score for spaCy entity"""
        
        # Base confidence from spaCy
        base_confidence = 0.8
        
        # Adjust based on entity length
        length_bonus = min(0.1, len(ent.text) / 20)
        
        # Adjust based on capitalization
        cap_bonus = 0.1 if ent.text[0].isupper() else 0
        
        # Adjust based on entity type reliability
        type_confidence = {
            "PERSON": 0.9,
            "ORG": 0.85,
            "GPE": 0.9,
            "LOC": 0.8,
            "PRODUCT": 0.7
        }
        
        type_bonus = type_confidence.get(ent.label_, 0.7) - 0.7
        
        return min(1.0, base_confidence + length_bonus + cap_bonus + type_bonus)
    
    def _extract_pattern_entities(self, text: str) -> List[EntityMention]:
        """Extract entities using regex patterns"""
        
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_text = match.group().strip()
                    
                    # Skip if too short or common words
                    if len(entity_text) < 3 or entity_text.lower() in ["the", "and", "for", "with"]:
                        continue
                    
                    entities.append(EntityMention(
                        text=entity_text,
                        start=match.start(),
                        end=match.end(),
                        entity_type=entity_type,
                        confidence=0.7  # Pattern-based confidence
                    ))
        
        return entities
    
    def _extract_question_entities(self, text: str) -> List[EntityMention]:
        """Extract entities based on question words and context"""
        
        entities = []
        text_lower = text.lower()
        
        # Look for question patterns
        for question_word, expected_types in self.question_patterns.items():
            if question_word in text_lower:
                # Extract potential entities after question words
                pattern = rf"\b{question_word}\s+(?:is|are|was|were|does|do|did)?\s*([A-Z][a-zA-Z\s]+?)(?:\?|$|,|\s+(?:and|or|but))"
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_text = match.group(1).strip()
                    
                    if len(entity_text) > 2:
                        # Use the first expected type
                        entity_type = expected_types[0] if expected_types else "UNKNOWN"
                        
                        entities.append(EntityMention(
                            text=entity_text,
                            start=match.start(1),
                            end=match.end(1),
                            entity_type=entity_type,
                            confidence=0.6  # Question-based confidence
                        ))
        
        return entities
    
    def _merge_overlapping_entities(self, entities: List[EntityMention]) -> List[EntityMention]:
        """Merge overlapping entity mentions"""
        
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start)
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # Check for overlap
            if next_entity.start <= current.end:
                # Merge entities - keep the one with higher confidence
                if next_entity.confidence > current.confidence:
                    current = EntityMention(
                        text=next_entity.text,
                        start=min(current.start, next_entity.start),
                        end=max(current.end, next_entity.end),
                        entity_type=next_entity.entity_type,
                        confidence=next_entity.confidence
                    )
                else:
                    current = EntityMention(
                        text=current.text,
                        start=min(current.start, next_entity.start),
                        end=max(current.end, next_entity.end),
                        entity_type=current.entity_type,
                        confidence=current.confidence
                    )
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged

class GraphEntityLinker:
    """Links extracted entities to graph nodes using various disambiguation methods"""
    
    def __init__(self, graph_nodes: Dict[str, Any], config: Dict[str, Any] = None):
        self.graph_nodes = graph_nodes
        self.config = config or {}
        
        # Linking configuration
        self.min_linking_score = self.config.get("min_linking_score", 0.5)
        self.max_candidates = self.config.get("max_candidates", 5)
        
        # Build entity index for fast lookup
        self.entity_index = self._build_entity_index()
        
        print("🔗 Graph Entity Linker initialized")
    
    def _build_entity_index(self) -> Dict[str, List[str]]:
        """Build index of entity text to graph node IDs"""
        
        index = defaultdict(list)
        
        for node_id, node_data in self.graph_nodes.items():
            # Index by node content
            if isinstance(node_data, dict):
                content = node_data.get("content", "")
                node_type = node_data.get("node_type", "")
            else:
                content = getattr(node_data, "content", "")
                node_type = getattr(node_data, "node_type", "")
            
            if content and node_type in ["entity", "person", "location", "organization"]:
                # Index exact match
                index[content.lower()].append(node_id)
                
                # Index individual words
                words = content.lower().split()
                for word in words:
                    if len(word) > 2:
                        index[word].append(node_id)
        
        return dict(index)
    
    def link_entities(self, entities: List[EntityMention]) -> List[LinkedEntity]:
        """Link entity mentions to graph nodes"""
        
        linked_entities = []
        
        for entity in entities:
            candidates = self._find_candidates(entity)
            
            if candidates:
                # Select best candidate
                best_candidate = max(candidates, key=lambda x: x[1])
                node_id, score, method = best_candidate
                
                if score >= self.min_linking_score:
                    # Get node type
                    node_data = self.graph_nodes.get(node_id, {})
                    if isinstance(node_data, dict):
                        node_type = node_data.get("node_type", "entity")
                    else:
                        node_type = getattr(node_data, "node_type", "entity")
                    
                    linked_entities.append(LinkedEntity(
                        mention=entity,
                        graph_node_id=node_id,
                        node_type=node_type,
                        linking_score=score,
                        disambiguation_method=method
                    ))
        
        return linked_entities
    
    def _find_candidates(self, entity: EntityMention) -> List[Tuple[str, float, str]]:
        """Find candidate graph nodes for entity mention"""
        
        candidates = []
        entity_text = entity.text.lower()
        
        # Method 1: Exact match
        if entity_text in self.entity_index:
            for node_id in self.entity_index[entity_text]:
                candidates.append((node_id, 1.0, "exact_match"))
        
        # Method 2: Partial match
        words = entity_text.split()
        for word in words:
            if word in self.entity_index:
                for node_id in self.entity_index[word]:
                    # Calculate partial match score
                    node_data = self.graph_nodes.get(node_id, {})
                    if isinstance(node_data, dict):
                        node_content = node_data.get("content", "").lower()
                    else:
                        node_content = getattr(node_data, "content", "").lower()
                    
                    score = self._calculate_similarity(entity_text, node_content)
                    if score > 0.3:  # Minimum similarity threshold
                        candidates.append((node_id, score, "partial_match"))
        
        # Remove duplicates and sort by score
        unique_candidates = {}
        for node_id, score, method in candidates:
            if node_id not in unique_candidates or score > unique_candidates[node_id][0]:
                unique_candidates[node_id] = (score, method)
        
        result = [(node_id, score, method) for node_id, (score, method) in unique_candidates.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result[:self.max_candidates]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

class QueryEntityLinker:
    """Main entity linker that combines extraction and linking for query processing"""
    
    def __init__(self, graph_nodes: Dict[str, Any], config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.entity_extractor = EntityExtractor(self.config.get("extraction", {}))
        self.graph_linker = GraphEntityLinker(graph_nodes, self.config.get("linking", {}))
        
        # Processing configuration
        self.min_confidence = self.config.get("min_confidence", 0.4)
        self.max_entities = self.config.get("max_entities", 10)
        
        print("🔗 Query Entity Linker initialized")
    
    def process_query(self, query: str) -> QueryEntityResult:
        """Process query to extract and link entities"""
        
        import time
        start_time = time.time()
        
        # Extract entity mentions
        entity_mentions = self.entity_extractor.extract_entities(query)
        
        # Link entities to graph nodes
        linked_entities = self.graph_linker.link_entities(entity_mentions)
        
        # Filter by confidence and limit count
        filtered_entities = [
            entity for entity in linked_entities
            if entity.linking_score >= self.min_confidence
        ]
        
        # Sort by linking score and limit
        filtered_entities.sort(key=lambda x: x.linking_score, reverse=True)
        filtered_entities = filtered_entities[:self.max_entities]
        
        # Calculate overall confidence
        overall_confidence = (
            np.mean([entity.linking_score for entity in filtered_entities])
            if filtered_entities else 0.0
        )
        
        processing_time = time.time() - start_time
        
        return QueryEntityResult(
            query=query,
            entities=filtered_entities,
            linking_confidence=overall_confidence,
            processing_time=processing_time,
            metadata={
                "total_mentions": len(entity_mentions),
                "linked_entities": len(linked_entities),
                "filtered_entities": len(filtered_entities)
            }
        )
    
    def get_entity_graph_nodes(self, query_result: QueryEntityResult) -> List[str]:
        """Get list of graph node IDs for linked entities"""
        
        return [entity.graph_node_id for entity in query_result.entities]

def create_entity_linker(graph_nodes: Dict[str, Any], config: Dict[str, Any] = None) -> QueryEntityLinker:
    """Create and configure entity linker"""
    return QueryEntityLinker(graph_nodes, config)

def demo_entity_linking():
    """Demo entity linking functionality"""
    
    print("🔗 Entity Linking Demo")
    
    # Mock graph nodes
    mock_graph_nodes = {
        "alice": {
            "content": "Alice",
            "node_type": "person"
        },
        "bob": {
            "content": "Bob", 
            "node_type": "person"
        },
        "apples": {
            "content": "apples",
            "node_type": "entity"
        },
        "farm": {
            "content": "farm",
            "node_type": "location"
        }
    }
    
    # Create entity linker
    linker = create_entity_linker(mock_graph_nodes)
    
    # Test queries
    test_queries = [
        "Who is Alice?",
        "What does Bob like?", 
        "Where are the apples grown?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        result = linker.process_query(query)
        
        print(f"   Entities found: {len(result.entities)}")
        print(f"   Linking confidence: {result.linking_confidence:.3f}")
        
        for entity in result.entities:
            print(f"     • '{entity.mention.text}' → {entity.graph_node_id} ({entity.linking_score:.3f})")
    
    return linker

if __name__ == "__main__":
    demo_entity_linking()
