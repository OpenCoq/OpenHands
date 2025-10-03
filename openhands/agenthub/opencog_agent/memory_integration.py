"""
Memory integration for OpenCoq Agent.

This module provides integration between OpenCoq AtomSpace and OpenHands memory system.
"""

from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from openhands.memory.memory import Memory
from openhands.events.action import RecallAction
from openhands.events.observation import RecallObservation
from openhands.events.event import Event

from .atomspace import AtomSpace, AtomType, Atom, TruthValue
from .reasoning import CognitiveReasoner


class OpenCogMemoryIntegration:
    """
    Integration layer between OpenCoq AtomSpace and OpenHands Memory system.
    
    This class allows the OpenCoq agent to store and retrieve knowledge
    from both the AtomSpace and the standard OpenHands memory system.
    """
    
    def __init__(
        self, 
        atomspace: AtomSpace, 
        cognitive_reasoner: CognitiveReasoner,
        memory: Optional[Memory] = None
    ):
        self.atomspace = atomspace
        self.cognitive_reasoner = cognitive_reasoner
        self.memory = memory
        
        # Cache for frequently accessed knowledge
        self.knowledge_cache: Dict[str, Any] = {}
        
        # Integration settings
        self.auto_sync = True
        self.sync_threshold = 0.7  # Truth value threshold for syncing to memory
    
    def store_knowledge(
        self, 
        concept: str, 
        description: str, 
        truth_value: Optional[TruthValue] = None,
        concept_type: AtomType = AtomType.CONCEPT_NODE
    ) -> Atom:
        """Store knowledge in the AtomSpace."""
        
        if truth_value is None:
            truth_value = TruthValue(0.8, 0.7)
        
        # Create concept atom
        concept_atom = self.atomspace.add_atom(
            concept_type,
            concept,
            truth_value=truth_value
        )
        
        # Create description atom and link
        if description:
            desc_atom = self.atomspace.add_atom(
                AtomType.CONCEPT_NODE,
                f"Description:{description[:100]}",  # Truncate long descriptions
                truth_value=TruthValue(0.9, 0.8)
            )
            
            # Create evaluation link
            predicate = self.atomspace.add_atom(
                AtomType.PREDICATE_NODE,
                "describedBy"
            )
            
            self.atomspace.add_atom(
                AtomType.EVALUATION_LINK,
                outgoing=[
                    predicate,
                    self.atomspace.add_atom(
                        AtomType.LIST_LINK,
                        outgoing=[concept_atom, desc_atom]
                    )
                ],
                truth_value=truth_value
            )
        
        # Auto-sync to memory if enabled and high confidence
        if (self.auto_sync and 
            self.memory and 
            truth_value.confidence >= self.sync_threshold):
            self._sync_to_memory(concept_atom, description)
        
        # Update cache
        self.knowledge_cache[concept] = {
            "atom": concept_atom,
            "description": description,
            "truth_value": truth_value.to_dict()
        }
        
        return concept_atom
    
    def retrieve_knowledge(self, concept: str) -> Optional[Dict[str, Any]]:
        """Retrieve knowledge about a concept."""
        
        # Check cache first
        if concept in self.knowledge_cache:
            return self.knowledge_cache[concept]
        
        # Search AtomSpace
        concept_atoms = self.atomspace.get_atoms_by_name(concept)
        if concept_atoms:
            atom = concept_atoms[0]
            
            # Find associated description
            description = self._find_description(atom)
            
            result = {
                "atom": atom,
                "concept": concept,
                "description": description,
                "truth_value": atom.truth_value.to_dict(),
                "attention_value": atom.attention_value.to_dict(),
                "related_concepts": self._find_related_concepts(atom)
            }
            
            # Cache result
            self.knowledge_cache[concept] = result
            return result
        
        # Fallback to memory system
        if self.memory:
            return self._retrieve_from_memory(concept)
        
        return None
    
    def _find_description(self, atom: Atom) -> Optional[str]:
        """Find description for an atom."""
        
        for incoming in atom.incoming:
            if (incoming.atom_type == AtomType.EVALUATION_LINK and
                len(incoming.outgoing) == 2):
                
                predicate, list_link = incoming.outgoing
                if (predicate.name == "describedBy" and
                    list_link.atom_type == AtomType.LIST_LINK and
                    len(list_link.outgoing) == 2):
                    
                    concept_ref, desc_atom = list_link.outgoing
                    if concept_ref == atom and desc_atom.name.startswith("Description:"):
                        return desc_atom.name[12:]  # Remove "Description:" prefix
        
        return None
    
    def _find_related_concepts(self, atom: Atom, limit: int = 5) -> List[str]:
        """Find concepts related to the given atom."""
        
        related = []
        
        # Find through inheritance links
        for incoming in atom.incoming:
            if incoming.atom_type == AtomType.INHERITANCE_LINK:
                for outgoing in incoming.outgoing:
                    if outgoing != atom and outgoing.name:
                        related.append(outgoing.name)
        
        # Find through similarity links
        for incoming in atom.incoming:
            if incoming.atom_type == AtomType.SIMILARITY_LINK:
                for outgoing in incoming.outgoing:
                    if outgoing != atom and outgoing.name:
                        related.append(outgoing.name)
        
        # Use cognitive reasoner for analogies
        analogies = self.cognitive_reasoner.find_analogies(atom, limit=limit)
        for analog_atom, _ in analogies:
            if analog_atom.name:
                related.append(analog_atom.name)
        
        return list(set(related))[:limit]  # Remove duplicates and limit
    
    def _sync_to_memory(self, atom: Atom, description: str):
        """Sync atom to memory system."""
        
        if not self.memory:
            return
        
        # Create a recall action to store in memory
        memory_content = {
            "concept": atom.name,
            "description": description,
            "truth_value": atom.truth_value.to_dict(),
            "type": "opencog_knowledge",
            "source": "atomspace"
        }
        
        # This would integrate with the memory system
        # Implementation depends on Memory class interface
        pass
    
    def _retrieve_from_memory(self, concept: str) -> Optional[Dict[str, Any]]:
        """Retrieve from memory system."""
        
        if not self.memory:
            return None
        
        # This would query the memory system for the concept
        # Implementation depends on Memory class interface
        return None
    
    def learn_from_event(self, event: Event):
        """Learn from an event by extracting knowledge."""
        
        event_type = type(event).__name__
        
        # Extract key information based on event type
        if hasattr(event, 'content'):
            content = str(event.content)
            
            # Create event concept
            event_atom = self.store_knowledge(
                f"Event:{event_type}",
                content[:200],  # Truncate long content
                TruthValue(0.7, 0.6)
            )
            
            # Extract concepts from content (simple keyword extraction)
            keywords = self._extract_keywords(content)
            for keyword in keywords:
                keyword_atom = self.store_knowledge(
                    keyword,
                    f"Concept from {event_type}",
                    TruthValue(0.6, 0.5)
                )
                
                # Create association
                self.atomspace.add_atom(
                    AtomType.SIMILARITY_LINK,
                    outgoing=[event_atom, keyword_atom],
                    truth_value=TruthValue(0.5, 0.4)
                )
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Simple keyword extraction from text."""
        
        # Simple implementation - in practice would use NLP
        words = text.lower().split()
        
        # Filter out common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        keywords = []
        for word in words:
            word = word.strip('.,!?;:"()[]{}')
            if (len(word) > 3 and 
                word not in stopwords and 
                word.isalpha()):
                keywords.append(word)
        
        # Return most frequent keywords
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, _ in word_counts.most_common(max_keywords)]
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get a summary of stored knowledge."""
        
        total_atoms = self.atomspace.size()
        concept_nodes = len(self.atomspace.get_atoms_by_type(AtomType.CONCEPT_NODE))
        links = total_atoms - concept_nodes
        
        # Get most attended concepts
        all_concepts = self.atomspace.get_atoms_by_type(AtomType.CONCEPT_NODE)
        top_concepts = sorted(
            all_concepts,
            key=lambda a: a.attention_value.sti,
            reverse=True
        )[:10]
        
        return {
            "total_atoms": total_atoms,
            "concept_nodes": concept_nodes,
            "links": links,
            "cache_size": len(self.knowledge_cache),
            "top_concepts": [atom.name for atom in top_concepts if atom.name]
        }
    
    def export_knowledge(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export knowledge to file or return as dict."""
        
        knowledge_data = {
            "atomspace": self.atomspace.export_to_dict(),
            "cache": self.knowledge_cache,
            "summary": self.get_knowledge_summary()
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(knowledge_data, f, indent=2, default=str)
        
        return knowledge_data
    
    def import_knowledge(self, data: Dict[str, Any]):
        """Import knowledge from data."""
        
        if "atomspace" in data:
            self.atomspace.import_from_dict(data["atomspace"])
        
        if "cache" in data:
            self.knowledge_cache.update(data["cache"])
    
    def clear_cache(self):
        """Clear the knowledge cache."""
        self.knowledge_cache.clear()