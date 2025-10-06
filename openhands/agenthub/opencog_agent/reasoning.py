"""
OpenCoq reasoning and pattern matching components.

This module provides pattern matching, attention allocation, and reasoning
capabilities for the OpenCoq agent system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
import re
import math
from dataclasses import dataclass
from .atomspace import AtomSpace, Atom, AtomType, TruthValue, AttentionValue


@dataclass
class Pattern:
    """Represents a pattern for matching atoms in the AtomSpace."""
    atom_type: Optional[AtomType] = None
    name_pattern: Optional[str] = None
    outgoing_patterns: Optional[List['Pattern']] = None
    truth_value_min: Optional[float] = None
    truth_value_max: Optional[float] = None
    
    def matches(self, atom: Atom) -> bool:
        """Check if the pattern matches the given atom."""
        # Check atom type
        if self.atom_type and atom.atom_type != self.atom_type:
            return False
        
        # Check name pattern
        if self.name_pattern and not re.match(self.name_pattern, atom.name):
            return False
        
        # Check truth value range
        if self.truth_value_min and atom.truth_value.strength < self.truth_value_min:
            return False
        if self.truth_value_max and atom.truth_value.strength > self.truth_value_max:
            return False
        
        # Check outgoing patterns
        if self.outgoing_patterns:
            if len(self.outgoing_patterns) != len(atom.outgoing):
                return False
            for pattern, outgoing_atom in zip(self.outgoing_patterns, atom.outgoing):
                if not pattern.matches(outgoing_atom):
                    return False
        
        return True


class PatternMatcher:
    """
    Pattern matching engine for the AtomSpace.
    
    Provides sophisticated pattern matching capabilities including
    variable binding and constraint satisfaction.
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.variable_bindings: Dict[str, Atom] = {}
    
    def find_matches(self, pattern: Pattern) -> List[Atom]:
        """Find all atoms that match the given pattern."""
        matches = []
        
        # Start with atoms of the specified type if available
        if pattern.atom_type:
            candidates = self.atomspace.get_atoms_by_type(pattern.atom_type)
        else:
            candidates = list(self.atomspace._atoms.values())
        
        for atom in candidates:
            if pattern.matches(atom):
                matches.append(atom)
        
        return matches
    
    def bind_pattern(
        self, 
        pattern: Pattern, 
        atom: Atom, 
        bindings: Optional[Dict[str, Atom]] = None
    ) -> Optional[Dict[str, Atom]]:
        """
        Attempt to bind a pattern to an atom, returning variable bindings.
        """
        if bindings is None:
            bindings = {}
        
        # Check if pattern matches atom structure
        if not pattern.matches(atom):
            return None
        
        # Handle variable binding
        if (pattern.name_pattern and 
            pattern.name_pattern.startswith('$') and 
            len(pattern.name_pattern) > 1):
            var_name = pattern.name_pattern[1:]  # Remove '$' prefix
            if var_name in bindings:
                if bindings[var_name] != atom:
                    return None
            else:
                bindings[var_name] = atom
        
        # Recursively bind outgoing patterns
        if pattern.outgoing_patterns:
            for p, a in zip(pattern.outgoing_patterns, atom.outgoing):
                result = self.bind_pattern(p, a, bindings)
                if result is None:
                    return None
                bindings.update(result)
        
        return bindings
    
    def query(self, query_pattern: Pattern) -> List[Dict[str, Atom]]:
        """
        Execute a query pattern and return all possible variable bindings.
        """
        results = []
        candidates = self.find_matches(query_pattern)
        
        for atom in candidates:
            bindings = self.bind_pattern(query_pattern, atom)
            if bindings is not None:
                results.append(bindings)
        
        return results


class AttentionBroker:
    """
    Attention allocation system for managing cognitive resources.
    
    Implements importance-based attention allocation with spreading
    activation and decay mechanisms.
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.attention_threshold = 0.1
        self.decay_rate = 0.95
        self.spread_factor = 0.1
        self.max_attention = 100.0
    
    def get_attentional_focus(self, limit: int = 10) -> List[Atom]:
        """Get the atoms currently in attentional focus."""
        all_atoms = list(self.atomspace._atoms.values())
        
        # Sort by STI (short-term importance)
        focused_atoms = sorted(
            all_atoms,
            key=lambda a: a.attention_value.sti,
            reverse=True
        )
        
        # Filter by attention threshold and limit
        result = []
        for atom in focused_atoms[:limit]:
            if atom.attention_value.sti > self.attention_threshold:
                result.append(atom)
        
        return result
    
    def stimulate_atom(self, atom: Atom, amount: float):
        """Increase the attention value of an atom."""
        current_sti = atom.attention_value.sti
        new_sti = min(self.max_attention, current_sti + amount)
        atom.attention_value.sti = new_sti
        
        # Spread activation to related atoms
        self._spread_activation(atom, amount * self.spread_factor)
    
    def _spread_activation(self, source_atom: Atom, amount: float):
        """Spread activation to related atoms."""
        # Spread to outgoing atoms
        for atom in source_atom.outgoing:
            atom.attention_value.sti += amount * 0.5
        
        # Spread to incoming atoms
        for atom in source_atom.incoming:
            atom.attention_value.sti += amount * 0.3
    
    def decay_attention(self):
        """Apply attention decay to all atoms."""
        for atom in self.atomspace._atoms.values():
            atom.attention_value.sti *= self.decay_rate
            atom.attention_value.lti *= self.decay_rate
            
            # Move STI to LTI gradually
            transfer_amount = atom.attention_value.sti * 0.01
            atom.attention_value.sti -= transfer_amount
            atom.attention_value.lti += transfer_amount
    
    def update_importance(self, atom: Atom, new_importance: float):
        """Update the importance value of an atom."""
        atom.attention_value.lti = max(0.0, min(self.max_attention, new_importance))


class CognitiveReasoner:
    """
    High-level reasoning engine that combines pattern matching and attention.
    
    Provides inference capabilities and knowledge-based reasoning.
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.pattern_matcher = PatternMatcher(atomspace)
        self.attention_broker = AttentionBroker(atomspace)
        self.inference_rules: List[Callable] = []
    
    def add_inference_rule(self, rule: Callable[[AtomSpace], List[Atom]]):
        """Add a custom inference rule."""
        self.inference_rules.append(rule)
    
    def infer(self, max_iterations: int = 10) -> List[Atom]:
        """
        Run inference process using available rules and attention.
        """
        new_atoms = []
        
        for _ in range(max_iterations):
            iteration_atoms = []
            
            # Apply all inference rules
            for rule in self.inference_rules:
                try:
                    rule_atoms = rule(self.atomspace)
                    iteration_atoms.extend(rule_atoms)
                except Exception as e:
                    # Log error but continue with other rules
                    continue
            
            if not iteration_atoms:
                break
            
            new_atoms.extend(iteration_atoms)
            
            # Update attention for new atoms
            for atom in iteration_atoms:
                self.attention_broker.stimulate_atom(atom, 1.0)
        
        return new_atoms
    
    def reason_about_goal(self, goal_description: str) -> List[Atom]:
        """
        Reason about a specific goal using available knowledge.
        """
        # Create goal atom
        goal_atom = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            name=f"Goal:{goal_description}",
            truth_value=TruthValue(strength=0.9, confidence=0.8)
        )
        
        # Stimulate goal atom
        self.attention_broker.stimulate_atom(goal_atom, 10.0)
        
        # Find related concepts
        related_concepts = []
        goal_words = goal_description.lower().split()
        
        for word in goal_words:
            pattern = Pattern(
                atom_type=AtomType.CONCEPT_NODE,
                name_pattern=f".*{word}.*"
            )
            matches = self.pattern_matcher.find_matches(pattern)
            related_concepts.extend(matches)
        
        # Create associations
        for concept in related_concepts:
            association = self.atomspace.add_atom(
                AtomType.SIMILARITY_LINK,
                outgoing=[goal_atom, concept],
                truth_value=TruthValue(strength=0.7, confidence=0.6)
            )
            self.attention_broker.stimulate_atom(association, 2.0)
        
        return [goal_atom] + related_concepts
    
    def evaluate_truth(self, statement_atom: Atom) -> float:
        """
        Evaluate the truth value of a statement based on available evidence.
        """
        if not statement_atom.truth_value:
            return 0.5  # Default neutral
        
        # Consider supporting evidence
        support_strength = 0.0
        support_count = 0
        
        # Look for supporting inheritance links
        for atom in self.atomspace._atoms.values():
            if (atom.atom_type == AtomType.INHERITANCE_LINK and 
                len(atom.outgoing) == 2 and
                statement_atom in atom.outgoing):
                support_strength += atom.truth_value.strength * atom.truth_value.confidence
                support_count += 1
        
        # Combine base truth value with supporting evidence
        base_truth = statement_atom.truth_value.strength
        if support_count > 0:
            evidence_factor = support_strength / support_count
            return (base_truth + evidence_factor) / 2.0
        
        return base_truth
    
    def find_analogies(self, source_atom: Atom, limit: int = 5) -> List[Tuple[Atom, float]]:
        """
        Find analogous atoms based on structural similarity.
        """
        analogies = []
        
        # Get atoms with similar structure
        candidates = self.atomspace.get_atoms_by_type(source_atom.atom_type)
        
        for candidate in candidates:
            if candidate == source_atom:
                continue
            
            similarity_score = self._compute_structural_similarity(source_atom, candidate)
            if similarity_score > 0.3:  # Threshold for analogies
                analogies.append((candidate, similarity_score))
        
        # Sort by similarity and return top matches
        analogies.sort(key=lambda x: x[1], reverse=True)
        return analogies[:limit]
    
    def _compute_structural_similarity(self, atom1: Atom, atom2: Atom) -> float:
        """Compute structural similarity between two atoms."""
        if atom1.atom_type != atom2.atom_type:
            return 0.0
        
        # Compare outgoing sets
        if len(atom1.outgoing) != len(atom2.outgoing):
            return 0.0
        
        if not atom1.outgoing and not atom2.outgoing:
            # Both are leaf nodes, compare names
            if atom1.name and atom2.name:
                return 1.0 if atom1.name == atom2.name else 0.5
            return 0.7  # Default similarity for unnamed leaf nodes
        
        # Recursive similarity for complex structures
        total_similarity = 0.0
        for out1, out2 in zip(atom1.outgoing, atom2.outgoing):
            total_similarity += self._compute_structural_similarity(out1, out2)
        
        return total_similarity / len(atom1.outgoing)


# Default inference rules

def inheritance_rule(atomspace: AtomSpace) -> List[Atom]:
    """Basic inheritance inference rule: if A inherits from B and B inherits from C, then A inherits from C."""
    new_atoms = []
    
    # Find all inheritance links
    inheritance_links = atomspace.get_atoms_by_type(AtomType.INHERITANCE_LINK)
    
    for link1 in inheritance_links:
        if len(link1.outgoing) != 2:
            continue
        
        a, b = link1.outgoing
        
        # Find links where B inherits from C
        for link2 in inheritance_links:
            if len(link2.outgoing) != 2 or link2 == link1:
                continue
            
            b2, c = link2.outgoing
            if b == b2:  # B inherits from C
                # Create A inherits from C
                new_link = atomspace.add_atom(
                    AtomType.INHERITANCE_LINK,
                    outgoing=[a, c],
                    truth_value=TruthValue(
                        strength=min(link1.truth_value.strength, link2.truth_value.strength),
                        confidence=min(link1.truth_value.confidence, link2.truth_value.confidence) * 0.9
                    )
                )
                new_atoms.append(new_link)
    
    return new_atoms


def similarity_rule(atomspace: AtomSpace) -> List[Atom]:
    """Similarity inference rule: if A is similar to B and B is similar to C, then A is similar to C."""
    new_atoms = []
    
    similarity_links = atomspace.get_atoms_by_type(AtomType.SIMILARITY_LINK)
    
    for link1 in similarity_links:
        if len(link1.outgoing) != 2:
            continue
        
        a, b = link1.outgoing
        
        for link2 in similarity_links:
            if len(link2.outgoing) != 2 or link2 == link1:
                continue
            
            b2, c = link2.outgoing
            if b == b2:
                # Create A similar to C
                new_link = atomspace.add_atom(
                    AtomType.SIMILARITY_LINK,
                    outgoing=[a, c],
                    truth_value=TruthValue(
                        strength=min(link1.truth_value.strength, link2.truth_value.strength) * 0.8,
                        confidence=min(link1.truth_value.confidence, link2.truth_value.confidence) * 0.8
                    )
                )
                new_atoms.append(new_link)
    
    return new_atoms