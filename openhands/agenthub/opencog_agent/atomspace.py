"""
OpenCoq AtomSpace implementation for knowledge representation.

This module provides a lightweight AtomSpace implementation optimized for
the OpenHands framework, supporting both symbolic and subsymbolic knowledge.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import json
from dataclasses import dataclass, field
from threading import Lock


class AtomType(Enum):
    """Core atom types for knowledge representation."""
    
    # Basic types
    NODE = "Node"
    LINK = "Link"
    
    # Specific node types
    CONCEPT_NODE = "ConceptNode"
    PREDICATE_NODE = "PredicateNode"
    SCHEMA_NODE = "SchemaNode"
    VARIABLE_NODE = "VariableNode"
    
    # Specific link types
    INHERITANCE_LINK = "InheritanceLink"
    SIMILARITY_LINK = "SimilarityLink"
    EVALUATION_LINK = "EvaluationLink"
    EXECUTION_LINK = "ExecutionLink"
    LIST_LINK = "ListLink"
    AND_LINK = "AndLink"
    OR_LINK = "OrLink"
    NOT_LINK = "NotLink"
    
    # Attention and memory types
    ATTENTION_VALUE = "AttentionValue"
    TRUTH_VALUE = "TruthValue"


@dataclass
class TruthValue:
    """Truth value for atoms with strength and confidence."""
    strength: float = 1.0  # [0, 1]
    confidence: float = 1.0  # [0, 1]
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def to_dict(self) -> Dict[str, float]:
        return {"strength": self.strength, "confidence": self.confidence}


@dataclass
class AttentionValue:
    """Attention value for atoms with STI, LTI, and VLTI."""
    sti: float = 0.0  # Short-term importance
    lti: float = 0.0  # Long-term importance
    vlti: float = 0.0  # Very long-term importance
    
    def to_dict(self) -> Dict[str, float]:
        return {"sti": self.sti, "lti": self.lti, "vlti": self.vlti}


class Atom:
    """Base class for all atoms in the AtomSpace."""
    
    def __init__(
        self,
        atom_type: AtomType,
        name: Optional[str] = None,
        outgoing: Optional[List['Atom']] = None,
        truth_value: Optional[TruthValue] = None,
        attention_value: Optional[AttentionValue] = None
    ):
        self.handle = str(uuid.uuid4())
        self.atom_type = atom_type
        self.name = name or ""
        self.outgoing = outgoing or []
        self.incoming: Set['Atom'] = set()
        self.truth_value = truth_value or TruthValue()
        self.attention_value = attention_value or AttentionValue()
        
        # Update incoming sets for outgoing atoms
        for atom in self.outgoing:
            atom.incoming.add(self)
    
    def __hash__(self) -> int:
        return hash(self.handle)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Atom):
            return False
        return self.handle == other.handle
    
    def __str__(self) -> str:
        if self.name:
            return f"{self.atom_type.value}({self.name})"
        elif self.outgoing:
            outgoing_str = ", ".join(str(atom) for atom in self.outgoing)
            return f"{self.atom_type.value}({outgoing_str})"
        else:
            return f"{self.atom_type.value}()"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert atom to dictionary representation."""
        return {
            "handle": self.handle,
            "type": self.atom_type.value,
            "name": self.name,
            "outgoing": [atom.handle for atom in self.outgoing],
            "truth_value": self.truth_value.to_dict(),
            "attention_value": self.attention_value.to_dict()
        }


class AtomSpace:
    """
    AtomSpace for storing and managing atoms with knowledge representation.
    
    This implementation provides core OpenCoq functionality optimized for
    the OpenHands framework.
    """
    
    def __init__(self):
        self._atoms: Dict[str, Atom] = {}
        self._type_index: Dict[AtomType, Set[Atom]] = {}
        self._name_index: Dict[str, Set[Atom]] = {}
        self._lock = Lock()
        
        # Initialize type index
        for atom_type in AtomType:
            self._type_index[atom_type] = set()
    
    def add_atom(
        self,
        atom_type: AtomType,
        name: Optional[str] = None,
        outgoing: Optional[List[Atom]] = None,
        truth_value: Optional[TruthValue] = None,
        attention_value: Optional[AttentionValue] = None
    ) -> Atom:
        """Add an atom to the AtomSpace."""
        with self._lock:
            # Check if atom already exists
            existing = self._find_existing_atom(atom_type, name, outgoing)
            if existing:
                # Update truth value if provided
                if truth_value:
                    existing.truth_value = truth_value
                if attention_value:
                    existing.attention_value = attention_value
                return existing
            
            # Create new atom
            atom = Atom(atom_type, name, outgoing, truth_value, attention_value)
            
            # Add to storage and indices
            self._atoms[atom.handle] = atom
            self._type_index[atom_type].add(atom)
            
            if name:
                if name not in self._name_index:
                    self._name_index[name] = set()
                self._name_index[name].add(atom)
            
            return atom
    
    def _find_existing_atom(
        self,
        atom_type: AtomType,
        name: Optional[str],
        outgoing: Optional[List[Atom]]
    ) -> Optional[Atom]:
        """Find existing atom with same type, name, and outgoing set."""
        candidates = self._type_index[atom_type]
        
        for atom in candidates:
            if atom.name == (name or "") and atom.outgoing == (outgoing or []):
                return atom
        
        return None
    
    def get_atom(self, handle: str) -> Optional[Atom]:
        """Get atom by handle."""
        return self._atoms.get(handle)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a specific type."""
        return list(self._type_index[atom_type])
    
    def get_atoms_by_name(self, name: str) -> List[Atom]:
        """Get all atoms with a specific name."""
        return list(self._name_index.get(name, set()))
    
    def remove_atom(self, atom: Union[Atom, str]) -> bool:
        """Remove an atom from the AtomSpace."""
        with self._lock:
            if isinstance(atom, str):
                atom_obj = self._atoms.get(atom)
                if not atom_obj:
                    return False
            else:
                atom_obj = atom
            
            # Remove from indices
            self._type_index[atom_obj.atom_type].discard(atom_obj)
            if atom_obj.name and atom_obj.name in self._name_index:
                self._name_index[atom_obj.name].discard(atom_obj)
                if not self._name_index[atom_obj.name]:
                    del self._name_index[atom_obj.name]
            
            # Update incoming/outgoing relationships
            for outgoing_atom in atom_obj.outgoing:
                outgoing_atom.incoming.discard(atom_obj)
            
            for incoming_atom in atom_obj.incoming:
                if atom_obj in incoming_atom.outgoing:
                    incoming_atom.outgoing.remove(atom_obj)
            
            # Remove from main storage
            del self._atoms[atom_obj.handle]
            return True
    
    def size(self) -> int:
        """Get the number of atoms in the AtomSpace."""
        return len(self._atoms)
    
    def clear(self):
        """Clear all atoms from the AtomSpace."""
        with self._lock:
            self._atoms.clear()
            for atom_set in self._type_index.values():
                atom_set.clear()
            self._name_index.clear()
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export AtomSpace to dictionary format."""
        return {
            "atoms": [atom.to_dict() for atom in self._atoms.values()],
            "size": self.size()
        }
    
    def import_from_dict(self, data: Dict[str, Any]):
        """Import AtomSpace from dictionary format."""
        self.clear()
        
        # First pass: create all atoms without outgoing links
        handle_map = {}
        for atom_data in data["atoms"]:
            atom_type = AtomType(atom_data["type"])
            name = atom_data["name"] if atom_data["name"] else None
            truth_value = TruthValue(**atom_data["truth_value"])
            attention_value = AttentionValue(**atom_data["attention_value"])
            
            atom = Atom(atom_type, name, [], truth_value, attention_value)
            atom.handle = atom_data["handle"]  # Preserve original handle
            handle_map[atom_data["handle"]] = atom
            
            self._atoms[atom.handle] = atom
            self._type_index[atom_type].add(atom)
            if name:
                if name not in self._name_index:
                    self._name_index[name] = set()
                self._name_index[name].add(atom)
        
        # Second pass: establish outgoing links
        for atom_data in data["atoms"]:
            atom = handle_map[atom_data["handle"]]
            for outgoing_handle in atom_data["outgoing"]:
                outgoing_atom = handle_map[outgoing_handle]
                atom.outgoing.append(outgoing_atom)
                outgoing_atom.incoming.add(atom)