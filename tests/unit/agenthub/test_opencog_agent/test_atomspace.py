"""
Tests for OpenCoq AtomSpace implementation.
"""

import unittest
from openhands.agenthub.opencog_agent.atomspace import (
    AtomSpace, Atom, AtomType, TruthValue, AttentionValue
)


class TestAtomSpace(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
    
    def test_create_atom(self):
        """Test basic atom creation."""
        atom = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "TestConcept",
            truth_value=TruthValue(0.8, 0.9)
        )
        
        self.assertIsNotNone(atom)
        self.assertEqual(atom.atom_type, AtomType.CONCEPT_NODE)
        self.assertEqual(atom.name, "TestConcept")
        self.assertEqual(atom.truth_value.strength, 0.8)
        self.assertEqual(atom.truth_value.confidence, 0.9)
    
    def test_duplicate_atom_handling(self):
        """Test that duplicate atoms are handled correctly."""
        atom1 = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "DuplicateTest"
        )
        
        atom2 = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "DuplicateTest"
        )
        
        # Should return the same atom
        self.assertEqual(atom1, atom2)
        self.assertEqual(self.atomspace.size(), 1)
    
    def test_create_link(self):
        """Test link creation between atoms."""
        concept1 = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "Animal"
        )
        
        concept2 = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "Dog"
        )
        
        inheritance_link = self.atomspace.add_atom(
            AtomType.INHERITANCE_LINK,
            outgoing=[concept2, concept1],
            truth_value=TruthValue(0.9, 0.8)
        )
        
        self.assertEqual(len(inheritance_link.outgoing), 2)
        self.assertIn(inheritance_link, concept1.incoming)
        self.assertIn(inheritance_link, concept2.incoming)
    
    def test_get_atoms_by_type(self):
        """Test retrieval of atoms by type."""
        # Create some atoms
        self.atomspace.add_atom(AtomType.CONCEPT_NODE, "Concept1")
        self.atomspace.add_atom(AtomType.CONCEPT_NODE, "Concept2")
        self.atomspace.add_atom(AtomType.PREDICATE_NODE, "Predicate1")
        
        concepts = self.atomspace.get_atoms_by_type(AtomType.CONCEPT_NODE)
        predicates = self.atomspace.get_atoms_by_type(AtomType.PREDICATE_NODE)
        
        self.assertEqual(len(concepts), 2)
        self.assertEqual(len(predicates), 1)
    
    def test_get_atoms_by_name(self):
        """Test retrieval of atoms by name."""
        atom = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "NamedConcept"
        )
        
        found_atoms = self.atomspace.get_atoms_by_name("NamedConcept")
        
        self.assertEqual(len(found_atoms), 1)
        self.assertEqual(found_atoms[0], atom)
    
    def test_remove_atom(self):
        """Test atom removal."""
        atom = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "ToRemove"
        )
        
        initial_size = self.atomspace.size()
        removed = self.atomspace.remove_atom(atom)
        
        self.assertTrue(removed)
        self.assertEqual(self.atomspace.size(), initial_size - 1)
        self.assertIsNone(self.atomspace.get_atom(atom.handle))
    
    def test_export_import(self):
        """Test export and import functionality."""
        # Create some test data
        concept = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "ExportTest",
            truth_value=TruthValue(0.7, 0.8)
        )
        
        predicate = self.atomspace.add_atom(
            AtomType.PREDICATE_NODE,
            "testPredicate"
        )
        
        # Export
        exported_data = self.atomspace.export_to_dict()
        
        # Clear and import
        self.atomspace.clear()
        self.assertEqual(self.atomspace.size(), 0)
        
        self.atomspace.import_from_dict(exported_data)
        
        # Verify import
        self.assertEqual(self.atomspace.size(), 2)
        imported_concepts = self.atomspace.get_atoms_by_name("ExportTest")
        self.assertEqual(len(imported_concepts), 1)
        self.assertEqual(imported_concepts[0].truth_value.strength, 0.7)
    
    def test_truth_value_bounds(self):
        """Test that truth values are properly bounded."""
        # Test values outside bounds
        tv = TruthValue(1.5, -0.5)  # Should be clamped
        
        self.assertEqual(tv.strength, 1.0)
        self.assertEqual(tv.confidence, 0.0)
    
    def test_attention_value(self):
        """Test attention value functionality."""
        atom = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "AttentionTest",
            attention_value=AttentionValue(sti=10.0, lti=5.0)
        )
        
        self.assertEqual(atom.attention_value.sti, 10.0)
        self.assertEqual(atom.attention_value.lti, 5.0)


if __name__ == '__main__':
    unittest.main()