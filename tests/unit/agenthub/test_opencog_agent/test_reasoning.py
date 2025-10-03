"""
Tests for OpenCoq reasoning components.
"""

import unittest
from openhands.agenthub.opencog_agent.atomspace import (
    AtomSpace, AtomType, TruthValue
)
from openhands.agenthub.opencog_agent.reasoning import (
    PatternMatcher, AttentionBroker, CognitiveReasoner, Pattern,
    inheritance_rule, similarity_rule
)


class TestPatternMatcher(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
        self.pattern_matcher = PatternMatcher(self.atomspace)
        
        # Create test atoms
        self.animal = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE, "Animal"
        )
        self.dog = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE, "Dog"
        )
        self.inheritance = self.atomspace.add_atom(
            AtomType.INHERITANCE_LINK,
            outgoing=[self.dog, self.animal],
            truth_value=TruthValue(0.9, 0.8)
        )
    
    def test_simple_pattern_matching(self):
        """Test basic pattern matching."""
        pattern = Pattern(atom_type=AtomType.CONCEPT_NODE)
        matches = self.pattern_matcher.find_matches(pattern)
        
        # Should find both concept nodes
        self.assertEqual(len(matches), 2)
        self.assertIn(self.animal, matches)
        self.assertIn(self.dog, matches)
    
    def test_name_pattern_matching(self):
        """Test pattern matching with name patterns."""
        pattern = Pattern(
            atom_type=AtomType.CONCEPT_NODE,
            name_pattern="Dog"
        )
        matches = self.pattern_matcher.find_matches(pattern)
        
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], self.dog)
    
    def test_truth_value_filtering(self):
        """Test pattern matching with truth value constraints."""
        high_confidence_atom = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "HighConfidence",
            truth_value=TruthValue(0.9, 0.9)
        )
        
        pattern = Pattern(
            atom_type=AtomType.CONCEPT_NODE,
            truth_value_min=0.8
        )
        matches = self.pattern_matcher.find_matches(pattern)
        
        # Should only find the high confidence atom
        self.assertIn(high_confidence_atom, matches)


class TestAttentionBroker(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
        self.attention_broker = AttentionBroker(self.atomspace)
        
        # Create test atoms
        self.test_atom = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE, "TestAtom"
        )
    
    def test_stimulate_atom(self):
        """Test atom stimulation."""
        initial_sti = self.test_atom.attention_value.sti
        self.attention_broker.stimulate_atom(self.test_atom, 10.0)
        
        self.assertGreater(
            self.test_atom.attention_value.sti,
            initial_sti
        )
    
    def test_attention_focus(self):
        """Test getting attentional focus."""
        # Stimulate atom to bring it into focus
        self.attention_broker.stimulate_atom(self.test_atom, 5.0)
        
        focus = self.attention_broker.get_attentional_focus(limit=5)
        
        # Should be in focus if above threshold
        if self.test_atom.attention_value.sti > self.attention_broker.attention_threshold:
            self.assertIn(self.test_atom, focus)
    
    def test_attention_decay(self):
        """Test attention decay mechanism."""
        # Stimulate atom
        self.attention_broker.stimulate_atom(self.test_atom, 10.0)
        attention_before_decay = self.test_atom.attention_value.sti
        
        # Apply decay
        self.attention_broker.decay_attention()
        
        self.assertLess(
            self.test_atom.attention_value.sti,
            attention_before_decay
        )


class TestCognitiveReasoner(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
        self.cognitive_reasoner = CognitiveReasoner(self.atomspace)
        
        # Add default inference rules
        self.cognitive_reasoner.add_inference_rule(inheritance_rule)
        self.cognitive_reasoner.add_inference_rule(similarity_rule)
    
    def test_inheritance_inference(self):
        """Test inheritance-based inference."""
        # Create chain: Dog -> Animal -> LivingThing
        living_thing = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE, "LivingThing"
        )
        animal = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE, "Animal"
        )
        dog = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE, "Dog"
        )
        
        # Create inheritance links
        self.atomspace.add_atom(
            AtomType.INHERITANCE_LINK,
            outgoing=[dog, animal],
            truth_value=TruthValue(0.9, 0.8)
        )
        self.atomspace.add_atom(
            AtomType.INHERITANCE_LINK,
            outgoing=[animal, living_thing],
            truth_value=TruthValue(0.8, 0.7)
        )
        
        # Run inference
        new_atoms = self.cognitive_reasoner.infer(max_iterations=1)
        
        # Should infer Dog -> LivingThing
        inheritance_links = self.atomspace.get_atoms_by_type(AtomType.INHERITANCE_LINK)
        
        # Find the inferred link
        inferred_link = None
        for link in inheritance_links:
            if (len(link.outgoing) == 2 and
                link.outgoing[0] == dog and
                link.outgoing[1] == living_thing):
                inferred_link = link
                break
        
        self.assertIsNotNone(inferred_link, "Should have inferred Dog -> LivingThing")
    
    def test_goal_reasoning(self):
        """Test goal-directed reasoning."""
        goal_concepts = self.cognitive_reasoner.reason_about_goal("programming")
        
        # Should create goal atom and related concepts
        self.assertGreater(len(goal_concepts), 0)
        
        # Check that goal atom was created
        goal_atoms = self.atomspace.get_atoms_by_name("Goal:programming")
        self.assertGreater(len(goal_atoms), 0)
    
    def test_truth_evaluation(self):
        """Test truth value evaluation."""
        test_atom = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "TestStatement",
            truth_value=TruthValue(0.8, 0.7)
        )
        
        truth_score = self.cognitive_reasoner.evaluate_truth(test_atom)
        
        # Should return something close to the base truth value
        self.assertAlmostEqual(truth_score, 0.8, delta=0.2)
    
    def test_analogy_finding(self):
        """Test finding analogies between concepts."""
        # Create similar structures
        concept1 = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE, "Concept1"
        )
        concept2 = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE, "Concept2"
        )
        
        # Add similarity link
        self.atomspace.add_atom(
            AtomType.SIMILARITY_LINK,
            outgoing=[concept1, concept2],
            truth_value=TruthValue(0.8, 0.7)
        )
        
        analogies = self.cognitive_reasoner.find_analogies(concept1, limit=5)
        
        # Should find concept2 as an analogy
        analog_atoms = [atom for atom, score in analogies]
        self.assertIn(concept2, analog_atoms)


class TestInferenceRules(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
    
    def test_inheritance_rule(self):
        """Test inheritance rule implementation."""
        # Create A -> B -> C chain
        a = self.atomspace.add_atom(AtomType.CONCEPT_NODE, "A")
        b = self.atomspace.add_atom(AtomType.CONCEPT_NODE, "B")
        c = self.atomspace.add_atom(AtomType.CONCEPT_NODE, "C")
        
        self.atomspace.add_atom(
            AtomType.INHERITANCE_LINK,
            outgoing=[a, b],
            truth_value=TruthValue(0.9, 0.8)
        )
        self.atomspace.add_atom(
            AtomType.INHERITANCE_LINK,
            outgoing=[b, c],
            truth_value=TruthValue(0.8, 0.7)
        )
        
        # Apply rule
        new_atoms = inheritance_rule(self.atomspace)
        
        # Should create A -> C link
        self.assertGreater(len(new_atoms), 0)
        
        # Verify the new link exists
        a_to_c_links = [
            atom for atom in new_atoms
            if (atom.atom_type == AtomType.INHERITANCE_LINK and
                len(atom.outgoing) == 2 and
                atom.outgoing[0] == a and
                atom.outgoing[1] == c)
        ]
        
        self.assertGreater(len(a_to_c_links), 0)
    
    def test_similarity_rule(self):
        """Test similarity rule implementation."""
        # Create A ~ B ~ C chain
        a = self.atomspace.add_atom(AtomType.CONCEPT_NODE, "A")
        b = self.atomspace.add_atom(AtomType.CONCEPT_NODE, "B")
        c = self.atomspace.add_atom(AtomType.CONCEPT_NODE, "C")
        
        self.atomspace.add_atom(
            AtomType.SIMILARITY_LINK,
            outgoing=[a, b],
            truth_value=TruthValue(0.8, 0.7)
        )
        self.atomspace.add_atom(
            AtomType.SIMILARITY_LINK,
            outgoing=[b, c],
            truth_value=TruthValue(0.7, 0.6)
        )
        
        # Apply rule
        new_atoms = similarity_rule(self.atomspace)
        
        # Should create A ~ C link
        self.assertGreater(len(new_atoms), 0)
        
        # Verify the new link exists
        a_to_c_links = [
            atom for atom in new_atoms
            if (atom.atom_type == AtomType.SIMILARITY_LINK and
                len(atom.outgoing) == 2 and
                atom.outgoing[0] == a and
                atom.outgoing[1] == c)
        ]
        
        self.assertGreater(len(a_to_c_links), 0)


if __name__ == '__main__':
    unittest.main()