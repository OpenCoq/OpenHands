#!/usr/bin/env python3
"""
OpenCoq Agent Demo Script

This script demonstrates the core capabilities of the OpenCoq cognitive
architecture integration with OpenHands.
"""

import sys
import json
from pathlib import Path

# Add the parent directory to sys.path to import OpenHands modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from openhands.agenthub.opencog_agent.atomspace import (
    AtomSpace, AtomType, TruthValue, AttentionValue
)
from openhands.agenthub.opencog_agent.reasoning import (
    CognitiveReasoner, PatternMatcher, AttentionBroker, Pattern,
    inheritance_rule, similarity_rule
)
from openhands.agenthub.opencog_agent.memory_integration import OpenCogMemoryIntegration


def demo_atomspace():
    """Demonstrate AtomSpace knowledge representation."""
    print("=" * 60)
    print("OpenCoq AtomSpace Demo")
    print("=" * 60)
    
    # Create AtomSpace
    atomspace = AtomSpace()
    
    # Create programming concepts
    print("\n1. Creating programming concepts...")
    programming = atomspace.add_atom(
        AtomType.CONCEPT_NODE,
        "Programming",
        truth_value=TruthValue(1.0, 0.9)
    )
    
    python = atomspace.add_atom(
        AtomType.CONCEPT_NODE,
        "Python",
        truth_value=TruthValue(0.9, 0.8)
    )
    
    debugging = atomspace.add_atom(
        AtomType.CONCEPT_NODE,
        "Debugging",
        truth_value=TruthValue(0.8, 0.7)
    )
    
    # Create relationships
    print("2. Creating relationships...")
    python_is_programming = atomspace.add_atom(
        AtomType.INHERITANCE_LINK,
        outgoing=[python, programming],
        truth_value=TruthValue(0.9, 0.8)
    )
    
    debugging_is_programming = atomspace.add_atom(
        AtomType.INHERITANCE_LINK,
        outgoing=[debugging, programming],
        truth_value=TruthValue(0.8, 0.7)
    )
    
    python_debugging_similarity = atomspace.add_atom(
        AtomType.SIMILARITY_LINK,
        outgoing=[python, debugging],
        truth_value=TruthValue(0.6, 0.5)
    )
    
    print(f"AtomSpace now contains {atomspace.size()} atoms")
    
    # Display some atoms
    print("\n3. Sample atoms:")
    for atom_type in [AtomType.CONCEPT_NODE, AtomType.INHERITANCE_LINK]:
        atoms = atomspace.get_atoms_by_type(atom_type)
        print(f"  {atom_type.value}: {len(atoms)} atoms")
        for atom in atoms[:3]:  # Show first 3
            print(f"    - {atom}")
    
    return atomspace


def demo_pattern_matching(atomspace):
    """Demonstrate pattern matching capabilities."""
    print("\n" + "=" * 60)
    print("Pattern Matching Demo")  
    print("=" * 60)
    
    pattern_matcher = PatternMatcher(atomspace)
    
    # Find all concept nodes
    print("\n1. Finding all concept nodes...")
    concept_pattern = Pattern(atom_type=AtomType.CONCEPT_NODE)
    concepts = pattern_matcher.find_matches(concept_pattern)
    print(f"Found {len(concepts)} concept nodes:")
    for concept in concepts:
        print(f"  - {concept}")
    
    # Find high-confidence atoms
    print("\n2. Finding high-confidence atoms...")
    high_conf_pattern = Pattern(
        atom_type=AtomType.CONCEPT_NODE,
        truth_value_min=0.8
    )
    high_conf_atoms = pattern_matcher.find_matches(high_conf_pattern)
    print(f"Found {len(high_conf_atoms)} high-confidence atoms:")
    for atom in high_conf_atoms:
        print(f"  - {atom} (confidence: {atom.truth_value.confidence:.2f})")
    
    # Pattern with name matching
    print("\n3. Finding atoms with 'Program' in name...")
    name_pattern = Pattern(
        atom_type=AtomType.CONCEPT_NODE,
        name_pattern=".*Program.*"
    )
    program_atoms = pattern_matcher.find_matches(name_pattern)
    print(f"Found {len(program_atoms)} atoms:")
    for atom in program_atoms:
        print(f"  - {atom}")


def demo_attention_system(atomspace):
    """Demonstrate attention allocation system."""
    print("\n" + "=" * 60)
    print("Attention System Demo")
    print("=" * 60)
    
    attention_broker = AttentionBroker(atomspace)
    
    # Get programming atom and stimulate it
    programming_atoms = atomspace.get_atoms_by_name("Programming")
    if programming_atoms:
        programming_atom = programming_atoms[0]
        
        print(f"\n1. Initial attention: {programming_atom.attention_value.sti:.2f}")
        
        # Stimulate the atom
        attention_broker.stimulate_atom(programming_atom, 10.0)
        print(f"2. After stimulation: {programming_atom.attention_value.sti:.2f}")
        
        # Get attentional focus
        print("\n3. Current attentional focus:")
        focused = attention_broker.get_attentional_focus(limit=5)
        for i, atom in enumerate(focused, 1):
            print(f"  {i}. {atom} (STI: {atom.attention_value.sti:.2f})")
        
        # Apply decay
        print("\n4. Applying attention decay...")
        attention_broker.decay_attention()
        print(f"   Programming attention after decay: {programming_atom.attention_value.sti:.2f}")


def demo_cognitive_reasoning(atomspace):
    """Demonstrate cognitive reasoning capabilities."""
    print("\n" + "=" * 60)
    print("Cognitive Reasoning Demo")
    print("=" * 60)
    
    cognitive_reasoner = CognitiveReasoner(atomspace)
    
    # Add inference rules
    cognitive_reasoner.add_inference_rule(inheritance_rule)
    cognitive_reasoner.add_inference_rule(similarity_rule)
    
    print("\n1. Running inference...")
    new_atoms = cognitive_reasoner.infer(max_iterations=3)
    print(f"Generated {len(new_atoms)} new inferences:")
    for atom in new_atoms:
        print(f"  - {atom} (strength: {atom.truth_value.strength:.2f})")
    
    # Goal-directed reasoning
    print("\n2. Goal-directed reasoning...")
    goal_concepts = cognitive_reasoner.reason_about_goal("learn Python programming")
    print(f"Identified {len(goal_concepts)} relevant concepts:")
    for concept in goal_concepts[:5]:  # Show top 5
        print(f"  - {concept} (attention: {concept.attention_value.sti:.2f})")
    
    # Find analogies
    print("\n3. Finding analogies...")
    python_atoms = atomspace.get_atoms_by_name("Python")
    if python_atoms:
        python_atom = python_atoms[0]
        analogies = cognitive_reasoner.find_analogies(python_atom, limit=3)
        print(f"Found {len(analogies)} analogies to Python:")
        for analog_atom, similarity in analogies:
            print(f"  - {analog_atom} (similarity: {similarity:.2f})")
    
    # Truth evaluation
    print("\n4. Truth evaluation...")
    if python_atoms:
        truth_score = cognitive_reasoner.evaluate_truth(python_atoms[0])
        print(f"Truth score for 'Python': {truth_score:.3f}")


def demo_memory_integration(atomspace):
    """Demonstrate memory integration capabilities."""
    print("\n" + "=" * 60)
    print("Memory Integration Demo")
    print("=" * 60)
    
    cognitive_reasoner = CognitiveReasoner(atomspace)
    memory_integration = OpenCogMemoryIntegration(atomspace, cognitive_reasoner)
    
    # Store some knowledge
    print("\n1. Storing knowledge...")
    algorithm_atom = memory_integration.store_knowledge(
        "Algorithm",
        "A step-by-step procedure for solving a problem",
        TruthValue(0.9, 0.8)
    )
    
    sorting_atom = memory_integration.store_knowledge(
        "Sorting",
        "Process of arranging data in a particular order",
        TruthValue(0.8, 0.7)
    )
    
    # Create relationship
    atomspace.add_atom(
        AtomType.INHERITANCE_LINK,
        outgoing=[sorting_atom, algorithm_atom],
        truth_value=TruthValue(0.9, 0.8)
    )
    
    # Retrieve knowledge
    print("\n2. Retrieving knowledge...")
    algorithm_info = memory_integration.retrieve_knowledge("Algorithm")
    if algorithm_info:
        print(f"Algorithm: {algorithm_info['description']}")
        print(f"Truth value: {algorithm_info['truth_value']}")
        print(f"Related concepts: {algorithm_info['related_concepts']}")
    
    # Get knowledge summary
    print("\n3. Knowledge summary:")
    summary = memory_integration.get_knowledge_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main():
    """Run the complete OpenCoq demo."""
    print("OpenCoq Cognitive Architecture Demo for OpenHands")
    print("Demonstrating advanced reasoning and knowledge representation")
    
    try:
        # Demo AtomSpace
        atomspace = demo_atomspace()
        
        # Demo pattern matching
        demo_pattern_matching(atomspace)
        
        # Demo attention system
        demo_attention_system(atomspace)
        
        # Demo cognitive reasoning
        demo_cognitive_reasoning(atomspace)
        
        # Demo memory integration
        demo_memory_integration(atomspace)
        
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print(f"Final AtomSpace size: {atomspace.size()} atoms")
        
        # Export knowledge for inspection
        knowledge_data = atomspace.export_to_dict()
        print(f"Knowledge exported: {len(knowledge_data['atoms'])} atoms")
        
        print("\nOpenCoq integration successful! 🧠✨")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())