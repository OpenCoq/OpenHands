"""
Reasoning Tool for OpenCoq Agent.

Provides access to cognitive reasoning capabilities including inference,
truth evaluation, and goal-directed reasoning.
"""

from typing import Dict, Any, List, Optional
from litellm import ChatCompletionToolParam

from openhands.events.action import ToolCallAction
from openhands.events.observation import ToolCallObservation


class ReasoningTool:
    """Tool for cognitive reasoning operations."""
    
    @staticmethod
    def get_tool() -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": "cognitive_reasoning",
                "description": "Perform cognitive reasoning operations including inference, truth evaluation, and goal-directed reasoning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["infer", "evaluate_truth", "reason_about_goal", "find_paths", "explain"],
                            "description": "Type of reasoning operation to perform"
                        },
                        "target": {
                            "type": "string",
                            "description": "Target concept, statement, or goal for reasoning"
                        },
                        "max_iterations": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of reasoning iterations"
                        },
                        "confidence_threshold": {
                            "type": "number",
                            "default": 0.7,
                            "description": "Minimum confidence threshold for results"
                        }
                    },
                    "required": ["operation", "target"]
                }
            }
        }
    
    @staticmethod
    def execute_reasoning(
        atomspace,
        cognitive_reasoner,
        attention_broker,
        **kwargs
    ) -> ToolCallObservation:
        """Execute a reasoning operation."""
        
        operation = kwargs.get("operation")
        target = kwargs.get("target")
        max_iterations = kwargs.get("max_iterations", 5)
        confidence_threshold = kwargs.get("confidence_threshold", 0.7)
        
        try:
            results = []
            
            if operation == "infer":
                # Run inference process
                new_atoms = cognitive_reasoner.infer(max_iterations=max_iterations)
                
                # Filter by confidence threshold
                high_confidence_atoms = []
                for atom in new_atoms:
                    if atom.truth_value.confidence >= confidence_threshold:
                        high_confidence_atoms.append(atom)
                
                results = [{
                    "inference": str(atom),
                    "type": atom.atom_type.value,
                    "truth_value": atom.truth_value.to_dict()
                } for atom in high_confidence_atoms]
                
                content = f"Inference Results:\n" + \
                         f"Generated {len(new_atoms)} new inferences, " + \
                         f"{len(high_confidence_atoms)} with high confidence:\n" + \
                         "\n".join([f"- {result['inference']}" for result in results])
            
            elif operation == "evaluate_truth":
                # Find atoms related to target and evaluate truth
                target_atoms = atomspace.get_atoms_by_name(target)
                if not target_atoms:
                    # Create a concept atom for the target
                    from ..atomspace import AtomType, TruthValue
                    target_atom = atomspace.add_atom(
                        AtomType.CONCEPT_NODE,
                        target,
                        truth_value=TruthValue(0.5, 0.5)  # Neutral initial value
                    )
                    target_atoms = [target_atom]
                
                for atom in target_atoms:
                    truth_score = cognitive_reasoner.evaluate_truth(atom)
                    results.append({
                        "statement": str(atom),
                        "truth_score": truth_score,
                        "confidence": atom.truth_value.confidence,
                        "evaluation": "likely_true" if truth_score > 0.7 else "likely_false" if truth_score < 0.3 else "uncertain"
                    })
                
                content = f"Truth Evaluation for '{target}':\n" + \
                         "\n".join([
                             f"- {result['statement']}: {result['evaluation']} (score: {result['truth_score']:.3f})"
                             for result in results
                         ])
            
            elif operation == "reason_about_goal":
                # Perform goal-directed reasoning
                related_atoms = cognitive_reasoner.reason_about_goal(target)
                
                results = [{
                    "concept": str(atom),
                    "type": atom.atom_type.value,
                    "relevance": atom.attention_value.sti,
                    "truth_value": atom.truth_value.to_dict()
                } for atom in related_atoms]
                
                # Sort by relevance (attention)
                results.sort(key=lambda x: x["relevance"], reverse=True)
                
                content = f"Goal Reasoning for '{target}':\n" + \
                         f"Found {len(related_atoms)} related concepts:\n" + \
                         "\n".join([
                             f"- {result['concept']} (relevance: {result['relevance']:.2f})"
                             for result in results[:10]  # Top 10
                         ])
            
            elif operation == "find_paths":
                # Find reasoning paths between concepts
                target_parts = target.split(" to ")
                if len(target_parts) != 2:
                    raise ValueError("Target must be in format 'concept1 to concept2'")
                
                source_name, dest_name = [part.strip() for part in target_parts]
                
                # Find source and destination atoms
                source_atoms = atomspace.get_atoms_by_name(source_name)
                dest_atoms = atomspace.get_atoms_by_name(dest_name)
                
                if not source_atoms or not dest_atoms:
                    content = f"Could not find concepts '{source_name}' or '{dest_name}' in knowledge base"
                else:
                    # Simple path finding through shared connections
                    source_atom = source_atoms[0]
                    dest_atom = dest_atoms[0]
                    
                    # Find common atoms in their incoming/outgoing sets
                    source_connected = set(source_atom.incoming) | set(source_atom.outgoing)
                    dest_connected = set(dest_atom.incoming) | set(dest_atom.outgoing)
                    
                    common_atoms = source_connected & dest_connected
                    
                    paths = []
                    for common in common_atoms:
                        paths.append({
                            "path": f"{source_name} -> {common.name} -> {dest_name}",
                            "strength": min(
                                common.truth_value.strength,
                                common.truth_value.confidence
                            )
                        })
                    
                    paths.sort(key=lambda x: x["strength"], reverse=True)
                    results = paths[:5]  # Top 5 paths
                    
                    content = f"Reasoning Paths from '{source_name}' to '{dest_name}':\n" + \
                             "\n".join([
                                 f"- {result['path']} (strength: {result['strength']:.3f})"
                                 for result in results
                             ])
            
            elif operation == "explain":
                # Explain why something might be true or false
                target_atoms = atomspace.get_atoms_by_name(target)
                
                if not target_atoms:
                    content = f"No knowledge found about '{target}'"
                else:
                    atom = target_atoms[0]
                    
                    # Find supporting evidence
                    evidence = []
                    
                    # Check inheritance relationships
                    for related in atom.incoming:
                        if related.atom_type.value == "InheritanceLink":
                            evidence.append({
                                "type": "inheritance",
                                "evidence": str(related),
                                "strength": related.truth_value.strength
                            })
                    
                    # Check similarity relationships
                    for related in atom.incoming:
                        if related.atom_type.value == "SimilarityLink":
                            evidence.append({
                                "type": "similarity", 
                                "evidence": str(related),
                                "strength": related.truth_value.strength
                            })
                    
                    evidence.sort(key=lambda x: x["strength"], reverse=True)
                    results = evidence[:5]  # Top 5 pieces of evidence
                    
                    content = f"Explanation for '{target}':\n"
                    if results:
                        content += "Supporting evidence:\n" + \
                                  "\n".join([
                                      f"- {result['type']}: {result['evidence']} (strength: {result['strength']:.3f})"
                                      for result in results
                                  ])
                    else:
                        content += "No strong supporting evidence found in knowledge base."
            
            return ToolCallObservation(
                content=content,
                tool_call_metadata={
                    "operation": operation,
                    "target": target,
                    "results_count": len(results)
                }
            )
            
        except Exception as e:
            return ToolCallObservation(
                content=f"Error in cognitive reasoning: {str(e)}",
                tool_call_metadata={"error": str(e)}
            )