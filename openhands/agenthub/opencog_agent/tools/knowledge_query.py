"""
Knowledge Query Tool for OpenCoq Agent.

Provides access to AtomSpace knowledge querying capabilities.
"""

from typing import Dict, Any, List, Optional
from litellm import ChatCompletionToolParam

from openhands.events.action import ToolCallAction
from openhands.events.observation import ToolCallObservation


class KnowledgeQueryTool:
    """Tool for querying the AtomSpace knowledge base."""
    
    @staticmethod
    def get_tool() -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": "query_knowledge",
                "description": "Query the cognitive knowledge base (AtomSpace) for information about concepts, relationships, and patterns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["concept", "relationship", "pattern", "analogies", "focus"],
                            "description": "Type of query to perform"
                        },
                        "query_term": {
                            "type": "string", 
                            "description": "The term or concept to query for"
                        },
                        "atom_type": {
                            "type": "string",
                            "enum": ["ConceptNode", "PredicateNode", "InheritanceLink", "SimilarityLink", "EvaluationLink"],
                            "description": "Specific atom type to search for (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum number of results to return"
                        }
                    },
                    "required": ["query_type", "query_term"]
                }
            }
        }
    
    @staticmethod
    def execute_query(
        atomspace,
        pattern_matcher,
        attention_broker,
        cognitive_reasoner,
        **kwargs
    ) -> ToolCallObservation:
        """Execute a knowledge query."""
        
        query_type = kwargs.get("query_type")
        query_term = kwargs.get("query_term")
        atom_type = kwargs.get("atom_type")
        limit = kwargs.get("limit", 10)
        
        try:
            results = []
            
            if query_type == "concept":
                # Search for concepts by name
                atoms = atomspace.get_atoms_by_name(query_term)
                for atom in atoms[:limit]:
                    results.append({
                        "atom": str(atom),
                        "type": atom.atom_type.value,
                        "truth_value": atom.truth_value.to_dict(),
                        "attention": atom.attention_value.to_dict()
                    })
            
            elif query_type == "relationship":
                # Find relationships involving the query term
                concept_atoms = atomspace.get_atoms_by_name(query_term)
                for concept in concept_atoms:
                    # Find incoming links (relationships where this atom is involved)
                    for related_atom in concept.incoming:
                        if len(results) >= limit:
                            break
                        results.append({
                            "relationship": str(related_atom),
                            "type": related_atom.atom_type.value,
                            "involves": [str(a) for a in related_atom.outgoing],
                            "truth_value": related_atom.truth_value.to_dict()
                        })
            
            elif query_type == "analogies":
                # Find analogous concepts
                concept_atoms = atomspace.get_atoms_by_name(query_term)
                if concept_atoms:
                    source_atom = concept_atoms[0]
                    analogies = cognitive_reasoner.find_analogies(source_atom, limit=limit)
                    for analog_atom, similarity in analogies:
                        results.append({
                            "analog": str(analog_atom),
                            "similarity_score": similarity,
                            "type": analog_atom.atom_type.value
                        })
            
            elif query_type == "focus":
                # Get current attentional focus
                focused_atoms = attention_broker.get_attentional_focus(limit=limit)
                for atom in focused_atoms:
                    if query_term.lower() in atom.name.lower():
                        results.append({
                            "atom": str(atom),
                            "attention_value": atom.attention_value.sti,
                            "type": atom.atom_type.value
                        })
            
            elif query_type == "pattern":
                # Pattern matching (simplified)
                from ..reasoning import Pattern
                from ..atomspace import AtomType
                
                # Create a simple pattern for the query term
                if atom_type:
                    pattern_type = AtomType(atom_type)
                else:
                    pattern_type = None
                
                pattern = Pattern(
                    atom_type=pattern_type,
                    name_pattern=f".*{query_term}.*"
                )
                
                matches = pattern_matcher.find_matches(pattern)
                for match in matches[:limit]:
                    results.append({
                        "match": str(match),
                        "type": match.atom_type.value,
                        "truth_value": match.truth_value.to_dict()
                    })
            
            return ToolCallObservation(
                content=f"Knowledge query results:\n" + 
                       f"Query: {query_type} for '{query_term}'\n" +
                       f"Results ({len(results)}):\n" +
                       "\n".join([f"- {result}" for result in results]),
                tool_call_metadata={"query_type": query_type, "results_count": len(results)}
            )
            
        except Exception as e:
            return ToolCallObservation(
                content=f"Error executing knowledge query: {str(e)}",
                tool_call_metadata={"error": str(e)}
            )