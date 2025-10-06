"""
Attention Tool for OpenCoq Agent.

Provides access to attention allocation and focus management capabilities.
"""

from typing import Dict, Any, List, Optional
from litellm import ChatCompletionToolParam

from openhands.events.action import ToolCallAction
from openhands.events.observation import ToolCallObservation


class AttentionTool:
    """Tool for managing cognitive attention and focus."""
    
    @staticmethod
    def get_tool() -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": "manage_attention",
                "description": "Manage cognitive attention including focus, stimulation, and attention allocation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["focus", "stimulate", "decay", "get_focus", "set_threshold"],
                            "description": "Attention management action to perform"
                        },
                        "target": {
                            "type": "string",
                            "description": "Target concept or atom name (required for focus/stimulate actions)"
                        },
                        "amount": {
                            "type": "number",
                            "default": 5.0,
                            "description": "Amount of attention/stimulation to apply"
                        },
                        "threshold": {
                            "type": "number",
                            "description": "New attention threshold (for set_threshold action)"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum number of items to return"
                        }
                    },
                    "required": ["action"]
                }
            }
        }
    
    @staticmethod
    def execute_attention_management(
        atomspace,
        attention_broker,
        **kwargs
    ) -> ToolCallObservation:
        """Execute attention management operation."""
        
        action = kwargs.get("action")
        target = kwargs.get("target")
        amount = kwargs.get("amount", 5.0)
        threshold = kwargs.get("threshold")
        limit = kwargs.get("limit", 10)
        
        try:
            results = []
            
            if action == "focus":
                if not target:
                    raise ValueError("Target required for focus action")
                
                # Find target atoms and stimulate them
                target_atoms = atomspace.get_atoms_by_name(target)
                if not target_atoms:
                    # Create concept if it doesn't exist
                    from ..atomspace import AtomType, TruthValue
                    target_atom = atomspace.add_atom(
                        AtomType.CONCEPT_NODE,
                        target,
                        truth_value=TruthValue(0.8, 0.7)
                    )
                    target_atoms = [target_atom]
                
                for atom in target_atoms:
                    attention_broker.stimulate_atom(atom, amount)
                    results.append({
                        "atom": str(atom),
                        "new_attention": atom.attention_value.sti,
                        "stimulation_amount": amount
                    })
                
                content = f"Focused attention on '{target}':\n" + \
                         "\n".join([
                             f"- {result['atom']}: attention = {result['new_attention']:.2f}"
                             for result in results
                         ])
            
            elif action == "stimulate":
                if not target:
                    raise ValueError("Target required for stimulate action")
                
                # Similar to focus but with explicit stimulation
                target_atoms = atomspace.get_atoms_by_name(target)
                
                if not target_atoms:
                    content = f"No atoms found for '{target}'"
                else:
                    for atom in target_atoms:
                        old_attention = atom.attention_value.sti
                        attention_broker.stimulate_atom(atom, amount)
                        results.append({
                            "atom": str(atom),
                            "old_attention": old_attention,
                            "new_attention": atom.attention_value.sti,
                            "change": atom.attention_value.sti - old_attention
                        })
                    
                    content = f"Stimulated '{target}':\n" + \
                             "\n".join([
                                 f"- {result['atom']}: {result['old_attention']:.2f} -> {result['new_attention']:.2f} " +
                                 f"(+{result['change']:.2f})"
                                 for result in results
                             ])
            
            elif action == "decay":
                # Apply attention decay
                old_focused = attention_broker.get_attentional_focus(limit=limit)
                attention_broker.decay_attention()
                new_focused = attention_broker.get_attentional_focus(limit=limit)
                
                results = [{
                    "old_focus_size": len(old_focused),
                    "new_focus_size": len(new_focused),
                    "decay_rate": attention_broker.decay_rate
                }]
                
                content = f"Applied attention decay:\n" + \
                         f"- Decay rate: {attention_broker.decay_rate}\n" + \
                         f"- Focused atoms: {len(old_focused)} -> {len(new_focused)}"
            
            elif action == "get_focus":
                # Get current attentional focus
                focused_atoms = attention_broker.get_attentional_focus(limit=limit)
                
                results = [{
                    "atom": str(atom),
                    "type": atom.atom_type.value,
                    "sti": atom.attention_value.sti,
                    "lti": atom.attention_value.lti,
                    "total_importance": atom.attention_value.sti + atom.attention_value.lti
                } for atom in focused_atoms]
                
                # Sort by total importance
                results.sort(key=lambda x: x["total_importance"], reverse=True)
                
                content = f"Current Attentional Focus ({len(focused_atoms)} atoms):\n" + \
                         "\n".join([
                             f"- {result['atom']}: STI={result['sti']:.2f}, LTI={result['lti']:.2f}"
                             for result in results
                         ])
            
            elif action == "set_threshold":
                if threshold is None:
                    raise ValueError("Threshold value required for set_threshold action")
                
                old_threshold = attention_broker.attention_threshold
                attention_broker.attention_threshold = threshold
                
                # Get new focus with updated threshold
                focused_atoms = attention_broker.get_attentional_focus(limit=limit)
                
                results = [{
                    "old_threshold": old_threshold,
                    "new_threshold": threshold,
                    "focused_atoms_count": len(focused_atoms)
                }]
                
                content = f"Updated attention threshold:\n" + \
                         f"- Old threshold: {old_threshold}\n" + \
                         f"- New threshold: {threshold}\n" + \
                         f"- Atoms in focus: {len(focused_atoms)}"
            
            # Add attention statistics
            all_atoms = list(atomspace._atoms.values())
            if all_atoms:
                total_attention = sum(atom.attention_value.sti for atom in all_atoms)
                avg_attention = total_attention / len(all_atoms)
                max_attention = max(atom.attention_value.sti for atom in all_atoms)
                
                content += f"\n\nAttention Statistics:\n" + \
                          f"- Total atoms: {len(all_atoms)}\n" + \
                          f"- Average attention: {avg_attention:.3f}\n" + \
                          f"- Maximum attention: {max_attention:.3f}\n" + \
                          f"- Current threshold: {attention_broker.attention_threshold}"
            
            return ToolCallObservation(
                content=content,
                tool_call_metadata={
                    "action": action,
                    "target": target,
                    "results_count": len(results)
                }
            )
            
        except Exception as e:
            return ToolCallObservation(
                content=f"Error in attention management: {str(e)}",
                tool_call_metadata={"error": str(e)}
            )