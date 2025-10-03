"""
OpenCoq Agent for OpenHands framework.

This agent integrates OpenCoq cognitive architecture with OpenHands,
providing advanced reasoning, memory, and knowledge representation capabilities.
"""

import json
import os
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam
    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.agenthub.codeact_agent.tools.bash import create_cmd_run_tool
from openhands.agenthub.codeact_agent.tools.browser import BrowserTool
from openhands.agenthub.codeact_agent.tools.finish import FinishTool
from openhands.agenthub.codeact_agent.tools.ipython import IPythonTool
from openhands.agenthub.codeact_agent.tools.str_replace_editor import create_str_replace_editor_tool
from openhands.agenthub.codeact_agent.tools.think import ThinkTool

# OpenCoq specific tools
from .tools.knowledge_query import KnowledgeQueryTool
from .tools.reasoning import ReasoningTool
from .tools.attention import AttentionTool

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.event import Event
from openhands.llm.llm_utils import check_tools
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager

# OpenCoq cognitive components
from .atomspace import AtomSpace, AtomType, Atom, TruthValue, AttentionValue
from .reasoning import (
    CognitiveReasoner, 
    PatternMatcher, 
    AttentionBroker, 
    Pattern,
    inheritance_rule,
    similarity_rule
)


class OpenCogAgentConfig(AgentConfig):
    """Configuration for OpenCoQ Agent with cognitive parameters."""
    
    # Cognitive system parameters
    attention_threshold: float = 0.1
    max_attention_atoms: int = 20
    reasoning_iterations: int = 5
    knowledge_decay_rate: float = 0.95
    inference_confidence_threshold: float = 0.7
    
    # Memory parameters
    atomspace_persistence: bool = True
    max_atomspace_size: int = 10000
    
    # Integration parameters
    use_cognitive_tools: bool = True
    cognitive_reasoning_weight: float = 0.3


class OpenCogAgent(Agent):
    """
    OpenCoq Agent integrating cognitive architecture with OpenHands.
    
    This agent enhances the standard agent capabilities with:
    - Knowledge representation using AtomSpace
    - Cognitive reasoning and inference
    - Attention-based resource allocation
    - Pattern matching and analogical reasoning
    """
    
    VERSION = '1.0'
    config_model = OpenCogAgentConfig
    
    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry):
        super().__init__(config, llm_registry)
        
        # Initialize cognitive components
        self.atomspace = AtomSpace()
        self.cognitive_reasoner = CognitiveReasoner(self.atomspace)
        self.pattern_matcher = PatternMatcher(self.atomspace)
        self.attention_broker = AttentionBroker(self.atomspace)
        
        # Configure cognitive parameters
        if isinstance(config, OpenCogAgentConfig):
            self.attention_broker.attention_threshold = config.attention_threshold
            self.attention_broker.max_attention = config.max_attention_atoms
            self.attention_broker.decay_rate = config.knowledge_decay_rate
            self.max_reasoning_iterations = config.reasoning_iterations
            self.inference_threshold = config.inference_confidence_threshold
            self.use_cognitive_tools = config.use_cognitive_tools
            self.cognitive_weight = config.cognitive_reasoning_weight
        else:
            # Default values for non-OpenCoq configs
            self.max_reasoning_iterations = 5
            self.inference_threshold = 0.7
            self.use_cognitive_tools = True
            self.cognitive_weight = 0.3
        
        # Add default inference rules
        self.cognitive_reasoner.add_inference_rule(inheritance_rule)
        self.cognitive_reasoner.add_inference_rule(similarity_rule)
        
        # Initialize knowledge base with common concepts
        self._initialize_knowledge_base()
        
        # Action queue for cognitive processing
        self.pending_actions = deque()
        
        # Cognitive state tracking
        self.current_goal: Optional[Atom] = None
        self.reasoning_history: List[Dict] = []
    
    def _initialize_knowledge_base(self):
        """Initialize AtomSpace with common programming and system concepts."""
        
        # Programming concepts
        programming = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE, 
            "Programming",
            truth_value=TruthValue(1.0, 0.9)
        )
        
        code = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE, 
            "Code",
            truth_value=TruthValue(1.0, 0.9)
        )
        
        debugging = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE, 
            "Debugging",
            truth_value=TruthValue(1.0, 0.8)
        )
        
        # Create relationships
        self.atomspace.add_atom(
            AtomType.INHERITANCE_LINK,
            outgoing=[code, programming],
            truth_value=TruthValue(0.9, 0.8)
        )
        
        self.atomspace.add_atom(
            AtomType.INHERITANCE_LINK,
            outgoing=[debugging, programming],
            truth_value=TruthValue(0.8, 0.8)
        )
        
        # System concepts
        system = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "System",
            truth_value=TruthValue(1.0, 0.9)
        )
        
        file_system = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "FileSystem",
            truth_value=TruthValue(1.0, 0.8)
        )
        
        self.atomspace.add_atom(
            AtomType.INHERITANCE_LINK,
            outgoing=[file_system, system],
            truth_value=TruthValue(0.9, 0.8)
        )
        
        # Task concepts
        task = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "Task",
            truth_value=TruthValue(1.0, 0.9)
        )
        
        problem_solving = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            "ProblemSolving",
            truth_value=TruthValue(1.0, 0.8)
        )
        
        self.atomspace.add_atom(
            AtomType.SIMILARITY_LINK,
            outgoing=[task, problem_solving],
            truth_value=TruthValue(0.8, 0.7)
        )
    
    def _get_tools(self) -> List['ChatCompletionToolParam']:
        """Get available tools including cognitive tools."""
        tools = []
        
        # Standard tools
        tools.append(create_cmd_run_tool(self.llm))
        tools.append(IPythonTool.get_tool())
        tools.append(create_str_replace_editor_tool(self.llm))
        tools.append(ThinkTool.get_tool())
        tools.append(FinishTool.get_tool())
        
        # Add browser tool if available
        try:
            tools.append(BrowserTool.get_tool())
        except Exception:
            pass
        
        # Add cognitive tools if enabled
        if self.use_cognitive_tools:
            tools.append(KnowledgeQueryTool.get_tool())
            tools.append(ReasoningTool.get_tool())
            tools.append(AttentionTool.get_tool())
        
        return tools
    
    def step(self, state: State) -> 'Action':
        """
        Enhanced step method with cognitive processing.
        """
        # Process current state with cognitive systems
        self._process_state_cognitively(state)
        
        # Apply attention decay
        self.attention_broker.decay_attention()
        
        # Run cognitive reasoning if we have a goal
        if self.current_goal:
            self._cognitive_reasoning_step()
        
        # Get messages and generate response
        messages = self._get_messages_for_llm(state)
        
        # Generate response with cognitive influence
        response = self._generate_cognitive_response(messages)
        
        # Parse response to actions
        actions = self.response_to_actions(response)
        
        # Store actions and return first one
        for action in actions:
            self.pending_actions.append(action)
        
        return self.pending_actions.popleft() if self.pending_actions else AgentFinishAction()
    
    def _process_state_cognitively(self, state: State):
        """Process current state information into the AtomSpace."""
        
        # Extract task information
        if hasattr(state, 'task') and state.task:
            task_description = str(state.task)
            
            # Create task atom
            task_atom = self.atomspace.add_atom(
                AtomType.CONCEPT_NODE,
                f"CurrentTask:{task_description[:50]}",
                truth_value=TruthValue(0.9, 0.8)
            )
            
            # Set as current goal
            self.current_goal = task_atom
            self.attention_broker.stimulate_atom(task_atom, 10.0)
        
        # Process recent events
        if hasattr(state, 'history'):
            for event in state.history[-5:]:  # Last 5 events
                self._process_event_cognitively(event)
    
    def _process_event_cognitively(self, event: Event):
        """Process an event into cognitive knowledge."""
        
        event_type = type(event).__name__
        
        # Create event concept
        event_atom = self.atomspace.add_atom(
            AtomType.CONCEPT_NODE,
            f"Event:{event_type}",
            truth_value=TruthValue(0.8, 0.7)
        )
        
        # Connect to current goal if available
        if self.current_goal:
            self.atomspace.add_atom(
                AtomType.EVALUATION_LINK,
                outgoing=[
                    self.atomspace.add_atom(AtomType.PREDICATE_NODE, "relatedTo"),
                    self.atomspace.add_atom(
                        AtomType.LIST_LINK,
                        outgoing=[event_atom, self.current_goal]
                    )
                ],
                truth_value=TruthValue(0.7, 0.6)
            )
        
        # Give some attention to the event
        self.attention_broker.stimulate_atom(event_atom, 2.0)
    
    def _cognitive_reasoning_step(self):
        """Perform one step of cognitive reasoning."""
        
        # Get atoms in attention focus
        focused_atoms = self.attention_broker.get_attentional_focus()
        
        if not focused_atoms:
            return
        
        # Run inference on focused atoms
        new_atoms = self.cognitive_reasoner.infer(max_iterations=self.max_reasoning_iterations)
        
        # Record reasoning step
        reasoning_step = {
            "focused_atoms": [str(atom) for atom in focused_atoms],
            "new_inferences": [str(atom) for atom in new_atoms],
            "atomspace_size": self.atomspace.size()
        }
        self.reasoning_history.append(reasoning_step)
        
        # Limit history size
        if len(self.reasoning_history) > 20:
            self.reasoning_history = self.reasoning_history[-20:]
    
    def _get_messages_for_llm(self, state: State) -> List[Message]:
        """Generate messages for the LLM including cognitive insights."""
        
        # Get base messages
        messages = self._get_base_messages(state)
        
        # Add cognitive context if available
        if self.current_goal:
            cognitive_context = self._generate_cognitive_context()
            if cognitive_context:
                cognitive_message = Message(
                    role="system",
                    content=f"Cognitive Context:\n{cognitive_context}"
                )
                messages.insert(-1, cognitive_message)  # Insert before last user message
        
        return messages
    
    def _get_base_messages(self, state: State) -> List[Message]:
        """Get base messages (simplified version)."""
        messages = []
        
        # System message
        system_msg = self.get_system_message()
        if system_msg:
            messages.append(Message(role="system", content=system_msg.content))
        
        # Add history
        if hasattr(state, 'history'):
            for event in state.history[-10:]:  # Last 10 events
                if hasattr(event, 'message'):
                    messages.append(Message(role="user", content=str(event.message)))
        
        return messages
    
    def _generate_cognitive_context(self) -> str:
        """Generate cognitive context for the LLM."""
        context_parts = []
        
        # Current goal information
        if self.current_goal:
            context_parts.append(f"Current Goal: {self.current_goal.name}")
            
            # Find related concepts
            related = self.cognitive_reasoner.find_analogies(self.current_goal, limit=3)
            if related:
                related_names = [atom.name for atom, _ in related]
                context_parts.append(f"Related Concepts: {', '.join(related_names)}")
        
        # Attention focus
        focused = self.attention_broker.get_attentional_focus(limit=5)
        if focused:
            focused_names = [atom.name for atom in focused if atom.name]
            if focused_names:
                context_parts.append(f"Current Focus: {', '.join(focused_names)}")
        
        # Recent reasoning
        if self.reasoning_history:
            last_reasoning = self.reasoning_history[-1]
            if last_reasoning["new_inferences"]:
                context_parts.append(f"Recent Insights: {len(last_reasoning['new_inferences'])} new connections found")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _generate_cognitive_response(self, messages: List[Message]) -> 'ModelResponse':
        """Generate response with cognitive influence."""
        
        # Standard LLM response
        prompt = self._create_prompt_from_messages(messages)
        tools = self._get_tools()
        
        response = self.llm.completion(
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],
            tools=tools
        )
        
        return response
    
    def _create_prompt_from_messages(self, messages: List[Message]) -> str:
        """Create prompt from messages."""
        return "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
    
    def response_to_actions(self, response: 'ModelResponse') -> List['Action']:
        """Convert LLM response to actions with cognitive enhancement."""
        
        # Use standard response parsing
        actions = codeact_function_calling.response_to_actions(response)
        
        # Enhance actions with cognitive insights
        for action in actions:
            self._enhance_action_cognitively(action)
        
        return actions
    
    def _enhance_action_cognitively(self, action: 'Action'):
        """Enhance action with cognitive insights."""
        
        # Add cognitive metadata
        if hasattr(action, 'metadata'):
            if not action.metadata:
                action.metadata = {}
            
            # Add attention information
            focused_atoms = self.attention_broker.get_attentional_focus(limit=3)
            if focused_atoms:
                action.metadata['cognitive_focus'] = [atom.name for atom in focused_atoms]
            
            # Add reasoning confidence
            if self.current_goal:
                truth_value = self.cognitive_reasoner.evaluate_truth(self.current_goal)
                action.metadata['goal_confidence'] = truth_value
    
    def get_system_message(self) -> Optional[MessageAction]:
        """Get system message with cognitive enhancement."""
        
        base_prompt = """You are OpenCoq Agent, an advanced AI assistant that combines traditional reasoning with cognitive architecture capabilities.

You have access to an AtomSpace for knowledge representation and can perform sophisticated reasoning, pattern matching, and attention-based processing.

Your cognitive capabilities include:
- Knowledge representation using symbolic atoms
- Pattern matching and analogical reasoning  
- Attention-based resource allocation
- Inference and rule-based reasoning
- Learning from interactions

Use these cognitive capabilities to provide more insightful and contextually aware responses. When solving problems, consider both logical reasoning and cognitive patterns that might be relevant.

Available tools include standard programming tools plus cognitive tools for knowledge querying, reasoning, and attention management."""

        return MessageAction(content=base_prompt)
    
    def reset(self) -> None:
        """Reset agent state including cognitive components."""
        super().reset()
        
        # Reset cognitive state
        self.current_goal = None
        self.reasoning_history.clear()
        self.pending_actions.clear()
        
        # Clear attention but keep knowledge base
        for atom in self.atomspace._atoms.values():
            atom.attention_value = AttentionValue()
    
    def export_knowledge(self) -> Dict:
        """Export current knowledge state."""
        return {
            "atomspace": self.atomspace.export_to_dict(),
            "reasoning_history": self.reasoning_history,
            "current_goal": self.current_goal.handle if self.current_goal else None
        }
    
    def import_knowledge(self, knowledge_data: Dict):
        """Import knowledge state."""
        if "atomspace" in knowledge_data:
            self.atomspace.import_from_dict(knowledge_data["atomspace"])
        
        if "reasoning_history" in knowledge_data:
            self.reasoning_history = knowledge_data["reasoning_history"]
        
        if "current_goal" in knowledge_data and knowledge_data["current_goal"]:
            self.current_goal = self.atomspace.get_atom(knowledge_data["current_goal"])