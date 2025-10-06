# OpenCoq Integration for OpenHands

This document describes the OpenCoq (OpenCog) cognitive architecture integration with the OpenHands framework.

## Overview

OpenCoq brings advanced cognitive capabilities to OpenHands through:

- **Symbolic Knowledge Representation**: AtomSpace for storing concepts and relationships
- **Cognitive Reasoning**: Pattern matching, inference, and analogical reasoning
- **Attention Management**: Dynamic focus allocation for important information
- **Memory Integration**: Seamless knowledge persistence and retrieval
- **Cognitive Tools**: Specialized tools for knowledge querying and reasoning

## Quick Start

### 1. Enable OpenCoq in Configuration

Add to your `config.toml`:

```toml
# Enable OpenCoq cognitive features
opencog_enabled = true
opencog_attention_threshold = 0.1
opencog_max_atomspace_size = 10000
opencog_reasoning_iterations = 5

# Use OpenCoq Agent
default_agent = "OpenCogAgent"

[agents.opencog]
agent_name = "OpenCogAgent"
attention_threshold = 0.1
reasoning_iterations = 5
use_cognitive_tools = true
```

### 2. Basic Usage

```python
from openhands.agenthub.opencog_agent import OpenCogAgent, OpenCogAgentConfig

# Create cognitive agent
config = OpenCogAgentConfig(
    attention_threshold=0.1,
    reasoning_iterations=5,
    use_cognitive_tools=True
)

agent = OpenCogAgent(config, llm_registry)
```

### 3. Test the Integration

Run the integration test:

```bash
cd openhands/agenthub/opencog_agent
python integration_test.py
```

## Architecture

```
OpenCoq Agent
├── AtomSpace              # Knowledge storage with atoms and links
├── CognitiveReasoner     # Inference engine with logical reasoning
├── PatternMatcher        # Query engine for pattern matching
├── AttentionBroker       # Resource management and focus control
├── Cognitive Tools       # User interface for cognitive operations
└── Memory Integration    # Bridge to OpenHands memory system
```

## Key Components

### AtomSpace
- **Atoms**: Basic knowledge units (concepts, predicates, schemas)
- **Links**: Relationships between atoms (inheritance, similarity, evaluation)
- **Truth Values**: Strength and confidence for uncertain knowledge
- **Attention Values**: Dynamic importance and focus allocation

### Reasoning Engine
- **Inference Rules**: Logical reasoning (inheritance, similarity)
- **Pattern Matching**: Complex query capabilities with variable binding
- **Analogical Reasoning**: Finding structural similarities between concepts
- **Goal-Directed Reasoning**: Focus on task-relevant knowledge

### Cognitive Tools
- **Knowledge Query**: Search AtomSpace for concepts and relationships
- **Reasoning Tool**: Perform inference, truth evaluation, path finding
- **Attention Tool**: Manage cognitive focus and stimulation

## Examples

### Store Knowledge
```python
# Create programming concepts
programming = agent.atomspace.add_atom(
    AtomType.CONCEPT_NODE,
    "Programming",
    truth_value=TruthValue(1.0, 0.9)
)

python = agent.atomspace.add_atom(
    AtomType.CONCEPT_NODE,
    "Python",
    truth_value=TruthValue(0.9, 0.8)
)

# Create relationship
agent.atomspace.add_atom(
    AtomType.INHERITANCE_LINK,
    outgoing=[python, programming],
    truth_value=TruthValue(0.9, 0.8)
)
```

### Query Knowledge
```python
# Find programming-related concepts
pattern = Pattern(
    atom_type=AtomType.CONCEPT_NODE,
    name_pattern=".*Program.*"
)
matches = agent.pattern_matcher.find_matches(pattern)
```

### Cognitive Reasoning
```python
# Reason about a goal
goal_concepts = agent.cognitive_reasoner.reason_about_goal(
    "implement sorting algorithm"
)

# Find analogies
analogies = agent.cognitive_reasoner.find_analogies(
    python_atom, limit=5
)

# Run inference
new_knowledge = agent.cognitive_reasoner.infer(max_iterations=10)
```

### Attention Management
```python
# Focus on important concepts
agent.attention_broker.stimulate_atom(important_concept, 10.0)

# Get current focus
focused_atoms = agent.attention_broker.get_attentional_focus()
```

## Integration Benefits

1. **Enhanced Problem Solving**: Logical reasoning and knowledge-based decisions
2. **Persistent Learning**: Accumulated knowledge across sessions
3. **Contextual Understanding**: Rich semantic relationships
4. **Explainable Reasoning**: Traceable decision processes
5. **Adaptive Behavior**: Dynamic attention and focus management

## Files Structure

```
openhands/agenthub/opencog_agent/
├── README.md                    # Detailed component documentation
├── __init__.py                  # Module exports
├── opencog_agent.py            # Main OpenCoq agent implementation
├── atomspace.py                # Knowledge representation system
├── reasoning.py                # Cognitive reasoning and inference
├── memory_integration.py       # Memory system bridge
├── demo.py                     # Usage demonstration
├── integration_test.py         # Comprehensive test suite
└── tools/                      # Cognitive tools
    ├── __init__.py
    ├── knowledge_query.py       # Knowledge querying tool
    ├── reasoning.py            # Reasoning operations tool
    └── attention.py            # Attention management tool
```

## Testing

The implementation includes comprehensive tests:

- `tests/unit/agenthub/test_opencog_agent/test_atomspace.py`
- `tests/unit/agenthub/test_opencog_agent/test_reasoning.py`
- `openhands/agenthub/opencog_agent/integration_test.py`

Run tests with:
```bash
# Unit tests
python -m pytest tests/unit/agenthub/test_opencog_agent/ -v

# Integration test
cd openhands/agenthub/opencog_agent
python integration_test.py
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `opencog_enabled` | `false` | Enable OpenCoq features |
| `opencog_attention_threshold` | `0.1` | Minimum attention for focus |
| `opencog_max_atomspace_size` | `10000` | Maximum atoms in AtomSpace |
| `opencog_reasoning_iterations` | `5` | Max reasoning steps per cycle |
| `attention_threshold` | `0.1` | Agent attention threshold |
| `reasoning_iterations` | `5` | Agent reasoning iterations |
| `use_cognitive_tools` | `true` | Enable cognitive tools |

## Future Extensions

- **Natural Language Processing**: Enhanced text understanding with semantic parsing
- **Machine Learning Integration**: Subsymbolic learning and neural-symbolic fusion  
- **Distributed Reasoning**: Multi-agent cognitive collaboration
- **Temporal Reasoning**: Time-based inference and causal modeling
- **Emotional Intelligence**: Affective reasoning and sentiment integration

## Contributing

The OpenCoq integration is designed to be extensible. Key areas for contribution:

1. **Inference Rules**: Add domain-specific reasoning rules
2. **Cognitive Tools**: Create specialized cognitive interfaces
3. **Integration Adapters**: Connect with other AI systems
4. **Performance Optimization**: Improve reasoning efficiency
5. **Domain Knowledge**: Add pre-built knowledge bases

## References

- [OpenCog Framework](https://opencog.org/)
- [AtomSpace Documentation](https://wiki.opencog.org/w/AtomSpace)
- [Cognitive Architectures](https://en.wikipedia.org/wiki/Cognitive_architecture)
- [Symbolic AI](https://en.wikipedia.org/wiki/Symbolic_artificial_intelligence)

---

**Status**: ✅ Production Ready  
**Version**: 1.0  
**Last Updated**: January 2025  

The OpenCoq integration provides a solid foundation for cognitive AI capabilities in OpenHands, enabling more sophisticated reasoning and knowledge-based problem solving.