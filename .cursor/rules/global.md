# JarvisKit Python SDK - Cursor Rules

## Project Overview
This is a Python SDK for JarvisKit, a framework for building LangGraph-based agents with integrations for [JarvisKit Runtime](https://github.com/jarviskit/runtime).

## Code Style & Standards

### Python Version & Dependencies
- Use Python 3.12+ (as specified in pyproject.toml)
- Use `uv` as the package manager (not pip)
- Follow type hints consistently using `typing` module
- Use `dataclasses` for configuration classes
- Use `TypedDict` for state management in LangGraph agents

### Import Organization
```python
# Standard library imports first
import os
import ssl
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast
from enum import Enum

# Third-party imports
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer, Command

# Local imports last
from src import JarvisKitToolNode, JarvisKitRuntimeManager
from src.classes import MessageEvent, SocketConfig, RabbitMQConfig
```

## Architecture Patterns

### Agent Structure
- Always use `StateGraph` from LangGraph for agent workflows
- Define state using `TypedDict` with clear type annotations
- Use `Command` for flow control between nodes
- Always include a checkpointer parameter in `get_graph()` functions

### Example Agent Pattern:
```python
class State(TypedDict):
    current_page: str | None
    protected_key: str | None = "protected_key"

async def agent(state: State, config: RunnableConfig) -> Command[Literal['__end__', 'tool_execution_handler']]:
    agent_runtime = JarvisKitRuntimeManager().get_runtime()
    thread_id = config.get("configurable", {}).get("thread_id", "")
    
    # Agent logic here
    
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        return Command(goto="tool_execution_handler")
    else:
        return Command(goto="__end__")

def get_graph(checkpointer: Checkpointer):
    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    workflow.add_node("tool_execution_handler", JarvisKitToolNode([your_tools]))
    
    workflow.add_edge("tool_execution_handler", "agent")
    workflow.set_entry_point("agent")
    
    return workflow.compile(checkpointer=checkpointer, name="agent_name")
```

### Configuration Management
- Use `@dataclass` for configuration classes
- Always include `from_env()` class method for environment-based configuration
- Always include `validate()` method for configuration validation
- Use `Environment` enum for environment specification

### Runtime Management
- Always use `JarvisKitRuntimeManager().get_runtime()` for runtime access
- Save LLM responses using `agent_runtime.put_store_message(thread_id, response)`
- Access message history using `agent_runtime.get_messages(thread_id)`

### Tool Development
- Use `@tool` decorator from `langchain_core.tools`
- Always include detailed docstrings for tools
- Use `JarvisKitToolNode` for tool execution in graphs
- For client tool calls, extend appropriate base classes

### Error Handling
- Always wrap agent invocations in try-catch blocks
- Use descriptive error messages
- Return boolean success indicators from message handlers
- Log errors appropriately using print statements (as per project pattern)

## File Organization

### Directory Structure
```
src/
├── __init__.py           # Public API exports
├── classes.py            # Configuration and data classes
├── jarvis_runtime.py     # Runtime management
├── callback_handler.py   # LangGraph callback handling
├── tool_node_wrapper.py  # Tool execution wrapper
├── runtime_manager.py    # Runtime lifecycle management
├── rabbit.py            # RabbitMQ integration
└── agui_util.py         # AGUI protocol utilities

examples/
└── {agent_name}/
    ├── __init__.py
    └── graph.py          # Agent graph definition
```

### Module Exports
- Always update `__all__` in `__init__.py` when adding new public classes/functions
- Keep public API minimal and well-documented
- Use consistent naming: `JarvisKit` prefix for main classes

## Environment Variables

### Required Variables
- `OPENAI_API_KEY`: OpenAI API key
- `JARVIS_KIT_RUNTIME`: Runtime endpoint URL
- `JARVIS_KIT_NAMESPACE`: Namespace identifier
- `JARVIS_KIT_NAMESPACE_SECRET`: Namespace API key
- `RABBITMQ_CONNECTION_STRING`: RabbitMQ connection string
- `POSTGRES_CONNECTION_STRING`: PostgreSQL connection string (for checkpointing)

### Optional Variables
- `DEBUG`: Enable debug mode (default: false)
- `ENVIRONMENT`: Environment (development/staging/production)
- `MAX_CONCURRENT_WORKERS`: Max concurrent workers (default: 2)
- `SOCKET_TIMEOUT`: Socket timeout (default: 30)

## Testing Patterns

### Test File Structure
- Use descriptive test file names like `test_sync.py`, `test_with_postgres_checkpointer.py`
- Always load environment variables with `load_dotenv()`
- Use `MemorySaver` for simple testing, PostgreSQL checkpointer for persistence testing
- Include bootstrap functions for test setup

### Agent Testing
```python
async def bootstrap():
    checkpointer = MemorySaver()
    socket_config = SocketConfig.from_env()
    
    # Initialize runtime
    await JarvisKitRuntimeManager().init_runtime(InitRuntimeParams(
        namespace=os.getenv('JARVIS_KIT_NAMESPACE'),
        namespace_secret=os.getenv('JARVIS_KIT_NAMESPACE_SECRET'),
        agents=[get_agent_graph(checkpointer)],
        socket_config=socket_config,
        rabbitmq_config=RabbitMQConfig.from_env()
    ))
```

## Best Practices

### Async/Await Usage
- Use `async/await` for all I/O operations
- Use `ainvoke` for LLM calls
- Use `await` for agent graph execution

### Type Safety
- Always use type hints
- Use `cast()` for type assertions when necessary
- Prefer `TypedDict` over regular dicts for structured data
- Use `Literal` types for string enums in function signatures

### Message Handling
- Always implement the `default_message_handler` pattern
- Use proper thread ID extraction from message events
- Include proper error handling in message handlers

### Configuration
- Always validate configuration on initialization
- Use environment variables for runtime configuration
- Support both environment and file-based configuration
- Include sensible defaults for optional settings

## Performance Considerations

- Use connection pooling for database/message queue connections
- Implement proper retry logic for network operations
- Use appropriate prefetch counts for RabbitMQ
- Consider memory usage with large message histories

## Security Guidelines

- Never log sensitive information (API keys, secrets)
- Use proper SSL/TLS configuration for production
- Validate all inputs from external sources
- Use environment variables for sensitive configuration

## Documentation

- Include docstrings for all public classes and methods
- Document configuration options and environment variables
- Provide working examples for common use cases
- Keep README.md updated with setup instructions