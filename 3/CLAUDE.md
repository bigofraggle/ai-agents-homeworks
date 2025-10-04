# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **ReAct (Reasoning + Acting) agent** implementation using LangGraph for educational purposes. The agent alternates between thinking and acting in a loop until it reaches a final answer.

**Core Pattern**: Agent → Tools → Agent (iterative loop)
- Agent decides to use tools or provide final answer
- Tools execute and return observations
- Loop continues until final answer (max 5 iterations)

## Essential Commands

### Setup
```bash
# Install dependencies
uv sync

# Or install directly
uv pip install -e .
```

### Running
```bash
# Run interactive agent
uv run python agent.py

# Run agent tests with detailed logging
uv run python test_agent.py

# Run memory/chat history tests
uv run python test_memory.py
```

### Environment Configuration
Required API keys in `.env`:
- `ANTHROPIC_API_KEY` - Claude API (required)
- `PINECONE_API_KEY` - Chat history storage (optional, uses mock mode if missing)
- `TAVILY_API_KEY` - Web search (optional)

## Architecture

### Core Components

**agent.py** - ReAct Agent with LangGraph
- `ReActAgent` class - Main agent orchestrator
- `AgentState` - TypedDict defining state flow through graph
- **Two-node workflow**:
  - `agent_node()` - Claude decides next action (uses `.bind_tools()`)
  - `tools_node()` - Executes tool calls, returns `ToolMessage`
- **Key method**: `should_continue()` - Routes based on `tool_calls` presence
- Uses message-based state (`HumanMessage`, `AIMessage`, `ToolMessage`)

**tools.py** - LangChain Tool Definitions
- `WebSearchTool(BaseTool)` - Tavily web search integration
- Each tool has `name`, `description`, `args_schema` (Pydantic model)
- `get_tools()` returns list of available tools
- Tools are automatically bound to LLM via `.bind_tools()`

**memory.py** - Chat History with Pinecone
- `ChatHistoryManager` - Vector storage for conversation history
- **Dual mode**:
  - Production: Uses Pinecone vector database
  - Mock: In-memory storage (auto-enabled if no API key)
- `_create_embedding()` - Simple bag-of-words embeddings (educational)
- `get_relevant_history()` - Retrieves context via cosine similarity

### State Flow

```
AgentState = {
    "input": str,           # User query
    "messages": List,       # Full conversation (HumanMessage, AIMessage, ToolMessage)
    "chat_history": List,   # Retrieved from Pinecone
    "iterations": int,      # Loop counter
    "final_answer": str     # Extracted from last AIMessage
}
```

### LangGraph Workflow

```
agent_node (Claude + tools bound)
    ↓
should_continue() checks tool_calls
    ↓
If tool_calls → tools_node → back to agent_node
If no tool_calls → END (extract final_answer from messages)
```

## Adding New Tools

1. Create tool class in `tools.py`:
```python
class MyToolInput(BaseModel):
    param: str = Field(description="...")

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "What it does"
    args_schema: Type[BaseModel] = MyToolInput

    def _run(self, param: str) -> str:
        # Implementation
        return result
```

2. Add to `get_tools()`:
```python
def get_tools():
    return [WebSearchTool(), MyTool()]
```

The agent will automatically discover and use it via Claude's native tool calling.

## Key Implementation Details

### Tool Calling Pattern
- **Not manual parsing** - Uses Claude's native tool calling API
- Agent returns `AIMessage` with `tool_calls` attribute
- Tools executed via `tool.invoke(args)`
- Results wrapped in `ToolMessage` with `tool_call_id`

### Memory System
- Automatically falls back to mock mode for testing
- Simple word-based embeddings (not production-quality)
- Retrieves top-k similar conversations on first iteration
- Chat history added to message context before first LLM call

### Iteration Control
- `max_iterations=5` prevents infinite loops
- Each agent→tools→agent cycle = 1 iteration
- Final answer extracted from last `AIMessage.content` in messages list

### Logging
- INFO level shows full reasoning trace
- Agent decisions, tool calls, and results all logged
- Use test scripts to see detailed execution flow

## Mock Mode Testing

Memory and tools gracefully degrade:
- No Pinecone key → in-memory storage
- No Tavily key → returns error message (doesn't crash)
- Allows testing without external dependencies

## Common Patterns

### Extracting Final Answer
```python
final_answer = ""
for message in reversed(result["messages"]):
    if isinstance(message, AIMessage) and message.content:
        final_answer = message.content
        break
```

### Checking Tool Calls
```python
if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
    # Route to tools node
```

### Adding History to Context
```python
for msg in history[-4:]:  # Last 4 messages
    if msg["role"] == "user":
        state["messages"].append(HumanMessage(content=msg["content"]))
    else:
        state["messages"].append(AIMessage(content=msg["content"]))
```
