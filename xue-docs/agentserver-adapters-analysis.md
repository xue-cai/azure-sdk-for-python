# Azure AI Agent Server — Adapter Analysis

## What Are These Adapters?

The **Azure AI Agent Server** system provides a way to host AI agents built with **any framework** behind a **unified OpenAI-compatible Responses API** (`POST /runs` or `POST /responses`). The architecture consists of:

- **`azure-ai-agentserver-core`** — The shared foundation: a Starlette/Uvicorn web server that exposes the OpenAI Responses API, handles streaming (SSE), tracing (OpenTelemetry), health checks, and defines the abstract `FoundryCBAgent` base class.
- **Framework adapters** — Each adapter wraps a specific agent framework's native objects and converts requests/responses between OpenAI format and the framework's native format.

### The Core Pattern

Every adapter follows the same 3-step lifecycle:

```
OpenAI Responses API Request (CreateResponse)
        │
        ▼
┌─────────────────────┐
│  Input Converter     │  ← Transforms OpenAI messages → framework-native input
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Framework Agent     │  ← Runs the actual agent (Claude SDK, LangGraph, etc.)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Output Converter    │  ← Transforms framework output → OpenAI Response / SSE stream
└─────────────────────┘
```

All adapters:
1. Extend `FoundryCBAgent` and implement `agent_run(context: AgentRunContext)`
2. Expose a factory function (`from_xxx(agent)`) as the public API
3. Support both **streaming** and **non-streaming** modes
4. Return `OpenAIResponse` (non-streaming) or `AsyncGenerator[ResponseStreamEvent]` (streaming)

---

## Core Package: `azure-ai-agentserver-core`

### `FoundryCBAgent` (Abstract Base)

```python
class FoundryCBAgent:
    """Base class for all agent adapters."""

    @abstractmethod
    async def agent_run(self, context: AgentRunContext) -> Union[Response, AsyncGenerator[ResponseStreamEvent, Any]]:
        """Execute the agent. Return a complete Response or stream events."""
        ...

    def run(self, port=8088):
        """Start blocking HTTP server (Starlette + Uvicorn)."""

    async def run_async(self, port=8088):
        """Start async HTTP server."""

    def init_tracing(self):
        """Setup OpenTelemetry (OTLP + Application Insights)."""
```

### `AgentRunContext`

Wraps the incoming request and provides:
- `context.request` → deserialized `CreateResponse` object
- `context.stream` → whether streaming was requested
- `context.conversation_id` → conversation thread ID
- `context.response_id` → unique response ID

### Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /runs` | Execute agent (main) |
| `POST /responses` | Execute agent (alias) |
| `GET /liveness` | Health check |
| `GET /readiness` | Readiness probe |

---

## Adapter 1: AgentFramework (`azure-ai-agentserver-agentframework`)

**Wraps:** Microsoft Agent Framework (`AgentProtocol` instances)
**Factory:** `from_agent_framework(agent)`
**Dependencies:** `agent-framework-azure-ai`, `agent-framework-core`

### Input Converter

```python
class AgentFrameworkInputConverter:
    """
    OpenAI messages → Agent Framework input types.
    
    Accepts: str | List[Dict] | None
    Returns: str | ChatMessage | list[ChatMessage] | None
    """
    def transform_input(self, input):
        # Handles implicit user messages: {"content": "hello"}
        # Handles explicit typed messages: {"type": "message", "role": "user", "content": [...]}
        # Extracts "input_text" content items
```

### Output Converters

**Non-streaming:** Converts `AgentRunResponse` → `OpenAIResponse`
- Maps `TextContent` → message item with `output_text`
- Maps `FunctionCallContent` → `function_call` item
- Maps `FunctionResultContent` → `function_call_output` item

**Streaming:** Uses a **state machine** with 3 streaming states:
```
_TextContentStreamingState          → text delta events
_FunctionCallStreamingState         → function call argument deltas
_FunctionCallOutputStreamingState   → function result events
```

### Unique Features
- **Richest content support**: Handles text, function calls, function results, errors, approval requests
- **Idle timeout**: Configurable via `AGENTS_ADAPTER_STREAM_TIMEOUT_S` env var (default: 300s)
- **Agent ID generation**: `AgentIdGenerator` builds IDs from context
- **Sync + async samples**: Both blocking and async usage patterns

### Sample

```python
from azure.ai.agentserver.agentframework import from_agent_framework
from agent_framework.azure import AzureOpenAIChatClient

agent = AzureOpenAIChatClient(credential=DefaultAzureCredential()).create_agent(
    instructions="You are a helpful weather agent.",
    tools=get_weather,
)
from_agent_framework(agent).run()  # Hosts on localhost:8088
```

---

## Adapter 2: LangGraph (`azure-ai-agentserver-langgraph`)

**Wraps:** LangGraph `CompiledStateGraph` instances
**Factory:** `from_langgraph(agent, state_converter=None)`
**Dependencies:** `langchain`, `langchain-openai`, `langchain-azure-ai`, `langgraph`

### Input Conversion (2-layer)

**Layer 1 — State Converter** (strategy pattern):
```python
class LanggraphStateConverter(ABC):
    """Abstract: converts between API requests and LangGraph state dicts."""
    @abstractmethod
    def request_to_state(self, context) -> Dict[str, Any]  # → {"messages": [...]}
    @abstractmethod
    def state_to_response(self, state, context) -> Response
    @abstractmethod
    async def state_to_response_stream(self, stream_state, context) -> AsyncGenerator

class LanggraphMessageStateConverter(LanggraphStateConverter):
    """Default for graphs using MessagesState. Auto-selected."""
```

**Layer 2 — Request Converter**:
```python
class LangGraphRequestConverter:
    """CreateResponse → LangGraph state dict ({"messages": [...]})"""
    # Handles: text, images, audio, files, function calls, tool outputs
    # Maps to LangChain message types: HumanMessage, SystemMessage, AIMessage, ToolMessage
```

### Output Converters

**Non-streaming:** `LangGraphResponseConverter` maps LangGraph state → Response items
- `AIMessage` → assistant message (text, tool_calls)
- `ToolMessage` → function_call_output item
- Handles multimodal content (text/image/audio/file)

**Streaming:** `LangGraphStreamResponseConverter` uses `ResponseEventGenerator` state machines per message type

### Unique Features
- **State converter abstraction**: Supports custom graph states beyond `MessagesState`
- **Multimodal support**: Text, images, audio, files
- **Checkpointing**: Samples show Redis and in-memory checkpointers
- **MCP integration**: Samples for MCP tool servers
- **Most samples** (7): react agent, calculator, RAG, custom state, MCP, Redis checkpointer

### Sample

```python
from azure.ai.agentserver.langgraph import from_langgraph
from langgraph.prebuilt import create_react_agent

model = AzureChatOpenAI(model="gpt-4o")
agent = create_react_agent(model, [get_word_length, calculator], MemorySaver())
from_langgraph(agent).run()
```

---

## Adapter 3: Claude SDK (`azure-ai-agentserver-claude`)

**Wraps:** Claude Agent SDK `ClaudeSDKClient` instances
**Factory:** `from_claude(agent)`
**Dependencies:** `claude-agent-sdk`

### Input Converter

```python
class ClaudeInputConverter:
    """
    OpenAI messages → plain text string for Claude.
    
    Accepts: str | List[Dict] | None
    Returns: str
    """
    # Flattens all message types into a single prompt string
    # Handles nested content arrays with "input_text" items
```

### Output Converters

**Non-streaming:** `ClaudeOutputNonStreamingConverter`
- Collects all Claude response chunks → single `OpenAIResponse`
- Builds assistant message with combined text

**Streaming:** `ClaudeOutputStreamingConverter`
- Emits the full SSE event sequence:
```
ResponseCreatedEvent → ResponseInProgressEvent →
  OutputItemAdded → ContentPartAdded →
    TextDelta (per chunk) →
  TextDone → ContentPartDone → OutputItemDone →
ResponseCompletedEvent
```

### Unique Features
- **MCP tool integration**: Tools defined via `@tool` decorator and registered on MCP servers
- **Tool naming convention**: `mcp__<server>__<tool>` (e.g., `mcp__calculator__add`)
- **Simplest input converter**: All inputs become a single string (Claude's native input format)
- **Calculator sample**: Demonstrates MCP-based tool calling

### Sample

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from azure.ai.agentserver.claude import from_claude

client = ClaudeSDKClient()
options = ClaudeAgentOptions(
    system_prompt="You are a helpful assistant.",
    mcp_servers={"calc": calc_server},
    allowed_tools=["mcp__calc__add"]
)
await client.start(prompt="", options=options)
from_claude(client).run()
```

---

## Adapter 4: Copilot SDK (`azure-ai-agentserver-copilotsdk`)

**Wraps:** A user-provided async/sync handler function
**Factory:** `from_copilot_sdk(handler, model=None, system_message=None)`
**Dependencies:** `github-copilot-sdk`

### Input Converter

```python
class CopilotSDKRequestConverter:
    """
    CreateResponse → plain text prompt.
    
    Extracts instructions + input, formats with role prefixes:
      "User: ...", "Assistant: ...", "System: ..."
    """
```

### Output Converters

**Non-streaming:** `CopilotSDKResponseConverter`
- Wraps handler return value as `ResponsesAssistantMessageItemResource`
- Adds metadata, timestamps, completion status

**Streaming:** `CopilotSDKStreamResponseConverter`
- Full SSE event sequence with sequence numbering
- Same event pattern as Claude adapter

### Unique Features
- **Handler-based**: Wraps a plain function, not a framework client object
- **Configurable model/system message** at adapter level
- **Simplest integration**: Just provide an async function that takes a prompt and returns text

### Sample

```python
from azure.ai.agentserver.copilotsdk import from_copilot_sdk

async def my_handler(prompt: str) -> str:
    return f"Echo: {prompt}"

from_copilot_sdk(my_handler).run()  # Hosts on localhost:8088
```

---

## Comparative Analysis

### Architecture Comparison

| Aspect | AgentFramework | LangGraph | Claude SDK | Copilot SDK |
|--------|---------------|-----------|------------|-------------|
| **Wraps** | `AgentProtocol` | `CompiledStateGraph` | `ClaudeSDKClient` | `Callable` (function) |
| **Factory** | `from_agent_framework(agent)` | `from_langgraph(agent, state_converter?)` | `from_claude(agent)` | `from_copilot_sdk(handler, model?, system_message?)` |
| **Input format** | `str \| ChatMessage \| list` | `{"messages": [LangChain msgs]}` | `str` (plain text) | `str` (role-prefixed) |
| **Output types** | Text, FunctionCall, FunctionResult | Text, ToolCall, ToolMessage, multimodal | Text only | Text only |
| **Streaming** | State machine (3 states) | Event generators per message | Sequential events | Sequential events |
| **Tool support** | Native function tools | LangChain `@tool` | MCP protocol | Via handler |
| **State mgmt** | Via framework | Checkpointers (Redis, memory) | Via Claude SDK | None (stateless) |
| **Tracing** | OTLP gRPC | OTLP HTTP + Azure AI | OTLP HTTP | None built-in |
| **Complexity** | ★★★★ High | ★★★★★ Highest | ★★★ Medium | ★★ Lowest |
| **Samples** | 5 | 7 | 2 | 1 |

### Input Conversion Complexity

```
              Simple                                              Complex
   ┌────────────┼──────────────┼──────────────┼──────────────────┐
   │            │              │              │                  │
Copilot SDK  Claude SDK  AgentFramework    LangGraph
(str prompt)  (str prompt)  (ChatMessage)   (LangChain msgs +
                                             multimodal +
                                             state dict)
```

- **Copilot SDK / Claude SDK**: Flatten everything to a string
- **AgentFramework**: Preserves message structure as `ChatMessage` objects
- **LangGraph**: Full LangChain message hierarchy with multimodal support (images, audio, files)

### Output Conversion Complexity

```
              Simple                                              Complex
   ┌────────────┼──────────────┼──────────────┼──────────────────┐
   │            │              │              │                  │
Copilot SDK  Claude SDK  LangGraph       AgentFramework
(text only)  (text only)  (text + tools   (text + function calls +
                           + multimodal)   function results +
                                           errors + approvals)
```

- **Copilot SDK / Claude SDK**: Only produce text output items
- **LangGraph**: Text + tool calls + multimodal content
- **AgentFramework**: Richest — text, function calls, function results, errors, approval requests

### Streaming Implementation Styles

1. **AgentFramework**: Stateful streaming with `_TextContentStreamingState`, `_FunctionCallStreamingState`, `_FunctionCallOutputStreamingState` — each manages its own buffer and event lifecycle
2. **LangGraph**: `ResponseEventGenerator` hierarchy with per-message-type generators
3. **Claude SDK**: Linear event emission — initial events → text deltas → completion events
4. **Copilot SDK**: Same linear pattern as Claude

### When to Use Each

| Use Case | Recommended Adapter |
|----------|-------------------|
| Microsoft Agent Framework agents with tool calling | **AgentFramework** |
| LangChain/LangGraph agents, RAG, complex workflows | **LangGraph** |
| Claude-powered agents with MCP tools | **Claude SDK** |
| Simple function-based agents, quick prototyping | **Copilot SDK** |
| Multimodal (images, audio, files) | **LangGraph** |
| Custom state management | **LangGraph** (state_converter) |
| Production tracing/observability | **AgentFramework** or **LangGraph** |

---

## Summary

All four adapters serve the same purpose: **host any agent framework behind Azure AI Foundry's OpenAI-compatible Responses API**. They share the `FoundryCBAgent` base class and the factory function pattern. The key differentiators are:

1. **Input/output fidelity**: LangGraph and AgentFramework preserve rich message structures; Claude and Copilot flatten to strings
2. **Tool support**: Each uses its framework's native tool system (Agent Framework functions, LangChain tools, MCP, handler functions)
3. **Streaming sophistication**: AgentFramework and LangGraph use state machines; Claude and Copilot use linear event sequences
4. **Extensibility**: LangGraph's `LanggraphStateConverter` abstraction allows custom graph states; others are more fixed
