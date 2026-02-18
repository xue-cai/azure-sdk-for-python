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

---

## The Contract Between Agent Code and Azure AI Agent Server

The Azure AI Agent Server defines a **contract** (an interface/protocol) that any agent must satisfy in order to be hosted on Azure AI Foundry as a container. This section explains that contract in detail, then shows how each adapter fulfills it, and how you would implement it directly in Python or TypeScript.

### What the Contract Is

The contract is deliberately simple — **one abstract method**:

```
agent_run(context) → Response | Stream<ResponseStreamEvent>
```

The server handles everything else: HTTP routing, request parsing, SSE framing, health checks, tracing, CORS, error handling. Your agent code only needs to:

1. **Accept** an `AgentRunContext` (which wraps a `CreateResponse` request)
2. **Return** either a complete `Response` object OR an async generator of `ResponseStreamEvent` objects

### Contract Details

#### 1. The HTTP Protocol

The server exposes an OpenAI Responses API-compatible HTTP interface:

```
POST /runs    (or POST /responses)
Content-Type: application/json

{
  "input": [                              // OpenAI-format input items
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "What's the weather?"}
      ]
    }
  ],
  "stream": true,                         // optional: request streaming
  "model": "gpt-4o",                      // optional
  "instructions": "You are helpful.",      // optional: system prompt
  "conversation": {"id": "conv_abc123"},  // optional: conversation thread
  "agent": {                              // optional: agent metadata
    "name": "weather-bot",
    "type": "agent",
    "version": "1.0"
  },
  "temperature": 0.7,                     // optional
  "tools": [...]                          // optional: tool definitions
}
```

**Non-streaming response** (JSON):
```json
{
  "id": "resp_abc123",
  "object": "response",
  "status": "completed",
  "created_at": "2026-02-18T12:00:00Z",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {"type": "output_text", "text": "The weather in Seattle is 15°C and cloudy."}
      ]
    }
  ]
}
```

**Streaming response** (SSE):
```
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"type":"response.created","response":{...}}

data: {"type":"response.in_progress","response":{...}}

data: {"type":"response.output_item.added","output_index":0,"item":{...}}

data: {"type":"response.content_part.added","output_index":0,...}

data: {"type":"response.output_text.delta","delta":"The weather "}

data: {"type":"response.output_text.delta","delta":"in Seattle "}

data: {"type":"response.output_text.delta","delta":"is 15°C and cloudy."}

data: {"type":"response.output_text.done","text":"The weather in Seattle is 15°C and cloudy."}

data: {"type":"response.content_part.done",...}

data: {"type":"response.output_item.done",...}

data: {"type":"response.completed","response":{...}}

data: [DONE]
```

#### 2. The Base Class (`FoundryCBAgent`)

```python
class FoundryCBAgent:
    """Base class for all agent adapters. Provides the HTTP server, 
    streaming support, tracing, and health check endpoints."""

    # ── The one method you MUST implement ──────────────────────────
    @abstractmethod
    async def agent_run(
        self, context: AgentRunContext
    ) -> Union[Response, Generator, AsyncGenerator]:
        """Execute the agent. 
        Return a Response (non-streaming) or yield ResponseStreamEvent (streaming)."""
        raise NotImplementedError

    # ── Optional overrides ─────────────────────────────────────────
    async def agent_liveness(self, request) -> Response:
        """Health check. Default: 200 OK."""
        return Response(status_code=200)

    async def agent_readiness(self, request) -> dict:
        """Readiness probe. Default: {"status": "ready"}."""
        return {"status": "ready"}

    def init_tracing_internal(self, exporter_endpoint=None, app_insights_conn_str=None):
        """Custom tracing setup hook. Called during init_tracing()."""
        pass

    # ── Server lifecycle (provided, not overridden) ────────────────
    def run(self, port: int = 8088) -> None:
        """Start blocking HTTP server (Starlette + Uvicorn)."""

    async def run_async(self, port: int = 8088) -> None:
        """Start async HTTP server in existing event loop."""
```

#### 3. The `AgentRunContext` Object

This is what your `agent_run()` receives — it wraps the deserialized HTTP request:

```python
class AgentRunContext:
    @property
    def raw_payload(self) -> dict:
        """The raw JSON body from the HTTP request."""

    @property
    def request(self) -> CreateResponse:
        """Deserialized request (extends OpenAI ResponseCreateParamsBase).
        Access fields like:
          - request.get("input")          # list of input items
          - request.get("instructions")   # system prompt
          - request.get("model")          # model name
          - request.get("tools")          # tool definitions
          - request.get("temperature")    # sampling temperature
        """

    @property
    def stream(self) -> bool:
        """Whether the client requested streaming."""

    @property
    def response_id(self) -> str:
        """Auto-generated unique response ID (e.g., 'resp_abc123...')."""

    @property
    def conversation_id(self) -> str:
        """Conversation thread ID (from request or auto-generated)."""

    @property
    def id_generator(self) -> IdGenerator:
        """Utility for generating child IDs (message IDs, function call IDs, etc.)."""
```

#### 4. The Streaming Event Sequence

When streaming, your generator must yield events in this order:

```
ResponseCreatedEvent              ─┐
ResponseInProgressEvent            │  "envelope" events
                                  ─┘
  ┌─ Per output item: ────────────────────────────────────────┐
  │ ResponseOutputItemAddedEvent                              │
  │   ┌─ Per content part: ─────────────────────────────┐     │
  │   │ ResponseContentPartAddedEvent                   │     │
  │   │ ResponseTextDeltaEvent (repeated, per chunk)    │     │
  │   │ ResponseTextDoneEvent                           │     │
  │   │ ResponseContentPartDoneEvent                    │     │
  │   └─────────────────────────────────────────────────┘     │
  │ ResponseOutputItemDoneEvent                               │
  └───────────────────────────────────────────────────────────┘

ResponseCompletedEvent            ── final event
```

For tool calls, the inner sequence changes:
```
ResponseOutputItemAddedEvent (type: function_call)
  ResponseFunctionCallArgumentsDeltaEvent (repeated)
  ResponseFunctionCallArgumentsDoneEvent
ResponseOutputItemDoneEvent
```

#### 5. What the Server Handles Automatically

You do **not** need to implement any of these — the base class provides them:

| Concern | How it's handled |
|---------|-----------------|
| **HTTP routing** | Starlette routes: `/runs`, `/responses`, `/liveness`, `/readiness` |
| **Request parsing** | `AgentRunContextMiddleware` deserializes JSON → `AgentRunContext` |
| **SSE framing** | Generator outputs are wrapped in `data: {json}\n\n` + `[DONE]` |
| **Error handling** | Generator init errors → HTTP 500; mid-stream errors → error SSE event + `[DONE]` |
| **Prefetching** | First stream event is pre-fetched to catch early errors before sending 200 |
| **ID generation** | `response_id` and `conversation_id` auto-generated via crypto-secure IDs |
| **Tracing** | OpenTelemetry spans with OTLP or Application Insights exporters |
| **CORS** | Enabled for all origins |
| **Health checks** | `/liveness` (200 OK) and `/readiness` ({"status":"ready"}) |

---

### How Each Adapter Fulfills the Contract

All four adapters implement `agent_run()` with the same pattern: **convert input → run framework → convert output**. Here's how they each do it:

#### AgentFramework Adapter

```python
class AgentFrameworkCBAgent(FoundryCBAgent):
    def __init__(self, agent: AgentProtocol):
        super().__init__()
        self.agent = agent

    async def agent_run(self, context: AgentRunContext):
        # 1. Convert OpenAI input → Agent Framework ChatMessage
        request_input = context.request.get("input")
        message = AgentFrameworkInputConverter().transform_input(request_input)

        if context.stream:
            # 2a. Streaming: wrap agent.run_stream() in event converter
            converter = AgentFrameworkOutputStreamingConverter(context)
            async def stream_updates():
                for ev in converter.initial_events():
                    yield ev
                async for update in self.agent.run_stream(message):
                    for event in converter.transform_output_for_streaming(update):
                        yield event
                for ev in converter.completion_events():
                    yield ev
            return stream_updates()
        else:
            # 2b. Non-streaming: run agent, convert result
            converter = AgentFrameworkOutputNonStreamingConverter(context)
            result = await self.agent.run(message)
            return converter.transform_output_for_response(result)
```

**Key detail**: Uses `asyncio.wait_for()` with configurable idle timeout on streaming — if the agent stops producing updates for N seconds, the stream is gracefully closed.

#### LangGraph Adapter

```python
class LangGraphAdapter(FoundryCBAgent):
    def __init__(self, graph: CompiledStateGraph, state_converter=None):
        super().__init__()
        self.graph = graph
        # Auto-select converter for MessagesState, or require custom one
        if not state_converter:
            if is_state_schema_valid(graph.builder.state_schema):
                self.state_converter = LanggraphMessageStateConverter()
            else:
                raise ValueError("state_converter required for non-MessagesState graph.")
        else:
            self.state_converter = state_converter

    async def agent_run(self, context: AgentRunContext):
        # 1. Convert OpenAI request → LangGraph state dict {"messages": [...]}
        input_data = self.state_converter.request_to_state(context)
        config = RunnableConfig(configurable={"thread_id": context.conversation_id})

        if context.stream:
            # 2a. Streaming via graph.astream()
            stream = self.graph.astream(input=input_data, config=config)
            async for result in self.state_converter.state_to_response_stream(stream, context):
                yield result
        else:
            # 2b. Non-streaming via graph.ainvoke()
            result = await self.graph.ainvoke(input_data, config=config)
            return self.state_converter.state_to_response(result, context)
```

**Key detail**: The `LanggraphStateConverter` abstraction is a strategy pattern — you can provide your own converter for custom graph states (not just `MessagesState`).

#### Claude SDK Adapter

```python
class ClaudeAdapter(FoundryCBAgent):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    async def agent_run(self, context: AgentRunContext):
        # 1. Convert OpenAI input → plain text string
        message = ClaudeInputConverter().transform_input(context.request.get("input"))

        if context.stream:
            converter = ClaudeOutputStreamingConverter(context)
            async def stream_updates():
                for ev in converter.initial_events():
                    yield ev
                await self.agent.send(message)
                async for update in self.agent.receive_response():
                    for event in converter.transform_output_for_streaming(update):
                        yield event
                for ev in converter.completion_events():
                    yield ev
            return stream_updates()
        else:
            converter = ClaudeOutputNonStreamingConverter(context)
            await self.agent.send(message)
            chunks = [chunk async for chunk in self.agent.receive_response()]
            return converter.transform_output_for_response(chunks)
```

**Key detail**: Uses Claude SDK's `send()` / `receive_response()` pattern rather than a single `run()` call.

#### Copilot SDK Adapter

```python
class CopilotSDKAdapter(FoundryCBAgent):
    def __init__(self, handler: Callable, model=None, system_message=None):
        super().__init__()
        self.handler = handler
        self.model = model or "gpt-4o"

    async def agent_run(self, context: AgentRunContext):
        # 1. Convert OpenAI request → plain text prompt (with role prefixes)
        prompt = CopilotSDKRequestConverter(context.request).convert()

        if context.stream:
            converter = CopilotSDKStreamResponseConverter(context)
            for event in converter.initial_events():
                yield event
            # Call the handler (supports both sync and async)
            if asyncio.iscoroutinefunction(self.handler):
                result = await self.handler(prompt)
            else:
                result = self.handler(prompt)
            for event in converter.content_events(result):
                yield event
            for event in converter.completion_events():
                yield event
        else:
            if asyncio.iscoroutinefunction(self.handler):
                result = await self.handler(prompt)
            else:
                result = self.handler(prompt)
            return CopilotSDKResponseConverter(context, result).convert()
```

**Key detail**: The simplest adapter — wraps a plain function, not a framework client. Supports both sync and async handlers.

---

### Implementing the Contract Directly (Without an Adapter)

If your agent doesn't use one of the supported frameworks, you can implement the contract directly against `FoundryCBAgent`.

#### Python — Minimal Custom Agent

From the official core README:

```python
import datetime
from azure.ai.agentserver.core import FoundryCBAgent
from azure.ai.agentserver.core.models import CreateResponse, Response as OpenAIResponse
from azure.ai.agentserver.core.models.projects import (
    ItemContentOutputText,
    ResponsesAssistantMessageItemResource,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)


def stream_events(text: str):
    """Sync generator that yields SSE-compatible stream events."""
    assembled = ""
    for i, token in enumerate(text.split(" ")):
        piece = token if i == len(text.split(" ")) - 1 else token + " "
        assembled += piece
        yield ResponseTextDeltaEvent(delta=piece)
    yield ResponseTextDoneEvent(text=assembled)


async def agent_run(request_body: CreateResponse):
    """The contract: accept a request, return a Response or generator."""
    if request_body.stream:
        return stream_events("I am a streaming agent.")

    return OpenAIResponse(
        metadata={},
        temperature=0.0,
        top_p=0.0,
        user="me",
        id="id",
        created_at=datetime.datetime.now(),
        output=[
            ResponsesAssistantMessageItemResource(
                status="completed",
                content=[
                    ItemContentOutputText(text="I am a non-streaming agent.", annotations=[])
                ],
            )
        ],
    )


my_agent = FoundryCBAgent()
my_agent.agent_run = agent_run

if __name__ == "__main__":
    my_agent.run()  # Starts server on localhost:8088
```

#### Python — Custom Agent with Tool Calls

```python
from azure.ai.agentserver.core import FoundryCBAgent, AgentRunContext
from azure.ai.agentserver.core.models import Response as OpenAIResponse


class WeatherAgent(FoundryCBAgent):
    async def agent_run(self, context: AgentRunContext):
        request = context.request
        input_items = request.get("input", [])

        # Extract user message
        user_text = ""
        for item in input_items:
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content", [])
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "input_text":
                        user_text = part["text"]

        # Simple logic (replace with real agent logic)
        if "weather" in user_text.lower():
            answer = "It's 72°F and sunny in Seattle."
        else:
            answer = f"You said: {user_text}"

        if context.stream:
            return self._stream_response(answer, context)
        else:
            return self._build_response(answer, context)

    async def _stream_response(self, text, context):
        """Yield the full SSE event sequence."""
        from azure.ai.agentserver.core.models.projects import (
            ResponseCreatedEvent, ResponseInProgressEvent,
            ResponseOutputItemAddedEvent, ResponseContentPartAddedEvent,
            ResponseTextDeltaEvent, ResponseTextDoneEvent,
            ResponseContentPartDoneEvent, ResponseOutputItemDoneEvent,
            ResponseCompletedEvent,
        )
        # Envelope
        yield ResponseCreatedEvent(response={"id": context.response_id, "status": "in_progress"})
        yield ResponseInProgressEvent(response={"id": context.response_id})
        # Content
        yield ResponseOutputItemAddedEvent(output_index=0, item={"type": "message", "role": "assistant"})
        yield ResponseContentPartAddedEvent(output_index=0, content_index=0, part={"type": "output_text"})
        # Stream word-by-word
        for word in text.split(" "):
            yield ResponseTextDeltaEvent(delta=word + " ")
        yield ResponseTextDoneEvent(text=text)
        yield ResponseContentPartDoneEvent(output_index=0, content_index=0)
        yield ResponseOutputItemDoneEvent(output_index=0)
        yield ResponseCompletedEvent(response={"id": context.response_id, "status": "completed"})

    def _build_response(self, text, context):
        import datetime
        from azure.ai.agentserver.core.models.projects import (
            ItemContentOutputText, ResponsesAssistantMessageItemResource,
        )
        return OpenAIResponse(
            id=context.response_id,
            created_at=datetime.datetime.now(),
            metadata={}, temperature=0.0, top_p=0.0, user="",
            output=[
                ResponsesAssistantMessageItemResource(
                    status="completed",
                    content=[ItemContentOutputText(text=text, annotations=[])],
                )
            ],
        )


if __name__ == "__main__":
    WeatherAgent().run()
```

#### TypeScript — How the Contract Would Look

> **Note**: As of February 2026, Azure AI Agent Server adapters exist only for Python. There is no published TypeScript SDK. However, the underlying contract is **HTTP + JSON** — any language can implement it. Here's how a TypeScript implementation would look, following the same architectural patterns:

```typescript
// ── The Contract (what you'd implement) ──────────────────────────

import { createServer, IncomingMessage, ServerResponse } from "http";

// Mirrors the Python CreateResponse
interface CreateResponse {
  input?: InputItem[];
  stream?: boolean;
  instructions?: string;
  model?: string;
  temperature?: number;
  conversation?: { id: string };
  agent?: { name: string; type: string; version: string };
  tools?: ToolDefinition[];
}

interface InputItem {
  type: string;
  role: string;
  content: ContentPart[];
}

interface ContentPart {
  type: string;      // "input_text", "input_image", etc.
  text?: string;
}

// Mirrors the Python Response
interface AgentResponse {
  id: string;
  object: "response";
  status: "completed" | "failed";
  created_at: string;
  output: OutputItem[];
}

interface OutputItem {
  type: "message";
  role: "assistant";
  status: "completed";
  content: { type: "output_text"; text: string }[];
}

// Mirrors ResponseStreamEvent
interface StreamEvent {
  type: string;
  [key: string]: unknown;
}

// ── The "FoundryCBAgent" equivalent ──────────────────────────────

abstract class FoundryCBAgent {
  /**
   * The ONE method you must implement.
   * Return a Response object (non-streaming) or an AsyncGenerator of events (streaming).
   */
  abstract agentRun(
    context: AgentRunContext
  ): Promise<AgentResponse | AsyncGenerator<StreamEvent>>;

  /** Start the HTTP server (equivalent to Python's .run()) */
  run(port: number = 8088): void {
    const server = createServer(async (req, res) => {
      if (req.method === "GET" && req.url === "/liveness") {
        res.writeHead(200);
        res.end();
        return;
      }
      if (req.method === "GET" && req.url === "/readiness") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ status: "ready" }));
        return;
      }
      if (req.method === "POST" && (req.url === "/runs" || req.url === "/responses")) {
        const body = await readBody(req);
        const payload: CreateResponse = JSON.parse(body);
        const context = new AgentRunContext(payload);

        try {
          const result = await this.agentRun(context);

          if (isAsyncGenerator(result)) {
            // SSE streaming
            res.writeHead(200, {
              "Content-Type": "text/event-stream",
              "Cache-Control": "no-cache",
              Connection: "keep-alive",
            });
            for await (const event of result) {
              res.write(`data: ${JSON.stringify(event)}\n\n`);
            }
            res.write("data: [DONE]\n\n");
            res.end();
          } else {
            // JSON response
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify(result));
          }
        } catch (err) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: String(err) }));
        }
        return;
      }
      res.writeHead(404);
      res.end();
    });

    server.listen(port, () => console.log(`Agent server listening on :${port}`));
  }
}

class AgentRunContext {
  readonly request: CreateResponse;
  readonly stream: boolean;
  readonly responseId: string;
  readonly conversationId: string;

  constructor(payload: CreateResponse) {
    this.request = payload;
    this.stream = payload.stream ?? false;
    this.responseId = `resp_${crypto.randomUUID()}`;
    this.conversationId = payload.conversation?.id ?? `conv_${crypto.randomUUID()}`;
  }
}
```

#### TypeScript — Example: LangGraph.js Adapter

```typescript
import { CompiledStateGraph } from "@langchain/langgraph";
import { HumanMessage, AIMessageChunk } from "@langchain/core/messages";

class LangGraphAdapter extends FoundryCBAgent {
  constructor(private graph: CompiledStateGraph) {
    super();
  }

  async agentRun(
    context: AgentRunContext
  ): Promise<AgentResponse | AsyncGenerator<StreamEvent>> {
    // 1. Convert OpenAI input → LangGraph messages
    const messages = this.convertInput(context.request);
    const config = { configurable: { thread_id: context.conversationId } };

    if (context.stream) {
      return this.streamResponse(messages, config, context);
    }

    // 2. Non-streaming: invoke and convert
    const result = await this.graph.invoke({ messages }, config);
    const lastMessage = result.messages[result.messages.length - 1];
    return {
      id: context.responseId,
      object: "response",
      status: "completed",
      created_at: new Date().toISOString(),
      output: [{
        type: "message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: lastMessage.content as string }],
      }],
    };
  }

  private async *streamResponse(
    messages: HumanMessage[],
    config: object,
    context: AgentRunContext
  ): AsyncGenerator<StreamEvent> {
    yield { type: "response.created", response: { id: context.responseId, status: "in_progress" } };
    yield { type: "response.in_progress", response: { id: context.responseId } };
    yield { type: "response.output_item.added", output_index: 0,
            item: { type: "message", role: "assistant" } };
    yield { type: "response.content_part.added", output_index: 0,
            content_index: 0, part: { type: "output_text" } };

    let fullText = "";
    for await (const event of this.graph.streamEvents({ messages }, { ...config, version: "v2" })) {
      if (event.event === "on_chat_model_stream") {
        const chunk = event.data.chunk as AIMessageChunk;
        const delta = chunk.content as string;
        if (delta) {
          fullText += delta;
          yield { type: "response.output_text.delta", delta };
        }
      }
    }

    yield { type: "response.output_text.done", text: fullText };
    yield { type: "response.content_part.done", output_index: 0, content_index: 0 };
    yield { type: "response.output_item.done", output_index: 0 };
    yield { type: "response.completed", response: { id: context.responseId, status: "completed" } };
  }

  private convertInput(request: CreateResponse): HumanMessage[] {
    const items = request.input ?? [];
    return items
      .filter((item) => item.role === "user")
      .map((item) => {
        const text = item.content
          .filter((c) => c.type === "input_text")
          .map((c) => c.text)
          .join("\n");
        return new HumanMessage(text);
      });
  }
}

// Usage:
// import { createReactAgent } from "@langchain/langgraph/prebuilt";
// const graph = createReactAgent({ llm: model, tools: [myTool] });
// new LangGraphAdapter(graph).run(8088);
```

#### TypeScript — Example: Simple Function Handler (Copilot SDK Style)

```typescript
/**
 * The simplest possible adapter — wraps an async function.
 * Equivalent to the Python CopilotSDKAdapter.
 */
class FunctionAdapter extends FoundryCBAgent {
  constructor(private handler: (prompt: string) => Promise<string>) {
    super();
  }

  async agentRun(
    context: AgentRunContext
  ): Promise<AgentResponse | AsyncGenerator<StreamEvent>> {
    // Extract text from input items
    const prompt = (context.request.input ?? [])
      .flatMap((item) => item.content ?? [])
      .filter((c) => c.type === "input_text")
      .map((c) => c.text)
      .join("\n");

    const result = await this.handler(prompt);

    if (context.stream) {
      return this.streamResult(result, context);
    }

    return {
      id: context.responseId,
      object: "response",
      status: "completed",
      created_at: new Date().toISOString(),
      output: [{
        type: "message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: result }],
      }],
    };
  }

  private async *streamResult(text: string, ctx: AgentRunContext): AsyncGenerator<StreamEvent> {
    yield { type: "response.created", response: { id: ctx.responseId, status: "in_progress" } };
    yield { type: "response.in_progress", response: { id: ctx.responseId } };
    yield { type: "response.output_item.added", output_index: 0,
            item: { type: "message", role: "assistant" } };
    yield { type: "response.content_part.added", output_index: 0,
            content_index: 0, part: { type: "output_text" } };

    // Stream word by word
    for (const word of text.split(" ")) {
      yield { type: "response.output_text.delta", delta: word + " " };
    }

    yield { type: "response.output_text.done", text };
    yield { type: "response.content_part.done", output_index: 0, content_index: 0 };
    yield { type: "response.output_item.done", output_index: 0 };
    yield { type: "response.completed", response: { id: ctx.responseId, status: "completed" } };
  }
}

// Usage:
// new FunctionAdapter(async (prompt) => `Echo: ${prompt}`).run(8088);
```

---

### Cross-Framework Comparison: The Same Agent in 4 Adapters

To make the contract concrete, here's how the **same simple agent** ("echo back the user's message") looks with each adapter:

#### Using AgentFramework Adapter
```python
from agent_framework import SimpleAgent
from azure.ai.agentserver.agentframework import from_agent_framework

agent = SimpleAgent(instructions="Echo back the user's message exactly.")
from_agent_framework(agent).run()
```

#### Using LangGraph Adapter
```python
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from azure.ai.agentserver.langgraph import from_langgraph

model = AzureChatOpenAI(model="gpt-4o")
graph = create_react_agent(model, tools=[])
from_langgraph(graph).run()
```

#### Using Claude SDK Adapter
```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from azure.ai.agentserver.claude import from_claude

client = ClaudeSDKClient()
await client.start(prompt="", options=ClaudeAgentOptions(
    system_prompt="Echo back the user's message exactly."
))
from_claude(client).run()
```

#### Using Copilot SDK Adapter
```python
from azure.ai.agentserver.copilotsdk import from_copilot_sdk

async def echo(prompt: str) -> str:
    return prompt

from_copilot_sdk(echo).run()
```

#### Using Core Directly (No Adapter)
```python
import datetime
from azure.ai.agentserver.core import FoundryCBAgent, AgentRunContext
from azure.ai.agentserver.core.models import Response as OpenAIResponse
from azure.ai.agentserver.core.models.projects import (
    ItemContentOutputText, ResponsesAssistantMessageItemResource,
)

class EchoAgent(FoundryCBAgent):
    async def agent_run(self, context: AgentRunContext):
        text = ""
        for item in (context.request.get("input") or []):
            for part in (item.get("content") or []):
                if part.get("type") == "input_text":
                    text += part["text"]
        return OpenAIResponse(
            id=context.response_id,
            created_at=datetime.datetime.now(),
            metadata={}, temperature=0.0, top_p=0.0, user="",
            output=[ResponsesAssistantMessageItemResource(
                status="completed",
                content=[ItemContentOutputText(text=text, annotations=[])],
            )],
        )

EchoAgent().run()
```

All five produce the **exact same HTTP API** — a client sending `POST /runs` with `{"input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]}` gets back the same OpenAI-compatible response structure from each.

---

## Summary

All four adapters serve the same purpose: **host any agent framework behind Azure AI Foundry's OpenAI-compatible Responses API**. They share the `FoundryCBAgent` base class and the factory function pattern. The key differentiators are:

1. **Input/output fidelity**: LangGraph and AgentFramework preserve rich message structures; Claude and Copilot flatten to strings
2. **Tool support**: Each uses its framework's native tool system (Agent Framework functions, LangChain tools, MCP, handler functions)
3. **Streaming sophistication**: AgentFramework and LangGraph use state machines; Claude and Copilot use linear event sequences
4. **Extensibility**: LangGraph's `LanggraphStateConverter` abstraction allows custom graph states; others are more fixed
