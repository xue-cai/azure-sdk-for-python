# Azure AI Agent Server — Adapter Analysis

## Table of Contents

1. [What Are These Adapters?](#what-are-these-adapters)
2. [The Contract: `FoundryCBAgent`](#the-contract-foundrycbagent)
3. [The `/responses` API Schema](#the-responses-api-schema)
4. [The Four Adapters](#the-four-adapters)
5. [Deep Dive: Input/Output Conversion](#deep-dive-inputoutput-conversion)
6. [Comparative Analysis](#comparative-analysis)
7. [Implementing the Contract Directly](#implementing-the-contract-directly)
8. [Summary](#summary)

---

## What Are These Adapters?

The **Azure AI Agent Server** hosts AI agents built with **any framework** behind a **unified OpenAI-compatible Responses API** (`POST /runs` or `POST /responses`). The architecture has two layers:

- **`azure-ai-agentserver-core`** — A Starlette/Uvicorn web server that exposes the OpenAI Responses API, handles streaming (SSE), tracing (OpenTelemetry), health checks, and defines the abstract `FoundryCBAgent` base class.
- **Framework adapters** — Each adapter wraps a specific agent framework and converts requests/responses between OpenAI format and the framework's native format.

Every adapter follows the same 3-step lifecycle:

```
OpenAI Responses API Request (CreateResponse)
        │
        ▼
┌─────────────────────┐
│  Input Converter     │  ← OpenAI messages → framework-native input
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Framework Agent     │  ← Runs the actual agent
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Output Converter    │  ← Framework output → OpenAI Response / SSE stream
└─────────────────────┘
```

---

## The Contract: `FoundryCBAgent`

The contract is deliberately simple — **one abstract method**:

```
agent_run(context) → Response | Stream<ResponseStreamEvent>
```

The server handles everything else. Your agent code only needs to:
1. **Accept** an `AgentRunContext` (wraps the deserialized HTTP request)
2. **Return** either a complete `Response` or an async generator of `ResponseStreamEvent` objects

### The Base Class

```python
class FoundryCBAgent:
    """Base class for all agent adapters."""

    # ── The one method you MUST implement ──────────────────────────
    @abstractmethod
    async def agent_run(
        self, context: AgentRunContext
    ) -> Union[Response, Generator, AsyncGenerator]:
        raise NotImplementedError

    # ── Optional overrides ─────────────────────────────────────────
    async def agent_liveness(self, request) -> Response:
        """Health check. Default: 200 OK."""
        return Response(status_code=200)

    async def agent_readiness(self, request) -> dict:
        """Readiness probe. Default: {"status": "ready"}."""
        return {"status": "ready"}

    def init_tracing_internal(self, exporter_endpoint=None, app_insights_conn_str=None):
        """Custom tracing setup hook."""
        pass

    # ── Server lifecycle (provided, not overridden) ────────────────
    def run(self, port: int = 8088) -> None:
        """Start blocking HTTP server (Starlette + Uvicorn)."""

    async def run_async(self, port: int = 8088) -> None:
        """Start async HTTP server in existing event loop."""
```

### `AgentRunContext`

This is what `agent_run()` receives:

```python
class AgentRunContext:
    @property
    def raw_payload(self) -> dict:
        """The raw JSON body from the HTTP request."""

    @property
    def request(self) -> CreateResponse:
        """Deserialized request. Access via request.get("input"), request.get("instructions"), etc."""

    @property
    def stream(self) -> bool:
        """Whether the client requested streaming."""

    @property
    def response_id(self) -> str:
        """Auto-generated unique response ID (e.g., 'resp_Xk9mN2...')."""

    @property
    def conversation_id(self) -> str:
        """Conversation thread ID (from request or auto-generated)."""

    @property
    def id_generator(self) -> IdGenerator:
        """Utility for generating child IDs (message, function call, etc.)."""
```

### What the Server Handles Automatically

| Concern | How it's handled |
|---------|-----------------|
| **HTTP routing** | Starlette routes: `POST /runs`, `POST /responses`, `GET /liveness`, `GET /readiness` |
| **Request parsing** | `AgentRunContextMiddleware` deserializes JSON → `AgentRunContext` |
| **SSE framing** | Generator outputs wrapped in `data: {json}\n\n` + `[DONE]` |
| **Error handling** | Generator init errors → HTTP 500; mid-stream errors → error SSE event + `[DONE]` |
| **Prefetching** | First stream event pre-fetched to catch early errors before sending 200 |
| **ID generation** | `response_id` and `conversation_id` auto-generated via crypto-secure IDs |
| **Tracing** | OpenTelemetry spans with OTLP or Application Insights exporters |
| **CORS** | Enabled for all origins |

---

## The `/responses` API Schema

### Request Schema (`POST /responses`)

```jsonc
{
  // ── Input items (the conversation history) ───────────────────
  "input": [
    {
      "type": "message",                          // "message" | "function_call" | "function_call_output"
      "role": "user",                             // "user" | "assistant" | "system"
      "content": [
        { "type": "input_text", "text": "What's the weather in Seattle?" }
      ]
    },
    { "content": "Follow-up question" },          // Implicit user message (no type/role)
    {                                             // Function call (from previous turn)
      "type": "function_call",
      "call_id": "call_abc123",
      "name": "get_weather",
      "arguments": "{\"location\": \"Seattle\"}"
    },
    {                                             // Function call result
      "type": "function_call_output",
      "call_id": "call_abc123",
      "output": "{\"temp\": 15, \"condition\": \"cloudy\"}"
    }
  ],

  // ── Optional fields ──────────────────────────────────────────
  "instructions": "You are a weather assistant.",
  "model": "gpt-4o",
  "stream": true,
  "temperature": 0.7,
  "top_p": 1.0,
  "tools": [...],
  "conversation": {"id": "conv_abc123"},
  "agent": {"name": "weather-bot", "type": "agent", "version": "1.0"},
  "metadata": {"response_id": "resp_abc123"}
}
```

**Content types** (`ItemContent.type`):

| Input types | Output types |
|-------------|-------------|
| `input_text` | `output_text` |
| `input_image` | `output_audio` |
| `input_audio` | `refusal` |
| `input_file` | |

### Response Schema (Non-Streaming)

```jsonc
{
  "id": "resp_abc123...",
  "object": "response",
  "status": "completed",
  "created_at": 1739884800,
  "agent": {"name": "weather-bot", "type": "agent", "version": "1.0"},
  "conversation": {"id": "conv_abc123..."},
  "metadata": {},
  "temperature": 0.7,
  "top_p": 1.0,
  "output": [
    {                                              // Assistant text message
      "type": "message", "id": "msg_abc123...", "role": "assistant", "status": "completed",
      "content": [
        { "type": "output_text", "text": "The weather in Seattle is 15°C and cloudy.", "annotations": [] }
      ]
    },
    {                                              // Function call (tool use)
      "type": "function_call", "id": "func_abc123...",
      "call_id": "call_xyz789", "name": "get_weather",
      "arguments": "{\"location\": \"Seattle\"}", "status": "completed"
    },
    {                                              // Function result
      "type": "function_call_output", "id": "funcout_abc123...",
      "call_id": "call_xyz789",
      "output": "[{\"type\": \"text\", \"text\": \"15°C, cloudy\"}]", "status": "completed"
    }
  ]
}
```

### Response Schema (Streaming SSE)

Each SSE `data:` line carries a `ResponseStreamEvent`. The full sequence for a **text response**:

```jsonc
// 1. Envelope events
{"type": "response.created", "sequence_number": 1,
 "response": {"id": "resp_...", "status": "in_progress", "created_at": 1739884800}}
{"type": "response.in_progress", "sequence_number": 2,
 "response": {"id": "resp_..."}}

// 2. Output item lifecycle
{"type": "response.output_item.added", "sequence_number": 3, "output_index": 0,
 "item": {"type": "message", "id": "msg_...", "role": "assistant", "status": "in_progress", "content": []}}
{"type": "response.content_part.added", "sequence_number": 4,
 "output_index": 0, "content_index": 0,
 "part": {"type": "output_text", "text": "", "annotations": []}}

// 3. Text deltas (repeated per chunk)
{"type": "response.output_text.delta", "sequence_number": 5, "output_index": 0, "content_index": 0,
 "delta": "The weather "}
{"type": "response.output_text.delta", "sequence_number": 6, "output_index": 0, "content_index": 0,
 "delta": "in Seattle is 15°C and cloudy."}

// 4. Finalization
{"type": "response.output_text.done", "sequence_number": 7, "output_index": 0, "content_index": 0,
 "text": "The weather in Seattle is 15°C and cloudy."}
{"type": "response.content_part.done", "sequence_number": 8, "output_index": 0, "content_index": 0,
 "part": {"type": "output_text", "text": "The weather in Seattle is 15°C and cloudy.", "annotations": []}}
{"type": "response.output_item.done", "sequence_number": 9, "output_index": 0,
 "item": {"type": "message", "id": "msg_...", "role": "assistant", "status": "completed",
          "content": [{"type": "output_text", "text": "The weather in Seattle is 15°C and cloudy.", "annotations": []}]}}
{"type": "response.completed", "sequence_number": 10,
 "response": {"id": "resp_...", "status": "completed"}}
[DONE]
```

For a **function call** stream, the inner events differ:

```jsonc
{"type": "response.output_item.added", "output_index": 0,
 "item": {"type": "function_call", "id": "func_...", "name": "get_weather", "call_id": "call_...", "status": "in_progress"}}
{"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\"location\":"}
{"type": "response.function_call_arguments.delta", "output_index": 0, "delta": " \"Seattle\"}"}
{"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "{\"location\": \"Seattle\"}"}
{"type": "response.output_item.done", "output_index": 0,
 "item": {"type": "function_call", "id": "func_...", "name": "get_weather", "call_id": "call_...",
          "arguments": "{\"location\": \"Seattle\"}", "status": "completed"}}
```

As a diagram:

```
ResponseCreatedEvent              ─┐
ResponseInProgressEvent            │  "envelope" events
                                  ─┘
  ┌─ Per output item: ────────────────────────────────────────┐
  │ ResponseOutputItemAddedEvent                              │
  │   ┌─ Per content part (text): ──────────────────────┐     │
  │   │ ResponseContentPartAddedEvent                   │     │
  │   │ ResponseTextDeltaEvent (repeated, per chunk)    │     │
  │   │ ResponseTextDoneEvent                           │     │
  │   │ ResponseContentPartDoneEvent                    │     │
  │   └─────────────────────────────────────────────────┘     │
  │   ┌─ Per content part (function call): ─────────────┐     │
  │   │ ResponseFunctionCallArgumentsDeltaEvent (×N)    │     │
  │   │ ResponseFunctionCallArgumentsDoneEvent          │     │
  │   └─────────────────────────────────────────────────┘     │
  │ ResponseOutputItemDoneEvent                               │
  └───────────────────────────────────────────────────────────┘
ResponseCompletedEvent            ── final event
```

---

## The Four Adapters

### Adapter 1: AgentFramework (`azure-ai-agentserver-agentframework`)

**Wraps:** Microsoft Agent Framework (`AgentProtocol` instances)
**Factory:** `from_agent_framework(agent)`
**Dependencies:** `agent-framework-azure-ai`, `agent-framework-core`

```python
class AgentFrameworkCBAgent(FoundryCBAgent):
    def __init__(self, agent: AgentProtocol):
        super().__init__()
        self.agent = agent

    async def agent_run(self, context: AgentRunContext):
        request_input = context.request.get("input")
        message = AgentFrameworkInputConverter().transform_input(request_input)

        if context.stream:
            converter = AgentFrameworkOutputStreamingConverter(context)
            async def stream_updates():
                for ev in converter.initial_events():
                    yield ev
                aiter = self.agent.run_stream(message).__aiter__()
                while True:
                    try:
                        update = await asyncio.wait_for(aiter.__anext__(), timeout=timeout_s)
                    except StopAsyncIteration:
                        break
                    for event in converter.transform_output_for_streaming(update):
                        yield event
                for ev in converter.completion_events():
                    yield ev
            return stream_updates()
        else:
            converter = AgentFrameworkOutputNonStreamingConverter(context)
            result = await self.agent.run(message)
            return converter.transform_output_for_response(result)
```

**Unique features:**
- **Richest content support**: Text, function calls, function results, errors, approval requests
- **Idle timeout**: Configurable via `AGENTS_ADAPTER_STREAM_TIMEOUT_S` env var (default: 300s)
- **Streaming state machine**: 3 states (`_TextContentStreamingState`, `_FunctionCallStreamingState`, `_FunctionCallOutputStreamingState`), each managing its own buffer and event lifecycle

**Quick start:**
```python
from azure.ai.agentserver.agentframework import from_agent_framework

agent = AzureOpenAIChatClient(credential=DefaultAzureCredential()).create_agent(
    instructions="You are a helpful weather agent.", tools=get_weather,
)
from_agent_framework(agent).run()  # localhost:8088
```

---

### Adapter 2: LangGraph (`azure-ai-agentserver-langgraph`)

**Wraps:** LangGraph `CompiledStateGraph` instances
**Factory:** `from_langgraph(agent, state_converter=None)`
**Dependencies:** `langchain`, `langchain-openai`, `langchain-azure-ai`, `langgraph`

```python
class LangGraphAdapter(FoundryCBAgent):
    def __init__(self, graph: CompiledStateGraph, state_converter=None):
        super().__init__()
        self.graph = graph
        # Auto-select LanggraphMessageStateConverter for MessagesState graphs
        if not state_converter:
            if is_state_schema_valid(graph.builder.state_schema):
                self.state_converter = LanggraphMessageStateConverter()
            else:
                raise ValueError("state_converter required for non-MessagesState graph.")
        else:
            self.state_converter = state_converter

    async def agent_run(self, context: AgentRunContext):
        input_data = self.state_converter.request_to_state(context)
        config = RunnableConfig(configurable={"thread_id": context.conversation_id})

        if context.stream:
            stream = self.graph.astream(input=input_data, config=config)
            async for result in self.state_converter.state_to_response_stream(stream, context):
                yield result
        else:
            result = await self.graph.ainvoke(input_data, config=config)
            return self.state_converter.state_to_response(result, context)
```

**Two-layer conversion** — the `LanggraphStateConverter` strategy pattern separates state conversion from the adapter:

```python
class LanggraphStateConverter(ABC):
    @abstractmethod
    def request_to_state(self, context) -> Dict[str, Any]      # → {"messages": [...]}
    @abstractmethod
    def state_to_response(self, state, context) -> Response
    @abstractmethod
    async def state_to_response_stream(self, stream, context) -> AsyncGenerator

class LanggraphMessageStateConverter(LanggraphStateConverter):
    """Default for graphs using MessagesState. Auto-selected."""
```

**Unique features:**
- **State converter abstraction**: Supports custom graph states beyond `MessagesState`
- **Multimodal support**: Text, images, audio, files
- **Checkpointing**: Samples with Redis and in-memory checkpointers
- **Most samples** (7): react agent, calculator, RAG, custom state, MCP, Redis checkpointer

**Quick start:**
```python
from azure.ai.agentserver.langgraph import from_langgraph

model = AzureChatOpenAI(model="gpt-4o")
agent = create_react_agent(model, [get_word_length, calculator], MemorySaver())
from_langgraph(agent).run()
```

---

### Adapter 3: Claude SDK (`azure-ai-agentserver-claude`)

**Wraps:** Claude Agent SDK `ClaudeSDKClient` instances
**Factory:** `from_claude(agent)`
**Dependencies:** `claude-agent-sdk`

```python
class ClaudeAdapter(FoundryCBAgent):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    async def agent_run(self, context: AgentRunContext):
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

**Unique features:**
- **MCP tool integration**: Tools defined via `@tool` decorator, registered on MCP servers
- **Tool naming convention**: `mcp__<server>__<tool>` (e.g., `mcp__calculator__add`)
- **send/receive pattern**: Uses `agent.send()` + `agent.receive_response()` rather than a single `run()` call

**Quick start:**
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

### Adapter 4: Copilot SDK (`azure-ai-agentserver-copilotsdk`)

**Wraps:** A user-provided async/sync handler function
**Factory:** `from_copilot_sdk(handler, model=None, system_message=None)`
**Dependencies:** `github-copilot-sdk`

```python
class CopilotSDKAdapter(FoundryCBAgent):
    def __init__(self, handler: Callable, model=None, system_message=None):
        super().__init__()
        self.handler = handler
        self.model = model or "gpt-4o"

    async def agent_run(self, context: AgentRunContext):
        prompt = CopilotSDKRequestConverter(context.request).convert()

        if context.stream:
            converter = CopilotSDKStreamResponseConverter(context)
            for event in converter.initial_events():
                yield event
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

**Unique features:**
- **Handler-based**: Wraps a plain function, not a framework client
- **Sync + async**: Supports both sync and async handler functions
- **Simplest integration**: Just provide a function that takes a prompt and returns text

**Quick start:**
```python
from azure.ai.agentserver.copilotsdk import from_copilot_sdk

async def my_handler(prompt: str) -> str:
    return f"Echo: {prompt}"

from_copilot_sdk(my_handler).run()  # localhost:8088
```

---

## Deep Dive: Input/Output Conversion

The converters vary dramatically in complexity because each framework has a different native format. Here's why, and what the conversion looks like in detail.

### Why Complexity Varies

```
              Least complex                              Most complex
   ┌──────────────┼──────────────┼──────────────┼──────────────┐
   │              │              │              │              │
Copilot SDK   Claude SDK   AgentFramework    LangGraph
(str → str)   (str → str)  (ChatMessage     (LangChain msgs
~50 lines     ~100 lines    + 3 content      + multimodal
                              types)           + state dict)
                             ~950 lines       ~500+ lines
```

The complexity is proportional to **how much of the OpenAI format the native framework can represent**.

### Input Conversion

Each converter answers: **how do you turn an OpenAI `input` array into what the framework expects?**

| Framework | Native input type | Why |
|-----------|------------------|-----|
| **AgentFramework** | `ChatMessage(role, text)` or `str` | Framework accepts structured chat messages with roles — close to OpenAI format |
| **LangGraph** | `{"messages": [HumanMessage, ...]}` | Richest type system — typed LangChain messages with multimodal + tool calls |
| **Claude SDK** | `str` (plain text) | `agent.send()` takes a single string — all structure is flattened |
| **Copilot SDK** | `str` (role-prefixed) | Handler takes a string — roles preserved as text prefixes |

#### AgentFramework: Preserves message structure

```python
# OpenAI request input:
[
    {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "You are helpful."}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
    {"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "Hi there!"}]},
]

# → AgentFrameworkInputConverter.transform_input() →

[
    ChatMessage(role=ChatRole.SYSTEM, text="You are helpful."),
    ChatMessage(role=ChatRole.USER, text="Hello"),
    ChatMessage(role=ChatRole.ASSISTANT, text="Hi there!"),
]
```

Handles three input shapes:
1. **Implicit user messages**: `{"content": "hello"}` (no type/role) → `"hello"` as string
2. **Explicit typed messages**: `{"type": "message", "role": "user", ...}` → `ChatMessage(role, text)`
3. **Mixed content**: Falls back to extracting text from ChatMessage objects

#### LangGraph: Maps to typed LangChain messages with multimodal support

```python
# OpenAI request:
{
    "instructions": "You are a weather bot.",
    "input": [
        {"type": "message", "role": "user",
         "content": [
             {"type": "input_text", "text": "What's the weather?"},
             {"type": "input_image", "url": "https://..."}
         ]},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather",
         "arguments": "{\"location\": \"Seattle\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "15°C, cloudy"},
    ]
}

# → LangGraphRequestConverter.convert() →

{
    "messages": [
        SystemMessage(content="You are a weather bot."),
        HumanMessage(content=[
            {"type": "text", "text": "What's the weather?"},
            {"type": "image", "url": "https://..."}           # input_image → image
        ]),
        AIMessage(content="", tool_calls=[
            ToolCall(id="call_1", name="get_weather", args={"location": "Seattle"})
        ]),
        ToolMessage(content="15°C, cloudy", tool_call_id="call_1"),
    ]
}
```

Content type remapping: `input_text` → `text`, `input_image` → `image`, `input_audio` → `audio`, `input_file` → `file`.

#### Claude SDK: Flattens everything to a string

```python
# OpenAI request input:
[
    {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "You are helpful."}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's the weather?"}]},
]

# → ClaudeInputConverter.transform_input() →

"You are helpful. What's the weather?"
```

All message roles are discarded. Multi-part content is joined with spaces.

#### Copilot SDK: Preserves roles as text prefixes

```python
# OpenAI request:
{
    "instructions": "You are a weather bot.",
    "input": [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "What's the weather?"}]},
        {"type": "function_call", "name": "get_weather",
         "arguments": "{\"location\": \"Seattle\"}"},
        {"type": "function_call_output", "output": "15°C, cloudy"},
    ]
}

# → CopilotSDKRequestConverter.convert() →

"System: You are a weather bot.\nUser: What's the weather?\n[Function Call: get_weather({\"location\": \"Seattle\"})]\n[Function Output: 15°C, cloudy]"
```

---

### Output Conversion

The reverse: **how do you turn the framework's response into OpenAI output items?**

| Framework | Native output type | Supported output items |
|-----------|-------------------|----------------------|
| **AgentFramework** | `AgentRunResponse` with `TextContent`, `FunctionCallContent`, `FunctionResultContent` | `message`, `function_call`, `function_call_output` |
| **LangGraph** | `AIMessage`, `ToolMessage`, `HumanMessage` with multimodal content | `message` (text, image, audio, file), `function_call`, `function_call_output` |
| **Claude SDK** | Stream of text chunks (strings or objects with `.content`) | `message` (text only) |
| **Copilot SDK** | Single string return value | `message` (text only) |

#### AgentFramework: Three content types → three output item types

```python
# AgentRunResponse with mixed content:
response.messages = [
    Message(contents=[TextContent(text="Let me check the weather...")]),
    Message(contents=[FunctionCallContent(name="get_weather", call_id="call_1", arguments={"location": "Seattle"})]),
    Message(contents=[FunctionResultContent(call_id="call_1", result=[TextContent(text="15°C, cloudy")])]),
    Message(contents=[TextContent(text="The weather in Seattle is 15°C and cloudy.")]),
]

# → AgentFrameworkOutputNonStreamingConverter.transform_output_for_response() →

{
    "id": "resp_...", "object": "response", "status": "completed",
    "output": [
        {"type": "message", "id": "msg_...", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Let me check the weather...", "annotations": []}]},

        {"type": "function_call", "id": "func_...", "status": "completed",
         "call_id": "call_1", "name": "get_weather", "arguments": "{\"location\": \"Seattle\"}"},

        {"type": "function_call_output", "id": "funcout_...", "status": "completed",
         "call_id": "call_1", "output": "[{\"type\": \"text\", \"text\": \"15°C, cloudy\"}]"},

        {"type": "message", "id": "msg_...", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "The weather in Seattle is 15°C and cloudy.", "annotations": []}]}
    ]
}
```

#### LangGraph: Maps LangChain messages to OpenAI items with multimodal support

```python
# LangGraph output state (list of step dicts):
[
    {"agent": {"messages": [
        AIMessage(content="", tool_calls=[ToolCall(id="call_1", name="get_weather", args={"location": "Seattle"})])
    ]}},
    {"tools": {"messages": [
        ToolMessage(content="15°C, cloudy", tool_call_id="call_1")
    ]}},
    {"agent": {"messages": [
        AIMessage(content="The weather in Seattle is 15°C and cloudy.")
    ]}},
]

# → LangGraphResponseConverter.convert() →

[
    FunctionToolCallItemResource(id="func_...", call_id="call_1", name="get_weather",
                                 arguments="{\"location\": \"Seattle\"}", status="completed"),
    FunctionToolCallOutputItemResource(id="funcout_...", call_id="call_1", output="15°C, cloudy"),
    ResponsesAssistantMessageItemResource(id="msg_...", status="completed",
        content=[ItemContent(type="output_text", text="The weather in Seattle is 15°C and cloudy.", annotations=[])]),
]
```

Bidirectional content type mapping: `text` → `output_text` (assistant) or `input_text` (user), `image` → `input_image` (user only), `audio` → `output_audio` (assistant) or `input_audio` (user).

#### Claude SDK: Collects text chunks into a single message

```python
# Claude response chunks (from agent.receive_response()):
[ChunkObject(content="The weather "), ChunkObject(content="in Seattle "), ChunkObject(content="is 15°C and cloudy.")]

# → ClaudeOutputNonStreamingConverter.transform_output_for_response() →

{"id": "resp_...", "status": "completed",
 "output": [{"type": "message", "id": "msg_...", "status": "completed",
             "content": [{"type": "output_text", "text": "The weather in Seattle is 15°C and cloudy.", "annotations": []}]}]}
```

#### Copilot SDK: Wraps a single string in response structure

```python
# Handler return value:
"The weather in Seattle is 15°C and cloudy."

# → CopilotSDKResponseConverter.convert() →

{"id": "resp_...", "status": "completed",
 "output": [{"type": "message", "id": "msg_...", "status": "completed",
             "content": [{"type": "output_text", "text": "The weather in Seattle is 15°C and cloudy.", "annotations": []}]}]}
```

---

### End-to-End Pipeline Example (AgentFramework)

```
Step 1: Client sends POST /responses
─────────────────────────────────────
{"instructions": "You are a weather bot.",
 "input": [{"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "What's the weather in Seattle?"}]}],
 "stream": false}
        │
        ▼
Step 2: Server parses → AgentRunContext
───────────────────────────────────────
context.request = CreateResponse(instructions=..., input=[...], stream=False)
context.response_id = "resp_Xk9mN2..."
context.conversation_id = "conv_Jp3qR7..."
        │
        ▼
Step 3: AgentFrameworkInputConverter.transform_input()
──────────────────────────────────────────────────────
Input:  [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's..."}]}]
Output: ChatMessage(role=ChatRole.USER, text="What's the weather in Seattle?")
        │
        ▼
Step 4: agent.run(message) — Agent Framework executes
──────────────────────────────────────────────────────
Framework calls the LLM → tool call → tool result → final answer.
Returns: AgentRunResponse(messages=[
    Message(contents=[FunctionCallContent(...)]),
    Message(contents=[FunctionResultContent(...)]),
    Message(contents=[TextContent(text="It's 15°C and cloudy in Seattle.")])])
        │
        ▼
Step 5: AgentFrameworkOutputNonStreamingConverter
─────────────────────────────────────────────────
Dispatches each content type to the matching OpenAI output item format.
        │
        ▼
Step 6: Server sends HTTP 200 JSON response
───────────────────────────────────────────
{"id": "resp_Xk9mN2...", "object": "response", "status": "completed",
 "output": [
   {"type": "function_call", ...},
   {"type": "function_call_output", ...},
   {"type": "message", "content": [{"type": "output_text", "text": "It's 15°C and cloudy in Seattle."}]}
 ]}
```

---

## Comparative Analysis

| Aspect | AgentFramework | LangGraph | Claude SDK | Copilot SDK |
|--------|---------------|-----------|------------|-------------|
| **Wraps** | `AgentProtocol` | `CompiledStateGraph` | `ClaudeSDKClient` | `Callable` (function) |
| **Factory** | `from_agent_framework(agent)` | `from_langgraph(agent, converter?)` | `from_claude(agent)` | `from_copilot_sdk(handler, model?, msg?)` |
| **Input format** | `ChatMessage \| str \| list` | `{"messages": [LangChain msgs]}` | `str` (plain text) | `str` (role-prefixed) |
| **Output types** | Text, FunctionCall, FunctionResult | Text, ToolCall, ToolMessage, multimodal | Text only | Text only |
| **Streaming** | State machine (3 states) | Event generators per message | Linear events | Linear events |
| **Tool support** | Native function tools | LangChain `@tool` | MCP protocol | Via handler |
| **State mgmt** | Via framework | Checkpointers (Redis, memory) | Via Claude SDK | None (stateless) |
| **Tracing** | OTLP gRPC | OTLP HTTP + Azure AI | OTLP HTTP | None built-in |
| **Complexity** | ★★★★ | ★★★★★ | ★★★ | ★★ |
| **Samples** | 5 | 7 | 2 | 1 |

### When to Use Each

| Use Case | Recommended Adapter |
|----------|-------------------|
| Microsoft Agent Framework agents with tool calling | **AgentFramework** |
| LangChain/LangGraph agents, RAG, complex workflows | **LangGraph** |
| Claude-powered agents with MCP tools | **Claude SDK** |
| Simple function-based agents, quick prototyping | **Copilot SDK** |
| Multimodal (images, audio, files) | **LangGraph** |
| Custom state management | **LangGraph** (custom state_converter) |
| Production tracing/observability | **AgentFramework** or **LangGraph** |

---

## Implementing the Contract Directly

If your agent doesn't use a supported framework, implement `FoundryCBAgent` directly.

### Python — Minimal Custom Agent

From the official core README:

```python
import datetime
from azure.ai.agentserver.core import FoundryCBAgent
from azure.ai.agentserver.core.models import CreateResponse, Response as OpenAIResponse
from azure.ai.agentserver.core.models.projects import (
    ItemContentOutputText, ResponsesAssistantMessageItemResource,
    ResponseTextDeltaEvent, ResponseTextDoneEvent,
)

def stream_events(text: str):
    assembled = ""
    for i, token in enumerate(text.split(" ")):
        piece = token if i == len(text.split(" ")) - 1 else token + " "
        assembled += piece
        yield ResponseTextDeltaEvent(delta=piece)
    yield ResponseTextDoneEvent(text=assembled)

async def agent_run(request_body: CreateResponse):
    if request_body.stream:
        return stream_events("I am a streaming agent.")

    return OpenAIResponse(
        metadata={}, temperature=0.0, top_p=0.0, user="me", id="id",
        created_at=datetime.datetime.now(),
        output=[ResponsesAssistantMessageItemResource(
            status="completed",
            content=[ItemContentOutputText(text="I am a non-streaming agent.", annotations=[])],
        )],
    )

my_agent = FoundryCBAgent()
my_agent.agent_run = agent_run
if __name__ == "__main__":
    my_agent.run()
```

### Python — Custom Agent with Streaming

```python
from azure.ai.agentserver.core import FoundryCBAgent, AgentRunContext
from azure.ai.agentserver.core.models import Response as OpenAIResponse
from azure.ai.agentserver.core.models.projects import (
    ItemContentOutputText, ResponsesAssistantMessageItemResource,
    ResponseCreatedEvent, ResponseInProgressEvent, ResponseCompletedEvent,
    ResponseOutputItemAddedEvent, ResponseContentPartAddedEvent,
    ResponseTextDeltaEvent, ResponseTextDoneEvent,
    ResponseContentPartDoneEvent, ResponseOutputItemDoneEvent,
)

class WeatherAgent(FoundryCBAgent):
    async def agent_run(self, context: AgentRunContext):
        # Extract user message
        user_text = ""
        for item in (context.request.get("input") or []):
            if isinstance(item, dict) and item.get("role") == "user":
                for part in (item.get("content") or []):
                    if isinstance(part, dict) and part.get("type") == "input_text":
                        user_text = part["text"]

        answer = "It's 72°F and sunny." if "weather" in user_text.lower() else f"You said: {user_text}"

        if context.stream:
            return self._stream(answer, context)
        return self._respond(answer, context)

    async def _stream(self, text, ctx):
        yield ResponseCreatedEvent(response={"id": ctx.response_id, "status": "in_progress"})
        yield ResponseInProgressEvent(response={"id": ctx.response_id})
        yield ResponseOutputItemAddedEvent(output_index=0, item={"type": "message", "role": "assistant"})
        yield ResponseContentPartAddedEvent(output_index=0, content_index=0, part={"type": "output_text"})
        for word in text.split(" "):
            yield ResponseTextDeltaEvent(delta=word + " ")
        yield ResponseTextDoneEvent(text=text)
        yield ResponseContentPartDoneEvent(output_index=0, content_index=0)
        yield ResponseOutputItemDoneEvent(output_index=0)
        yield ResponseCompletedEvent(response={"id": ctx.response_id, "status": "completed"})

    def _respond(self, text, ctx):
        import datetime
        return OpenAIResponse(
            id=ctx.response_id, created_at=datetime.datetime.now(),
            metadata={}, temperature=0.0, top_p=0.0, user="",
            output=[ResponsesAssistantMessageItemResource(
                status="completed",
                content=[ItemContentOutputText(text=text, annotations=[])],
            )],
        )

if __name__ == "__main__":
    WeatherAgent().run()
```

### TypeScript — Implementing the Contract

> **Note**: As of February 2026, Azure AI Agent Server adapters exist only for Python. There is no published TypeScript SDK. However, the underlying contract is HTTP + JSON — any language can implement it.

```typescript
import { createServer } from "http";

// ── Types mirroring the Python SDK ───────────────────────────────
interface CreateResponse {
  input?: InputItem[];
  stream?: boolean;
  instructions?: string;
  model?: string;
  temperature?: number;
  conversation?: { id: string };
  agent?: { name: string; type: string; version: string };
}

interface InputItem { type: string; role: string; content: ContentPart[]; }
interface ContentPart { type: string; text?: string; }
interface AgentResponse { id: string; object: "response"; status: string; created_at: string; output: OutputItem[]; }
interface OutputItem { type: "message"; role: "assistant"; status: "completed"; content: { type: "output_text"; text: string }[]; }
interface StreamEvent { type: string; [key: string]: unknown; }

// ── Base class ───────────────────────────────────────────────────
abstract class FoundryCBAgent {
  abstract agentRun(context: AgentRunContext): Promise<AgentResponse | AsyncGenerator<StreamEvent>>;

  run(port: number = 8088): void {
    const server = createServer(async (req, res) => {
      if (req.method === "GET" && req.url === "/liveness") { res.writeHead(200); res.end(); return; }
      if (req.method === "GET" && req.url === "/readiness") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ status: "ready" })); return;
      }
      if (req.method === "POST" && (req.url === "/runs" || req.url === "/responses")) {
        const body = await readBody(req);
        const context = new AgentRunContext(JSON.parse(body));
        try {
          const result = await this.agentRun(context);
          if (isAsyncGenerator(result)) {
            res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
            for await (const event of result) { res.write(`data: ${JSON.stringify(event)}\n\n`); }
            res.write("data: [DONE]\n\n"); res.end();
          } else {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify(result));
          }
        } catch (err) { res.writeHead(500); res.end(JSON.stringify({ error: String(err) })); }
      }
    });
    server.listen(port, () => console.log(`Agent server on :${port}`));
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

### TypeScript — LangGraph.js Adapter Example

```typescript
import { CompiledStateGraph } from "@langchain/langgraph";
import { HumanMessage, AIMessageChunk } from "@langchain/core/messages";

class LangGraphAdapter extends FoundryCBAgent {
  constructor(private graph: CompiledStateGraph) { super(); }

  async agentRun(context: AgentRunContext): Promise<AgentResponse | AsyncGenerator<StreamEvent>> {
    const messages = (context.request.input ?? [])
      .filter((item) => item.role === "user")
      .map((item) => new HumanMessage(
        item.content.filter((c) => c.type === "input_text").map((c) => c.text).join("\n")
      ));
    const config = { configurable: { thread_id: context.conversationId } };

    if (context.stream) return this.streamResponse(messages, config, context);

    const result = await this.graph.invoke({ messages }, config);
    const last = result.messages[result.messages.length - 1];
    return {
      id: context.responseId, object: "response", status: "completed",
      created_at: new Date().toISOString(),
      output: [{ type: "message", role: "assistant", status: "completed",
                 content: [{ type: "output_text", text: last.content as string }] }],
    };
  }

  private async *streamResponse(messages: HumanMessage[], config: object, ctx: AgentRunContext) {
    yield { type: "response.created", response: { id: ctx.responseId, status: "in_progress" } };
    yield { type: "response.in_progress", response: { id: ctx.responseId } };
    yield { type: "response.output_item.added", output_index: 0, item: { type: "message", role: "assistant" } };
    yield { type: "response.content_part.added", output_index: 0, content_index: 0, part: { type: "output_text" } };
    let fullText = "";
    for await (const event of this.graph.streamEvents({ messages }, { ...config, version: "v2" })) {
      if (event.event === "on_chat_model_stream") {
        const delta = (event.data.chunk as AIMessageChunk).content as string;
        if (delta) { fullText += delta; yield { type: "response.output_text.delta", delta }; }
      }
    }
    yield { type: "response.output_text.done", text: fullText };
    yield { type: "response.content_part.done", output_index: 0, content_index: 0 };
    yield { type: "response.output_item.done", output_index: 0 };
    yield { type: "response.completed", response: { id: ctx.responseId, status: "completed" } };
  }
}

// Usage: new LangGraphAdapter(createReactAgent({ llm: model, tools })).run(8088);
```

### TypeScript — Simple Function Adapter Example

```typescript
class FunctionAdapter extends FoundryCBAgent {
  constructor(private handler: (prompt: string) => Promise<string>) { super(); }

  async agentRun(context: AgentRunContext): Promise<AgentResponse | AsyncGenerator<StreamEvent>> {
    const prompt = (context.request.input ?? [])
      .flatMap((item) => item.content ?? [])
      .filter((c) => c.type === "input_text").map((c) => c.text).join("\n");
    const result = await this.handler(prompt);

    if (context.stream) return this.streamResult(result, context);
    return {
      id: context.responseId, object: "response", status: "completed",
      created_at: new Date().toISOString(),
      output: [{ type: "message", role: "assistant", status: "completed",
                 content: [{ type: "output_text", text: result }] }],
    };
  }

  private async *streamResult(text: string, ctx: AgentRunContext) {
    yield { type: "response.created", response: { id: ctx.responseId, status: "in_progress" } };
    yield { type: "response.in_progress", response: { id: ctx.responseId } };
    yield { type: "response.output_item.added", output_index: 0, item: { type: "message", role: "assistant" } };
    yield { type: "response.content_part.added", output_index: 0, content_index: 0, part: { type: "output_text" } };
    for (const word of text.split(" ")) yield { type: "response.output_text.delta", delta: word + " " };
    yield { type: "response.output_text.done", text };
    yield { type: "response.content_part.done", output_index: 0, content_index: 0 };
    yield { type: "response.output_item.done", output_index: 0 };
    yield { type: "response.completed", response: { id: ctx.responseId, status: "completed" } };
  }
}

// Usage: new FunctionAdapter(async (prompt) => `Echo: ${prompt}`).run(8088);
```

---

## Summary

All four adapters serve the same purpose: **host any agent framework behind Azure AI Foundry's OpenAI-compatible Responses API**. They share the `FoundryCBAgent` base class and the factory function pattern. The key differentiators are:

1. **Input/output fidelity**: LangGraph and AgentFramework preserve rich message structures; Claude and Copilot flatten to strings
2. **Tool support**: Each uses its framework's native tool system (Agent Framework functions, LangChain tools, MCP, handler functions)
3. **Streaming sophistication**: AgentFramework and LangGraph use state machines; Claude and Copilot use linear event sequences
4. **Extensibility**: LangGraph's `LanggraphStateConverter` abstraction allows custom graph states; others are more fixed
