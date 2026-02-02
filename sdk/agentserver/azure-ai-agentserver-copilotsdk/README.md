# Azure AI Agent Server Adapter for GitHub Copilot SDK

## Getting started

```bash
pip install azure-ai-agentserver-copilotsdk
```

### Prerequisites

Before using this adapter, you need to:

1. **Install the GitHub Copilot CLI**: Follow the [Copilot CLI installation guide](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli) to install the CLI, or ensure `copilot` is available in your PATH.

2. **Have a GitHub Copilot subscription**: A GitHub Copilot subscription is required to use the GitHub Copilot SDK. Refer to the [GitHub Copilot pricing page](https://github.com/features/copilot#pricing).


## Key concepts

Azure AI Agent Server wraps your GitHub Copilot SDK-based agent, and hosts it on the cloud using Azure AI Foundry.

The adapter provides a bridge between the GitHub Copilot SDK and the Azure AI Agent Server, allowing you to:
- Use the powerful Copilot agent runtime in your Azure-hosted agents
- Leverage Copilot's built-in tools for file operations, Git, web requests, and more
- Define custom tools using the Copilot SDK's tool definition system


## Examples

### Basic Usage

```python
from copilot import CopilotClient
from azure.ai.agentserver.copilotsdk import from_copilot_sdk

# Define your handler function that uses the Copilot SDK
async def my_agent_handler(prompt: str) -> str:
    """Handle incoming prompts using the Copilot SDK."""
    client = CopilotClient()
    await client.start()

    try:
        session = await client.create_session({"model": "gpt-4o"})

        # Collect the response
        response_text = ""
        done = asyncio.Event()

        def on_event(event):
            nonlocal response_text
            if event.type.value == "assistant.message":
                response_text = event.data.content
            elif event.type.value == "session.idle":
                done.set()

        session.on(on_event)
        await session.send({"prompt": prompt})
        await done.wait()

        await session.destroy()
        return response_text

    finally:
        await client.stop()

if __name__ == "__main__":
    # Host the agent on http://localhost:8088
    from_copilot_sdk(my_agent_handler).run()
```

### With Custom Tools

```python
from pydantic import BaseModel, Field
from copilot import CopilotClient, define_tool
from azure.ai.agentserver.copilotsdk import from_copilot_sdk

class LookupParams(BaseModel):
    query: str = Field(description="The search query")

@define_tool(description="Search for information")
async def search(params: LookupParams) -> str:
    # Your custom search logic here
    return f"Results for: {params.query}"

async def my_agent_handler(prompt: str) -> str:
    client = CopilotClient()
    await client.start()

    try:
        session = await client.create_session({
            "model": "gpt-4o",
            "tools": [search],
        })

        response_text = ""
        done = asyncio.Event()

        def on_event(event):
            nonlocal response_text
            if event.type.value == "assistant.message":
                response_text = event.data.content
            elif event.type.value == "session.idle":
                done.set()

        session.on(on_event)
        await session.send({"prompt": prompt})
        await done.wait()

        await session.destroy()
        return response_text

    finally:
        await client.stop()

if __name__ == "__main__":
    from_copilot_sdk(my_agent_handler).run()
```


## Testing Locally

1. Install the dependencies:
   ```bash
   pip install azure-ai-agentserver-copilotsdk
   ```

2. Make sure the Copilot CLI is installed and you're authenticated:
   ```bash
   copilot --version
   ```

3. Run your agent:
   ```bash
   python your_agent.py
   ```

4. Test with curl:
   ```bash
   curl -X POST http://localhost:8088/runs \
     -H "Content-Type: application/json" \
     -d '{"input": "Hello, what can you help me with?"}'
   ```


## Troubleshooting

First run your agent with azure-ai-agentserver-copilotsdk locally.

If it works locally but fails on cloud:
- Check your logs in the Application Insights connected to your Azure AI Foundry Project
- Ensure the Copilot CLI is properly installed in your deployment environment
- Verify your GitHub Copilot subscription is active


## Next steps

Please visit the [Samples](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/agentserver/azure-ai-agentserver-copilotsdk/samples) folder for more examples.

For more information about the GitHub Copilot SDK, see:
- [Copilot SDK GitHub Repository](https://github.com/github/copilot-sdk)
- [Copilot SDK Python Documentation](https://pypi.org/project/github-copilot-sdk/)


## Contributing

This project welcomes contributions and suggestions. Most contributions require
you to agree to a Contributor License Agreement (CLA) declaring that you have
the right to, and actually do, grant us the rights to use your contribution.
For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether
you need to provide a CLA and decorate the PR appropriately (e.g., label,
comment). Simply follow the instructions provided by the bot. You will only
need to do this once across all repos using our CLA.

This project has adopted the
[Microsoft Open Source Code of Conduct][code_of_conduct]. For more information,
see the Code of Conduct FAQ or contact opencode@microsoft.com with any
additional questions or comments.
