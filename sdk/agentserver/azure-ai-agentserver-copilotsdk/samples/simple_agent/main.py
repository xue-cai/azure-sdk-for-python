"""
Simple agent example using the GitHub Copilot SDK adapter.

This sample demonstrates how to create a basic agent that uses the GitHub Copilot SDK
and hosts it using the Azure AI Agent Server adapter.

Prerequisites:
    1. Install the Copilot CLI: https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli
    2. Have an active GitHub Copilot subscription
    3. Install dependencies: pip install -r requirements.txt
    4. Copy .env-template to .env and add your GitHub token (optional if already authenticated)

Usage:
    python main.py

    The agent will start on http://localhost:8088

Testing:
    curl -X POST http://localhost:8088/runs \
        -H "Content-Type: application/json" \
        -d '{"input": "What is 2 + 2?"}'
"""

import asyncio
import os

from dotenv import load_dotenv

from azure.ai.agentserver.copilotsdk import from_copilot_sdk

load_dotenv()


async def simple_copilot_handler(prompt: str) -> str:
    """
    Handle incoming prompts using the GitHub Copilot SDK.

    This handler creates a Copilot SDK session, sends the prompt,
    and returns the response.

    :param prompt: The user's prompt to process.
    :type prompt: str

    :return: The Copilot's response.
    :rtype: str
    """
    # Import the Copilot SDK
    from copilot import CopilotClient

    # Create and start the client
    client = CopilotClient()
    await client.start()

    try:
        # Get GitHub token from environment if available
        github_token = os.getenv("GITHUB_TOKEN")

        # Create a session with the specified model
        session_config = {"model": "gpt-4o"}
        if github_token:
            session_config["github_token"] = github_token

        session = await client.create_session(session_config)

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

        # Send the prompt and wait for completion
        await session.send({"prompt": prompt})
        await done.wait()

        # Clean up the session
        await session.destroy()

        return response_text

    finally:
        # Always stop the client
        await client.stop()


if __name__ == "__main__":
    # Create and run the agent server
    # This will host the agent on http://localhost:8088
    from_copilot_sdk(simple_copilot_handler).run()
