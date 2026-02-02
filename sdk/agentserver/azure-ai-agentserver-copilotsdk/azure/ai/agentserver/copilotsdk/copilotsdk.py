# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation,broad-exception-caught
import asyncio
import os
import time
from typing import Any, AsyncGenerator, Callable, Optional, Union

from azure.ai.agentserver.core import AgentRunContext, FoundryCBAgent
from azure.ai.agentserver.core.constants import Constants
from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.models import Response as OpenAIResponse, ResponseStreamEvent

from .models import CopilotSDKRequestConverter, CopilotSDKResponseConverter, CopilotSDKStreamResponseConverter

logger = get_logger()


class CopilotSDKAdapter(FoundryCBAgent):
    """
    Adapter for GitHub Copilot SDK.

    This class wraps a handler function that uses the GitHub Copilot SDK and provides a unified interface
    for running agents in both streaming and non-streaming modes. It handles input and output
    conversion between the Copilot SDK and the expected formats for FoundryCB agents.
    """

    def __init__(
        self,
        handler: Callable,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
    ):
        """
        Initialize the CopilotSDKAdapter.

        :param handler: An async function that takes a prompt (str) and returns a response (str).
                       Alternatively, it can take (prompt, context) and return a response.
        :type handler: Callable
        :param model: The model to use for Copilot SDK session (e.g., "gpt-5", "claude-sonnet-4.5").
                     Defaults to "gpt-4o" if not specified.
        :type model: Optional[str]
        :param system_message: Optional system message to set context for conversations.
        :type system_message: Optional[str]
        """
        super().__init__()
        self.handler = handler
        self.model = model or "gpt-4o"
        self.system_message = system_message
        logger.info(f"Initialized CopilotSDKAdapter with model: {self.model}")

    async def agent_run(
        self, context: AgentRunContext
    ) -> Union[OpenAIResponse, AsyncGenerator[ResponseStreamEvent, Any]]:
        """
        Run the agent with the provided context.

        :param context: The context for the agent run.
        :type context: AgentRunContext

        :return: The response from the agent run.
        :rtype: Union[OpenAIResponse, AsyncGenerator[ResponseStreamEvent, Any]]
        """
        logger.info(f"Starting agent_run with stream={context.stream}")

        # Convert the request to a prompt
        request_converter = CopilotSDKRequestConverter(context.request)
        prompt = request_converter.convert()
        logger.debug(f"Converted request to prompt: {prompt[:100]}..." if len(prompt) > 100 else prompt)

        if context.stream:
            logger.info("Running agent in streaming mode")
            return self._run_streaming(prompt, context)

        # Non-streaming path
        logger.info("Running agent in non-streaming mode")
        return await self._run_non_streaming(prompt, context)

    async def _run_non_streaming(self, prompt: str, context: AgentRunContext) -> OpenAIResponse:
        """
        Run the agent in non-streaming mode.

        :param prompt: The prompt to send to the handler.
        :type prompt: str
        :param context: The context for the agent run.
        :type context: AgentRunContext

        :return: The response from the agent.
        :rtype: OpenAIResponse
        """
        try:
            # Call the handler with the prompt
            if asyncio.iscoroutinefunction(self.handler):
                result = await self.handler(prompt)
            else:
                result = self.handler(prompt)

            logger.debug(f"Handler returned result: {type(result)}")

            # Convert the result to OpenAI response format
            response_converter = CopilotSDKResponseConverter(context, result)
            response = response_converter.convert()

            logger.info("Agent run completed successfully")
            return response

        except Exception as e:
            logger.error(f"Error during agent run: {e}")
            raise

    async def _run_streaming(
        self, prompt: str, context: AgentRunContext
    ) -> AsyncGenerator[ResponseStreamEvent, Any]:
        """
        Run the agent in streaming mode.

        :param prompt: The prompt to send to the handler.
        :type prompt: str
        :param context: The context for the agent run.
        :type context: AgentRunContext

        :return: An async generator yielding response stream events.
        :rtype: AsyncGenerator[ResponseStreamEvent, Any]
        """
        stream_converter = CopilotSDKStreamResponseConverter(context)

        # Emit initial events
        for event in stream_converter.initial_events():
            yield event

        try:
            # Call the handler
            if asyncio.iscoroutinefunction(self.handler):
                result = await self.handler(prompt)
            else:
                result = self.handler(prompt)

            # Emit content events
            for event in stream_converter.content_events(result):
                yield event

            # Emit completion events
            for event in stream_converter.completion_events():
                yield event

            logger.info("Streaming completed successfully")

        except Exception as e:
            logger.error(f"Error during streaming agent run: {e}")
            raise

    def get_trace_attributes(self):
        """Get trace attributes for telemetry."""
        attrs = super().get_trace_attributes()
        attrs["service.namespace"] = "azure.ai.agentserver.copilotsdk"
        return attrs

    def get_agent_identifier(self) -> str:
        """Get the agent identifier for telemetry."""
        agent_name = os.getenv(Constants.AGENT_NAME)
        if agent_name:
            return agent_name
        agent_id = os.getenv(Constants.AGENT_ID)
        if agent_id:
            return agent_id
        return "HostedAgent-CopilotSDK"
