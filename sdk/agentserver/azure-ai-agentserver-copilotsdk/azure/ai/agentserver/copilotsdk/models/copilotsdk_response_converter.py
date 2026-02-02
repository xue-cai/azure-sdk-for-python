# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation
import time
from typing import Any

from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.models import Response
from azure.ai.agentserver.core.models import projects as project_models
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext

logger = get_logger()


class CopilotSDKResponseConverter:
    """
    Converter for transforming Copilot SDK handler output into OpenAI-style Response format.

    This class takes the string result from the Copilot SDK handler and converts it
    into the proper Response format expected by the FoundryCB agent server.
    """

    def __init__(self, context: AgentRunContext, result: Any):
        """
        Initialize the response converter.

        :param context: The agent run context.
        :type context: AgentRunContext
        :param result: The result from the Copilot SDK handler.
        :type result: Any
        """
        self.context = context
        self.result = result

    def convert(self) -> Response:
        """
        Convert the Copilot SDK result to an OpenAI-style Response.

        :return: The converted Response object.
        :rtype: Response
        """
        # Convert the result to a string if it isn't already
        if isinstance(self.result, str):
            content = self.result
        else:
            content = str(self.result)

        # Create the output item
        output_item = project_models.ResponsesAssistantMessageItemResource(
            content=[
                project_models.ItemContent(
                    {
                        "type": project_models.ItemContentType.OUTPUT_TEXT,
                        "text": content,
                        "annotations": [],
                    }
                )
            ],
            id=self.context.id_generator.generate_message_id(),
            status="completed",
        )

        # Build the response
        agent_id = self.context.get_agent_id_object()
        conversation = self.context.get_conversation_object()

        response = Response(
            object="response",
            id=self.context.response_id,
            agent=agent_id,
            conversation=conversation,
            metadata=self.context.request.get("metadata"),
            created_at=int(time.time()),
            output=[output_item],
        )

        return response
