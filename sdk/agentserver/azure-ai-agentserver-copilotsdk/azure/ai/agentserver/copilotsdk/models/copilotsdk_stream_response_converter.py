# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation
import time
from typing import Any, List

from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.models import ResponseStreamEvent
from azure.ai.agentserver.core.models import projects as project_models
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext

logger = get_logger()


class CopilotSDKStreamResponseConverter:
    """
    Converter for transforming Copilot SDK handler output into streaming ResponseStreamEvent format.

    This class handles the conversion of handler output into the proper streaming event format
    expected by the FoundryCB agent server.
    """

    def __init__(self, context: AgentRunContext):
        """
        Initialize the stream response converter.

        :param context: The agent run context.
        :type context: AgentRunContext
        """
        self.context = context
        self.sequence_number = 0
        self.output_item_id = None
        self.content_part_index = 0

    def initial_events(self) -> List[ResponseStreamEvent]:
        """
        Generate the initial events for a streaming response.

        :return: List of initial response stream events.
        :rtype: List[ResponseStreamEvent]
        """
        events = []
        agent_id = self.context.get_agent_id_object()
        conversation = self.context.get_conversation_object()

        # Response created event
        response_dict = {
            "object": "response",
            "agent_id": agent_id,
            "conversation": conversation,
            "id": self.context.response_id,
            "status": "in_progress",
            "created_at": int(time.time()),
        }
        created_event = project_models.ResponseCreatedEvent(
            response=project_models.Response(response_dict),
            sequence_number=self.sequence_number,
        )
        self.sequence_number += 1
        events.append(created_event)

        # Response in progress event
        in_progress_event = project_models.ResponseInProgressEvent(
            response=project_models.Response(response_dict),
            sequence_number=self.sequence_number,
        )
        self.sequence_number += 1
        events.append(in_progress_event)

        return events

    def content_events(self, result: Any) -> List[ResponseStreamEvent]:
        """
        Generate content events for the streaming response.

        :param result: The result from the Copilot SDK handler.
        :type result: Any

        :return: List of content stream events.
        :rtype: List[ResponseStreamEvent]
        """
        events = []

        # Convert result to string
        if isinstance(result, str):
            content = result
        else:
            content = str(result)

        # Generate output item ID
        self.output_item_id = self.context.id_generator.generate_message_id()

        # Output item added event
        output_item = project_models.ResponsesAssistantMessageItemResource(
            content=[],
            id=self.output_item_id,
            status="in_progress",
        )
        item_added_event = project_models.ResponseOutputItemAddedEvent(
            item=output_item,
            output_index=0,
            sequence_number=self.sequence_number,
        )
        self.sequence_number += 1
        events.append(item_added_event)

        # Content part added event
        content_part = project_models.ItemContent(
            {
                "type": project_models.ItemContentType.OUTPUT_TEXT,
                "text": "",
                "annotations": [],
            }
        )
        content_part_added_event = project_models.ResponseContentPartAddedEvent(
            item_id=self.output_item_id,
            output_index=0,
            content_index=self.content_part_index,
            part=content_part,
            sequence_number=self.sequence_number,
        )
        self.sequence_number += 1
        events.append(content_part_added_event)

        # Text delta event with the full content
        text_delta_event = project_models.ResponseTextDeltaEvent(
            item_id=self.output_item_id,
            output_index=0,
            content_index=self.content_part_index,
            delta=content,
            sequence_number=self.sequence_number,
        )
        self.sequence_number += 1
        events.append(text_delta_event)

        # Text done event
        text_done_event = project_models.ResponseTextDoneEvent(
            item_id=self.output_item_id,
            output_index=0,
            content_index=self.content_part_index,
            text=content,
            sequence_number=self.sequence_number,
        )
        self.sequence_number += 1
        events.append(text_done_event)

        # Content part done event
        content_part_done = project_models.ItemContent(
            {
                "type": project_models.ItemContentType.OUTPUT_TEXT,
                "text": content,
                "annotations": [],
            }
        )
        content_part_done_event = project_models.ResponseContentPartDoneEvent(
            item_id=self.output_item_id,
            output_index=0,
            content_index=self.content_part_index,
            part=content_part_done,
            sequence_number=self.sequence_number,
        )
        self.sequence_number += 1
        events.append(content_part_done_event)

        # Output item done event
        output_item_done = project_models.ResponsesAssistantMessageItemResource(
            content=[content_part_done],
            id=self.output_item_id,
            status="completed",
        )
        output_item_done_event = project_models.ResponseOutputItemDoneEvent(
            item=output_item_done,
            output_index=0,
            sequence_number=self.sequence_number,
        )
        self.sequence_number += 1
        events.append(output_item_done_event)

        return events

    def completion_events(self) -> List[ResponseStreamEvent]:
        """
        Generate completion events for the streaming response.

        :return: List of completion stream events.
        :rtype: List[ResponseStreamEvent]
        """
        events = []
        agent_id = self.context.get_agent_id_object()
        conversation = self.context.get_conversation_object()

        # Response completed event
        response_dict = {
            "object": "response",
            "agent_id": agent_id,
            "conversation": conversation,
            "id": self.context.response_id,
            "status": "completed",
            "created_at": int(time.time()),
        }
        completed_event = project_models.ResponseCompletedEvent(
            response=project_models.Response(response_dict),
            sequence_number=self.sequence_number,
        )
        self.sequence_number += 1
        events.append(completed_event)

        return events
