# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation
from typing import List

from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.models import CreateResponse, projects as project_models

logger = get_logger()


class CopilotSDKRequestConverter:
    """
    Converter for transforming incoming CreateResponse requests into prompts for the Copilot SDK.

    This class extracts the user's input from the OpenAI-style request format and converts it
    into a simple string prompt that can be passed to the Copilot SDK handler.
    """

    def __init__(self, data: CreateResponse):
        """
        Initialize the request converter.

        :param data: The CreateResponse request data.
        :type data: CreateResponse
        """
        self.data = data

    def convert(self) -> str:
        """
        Convert the CreateResponse input to a string prompt for the Copilot SDK.

        :return: The converted prompt string.
        :rtype: str
        """
        prompt_parts = []

        # Add instructions if provided
        instructions = self.data.get("instructions")
        if instructions and isinstance(instructions, str):
            prompt_parts.append(f"System: {instructions}")

        # Process input
        input_data = self.data.get("input")
        if isinstance(input_data, str):
            prompt_parts.append(input_data)
        elif isinstance(input_data, list):
            for item in input_data:
                message = self._convert_input_item(item)
                if message:
                    prompt_parts.append(message)
        else:
            logger.warning(f"Unsupported input type: {type(input_data)}")

        return "\n".join(prompt_parts)

    def _convert_input_item(self, item: dict) -> str:
        """
        Convert a single input item to a string message.

        :param item: The input item to convert.
        :type item: dict

        :return: The converted message string.
        :rtype: str
        """
        item_type = item.get("type", project_models.ItemType.MESSAGE)

        if item_type == project_models.ItemType.MESSAGE:
            return self._convert_message(item)
        if item_type == project_models.ItemType.FUNCTION_CALL:
            return self._convert_function_call(item)
        if item_type == project_models.ItemType.FUNCTION_CALL_OUTPUT:
            return self._convert_function_call_output(item)

        logger.warning(f"Unsupported item type: {item_type}")
        return ""

    def _convert_message(self, message: dict) -> str:
        """
        Convert a message item to a string.

        :param message: The message item.
        :type message: dict

        :return: The converted message string.
        :rtype: str
        """
        content = message.get("content")
        role = message.get("role", project_models.ResponsesMessageRole.USER)

        if not content:
            return ""

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = self._extract_text_from_content_list(content)
        else:
            logger.warning(f"Unsupported content type: {type(content)}")
            return ""

        # Format with role prefix for context
        if role == project_models.ResponsesMessageRole.USER:
            return f"User: {text}"
        elif role == project_models.ResponsesMessageRole.ASSISTANT:
            return f"Assistant: {text}"
        elif role == project_models.ResponsesMessageRole.SYSTEM:
            return f"System: {text}"
        return text

    def _extract_text_from_content_list(self, content_list: List[dict]) -> str:
        """
        Extract text from a list of content items.

        :param content_list: The list of content items.
        :type content_list: List[dict]

        :return: The extracted text.
        :rtype: str
        """
        text_parts = []
        for item in content_list:
            if item.get("type") in ["text", "input_text", "output_text"]:
                text = item.get("text", "")
                if text:
                    text_parts.append(text)
        return " ".join(text_parts)

    def _convert_function_call(self, item: dict) -> str:
        """
        Convert a function call item to a string representation.

        :param item: The function call item.
        :type item: dict

        :return: The converted string.
        :rtype: str
        """
        name = item.get("name", "unknown")
        arguments = item.get("arguments", "{}")
        return f"[Function Call: {name}({arguments})]"

    def _convert_function_call_output(self, item: dict) -> str:
        """
        Convert a function call output item to a string representation.

        :param item: The function call output item.
        :type item: dict

        :return: The converted string.
        :rtype: str
        """
        output = item.get("output", "")
        return f"[Function Output: {output}]"
