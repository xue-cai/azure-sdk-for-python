import pytest

from azure.ai.agentserver.core import models
from azure.ai.agentserver.copilotsdk.models import CopilotSDKRequestConverter


@pytest.mark.unit
def test_convert_simple_string_input():
    """Test conversion of simple string input to prompt."""
    create_response = models.CreateResponse(
        input="Hello, how are you?",
    )

    converter = CopilotSDKRequestConverter(create_response)
    result = converter.convert()

    assert result == "Hello, how are you?"


@pytest.mark.unit
def test_convert_string_input_with_instructions():
    """Test conversion of string input with instructions."""
    create_response = models.CreateResponse(
        input="What's the weather?",
        instructions="You are a helpful assistant.",
    )

    converter = CopilotSDKRequestConverter(create_response)
    result = converter.convert()

    assert "System: You are a helpful assistant." in result
    assert "What's the weather?" in result


@pytest.mark.unit
def test_convert_message_list_input():
    """Test conversion of message list input."""
    input_data = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    create_response = models.CreateResponse(
        input=input_data,
    )

    converter = CopilotSDKRequestConverter(create_response)
    result = converter.convert()

    assert "User: Hello" in result
    assert "Assistant: Hi there!" in result
    assert "User: How are you?" in result


@pytest.mark.unit
def test_convert_system_message():
    """Test conversion of system message."""
    input_data = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Help me"},
    ]
    create_response = models.CreateResponse(
        input=input_data,
    )

    converter = CopilotSDKRequestConverter(create_response)
    result = converter.convert()

    assert "System: You are an AI assistant." in result
    assert "User: Help me" in result


@pytest.mark.unit
def test_convert_content_list():
    """Test conversion of message with content list."""
    input_data = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Hello"},
                {"type": "input_text", "text": "World"},
            ],
        },
    ]
    create_response = models.CreateResponse(
        input=input_data,
    )

    converter = CopilotSDKRequestConverter(create_response)
    result = converter.convert()

    assert "Hello" in result
    assert "World" in result


@pytest.mark.unit
def test_convert_function_call():
    """Test conversion of function call item."""
    input_data = [
        {
            "type": "function_call",
            "call_id": "call_001",
            "name": "get_weather",
            "arguments": '{"location": "Seattle"}',
        },
    ]
    create_response = models.CreateResponse(
        input=input_data,
    )

    converter = CopilotSDKRequestConverter(create_response)
    result = converter.convert()

    assert "Function Call" in result
    assert "get_weather" in result


@pytest.mark.unit
def test_convert_function_call_output():
    """Test conversion of function call output item."""
    input_data = [
        {
            "type": "function_call_output",
            "call_id": "call_001",
            "output": "Sunny, 72°F",
        },
    ]
    create_response = models.CreateResponse(
        input=input_data,
    )

    converter = CopilotSDKRequestConverter(create_response)
    result = converter.convert()

    assert "Function Output" in result
    assert "Sunny, 72°F" in result
