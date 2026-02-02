# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from typing import TYPE_CHECKING, Callable, Optional

from ._version import VERSION

if TYPE_CHECKING:  # pragma: no cover
    from .copilotsdk import CopilotSDKAdapter


def from_copilot_sdk(
    handler: Callable,
    model: Optional[str] = None,
    system_message: Optional[str] = None,
) -> "CopilotSDKAdapter":
    """
    Create a CopilotSDKAdapter from a prompt handler function.

    :param handler: An async function that takes a prompt (str) and returns a response (str).
                   This function will be called for each incoming request.
    :type handler: Callable
    :param model: The model to use for the Copilot SDK session (e.g., "gpt-5", "claude-sonnet-4.5").
                 Defaults to "gpt-4o" if not specified.
    :type model: Optional[str]
    :param system_message: Optional system message to set context for conversations.
    :type system_message: Optional[str]

    :return: A CopilotSDKAdapter instance ready to be run.
    :rtype: CopilotSDKAdapter
    """
    from .copilotsdk import CopilotSDKAdapter

    return CopilotSDKAdapter(handler=handler, model=model, system_message=system_message)


__all__ = ["from_copilot_sdk"]
__version__ = VERSION
