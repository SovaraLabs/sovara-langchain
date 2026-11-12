"""Record LangChain and LangGraph model and tool calls, with lessons per model call.

Use one of these per model, not both:

- ``SovaraMiddleware`` for ``create_agent`` (needs ``langchain>=1.1``); it also records tool
  calls.
- ``wrap_chat_model(model)`` for any other use of a chat model, such as custom LangGraph
  nodes, ``with_structured_output`` or prebuilt agents (needs only ``langchain-core``).

Call ``instrument_tools()`` once at startup to also record tool calls that the middleware does not
own, such as the tools of a LangGraph agent built on a wrapped model. Each export loads on first
access, so the wrapper does not import ``langchain``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sovara_langchain._chat_model import SovaraChatModel, wrap_chat_model
    from sovara_langchain._middleware import SovaraMiddleware

__all__ = ["SovaraChatModel", "SovaraMiddleware", "instrument_tools", "wrap_chat_model"]

_MODULES = {
    "SovaraChatModel": "sovara_langchain._chat_model",
    "wrap_chat_model": "sovara_langchain._chat_model",
    "SovaraMiddleware": "sovara_langchain._middleware",
}


def instrument_tools() -> None:
    """Record LangChain tool calls inside Sovara traces; idempotent.

    Patches ``langchain_core`` tool execution. Tool calls the middleware already records are
    skipped, so using both never records a tool twice.
    """
    from sovara_langchain._tool_patch import langchain_patch

    langchain_patch()


def __getattr__(name: str) -> Any:
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
