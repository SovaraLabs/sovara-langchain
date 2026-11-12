"""Convert LangChain messages, tools and responses into Sovara conversation data.

Content is read through LangChain's standard ``content_blocks``, so provider-specific
formats are normalized by LangChain. Tool calls come from ``AIMessage.tool_calls``, which
LangChain keeps authoritative for every provider. Also holds the step helpers the
middleware and the chat-model wrapper share.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import ensure_config
from langchain_core.utils.function_calling import convert_to_openai_tool

from sovara import (
    ContentBlock,
    ConversationInput,
    ConversationOutput,
    LlmCall,
    MediaBlock,
    Message,
    ProviderBlock,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolSpec,
    Usage,
)
from sovara import integration as instrumentation
from sovara.integration import Ending, json_value
from sovara_langchain._log import warn_once

try:
    from langgraph.errors import GraphBubbleUp as _BUBBLE_UP
except ImportError:  # LangGraph is optional for the chat-model wrapper.
    _BUBBLE_UP = None

METADATA = {"framework": "langchain"}


def graph_location() -> dict[str, Any]:
    """Return where in a LangGraph graph the current call runs, or nothing outside one.

    ``langgraph_nodes`` lists the node names from the outermost graph in, so a supervisor's
    sub-assistant shows as ``["sql_assistant", "agent"]``. ``langgraph_namespace`` also
    carries LangGraph's task IDs, so it tells repeated or parallel delegations apart.
    """
    metadata = ensure_config().get("metadata") or {}
    namespace = metadata.get("langgraph_checkpoint_ns")
    if not isinstance(namespace, str) or not namespace:
        return {}
    nodes = [part.split(":", 1)[0] for part in namespace.split("|") if part]
    return {"langgraph_nodes": nodes, "langgraph_namespace": namespace}


def step_metadata() -> dict[str, Any]:
    """Return the metadata of a recorded LangChain model or tool call."""
    return {**METADATA, **graph_location()}


_ROLES = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}
_CHAT_ROLES = {"system", "developer", "user", "assistant", "tool"}
_MEDIA = {"image", "audio", "video", "file"}


def call_id(value: Any) -> str | None:
    """Return a tool call ID, or None when it is missing or blank."""
    return (str(value).strip() or None) if value is not None else None


def _role(message: BaseMessage) -> Any:
    # Streamed chunks report types such as "AIMessageChunk".
    kind = message.type.removesuffix("MessageChunk").lower()
    if kind == "chat":
        role = getattr(message, "role", None)
        return role if role in _CHAT_ROLES else "user"
    return _ROLES.get(kind, "user")


def _provider(block: dict[str, Any]) -> ProviderBlock:
    return ProviderBlock(format=f"langchain.{block.get('type')}", data=json_value(block))


def _media(block: dict[str, Any]) -> ContentBlock:
    url, data = block.get("url") or None, block.get("base64") or None
    if (url is None) == (data is None):
        return _provider(block)
    kind = block["type"]
    default = "application/octet-stream" if kind == "file" else f"{kind}/*"
    return MediaBlock(mime_type=block.get("mime_type") or default, url=url, data=data)


def _block(block: Any) -> ContentBlock | None:
    if not isinstance(block, dict):
        return TextBlock(text=block) if isinstance(block, str) and block else None
    kind = block.get("type")
    if kind == "text":
        return TextBlock(text=str(block["text"])) if block.get("text") else None
    if kind == "reasoning":
        return ReasoningBlock(text=str(block.get("reasoning", "")))
    if kind == "tool_call":
        return None
    if kind in _MEDIA:
        return _media(block)
    return _provider(block)


def _blocks(message: BaseMessage) -> list[ContentBlock]:
    return [block for raw in message.content_blocks if (block := _block(raw)) is not None]


def message(message: BaseMessage) -> Message:
    """Convert one LangChain message."""
    if isinstance(message, ToolMessage):
        blocks = _blocks(message)
        texts = [block.text for block in blocks if isinstance(block, TextBlock)]
        output = "".join(texts) if len(texts) == len(blocks) else json_value(message.content)
        result = ToolResultBlock(
            output=output,
            call_id=call_id(message.tool_call_id),
            is_error=message.status == "error",
        )
        return Message(role="tool", content=(result,))
    content = _blocks(message)
    if isinstance(message, AIMessage):
        content += [
            ToolCallBlock(
                name=call["name"],
                arguments=json_value(call["args"]),
                call_id=call_id(call.get("id")),
            )
            for call in message.tool_calls
        ]
    return Message(role=_role(message), content=tuple(content))


def _tool(tool: Any) -> ToolSpec:
    try:
        function = convert_to_openai_tool(tool)["function"]
    except Exception:
        name = tool.get("name") or tool.get("type") if isinstance(tool, dict) else None
        return ToolSpec(name=str(name or type(tool).__name__))
    parameters = json_value(function.get("parameters") or {})
    return ToolSpec(
        name=function["name"],
        description=function.get("description"),
        parameters=parameters if isinstance(parameters, dict) else {},
    )


def conversation(
    messages: Iterable[BaseMessage], tools: Sequence[Any] = (), system: BaseMessage | None = None
) -> ConversationInput:
    """Convert a model request: optional system message, messages and tools."""
    prefix = [system] if system is not None and system.content_blocks else []
    return ConversationInput(
        messages=tuple(message(item) for item in [*prefix, *messages]),
        tools=tuple(_tool(tool) for tool in tools),
    )


def _count(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _usage(message: AIMessage) -> Usage | None:
    usage = message.usage_metadata
    if not usage:
        return None
    details = usage.get("input_token_details") or {}
    cache_read, cache_write = details.get("cache_read"), details.get("cache_creation")
    raw = usage.get("input_tokens")
    return Usage(
        input_tokens=max(0, raw - _count(cache_read) - _count(cache_write))
        if isinstance(raw, int)
        else None,
        output_tokens=usage.get("output_tokens"),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def output(
    response: AIMessage | None, generation_info: dict[str, Any] | None = None
) -> ConversationOutput:
    """Convert a model response; None becomes an empty assistant message.

    Args:
        response: The model's message.
        generation_info: Metadata LangChain merges into the message only after the model
            returns, such as the OpenAI finish reason.
    """
    if response is None:
        return ConversationOutput(message=Message(role="assistant", content=()))
    metadata = {**(generation_info or {}), **(response.response_metadata or {})}
    stop = metadata.get("finish_reason") or metadata.get("stop_reason")
    return ConversationOutput(
        message=message(response),
        usage=_usage(response),
        stop_reason=stop if isinstance(stop, str) else None,
    )


def model_identity(model: Any) -> tuple[str | None, str | None]:
    """Return the model's provider and name from its LangSmith parameters."""
    try:
        params = model._get_ls_params()
    except Exception:
        return None, None

    def text(key: str) -> str | None:
        value = params.get(key)
        return value if isinstance(value, str) and value.strip() else None

    return text("ls_provider"), text("ls_model_name")


def with_lessons(messages: Sequence[Any], text: str) -> list[Any]:
    """Return a copy with one user message holding lessons, before an assistant prefill."""
    temporary = list(messages)
    last = temporary[-1] if temporary else None
    prefill = isinstance(last, AIMessage) and not last.tool_calls and bool(last.content)
    temporary.insert(len(temporary) - 1 if prefill else len(temporary), HumanMessage(content=text))
    return temporary


def llm_call(
    model: Any,
    messages: Sequence[BaseMessage],
    tools: Sequence[Any] = (),
    system: BaseMessage | None = None,
) -> LlmCall | None:
    """Describe a model call, or return None to run it unrecorded.

    Returns None outside a recording, while capture is suppressed (for example inside a
    call the middleware owns) or when the request cannot be converted.
    """
    if not instrumentation.capture_active():
        return None
    try:
        provider, name = model_identity(model)
        return LlmCall(
            input=conversation(messages, tools, system),
            provider=provider,
            model=name,
            metadata=step_metadata(),
        )
    except Exception as exc:
        warn_once(
            "langchain-convert",
            f"Sovara could not record a LangChain model call. error_type={type(exc).__name__}",
        )
        return None


def safe_output(
    response: AIMessage | None, generation_info: dict[str, Any] | None = None
) -> ConversationOutput:
    """Convert a model response, recording an empty one if conversion fails."""
    try:
        return output(response, generation_info)
    except Exception as exc:
        warn_once(
            "langchain-output",
            f"Sovara could not record a LangChain model response. error_type={type(exc).__name__}",
        )
        return output(None)


def raised(exc: BaseException) -> tuple[Ending, tuple[Any, ...]]:
    """Return the step method and arguments for an exception leaving a call."""
    if _BUBBLE_UP is not None and isinstance(exc, _BUBBLE_UP):
        return "cancel", ("Interrupted",)
    return instrumentation.raised(exc)
