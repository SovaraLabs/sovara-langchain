"""Record synchronous and asynchronous LangChain tool execution."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

from sovara import ErrorInfo, ToolCall
from sovara import integration as instrumentation
from sovara.integration import Ending, json_value
from sovara_langchain._log import logger
from sovara_langchain._messages import graph_location

# Tool call ID recorded by SovaraMiddleware. Only that tool is skipped here; model and
# HTTP calls inside it are still recorded.
_owned_tool_call_id: ContextVar[str | None] = ContextVar("owned_tool_call_id", default=None)
_handled_errors: ContextVar[list[str] | None] = ContextVar("handled_tool_errors", default=None)


@contextmanager
def own_tool_call(tool_call_id: str | None):
    """Mark a tool call as recorded by the caller, so this patch skips it."""
    token = _owned_tool_call_id.set(tool_call_id)
    try:
        yield
    finally:
        _owned_tool_call_id.reset(token)


@contextmanager
def _capture_tool_errors():
    """Keep handled errors local to this invocation, including across copied contexts."""
    errors: list[str] = []
    token = _handled_errors.set(errors)
    try:
        yield errors
    finally:
        _handled_errors.reset(token)


def _skip(tool_call_id: Any) -> bool:
    if not instrumentation.capture_active():
        return True
    return bool(tool_call_id) and str(tool_call_id) == _owned_tool_call_id.get()


def _tool_call(tool: Any, tool_input: Any, tool_call_id: Any) -> ToolCall:
    metadata = json_value(
        {
            "tool_class": tool.__class__.__name__,
            "response_format": getattr(tool, "response_format", None),
            **graph_location(),
        }
    )
    return ToolCall(
        name=tool.name,
        arguments=json_value(tool_input),
        call_id=(str(tool_call_id).strip() or None) if tool_call_id is not None else None,
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def tool_outcome(
    result: Any, *, failed: bool = False
) -> tuple[Ending, tuple[Any, ...], dict[str, Any]]:
    """Return the step method and arguments for a tool result, unwrapping ToolMessage."""
    is_message = getattr(result, "tool_call_id", None) is not None and hasattr(result, "content")
    content = result.content if is_message else result
    output = json_value(content)
    if failed or (is_message and getattr(result, "status", None) == "error"):
        message = content if isinstance(content, str) else None
        return "fail", (ErrorInfo(type="ToolError", message=message),), {"output": output}
    return "finish", (output,), {}


def langchain_patch():
    """Wrap LangChain tool run/arun methods once, if langchain_core is installed.

    Capture requires a recording and enabled instrumentation. Tool
    input and results pass through unchanged; exceptions are reported before being
    re-raised. Methods are replaced globally on BaseTool.
    """
    try:
        from langchain_core.tools import base
        from langchain_core.tools.base import BaseTool
    except ImportError:
        logger.info("langchain_core not installed, skipping LangChain tool patches")
        return

    if getattr(BaseTool.run, "__sovara_langchain_patched__", False):
        return

    original_run = BaseTool.run
    original_arun = BaseTool.arun
    original_format_output = base._format_output

    @wraps(original_format_output)
    def format_output(content, artifact, tool_call_id, name, status):
        # LangChain drops the status when an ID-less call returns plain content.
        errors = _handled_errors.get()
        if errors is not None and status == "error":
            errors.append(status)
        return original_format_output(content, artifact, tool_call_id, name, status)

    @wraps(original_run)
    def patched_run(self, tool_input, *args, tool_call_id=None, **kwargs):
        def run():
            return original_run(self, tool_input, *args, tool_call_id=tool_call_id, **kwargs)

        with _capture_tool_errors() as errors:
            if _skip(tool_call_id):
                with own_tool_call(None):
                    return run()
            step = instrumentation.begin(_tool_call(self, tool_input, tool_call_id))
            try:
                result = run()
            except BaseException as exc:
                instrumentation.report_raised(step, exc)
                raise
            ending, report_args, report_kwargs = tool_outcome(result, failed=bool(errors))
            instrumentation.report(step, ending, *report_args, **report_kwargs)
            return result

    @wraps(original_arun)
    async def patched_arun(self, tool_input, *args, tool_call_id=None, **kwargs):
        async def arun():
            return await original_arun(self, tool_input, *args, tool_call_id=tool_call_id, **kwargs)

        with _capture_tool_errors() as errors:
            if _skip(tool_call_id):
                with own_tool_call(None):
                    return await arun()
            step = await instrumentation.abegin(_tool_call(self, tool_input, tool_call_id))
            try:
                result = await arun()
            except BaseException as exc:
                await instrumentation.areport_raised(step, exc)
                raise
            ending, report_args, report_kwargs = tool_outcome(result, failed=bool(errors))
            await instrumentation.areport(step, ending, *report_args, **report_kwargs)
            return result

    patched_run.__sovara_langchain_patched__ = True  # pyright: ignore[reportAttributeAccessIssue]
    patched_arun.__sovara_langchain_patched__ = True  # pyright: ignore[reportAttributeAccessIssue]
    base._format_output = format_output
    BaseTool.run = patched_run
    BaseTool.arun = patched_arun
    logger.info("langchain_core tool patch applied")
