"""Record LangChain ``create_agent`` model and tool calls, with lessons per model call.

Requires ``langchain>=1.1`` (``pip install "sovara-langchain[agent]"``). Add the middleware
last, so it records the request other middleware finalized (model, prompt, messages) and each
attempt of a middleware retry as its own step::

    agent = create_agent(model, tools, middleware=[*other_middleware, SovaraMiddleware()])
    with client.trace("support"), client.turn(turn_input=question):
        agent.invoke({"messages": [{"role": "user", "content": question}]})

Outside a recording the middleware passes calls through. Lessons go into one temporary
user message on the model request only; the agent's messages are never changed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from sovara import Execution, LlmCall, Step, ToolCall
from sovara import integration as instrumentation
from sovara.integration import Ending, json_value
from sovara_langchain import _messages as convert
from sovara_langchain._tool_patch import own_tool_call, tool_outcome

__all__ = ["SovaraMiddleware"]


def _llm_call(request: ModelRequest) -> LlmCall | None:
    return convert.llm_call(request.model, request.messages, request.tools, request.system_message)


def _with_lessons(request: ModelRequest, step: Step) -> tuple[ModelRequest, Execution | None]:
    context = step.lesson_context
    if context is None:
        return request, None
    messages = convert.with_lessons(request.messages, context.text)
    execution = Execution(
        input=convert.conversation(messages, request.tools, request.system_message),
        injected_retrieval_id=context.retrieval_id,
    )
    return request.override(messages=messages), execution


def _model_output(response: Any) -> Any:
    # Newer langchain versions may wrap it in an ExtendedModelResponse.
    response = getattr(response, "model_response", response)
    if not isinstance(response, AIMessage):
        result = getattr(response, "result", ())
        response = next((m for m in result if isinstance(m, AIMessage)), None)
    return convert.safe_output(response)


def _tool_call(request: Any) -> ToolCall:
    call = request.tool_call
    return ToolCall(
        name=call["name"],
        arguments=json_value(call.get("args")),
        call_id=convert.call_id(call.get("id")),
        metadata=convert.step_metadata(),
    )


def _tool_result(result: Any) -> tuple[Ending, tuple[Any, ...], dict[str, Any]]:
    if not isinstance(result, Command):
        return tool_outcome(result)
    update = result.update if isinstance(result.update, dict) else {}
    messages = [m.content for m in update.get("messages", ()) if isinstance(m, ToolMessage)]
    return "finish", (json_value(messages) if messages else None,), {}


class SovaraMiddleware(AgentMiddleware):
    """Record each model and tool call of a LangChain agent as a Sovara step."""

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        """Record the model call, adding retrieved lessons to this request only."""
        call = _llm_call(request)
        step = instrumentation.begin(call, lessons=True) if call is not None else None
        if step is None:
            return handler(request)
        request, execution = _with_lessons(request, step)
        try:
            with step.activate():
                response = handler(request)
        except BaseException as exc:
            instrumentation.report_raised(step, exc, classify=convert.raised, execution=execution)
            raise
        instrumentation.report(step, "finish", _model_output(response), execution=execution)
        return response

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[Any]]
    ) -> Any:
        """Asynchronous form of wrap_model_call()."""
        call = _llm_call(request)
        step = await instrumentation.abegin(call, lessons=True) if call is not None else None
        if step is None:
            return await handler(request)
        request, execution = _with_lessons(request, step)
        try:
            with step.activate():
                response = await handler(request)
        except BaseException as exc:
            await instrumentation.areport_raised(
                step, exc, classify=convert.raised, execution=execution
            )
            raise
        await instrumentation.areport(step, "finish", _model_output(response), execution=execution)
        return response

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Record the tool call; the existing tool patch skips it."""
        step = instrumentation.begin(_tool_call(request))
        if step is None:
            return handler(request)
        try:
            with own_tool_call(request.tool_call.get("id")):
                result = handler(request)
        except BaseException as exc:
            instrumentation.report_raised(step, exc, classify=convert.raised)
            raise
        ending, args, kwargs = _tool_result(result)
        instrumentation.report(step, ending, *args, **kwargs)
        return result

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        """Asynchronous form of wrap_tool_call()."""
        step = await instrumentation.abegin(_tool_call(request))
        if step is None:
            return await handler(request)
        try:
            with own_tool_call(request.tool_call.get("id")):
                result = await handler(request)
        except BaseException as exc:
            await instrumentation.areport_raised(step, exc, classify=convert.raised)
            raise
        ending, args, kwargs = _tool_result(result)
        await instrumentation.areport(step, ending, *args, **kwargs)
        return result
