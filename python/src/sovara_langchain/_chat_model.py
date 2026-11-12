"""Wrap a LangChain chat model so every call it makes is recorded with lessons.

``wrap_chat_model(model)`` returns a chat model that records each ``invoke``, ``stream``,
``batch`` and async call, including calls through ``bind_tools`` and
``with_structured_output``. Lessons go into one temporary user message on that request
only; the caller's messages and graph state are never changed. Generation, tool binding,
structured output and parsing stay with the wrapped model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableParallel,
    RunnableSequence,
    RunnableWithFallbacks,
)
from langchain_core.runnables.passthrough import RunnableAssign
from pydantic import PrivateAttr

from sovara import ConversationOutput, Execution, SovaraConfigError, Step
from sovara import integration as instrumentation
from sovara_langchain import _messages as convert

__all__ = ["SovaraChatModel", "wrap_chat_model"]


def _prepare(
    model: BaseChatModel, messages: list[BaseMessage], kwargs: dict[str, Any]
) -> tuple[Step | None, list[BaseMessage], Execution | None]:
    call = convert.llm_call(model, messages, kwargs.get("tools") or ())
    step = instrumentation.begin(call, lessons=True) if call is not None else None
    return _with_lessons(step, messages, kwargs)


async def _aprepare(
    model: BaseChatModel, messages: list[BaseMessage], kwargs: dict[str, Any]
) -> tuple[Step | None, list[BaseMessage], Execution | None]:
    call = convert.llm_call(model, messages, kwargs.get("tools") or ())
    step = await instrumentation.abegin(call, lessons=True) if call is not None else None
    return _with_lessons(step, messages, kwargs)


def _with_lessons(
    step: Step | None, messages: list[BaseMessage], kwargs: dict[str, Any]
) -> tuple[Step | None, list[BaseMessage], Execution | None]:
    context = step.lesson_context if step is not None else None
    if context is None:
        return step, messages, None
    sent = convert.with_lessons(messages, context.text)
    execution = Execution(
        input=convert.conversation(sent, kwargs.get("tools") or ()),
        injected_retrieval_id=context.retrieval_id,
    )
    return step, sent, execution


def _result_output(result: ChatResult) -> ConversationOutput:
    if not result.generations:
        return convert.safe_output(None)
    generation = result.generations[0]
    return convert.safe_output(generation.message, generation.generation_info)  # type: ignore[arg-type]


def _chunks_output(chunks: ChatGenerationChunk | None) -> ConversationOutput:
    if chunks is None:
        return convert.safe_output(None)
    return convert.safe_output(chunks.message, chunks.generation_info)  # type: ignore[arg-type]


def _rebind(node: Any, old: Any, new: Any) -> tuple[Any, bool]:
    """Return ``node`` with the runnable ``old`` replaced by ``new``, and whether it was found."""
    if node is old:
        return new, True
    if isinstance(node, RunnableBinding):
        bound, found = _rebind(node.bound, old, new)
        return (node.model_copy(update={"bound": bound}), True) if found else (node, False)
    if isinstance(node, RunnableSequence):
        first, in_first = _rebind(node.first, old, new)
        middle = [_rebind(step, old, new) for step in node.middle]
        last, in_last = _rebind(node.last, old, new)
        found = in_first or in_last or any(hit for _, hit in middle)
        update = {"first": first, "middle": [step for step, _ in middle], "last": last}
        return (node.model_copy(update=update), True) if found else (node, False)
    if isinstance(node, RunnableParallel):
        steps = {key: _rebind(step, old, new) for key, step in node.steps__.items()}
        if not any(hit for _, hit in steps.values()):
            return node, False
        return node.model_copy(update={"steps__": {k: s for k, (s, _) in steps.items()}}), True
    if isinstance(node, RunnableWithFallbacks):
        runnable, in_runnable = _rebind(node.runnable, old, new)
        fallbacks = [_rebind(step, old, new) for step in node.fallbacks]
        if not in_runnable and not any(hit for _, hit in fallbacks):
            return node, False
        update = {"runnable": runnable, "fallbacks": [step for step, _ in fallbacks]}
        return node.model_copy(update=update), True
    if isinstance(node, RunnableAssign):
        mapper, found = _rebind(node.mapper, old, new)
        return (node.model_copy(update={"mapper": mapper}), True) if found else (node, False)
    return node, False


class SovaraChatModel(BaseChatModel):
    """A chat model that records each call of the wrapped model as a Sovara step.

    Create it with ``wrap_chat_model``. Calls go to the wrapped model's own generation and
    streaming methods; outside a recording they pass through unchanged.

    Public attributes the wrapper does not define, such as ``temperature`` or ``model_name``,
    are read from and set on the wrapped model. Chat-model settings (callbacks, tags, cache and
    the other ``BaseChatModel`` fields) are copied when wrapping; setting one later sets it on
    both models.
    """

    # Private, because providers such as ChatAnthropic use a public ``model`` field.
    _model: BaseChatModel = PrivateAttr()

    def __init__(self, model: BaseChatModel, **settings: Any) -> None:
        """Wrap ``model``; ``settings`` are its ``BaseChatModel`` fields."""
        super().__init__(**settings)
        self._model = model

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)  # type: ignore[misc]
        except AttributeError:
            if name.startswith("_"):
                raise
            private = self.__pydantic_private__ or {}
            if "_model" not in private:
                raise
            return getattr(private["_model"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name in BaseChatModel.model_fields:
            super().__setattr__(name, value)
            setattr(self._model, name, value)
        else:
            setattr(self._model, name, value)

    @property
    def wrapped_model(self) -> BaseChatModel:
        """Return the model whose calls this wrapper records."""
        return self._model

    @property
    def _llm_type(self) -> str:
        return self._model._llm_type

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self._model._identifying_params

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        return self._model._get_ls_params(stop=stop, **kwargs)

    def _should_stream(self, *, async_api: bool, run_manager: Any = None, **kwargs: Any) -> bool:
        return self._model._should_stream(async_api=async_api, run_manager=run_manager, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        step, sent, execution = _prepare(self._model, messages, kwargs)
        if step is None:
            return self._model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        try:
            with step.activate():
                result = self._model._generate(sent, stop=stop, run_manager=run_manager, **kwargs)
        except BaseException as exc:
            instrumentation.report_raised(step, exc, classify=convert.raised, execution=execution)
            raise
        instrumentation.report(step, "finish", _result_output(result), execution=execution)
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        step, sent, execution = await _aprepare(self._model, messages, kwargs)
        if step is None:
            return await self._model._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        try:
            with step.activate():
                result = await self._model._agenerate(
                    sent, stop=stop, run_manager=run_manager, **kwargs
                )
        except BaseException as exc:
            await instrumentation.areport_raised(
                step, exc, classify=convert.raised, execution=execution
            )
            raise
        await instrumentation.areport(step, "finish", _result_output(result), execution=execution)
        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        step, sent, execution = _prepare(self._model, messages, kwargs)
        if step is None:
            yield from self._model._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return
        chunks: ChatGenerationChunk | None = None
        iterator = self._model._stream(sent, stop=stop, run_manager=run_manager, **kwargs)
        try:
            while True:
                # Activation covers each pull from the model, never the consumer's code.
                with step.activate():
                    chunk = next(iterator, None)
                if chunk is None:
                    break
                chunks = chunk if chunks is None else chunks + chunk
                yield chunk
        except BaseException as exc:
            ending, args = (
                ("cancel", ("Closed",)) if isinstance(exc, GeneratorExit) else (convert.raised(exc))
            )
            instrumentation.report(
                step,
                ending,
                *args,
                output=_chunks_output(chunks),
                execution=execution,
                propagating=True,
            )
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
            raise
        instrumentation.report(step, "finish", _chunks_output(chunks), execution=execution)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        step, sent, execution = await _aprepare(self._model, messages, kwargs)
        if step is None:
            async for chunk in self._model._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
            return
        chunks: ChatGenerationChunk | None = None
        iterator = self._model._astream(sent, stop=stop, run_manager=run_manager, **kwargs)
        try:
            while True:
                with step.activate():
                    chunk = await anext(iterator, None)
                if chunk is None:
                    break
                chunks = chunk if chunks is None else chunks + chunk
                yield chunk
        except BaseException as exc:
            ending, args = (
                ("cancel", ("Closed",)) if isinstance(exc, GeneratorExit) else (convert.raised(exc))
            )
            partial = _chunks_output(chunks)
            # A closed async generator may still await; a cancelled task must not.
            if isinstance(exc, (Exception, GeneratorExit)):
                await instrumentation.areport(
                    step, ending, *args, output=partial, execution=execution, propagating=True
                )
            else:
                instrumentation.report_in_background(
                    step, ending, *args, output=partial, execution=execution
                )
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await aclose()
            raise
        await instrumentation.areport(step, "finish", _chunks_output(chunks), execution=execution)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        """Bind tools with the wrapped model's own implementation, keeping calls recorded.

        Raises:
            SovaraConfigError: The wrapped model did not return a binding of itself, so its
                calls could not be recorded.
        """
        native = self._model.bind_tools(tools, **kwargs)
        if not (isinstance(native, RunnableBinding) and native.bound is self._model):
            raise SovaraConfigError(
                f"{type(self._model).__name__}.bind_tools did not return a binding of the "
                "model, so Sovara cannot record its calls."
            )
        return native.model_copy(update={"bound": self})

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        """Use the wrapped model's own structured output, keeping its calls recorded.

        Parsing, ``include_raw`` and errors behave exactly as for the wrapped model.

        Raises:
            SovaraConfigError: The wrapped model's runnable does not call the model in a form
                Sovara can redirect.
        """
        native = self._model.with_structured_output(schema, **kwargs)
        rebound, found = _rebind(native, self._model, self)
        if not found:
            raise SovaraConfigError(
                f"{type(self._model).__name__}.with_structured_output returned a runnable "
                "Sovara cannot record."
            )
        return rebound


def wrap_chat_model(model: BaseChatModel) -> SovaraChatModel:
    """Return a chat model that records every call of ``model``; wrapping twice is a no-op.

    The wrapper takes the wrapped model's callbacks, cache, rate limiter and other chat-model
    settings, since calls go to its generation methods directly. Other attributes are
    forwarded to ``model``.

    Raises:
        SovaraConfigError: ``model`` is not a LangChain chat model.
    """
    if isinstance(model, SovaraChatModel):
        return model
    if not isinstance(model, BaseChatModel):
        raise SovaraConfigError("wrap_chat_model() needs a LangChain BaseChatModel.")
    settings = {name: getattr(model, name) for name in BaseChatModel.model_fields}
    return SovaraChatModel(model=model, **settings)
