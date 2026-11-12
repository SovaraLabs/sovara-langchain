"""The LangChain middleware and chat-model wrapper against a real Sovara exec server."""

import asyncio
import json
from typing import Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, SecretStr, ValidationError

import sovara_langchain
from sovara import (
    RecordingError,
    SovaraClient,
    SovaraConfigError,
)
from sovara.integration import capture_active
from sovara.runner.monkey_patching.patches.httpx_patch import patch_httpx_send
from sovara_langchain import SovaraChatModel, SovaraMiddleware, wrap_chat_model
from tests.e2e.support import (
    AGENT_TOKEN,
    LESSON_BLOCK,
    QUESTION,
    ScriptedModel,
    applied_lessons,
    async_server_client,
    enable_lessons,
    json_of,
    lesson_last,
    recorded_steps,
)

# Whether capture was suppressed while each lookup ran.
SUPPRESSED_IN_LOOKUP: list[bool] = []


@tool
def lookup(topic: str) -> str:
    """Look up the team policy for a topic."""
    SUPPRESSED_IN_LOOKUP.append(not capture_active())
    return f"policy for {topic}"


@tool
def broken(topic: str) -> str:
    """Fail for every topic."""
    raise ToolException(f"no policy for {topic}")


broken.handle_tool_error = True


def _calling(name: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": {"topic": "ci"}, "id": "call-1"}]
    )


def test_agent_records_model_and_tool_steps_with_temporary_lessons(server):
    app, client = server
    sovara_langchain.instrument_tools()  # The tool patch skips calls the middleware owns.
    first = _calling("lookup")
    first.usage_metadata = {
        "input_tokens": 30,
        "output_tokens": 5,
        "total_tokens": 35,
        "input_token_details": {"cache_read": 10},
    }
    model = ScriptedModel(replies=[first, AIMessage(content="Rerun the job.")])
    agent = create_agent(
        model, [lookup], system_prompt="Be brief.", middleware=[SovaraMiddleware()]
    )

    with client.trace("agent", capture_logs=False) as trace_id:
        enable_lessons(app)
        with client.turn(turn_input=QUESTION):
            result = agent.invoke({"messages": [HumanMessage(QUESTION)]})

    assert result["messages"][-1].content == "Rerun the job."
    assert all(LESSON_BLOCK not in str(m.content) for m in result["messages"])
    assert len(model.seen) == 2 and all(lesson_last(sent) for sent in model.seen)
    assert isinstance(model.seen[0][0], SystemMessage)

    llm, tool_step, answer = steps = recorded_steps(app, trace_id)
    assert [s.step_type for s in steps] == ["llm_call", "tool_call", "llm_call"]
    assert all(s.status == "success" for s in steps)
    call = json_of(llm.call_json)
    assert call["metadata"]["framework"] == "langchain"
    assert call["metadata"]["langgraph_nodes"] == ["model"]
    sent = call["input"]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert [t["name"] for t in sent["tools"]] == ["lookup"]
    assert LESSON_BLOCK not in str(llm.call_json)
    executed = json_of(llm.execution_json)
    assert executed["input"]["messages"][-1]["content"][0]["text"] == LESSON_BLOCK
    assert executed["injected_retrieval_id"]
    assert (llm.input_tokens, llm.cache_read_tokens, llm.output_tokens) == (20, 10, 5)
    assert json_of(tool_step.call_json)["call_id"] == "call-1"
    # Model calls inside the tool are still recorded.
    assert SUPPRESSED_IN_LOOKUP[-1] is False
    assert json_of(tool_step.outcome_json)["output"] == "policy for ci"
    assert {llm.step_uuid, answer.step_uuid} <= applied_lessons(app, trace_id)


def _chunk(delta: dict, finish_reason: str | None = None) -> bytes:
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    body = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-5.4"}
    return f"data: {json.dumps({**body, 'choices': [choice]})}\n\n".encode()


def _openai(bodies: list[dict], model: str = "gpt-5.4") -> ChatOpenAI:
    """Return ChatOpenAI over a fake provider whose HTTP client Sovara instruments."""

    def provider(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("stream"):
            events = [
                _chunk({"role": "assistant", "content": "Rerun"}),
                _chunk({"content": " the job."}),
                _chunk({}, "stop"),
                b"data: [DONE]\n\n",
            ]
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=b"".join(events)
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5.4",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Rerun the job."},
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(provider))
    patch_httpx_send(http, httpx.Client)
    return ChatOpenAI(
        model=model,
        api_key=SecretStr("test-key"),
        http_client=http,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
    )


def test_owned_model_call_is_not_recorded_again_by_http_instrumentation(server):
    app, client = server
    bodies: list[dict] = []
    agent = create_agent(_openai(bodies), [], middleware=[SovaraMiddleware()])

    with client.trace("openai", capture_logs=False) as trace_id:
        enable_lessons(app)
        agent.invoke({"messages": [HumanMessage(QUESTION)]})

    (step,) = recorded_steps(app, trace_id)
    call = json_of(step.call_json)
    assert call["input"]["format"] == "conversation"
    assert (call["provider"], call["model"]) == ("openai", "gpt-5.4")
    assert bodies[0]["messages"][-1] == {"role": "user", "content": LESSON_BLOCK}


def test_async_failures_are_recorded_and_surface_unchanged(server):
    app, client = server
    tools = ScriptedModel(replies=[_calling("broken"), AIMessage(content="Gave up.")])
    failing = ScriptedModel(error="provider down")

    async def run():
        async with async_server_client(app) as http:
            async_client = SovaraClient(
                project_name=client.project_name,
                base_url=app.url,
                agent_token=AGENT_TOKEN,
                async_http_client=http,
            )
            async with async_client.trace("async", capture_logs=False) as trace_id:
                await create_agent(tools, [broken], middleware=[SovaraMiddleware()]).ainvoke(
                    {"messages": [HumanMessage(QUESTION)]}
                )
                with pytest.raises(RuntimeError, match="provider down"):
                    await create_agent(failing, [], middleware=[SovaraMiddleware()]).ainvoke(
                        {"messages": [HumanMessage(QUESTION)]}
                    )
            await async_client.aclose()
        return trace_id

    steps = recorded_steps(app, asyncio.run(run()))
    assert [(s.step_type, s.status) for s in steps] == [
        ("llm_call", "success"),
        ("tool_call", "error"),
        ("llm_call", "success"),
        ("llm_call", "error"),
    ]
    assert "no policy for ci" in str(steps[1].outcome_json)
    assert "provider down" in str(steps[3].outcome_json)


def test_calls_pass_through_outside_a_recording_or_when_preparation_fails(server, monkeypatch):
    app, client = server
    model = ScriptedModel(replies=[AIMessage(content="one"), AIMessage(content="two")])
    agent = create_agent(model, [], middleware=[SovaraMiddleware()])
    assert agent.invoke({"messages": [HumanMessage(QUESTION)]})["messages"][-1].content == "one"

    def unavailable(*args, **kwargs):
        raise RecordingError("unavailable", "Sovara is unavailable.")

    monkeypatch.setattr("sovara._steps.start_step", unavailable)
    with client.trace("unavailable", capture_logs=False) as trace_id:
        result = agent.invoke({"messages": [HumanMessage(QUESTION)]})
    assert result["messages"][-1].content == "two"
    assert [len(sent) for sent in model.seen] == [1, 1]
    assert recorded_steps(app, trace_id) == []


def test_wrapped_model_in_a_langgraph_node_records_one_step_with_lessons(server):
    app, client = server
    bodies: list[dict] = []
    model = wrap_chat_model(_openai(bodies)).bind_tools([lookup])

    def answer(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState).add_node("answer", answer)
    graph = graph.add_edge(START, "answer").compile()
    with client.trace("graph", capture_logs=False) as trace_id:
        enable_lessons(app)
        result = graph.invoke({"messages": [HumanMessage(QUESTION)]})

    assert [m.content for m in result["messages"]] == [QUESTION, "Rerun the job."]
    assert bodies[0]["messages"][-1] == {"role": "user", "content": LESSON_BLOCK}
    assert [t["function"]["name"] for t in bodies[0]["tools"]] == ["lookup"]
    (step,) = recorded_steps(app, trace_id)
    call = json_of(step.call_json)
    assert (call["provider"], call["model"]) == ("openai", "gpt-5.4")
    assert [t["name"] for t in call["input"]["tools"]] == ["lookup"]
    assert LESSON_BLOCK not in str(step.call_json)
    assert json_of(step.outcome_json)["output"]["stop_reason"] == "stop"
    assert step.step_uuid in applied_lessons(app, trace_id)


def test_supervisor_sub_assistant_steps_name_the_assistant_and_its_delegation(server):
    app, client = server
    model = wrap_chat_model(ScriptedModel(replies=[AIMessage(content=c) for c in "abc"]))

    def answer(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    assistant = StateGraph(MessagesState).add_node("agent", answer)
    assistant = assistant.add_edge(START, "agent").compile()
    # The supervisor answers, then delegates to two sub-assistants built from one graph.
    graph = StateGraph(MessagesState).add_node("supervisor", answer)
    graph = graph.add_node("sql_assistant", assistant).add_node("reviewer", assistant)
    graph = graph.add_edge(START, "supervisor").add_edge("supervisor", "sql_assistant")
    graph = graph.add_edge("sql_assistant", "reviewer").compile()
    with client.trace("supervisor", capture_logs=False) as trace_id:
        graph.invoke({"messages": [HumanMessage(QUESTION)]})

    steps = recorded_steps(app, trace_id)
    stored = [json_of(step.input_json)["raw"]["_meta"] for step in steps]
    assert [meta["langgraph_nodes"] for meta in stored] == [
        ["supervisor"],
        ["sql_assistant", "agent"],
        ["reviewer", "agent"],
    ]
    assert len({meta["langgraph_namespace"] for meta in stored}) == 3
    assert all(meta["framework"] == "langchain" for meta in stored)


def test_streams_record_the_aggregate_and_early_close_cancels_with_partial_output(server):
    app, client = server
    model = wrap_chat_model(_openai([]))

    with client.trace("stream", capture_logs=False) as trace_id:
        assert "".join(str(c.content) for c in model.stream(QUESTION)) == "Rerun the job."
        chunks = model.stream(QUESTION)
        assert next(chunks).content == "Rerun"
        chunks.close()  # type: ignore[attr-defined]

    done, closed = recorded_steps(app, trace_id)
    done_output = json_of(done.outcome_json)["output"]
    assert done_output["message"] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Rerun the job."}],
    }
    assert done_output["stop_reason"] == "stop"
    cancelled = json_of(closed.outcome_json)
    assert (cancelled["status"], cancelled["reason"]) == ("cancelled", "Closed")
    assert cancelled["output"]["message"]["content"] == [{"type": "text", "text": "Rerun"}]


class Verdict(BaseModel):
    """The verdict on a CI failure."""

    cause: str


def _verdict(args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "Verdict", "args": args, "id": "call-1"}])


def test_structured_output_keeps_native_parsing_and_records_the_model_call(server):
    app, client = server
    native = ScriptedModel(
        replies=[_verdict({"cause": "flaky"}), _verdict({"cause": "flaky"}), _verdict({})]
    )
    model = wrap_chat_model(native)

    with client.trace("structured", capture_logs=False) as trace_id:
        enable_lessons(app)
        assert model.with_structured_output(Verdict).invoke(QUESTION) == Verdict(cause="flaky")
        raw = model.with_structured_output(Verdict, include_raw=True).invoke(QUESTION)
        with pytest.raises(ValidationError):
            model.with_structured_output(Verdict).invoke(QUESTION)

    assert raw["parsed"] == Verdict(cause="flaky") and raw["parsing_error"] is None
    assert all(lesson_last(sent) for sent in native.seen)
    steps = recorded_steps(app, trace_id)
    assert [s.status for s in steps] == ["success"] * 3
    assert [t["name"] for t in json_of(steps[0].call_json)["input"]["tools"]] == ["Verdict"]


def test_wrapper_failures_pass_through_and_the_middleware_owns_wrapped_calls(server, monkeypatch):
    app, client = server
    sovara_langchain.instrument_tools()
    failing = wrap_chat_model(ScriptedModel(error="provider down"))
    streaming = wrap_chat_model(_openai([]))
    inner = ScriptedModel(replies=[AIMessage(content="one"), AIMessage(content="two")])
    agent = create_agent(wrap_chat_model(inner), [], middleware=[SovaraMiddleware()])

    async def run():
        async with async_server_client(app) as http:
            async_client = SovaraClient(
                project_name=client.project_name,
                base_url=app.url,
                agent_token=AGENT_TOKEN,
                async_http_client=http,
            )
            async with async_client.trace("async", capture_logs=False) as trace_id:
                with pytest.raises(RuntimeError, match="provider down"):
                    await failing.ainvoke(QUESTION)
                assert "".join([str(c.content) async for c in streaming.astream(QUESTION)]) == (
                    "Rerun the job."
                )
                chunks = streaming.astream(QUESTION)
                assert (await anext(chunks)).content == "Rerun"
                await chunks.aclose()  # type: ignore[attr-defined]
                await asyncio.sleep(0)  # LangChain leaves closing the wrapper's stream to asyncio.
            await async_client.aclose()
        return trace_id

    failed, streamed, closed = recorded_steps(app, asyncio.run(run()))
    assert failed.status == "error" and "provider down" in str(failed.outcome_json)
    assert json_of(streamed.outcome_json)["output"]["stop_reason"] == "stop"
    assert json_of(closed.outcome_json)["status"] == "cancelled"

    with client.trace("both", capture_logs=False) as trace_id:
        enable_lessons(app)
        agent.invoke({"messages": [HumanMessage(QUESTION)]})
    (step,) = recorded_steps(app, trace_id)
    assert lesson_last(inner.seen[0]) and len(inner.seen[0]) == 2

    def unavailable(*args, **kwargs):
        raise RecordingError("unavailable", "Sovara is unavailable.")

    monkeypatch.setattr("sovara._steps.start_step", unavailable)
    with client.trace("unavailable", capture_logs=False) as trace_id:
        assert wrap_chat_model(inner).invoke(QUESTION).content == "two"
        assert lookup.invoke({"topic": "ci"}) == "policy for ci"
    assert recorded_steps(app, trace_id) == []


class NamedModel(ScriptedModel):
    """A scripted model with a name, that fails its first ``failures`` calls."""

    label: str = "model-a"
    failures: int = 0

    def _get_ls_params(self, stop=None, **kwargs):
        return {"ls_provider": "scripted", "ls_model_name": self.label, "ls_model_type": "chat"}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if len(self.seen) < self.failures:
            self.seen.append(list(messages))
            raise RuntimeError("rate limited")
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_middleware_last_records_what_earlier_middleware_sent(server):
    app, client = server
    original = NamedModel(label="model-a")
    routed = NamedModel(
        label="model-b",
        failures=1,
        replies=[AIMessage(content="never sent"), AIMessage(content="Rerun the job.")],
    )

    class Route(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return handler(
                request.override(model=routed, system_message=SystemMessage("Answer in one line."))
            )

    class Retry(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            try:
                return handler(request)
            except RuntimeError:
                return handler(request)

    agent = create_agent(
        original, [], system_prompt="Be brief.", middleware=[Route(), Retry(), SovaraMiddleware()]
    )
    with client.trace("composed", capture_logs=False) as trace_id:
        enable_lessons(app)
        with client.turn(turn_input=QUESTION):
            result = agent.invoke({"messages": [HumanMessage(QUESTION)]})

    # The model that ran is the one recorded, with the prompt it received.
    assert result["messages"][-1].content == "Rerun the job."
    assert original.seen == []
    failed, retried = steps = recorded_steps(app, trace_id)
    for step in steps:
        call = json_of(step.call_json)
        assert (call["provider"], call["model"]) == ("scripted", "model-b")
        assert "Answer in one line." in step.call_json and "Be brief." not in step.call_json

    # One step per attempt: the failure stays visible, then the retry succeeds.
    assert [step.status for step in steps] == ["error", "success"]
    assert "rate limited" in str(failed.outcome_json)

    # Each attempt gets the lessons once, reported as executed, never kept in the agent's state.
    assert len(routed.seen) == 2
    for sent, step in zip(routed.seen, steps, strict=True):
        assert lesson_last(sent) and sum(m.content == LESSON_BLOCK for m in sent) == 1
        assert sent[0].content == "Answer in one line."
        executed = json_of(step.execution_json)
        assert executed["injected_retrieval_id"]
        assert executed["input"]["messages"][-1]["content"][0]["text"] == LESSON_BLOCK
        assert "Answer in one line." in step.execution_json
    assert all(LESSON_BLOCK not in str(m.content) for m in result["messages"])
    assert applied_lessons(app, trace_id) == {failed.step_uuid, retried.step_uuid}


def test_prebuilt_agents_and_concurrent_calls_each_record_their_own_steps(server):
    app, client = server
    native = ScriptedModel(replies=[_calling("lookup"), AIMessage(content="Rerun the job.")])
    agent = create_react_agent(wrap_chat_model(native), [lookup])
    questions: list[Any] = [f"{QUESTION} ({n})" for n in range(3)]
    batched = ScriptedModel(replies=[AIMessage(content="ok")] * 3)

    async def run():
        async with async_server_client(app) as http:
            async_client = SovaraClient(
                project_name=client.project_name,
                base_url=app.url,
                agent_token=AGENT_TOKEN,
                async_http_client=http,
            )
            async with async_client.trace("batch", capture_logs=False) as trace_id:
                enable_lessons(app)
                await wrap_chat_model(batched).abatch(questions)
            await async_client.aclose()
        return trace_id

    with client.trace("prebuilt", capture_logs=False) as trace_id:
        enable_lessons(app)
        result = agent.invoke({"messages": [HumanMessage(QUESTION)]})
    assert result["messages"][-1].content == "Rerun the job."
    assert all(lesson_last(sent) for sent in native.seen)
    llm_steps = [s for s in recorded_steps(app, trace_id) if s.step_type == "llm_call"]
    assert len(llm_steps) == 2

    steps = recorded_steps(app, asyncio.run(run()))
    sent = sorted(
        m.content for request in batched.seen for m in request if m.content != LESSON_BLOCK
    )
    assert sent == questions and all(lesson_last(request) for request in batched.seen)
    inputs = sorted(
        json_of(s.call_json)["input"]["messages"][0]["content"][0]["text"] for s in steps
    )
    assert inputs == questions
    assert all(json_of(s.execution_json)["injected_retrieval_id"] for s in steps)


class _Starts(BaseCallbackHandler):
    def __init__(self) -> None:
        self.count = 0

    def on_chat_model_start(self, *args, **kwargs) -> None:
        del args, kwargs
        self.count += 1


def test_wrapper_behaves_like_the_wrapped_model_and_rejects_what_it_cannot_record():
    bodies: list[dict] = []
    native = _openai(bodies, model="gpt-4.1")
    native.tags = ["prod"]
    model = wrap_chat_model(native)
    assert isinstance(model, SovaraChatModel) and wrap_chat_model(model) is model
    assert (model._llm_type, model.model_name, model.tags) == ("openai-chat", "gpt-4.1", ["prod"])
    assert model._get_ls_params()["ls_provider"] == "openai"
    assert model.root_client is native.root_client
    assert getattr(model, "deployment_name", "none") == "none"

    starts = _Starts()
    model.temperature = 0.1
    model.callbacks = [starts]
    model.invoke(QUESTION)
    assert native.temperature == 0.1 and bodies[-1]["temperature"] == 0.1
    assert starts.count == 1 and native.callbacks == [starts]

    class Unbindable(ScriptedModel):
        def bind_tools(self, tools, **kwargs):
            return ScriptedModel().bind(tools=tools)

    with pytest.raises(SovaraConfigError, match="bind_tools"):
        wrap_chat_model(Unbindable()).bind_tools([lookup])
    with pytest.raises(SovaraConfigError, match="BaseChatModel"):
        wrap_chat_model(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("tool_input", "call_id"),
    [
        ({"topic": "ci"}, None),
        (
            {"type": "tool_call", "name": "lookup", "args": {"topic": "ci"}, "id": "call-1"},
            "call-1",
        ),
    ],
)
def test_direct_tools_record_with_or_without_a_model_call_id(server, tool_input, call_id):
    app, client = server
    sovara_langchain.instrument_tools()
    sovara_langchain.instrument_tools()
    with client.trace("direct", capture_logs=False) as trace_id:
        result = lookup.invoke(tool_input)
    assert (result.content if isinstance(result, ToolMessage) else result) == "policy for ci"
    steps = recorded_steps(app, trace_id)
    assert len(steps) == 1
    step = steps[0]
    assert json_of(step.call_json)["call_id"] == call_id
    assert json_of(step.call_json)["arguments"] == {"topic": "ci"}
    assert json_of(step.outcome_json) == {"status": "success", "output": "policy for ci"}
    assert lookup.invoke({"topic": "ci"}) == "policy for ci"


@pytest.mark.parametrize(
    ("error", "status"),
    [(RuntimeError("failed"), "error"), (asyncio.CancelledError(), "cancelled")],
)
def test_direct_tool_errors_are_recorded_and_propagated(server, error, status):
    app, client = server
    sovara_langchain.instrument_tools()

    @tool
    def fails() -> str:
        """Fail the call."""
        raise error

    with client.trace("failed tool", capture_logs=False) as trace_id:
        with pytest.raises(type(error)) as raised:
            fails.invoke({})
    assert raised.value is error
    steps = recorded_steps(app, trace_id)
    assert len(steps) == 1
    step = steps[0]
    assert json_of(step.outcome_json)["status"] == status


def test_async_nested_tools_and_models_record_once_under_middleware(server):
    app, client = server
    sovara_langchain.instrument_tools()
    inner_model = wrap_chat_model(ScriptedModel(replies=[AIMessage(content="nested")]))

    @tool("lookup")
    async def inner(topic: str) -> str:
        """Look up a topic."""
        return f"policy for {topic}"

    @tool
    async def outer(topic: str) -> str:
        """Look up two topics and ask a model."""
        await asyncio.gather(inner.ainvoke({"topic": topic}), inner.ainvoke({"topic": "other"}))
        return str((await inner_model.ainvoke(topic)).content)

    agent = create_agent(
        ScriptedModel(replies=[_calling("outer"), AIMessage(content="done")]),
        [outer],
        middleware=[SovaraMiddleware()],
    )

    async def run():
        async with async_server_client(app) as http:
            async_client = SovaraClient(
                project_name=client.project_name,
                base_url=app.url,
                agent_token=AGENT_TOKEN,
                async_http_client=http,
            )
            async with async_client.trace("nested", capture_logs=False) as trace_id:
                result = await agent.ainvoke({"messages": [HumanMessage(QUESTION)]})
                assert result["messages"][-1].content == "done"
            await async_client.aclose()
        return trace_id

    steps = recorded_steps(app, asyncio.run(run()))
    calls = [json_of(step.call_json) for step in steps]
    assert [call["kind"] for call in calls] == ["llm", "tool", "tool", "tool", "llm", "llm"]
    assert [call["name"] for call in calls[1:4]] == ["outer", "lookup", "lookup"]
    assert [call["call_id"] for call in calls[1:4]] == ["call-1", None, None]
    assert sorted(call["arguments"]["topic"] for call in calls[2:4]) == ["ci", "other"]
    assert all(step.status == "success" for step in steps)


@pytest.mark.parametrize(
    ("tool_input", "expected"),
    [({"topic": "ci"}, "no policy for ci"), ({}, "Tool input validation error")],
)
def test_handled_direct_tool_errors_do_not_fail_the_parent(server, tool_input, expected):
    app, client = server
    sovara_langchain.instrument_tools()
    fails = broken.model_copy(update={"handle_validation_error": True})

    @tool
    def outer() -> str:
        """Return the inner tool's handled error."""
        return fails.invoke(tool_input)

    with client.trace("handled error", capture_logs=False) as trace_id:
        assert outer.invoke({}) == expected
    steps = recorded_steps(app, trace_id)
    assert [step.status for step in steps] == ["success", "error"]
    assert [json_of(step.outcome_json)["output"] for step in steps] == [expected, expected]


@pytest.mark.parametrize(
    ("tool_input", "expected"),
    [({"topic": "ci"}, "no policy for ci"), ({}, "Tool input validation error")],
)
def test_async_handled_direct_tool_errors_do_not_fail_the_parent(server, tool_input, expected):
    app, client = server
    sovara_langchain.instrument_tools()

    @tool
    async def fails(topic: str) -> str:
        """Fail for every topic."""
        raise ToolException(f"no policy for {topic}")

    fails.handle_tool_error = True
    fails.handle_validation_error = True

    @tool
    async def outer() -> str:
        """Return the inner tool's handled error."""
        return await fails.ainvoke(tool_input)

    async def run():
        async with async_server_client(app) as http:
            async_client = SovaraClient(
                project_name=client.project_name,
                base_url=app.url,
                agent_token=AGENT_TOKEN,
                async_http_client=http,
            )
            async with async_client.trace("handled error", capture_logs=False) as trace_id:
                assert await outer.ainvoke({}) == expected
            await async_client.aclose()
        return trace_id

    steps = recorded_steps(app, asyncio.run(run()))
    assert [step.status for step in steps] == ["success", "error"]
    assert [json_of(step.outcome_json)["output"] for step in steps] == [expected, expected]
