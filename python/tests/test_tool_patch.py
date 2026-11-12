"""The tool patch records LangChain tool runs and passes provider arguments through."""

import uuid

import pytest

from tests.fake_steps import fake_steps, recording_context


@pytest.mark.anyio
async def test_langchain_tool_is_recorded_and_keeps_provider_arguments(monkeypatch):
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import StructuredTool
    from langchain_core.tools.base import BaseTool

    from sovara_langchain import _tool_patch as patch_module

    calls = []
    results = iter(["result", ToolMessage(content="bad", tool_call_id="c2", status="error")])

    def original(_self, _input, **kwargs):
        calls.append(kwargs)
        return next(results)

    async def original_async(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(BaseTool, "run", original)
    monkeypatch.setattr(BaseTool, "arun", original_async)
    patch_module.langchain_patch()
    steps = fake_steps(monkeypatch)
    tool = StructuredTool.from_function(lambda text: text, name="echo", description="Echo")
    provider_id = uuid.uuid4()

    with recording_context():
        assert tool.run({"text": "a"}, tool_call_id="c1", run_id=provider_id) == "result"
        failed = await tool.arun({"text": "b"}, tool_call_id="c2", run_id=provider_id)

    assert failed.status == "error"
    assert [kwargs["run_id"] for kwargs in calls] == [provider_id, provider_id]
    assert [step.call.call_id for step in steps] == ["c1", "c2"]
    assert steps[0].call.name == "echo" and steps[0].call.arguments == {"text": "a"}
    assert steps[0].call.metadata["tool_class"] == "StructuredTool"
    assert (steps[0].ending, steps[0].output) == ("finish", "result")
    assert steps[1].ending == "fail"
    assert steps[1].reports[0][1].type == "ToolError"
    assert steps[1].reports[0][2]["output"] == "bad"
