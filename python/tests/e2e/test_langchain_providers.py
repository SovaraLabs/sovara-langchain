"""The chat-model wrapper over LangChain provider classes, with each provider client faked.

Each provider package is optional; its cases skip when it is missing.
"""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, SecretStr

from sovara_langchain import wrap_chat_model
from tests.e2e.support import (
    LESSON_BLOCK,
    enable_lessons,
    json_of,
    recorded_steps,
)


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Answer(BaseModel):
    """The final answer."""

    answer: str


HISTORY = [
    HumanMessage("What is 1 + 2?"),
    AIMessage("", tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "call-0"}]),
    ToolMessage("3", tool_call_id="call-0"),
]


def _args(name: str) -> dict:
    return {"a": 1, "b": 2} if name == "add" else {"answer": "3"}


def _openai_reply(body: dict) -> dict:
    tools = body.get("tools") or []
    if tools:
        name = tools[0]["function"]["name"]
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(_args(name))},
                }
            ],
        }
    else:
        message = {"role": "assistant", "content": json.dumps(_args("Answer"))}
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": body["model"],
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    }


def _anthropic_reply(body: dict) -> dict:
    tools = body.get("tools") or []
    content: list[dict] = (
        [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": tools[0]["name"],
                "input": _args(tools[0]["name"]),
            }
        ]
        if tools
        else [{"type": "text", "text": json.dumps(_args("Answer"))}]
    )
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": content,
        "stop_reason": "tool_use" if tools else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def _http(requests: list, reply: Callable[[dict], dict]) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json=reply(body))

    return httpx.Client(transport=httpx.MockTransport(handle))


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _openai(requests: list) -> tuple[Any, Callable[[Any], str]]:
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model="gpt-4.1", api_key=SecretStr("test-key"), http_client=_http(requests, _openai_reply)
    )
    return model, lambda body: _text(body["messages"][-1]["content"])


def _azure(requests: list) -> tuple[Any, Callable[[Any], str]]:
    from langchain_openai import AzureChatOpenAI

    model = AzureChatOpenAI(
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4.1",
        api_version="2024-10-21",
        api_key=SecretStr("test-key"),
        http_client=_http(requests, _openai_reply),
    )
    return model, lambda body: _text(body["messages"][-1]["content"])


def _anthropic(requests: list) -> tuple[Any, Callable[[Any], str]]:
    anthropic = pytest.importorskip("anthropic")
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(
        model="claude-sonnet-4-5",  # pyright: ignore[reportCallIssue]
        api_key=SecretStr("test-key"),
        max_tokens=1024,  # pyright: ignore[reportCallIssue]
    )  # pyright: ignore[reportCallIssue]
    # ChatAnthropic builds its client lazily in this cached property.
    vars(model)["_client"] = anthropic.Anthropic(
        api_key="test-key", http_client=_http(requests, _anthropic_reply)
    )
    return model, lambda body: _text(body["messages"][-1]["content"])


def _anthropic_vertex(requests: list) -> tuple[Any, Callable[[Any], str]]:
    from langchain_google_vertexai.model_garden import (  # pyright: ignore[reportMissingImports]
        ChatAnthropicVertex,
    )

    model = ChatAnthropicVertex(
        model_name="claude-sonnet-4-5",
        project="test-project",
        location="us-east5",
        max_tokens=1024,
        access_token="test-token",
        http_client=_http(requests, _anthropic_reply),
    )
    return model, lambda body: _text(body["messages"][-1]["content"])


def _bedrock(requests: list) -> tuple[Any, Callable[[Any], str]]:
    boto3 = pytest.importorskip("boto3")
    from botocore.stub import Stubber
    from langchain_aws import ChatBedrockConverse  # pyright: ignore[reportMissingImports]

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    client.meta.events.register(
        "provide-client-params.bedrock-runtime.Converse",
        lambda params, **_: requests.append(params),
    )
    stub = Stubber(client)
    for name in ("add", "Answer"):
        use = {"toolUseId": "tool-1", "name": name, "input": _args(name)}
        stub.add_response(
            "converse",
            {
                "output": {"message": {"role": "assistant", "content": [{"toolUse": use}]}},
                "stopReason": "tool_use",
                "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
                "metrics": {"latencyMs": 1},
            },
        )
    stub.activate()
    model = ChatBedrockConverse(
        model="anthropic.claude-sonnet-4-5", client=client, region_name="us-east-1"
    )
    return model, lambda params: _text(params["messages"][-1]["content"])


def _vertex(requests: list) -> tuple[Any, Callable[[Any], str]]:
    from google.cloud.aiplatform_v1beta1.types import (
        Candidate,
        Content,
        FunctionCall,
        GenerateContentResponse,
        Part,
    )
    from langchain_google_vertexai import ChatVertexAI  # pyright: ignore[reportMissingImports]

    class PredictionClient:
        def generate_content(self, request, **kwargs):
            del kwargs
            requests.append(request)
            name = request.tools[0].function_declarations[0].name
            part = Part(function_call=FunctionCall(name=name, args=_args(name)))
            candidate = Candidate(content=Content(role="model", parts=[part]), finish_reason=1)
            return GenerateContentResponse(candidates=[candidate])

    model = ChatVertexAI(model="gemini-2.5-pro", project="test-project", location="us-central1")
    model.client = PredictionClient()  # ChatVertexAI uses a client that is already set.
    return model, lambda request: " ".join(part.text for part in request.contents[-1].parts)


PROVIDERS = {
    "ChatOpenAI": ("langchain_openai", _openai),
    "AzureChatOpenAI": ("langchain_openai", _azure),
    "ChatAnthropic": ("langchain_anthropic", _anthropic),
    "ChatAnthropicVertex": ("langchain_google_vertexai", _anthropic_vertex),
    "ChatBedrockConverse": ("langchain_aws", _bedrock),
    "ChatVertexAI": ("langchain_google_vertexai", _vertex),
}


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_provider_requests_end_with_lessons_and_keep_tool_calls_and_structured_output(
    server, provider
):
    app, client = server
    package, build = PROVIDERS[provider]
    pytest.importorskip(package)
    requests: list = []
    native, last_text = build(requests)
    model = wrap_chat_model(native)

    with client.trace(provider, capture_logs=False) as trace_id:
        enable_lessons(app)
        reply = model.bind_tools([add]).invoke(HISTORY)
        parsed = model.with_structured_output(Answer).invoke("What is 1 + 2?")

    assert [call["name"] for call in reply.tool_calls] == ["add"]
    assert parsed == Answer(answer="3")
    assert len(requests) == 2 and all(LESSON_BLOCK in last_text(sent) for sent in requests)
    steps = recorded_steps(app, trace_id)
    assert [s.status for s in steps] == ["success", "success"]
    calls = [json_of(s.call_json) for s in steps]
    assert all(call["provider"] and call["model"] for call in calls)
    assert [t["name"] for t in calls[0]["input"]["tools"]] == ["add"]
    assert all(LESSON_BLOCK not in json.dumps(call) for call in calls)
