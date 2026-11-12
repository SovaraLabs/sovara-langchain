"""Package tests that need no Sovara server; end-to-end tests are in tests/e2e."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from sovara import (
    MediaBlock,
    ProviderBlock,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from sovara_langchain import SovaraChatModel, wrap_chat_model
from sovara_langchain import _messages as convert


def test_calls_pass_through_unrecorded_outside_a_trace():
    native = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
    model = wrap_chat_model(native)

    assert isinstance(model, SovaraChatModel) and wrap_chat_model(model) is model
    assert model.invoke("hello").content == "hi"


def test_messages_convert_through_standard_content_blocks():
    thinking = AIMessage(
        content=[
            {"type": "thinking", "thinking": "Check the logs.", "signature": "sig"},
            {"type": "text", "text": "Calling lookup."},
        ],
        tool_calls=[{"name": "lookup", "args": {"topic": "ci"}, "id": " "}],
        response_metadata={"model_provider": "anthropic"},
    )
    assert convert.message(thinking).content == (
        ReasoningBlock(text="Check the logs."),
        TextBlock(text="Calling lookup."),
        ToolCallBlock(name="lookup", arguments={"topic": "ci"}, call_id=None),
    )
    image = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "image", "file_id": "file-1"},
        ]
    )
    media, by_id = convert.message(image).content
    assert media == MediaBlock(mime_type="image/png", data="AAA")
    assert isinstance(by_id, ProviderBlock) and by_id.format == "langchain.image"
    failed = ToolMessage(content="boom", tool_call_id="call-1", status="error")
    assert convert.message(failed).content == (
        ToolResultBlock(output="boom", call_id="call-1", is_error=True),
    )
    assert convert.conversation([], system=SystemMessage(content="")).messages == ()
