from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import sovara_langchain
from sovara import SovaraClient, auto_instrument

auto_instrument()
sovara_langchain.instrument_tools()


STREAM_TOKEN = "SOVARA_STREAM_LANGGRAPH_OK"
sovara_client = SovaraClient(project_name="langgraph-streaming")


def _chunk_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def main():
    agent = create_react_agent(
        model=ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=16,
            use_responses_api=False,
        ),
        tools=[],
    )

    chunks = []
    stream = agent.stream(
        {"messages": [{"role": "user", "content": f"Output exactly {STREAM_TOKEN}."}]},
        stream_mode="messages",
    )
    for message_chunk, _metadata in stream:
        text = _chunk_text(message_chunk)
        if text:
            print(text, end="", flush=True)
            chunks.append(text)
    print()

    text = "".join(chunks)
    if STREAM_TOKEN not in text:
        raise RuntimeError("LangGraph stream did not include the expected probe token")


if __name__ == "__main__":
    with sovara_client.trace("LangGraph Streaming"):
        main()
