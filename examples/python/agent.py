from langchain.agents import create_agent
import sovara_langchain
from sovara import SovaraClient, auto_instrument

auto_instrument()
sovara_langchain.instrument_tools()


sovara_client = SovaraClient(project_name="debug-langchain")


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def main():
    agent = create_agent(
        model="claude-sonnet-4-5-20250929",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    result = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})
    print(result)


if __name__ == "__main__":
    with sovara_client.trace("LangChain Agent"):
        main()
