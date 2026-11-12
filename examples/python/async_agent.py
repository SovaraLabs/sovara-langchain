import asyncio
from langchain.agents import create_agent
import sovara_langchain
from sovara import SovaraClient, auto_instrument

auto_instrument()
sovara_langchain.instrument_tools()


sovara_client = SovaraClient(project_name="debug-langchain")


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def main():
    agent = create_agent(
        model="claude-sonnet-4-5-20250929",
        tools=[get_weather, multiply, add],
        system_prompt="You are a helpful assistant.",
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 7 * 8, then add 5? Also what's the weather in SF?",
                }
            ]
        }
    )
    print(result)


if __name__ == "__main__":

    async def _run():
        async with sovara_client.trace("LangChain Async Agent"):
            await main()

    asyncio.run(_run())
