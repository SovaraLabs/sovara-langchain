"""LangChain tool-calling example backed by OpenAI.

Requires `OPENAI_API_KEY` in the environment.
"""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import sovara_langchain
from sovara import SovaraClient, auto_instrument

auto_instrument()
sovara_langchain.instrument_tools()


sovara_client = SovaraClient(project_name="debug-langchain")


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    city_normalized = city.strip().lower()
    weather_by_city = {
        "san francisco": "Sunny, 18C, light ocean breeze.",
        "new york": "Cloudy, 12C, chance of rain.",
        "zurich": "Cool, 10C, partly cloudy.",
    }
    return weather_by_city.get(city_normalized, f"No forecast is available for {city}.")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this example.")

    agent = create_agent(
        model=ChatOpenAI(
            model="gpt-5.4-mini",
            temperature=0,
            use_responses_api=False,
        ),
        tools=[get_weather],
        system_prompt="You are a concise assistant. Use tools when they help answer accurately.",
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in San Francisco? Keep it short.",
                }
            ]
        }
    )

    messages = result["messages"]
    print("LangChain tool agent succeeded.")
    print(f"Final answer: {messages[-1].content!r}")


if __name__ == "__main__":
    with sovara_client.trace("LangChain Tool Agent"):
        main()
