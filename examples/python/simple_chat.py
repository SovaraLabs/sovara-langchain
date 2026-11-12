from langchain.chat_models import init_chat_model
import sovara_langchain
from sovara import SovaraClient, auto_instrument

auto_instrument()
sovara_langchain.instrument_tools()


sovara_client = SovaraClient(project_name="debug-langchain")


def main():

    model = init_chat_model("gpt-5.1")
    response = model.invoke("Why do parrots talk? Keep it short.")

    prompt2 = f"Summarize the following. Keep it short.\n\n{response.content}"
    response2 = model.invoke(prompt2)

    print(response2.content)


if __name__ == "__main__":
    with sovara_client.trace("LangChain Simple Chat"):
        main()
