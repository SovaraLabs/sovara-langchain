# Changelog

## Unreleased

- First release, moved out of the `sovara` SDK (`sovara.langchain`): `SovaraMiddleware` for
  `create_agent`, `wrap_chat_model` for other chat-model use, and `instrument_tools()` for tool calls.
- Model and tool steps record the LangGraph nodes they ran in (`langgraph_nodes`, for example
  `["sql_assistant", "agent"]`) and LangGraph's namespace, so supervisor sub-assistants and
  each delegation can be told apart.
