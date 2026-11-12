# Changelog

## Unreleased

- First release, moved out of `@sovara/sdk` (`@sovara/sdk/langchain`): `sovaraMiddleware` for
  `createAgent` and `wrapChatModel` (`@sovara/langchain/chat-model`) for other chat-model use.
- Model and tool steps record the LangGraph nodes they ran in (`langgraph_nodes`, for example
  `["sql_assistant", "agent"]`) and LangGraph's namespace, so supervisor sub-assistants and
  each delegation can be told apart.
