# Sovara for LangChain (Python)

Record LangChain and LangGraph model and tool calls in Sovara, and add relevant lessons to each
model request.

```bash
uv add "sovara-langchain[agent]"
# or: pip install "sovara-langchain[agent]"
```

Omit `[agent]` if you only use `wrap_chat_model`.

## Use

Inside a Sovara trace, pick one per model:

```python
import sovara_langchain
from sovara_langchain import SovaraMiddleware, wrap_chat_model

# create_agent: add the middleware last. It records model and tool calls.
agent = create_agent(model, tools, middleware=[*other_middleware, SovaraMiddleware()])

# Anything else (LangGraph nodes, create_react_agent, with_structured_output): wrap the model,
# and call instrument_tools() once at startup to record tool calls too.
sovara_langchain.instrument_tools()
model = wrap_chat_model(ChatOpenAI(model="gpt-5.4")).bind_tools(tools)
```

- `instrument_tools()` also records tools called directly, without a model-generated tool call.
- Lessons are one temporary user message at the end of the request. The agent's messages and
  graph state are never changed.
- Outside a trace, calls pass through unrecorded. If Sovara is unavailable, calls still run.
- Last in the list, the middleware records the model, prompt and messages that other middleware
  settled on, and each retry by other middleware as its own step.
- Don't use the middleware and the wrapper for the same model.
- Model calls are recorded as conversations, so **Rerun** is not available for them.

Requires `langchain-core` 1.x (`langchain` 1.1+ for the middleware) and an exec server with
runner-api 6.

## Development

See [Development](https://github.com/SovaraLabs/sovara-langchain#development).

## License

MIT.
