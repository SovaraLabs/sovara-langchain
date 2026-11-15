# Sovara for LangChain (Python)

Sovara helps teams manage domain knowledge and put it to work in AI agents.
Capture business rules, expert guidance, and lessons from experience in a shared
knowledge base. Deliver relevant knowledge to agents as they work, review how
they use it, and improve it together.

Bring your team's domain knowledge to LangChain and LangGraph agents with Sovara.

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

Use either the middleware or the wrapper for a model, and keep the middleware
last. Calls outside a Sovara trace are not recorded; agents continue running if
Sovara is unavailable. Recorded model conversations do not support **Rerun**.

[Enable knowledge injection](https://docs.sovara-labs.com/sovaradb/auto-injection)
and see the [integration guide](https://docs.sovara-labs.com/integrations/langchain)
for supported usage. Runnable examples are in
[examples](https://github.com/SovaraLabs/sovara-langchain/tree/main/examples).

Requires `langchain-core` 1.x (`langchain` 1.1+ for the middleware).

## License

MIT.
