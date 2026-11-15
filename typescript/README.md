# Sovara for LangChain (TypeScript)

Sovara helps teams manage domain knowledge and put it to work in AI agents.
Capture business rules, expert guidance, and lessons from experience in a shared
knowledge base. Deliver relevant knowledge to agents as they work, review how
they use it, and improve it together.

Bring your team's domain knowledge to LangChain and LangGraph agents with Sovara.

```bash
npm install @sovara/langchain @sovara/sdk
```

## Use

Inside a Sovara trace, pick one per model:

```ts
import { sovaraMiddleware } from "@sovara/langchain";
import { instrumentTools } from "@sovara/langchain/tools";
import { wrapChatModel } from "@sovara/langchain/chat-model";

// createAgent: add the middleware last. It records model and tool calls.
const agent = createAgent({ model, tools, middleware: [...otherMiddleware, sovaraMiddleware()] });

// Anything else (LangGraph nodes, withStructuredOutput): wrap the model
// and call instrumentTools() once at startup to record tool calls too.
instrumentTools();
const wrapped = wrapChatModel(new ChatOpenAI({ model: "gpt-5.4" })).bindTools(tools);
```

Use either the middleware or the wrapper for a model, and keep the middleware
last. Calls outside a Sovara trace are not recorded; agents continue running if
Sovara is unavailable. Recorded model conversations do not support **Rerun**.

[Enable knowledge injection](https://docs.sovara-labs.com/sovaradb/auto-injection)
and see the [integration guide](https://docs.sovara-labs.com/integrations/langchain)
for supported usage. Runnable examples are in
[examples](https://github.com/SovaraLabs/sovara-langchain/tree/main/examples).

Requires Node.js 24.22.1+ and `@langchain/core` 1.2+ (`langchain` 1.5+ for the middleware).

## License

MIT.
