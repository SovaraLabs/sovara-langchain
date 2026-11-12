# Sovara for LangChain (TypeScript)

Record LangChain and LangGraph model and tool calls in Sovara, and add relevant lessons to each
model request.

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

- Lessons are one temporary user message at the end of the request. The agent's messages are
  never changed.
- Outside a trace, calls pass through unrecorded. If Sovara is unavailable, calls still run.
- Last in the list, the middleware records the model, prompt and messages that other middleware
  settled on, and each retry by other middleware as its own step.
- Don't use the middleware and the wrapper for the same model.
- Model calls are recorded as conversations, so **Rerun** is not available for them.

Requires Node.js 24.22.1+, `@langchain/core` 1.2+ (`langchain` 1.5+ for the middleware) and an
exec server with runner-api 6.

## Development

See [Development](https://github.com/SovaraLabs/sovara-langchain#development).

## License

MIT.
