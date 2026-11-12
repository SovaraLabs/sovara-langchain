import { AsyncLocalStorage } from "node:async_hooks";
import { isDirectToolOutput, ToolMessage } from "@langchain/core/messages";
import { StructuredTool } from "@langchain/core/tools";
import { begin, isCaptureActive, jsonValue, report, type Ending } from "@sovara/sdk/integration";
import { callId, raised, stepMetadata } from "./messages.js";

const ownedCall = new AsyncLocalStorage<string | undefined>();

/** Let middleware own this invocation while nested calls remain recordable. */
export function ownToolCall<T>(id: string | undefined, fn: () => T): T {
  return ownedCall.run(id, fn);
}

export function toolOutcome(result: unknown): [Ending, unknown[]] {
  if (ToolMessage.isInstance(result)) {
    const content = jsonValue(result.content);
    if (result.status === "error") {
      const message = typeof result.content === "string" ? result.content : null;
      return ["fail", [{ type: "ToolError", message }, { output: content }]];
    }
    return ["finish", [content]];
  }
  // LangGraph's marker avoids requiring LangGraph for tool-only consumers.
  if (isDirectToolOutput(result) && (result as { lg_name?: unknown }).lg_name === "Command") {
    const value = (result as { update?: { messages?: unknown } }).update?.messages;
    const messages = (Array.isArray(value) ? value : [value]).filter((item) => ToolMessage.isInstance(item));
    return ["finish", [messages.length ? jsonValue(messages.map((item) => item.content)) : null]];
  }
  return ["finish", [jsonValue(result)]];
}

const patched = Symbol.for("sovara.langchain.tools");

/** Record LangChain tool calls inside Sovara traces; call once at startup. */
export function instrumentTools(): void {
  const original = StructuredTool.prototype.call;
  if ((original as unknown as Record<symbol, boolean>)[patched]) return;
  const wrapped: typeof original = async function (this: StructuredTool, arg, config, tags) {
    const run = () => original.call(this, arg, config, tags);
    if (!isCaptureActive()) return run();
    const input = arg && typeof arg === "object" && "type" in arg && arg.type === "tool_call" ? arg : null;
    const configured = config && "toolCall" in config ? config.toolCall : null;
    const id = callId(input?.id ?? configured?.id);
    if (id && id === ownedCall.getStore()) return ownedCall.run(undefined, run);
    const step = await begin({
      kind: "tool",
      name: this.name,
      arguments: jsonValue(input ? input.args : arg),
      call_id: id,
      metadata: stepMetadata(),
    });
    let result: unknown;
    try {
      result = await run();
    } catch (error) {
      await report(step, ...raised(error), { propagating: true });
      throw error;
    }
    await report(step, ...toolOutcome(result));
    return result;
  };
  Object.defineProperty(wrapped, patched, { value: true });
  StructuredTool.prototype.call = wrapped;
}
