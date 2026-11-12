import { AIMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";
import { createMiddleware, ToolInvocationError } from "langchain";
import type { Execution, LlmCall, Output, ToolCall } from "@sovara/sdk";
import { begin, jsonValue, report } from "@sovara/sdk/integration";
import type { Step } from "@sovara/sdk";
import { ownToolCall, toolOutcome } from "./tools.js";
export { instrumentTools } from "./tools.js";
import {
  bubblesUp,
  callId,
  conversation,
  llmCall as describeCall,
  raised,
  safeOutput,
  stepMetadata,
  withLessons,
} from "./messages.js";

/**
 * Record LangChain `createAgent` model and tool calls, with lessons per model call.
 *
 * Requires `langchain` 1.5 or newer. Add the middleware last, so it records the request other
 * middleware finalized (model, prompt, messages) and each attempt of a middleware retry as its
 * own step:
 *
 * ```ts
 * const agent = createAgent({ model, tools, middleware: [...otherMiddleware, sovaraMiddleware()] });
 * await client.trace("support", () => client.turn(question, () => agent.invoke({ messages })));
 * ```
 *
 * Outside a recording the middleware passes calls through. Lessons go into one temporary
 * user message on the model request only; the agent's messages are never changed.
 */

interface ModelRequestLike {
  model: unknown;
  messages: BaseMessage[];
  systemMessage?: BaseMessage;
  tools?: unknown[];
}

interface ToolCallRequestLike {
  toolCall: { name: string; args?: unknown; id?: string };
  runtime?: { signal?: AbortSignal };
}

function llmCall(request: ModelRequestLike): LlmCall | null {
  return describeCall(request.model, request.messages, request.tools, request.systemMessage);
}

function lessons<T extends ModelRequestLike>(request: T, step: Step): { request: T; execution?: Execution } {
  const context = step.lessonContext;
  if (!context) return { request };
  const messages = withLessons(request.messages, context.text);
  return {
    request: { ...request, messages },
    execution: {
      input: conversation(messages, request.tools, request.systemMessage),
      injected_retrieval_id: context.retrieval_id,
    },
  };
}

function responseMessage(response: unknown): AIMessage | null {
  if (AIMessage.isInstance(response)) return response;
  const messages = (response as { messages?: unknown[] } | null)?.messages;
  return (Array.isArray(messages) ? messages.find((item) => AIMessage.isInstance(item)) : undefined) ?? null;
}

function modelOutput(response: unknown): Output {
  return safeOutput(responseMessage(response));
}

function toolCall(request: ToolCallRequestLike): ToolCall {
  return {
    kind: "tool",
    name: request.toolCall.name,
    arguments: jsonValue(request.toolCall.args),
    call_id: callId(request.toolCall.id),
    metadata: stepMetadata(),
  };
}

/** The error message the agent's default tool error handling would have returned. */
function toolError(request: ToolCallRequestLike, error: unknown): ToolMessage {
  return new ToolMessage({
    content: error instanceof ToolInvocationError ? error.message : `${error}\n Please fix your mistakes.`,
    tool_call_id: request.toolCall.id ?? "",
    name: request.toolCall.name,
    status: "error",
  });
}

/** Middleware that records each model and tool call of a LangChain agent as a Sovara step. */
export function sovaraMiddleware() {
  return createMiddleware({
    name: "SovaraMiddleware",
    wrapModelCall: async (request, handler) => {
      const call = llmCall(request);
      const step = call ? await begin(call, { lessons: true }) : null;
      if (!step) return handler(request);
      const prepared = lessons(request, step);
      let response: Awaited<ReturnType<typeof handler>>;
      try {
        response = await step.activate(() => handler(prepared.request));
      } catch (error) {
        const [ending, args] = raised(error, { execution: prepared.execution });
        await report(step, ending, args, { propagating: true });
        throw error;
      }
      await report(step, "finish", [modelOutput(response), { execution: prepared.execution }]);
      return response;
    },
    wrapToolCall: async (request, handler) => {
      const step = await begin(toolCall(request));
      let result: Awaited<ReturnType<typeof handler>>;
      try {
        // Not activated: model calls inside the tool are recorded as their own steps.
        result = await ownToolCall(request.toolCall.id, () => handler(request));
      } catch (error) {
        // Any wrapToolCall makes a thrown tool error fatal to the agent, so return the
        // message the agent would have produced without middleware.
        if (bubblesUp(error) || request.runtime?.signal?.aborted) {
          await report(step, ...raised(error), { propagating: true });
          throw error;
        }
        const message = toolError(request, error);
        await report(step, "fail", [error, { output: jsonValue(message.content) }], { propagating: true });
        return message;
      }
      await report(step, ...toolOutcome(result));
      return result;
    },
  });
}
