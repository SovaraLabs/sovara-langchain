import { AIMessage, HumanMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";
import { AsyncLocalStorageProviderSingleton } from "@langchain/core/singletons";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import type {
  ContentBlock,
  ConversationInput,
  ConversationOutput,
  Message,
  ProviderBlock,
  ToolSpec,
  Usage,
  Execution,
  JsonObject,
  LlmCall,
  Output,
} from "@sovara/sdk";
import { isCaptureActive } from "@sovara/sdk/integration";
import { jsonValue, type Ending } from "@sovara/sdk/integration";
import { warnOnce } from "./log.js";

/**
 * Convert LangChain messages, tools and responses into Sovara conversation data.
 *
 * Content is read through LangChain's standard `contentBlocks`, so provider-specific formats
 * are normalized by LangChain. Tool calls come from `AIMessage.tool_calls`, which LangChain
 * keeps authoritative for every provider. Also holds the step helpers the middleware and the
 * chat-model wrapper share.
 */

export const METADATA = { framework: "langchain" };

/**
 * Where in a LangGraph graph the current call runs, or nothing outside one.
 *
 * `langgraph_nodes` lists the node names from the outermost graph in, so a supervisor's
 * sub-assistant shows as `["sql_assistant", "agent"]`. `langgraph_namespace` also carries
 * LangGraph's task IDs, so it tells repeated or parallel delegations apart.
 */
export function graphLocation(): JsonObject {
  const metadata = AsyncLocalStorageProviderSingleton.getRunnableConfig()?.metadata as
    | Record<string, unknown>
    | undefined;
  const namespace = metadata?.langgraph_checkpoint_ns;
  if (typeof namespace !== "string" || !namespace) return {};
  const nodes = namespace.split("|").filter(Boolean).map((part) => part.split(":")[0]!);
  return { langgraph_nodes: nodes, langgraph_namespace: namespace };
}

/** The metadata of a recorded LangChain model or tool call. */
export function stepMetadata(): JsonObject {
  return { ...METADATA, ...graphLocation() };
}

const ROLES: Record<string, Message["role"]> = {
  system: "system",
  human: "user",
  ai: "assistant",
  tool: "tool",
};
const CHAT_ROLES = new Set(["system", "developer", "user", "assistant", "tool"]);
const MEDIA = new Set(["image", "audio", "video", "file"]);

type Block = Record<string, unknown>;

/** A tool call ID, or null when it is missing or blank. */
export function callId(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  return String(value).trim() || null;
}

function role(message: BaseMessage): Message["role"] {
  if (message.type === "generic") {
    const named = (message as { role?: unknown }).role;
    return typeof named === "string" && CHAT_ROLES.has(named) ? (named as Message["role"]) : "user";
  }
  return ROLES[message.type] ?? "user";
}

function provider(block: Block): ProviderBlock {
  return { type: "provider_block", format: `langchain.${String(block.type)}`, data: jsonValue(block) };
}

function base64(data: unknown): string | null {
  if (typeof data === "string") return data || null;
  if (data instanceof Uint8Array) return Buffer.from(data).toString("base64") || null;
  return null;
}

function media(block: Block): ContentBlock {
  const url = typeof block.url === "string" && block.url ? block.url : null;
  const data = base64(block.data);
  if ((url === null) === (data === null)) return provider(block);
  const kind = String(block.type);
  const mime = block.mimeType ?? block.mime_type;
  return {
    type: "media",
    mime_type: typeof mime === "string" && mime ? mime : kind === "file" ? "application/octet-stream" : `${kind}/*`,
    url,
    data,
  };
}

function block(raw: unknown): ContentBlock | null {
  if (typeof raw === "string") return raw ? { type: "text", text: raw } : null;
  if (!raw || typeof raw !== "object") return null;
  const value = raw as Block;
  switch (value.type) {
    case "text":
      return value.text ? { type: "text", text: String(value.text) } : null;
    case "reasoning":
      return { type: "reasoning", text: String(value.reasoning ?? "") };
    case "tool_call":
      return null;
    default:
      return MEDIA.has(String(value.type)) ? media(value) : provider(value);
  }
}

function blocks(message: BaseMessage): ContentBlock[] {
  return message.contentBlocks.map(block).filter((item): item is ContentBlock => item !== null);
}

/** Convert one LangChain message. */
export function message(item: BaseMessage): Message {
  if (ToolMessage.isInstance(item)) {
    const content = blocks(item);
    const texts = content.flatMap((part) => (part.type === "text" ? [part.text] : []));
    return {
      role: "tool",
      content: [
        {
          type: "tool_result",
          output: texts.length === content.length ? texts.join("") : jsonValue(item.content),
          call_id: callId(item.tool_call_id),
          is_error: item.status === "error",
        },
      ],
    };
  }
  const content = blocks(item);
  if (AIMessage.isInstance(item)) {
    for (const call of item.tool_calls ?? []) {
      content.push({ type: "tool_call", name: call.name, arguments: jsonValue(call.args), call_id: callId(call.id) });
    }
  }
  return { role: role(item), content };
}

function tool(value: unknown): ToolSpec {
  try {
    const fn = convertToOpenAITool(value as Parameters<typeof convertToOpenAITool>[0]).function;
    const parameters = jsonValue(fn.parameters ?? {});
    return {
      name: fn.name,
      description: fn.description ?? null,
      parameters: parameters && typeof parameters === "object" && !Array.isArray(parameters) ? parameters : {},
    };
  } catch {
    const named = value && typeof value === "object" ? ((value as Block).name ?? (value as Block).type) : undefined;
    return { name: typeof named === "string" && named ? named : "tool" };
  }
}

/** Convert a model request: optional system message, messages and tools. */
export function conversation(
  messages: readonly BaseMessage[],
  tools: readonly unknown[] = [],
  system?: BaseMessage | null,
): ConversationInput {
  const prefix = system && blocks(system).length > 0 ? [system] : [];
  return {
    format: "conversation",
    messages: [...prefix, ...messages].map(message),
    tools: tools.map(tool),
  };
}

function count(value: unknown): number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : 0;
}

function usage(response: AIMessage): Usage | null {
  const metadata = response.usage_metadata;
  if (!metadata) return null;
  const details = metadata.input_token_details ?? {};
  const cacheRead = details.cache_read ?? null;
  const cacheWrite = details.cache_creation ?? null;
  const raw = metadata.input_tokens;
  return {
    input_tokens: typeof raw === "number" ? Math.max(0, raw - count(cacheRead) - count(cacheWrite)) : null,
    output_tokens: metadata.output_tokens ?? null,
    cache_read_tokens: cacheRead,
    cache_write_tokens: cacheWrite,
  };
}

/**
 * Convert a model response; null becomes an empty assistant message.
 *
 * `generationInfo` is metadata LangChain merges into the message only after the model returns,
 * such as the OpenAI finish reason.
 */
export function output(response: AIMessage | null, generationInfo?: Record<string, unknown>): ConversationOutput {
  if (!response) return { message: { role: "assistant", content: [] } };
  const metadata = { ...generationInfo, ...response.response_metadata } as Block;
  const stop = metadata.finish_reason ?? metadata.stop_reason;
  return {
    message: message(response),
    usage: usage(response),
    stop_reason: typeof stop === "string" ? stop : null,
  };
}

/** The model's provider and name from its LangSmith parameters. */
export function modelIdentity(model: unknown): { provider: string | null; model: string | null } {
  try {
    const params = (model as { getLsParams?: (options: object) => Block }).getLsParams?.({}) ?? {};
    const text = (value: unknown) => (typeof value === "string" && value.trim() ? value : null);
    return { provider: text(params.ls_provider), model: text(params.ls_model_name) };
  } catch {
    return { provider: null, model: null };
  }
}

/** A copy with one user message holding lessons, before an assistant prefill. */
export function withLessons(messages: readonly BaseMessage[], text: string): BaseMessage[] {
  const temporary = [...messages];
  const last = temporary.at(-1);
  const prefill =
    last !== undefined && AIMessage.isInstance(last) && !last.tool_calls?.length && blocks(last).length > 0;
  temporary.splice(prefill ? temporary.length - 1 : temporary.length, 0, new HumanMessage({ content: text }));
  return temporary;
}

export function errorName(error: unknown): string {
  return error instanceof Error ? error.name : typeof error;
}

/**
 * Describe a model call, or return null to run it unrecorded: outside a recording, while capture
 * is suppressed (for example inside a call the middleware owns) or when conversion fails.
 */
export function llmCall(
  model: unknown,
  messages: readonly BaseMessage[],
  tools: readonly unknown[] = [],
  system?: BaseMessage,
): LlmCall | null {
  if (!isCaptureActive()) return null;
  try {
    const identity = modelIdentity(model);
    return {
      kind: "llm",
      input: conversation(messages, tools, system),
      provider: identity.provider,
      model: identity.model,
      metadata: stepMetadata(),
    };
  } catch (error) {
    warnOnce("langchain-convert", `[sovara] Could not record a LangChain model call. error_type=${errorName(error)}`);
    return null;
  }
}

/** Convert a model response, recording an empty one if conversion fails. */
export function safeOutput(response: AIMessage | null, generationInfo?: Record<string, unknown>): Output {
  try {
    return output(response, generationInfo);
  } catch (error) {
    warnOnce("langchain-output", `[sovara] Could not record a LangChain model response. error_type=${errorName(error)}`);
    return output(null);
  }
}

export function bubblesUp(error: unknown): boolean {
  return (error as { is_bubble_up?: unknown } | null)?.is_bubble_up === true;
}

/** The step ending and arguments for an error leaving a call. */
export function raised(error: unknown, options: { execution?: Execution; output?: Output } = {}): [Ending, unknown[]] {
  if (bubblesUp(error)) return ["cancel", ["Interrupted", options]];
  if (errorName(error) === "AbortError") return ["cancel", ["Aborted", options]];
  return ["fail", [error, options]];
}
