import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, HumanMessage, SystemMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { DynamicTool, tool, type StructuredTool } from "@langchain/core/tools";
import { createAgent, createMiddleware } from "langchain";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { instrumentTools, sovaraMiddleware } from "../src/index.js";
import { wrapChatModel } from "../src/chatModel.js";
import { conversation, message } from "../src/messages.js";
import { inline, traced } from "./support/traced.js";

const LESSONS = { retrieval_id: "retrieval-1", text: "<lessons>Prefer small diffs.</lessons>" };
const QUESTION = "Why does CI fail?";

class ScriptedModel extends BaseChatModel {
  seen: BaseMessage[][] = [];
  replies: AIMessage[];
  error: string | undefined;

  constructor(replies: AIMessage[], error?: string) {
    super({});
    this.replies = replies;
    this.error = error;
  }

  _llmType() {
    return "scripted";
  }

  async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    this.seen.push([...messages]);
    if (this.error) throw new Error(this.error);
    const reply = this.replies[this.seen.length - 1]!;
    return { generations: [{ message: reply, text: "" }] };
  }

  override bindTools() {
    return this;
  }
}

const calling = (name: string) =>
  new AIMessage({ content: "", tool_calls: [{ name, args: { topic: "ci" }, id: "call-1" }] });

const lookup = tool(async ({ topic }) => `policy for ${topic}`, {
  name: "lookup",
  description: "Look up the team policy for a topic.",
  schema: z.object({ topic: z.string() }),
});

const broken = tool(
  async ({ topic }): Promise<string> => {
    throw new Error(`no policy for ${topic}`);
  },
  { name: "broken", description: "Fail for every topic.", schema: z.object({ topic: z.string() }) },
);

const ask = { messages: [new HumanMessage(QUESTION)] };
const lessonLast = (sent: BaseMessage[]) =>
  HumanMessage.isInstance(sent.at(-1)) && sent.at(-1)!.content === LESSONS.text;

// Repeated setup must not record a tool twice.
instrumentTools();
instrumentTools();

/** A scripted model with a name, that fails its first `failures` calls. */
class NamedModel extends ScriptedModel {
  constructor(
    readonly label: string,
    replies: AIMessage[],
    readonly failures = 0,
  ) {
    super(replies);
  }

  override getLsParams() {
    return { ls_provider: "scripted", ls_model_name: this.label, ls_model_type: "chat" as const };
  }

  override async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    if (this.seen.length < this.failures) {
      this.seen.push([...messages]);
      throw new Error("rate limited");
    }
    return super._generate(messages);
  }
}

describe("LangChain createAgent middleware", () => {
  it("records model and tool steps and sends lessons only on the model request", async () => {
    const first = calling("lookup");
    first.usage_metadata = {
      input_tokens: 30,
      output_tokens: 5,
      total_tokens: 35,
      input_token_details: { cache_read: 10 },
    };
    const model = new ScriptedModel([first, new AIMessage("Rerun the job.")]);
    const agent = createAgent({ model, tools: [lookup], systemPrompt: "Be brief.", middleware: [sovaraMiddleware()] });

    const { result, calls, finalizes } = await traced(() => agent.invoke(ask), { lessonContext: LESSONS });

    expect(result.messages.at(-1)?.content).toBe("Rerun the job.");
    expect(result.messages.some((m) => m.content === LESSONS.text)).toBe(false);
    expect(model.seen.map(lessonLast)).toEqual([true, true]);
    expect(SystemMessage.isInstance(model.seen[0]![0])).toBe(true);

    expect(calls.map((call) => (call as { kind: string }).kind)).toEqual(["llm", "tool", "llm"]);
    const llm = calls[0] as { input: { messages: { role: string }[]; tools: { name: string }[] }; metadata: unknown };
    expect(llm.metadata).toEqual({
      framework: "langchain",
      langgraph_nodes: ["model_request"],
      langgraph_namespace: expect.stringMatching(/^model_request:/),
    });
    expect(llm.input.messages.map((m) => m.role)).toEqual(["system", "user"]);
    expect(llm.input.tools.map((t) => t.name)).toEqual(["lookup"]);
    expect(JSON.stringify(llm)).not.toContain(LESSONS.text);
    expect(calls[1]).toMatchObject({ kind: "tool", name: "lookup", call_id: "call-1", arguments: { topic: "ci" } });

    const execution = inline(finalizes[0]!, "execution") as { injected_retrieval_id: string; input: unknown };
    expect(execution.injected_retrieval_id).toBe("retrieval-1");
    expect(JSON.stringify(execution.input)).toContain(LESSONS.text);
    const outcome = inline(finalizes[0]!, "outcome") as { output: { usage: unknown } };
    expect(outcome.output.usage).toEqual({
      input_tokens: 20,
      output_tokens: 5,
      cache_read_tokens: 10,
      cache_write_tokens: null,
    });
    expect(inline(finalizes[1]!, "outcome")).toEqual({ status: "success", output: "policy for ci" });
  });

  it("keeps the agent's tool error handling and records the failure", async () => {
    const replies = () => [calling("broken"), new AIMessage("Gave up.")];
    const recorded = new ScriptedModel(replies());
    const agent = createAgent({ model: recorded, tools: [broken], middleware: [sovaraMiddleware()] });
    const { result, finalizes } = await traced(() => agent.invoke(ask));

    expect(result.messages.at(-1)?.content).toBe("Gave up.");
    const sent = recorded.seen[1]!.at(-1) as ToolMessage;
    expect(sent.status).toBe("error");
    expect(sent.content).toBe("Error: no policy for ci\n Please fix your mistakes.");
    expect(inline(finalizes[1]!, "outcome")).toMatchObject({
      status: "error",
      error: { message: "no policy for ci" },
      output: "Error: no policy for ci\n Please fix your mistakes.",
    });

    const unrecorded = new ScriptedModel(replies());
    const plain = createAgent({ model: unrecorded, tools: [broken], middleware: [sovaraMiddleware()] });
    expect((await plain.invoke(ask)).messages.at(-1)?.content).toBe("Gave up.");
    expect(unrecorded.seen.map((sent) => sent.length)).toEqual([1, 3]);
  });

  it("records a failed model call and rethrows its error", async () => {
    const agent = createAgent({ model: new ScriptedModel([], "provider down"), tools: [], middleware: [sovaraMiddleware()] });
    const { finalizes } = await traced(async () => {
      await expect(agent.invoke(ask)).rejects.toThrow("provider down");
    });
    expect(inline(finalizes[0]!, "outcome")).toMatchObject({ status: "error", error: { message: "provider down" } });
  });

  it("placed last, records what earlier middleware sent: switched model, new prompt, each retry", async () => {
    const original = new NamedModel("model-a", []);
    const routed = new NamedModel("model-b", [new AIMessage("never sent"), new AIMessage("Rerun the job.")], 1);
    const route = createMiddleware({
      name: "Route",
      wrapModelCall: (request, handler) =>
        handler({ ...request, model: routed, systemMessage: new SystemMessage("Answer in one line.") }),
    });
    const retry = createMiddleware({
      name: "Retry",
      wrapModelCall: async (request, handler) => {
        try {
          return await handler(request);
        } catch {
          return handler(request);
        }
      },
    });
    const agent = createAgent({
      model: original,
      tools: [],
      systemPrompt: "Be brief.",
      middleware: [route, retry, sovaraMiddleware()],
    });

    const { result, calls, finalizes } = await traced(() => agent.invoke(ask), { lessonContext: LESSONS });

    // The model that ran is the one recorded, with the prompt it received.
    expect(result.messages.at(-1)?.content).toBe("Rerun the job.");
    expect(original.seen).toEqual([]);
    expect(routed.seen).toHaveLength(2);
    expect(calls).toHaveLength(2);
    for (const call of calls as { provider: string; model: string; input: unknown }[]) {
      expect(call).toMatchObject({ provider: "scripted", model: "model-b" });
      expect(JSON.stringify(call.input)).toContain("Answer in one line.");
      expect(JSON.stringify(call.input)).not.toContain("Be brief.");
    }

    // One step per attempt: the failure stays visible, then the retry succeeds.
    expect(finalizes.map((body) => (inline(body, "outcome") as { status: string }).status)).toEqual([
      "error",
      "success",
    ]);
    expect(inline(finalizes[0]!, "outcome")).toMatchObject({ error: { message: "rate limited" } });

    // Each attempt gets the lessons once, reported as executed, and never kept in the agent's state.
    for (const [index, sent] of routed.seen.entries()) {
      expect(lessonLast(sent)).toBe(true);
      expect(sent.filter((m) => m.content === LESSONS.text)).toHaveLength(1);
      expect((sent[0] as SystemMessage).content).toBe("Answer in one line.");
      const execution = inline(finalizes[index]!, "execution") as { injected_retrieval_id: string; input: unknown };
      expect(execution.injected_retrieval_id).toBe("retrieval-1");
      expect(JSON.stringify(execution.input)).toContain(LESSONS.text);
      expect(JSON.stringify(execution.input)).toContain("Answer in one line.");
    }
    expect(result.messages.some((m) => m.content === LESSONS.text)).toBe(false);
  });

  it("converts standard content blocks", () => {
    const thinking = new AIMessage({
      content: [
        { type: "thinking", thinking: "Check the logs.", signature: "sig" },
        { type: "text", text: "Calling lookup." },
      ],
      tool_calls: [{ name: "lookup", args: { topic: "ci" }, id: " " }],
      response_metadata: { model_provider: "anthropic" },
    });
    expect(message(thinking).content).toEqual([
      { type: "reasoning", text: "Check the logs." },
      { type: "text", text: "Calling lookup." },
      { type: "tool_call", name: "lookup", arguments: { topic: "ci" }, call_id: null },
    ]);
    const image = new HumanMessage({
      content: [
        { type: "image_url", image_url: { url: "data:image/png;base64,AAA" } },
        { type: "image", fileId: "file-1" },
      ],
    });
    const [media, byId] = message(image).content;
    expect(media).toEqual({ type: "media", mime_type: "image/png", url: null, data: "AAA" });
    expect(byId).toMatchObject({ type: "provider_block", format: "langchain.image" });
    expect(message(new ToolMessage({ content: "boom", tool_call_id: "call-1", status: "error" })).content).toEqual([
      { type: "tool_result", output: "boom", call_id: "call-1", is_error: true },
    ]);
    expect(conversation([], [], new SystemMessage("")).messages).toEqual([]);
  });
});

const echo = new DynamicTool({ name: "echo", description: "Echo text.", func: async (text) => text });

describe("tool instrumentation", () => {
  it.each([
    { tool: lookup, input: { topic: "ci" }, args: { topic: "ci" }, id: null, output: "policy for ci" },
    { tool: lookup, input: { type: "tool_call", name: "lookup", args: { topic: "ci" }, id: "call-1" }, args: { topic: "ci" }, id: "call-1", output: "policy for ci" },
    { tool: echo, input: "hello", args: { input: "hello" }, id: null, output: "hello" },
  ])("records direct $tool.name calls ($id)", async ({ tool, input, args, id, output }) => {
    const { result, calls, finalizes } = await traced(() => (tool as StructuredTool).invoke(input));
    expect(ToolMessage.isInstance(result) ? result.content : result).toBe(output);
    expect(calls).toEqual([{ kind: "tool", name: tool.name, arguments: args, call_id: id, metadata: { framework: "langchain" } }]);
    expect(inline(finalizes[0]!, "outcome")).toEqual({ status: "success", output });
  });

  it.each([
    { error: new Error("failed"), status: "error" },
    { error: Object.assign(new Error("cancelled"), { name: "AbortError" }), status: "cancelled" },
  ])("preserves tool $status", async ({ error, status }) => {
    const fails = tool(async () => { throw error; }, { name: "fails", description: "Fail.", schema: z.object({}) });
    const { calls, finalizes } = await traced(async () => {
      await expect(fails.invoke({})).rejects.toBe(error);
    });
    expect(calls).toHaveLength(1);
    expect(inline(finalizes[0]!, "outcome")).toMatchObject({ status });
    await expect(fails.invoke({})).rejects.toBe(error);
  });

  it("records concurrent nested tools once when middleware owns their parent", async () => {
    const nestedModel = wrapChatModel(new ScriptedModel([new AIMessage("nested")]));
    const outer = tool(async ({ topic }) => {
      await Promise.all([lookup.invoke({ topic }), lookup.invoke({ topic: "other" })]);
      return (await nestedModel.invoke(topic)).content;
    }, { name: "outer", description: "Look up two topics.", schema: z.object({ topic: z.string() }) });
    const model = new ScriptedModel([calling("outer"), new AIMessage("done")]);
    const agent = createAgent({ model, tools: [outer], middleware: [sovaraMiddleware()] });
    const { calls, finalizes } = await traced(() => agent.invoke(ask));
    expect(calls.map((call: any) => call.kind)).toEqual(["llm", "tool", "tool", "tool", "llm", "llm"]);
    expect(calls.slice(1, 4)).toMatchObject([
      { name: "outer", call_id: "call-1" },
      { name: "lookup", call_id: null, arguments: { topic: "ci" } },
      { name: "lookup", call_id: null, arguments: { topic: "other" } },
    ]);
    expect(finalizes.map((body) => inline(body, "outcome"))).toEqual(expect.arrayContaining([
      { status: "success", output: "policy for ci" },
      { status: "success", output: "policy for other" },
    ]));
  });
});
