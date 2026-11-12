import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, AIMessageChunk, HumanMessage, type BaseMessage } from "@langchain/core/messages";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import { tool } from "@langchain/core/tools";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { END, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph";
import { createAgent } from "langchain";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { SovaraConfigError } from "@sovara/sdk";
import { SovaraChatModel, wrapChatModel } from "../src/chatModel.js";
import { sovaraMiddleware } from "../src/index.js";
import { inline, traced } from "./support/traced.js";

const LESSONS = { retrieval_id: "retrieval-1", text: "<lessons>Prefer small diffs.</lessons>" };
const QUESTION = "Why does CI fail?";

class ScriptedModel extends BaseChatModel {
  seen: BaseMessage[][] = [];
  options: Record<string, unknown>[] = [];
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

  async _generate(messages: BaseMessage[], options: this["ParsedCallOptions"]): Promise<ChatResult> {
    this.seen.push([...messages]);
    this.options.push(options as Record<string, unknown>);
    if (this.error) throw new Error(this.error);
    const reply = this.replies[this.seen.length - 1]!;
    return { generations: [{ message: reply, text: "", generationInfo: { finish_reason: "stop" } }] };
  }

  override bindTools(tools: Parameters<NonNullable<BaseChatModel["bindTools"]>>[0]) {
    return this.withConfig({ tools: tools.map((item) => convertToOpenAITool(item)) } as never);
  }
}

class StreamingModel extends ScriptedModel {
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    this.seen.push([...messages]);
    yield new ChatGenerationChunk({ message: new AIMessageChunk("Rerun"), text: "Rerun" });
    yield new ChatGenerationChunk({
      message: new AIMessageChunk(" the job."),
      text: " the job.",
      generationInfo: { finish_reason: "stop" },
    });
  }
}

/** Binds like ChatOpenAI: a copy holding defaultOptions, which only invoke and stream apply. */
class CloningModel extends ScriptedModel {
  defaultOptions: Record<string, unknown> = {};

  override withConfig(config: Record<string, unknown>): any {
    const copy = new CloningModel(this.replies);
    Object.assign(copy, { seen: this.seen, options: this.options });
    copy.defaultOptions = { ...this.defaultOptions, ...config };
    return copy;
  }

  override invoke(input: Parameters<BaseChatModel["invoke"]>[0], options?: Record<string, unknown>) {
    return super.invoke(input, { ...this.defaultOptions, ...options });
  }
}

const lookup = tool(async ({ topic }) => `policy for ${topic}`, {
  name: "lookup",
  description: "Look up the team policy for a topic.",
  schema: z.object({ topic: z.string() }),
});

// LangChain's default structured output parses only message chunks with a matching tool call.
const verdict = (args: Record<string, unknown>) =>
  new AIMessageChunk({ content: "", tool_calls: [{ name: "Verdict", args, id: "call-1" }] });
const Verdict = z.object({ cause: z.string() }).describe("The verdict on a CI failure.");
const structured = { name: "Verdict" };

const lessonLast = (sent: BaseMessage[]) =>
  HumanMessage.isInstance(sent.at(-1)) && sent.at(-1)!.content === LESSONS.text;
const outcomeOf = (finalize: Record<string, unknown>) =>
  inline(finalize, "outcome") as { status: string; reason?: string; output: { message: unknown; stop_reason: unknown } };

describe("LangChain chat-model wrapper", () => {
  it("names the sub-assistant and its delegation on each step of a supervisor graph", async () => {
    const native = new ScriptedModel([new AIMessage("a"), new AIMessage("b"), new AIMessage("c")]);
    const model = wrapChatModel(native);
    const assistant = new StateGraph(MessagesAnnotation)
      .addNode("agent", async (state) => ({ messages: [await model.invoke(state.messages)] }))
      .addEdge(START, "agent")
      .compile();
    // The supervisor answers itself, then delegates to two sub-assistants built from the same graph.
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("supervisor", async (state) => ({ messages: [await model.invoke(state.messages)] }))
      .addNode("sql_assistant", assistant)
      .addNode("reviewer", assistant)
      .addEdge(START, "supervisor")
      .addEdge("supervisor", "sql_assistant")
      .addEdge("sql_assistant", "reviewer")
      .addEdge("reviewer", END)
      .compile();

    const { calls } = await traced(() => graph.invoke({ messages: [new HumanMessage(QUESTION)] }));
    const located = calls.map((call) => (call as { metadata: Record<string, unknown> }).metadata);
    expect(located.map((m) => m.langgraph_nodes)).toEqual([
      ["supervisor"],
      ["sql_assistant", "agent"],
      ["reviewer", "agent"],
    ]);
    expect(new Set(located.map((m) => m.langgraph_namespace)).size).toBe(3);
  });

  it("records one step with lessons for a bound model in a LangGraph node", async () => {
    const native = new ScriptedModel([new AIMessage("Rerun the job.")]);
    const model = wrapChatModel(native).bindTools([lookup]);
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("answer", async (state) => ({ messages: [await model.invoke(state.messages)] }))
      .addEdge(START, "answer")
      .addEdge("answer", END)
      .compile();

    const { result, calls, finalizes } = await traced(
      () => graph.invoke({ messages: [new HumanMessage(QUESTION)] }),
      { lessonContext: LESSONS },
    );

    expect(result.messages.map((m) => m.content)).toEqual([QUESTION, "Rerun the job."]);
    expect(lessonLast(native.seen[0]!)).toBe(true);
    expect((native.options[0]!.tools as { function: { name: string } }[])[0]!.function.name).toBe("lookup");
    expect(calls).toHaveLength(1);
    const call = calls[0] as { input: { tools: { name: string }[] }; metadata: unknown };
    expect(call.input.tools.map((t) => t.name)).toEqual(["lookup"]);
    expect(call.metadata).toEqual({
      framework: "langchain",
      langgraph_nodes: ["answer"],
      langgraph_namespace: expect.stringMatching(/^answer:/),
    });
    expect(JSON.stringify(call)).not.toContain(LESSONS.text);
    expect(inline(finalizes[0]!, "execution")).toMatchObject({ injected_retrieval_id: "retrieval-1" });
    expect(outcomeOf(finalizes[0]!).output.stop_reason).toBe("stop");
  });

  it("records streams as their aggregate, cancels early closes and falls back without native streaming", async () => {
    const model = wrapChatModel(new StreamingModel([]));
    const plain = wrapChatModel(new ScriptedModel([new AIMessage("Rerun the job.")]));

    const { finalizes } = await traced(async () => {
      let text = "";
      for await (const chunk of await model.stream(QUESTION)) text += chunk.content;
      expect(text).toBe("Rerun the job.");
      for await (const chunk of await model.stream(QUESTION)) {
        expect(chunk.content).toBe("Rerun");
        break;
      }
      const chunks = [];
      for await (const chunk of await plain.stream(QUESTION)) chunks.push(chunk.content);
      expect(chunks).toEqual(["Rerun the job."]);
    });

    const [done, closed, fallback] = finalizes.map(outcomeOf);
    expect(done).toMatchObject({
      status: "success",
      output: { message: { role: "assistant", content: [{ type: "text", text: "Rerun the job." }] }, stop_reason: "stop" },
    });
    expect(closed).toMatchObject({ status: "cancelled", reason: "Closed" });
    // LangChain's stream() reads ahead, so the partial output may hold more than was consumed.
    expect(JSON.stringify(closed!.output.message)).toContain('"text":"Rerun');
    expect(fallback!.status).toBe("success");
    expect(finalizes).toHaveLength(3);
  });

  it("keeps native structured output, include_raw and parse errors", async () => {
    const unparsable = new AIMessageChunk("No verdict.");
    const native = new ScriptedModel([verdict({ cause: "flaky" }), unparsable, unparsable]);
    const model = wrapChatModel(native);

    const { result, finalizes, calls } = await traced(
      async () => {
        const parsed = await model.withStructuredOutput(Verdict, structured).invoke(QUESTION);
        const raw = await model.withStructuredOutput(Verdict, { ...structured, includeRaw: true }).invoke(QUESTION);
        await expect(model.withStructuredOutput(Verdict, structured).invoke(QUESTION)).rejects.toThrow("No tool calls");
        return { parsed, raw };
      },
      { lessonContext: LESSONS },
    );

    expect(result.parsed).toEqual({ cause: "flaky" });
    expect(result.raw.parsed).toBeNull();
    expect(AIMessage.isInstance(result.raw.raw)).toBe(true);
    expect(native.seen.every(lessonLast)).toBe(true);
    expect(finalizes.map((f) => outcomeOf(f).status)).toEqual(["success", "success", "success"]);
    expect((calls[0] as { input: { tools: { name: string }[] } }).input.tools.map((t) => t.name)).toEqual(["Verdict"]);
  });

  it("records models that bind tools and structured output by copying themselves", async () => {
    const native = new CloningModel([new AIMessage("Rerun the job."), verdict({ cause: "flaky" })]);
    const model = wrapChatModel(native);

    const { result, calls, finalizes } = await traced(async () => {
      await model.bindTools([lookup]).invoke(QUESTION);
      return model.withStructuredOutput(Verdict, structured).invoke(QUESTION);
    });

    expect(result).toEqual({ cause: "flaky" });
    const sentTools = native.options.map((options) =>
      (options.tools as { function: { name: string } }[]).map((t) => t.function.name));
    expect(sentTools).toEqual([["lookup"], ["Verdict"]]);
    expect(calls.map((call) => (call as { input: { tools: { name: string }[] } }).input.tools.map((t) => t.name)))
      .toEqual([["lookup"], ["Verdict"]]);
    expect(finalizes.map((f) => outcomeOf(f).status)).toEqual(["success", "success"]);
  });

  it("passes failures through, leaves middleware-owned calls to the middleware and runs unrecorded outside a trace", async () => {
    const failing = wrapChatModel(new ScriptedModel([], "provider down"));
    const inner = new ScriptedModel([new AIMessage("one"), new AIMessage("two")]);
    const agent = createAgent({ model: wrapChatModel(inner), tools: [], middleware: [sovaraMiddleware()] });

    const { finalizes } = await traced(
      async () => {
        await expect(failing.invoke(QUESTION)).rejects.toThrow("provider down");
        await agent.invoke({ messages: [new HumanMessage(QUESTION)] });
      },
      { lessonContext: LESSONS },
    );

    expect(finalizes.map((f) => outcomeOf(f).status)).toEqual(["error", "success"]);
    expect(inner.seen[0]!.filter((m) => m.content === LESSONS.text)).toHaveLength(1);
    expect((await wrapChatModel(inner).invoke(QUESTION)).content).toBe("two");
  });

  it("delegates the model's identity and rejects what it cannot record", () => {
    const native = new ScriptedModel([]);
    native.tags = ["prod"];
    const model = wrapChatModel(native);
    expect(model).toBeInstanceOf(SovaraChatModel);
    expect(wrapChatModel(model)).toBe(model);
    expect(model._llmType()).toBe("scripted");
    expect(model.tags).toEqual(["prod"]);
    expect(model.getLsParams({} as never)).toEqual(native.getLsParams({} as never));

    class Unbindable extends ScriptedModel {
      override bindTools() {
        return new ScriptedModel([]).withConfig({});
      }
    }
    expect(() => wrapChatModel(new Unbindable([])).bindTools([lookup])).toThrow(SovaraConfigError);
    expect(() => wrapChatModel({} as BaseChatModel)).toThrow(SovaraConfigError);
  });
});
