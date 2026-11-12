import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { AIMessage, BaseMessage } from "@langchain/core/messages";
import type { ChatGenerationChunk, ChatResult } from "@langchain/core/outputs";
import {
  Runnable,
  RunnableAssign,
  RunnableBinding,
  RunnableMap,
  RunnableSequence,
  RunnableWithFallbacks,
} from "@langchain/core/runnables";
import type { Execution, Output } from "@sovara/sdk";
import { SovaraConfigError } from "@sovara/sdk";
import { begin, report } from "@sovara/sdk/integration";
import type { Step } from "@sovara/sdk";
import { conversation, llmCall, raised, safeOutput, withLessons } from "./messages.js";

/**
 * Wrap a LangChain chat model so every call it makes is recorded with lessons.
 *
 * `wrapChatModel(model)` returns a chat model that records each `invoke`, `stream`, `batch`
 * and streamed event call, including calls through `bindTools` and `withStructuredOutput`.
 * Lessons go into one temporary user message on that request only; the caller's messages and
 * graph state are never changed. Generation, tool binding, structured output and parsing stay
 * with the wrapped model. Needs only `@langchain/core`.
 */

type Options = BaseChatModel["ParsedCallOptions"];

interface Prepared {
  step: Step | null;
  messages: BaseMessage[];
  execution?: Execution;
}

async function prepare(model: BaseChatModel, messages: BaseMessage[], options: Options): Promise<Prepared> {
  const tools = (options as { tools?: unknown[] }).tools ?? [];
  const call = llmCall(model, messages, tools);
  const step = call ? await begin(call, { lessons: true }) : null;
  const context = step?.lessonContext;
  if (!step || !context) return { step, messages };
  const sent = withLessons(messages, context.text);
  return {
    step,
    messages: sent,
    execution: { input: conversation(sent, tools), injected_retrieval_id: context.retrieval_id },
  };
}

function resultOutput(result: ChatResult): Output {
  const generation = result.generations[0];
  return safeOutput((generation?.message as AIMessage | undefined) ?? null, generation?.generationInfo);
}

function chunksOutput(chunks: ChatGenerationChunk | undefined): Output {
  return safeOutput((chunks?.message as AIMessage | undefined) ?? null, chunks?.generationInfo);
}

function streamsNatively(model: BaseChatModel): boolean {
  return model._streamResponseChunks !== BaseChatModel.prototype._streamResponseChunks;
}

/** A shallow copy keeping the runnable's class and every field except those replaced. */
function copyWith<T extends object>(node: T, update: Record<string, unknown>): T {
  return Object.assign(Object.create(Object.getPrototypeOf(node)) as T, node, update);
}

/**
 * Whether `node` is a configured copy of `model`: some models (ChatOpenAI) implement
 * `bindTools` and `withStructuredOutput` by returning a clone with bound defaults instead of
 * a binding.
 */
function isConfiguredCopy(node: unknown, model: BaseChatModel): node is BaseChatModel {
  return node instanceof BaseChatModel && node !== model && node.constructor === model.constructor;
}

/**
 * `node` with the runnable `old` replaced by `next`, and any configured copy of `old` wrapped,
 * or null when neither is in it.
 */
function rebind(node: unknown, old: BaseChatModel, next: Runnable): unknown {
  if (node === old) return next;
  if (isConfiguredCopy(node, old)) return new SovaraChatModel(node);
  if (RunnableBinding.isRunnableBinding(node)) {
    const bound = rebind(node.bound, old, next);
    return bound === null ? null : copyWith(node, { bound });
  }
  if (node instanceof RunnableSequence) {
    // These fields are protected in LangChain's types but stable runtime properties.
    const { first, middle, last } = node as unknown as { first: unknown; middle: unknown[]; last: unknown };
    const steps = [first, ...middle, last];
    const rebound = steps.map((step) => rebind(step, old, next));
    if (rebound.every((step) => step === null)) return null;
    const kept = rebound.map((step, index) => step ?? steps[index]);
    return copyWith(node, { first: kept[0], middle: kept.slice(1, -1), last: kept.at(-1) });
  }
  if (node instanceof RunnableMap) {
    const { steps } = node as unknown as { steps: Record<string, unknown> };
    const entries = Object.entries(steps).map(([key, step]) => [key, step, rebind(step, old, next)] as const);
    if (entries.every(([, , step]) => step === null)) return null;
    return copyWith(node, { steps: Object.fromEntries(entries.map(([key, step, hit]) => [key, hit ?? step])) });
  }
  if (node instanceof RunnableWithFallbacks) {
    const runnable = rebind(node.runnable, old, next);
    const fallbacks = node.fallbacks.map((step) => rebind(step, old, next));
    if (runnable === null && fallbacks.every((step) => step === null)) return null;
    return copyWith(node, {
      runnable: runnable ?? node.runnable,
      fallbacks: fallbacks.map((step, index) => step ?? node.fallbacks[index]),
    });
  }
  if (node instanceof RunnableAssign) {
    const mapper = rebind(node.mapper, old, next);
    return mapper === null ? null : copyWith(node, { mapper });
  }
  return null;
}

/**
 * A chat model that records each call of the wrapped model as a Sovara step.
 *
 * Create it with `wrapChatModel`. Calls go to the wrapped model's own generation and streaming
 * methods; outside a recording they pass through unchanged.
 */
export class SovaraChatModel extends BaseChatModel {
  readonly model: BaseChatModel;

  static lc_name(): string {
    return "SovaraChatModel";
  }

  constructor(model: BaseChatModel) {
    super({
      callbacks: model.callbacks,
      tags: model.tags,
      metadata: model.metadata,
      verbose: model.verbose,
      cache: model.cache,
    });
    this.model = model;
    this.caller = model.caller;
    // LangChain streams only models that override _streamResponseChunks, as this one does.
    this.disableStreaming = model.disableStreaming || !streamsNatively(model);
  }

  _llmType(): string {
    // BaseChatModel calls this while constructing, before `model` is set.
    return this.model?._llmType() ?? "sovara";
  }

  override get callKeys(): string[] {
    return this.model.callKeys;
  }

  override get profile() {
    return this.model.profile;
  }

  override getLsParams(options: this["ParsedCallOptions"]) {
    return this.model.getLsParams(options);
  }

  override invocationParams(options?: this["ParsedCallOptions"]) {
    return this.model.invocationParams(options);
  }

  /**
   * `options` over the wrapped model's bound defaults. Models that bind tools or structured
   * output by cloning themselves (ChatOpenAI) apply those defaults only in `invoke` and `stream`,
   * which the wrapper bypasses.
   */
  private withDefaults(options: Options): Options {
    const defaults = (this.model as { defaultOptions?: Partial<Options> }).defaultOptions;
    return defaults ? { ...defaults, ...options } : options;
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    options = this.withDefaults(options);
    const { step, messages: sent, execution } = await prepare(this.model, messages, options);
    if (!step) return this.model._generate(messages, options, runManager);
    let result: ChatResult;
    try {
      result = await step.activate(() => this.model._generate(sent, options, runManager));
    } catch (error) {
      await report(step, ...raised(error, { execution }), { propagating: true });
      throw error;
    }
    await report(step, "finish", [resultOutput(result), { execution }]);
    return result;
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    options = this.withDefaults(options);
    const { step, messages: sent, execution } = await prepare(this.model, messages, options);
    if (!step) {
      yield* this.model._streamResponseChunks(messages, options, runManager);
      return;
    }
    const iterator = this.model._streamResponseChunks(sent, options, runManager);
    let chunks: ChatGenerationChunk | undefined;
    let ended = false;
    try {
      while (true) {
        // Activation covers each pull from the model, never the consumer's code.
        const next = await step.activate(() => iterator.next());
        if (next.done) break;
        chunks = chunks ? chunks.concat(next.value) : next.value;
        yield next.value;
      }
      ended = true;
    } catch (error) {
      ended = true;
      await report(step, ...raised(error, { execution, output: chunksOutput(chunks) }), { propagating: true });
      throw error;
    } finally {
      // Neither finished nor failed: the consumer stopped reading early.
      if (!ended) {
        await report(step, "cancel", ["Closed", { execution, output: chunksOutput(chunks) }], { propagating: true });
        await iterator.return?.(undefined);
      }
    }
    // Outside the try, so a failed report never marks the finished call as failed.
    await report(step, "finish", [chunksOutput(chunks), { execution }]);
  }

  override bindTools(tools: Parameters<NonNullable<BaseChatModel["bindTools"]>>[0], kwargs?: Partial<Options>) {
    const native: unknown = this.model.bindTools?.(tools, kwargs);
    if (isConfiguredCopy(native, this.model)) {
      return new SovaraChatModel(native) as unknown as ReturnType<NonNullable<BaseChatModel["bindTools"]>>;
    }
    if (!RunnableBinding.isRunnableBinding(native) || native.bound !== this.model) {
      throw new SovaraConfigError(
        `${this.model.constructor.name}.bindTools did not return a binding of the model, so Sovara cannot record its calls.`,
      );
    }
    return copyWith(native, { bound: this }) as ReturnType<NonNullable<BaseChatModel["bindTools"]>>;
  }

  /** Use the wrapped model's own structured output, keeping its calls recorded. */
  // `any` matches every overload of the base method.
  override withStructuredOutput(schema: any, config?: any): any {
    const native = this.model.withStructuredOutput(schema, config);
    const rebound = rebind(native, this.model, this);
    if (rebound === null) {
      throw new SovaraConfigError(
        `${this.model.constructor.name}.withStructuredOutput returned a runnable Sovara cannot record.`,
      );
    }
    return rebound;
  }
}

/**
 * A chat model that records every call of `model`; wrapping twice returns the same wrapper.
 *
 * The wrapper takes the wrapped model's callbacks, cache, retry settings and other chat-model
 * settings, since calls go to its generation methods directly.
 */
export function wrapChatModel(model: BaseChatModel): SovaraChatModel {
  if (model instanceof SovaraChatModel) return model;
  if (!(model instanceof BaseChatModel)) throw new SovaraConfigError("wrapChatModel() needs a LangChain BaseChatModel.");
  return new SovaraChatModel(model);
}
