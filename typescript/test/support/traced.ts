import { SovaraClient } from "@sovara/sdk";
import { fakeExecServer, type FakeExecServerOptions, type FakeRequest } from "./fakeExecServer.js";

/** Run fn inside a trace recorded by a fresh fake exec server. */
export async function traced<T>(fn: () => Promise<T>, options: FakeExecServerOptions = {}) {
  const server = fakeExecServer(options);
  const client = new SovaraClient({ projectName: "Project", url: server.url, fetch: server.fetch });
  const result = await client.trace("trace", fn, { captureLogs: false });
  return { server, result, ...stepsOf(server.requests) };
}

/** The step prepares (POST .../steps) and finalizes, in order. */
export function stepsOf(requests: FakeRequest[]) {
  const prepares = requests.filter((r) => /\/exec\/traces\/[^/]+\/steps$/.test(r.path)).map((r) => r.body!);
  const finalizes = requests.filter((r) => r.path.endsWith("/finalize")).map((r) => r.body!);
  return { prepares, finalizes, calls: prepares.map((body) => (body.call as { value: unknown }).value) };
}

/** The inline value of a finalize field such as outcome or execution. */
export const inline = (body: Record<string, unknown>, key: string) =>
  (body[key] as { value?: unknown } | null)?.value ?? null;
