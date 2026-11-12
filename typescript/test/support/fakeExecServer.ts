import { randomUUID } from "node:crypto";

export type FakeRequest = {
  method: string;
  path: string;
  headers: Record<string, string>;
  body: Record<string, unknown> | undefined;
};

type Recording = { token: string; recordingId: string; closed: boolean; projectId: string };

type Override = (request: FakeRequest) => Response | undefined | Promise<Response | undefined>;

export type FakeExecServerOptions = {
  /** Answer a request before the default routes; return undefined to fall through. */
  override?: Override;
  lessonContext?: { retrieval_id: string; text: string } | null;
  preparedRequest?: Record<string, unknown> | null;
  contracts?: number[];
  features?: Record<string, unknown>;
};

const json = (payload: unknown, status = 200) => Response.json(payload, { status });
const apiError = (status: number, error: string, message = error) => json({ error, message }, status);

/** An in-memory runner-api 6 exec server answering through a fetch function. */
export function fakeExecServer(options: FakeExecServerOptions = {}) {
  const requests: FakeRequest[] = [];
  const recordings = new Map<string, Recording>();
  const replies = new Map<string, unknown>();
  let counter = 0;

  function open(): Record<string, unknown> {
    counter += 1;
    const traceId = `trace-${counter}`;
    const recording = {
      token: `token-${counter}`,
      recordingId: `recording-${counter}`,
      closed: false,
      projectId: "project-1",
    };
    recordings.set(traceId, recording);
    return {
      trace_id: traceId,
      recording_id: recording.recordingId,
      trace_write_token: recording.token,
      project_id: recording.projectId,
      appended: false,
    };
  }

  function route(request: FakeRequest): Response {
    const { method, path, body, headers } = request;
    if (method === "GET" && path === "/exec/health") {
      return json({
        status: "ok",
        service: "sovara-exec-server",
        compatibility: { component: "exec-server", contracts: { "runner-api": options.contracts ?? [5, 6] } },
        features: options.features ?? {},
      });
    }
    if (headers["x-sovara-contract-version"] !== "6") return apiError(409, "incompatible_contract");
    if (method === "POST" && path === "/exec/traces") return json(open());
    if (method === "POST" && path === "/exec/eval-runs") return json({ eval_run_id: `eval-${++counter}` });
    if (method === "POST" && path === "/exec/log-fields") return json({ ok: true });
    const match = /^\/exec\/traces\/([^/]+)(\/.*)$/.exec(path);
    const recording = match ? recordings.get(match[1]!) : undefined;
    if (!match || !recording) return apiError(404, "not_found");
    if (headers["x-sovara-trace-token"] !== recording.token) return apiError(403, "forbidden");
    if (recording.closed) return apiError(409, "recording_inactive");
    const rest = match[2]!;
    if (rest === "/complete") {
      recording.closed = true;
      return json({ ok: true });
    }
    if (rest === "/subtraces") return json(open());
    if (rest === "/turns") return json({ recording_id: recording.recordingId, turn_id: `turn-${++counter}` });
    if (rest === "/log") return json({ ok: true });
    if (rest === "/steps") {
      return json({
        ok: true,
        step_uuid: body?.step_uuid,
        lesson_context: body?.lessons ? options.lessonContext ?? null : null,
        prepared_request: body?.lessons ? options.preparedRequest ?? null : null,
      });
    }
    if (rest === "/steps/record") return json({ ok: true, step_uuid: body?.step_uuid });
    if (/^\/steps\/[^/]+\/finalize$/.test(rest)) return json({ ok: true });
    if (rest === "/lesson-retrievals") return json({ ok: true, lesson_context: options.lessonContext ?? null });
    if (rest === "/uploads") {
      const size = body?.total_size as number;
      return json({ upload_id: "upload-1", part_size: 4, part_count: Math.ceil(size / 4) });
    }
    if (/^\/uploads\/[^/]+\/(parts\/\d+|complete)$/.test(rest)) return json({ ok: true });
    return apiError(404, "not_found");
  }

  const fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(String(input));
    const headers = Object.fromEntries(new Headers(init?.headers).entries());
    const text = typeof init?.body === "string" ? init.body : undefined;
    const request: FakeRequest = {
      method: init?.method ?? "GET",
      path: url.pathname,
      headers,
      body: text ? JSON.parse(text) as Record<string, unknown> : undefined,
    };
    requests.push(request);
    const overridden = await options.override?.(request);
    if (overridden) return overridden;
    const key = headers["idempotency-key"];
    if (key && replies.has(`${request.path}:${key}`)) return json(replies.get(`${request.path}:${key}`));
    const response = route(request);
    if (key && response.ok) replies.set(`${request.path}:${key}`, await response.clone().json());
    return response;
  }) as typeof globalThis.fetch;

  return {
    fetch,
    requests,
    url: `http://fake-exec-${randomUUID()}.test`,
    /** Requests excluding health probes, as "METHOD path". */
    calls: () => requests.filter((r) => !r.path.endsWith("/health")).map((r) => `${r.method} ${r.path}`),
  };
}
