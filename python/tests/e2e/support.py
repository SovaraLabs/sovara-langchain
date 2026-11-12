"""A real Sovara exec server for end-to-end tests, and helpers to read what it recorded.

The server is sovara's ``tests.integration_backend``, run from a sovara checkout:
``SOVARA_SOURCE_DIR`` or, by default, ``../sovara`` next to this repository. It needs that
checkout's ``uv sync`` and the dev Postgres (``./dev.sh`` there). It returns ``LESSON``
rendered as ``LESSON_BLOCK`` and never calls a model.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from sovara import SovaraClient, get_trace_id

LESSON_BLOCK = "<lessons>Prefer small diffs.</lessons>"
QUESTION = "Why does CI fail?"
# The test server accepts any credential.
AGENT_TOKEN = "e2e-test"


def sovara_source() -> Path | None:
    """Return the sovara checkout that runs the server, or None when there is none."""
    configured = os.environ.get("SOVARA_SOURCE_DIR")
    source = Path(configured) if configured else Path(__file__).resolve().parents[4] / "sovara"
    if (source / "tests" / "integration_backend.py").is_file():
        return source
    return None


@dataclass
class Backend:
    """A running test server and the step requests sent to it, by step UUID."""

    url: str
    sent_steps: dict[str, dict[str, str | None]] = field(default_factory=dict)

    def enable_lessons(self) -> None:
        """Turn on lessons for the current trace."""
        response = httpx.post(f"{self.url}/e2e/recordings/{get_trace_id()}/lessons")
        response.raise_for_status()

    def recording(self, trace_id: str | None) -> dict[str, Any]:
        """Return the trace's stored steps and lesson retrievals."""
        assert trace_id is not None
        response = httpx.get(f"{self.url}/e2e/recordings/{trace_id}", timeout=30)
        response.raise_for_status()
        return response.json()

    def client(self, project_name: str | None = None) -> SovaraClient:
        """Return a client for a project that records every step request it sends."""
        http = httpx.Client(base_url=self.url, event_hooks={"request": [self._capture]})
        return SovaraClient(
            project_name=project_name or f"sdk langchain {uuid.uuid4()}",
            base_url=self.url,
            agent_token=AGENT_TOKEN,
            http_client=http,
        )

    def async_http(self) -> httpx.AsyncClient:
        """Return an async HTTP client that records step requests like ``client``."""

        async def capture(request: httpx.Request) -> None:
            await request.aread()
            self._capture(request)

        return httpx.AsyncClient(base_url=self.url, event_hooks={"request": [capture]})

    def _capture(self, request: httpx.Request) -> None:
        path = request.url.path
        if request.method != "POST" or "/steps" not in path:
            return
        body = json.loads(request.read() or b"{}")
        step_uuid = body.get("step_uuid") or path.split("/steps/")[1].split("/")[0]
        values = self.sent_steps.setdefault(step_uuid, {})
        for name in ("call", "outcome", "execution"):
            if name in body:
                values[f"{name}_json"] = _sent(body[name])


def run_backend(source: Path) -> Iterator[Backend]:
    """Start the test server from ``source`` and stop it afterwards."""
    python = source / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    process = subprocess.Popen(
        [str(python), "-m", "tests.integration_backend"],
        cwd=source,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        # Empty when the server exits first; its error is in the output above.
        lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True).start()
        url = lines.get(timeout=120).strip()
        assert url.startswith("http://"), "The Sovara test server did not start; see its output."
        httpx.get(f"{url}/health", timeout=30).raise_for_status()
        yield Backend(url)
    finally:
        process.terminate()
        process.wait(timeout=30)


class ScriptedModel(BaseChatModel):
    """Answer with scripted replies and remember every request it received."""

    replies: list[AIMessage] = Field(default_factory=list)
    seen: list[list] = Field(default_factory=list)
    error: str | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del stop, run_manager, kwargs
        self.seen.append(list(messages))
        if self.error:
            raise RuntimeError(self.error)
        return ChatResult(generations=[ChatGeneration(message=self.replies[len(self.seen) - 1])])

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(item) for item in tools], **kwargs)


def enable_lessons(app: Backend) -> None:
    """Turn on lesson retrieval for the current recording."""
    app.enable_lessons()


def json_of(text: str | None) -> Any:
    """Parse a stored JSON column that must be present."""
    assert text is not None
    return json.loads(text)


def _sent(value: Any) -> str | None:
    if isinstance(value, dict) and value.get("storage") == "inline":
        value = value["value"]
    return None if value is None else json.dumps(value)


def async_server_client(app: Backend) -> httpx.AsyncClient:
    """Return an async client for the server that captures sent steps like the sync one."""
    return app.async_http()


@dataclass
class RecordedStep:
    """A stored step plus the call, outcome and execution JSON the SDK sent for it."""

    row: dict[str, Any]
    call_json: str | None = None
    outcome_json: str | None = None
    execution_json: str | None = None

    def __getattr__(self, name: str) -> Any:
        try:
            return self.row[name]
        except KeyError:
            raise AttributeError(name) from None


def recorded_steps(app: Backend, trace_id: str | None) -> list[RecordedStep]:
    """Return the trace's stored steps in recording order, with what the SDK sent."""
    rows = app.recording(trace_id)["steps"]
    return [RecordedStep(row, **app.sent_steps.get(row["step_uuid"], {})) for row in rows]


def applied_lessons(app: Backend, trace_id: str) -> set[str]:
    """Return the UUIDs of steps that applied lessons."""
    retrievals = app.recording(trace_id)["lesson_retrievals"]
    return {row["step_uuid"] for row in retrievals if row["applied_lessons"]}


def lesson_last(messages) -> bool:
    """Whether the request ends with the lessons message."""
    return isinstance(messages[-1], HumanMessage) and messages[-1].content == LESSON_BLOCK
