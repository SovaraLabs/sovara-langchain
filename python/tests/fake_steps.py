"""Fake step preparation, so instrumentation can be tested without a Sovara server."""

import threading
from contextlib import contextmanager
from typing import Any

from sovara import _recording, _steps
from sovara.runner import context_manager


class FakeStep:
    """Stands in for a prepared Step; records how instrumentation ended it."""

    def __init__(self, call: Any, *, lessons: bool):
        self.call = call
        self.recorded = True
        self.lessons = lessons
        self.prepared_request = None
        self.reports: list[tuple[str, Any, dict[str, Any]]] = []
        self.report_thread: int | None = None

    def activate(self):
        return context_manager.activate_step()

    def _report(self, ending: str, value: Any, kwargs: dict[str, Any]) -> None:
        self.report_thread = threading.get_ident()
        self.reports.append((ending, value, kwargs))

    def finish(self, output=None, **kwargs):
        self._report("finish", output, kwargs)

    def fail(self, error=None, **kwargs):
        self._report("fail", error, kwargs)

    def cancel(self, reason=None, **kwargs):
        self._report("cancel", reason, kwargs)

    async def afinish(self, output=None, **kwargs):
        self.finish(output, **kwargs)

    async def afail(self, error=None, **kwargs):
        self.fail(error, **kwargs)

    async def acancel(self, reason=None, **kwargs):
        self.cancel(reason, **kwargs)

    @property
    def ending(self) -> str:
        assert len(self.reports) == 1, self.reports
        return self.reports[0][0]

    @property
    def output(self) -> Any:
        ending, value, kwargs = self.reports[0]
        return value if ending == "finish" else kwargs.get("output")


def fake_steps(monkeypatch) -> list[FakeStep]:
    """Replace step preparation with FakeSteps; returns the steps started, in order."""
    steps: list[FakeStep] = []

    def start_step(call, *, lessons=False, step_uuid=None):
        del step_uuid
        step = FakeStep(call, lessons=lessons)
        steps.append(step)
        return step

    async def astart_step(call, *, lessons=False, step_uuid=None):
        return start_step(call, lessons=lessons, step_uuid=step_uuid)

    monkeypatch.setattr(_steps, "start_step", start_step)
    monkeypatch.setattr(_steps, "astart_step", astart_step)
    return steps


@contextmanager
def recording_context(trace_id: str = "trace-1", *, turn_id: str | None = None):
    """Bind a recording to the current context without contacting a server."""
    credentials = _recording.TraceWriteCredentials(
        trace_id=trace_id,
        recording_id=turn_id or trace_id,
        _trace_write_token=f"token-{turn_id or trace_id}",
    )
    recording = context_manager.RecordingContext(
        owner=object(),
        credentials=credentials,
        lesson_scope_paths=None,
    )
    with context_manager.bind_recording_context(recording):
        yield recording
