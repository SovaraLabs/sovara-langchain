# Sovara in CodeMie

CodeMie records its LangChain and LangGraph model and tool calls with
[`sovara-langchain`](../../python), and Sovara adds matching lessons to
each model request.

- A conversation is a trace (`codemie:<conversation_id>`); each message is a turn.
- A workflow execution is a trace (`codemie:workflow:<execution_id>`); each run or resume is a
  turn, each agent-node run a subtrace named after its assistant, so parallel branches and retry
  attempts stay apart. Workflow-level model calls (for example the result finalizer) are recorded
  in the trace itself.
- An agent run that starts inside another recorded run is a subtrace of it.

The integration is LangChain-only. It never calls `sovara.auto_instrument()`, so `httpx`,
`requests`, `aiohttp` and `urllib3` are left untouched. `sovara_langchain.instrument_tools()` patches
only LangChain's tool execution; leave it out to record model calls without separate tool steps.

## Setup

1. Add `sovara-langchain` to CodeMie's dependencies, and set `SOVARA_ENABLED=true`.
2. Apply the change below: five small edits and a new `core/sovara.py`.
   - `get_llm_by_credentials` builds every chat model, including the `RouterChatModel`
     candidates, so wrapping there covers agents, the supervisor, LangGraph nodes, chains and
     `with_structured_output`, with one step per routed attempt.
   - `get_run_config` returns the observability context every agent run (`invoke`,
     `invoke_task`, `generate`, `stream`) enters; the Sovara turn joins it there, next to
     Langfuse or Phoenix.
   - `PureChatChain` passes that context to LangChain without entering it, so it enters it now.
     With Langfuse or Phoenix enabled, pure chat now also runs inside their trace context.
   - `WorkflowExecutor` wraps each workflow run, so the nodes join one execution trace.

```diff
--- a/src/codemie/configs/config.py
+++ b/src/codemie/configs/config.py
@@ -160,6 +160,10 @@
     VERTEX_AI_ANTHROPIC_ENABLE_PROMPT_CACHE: bool = False
 
+    # Record model calls with Sovara and add its lessons to each request.
+    SOVARA_ENABLED: bool = False
+    SOVARA_PROJECT: str = "codemie"
+
     ELASTIC_APPLICATION_INDEX: str = "applications"
--- a/src/codemie/core/dependecies.py
+++ b/src/codemie/core/dependecies.py
@@ -333,6 +333,12 @@
         logger.debug(f"Prompt caching not supported for Vertex AI Claude models: {llm_model}")
+
+    if config.SOVARA_ENABLED:
+        # Wrap last so the wrapper keeps the callbacks set above.
+        from sovara_langchain import wrap_chat_model
+
+        llm = wrap_chat_model(llm)
     return llm
--- a/src/codemie/agents/utils.py
+++ b/src/codemie/agents/utils.py
@@ -28,7 +28,7 @@
-from codemie.configs import logger
+from codemie.configs import config, logger
@@ -366,6 +366,28 @@
 def get_run_config(
+    request: Optional[AssistantChatRequest],
+    llm_model: str,
+    agent_name: str,
+    conversation_id: Optional[str] = None,
+    **kwargs,
+) -> Dict[str, Any]:
+    """Return the observability run configuration, with a Sovara turn when Sovara is enabled."""
+    run_config = _observability_run_config(request, llm_model, agent_name, conversation_id, **kwargs)
+    if config.SOVARA_ENABLED and conversation_id:
+        from codemie.core.sovara import sovara_turn
+
+        run_config["_trace_ctx"] = sovara_turn(
+            conversation_id,
+            request.text if request else None,
+            run_config.get("_trace_ctx"),
+            agent_name=agent_name,
+            workflow_execution_id=getattr(request, "workflow_execution_id", None),
+        )
+    return run_config
+
+
+def _observability_run_config(
     request: Optional[AssistantChatRequest],
--- a/src/codemie/chains/pure_chat_chain.py
+++ b/src/codemie/chains/pure_chat_chain.py
@@ -13,6 +13,7 @@
 import traceback
+import contextlib
 import json
@@ -65,10 +66,10 @@ class PureChatChain(StreamingChain):
     def invoke(self, inputs: Dict[str, Any]) -> GenerationResult:
-        response = self._chain.invoke(
-            input=inputs,
-            config=self._get_config(),
-        )
+        config = self._get_config()
+        # Enter the observability trace context, as the agents do.
+        with config.pop("_trace_ctx", contextlib.nullcontext()):
+            response = self._chain.invoke(input=inputs, config=config)
@@ -85,15 +86,13 @@ class PureChatChain(StreamingChain):
             chain = self._chain.with_config(callbacks=[ChainStreamingCallback(self.thread_generator)])
-            stream = chain.stream(
-                input=self._get_inputs(),
-                config=self._get_config(),
-            )
-            for chunk in stream:
-                if not chunk:
-                    continue
+            config = self._get_config()
+            with config.pop("_trace_ctx", contextlib.nullcontext()):
+                for chunk in chain.stream(input=self._get_inputs(), config=config):
+                    if not chunk:
+                        continue
 
-                chunks_collector.append(chunk)
+                    chunks_collector.append(chunk)
--- a/src/codemie/workflows/workflow.py
+++ b/src/codemie/workflows/workflow.py
@@ -1030,7 +1030,10 @@ class WorkflowExecutor:
                 try:
-                    self._run_workflow_execution(graph_config, chunks_collector)
+                    from codemie.core.sovara import sovara_workflow
+
+                    with sovara_workflow(self.execution_id, self.workflow_config.name, self.user_input):
+                        self._run_workflow_execution(graph_config, chunks_collector)
                     workflow_succeeded = True
```

New file `src/codemie/core/sovara.py`:

```python
"""Record CodeMie runs in Sovara: a trace per conversation or workflow execution.

- A chat message is a turn of its conversation's trace.
- Each workflow agent-node run is a subtrace of its execution's trace, named after the agent, so
  parallel branches and retried attempts stay apart.
- An agent run that starts inside another recorded run is a subtrace of it.
"""

from contextlib import ExitStack, contextmanager, nullcontext
from functools import lru_cache
from typing import Any, Iterator, Optional

from codemie.configs import config, logger


@lru_cache(maxsize=1)
def _client():
    import sovara_langchain
    from sovara import SovaraClient

    # Records tool calls too; model calls are recorded by the wrapper in get_llm_by_credentials.
    sovara_langchain.instrument_tools()
    return SovaraClient(project_name=config.SOVARA_PROJECT)


def _enter_recording(stack: ExitStack, client, run: dict) -> None:
    if client.get_trace_id() is not None:
        stack.enter_context(client.subtrace(run["agent_name"]))
    elif run["workflow_execution_id"]:
        trace_id = f"codemie:workflow:{run['workflow_execution_id']}"
        stack.enter_context(client.trace("CodeMie workflow", trace_id=trace_id, capture_logs=False))
        stack.enter_context(client.subtrace(run["agent_name"]))
    else:
        # capture_logs=False: concurrent requests share the process's stdout and stderr.
        trace_id = f"codemie:{run['conversation_id']}"
        stack.enter_context(client.trace(run["agent_name"], trace_id=trace_id, capture_logs=False))
        stack.enter_context(client.turn(turn_input=run["user_input"]))


@contextmanager
def sovara_turn(
    conversation_id: str,
    user_input: Optional[str],
    inner: Any = None,
    *,
    agent_name: Optional[str] = None,
    workflow_execution_id: Optional[str] = None,
) -> Iterator[None]:
    """Record the agent run in Sovara, then enter `inner` (the observability trace context).

    Sovara failures never fail the run: it continues unrecorded.
    """
    from sovara.integration import short_retries

    run = {
        "conversation_id": conversation_id,
        "user_input": user_input,
        "agent_name": agent_name or "CodeMie agent",
        "workflow_execution_id": workflow_execution_id,
    }
    with ExitStack() as stack:
        # Retry Sovara briefly: the default budget rides out a restart for up to a minute, which
        # would hold every chat request while Sovara is down.
        stack.enter_context(short_retries())
        try:
            _enter_recording(stack, _client(), run)
        except Exception as exc:
            logger.warning(f"Sovara recording unavailable; continuing unrecorded. error_type={type(exc).__name__}")
        if inner is not None:
            stack.enter_context(inner)
        yield


def sovara_workflow(execution_id: Optional[str], workflow_name: Optional[str], user_input: Optional[str]):
    """Record one workflow run (or resume) as a turn of its execution's trace.

    Agent nodes inside it become subtraces; workflow-level model calls (for example the result
    finalizer) are recorded in the trace itself.
    """
    if not (config.SOVARA_ENABLED and execution_id):
        return nullcontext()
    return _workflow_recording(execution_id, workflow_name or "CodeMie workflow", user_input)


@contextmanager
def _workflow_recording(execution_id: str, workflow_name: str, user_input: Optional[str]) -> Iterator[None]:
    from sovara.integration import short_retries

    with ExitStack() as stack:
        stack.enter_context(short_retries())
        try:
            client = _client()
            trace_id = f"codemie:workflow:{execution_id}"
            stack.enter_context(client.trace(workflow_name, trace_id=trace_id, capture_logs=False))
            stack.enter_context(client.turn(turn_input=user_input))
        except Exception as exc:
            logger.warning(f"Sovara recording unavailable; continuing unrecorded. error_type={type(exc).__name__}")
        yield
```

## Behavior

- Sovara failures never fail a CodeMie run: it continues unrecorded. That includes a failure
  completing the recording after a successful run: inside `short_retries()` it is only logged
  (synchronous runs send the completion in the background).
- A slow Sovara delays a chat by at most a few requests' worth: inside `short_retries()` each
  request gives up after 5 s (10 s for a model step that waits for lessons, which then runs
  without lessons). In synchronous runs step finishes and completion don't wait at all, so a
  chat with one model call waits on three Sovara requests (open the trace, start the turn,
  prepare the model step) and each tool call adds one. Asynchronous runs also wait for the
  finishes and the completion, each capped the same way.
- Concurrent conversations and simultaneous turns of one conversation are recorded apart.
- Supervisor sub-assistants run as nodes of the coordinator's graph, so their model calls and
  handoffs are recorded in the coordinator's turn. Each model and tool step stores the LangGraph
  nodes it ran in (for example `["sql_assistant", "agent"]`) and LangGraph's namespace, which
  tells repeated or parallel delegations apart. They are kept in the step's recorded input
  (`_meta`), which the trace view does not show yet. Sub-assistants share the coordinator's
  lesson scope.
- CodeMie's own tests that mock the model in `get_llm_by_credentials` need `SOVARA_ENABLED`
  off (the default): `wrap_chat_model` accepts only LangChain chat models.
