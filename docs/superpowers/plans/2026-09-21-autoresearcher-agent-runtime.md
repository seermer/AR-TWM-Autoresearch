# AutoResearcher Agent Runtime Implementation Plan (Plan 2 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an agent version in a sandbox, where it talks to an LLM only through a recording gateway and to the kernel only through privileged MCP tools, produces data commits and a recipe, and is verified against the fixed `edit_self` / `improve_recipe` contract. Every step is recorded.

**Architecture:** The kernel starts two HTTP services on Unix domain sockets in a per-run socket directory: a recording OpenAI-compatible **gateway** and an MCP **tool server**. Each agent call runs in a fresh Docker container with **no network** (`--network none`), running as the host user, with the socket directory mounted. The fixed `ar_contract` package (mounted read-only) gives agent code preconfigured clients and runs the entry point. Agent code lives in a per-run git repo (`agents.git`). The seed agent's single-agent harness is an explicit LangGraph ReAct graph in the agent's own code: it reproduces `langchain.agents.create_agent` exactly, then adds reported tool errors and Claude-Code-style auto-compaction. Its multi-agent orchestration is plain async Python. Both are agent code, so self-improvement can change them.

**Tech Stack:** Python 3.12 (`autoresearcher` env); FastAPI and uvicorn (UDS); `mcp==2.2.0` (`MCPServer`, `httpx2` client transport); `langgraph==1.2.11`, `langchain-core==1.6.3` and `langchain-openai==1.6.2` (`langchain==1.4.2` in tests only, as the reference for the harness); `huggingface_hub`; `zstandard`; the Docker 27.3.1 CLI (no Python Docker library); the git CLI.

**Spec:** `docs/superpowers/specs/2026-09-17-autoresearcher-design.md`. Sections implemented: §4.1–4.2 (gateway, tools, sandbox, contract components), §5.2, §5.5 (aspect check), §9, §10 (except the GPU generator backends, see Plan 3), §13.1–13.3 (gateway and tool-call capture), §14.2 rows for `edit_self` / contract / `improve_recipe` / kernel tools / gateway, §16.1 (gateway, contract, GPU job API), §16.3 item 4. Also read `docs/superpowers/plans/verification-log.md`: its findings 1–9 bind this plan wherever it launches or kills processes.

## Global Constraints

- Agents never touch WBench, the model inference path, the kernel or `.env` (spec §1.1.5). Nothing in this plan mounts `WorldModel/`, `WBench/`, `AutoResearcher/kernel`, weights or `.env` into a container.
- `OPENAI_API_KEY` is held only by the gateway process and is never passed into a container (§2.1). Containers get a per-container token instead.
- Agent LLMs are OpenAI models only (§1.1.9). The gateway enforces `gateway.model_allowlist`.
- Telemetry is fail-closed (§13.1.2): the gateway persists the request **before** forwarding and the response **before** returning. If either write fails, the call fails.
- No truncation or sampling of telemetry (§13.1.3). Secrets are redacted: `OPENAI_API_KEY`, `HF_TOKEN`, `VLM_API_KEY` and container tokens.
- Tool errors return to the agent as tool errors, never as kernel exceptions (§10, §14.2).
- GPU tools are asynchronous jobs. `job.wait` never blocks longer than `tools.job_wait_max_s` (300 s) (§10).
- Every process the kernel starts runs in its own session and is killed by **process group**. GPU memory and scratch space are reclaimed after a kill (verification-log findings 6–7).
- Containers run with `--user <host uid>:<host gid>`, so the kernel can delete everything they write.
- Never use system or `base` Python: `conda run --no-capture-output -n autoresearcher ...` (and `alayaworld` for WorldModel tools).
- GPU policy: device lists always explicit; default `0,1,2,3`; any indices, minimum 4 (memory `gpu-5-off-limits`).
- The default unit suite uses no GPU and no network. Tests that need Docker are marked `docker`, and are selected explicitly like `gpu` and `manual`.
- Portability: no absolute paths in tracked code or config; everything resolves from `KernelConfig` (`docs/PORTABILITY.md`).

## Verified facts this plan relies on (pre-plan spikes, 2026-09-21; revised the same day for the harness)

Each of these was checked on this machine before the plan was written. Several differ from what the library documentation or older versions suggest, so do not "fix" them back.

1. **MCP 2.x** renamed `FastMCP` to `mcp.server.mcpserver.MCPServer`. The server app is `MCPServer.streamable_http_app(...)`.
2. **The MCP 2.x client uses `httpx2`, not `httpx`.** `mcp.client.streamable_http.streamable_http_client(url, http_client=httpx2.AsyncClient(transport=httpx2.AsyncHTTPTransport(uds=...)))` yields `(read, write)` for `mcp.ClientSession(read, write, read_timeout_seconds=...)`. Verified end to end over a Unix socket.
3. **Host header over UDS.** The MCP server's DNS-rebinding guard rejects `Host: localhost` (no port) with **421**. Pass `transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=["localhost"], allowed_origins=[])`, and address both services as `http://localhost/...` over the socket.
4. **Session idle timeout.** `streamable_http_app` defaults to `session_idle_timeout=1800`, which silently kills the session of an agent that makes no tool call for 30 min. Use `session_idle_timeout=None`.
5. **Client timeout.** `data_ingest` runs WorldModel's checker per clip and takes minutes; `recipe_check` can wait on the GPU lock and then run 3,600 s gate steps; `hf_download` can fetch up to 20 GiB. `ar_contract.client.mcp_session` sets `read_timeout_seconds=MCP_TIMEOUT_S` and the same `httpx2` timeout, with `MCP_TIMEOUT_S = 14400` (4 h). *(As built: the first draft said 900 s; a client timeout shorter than a server operation makes the agent retry work that is still running. The phase hard cap bounds real hangs.)*
6. **Mock Responses API.** A scripted `/v1/responses` server works with `langchain-openai`'s `ChatOpenAI(use_responses_api=True)` when each response carries `id, object="response", created_at, status="completed", model, output, parallel_tool_calls, tool_choice, tools, usage{input_tokens, output_tokens, total_tokens, input_tokens_details{cached_tokens}, output_tokens_details{reasoning_tokens}}`. A tool call is an output item `{"type":"function_call","id","call_id","name","arguments","status":"completed"}`. `ChatOpenAI` turns these into `tool_calls`, sends `function_call_output` items on the next request, reports `usage` as `usage_metadata`, and sends `stream: false`, so the gateway needs no streaming. `bind_tools(tools, tool_choice="none")` sends `tool_choice: "none"`, and `image_url` content blocks become `input_image` items. Its resent history equals the previous request plus response under the gateway's prefix normalization (Task 4), checked for tool-call and text turns, so conversation linking works; after an auto-compaction the history loses that prefix and a new conversation starts. *(Task 19: the client now uses Chat Completions (`use_responses_api=False`); the mock serves both formats from the same scripts, and the gateway's Chat Completions prefix linking compares assistant messages by content, tool-call ids/names/parsed arguments and `reasoning_content`.)*
7. **Sandbox networking.** A Docker `--internal` network blocks the internet and DNS, **but the host stays reachable at the bridge IP**: SSH on 22 and another host service were open from inside. `--network none` plus the UDS socket directory gives complete isolation. Verified: the agent loop completes, host ports are unreachable, the internet is blocked, and files are owned by the host uid.
8. **ffmpeg 6.1.1** writes rotation with `ffmpeg -display_rotation 90 -i in.mp4 -c copy out.mp4`, and ffprobe reports it as `side_data_list:[{"rotation":90}]`. A 552x414 clip with `sample_aspect_ratio` `4:3` displays at 16:9.
9. **Packages.** The agent layer is `langgraph` 1.2.11 (with `langgraph-prebuilt` 1.1.0), `langchain-core` 1.6.3, `langchain-openai` 1.6.2, `mcp` 2.2.0 and `httpx2` 2.13.0; `pip check` is clean. `langchain` 1.4.2 is installed for tests only. `openai-agents` and `langgraph-checkpoint-sqlite` were installed by the first spike and are no longer used (Task 1 removes them).
10. **Caller identity in a tool.** A tool that declares `ctx: Context` reads the caller's header as `ctx.request_context.request.headers["authorization"]`. Verified end to end over streamable HTTP.
11. **Tool names.** OpenAI function names must match `^[a-zA-Z0-9_-]+$`, and LangChain passes tool names through as function names. The spec's dotted names (`data.ingest`) would be rejected upstream, so tools are registered as `data_ingest` and so on.
12. **Socket path length.** `AF_UNIX` paths are capped at 107 bytes. A socket under `runs/<run_id>/sock/` is 125 bytes, and `bind` fails with "AF_UNIX path too long". Sockets live in a short per-run directory under the system temp dir.
13. **One event loop.** One MCP session lives inside one event loop, so the harness, the roles and the orchestration are all async and share one `mcp_session()` per entry-point call.
14. **The `create_agent` reference.** The installed `langchain` 1.4.2 `agents/factory.py` is byte-identical to commit 4af7ab8. With no middleware and no `response_format`, its graph is START → model → (END, or one `Send("tools", [call])` per tool call) → model, with `recursion_limit` 9999, and it binds the tools on every model call. The Task 16 graph matches it message for message on 17 scenarios, and each of 5 mutations breaks the comparison.
15. **Tool errors in `create_agent`.** Its default `ToolNode` handler turns only invalid arguments (`ToolInvocationError`) into a message; any other tool exception is re-raised and ends the run. An unknown tool name gets `Error: <name> is not a valid tool, try one of [...]`.
16. **MCP 2.x attributes are snake_case**: `CallToolResult.is_error`, `.structured_content`, `Tool.input_schema`. `isError` and `inputSchema` exist only as JSON aliases, not as Python attributes. The first draft of this plan used `.isError`, which raises `AttributeError`.
17. **No tokenizer offline.** `tiktoken` downloads its encodings on first use and containers have no network, so the harness estimates context size from the last reply's `usage_metadata` plus about 4 characters per token for the messages after it. Image blocks are counted as a fixed 1,500 tokens: their base64 length says nothing about vision tokens (a 300 KB frame would otherwise count as about 75,000 tokens and force a needless compaction).
18. **`ChatOpenAI` needs both HTTP clients on the socket.** With only `http_async_client` set, a sync `invoke()` builds its own default client, dials TCP `localhost:80`, and fails with `Connection error` (reproduced). `chat_model()` passes `http_client` and `http_async_client`, both over the gateway socket; then `invoke()` and `ainvoke()` both work.

## Plan sequence (revised)

| Plan | Scope | Status |
|---|---|---|
| 1 | Kernel foundations | done, merged to `main` |
| **2** | **Agent runtime (this plan)** | — |
| 3 | GPU data sources: `rollout.alayaworld`, `annotate.camera` (ViGeo), `rollout.ltx25`, `rollout.wan22`, each in its own conda env, verified with a smoke rollout before it is enabled (§16.3 items 5–6) | written after Plan 2 lands |
| 4 | The loop: selection, cycle, retries, per-attempt training dirs, resume/stop, liveness, dashboard | interface contract at the end of this plan; tasks written after Plan 3 |

The GPU generators were split out because they are independent of the runtime, not because they are infeasible. Each GPU job holds the node's whole GPU set (default 4 × 24 GB), so LTX-2.5 (22B; bf16 checkpoint 42 GB, int8 21.5 GB on disk) runs sharded across the 4 cards, the same way the 22B LTX-2.3-based AlayaWorld already runs on this machine; the NVFP4 file is the only variant that needs Blackwell and is not used. Wan 2.2 TI2V-5B is downloaded in the official repo's format, so it runs from the official Wan2.2 code in a dedicated conda env (spec: conflicting tools get their own env) rather than through `diffusers`. Plan 2 ships the job API with a registry and a fake backend; Plan 3 plugs real backends in. Until then, agents build data from Hugging Face datasets and the archive-wide clip pool.

## Spec amendments made with this plan

- **§9.5 Network:** "a dedicated Docker network reaching only the gateway and the tool server" becomes: no network (`--network none`); the gateway and tool server are reached over Unix domain sockets in a mounted socket directory. Reason: fact 7 (an internal bridge network exposes host services).
- **§10 tool names:** registered with underscores (`data_ingest`), because OpenAI function names forbid dots (fact 11).
- Not an amendment: §13.2 already specifies `.json.zst` payloads. Plan 1 stored plain `.json`; Task 1 brings the code into line and keeps old payloads readable.

- **§1.1 item 9, §9.1–9.3 (frameworks, revised the same day):** the single-agent harness is an explicit LangGraph ReAct graph in agent code, reproducing `create_agent` plus reported tool errors and auto-compaction; multi-agent orchestration is plain Python; the OpenAI Agents SDK is dropped. New §9.1.1 (edit components: each `edit_self` plans exactly one of prompts, tools, harness, orchestration or knowledge, recorded in `EditResult.component`, not enforced) and §9.1.2 (harness behaviour). §9.2 loses the SDK trace processor; §13.2, §13.3, §17 (`agents.*`) and §18 follow.

All amendments are applied to the spec in the same commits as this plan.

Amendments made during implementation (applied to the spec in Task 18):
- **§5.2 attempt refs:** `refs/attempts/<node>/<phase>-<k>` (for example `edit_self-2`), not `refs/attempts/<node>/<k>`, because both phases have attempts.
- **§9.1 layout and §9.1.1 paths:** the seed agent is `entry.py` (with the settings), `harness.py`, `orchestration.py`, `tools.py`, `prompts/`, `knowledge/data_building.md` and `memory/README.md` (simplicity ruling, File structure).
- **§9.1.2 compaction placement:** the compaction check runs inside the model node on the merged state; the graph is exactly `create_agent`'s (Task 16).
- **§9.5 mounts:** `/agent` is read-only for `improve_recipe`; only `edit_self` changes code.
- **§10 `hf.search`:** returns ids, license, tags, downloads and last-modified time, not sizes; `hf.download` checks sizes before transferring.

## File structure

```
kernel/ar_kernel/
  telemetry/recorder.py        MODIFY  run_id + component on every event; zstd payloads
  data/probe.py                MODIFY  rotation + SAR; display aspect
  gateway/__init__.py          CREATE
  gateway/store.py             CREATE  fail-closed call records + conversation linking
  gateway/app.py               CREATE  FastAPI app: auth, allowlist, forward/mock, record
  gateway/mock.py              CREATE  scripted responses for smoke runs and tests
  tools/__init__.py            CREATE
  tools/context.py             CREATE  token -> (node, phase, attempt) registry; path mapping
  tools/server.py              CREATE  MCPServer factory, telemetry + error wrapping
  tools/data_tools.py          CREATE  video.probe, data.ingest, data.query, data.commit, recipe.check
  tools/hf_tools.py            CREATE  hf.search, hf.download
  tools/jobs.py                CREATE  GPU job queue + backend registry + job.* tools
  services.py                  CREATE  start/stop gateway + tool server on UDS for a run
  sandbox/__init__.py          CREATE
  sandbox/image.py             CREATE  build the agent image, cached by requirements hash
  sandbox/runner.py            CREATE  one container per call: mounts, limits, kill, stats, diff
  vcs/__init__.py              CREATE
  vcs/agents_repo.py           CREATE  agents.git: seed, node branches, attempt refs, diffs
  context_bundle.py            CREATE  EditContext / RecipeContext bundles from the archive
  contract/__init__.py         CREATE
  contract/verify.py           CREATE  static, import, smoke run (mock gateway + tools)
  agent_phase.py               CREATE  run_edit_self / run_improve_recipe (Plan 4's entry points)
contract/ar_contract/          CREATE  kernel-owned, mounted read-only into containers
  __init__.py  models.py  client.py  run.py
seed_agent/agent/              CREATE  initial agent code (the root node's commit)
  entry.py            edit_self / improve_recipe + the settings constants (Task 17)
  harness.py          single-agent inner loop: ReAct graph + auto-compaction (Task 16)
  orchestration.py    roles + improve_recipe workflow + edit_self workflow, plain Python (Task 17)
  tools.py            file/bash, media, MCP adapter, submit_* tools (Task 17)
  requirements.txt  prompts/*.md  knowledge/data_building.md  memory/README.md
docker/agent.Dockerfile        CREATE
configs/kernel.yaml            MODIFY  gateway, agents, tools, sandbox, timeouts, generators blocks
tests/                         CREATE  one test file per component (named per task)
```

*As built (2026-09-23, user requirement):* agent code is what self-improvement reads and rewrites every cycle, so it prefers simplicity over everything: fewer, shorter files and straightforward code. The seed agent was therefore built as the four modules above instead of the first draft's `settings.py`, `harness/{react,compact}.py`, `orchestration/{roles,task,meta}.py` and `tools/{files,media,kernel,submit}.py` packages, with the same behaviour and tests. `knowledge/README.md` was dropped (the edit components table and `memory/README.md` already say what `knowledge/` is for).

---
### Task 1: Dependencies, config blocks, telemetry v2

Every later task imports these packages, reads these config keys, and emits events through this recorder. That is why they are one task.

**Files:**
- Modify: `pyproject.toml`
- Modify: `configs/kernel.yaml`
- Modify: `kernel/ar_kernel/telemetry/recorder.py`
- Modify: `tests/test_telemetry.py`

**Interfaces:**
- Consumes: the Plan 1 `Recorder(run_dir, redact=())`.
- Produces:
  - `Recorder(run_dir, redact=(), run_id: str | None = None)`. `run_id` defaults to `Path(run_dir).name`.
  - `Recorder.event(type, *, node="run", phase="-", attempt=0, payload=None, span_id=None, parent_span_id=None, component="kernel", **fields) -> str`. Every record now carries `run_id` and `component` (spec §13.1.4).
  - `Recorder.span(...)` also accepts `component=`.
  - `Recorder.add_redaction(secret: str) -> None`. Used for container tokens issued after the recorder exists.
  - `Recorder.store_payload(obj) -> str` writes `payloads/<sha256>.json.zst`. The digest is of the *uncompressed* JSON bytes.
  - `Recorder.load_payload(digest) -> dict` reads `.json.zst`, falling back to legacy `.json`.
  - Config keys: `gateway.*`, `agents.*`, `sandbox.*`, `timeouts.*`, `generators.*`, `tools.hf_download_max_bytes`.

- [ ] **Step 1: Pin the dependencies and make `ar_contract` importable**

In `pyproject.toml`, replace the `dependencies` line and the package-find block:

```toml
dependencies = [
    "pyyaml>=6.0", "numpy>=1.26", "pillow>=10.0", "imagehash>=4.3",
    "fastapi>=0.141", "uvicorn>=0.53", "httpx>=0.28", "pydantic>=2.12",
    "mcp==2.2.0", "openai>=3.0,<4",
    "langgraph==1.2.11", "langchain-core==1.6.3", "langchain-openai==1.6.2",
    "huggingface_hub>=1.32", "zstandard>=0.25",
]
```

```toml
[tool.setuptools.packages.find]
where = ["kernel", "contract"]
```

and the test extra (`langchain` is the reference `create_agent` for `tests/test_seed_harness.py`; it is not in the agent image):

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "langchain==1.4.2"]
```

Register the `docker` marker and deselect it by default:

```toml
markers = [
    "manual: GPU verification, run explicitly",
    "gpu: launches a real GPU job (the gate's describe step); run with -m gpu",
    "docker: starts real containers; run with -m docker",
]
addopts = "-q -m 'not manual and not gpu and not docker'"
```

Run: `conda run --no-capture-output -n autoresearcher python -m pip uninstall -y openai-agents langgraph-checkpoint-sqlite && conda run --no-capture-output -n autoresearcher python -m pip install -e '.[dev]' && conda run --no-capture-output -n autoresearcher python -m pip check`
Expected: `No broken requirements found.`

- [ ] **Step 2: Add the config blocks**

Append to `configs/kernel.yaml`. Keep the existing `tools:` block's `job_wait_max_s` and add the new key under it:

```yaml
gateway:
  model_allowlist: []            # OPENAI_MODEL from the environment is always allowed too
  upstream_timeout_s: 600
  upstream_retries: 5            # 429/5xx, exponential backoff 1, 2, 4, ... s
  upstream_outage_pause_min: 15
agents:                          # passed to agent containers as AR_CONTEXT_WINDOW / AR_COMPACT_AT
  context_window_tokens: 128000  # set to the agent model's context window
  compact_at: 0.85               # the harness auto-compacts at this fraction of the window
sandbox:
  image: ar-agent
  cpus: 16
  memory_gb: 64
timeouts:                        # soft timeouts (spec 14.5); Plan 2 enforces the hard cap = 4x soft
  edit_self_s: 7200
  improve_recipe_s: 86400
  contract_import_s: 60
  contract_smoke_s: 900
generators:                      # every variant disabled until Plan 3 verifies it (spec 16.3 item 5)
  alayaworld: {dmd4: {enabled: false}, ar30: {enabled: false}}
  ltx25: {dev: {enabled: false}, distilled: {enabled: false}}
  wan22: {ti2v-5b: {enabled: false}}
```

and under the existing `tools:` key:

```yaml
tools:
  job_wait_max_s: 300
  hf_download_max_bytes: 21474836480   # 20 GiB per hf.download call
```

- [ ] **Step 3: Write the failing telemetry tests**

The two redaction tests added in the Plan 1 review read `payloads/<digest>.json` directly. Payloads move to `.json.zst`, so change them to read through the recorder. In `tests/test_telemetry.py`, replace each

```python
    assert "sk-SECRET123" not in (tmp_path / "telemetry" / "payloads" / f"{digest}.json").read_text()
```

with

```python
    assert "sk-SECRET123" not in json.dumps(rec.load_payload(digest))
    assert b"sk-SECRET123" not in _raw_payload(tmp_path, digest)
```

and append:

```python
import json
import zstandard


def _raw_payload(run_dir, digest) -> bytes:
    """Decompressed on-disk bytes, so a test proves the secret never reached the disk."""
    path = run_dir / "telemetry" / "payloads" / f"{digest}.json.zst"
    return zstandard.ZstdDecompressor().decompress(path.read_bytes())


def test_every_event_carries_run_id_and_component(tmp_path):
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path / "run42")
    rec.event("x.happened", node="n1", component="gateway")
    event = rec.read_events("n1")[0]
    assert event["run_id"] == "run42"
    assert event["component"] == "gateway"


def test_component_defaults_to_kernel(tmp_path):
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    rec.event("y")
    assert rec.read_events()[0]["component"] == "kernel"


def test_payloads_are_zstd_and_round_trip(tmp_path):
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    big = {"messages": ["the same long prompt " * 200] * 20}
    digest = rec.store_payload(big)
    path = tmp_path / "telemetry" / "payloads" / f"{digest}.json.zst"
    assert path.exists()
    assert path.stat().st_size < len(json.dumps(big)) / 5
    assert rec.load_payload(digest) == big


def test_legacy_uncompressed_payloads_stay_readable(tmp_path):
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    legacy = tmp_path / "telemetry" / "payloads" / "abc123.json"
    legacy.write_text('{"old": true}')
    assert rec.load_payload("abc123") == {"old": True}


def test_redaction_added_after_construction_applies(tmp_path):
    """Container tokens are issued after the recorder exists."""
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    rec.add_redaction("tok-LATE-ISSUED")
    digest = rec.store_payload({"auth": "Bearer tok-LATE-ISSUED"})
    assert b"tok-LATE-ISSUED" not in _raw_payload(tmp_path, digest)
```

- [ ] **Step 4: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_telemetry.py -p no:cacheprovider`
Expected: FAIL. `run_id` is missing from the event, no `.json.zst` file exists, and `add_redaction` is not defined.

- [ ] **Step 5: Implement**

In `kernel/ar_kernel/telemetry/recorder.py`:

```python
import zstandard
```

Replace `__init__`:

```python
    def __init__(self, run_dir: Path, redact: Iterable[str] = (), run_id: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.redact = [s for s in redact if s]
        self._events = self.run_dir / "telemetry" / "events"
        self._payloads = self.run_dir / "telemetry" / "payloads"
        for directory in (self._events, self._payloads):
            directory.mkdir(parents=True, exist_ok=True)

    def add_redaction(self, secret: str) -> None:
        if secret and secret not in self.redact:
            self.redact.append(secret)
```

Replace `store_payload` and `load_payload`:

```python
    def store_payload(self, obj: dict) -> str:
        blob = self._scrub_text(json.dumps(self._scrub(obj), sort_keys=True, default=str)).encode()
        digest = hashlib.sha256(blob).hexdigest()
        target = self._payloads / f"{digest}.json.zst"
        if not target.exists():
            try:
                tmp = target.with_name(target.name + ".tmp")
                with tmp.open("wb") as handle:
                    handle.write(zstandard.ZstdCompressor(level=10).compress(blob))
                    handle.flush()
                    # The event line that references this payload is fsync'd; the
                    # payload must be durable first or a crash leaves a dangling digest.
                    os.fsync(handle.fileno())
                os.replace(tmp, target)
            except OSError as exc:
                raise TelemetryError(f"cannot write payload {digest}: {exc}") from exc
        return digest

    def load_payload(self, digest: str) -> dict:
        compressed = self._payloads / f"{digest}.json.zst"
        if compressed.exists():
            return json.loads(zstandard.ZstdDecompressor().decompress(compressed.read_bytes()))
        return json.loads((self._payloads / f"{digest}.json").read_text())   # pre-zstd runs
```

In `event`, add the parameter `component: str = "kernel"` before `**fields`, and add two keys to `record` directly after `"ts_mono"`:

```python
            "run_id": self.run_id,
            "component": component,
```

In `span`, add `component: str = "kernel"` to the signature and pass `component=component` to both `self.event(...)` calls inside it.

- [ ] **Step 6: Run the whole suite**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -p no:cacheprovider`
Expected: PASS. Every pre-existing test is still green, and `grep -rn '"payloads"' tests/` shows no other test reading payload files directly.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml configs/kernel.yaml kernel/ar_kernel/telemetry/recorder.py tests/test_telemetry.py
git commit -m "feat(telemetry): run_id and component on every event; zstd payloads; late redaction"
```

---

### Task 2: Display aspect ratio and rotation at ingest

A deferred Plan 1 review item that must land before agents fetch arbitrary web video. The aspect check compares **coded** width/height, which is wrong in both directions:
- A `rotate=90` phone clip (coded 736x414, shown portrait) passes as 16:9. WorldModel decodes frames in stored orientation, so it would train on sideways frames.
- A 552x414 clip with 4:3 pixels (`sample_aspect_ratio` `4:3`) displays at exactly 16:9, and is wrongly rejected. Resizing it to `sample.width x sample.height` without cropping applies that stretch, so it is valid training data.

**Files:**
- Modify: `kernel/ar_kernel/data/probe.py`
- Modify: `kernel/ar_kernel/data/ingest.py` (in `_check_and_store`, right after `probe_video`)
- Test: `tests/test_probe.py`, `tests/test_ingest.py`

**Interfaces:**
- Consumes: the Plan 1 `probe_video(path) -> VideoInfo`, `aspect_ok(info, tolerance) -> bool`.
- Produces: `VideoInfo` gains `rotation: int = 0` (degrees, normalized to 0/90/180/270) and `sar: float = 1.0`, plus the property `display_aspect -> float`. `aspect_ok` checks `display_aspect`. Ingest rejects any clip with `rotation != 0`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_probe.py`:

```python
import subprocess

from ar_kernel.data.probe import aspect_ok, probe_video
from conftest import make_mp4


def _rotated(src, dst, degrees=90):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", str(degrees),
                    "-i", str(src), "-c", "copy", str(dst)], check=True)
    return dst


def _with_sar(dst, width, height, sar):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=size={width}x{height}:rate=24", "-t", "3",
                    "-vf", f"setsar={sar}", "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


def test_rotation_is_reported(tmp_path):
    info = probe_video(_rotated(make_mp4(tmp_path / "a.mp4"), tmp_path / "r.mp4"))
    assert info.rotation == 90


def test_rotated_landscape_is_not_sixteen_by_nine(tmp_path):
    """Coded 736x414 but displayed portrait: must fail the aspect check."""
    info = probe_video(_rotated(make_mp4(tmp_path / "a.mp4"), tmp_path / "r.mp4"))
    assert not aspect_ok(info, 0.02)


def test_anamorphic_clip_that_displays_sixteen_by_nine_passes(tmp_path):
    info = probe_video(_with_sar(tmp_path / "s.mp4", 552, 414, "4/3"))
    assert abs(info.sar - 4 / 3) < 1e-6
    assert aspect_ok(info, 0.02)


def test_square_pixel_four_by_three_still_fails(tmp_path):
    assert not aspect_ok(probe_video(_with_sar(tmp_path / "f.mp4", 552, 414, "1")), 0.02)


def test_unrotated_clip_reports_zero_rotation(tmp_path):
    info = probe_video(make_mp4(tmp_path / "plain.mp4"))
    assert info.rotation == 0 and info.sar == 1.0
```

Append to `tests/test_ingest.py`:

```python
def test_rotated_clip_is_rejected_with_a_fix_hint(tmp_path):
    """WorldModel decodes frames in stored orientation; a rotated clip would train sideways."""
    import subprocess
    ing = _ingestor(tmp_path)
    cand = _candidate(tmp_path, "rot")
    rotated = cand.video.with_name("v_rot.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", "180",
                    "-i", str(cand.video), "-c", "copy", str(rotated)], check=True)
    rotated.replace(cand.video)
    result = ing.ingest([cand], node_id="n1")[0]
    assert not result.accepted
    assert any("rotation" in r and "re-encode" in r for r in result.reasons)
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_probe.py tests/test_ingest.py -k "rotat or anamorphic or square" -p no:cacheprovider`
Expected: FAIL (`VideoInfo` has no `rotation`; the rotated clip is accepted).

- [ ] **Step 3: Implement the probe changes**

Replace `kernel/ar_kernel/data/probe.py` below the imports with:

```python
TARGET_ASPECT = 16 / 9

@dataclass(frozen=True)
class VideoInfo:
    frames: int
    fps: float
    width: int                 # coded (stored) pixels
    height: int
    duration: float
    rotation: int = 0          # display rotation in degrees: 0, 90, 180 or 270
    sar: float = 1.0           # sample (pixel) aspect ratio

    @property
    def display_aspect(self) -> float:
        """Aspect as shown: coded width x pixel aspect, swapped for 90/270 rotation."""
        width, height = self.width * self.sar, float(self.height)
        if self.rotation in (90, 270):
            width, height = height, width
        return width / height


def _ratio(text: str | None) -> float:
    if not text or ":" not in text:
        return 1.0
    num, den = (float(x) for x in text.split(":"))
    return num / den if num > 0 and den > 0 else 1.0     # "0:1" / "N/A" mean square pixels


def _rotation(stream: dict) -> int:
    raw = 0.0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            raw = float(side["rotation"])
    if not raw:
        raw = float((stream.get("tags") or {}).get("rotate") or 0)   # pre-6.0 muxers
    return int(round(raw)) % 360


def probe_video(path: Path) -> VideoInfo:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries",
         "stream=nb_read_frames,avg_frame_rate,width,height,duration,sample_aspect_ratio"
         ":stream_tags=rotate:stream_side_data=rotation",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    num, den = (int(x) for x in stream["avg_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    frames = int(stream["nb_read_frames"])
    duration = float(stream.get("duration") or (frames / fps if fps else 0.0))
    return VideoInfo(frames=frames, fps=fps, width=int(stream["width"]),
                     height=int(stream["height"]), duration=duration,
                     rotation=_rotation(stream), sar=_ratio(stream.get("sample_aspect_ratio")))


def aspect_ok(info: VideoInfo, tolerance: float) -> bool:
    return abs(info.display_aspect - TARGET_ASPECT) <= TARGET_ASPECT * tolerance
```

- [ ] **Step 4: Reject rotated clips at ingest**

In `kernel/ar_kernel/data/ingest.py`, in `_check_and_store`, directly after `info = probe_video(video)` and before the aspect check:

```python
        if info.rotation:
            reasons = [f"video carries a {info.rotation} degree display rotation; WorldModel decodes "
                       f"frames in stored orientation, so they would train rotated. re-encode with "
                       f"the rotation applied (ffmpeg applies it when transcoding: "
                       f"ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4)"]
            self.recorder.event("ingest.rejected", node=node_id, phase="ingest",
                                payload={"reasons": reasons})
            return IngestResult(accepted=False, reasons=reasons)
```

Also change the aspect rejection message to report the display aspect:

```python
            reasons = [f"display aspect {info.display_aspect:.4f} (coded {info.width}x{info.height}, "
                       f"sar {info.sar:.4f}) is not within {self.tolerance:.0%} of 16:9"]
```

- [ ] **Step 5: Run the probe and ingest tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_probe.py tests/test_ingest.py -p no:cacheprovider`
Expected: PASS, including every pre-existing ingest test. (`test_four_by_three_clip_is_rejected_before_the_checker` still fails a square-pixel 4:3 clip.)

- [ ] **Step 6: Commit**

```bash
git add kernel/ar_kernel/data/probe.py kernel/ar_kernel/data/ingest.py tests/test_probe.py tests/test_ingest.py
git commit -m "fix(ingest): check display aspect and reject rotated video before agents fetch web data"
```

---

### Task 3: The `ar_contract` package

Kernel-owned and mounted read-only at `/ar_contract` in every container (spec §9.2). It holds the schemas the kernel validates results against, clients preconfigured for the sockets (facts 2–6), and the runner. This is the only way agent code gets a correctly configured LLM client and MCP connection, so the socket details live here once.

**Files:**
- Create: `contract/ar_contract/__init__.py`, `models.py`, `client.py`, `run.py`
- Test: `tests/test_contract_package.py`

**Interfaces:**
- Produces (imported by agent code, the kernel's context builder and contract verification):
  - `ar_contract.models`: `EditContext`, `EditResult` (`summary`, optional `component`), `RecipeContext`, `RecipeResult` (pydantic v2), `EditComponent = Literal["prompts", "tools", "harness", "orchestration", "knowledge"]` and `EDIT_COMPONENTS = typing.get_args(EditComponent)` (defined once; Task 17 imports `EditComponent`), and `CONTEXT_MODELS = {"edit_self": EditContext, "improve_recipe": RecipeContext}`, `RESULT_MODELS = {"edit_self": EditResult, "improve_recipe": RecipeResult}`.
  - `ar_contract.client`: `socket_dir() -> str` (env `AR_SOCKET_DIR`, default `/run/ar`), `token() -> str` (env `AR_TOKEN`), `default_model() -> str` (env `AR_DEFAULT_MODEL`), `chat_model(model=None, **kwargs) -> langchain_openai.ChatOpenAI` (gateway socket for both the sync and the async client, fact 18; Chat Completions since Task 19, `max_retries=0`), and `mcp_session(sockets=None, auth_token=None)`, an async context manager yielding an initialized `mcp.ClientSession` on the tool server. Both read the environment when called.
  - `ar_contract.run.main(argv) -> int`, run as `python -m ar_contract.run <edit_self|improve_recipe>`. It reads `$AR_CONTEXT_DIR/context.json` (default `/context`), imports `agent.entry` from `$AR_AGENT_DIR` (default `/agent`), calls the entry point (sync or async), validates the result, and writes `$AR_WORKSPACE/result.json` (default `/workspace`) as `{"ok": true, "result": {...}}` or `{"ok": false, "error": "...", "traceback": "..."}`. Exit code 0 means ok.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_contract_package.py
"""ar_contract runs inside the container; these tests run it in-process with
temp directories standing in for /agent, /context and /workspace."""
import json
import textwrap

import pytest

from ar_contract import models
from ar_contract.run import main


def _edit_ctx(**over):
    base = {"nodes_remaining": 5, "attempt": 1, "max_attempts": 3}
    return {**base, **over}


def _layout(tmp_path, entry_src, ctx):
    agent = tmp_path / "agent_repo"
    (agent / "agent").mkdir(parents=True)
    (agent / "agent" / "__init__.py").write_text("")
    (agent / "agent" / "entry.py").write_text(textwrap.dedent(entry_src))
    (tmp_path / "context").mkdir()
    (tmp_path / "context" / "context.json").write_text(json.dumps(ctx))
    (tmp_path / "workspace").mkdir()
    return agent


@pytest.fixture
def env(tmp_path, monkeypatch):
    def setup(entry_src, ctx):
        agent = _layout(tmp_path, entry_src, ctx)
        monkeypatch.setenv("AR_AGENT_DIR", str(agent))
        monkeypatch.setenv("AR_CONTEXT_DIR", str(tmp_path / "context"))
        monkeypatch.setenv("AR_WORKSPACE", str(tmp_path / "workspace"))
        monkeypatch.delitem(__import__("sys").modules, "agent.entry", raising=False)
        monkeypatch.delitem(__import__("sys").modules, "agent", raising=False)
        return tmp_path / "workspace" / "result.json"
    return setup


GOOD = """
from ar_contract.models import EditResult, RecipeResult
def edit_self(ctx):
    return EditResult(summary=f"attempt {ctx.attempt}")
def improve_recipe(ctx):
    return RecipeResult(data_commit="c1", recipe={"optimizer.lr": 1e-5}, rationale="why")
"""


def test_good_edit_self_writes_ok_result(env):
    out = env(GOOD, _edit_ctx())
    assert main(["edit_self"]) == 0
    assert json.loads(out.read_text()) == {"ok": True, "result": {"summary": "attempt 1", "component": None}}


def test_async_entry_points_are_supported(env):
    out = env("""
from ar_contract.models import EditResult
async def edit_self(ctx):
    return EditResult(summary="async ok")
def improve_recipe(ctx):
    raise NotImplementedError
""", _edit_ctx())
    assert main(["edit_self"]) == 0
    assert json.loads(out.read_text())["result"]["summary"] == "async ok"


def test_raising_entry_point_writes_error_and_nonzero_exit(env):
    out = env("""
def edit_self(ctx):
    raise RuntimeError("boom")
def improve_recipe(ctx):
    pass
""", _edit_ctx())
    assert main(["edit_self"]) == 1
    body = json.loads(out.read_text())
    assert body["ok"] is False and "boom" in body["error"] and "Traceback" in body["traceback"]


def test_invalid_result_is_a_schema_failure(env):
    out = env("""
def edit_self(ctx):
    return {"summary": ""}          # empty summary violates min_length=1
def improve_recipe(ctx):
    pass
""", _edit_ctx())
    assert main(["edit_self"]) == 1
    assert "summary" in json.loads(out.read_text())["error"]


def test_dict_results_are_validated_and_accepted(env):
    out = env("""
def edit_self(ctx):
    return {"summary": "plain dict is fine"}
def improve_recipe(ctx):
    pass
""", _edit_ctx())
    assert main(["edit_self"]) == 0
    assert json.loads(out.read_text())["result"]["summary"] == "plain dict is fine"


def test_unknown_kind_is_rejected(env):
    env(GOOD, _edit_ctx())
    assert main(["drop_tables"]) == 2


def test_recipe_result_requires_commit_recipe_and_rationale():
    with pytest.raises(Exception):
        models.RecipeResult(data_commit="", recipe={}, rationale="x")
    ok = models.RecipeResult(data_commit="c", recipe={"optimizer.lr": 1e-5}, rationale="r")
    assert ok.recipe["optimizer.lr"] == 1e-5


def test_edit_result_component_is_optional_and_checked():
    assert models.EDIT_COMPONENTS == ("prompts", "tools", "harness", "orchestration", "knowledge")
    assert models.EditResult(summary="s").component is None
    assert models.EditResult(summary="s", component="harness").component == "harness"
    with pytest.raises(Exception):
        models.EditResult(summary="s", component="everything")


def test_clients_speak_over_the_socket_directory(monkeypatch, tmp_path):
    """Facts 2-6: the chat model talks to the gateway socket (Responses API, no client-side
    retries); the MCP session is an async context manager. Built without connecting; Task 6
    and Task 17 exercise both over real sockets."""
    monkeypatch.setenv("AR_SOCKET_DIR", str(tmp_path))
    monkeypatch.setenv("AR_TOKEN", "tok-abc")
    monkeypatch.setenv("AR_DEFAULT_MODEL", "gpt-x")
    import inspect
    import httpx
    from ar_contract import client
    model = client.chat_model()
    assert model.model_name == "gpt-x" and model.openai_api_base == "http://localhost/v1"
    assert model.openai_api_key.get_secret_value() == "tok-abc"
    assert model.use_responses_api is True and model.max_retries == 0
    assert isinstance(model.http_client, httpx.Client)             # sync invoke() also uses the socket
    assert isinstance(model.http_async_client, httpx.AsyncClient)
    assert inspect.isasyncgenfunction(client.mcp_session.__wrapped__)
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_contract_package.py -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_contract'`.

- [ ] **Step 3: Implement the models**

```python
# contract/ar_contract/__init__.py
"""Fixed contract between the kernel and agent code. Kernel-owned; mounted read-only."""
```

```python
# contract/ar_contract/models.py
"""Schemas the kernel validates every agent call against (spec 9.3)."""
from __future__ import annotations

import typing
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# The five parts of an agent version an edit_self plan picks from (spec 9.1.1).
EditComponent = Literal["prompts", "tools", "harness", "orchestration", "knowledge"]
EDIT_COMPONENTS = typing.get_args(EditComponent)


class _Ctx(BaseModel):
    model_config = ConfigDict(extra="allow")     # the kernel may add fields; agents ignore unknowns
    lineage: list[dict[str, Any]] = Field(default_factory=list)
    archive: dict[str, Any] = Field(default_factory=dict)
    nodes_remaining: int
    attempt: int
    max_attempts: int
    retry: dict[str, Any] | None = None          # the failed attempt's report, when retrying
    dry_run: bool = False


class EditContext(_Ctx):
    agent_dir: str = "/agent"


class RecipeContext(_Ctx):
    workspace: str = "/workspace"
    clip_pool: list[dict[str, Any]] = Field(default_factory=list)
    parent_data_commit: str | None = None
    parent_recipe: dict[str, Any] = Field(default_factory=dict)
    base_recipe: dict[str, Any] = Field(default_factory=dict)
    tunable_rules: dict[str, Any] = Field(default_factory=dict)
    resolution_allowlist: list[list[int]] = Field(default_factory=list)
    lora_allowlist: list[list[int]] = Field(default_factory=list)
    format_rules: str = ""
    n_gpus: int = 4
    tools: list[str] = Field(default_factory=list)   # enabled kernel tool names


class EditResult(BaseModel):
    summary: str = Field(min_length=1)
    # The component the edit plan chose. Recorded with the node, never enforced.
    component: EditComponent | None = None


class RecipeResult(BaseModel):
    data_commit: str = Field(min_length=1)
    recipe: dict[str, Any]
    rationale: str = Field(min_length=1)


CONTEXT_MODELS = {"edit_self": EditContext, "improve_recipe": RecipeContext}
RESULT_MODELS = {"edit_self": EditResult, "improve_recipe": RecipeResult}
```

- [ ] **Step 4: Implement the clients**

```python
# contract/ar_contract/client.py
"""Preconfigured clients for the two kernel services, reached over Unix sockets.

The container has no network. The gateway and tool server listen on sockets in
SOCKET_DIR. Details verified on this stack (see the Plan 2 facts):
- the chat model uses httpx over the gateway socket; the MCP 2.x client REQUIRES httpx2;
- the host is 'localhost' (the MCP server's rebinding guard rejects anything else);
- MCP calls such as data_ingest run the dataset checker and take minutes, so the
  session read timeout is long;
- the gateway owns upstream retries, so clients never retry.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import httpx2

MCP_TIMEOUT_S = 14400.0      # 4 h (fact 5): recipe_check and hf_download can outlast 900 s
LLM_TIMEOUT_S = 900.0


def socket_dir() -> str:
    return os.environ.get("AR_SOCKET_DIR", "/run/ar")


def token() -> str:
    return os.environ.get("AR_TOKEN", "")


def default_model() -> str:
    return os.environ.get("AR_DEFAULT_MODEL", "mock-model")


def chat_model(model: str | None = None, **kwargs):
    """A LangChain ChatOpenAI bound to the gateway (Responses API, non-streaming).

    Both clients go over the socket: without an explicit sync client, a sync invoke()
    would try TCP localhost:80 and fail inside the network-less container."""
    from langchain_openai import ChatOpenAI
    sock = os.path.join(socket_dir(), "gateway.sock")
    return ChatOpenAI(model=model or default_model(), base_url="http://localhost/v1", api_key=token(),
                      use_responses_api=True, max_retries=0,
                      http_client=httpx.Client(transport=httpx.HTTPTransport(uds=sock),
                                               timeout=LLM_TIMEOUT_S),
                      http_async_client=httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=sock),
                                                          timeout=LLM_TIMEOUT_S),
                      **kwargs)


@asynccontextmanager
async def mcp_session(sockets: str | Path | None = None, auth_token: str | None = None):
    """An initialized MCP ClientSession on the kernel tool server."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    sock = os.path.join(str(sockets or socket_dir()), "tools.sock")
    client = httpx2.AsyncClient(transport=httpx2.AsyncHTTPTransport(uds=sock),
                                headers={"Authorization": f"Bearer {auth_token or token()}"},
                                timeout=MCP_TIMEOUT_S)
    async with client, streamable_http_client("http://localhost/mcp", http_client=client) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=MCP_TIMEOUT_S) as session:
            await session.initialize()
            yield session
```

- [ ] **Step 5: Implement the runner**

```python
# contract/ar_contract/run.py
"""python -m ar_contract.run <edit_self|improve_recipe>  -- the only entry the kernel runs."""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
import traceback

from .models import CONTEXT_MODELS, RESULT_MODELS


def _write(workspace: str, body: dict) -> None:
    path = os.path.join(workspace, "result.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(body, handle, default=str)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    kind = argv[0] if argv else ""
    workspace = os.environ.get("AR_WORKSPACE", "/workspace")
    if kind not in CONTEXT_MODELS:
        print(f"usage: python -m ar_contract.run <{'|'.join(CONTEXT_MODELS)}>", file=sys.stderr)
        return 2
    try:
        context_dir = os.environ.get("AR_CONTEXT_DIR", "/context")
        with open(os.path.join(context_dir, "context.json"), encoding="utf-8") as handle:
            ctx = CONTEXT_MODELS[kind].model_validate(json.load(handle))
        sys.path.insert(0, os.environ.get("AR_AGENT_DIR", "/agent"))
        entry = importlib.import_module("agent.entry")
        value = getattr(entry, kind)(ctx)
        if inspect.isawaitable(value):
            async def _settle(awaitable):
                return await awaitable
            value = asyncio.run(_settle(value))      # coroutines and other awaitables alike
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        result = RESULT_MODELS[kind].model_validate(value)
    except Exception as exc:                  # noqa: BLE001 -- every failure becomes a result
        _write(workspace, {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()})
        return 1
    _write(workspace, {"ok": True, "result": result.model_dump()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_contract_package.py -p no:cacheprovider`
Expected: PASS (9 tests).

- [ ] **Step 7: Commit**

```bash
git add contract/ tests/test_contract_package.py
git commit -m "feat(contract): ar_contract models, UDS clients and the entry-point runner"
```

---
### Task 4: Caller tokens, path mapping and the gateway call store

Two pieces of pure logic that the gateway and the tool server share. Both carry
security weight and are easy to get subtly wrong, so they get their own task
and exhaustive tests before any HTTP is involved.

**Files:**
- Create: `kernel/ar_kernel/tools/__init__.py` (empty), `kernel/ar_kernel/tools/context.py`
- Create: `kernel/ar_kernel/gateway/__init__.py` (empty), `kernel/ar_kernel/gateway/store.py`
- Test: `tests/test_tool_context.py`, `tests/test_gateway_store.py`

**Interfaces:**
- Consumes: `Recorder.event(..., component=)`, `Recorder.add_redaction` (Task 1).
- Produces:
  - `tools.context.Caller` (frozen dataclass): `token, node, phase, attempt, workspace_host: Path, staging_host: Path, mock_script: str | None`.
  - `tools.context.TokenRegistry(recorder)`: `.issue(*, node, phase, attempt, workspace_host, staging_host, mock_script=None) -> Caller`, `.lookup(token) -> Caller | None`, `.revoke(token) -> None`.
  - `tools.context.bearer(header: str | None) -> str | None`.
  - `tools.context.to_host(caller, container_path: str) -> Path`, `tools.context.to_container(caller, host_path: Path) -> str`, and `tools.context.PathError(ValueError)`.
  - `gateway.store.CallStore(recorder)`: `.begin(caller, endpoint, body) -> dict` returns `{call_id, conversation_id, turn_index, parent_call_id}`; `.end(meta, caller, *, status, body, latency_s, attempts) -> None`. Both raise `TelemetryError` if persistence fails (fail-closed).
  - *As built:* `TokenRegistry.on_revoke(fn)` registers a listener called with the token on `.revoke`, and `CallStore.forget(token)` drops that token's conversation-linking state; the store also drops a call's stored `request` once its normalized `prefix` is kept. Task 5's `create_gateway_app` wires `registry.on_revoke(store.forget)`, so linking memory is bounded by the live containers.

Container mount layout (used by every later task):
`/workspace` ← `caller.workspace_host`; `/workspace/staging` ← `caller.staging_host`,
a nested bind mount. `staging_host` is always under `<run_dir>/staging/`, the
only place the Plan 1 ingest guard accepts files from.

- [ ] **Step 1: Write the failing tests for tokens and paths**

```python
# tests/test_tool_context.py
import pytest

from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import PathError, TokenRegistry, bearer, to_container, to_host


@pytest.fixture
def caller(tmp_path):
    reg = TokenRegistry(Recorder(tmp_path))
    ws = tmp_path / "nodes" / "n1" / "attempts" / "recipe-1" / "workspace"
    st = tmp_path / "staging" / "n1" / "recipe-1"
    ws.mkdir(parents=True), st.mkdir(parents=True)
    return reg, reg.issue(node="n1", phase="improve_recipe", attempt=1,
                          workspace_host=ws, staging_host=st)


def test_issued_token_is_unguessable_and_resolves(caller):
    reg, c = caller
    assert len(c.token) >= 40 and c.token.startswith("ar-")
    assert reg.lookup(c.token) == c
    assert reg.lookup("ar-guess") is None


def test_revoked_token_no_longer_resolves(caller):
    reg, c = caller
    reg.revoke(c.token)
    assert reg.lookup(c.token) is None


def test_issued_tokens_are_redacted_from_telemetry(tmp_path):
    rec = Recorder(tmp_path)
    c = TokenRegistry(rec).issue(node="n", phase="p", attempt=1,
                                 workspace_host=tmp_path, staging_host=tmp_path)
    assert c.token in rec.redact


def test_bearer_parsing():
    assert bearer("Bearer abc") == "abc"
    assert bearer("bearer abc") == "abc"
    assert bearer(None) is None and bearer("Basic x") is None


def test_staging_paths_map_to_the_staging_mount(caller):
    _, c = caller
    assert to_host(c, "/workspace/staging/hf/clip.mp4") == c.staging_host / "hf" / "clip.mp4"
    assert to_host(c, "/workspace/notes.txt") == c.workspace_host / "notes.txt"


@pytest.mark.parametrize("bad", [
    "/etc/passwd", "/store/blobs/video/x.mp4", "/workspace/../../etc/passwd",
    "/workspace/staging/../../../../kernel", "relative/path.mp4", "",
])
def test_paths_outside_the_workspace_are_refused(caller, bad):
    _, c = caller
    with pytest.raises(PathError):
        to_host(c, bad)


def test_symlink_escaping_the_workspace_is_refused(caller, tmp_path):
    """The agent controls /workspace; a link inside it must not reach the host."""
    _, c = caller
    (c.staging_host / "evil").symlink_to(tmp_path.parent)
    with pytest.raises(PathError):
        to_host(c, "/workspace/staging/evil/anything")


def test_round_trip_to_container(caller):
    _, c = caller
    host = c.staging_host / "a" / "b.mp4"
    assert to_container(c, host) == "/workspace/staging/a/b.mp4"
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_tool_context.py -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.tools'`.

- [ ] **Step 3: Implement `tools/context.py`**

```python
# kernel/ar_kernel/tools/context.py
"""Who is calling, and which host paths their container paths may name.

Every container gets its own token. The gateway and the tool server resolve it
to a Caller, which attributes telemetry (node, phase, attempt) and bounds every
path argument to that container's workspace. Anything else is refused.
"""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

WORKSPACE = PurePosixPath("/workspace")
STAGING = WORKSPACE / "staging"


class PathError(ValueError):
    """A path argument does not name something inside the caller's workspace."""


@dataclass(frozen=True)
class Caller:
    token: str
    node: str
    phase: str
    attempt: int
    workspace_host: Path
    staging_host: Path
    mock_script: str | None = None


class TokenRegistry:
    def __init__(self, recorder) -> None:
        self._recorder = recorder
        self._by_token: dict[str, Caller] = {}
        self._lock = threading.Lock()

    def issue(self, *, node: str, phase: str, attempt: int, workspace_host: Path,
              staging_host: Path, mock_script: str | None = None) -> Caller:
        token = "ar-" + secrets.token_urlsafe(32)
        self._recorder.add_redaction(token)
        caller = Caller(token, node, phase, attempt, Path(workspace_host), Path(staging_host),
                        mock_script)
        with self._lock:
            self._by_token[token] = caller
        return caller

    def lookup(self, token: str | None) -> Caller | None:
        with self._lock:
            return self._by_token.get(token or "")

    def revoke(self, token: str) -> None:
        with self._lock:
            self._by_token.pop(token, None)


def bearer(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    return value.strip() or None if scheme.lower() == "bearer" else None


def to_host(caller: Caller, container_path: str) -> Path:
    """Map a container path to the host, refusing anything outside the workspace.

    Resolution follows symlinks, so a link the agent planted inside /workspace
    cannot point the kernel at a host file.
    """
    if not container_path or not container_path.startswith("/"):
        raise PathError(f"{container_path!r} is not an absolute container path")
    path = PurePosixPath(container_path)
    if path == STAGING or STAGING in path.parents:
        base, root = caller.staging_host, STAGING
    elif path == WORKSPACE or WORKSPACE in path.parents:
        base, root = caller.workspace_host, WORKSPACE
    else:
        raise PathError(f"{container_path} is outside /workspace")
    host = (base / path.relative_to(root)).resolve()
    if not host.is_relative_to(base.resolve()):
        raise PathError(f"{container_path} escapes its mount")
    return host


def to_container(caller: Caller, host_path: Path) -> str:
    host = Path(host_path).resolve()
    for base, root in ((caller.staging_host, STAGING), (caller.workspace_host, WORKSPACE)):
        if host.is_relative_to(base.resolve()):
            return str(root / host.relative_to(base.resolve()))
    raise PathError(f"{host} is not inside this caller's workspace")
```

- [ ] **Step 4: Run the path tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_tool_context.py -p no:cacheprovider`
Expected: PASS (14 tests counting the parametrized ones).

- [ ] **Step 5: Write the failing tests for the call store**

```python
# tests/test_gateway_store.py
import pytest

from ar_kernel.gateway.store import CallStore
from ar_kernel.telemetry.recorder import Recorder, TelemetryError
from ar_kernel.tools.context import TokenRegistry


@pytest.fixture
def env(tmp_path):
    rec = Recorder(tmp_path)
    caller = TokenRegistry(rec).issue(node="n1", phase="improve_recipe", attempt=2,
                                      workspace_host=tmp_path, staging_host=tmp_path)
    return rec, CallStore(rec), caller


def _resp(rid, output):
    return {"id": rid, "output": output, "usage": {"input_tokens": 3, "output_tokens": 2}}


FIRST_INPUT = [{"role": "user", "content": "build a dataset"}]
FIRST_OUTPUT = [{"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "data_query",
                 "arguments": "{}", "status": "completed"}]


def test_request_and_response_are_recorded_with_attribution(env):
    rec, store, caller = env
    meta = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(meta, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0.5, attempts=1)
    events = rec.read_events("n1")
    kinds = [e["type"] for e in events]
    assert kinds == ["llm.request", "llm.response"]
    assert all(e["component"] == "gateway" and e["attempt"] == 2 for e in events)
    assert events[0]["phase"] == "improve_recipe"
    assert rec.load_payload(events[1]["payload"])["body"]["id"] == "r1"


def test_persistence_failure_raises_so_the_call_is_never_forwarded(env, monkeypatch):
    rec, store, caller = env
    def broken(*a, **k):
        raise TelemetryError("disk full")
    monkeypatch.setattr(rec, "event", broken)
    with pytest.raises(TelemetryError):
        store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})


def test_previous_response_id_links_the_conversation(env):
    _, store, caller = env
    m1 = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(m1, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0, attempts=1)
    m2 = store.begin(caller, "/v1/responses",
                     {"model": "m", "previous_response_id": "r1",
                      "input": [{"type": "function_call_output", "call_id": "c1", "output": "[]"}]})
    assert m2["conversation_id"] == m1["conversation_id"]
    assert m2["parent_call_id"] == m1["call_id"] and m2["turn_index"] == 1


def test_resent_history_links_by_prefix(env):
    """How a Responses client continues a run (verified for ChatOpenAI, fact 6): it resends
    the prior input and output items, plus the tool output."""
    _, store, caller = env
    m1 = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(m1, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0, attempts=1)
    resent = FIRST_INPUT + [dict(FIRST_OUTPUT[0], status=None)] + [
        {"type": "function_call_output", "call_id": "c1", "output": "[]"}]
    m2 = store.begin(caller, "/v1/responses", {"model": "m", "input": resent})
    assert m2["conversation_id"] == m1["conversation_id"] and m2["turn_index"] == 1


def test_unrelated_request_starts_a_new_conversation(env):
    _, store, caller = env
    m1 = store.begin(caller, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(m1, caller, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0, attempts=1)
    m2 = store.begin(caller, "/v1/responses",
                     {"model": "m", "input": [{"role": "user", "content": "something else"}]})
    assert m2["conversation_id"] != m1["conversation_id"] and m2["turn_index"] == 0


def test_chat_completions_link_by_message_prefix(env):
    _, store, caller = env
    msgs = [{"role": "user", "content": "hi"}]
    m1 = store.begin(caller, "/v1/chat/completions", {"model": "m", "messages": msgs})
    store.end(m1, caller, status=200, latency_s=0, attempts=1,
              body={"id": "cc1", "choices": [{"message": {"role": "assistant", "content": "yo"}}]})
    m2 = store.begin(caller, "/v1/chat/completions", {"model": "m", "messages": msgs + [
        {"role": "assistant", "content": "yo"}, {"role": "user", "content": "more"}]})
    assert m2["conversation_id"] == m1["conversation_id"]


def test_prefix_matching_never_crosses_containers(tmp_path):
    rec = Recorder(tmp_path)
    reg, store = TokenRegistry(rec), CallStore(rec)
    a = reg.issue(node="n1", phase="p", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    b = reg.issue(node="n2", phase="p", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    m1 = store.begin(a, "/v1/responses", {"model": "m", "input": FIRST_INPUT})
    store.end(m1, a, status=200, body=_resp("r1", FIRST_OUTPUT), latency_s=0, attempts=1)
    m2 = store.begin(b, "/v1/responses", {"model": "m", "input": FIRST_INPUT + FIRST_OUTPUT})
    assert m2["conversation_id"] != m1["conversation_id"]
```

- [ ] **Step 6: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_gateway_store.py -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.gateway'`.

- [ ] **Step 7: Implement `gateway/store.py`**

```python
# kernel/ar_kernel/gateway/store.py
"""Fail-closed LLM call records with conversation linking (spec 13.3, row "LLM calls").

Linking, in order: an explicit previous_response_id; a Responses API conversation
id; otherwise the request's history begins with a prior call's request + response
(how ChatOpenAI continues a run, fact 6). Prefix matching only looks at calls from the
same container token.
"""
from __future__ import annotations

import json
import threading
import uuid
from typing import Any

_VOLATILE = {"id", "status"}          # differ between an output item and its resent copy


def _items(endpoint: str, body: dict) -> list[Any]:
    if endpoint.endswith("/chat/completions"):
        return list(body.get("messages") or [])
    raw = body.get("input")
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    return list(raw or [])


def _out_items(endpoint: str, body: dict) -> list[Any]:
    if endpoint.endswith("/chat/completions"):
        return [c.get("message") for c in body.get("choices") or [] if c.get("message")]
    return list(body.get("output") or [])


def _norm(item: Any) -> str:
    if isinstance(item, dict):
        item = {k: v for k, v in item.items() if k not in _VOLATILE and v is not None}
    return json.dumps(item, sort_keys=True, default=str)


class CallStore:
    def __init__(self, recorder) -> None:
        self._rec = recorder
        self._lock = threading.Lock()
        self._calls: dict[str, dict] = {}          # call_id -> linking state
        self._by_response: dict[str, str] = {}     # response id -> call_id
        self._last_in_conv: dict[str, str] = {}    # conversation id -> latest call_id

    def _link(self, caller, endpoint: str, body: dict) -> tuple[str, int, str | None]:
        prev = body.get("previous_response_id")
        if prev and prev in self._by_response:
            parent = self._calls[self._by_response[prev]]
            return parent["conversation_id"], parent["turn_index"] + 1, parent["call_id"]
        conv = body.get("conversation")
        if isinstance(conv, dict):
            conv = conv.get("id")
        if conv:
            cid = f"oai:{conv}"
            parent_id = self._last_in_conv.get(cid)
            turn = self._calls[parent_id]["turn_index"] + 1 if parent_id else 0
            return cid, turn, parent_id
        current = [_norm(i) for i in _items(endpoint, body)]
        for call in reversed(list(self._calls.values())):
            if call["token"] != caller.token or call["endpoint"] != endpoint or not call["done"]:
                continue
            prefix = call["prefix"]
            if prefix and len(prefix) <= len(current) and current[:len(prefix)] == prefix:
                return call["conversation_id"], call["turn_index"] + 1, call["call_id"]
        return uuid.uuid4().hex, 0, None

    def begin(self, caller, endpoint: str, body: dict) -> dict:
        with self._lock:
            conv, turn, parent = self._link(caller, endpoint, body)
            meta = {"call_id": uuid.uuid4().hex, "conversation_id": conv,
                    "turn_index": turn, "parent_call_id": parent}
            # Persist BEFORE the caller forwards anything; a TelemetryError propagates.
            self._rec.event("llm.request", node=caller.node, phase=caller.phase,
                            attempt=caller.attempt, component="gateway",
                            payload={"endpoint": endpoint, "body": body, **meta},
                            call_id=meta["call_id"], conversation_id=conv,
                            turn_index=turn, parent_call_id=parent, model=body.get("model"))
            self._calls[meta["call_id"]] = {**meta, "token": caller.token, "endpoint": endpoint,
                                            "request": [_norm(i) for i in _items(endpoint, body)],
                                            "prefix": None, "done": False}
            self._last_in_conv[conv] = meta["call_id"]
            return meta

    def end(self, meta: dict, caller, *, status: int, body: dict, latency_s: float,
            attempts: int) -> None:
        with self._lock:
            self._rec.event("llm.response", node=caller.node, phase=caller.phase,
                            attempt=caller.attempt, component="gateway",
                            payload={"status": status, "body": body, "latency_s": latency_s,
                                     "attempts": attempts, **meta},
                            call_id=meta["call_id"], conversation_id=meta["conversation_id"],
                            status=status, latency_s=latency_s, attempts=attempts,
                            usage=body.get("usage") if isinstance(body, dict) else None)
            call = self._calls[meta["call_id"]]
            endpoint = call["endpoint"]
            call["prefix"] = call["request"] + [_norm(i) for i in _out_items(endpoint, body or {})]
            call["done"] = True
            if isinstance(body, dict) and body.get("id"):
                self._by_response[body["id"]] = meta["call_id"]
```

- [ ] **Step 8: Run both test files**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_tool_context.py tests/test_gateway_store.py -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add kernel/ar_kernel/tools kernel/ar_kernel/gateway tests/test_tool_context.py tests/test_gateway_store.py
git commit -m "feat(gateway,tools): caller tokens, workspace path mapping, fail-closed call store"
```

---

### Task 5: The recording gateway

An OpenAI-compatible HTTP service. It is the only route from a container to an LLM. It authenticates the per-container token, enforces the model allowlist, persists the request before forwarding and the response before returning, retries upstream 429/5xx with backoff, and serves scripted responses in mock mode (contract smoke runs, tests, and every run before an API key exists).

**Files:**
- Create: `kernel/ar_kernel/gateway/mock.py`, `kernel/ar_kernel/gateway/app.py`
- Test: `tests/test_gateway_app.py`

**Interfaces:**
- Consumes: `CallStore`, `TokenRegistry`, `bearer` (Task 4).
- Produces:
  - `gateway.mock.MockBook`: `.add(name, outputs: list[list[dict]])` registers a script, where each element is one response's `output` list. `.next(name, token) -> list[dict]` returns responses in order and repeats the last one when the script is exhausted. `.response(model, output) -> dict` builds the envelope from verified fact 6. `MockBook.default()` has scripts `smoke` (one assistant message `"ok"`) and `final:<text>` handled on the fly.
  - `gateway.mock.message(text) -> dict` and `gateway.mock.function_call(name, args: dict, call_id) -> dict` are output-item builders for tests and scripts.
  - `gateway.app.Upstream(base_url, api_key, *, timeout_s, retries, transport=None, sleep=asyncio.sleep)`: `await .post(path, body) -> (status, body, attempts)`. `base_url` follows the OpenAI SDK convention and includes the version (`https://api.openai.com/v1`, the default when it is empty or `None`); it is used as given, never rewritten, because providers differ in their version paths. An upstream body that is not JSON (an HTML error page from a proxy) becomes an error payload, and a non-JSON 2xx becomes a **502**; both are retried like any 5xx.
  - `gateway.app.create_gateway_app(*, registry, store, allowed_models: set[str], upstream: Upstream | None, mocks: MockBook) -> FastAPI`. It serves `POST /v1/responses` and `POST /v1/chat/completions`.

Behavior, per request: token → Caller, else **401** · `stream: true` → **400** (the gateway records complete responses; `ChatOpenAI.ainvoke` sends `stream: false`, fact 6) · model not allowed → **403** · `store.begin` (a `TelemetryError` → **500**, nothing forwarded) · mock if `caller.mock_script` or `upstream is None`, else upstream · any exception while producing the response (a gateway bug, an unknown mock script) → **502** with an error body, still passed to `store.end`, so every recorded request gets a recorded response · `store.end` (a `TelemetryError` → **500**) · return the upstream status and body.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gateway_app.py
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from ar_kernel.gateway.app import Upstream, create_gateway_app
from ar_kernel.gateway.mock import MockBook, function_call, message
from ar_kernel.gateway.store import CallStore
from ar_kernel.telemetry.recorder import Recorder, TelemetryError
from ar_kernel.tools.context import TokenRegistry


def _upstream_that(handler, seen):
    def wrapped(request: httpx.Request):
        seen.append(json.loads(request.content))
        return handler(request)
    async def no_sleep(_s):
        pass
    return Upstream("https://api.example/v1", "sk-REALKEY", timeout_s=5, retries=3,
                    transport=httpx.MockTransport(wrapped), sleep=no_sleep)


def _ok(request):
    return httpx.Response(200, json={"id": "resp_up", "object": "response", "output": [],
                                     "usage": {"input_tokens": 1, "output_tokens": 1}})


@pytest.fixture
def make(tmp_path):
    def build(handler=_ok, mock_script=None, allowed=frozenset({"gpt-x"})):
        rec = Recorder(tmp_path)
        reg, store, seen = TokenRegistry(rec), CallStore(rec), []
        caller = reg.issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path,
                           staging_host=tmp_path, mock_script=mock_script)
        app = create_gateway_app(registry=reg, store=store, allowed_models=set(allowed),
                                 upstream=_upstream_that(handler, seen), mocks=MockBook.default())
        return TestClient(app), caller, rec, seen
    return build


def _post(client, token, body, path="/v1/responses"):
    return client.post(path, json=body, headers={"Authorization": f"Bearer {token}"})


def test_unknown_token_is_401(make):
    client, _, _, seen = make()
    assert _post(client, "ar-nope", {"model": "gpt-x", "input": "hi"}).status_code == 401
    assert seen == []


def test_disallowed_model_is_403_and_never_forwarded(make):
    client, caller, _, seen = make()
    assert _post(client, caller.token, {"model": "gpt-other", "input": "hi"}).status_code == 403
    assert seen == []


def test_streaming_is_refused(make):
    client, caller, _, _ = make()
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi", "stream": True}).status_code == 400


def test_forwarded_call_is_recorded_and_upstream_key_never_reaches_telemetry(make, tmp_path):
    client, caller, rec, seen = make()
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 200 and r.json()["id"] == "resp_up"
    assert len(seen) == 1
    kinds = [e["type"] for e in rec.read_events("n1")]
    assert kinds == ["llm.request", "llm.response"]
    blob = b"".join(p.read_bytes() for p in (tmp_path / "telemetry").rglob("*") if p.is_file())
    assert b"sk-REALKEY" not in blob


def test_request_is_persisted_before_forwarding(make, monkeypatch):
    """Fail-closed (spec 13.1.2): if the request cannot be recorded, nothing is sent."""
    client, caller, rec, seen = make()
    def broken(*a, **k):
        raise TelemetryError("disk full")
    monkeypatch.setattr(rec, "event", broken)
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 500
    assert seen == []


def test_upstream_429_is_retried_then_succeeds(make):
    calls = {"n": 0}
    def flaky(request):
        calls["n"] += 1
        return httpx.Response(429, json={"error": "slow down"}) if calls["n"] < 3 else _ok(request)
    client, caller, rec, _ = make(handler=flaky)
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 200
    response_event = [e for e in rec.read_events("n1") if e["type"] == "llm.response"][0]
    assert response_event["attempts"] == 3


def test_persistent_upstream_failure_returns_its_status(make):
    client, caller, _, seen = make(handler=lambda r: httpx.Response(503, json={"error": "down"}))
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 503
    assert len(seen) == 4                     # 1 try + 3 retries


def test_non_json_upstream_error_is_retried_recorded_and_returned(make):
    """Proxies answer 502/503/504 with HTML; that must not crash the gateway."""
    client, caller, rec, seen = make(handler=lambda r: httpx.Response(502, text="<html>Bad Gateway</html>"))
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 502 and "non-JSON" in r.json()["error"]["message"]
    assert len(seen) == 4                     # retried like any 5xx
    response_event = [e for e in rec.read_events("n1") if e["type"] == "llm.response"][0]
    assert response_event["attempts"] == 4 and response_event["status"] == 502


def test_non_json_success_body_becomes_a_502(make):
    client, caller, _, _ = make(handler=lambda r: httpx.Response(200, text="not json"))
    assert _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).status_code == 502


def test_unexpected_gateway_exception_is_still_recorded(make, monkeypatch):
    async def bug(self, path, body):
        raise RuntimeError("gateway bug")
    monkeypatch.setattr(Upstream, "post", bug)
    client, caller, rec, _ = make()
    r = _post(client, caller.token, {"model": "gpt-x", "input": "hi"})
    assert r.status_code == 502 and "gateway bug" in r.json()["error"]["message"]
    assert [e["type"] for e in rec.read_events("n1")] == ["llm.request", "llm.response"]


def test_upstream_base_url_defaults_to_openai_v1_and_is_not_rewritten():
    import asyncio
    seen = []
    def record(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={})
    for base in (None, "", "https://proxy.example/api/v3/"):
        up = Upstream(base, "k", timeout_s=5, retries=0, transport=httpx.MockTransport(record))
        asyncio.run(up.post("/responses", {}))
    assert seen == ["https://api.openai.com/v1/responses", "https://api.openai.com/v1/responses",
                    "https://proxy.example/api/v3/responses"]


def test_mock_mode_serves_scripted_output_without_upstream(make):
    client, caller, rec, seen = make(mock_script="smoke")
    body = _post(client, caller.token, {"model": "gpt-x", "input": "hi"}).json()
    assert body["output"][0]["content"][0]["text"] == "ok"
    assert seen == []
    assert [e["type"] for e in rec.read_events("n1")] == ["llm.request", "llm.response"]


def test_mock_script_is_consumed_in_order_then_repeats_its_last_step(tmp_path):
    rec = Recorder(tmp_path)
    reg, store = TokenRegistry(rec), CallStore(rec)
    book = MockBook()
    book.add("two_steps", [[function_call("data_query", {}, "c1")], [message("done")]])
    caller = reg.issue(node="n", phase="p", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path, mock_script="two_steps")
    client = TestClient(create_gateway_app(registry=reg, store=store, allowed_models={"gpt-x"},
                                           upstream=None, mocks=book))
    kinds = [_post(client, caller.token, {"model": "gpt-x", "input": "x"}).json()["output"][0]["type"]
             for _ in range(3)]
    assert kinds == ["function_call", "message", "message"]


def test_chat_completions_is_forwarded_to_the_matching_path(make):
    seen_paths = []
    def handler(request):
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"id": "cc", "choices": [{"message": {"role": "assistant", "content": "k"}}]})
    client, caller, _, _ = make(handler=handler)
    r = _post(client, caller.token, {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]},
              path="/v1/chat/completions")
    assert r.status_code == 200 and seen_paths == ["/v1/chat/completions"]
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_gateway_app.py -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.gateway.app'`.

- [ ] **Step 3: Implement the mock book**

```python
# kernel/ar_kernel/gateway/mock.py
"""Scripted Responses API outputs. The envelope is the minimum ChatOpenAI
accepts (verified fact 6 in the Plan 2 document)."""
from __future__ import annotations

import json
import threading
import time
import uuid


def message(text: str) -> dict:
    return {"type": "message", "id": f"msg_{uuid.uuid4().hex[:12]}", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def function_call(name: str, args: dict, call_id: str) -> dict:
    return {"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:12]}", "call_id": call_id,
            "name": name, "arguments": json.dumps(args), "status": "completed"}


class MockBook:
    def __init__(self) -> None:
        self._scripts: dict[str, list[list[dict]]] = {}
        self._cursor: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    @classmethod
    def default(cls) -> "MockBook":
        book = cls()
        book.add("smoke", [[message("ok")]])
        return book

    def add(self, name: str, outputs: list[list[dict]]) -> None:
        if not outputs:
            raise ValueError("a mock script needs at least one response")
        self._scripts[name] = outputs

    def next(self, name: str, token: str) -> list[dict]:
        if name.startswith("final:"):
            return [message(name[len("final:"):])]
        script = self._scripts.get(name)
        if script is None:
            raise KeyError(f"no mock script {name!r}")
        with self._lock:
            i = self._cursor.get((name, token), 0)
            self._cursor[(name, token)] = i + 1
        return script[min(i, len(script) - 1)]

    @staticmethod
    def response(model: str, output: list[dict]) -> dict:
        return {"id": f"resp_{uuid.uuid4().hex}", "object": "response",
                "created_at": int(time.time()), "status": "completed", "model": model,
                "output": output, "parallel_tool_calls": True, "tool_choice": "auto",
                "tools": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}}
```

- [ ] **Step 4: Implement the app**

```python
# kernel/ar_kernel/gateway/app.py
"""The only route from a container to an LLM (spec 4.2, 13.1, 13.3)."""
from __future__ import annotations

import asyncio
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..telemetry.recorder import TelemetryError
from ..tools.context import bearer
from .mock import MockBook
from .store import CallStore

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}
# The OpenAI SDK convention: the base URL includes the API version (spec 2: OPENAI_BASE_URL).
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class Upstream:
    def __init__(self, base_url: str | None, api_key: str, *, timeout_s: float, retries: int,
                 transport: httpx.AsyncBaseTransport | None = None, sleep=asyncio.sleep) -> None:
        # Used as given: "/responses" is appended, so the URL must already end in its version
        # ("/v1"). Appending "/v1" here would break providers whose version path differs.
        self._base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout_s
        self._retries = retries
        self._transport = transport
        self._sleep = sleep

    async def post(self, path: str, body: dict) -> tuple[int, dict, int]:
        # path is "/responses" or "/chat/completions"; the base URL already ends in /v1
        attempts, status, payload = 0, 599, {"error": "no attempt made"}
        async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
            for attempt in range(self._retries + 1):
                attempts = attempt + 1
                try:
                    r = await client.post(self._base + path, json=body, headers=self._headers)
                    status = r.status_code
                    try:
                        payload = r.json() if r.content else {}
                    except ValueError:        # e.g. an HTML error page from a proxy in front of the API
                        payload = {"error": {"message": f"upstream returned HTTP {status} with a "
                                                        f"non-JSON body", "body": r.text}}   # full body: no telemetry truncation (§13.1.3)
                        status = status if status >= 400 else 502   # never pass garbage on as success
                except httpx.HTTPError as exc:
                    status, payload = 599, {"error": f"{type(exc).__name__}: {exc}"}
                if status not in RETRYABLE and status != 599:
                    break
                if attempt < self._retries:
                    await self._sleep(2 ** attempt)
        return status, payload, attempts


def create_gateway_app(*, registry, store: CallStore, allowed_models: set[str],
                       upstream: Upstream | None, mocks: MockBook) -> FastAPI:
    app = FastAPI(title="ar-gateway")

    async def handle(request: Request, endpoint: str, upstream_path: str) -> JSONResponse:
        caller = registry.lookup(bearer(request.headers.get("authorization")))
        if caller is None:
            return JSONResponse({"error": {"message": "unknown or revoked token"}}, status_code=401)
        body = await request.json()
        if body.get("stream"):
            return JSONResponse({"error": {"message": "streaming is not supported by the gateway; "
                                                      "use non-streaming calls"}},
                                status_code=400)
        if body.get("model") not in allowed_models:
            return JSONResponse({"error": {"message": f"model {body.get('model')!r} is not in the "
                                                      f"allowlist {sorted(allowed_models)}"}},
                                status_code=403)
        try:
            meta = store.begin(caller, endpoint, body)
        except TelemetryError as exc:
            return JSONResponse({"error": {"message": f"telemetry unavailable: {exc}"}}, status_code=500)
        started = time.monotonic()
        try:
            if caller.mock_script or upstream is None:
                status, payload, attempts = 200, MockBook.response(
                    body.get("model", ""), mocks.next(caller.mock_script or "smoke", caller.token)), 1
            else:
                status, payload, attempts = await upstream.post(upstream_path, body)
        except Exception as exc:  # noqa: BLE001 -- the request is recorded; its failure must be too
            status, payload, attempts = 502, {"error": {"message": f"gateway: {type(exc).__name__}: {exc}"}}, 1
        try:
            store.end(meta, caller, status=status, body=payload,
                      latency_s=time.monotonic() - started, attempts=attempts)
        except TelemetryError as exc:
            return JSONResponse({"error": {"message": f"telemetry unavailable: {exc}"}}, status_code=500)
        return JSONResponse(payload, status_code=status)

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await handle(request, "/v1/responses", "/responses")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await handle(request, "/v1/chat/completions", "/chat/completions")

    return app
```

- [ ] **Step 5: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_gateway_app.py -p no:cacheprovider`
Expected: PASS (16 tests).

- [ ] **Step 6: Commit**

```bash
git add kernel/ar_kernel/gateway tests/test_gateway_app.py
git commit -m "feat(gateway): recording OpenAI-compatible gateway with allowlist, retries and mock mode"
```

---

### Task 6: Tool server core and UDS service lifecycle

The MCP server factory that every kernel tool registers on, plus starting and stopping both services on a run's socket directory. Each tool resolves its caller from the `Authorization` header through the request context (verified: `ctx.request_context.request.headers`), records `tool.call` and then `tool.result` or `tool.error`, and turns *any* exception into a tool error for the agent. Blocking work runs in a thread, so one slow tool does not stall the server.

**Files:**
- Create: `kernel/ar_kernel/tools/server.py`, `kernel/ar_kernel/services.py`
- Test: `tests/test_tool_server.py`

**Interfaces:**
- Consumes: `TokenRegistry`, `bearer` (Task 4); `create_gateway_app` (Task 5).
- Produces:
  - `tools.server.ToolError`: the message the agent sees. *As built:* it subclasses the MCP SDK's `mcp.server.mcpserver.exceptions.ToolError` (still an `Exception`), because MCP 2.2 masks the message of any other exception type as `Error executing tool <name>`.
  - `tools.server.ToolKit(registry, recorder)`: `await .call(ctx, name: str, args: dict, fn: Callable[[Caller], Any]) -> Any`. Resolves the caller, records, runs `fn(caller)` in a worker thread, and re-raises every failure as `ToolError`.
  - `tools.server.build_tool_app(mcp: MCPServer) -> Starlette`, with the transport settings from facts 3–4.
  - `tools.server.new_mcp() -> MCPServer`.
  - `services.socket_dir_for(run_dir: Path) -> Path`, a short per-run directory under the system temp dir (fact 12: Unix socket paths are capped at 107 bytes).
  - `services.RunServices`: `.start(gateway_app, tools_app) -> None`, `.stop() -> None`, `.socket_dir: Path`. It binds `gateway.sock` and `tools.sock` with mode 0600, in a directory with mode 0700.
  - *As built (review fix):* `.stop()` raises `RuntimeError` if a server thread is still alive after its join timeout, instead of silently leaking it.

Tool names use underscores: OpenAI function names must match `^[a-zA-Z0-9_-]+$`, and LangChain passes tool names through as function names. So spec §10's `data.ingest` is registered as `data_ingest`, and so on.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_server.py
"""Exercise the tool server the way a container does: over a Unix socket, with
the ar_contract client configuration."""
import asyncio
import json
import os

import pytest
from mcp.server.mcpserver import Context

from ar_kernel.services import RunServices, socket_dir_for
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.server import ToolError, ToolKit, build_tool_app, new_mcp


def _server(tmp_path):
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    kit = ToolKit(reg, rec)
    mcp = new_mcp()

    @mcp.tool(name="echo_caller")
    async def echo_caller(word: str, ctx: Context) -> dict:
        return await kit.call(ctx, "echo_caller", {"word": word},
                              lambda c: {"node": c.node, "word": word})

    @mcp.tool(name="always_fails")
    async def always_fails(ctx: Context) -> dict:
        def boom(_c):
            raise ToolError("the download was too large")
        return await kit.call(ctx, "always_fails", {}, boom)

    @mcp.tool(name="kernel_bug")
    async def kernel_bug(ctx: Context) -> dict:
        return await kit.call(ctx, "kernel_bug", {}, lambda c: 1 / 0)

    return rec, reg, build_tool_app(mcp)


async def _call(sock_dir, token, tool, args):
    from ar_contract.client import mcp_session
    async with mcp_session(sock_dir, token) as session:
        return await session.call_tool(tool, args)


@pytest.fixture
def live(tmp_path):
    rec, reg, tools_app = _server(tmp_path)
    from fastapi import FastAPI
    services = RunServices(socket_dir_for(tmp_path / "run"))
    services.start(FastAPI(), tools_app)
    caller = reg.issue(node="n7", phase="improve_recipe", attempt=1,
                       workspace_host=tmp_path, staging_host=tmp_path)
    yield rec, reg, services, caller
    services.stop()


def test_socket_dir_is_short_and_private(tmp_path):
    d = socket_dir_for(tmp_path / ("x" * 150))
    assert len(str(d / "gateway.sock")) < 100
    assert socket_dir_for(tmp_path / ("x" * 150)) == d          # deterministic per run


def test_tool_sees_its_caller_over_the_socket(live):
    rec, _, services, caller = live
    result = asyncio.run(_call(services.socket_dir, caller.token, "echo_caller", {"word": "hi"}))
    assert not result.is_error
    assert json.loads(result.content[0].text) == {"node": "n7", "word": "hi"}
    kinds = [e["type"] for e in rec.read_events("n7")]
    assert kinds == ["tool.call", "tool.result"]
    assert all(e["component"] == "tools" for e in rec.read_events("n7"))


def test_tool_error_reaches_the_agent_as_an_error_result(live):
    rec, _, services, caller = live
    result = asyncio.run(_call(services.socket_dir, caller.token, "always_fails", {}))
    assert result.is_error and "too large" in result.content[0].text
    assert [e["type"] for e in rec.read_events("n7")][-1] == "tool.error"


def test_unexpected_kernel_exception_is_contained(live):
    """A bug in a tool must come back as a tool error, never take the server down."""
    rec, _, services, caller = live
    result = asyncio.run(_call(services.socket_dir, caller.token, "kernel_bug", {}))
    assert result.is_error and "ZeroDivisionError" in result.content[0].text
    err = [e for e in rec.read_events("n7") if e["type"] == "tool.error"][0]
    assert "Traceback" in rec.load_payload(err["payload"])["traceback"]
    # the server still answers afterwards
    assert not asyncio.run(_call(services.socket_dir, caller.token, "echo_caller", {"word": "x"})).is_error


def test_unknown_token_is_refused(live):
    _, _, services, _ = live
    result = asyncio.run(_call(services.socket_dir, "ar-forged", "echo_caller", {"word": "hi"}))
    assert result.is_error and "token" in result.content[0].text


def test_sockets_are_owner_only(live):
    _, _, services, _ = live
    assert oct(os.stat(services.socket_dir).st_mode & 0o777) == "0o700"
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_tool_server.py -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the tool server core**

```python
# kernel/ar_kernel/tools/server.py
"""MCP tool server core: caller resolution, telemetry and error containment for
every kernel tool (spec 10, 13.3 row "Tool calls")."""
from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, Callable

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError as _MCPToolError
from mcp.server.transport_security import TransportSecuritySettings

from .context import bearer


class ToolError(_MCPToolError):
    """A failure the agent should see and may act on. Subclasses the SDK's ToolError:
    the SDK masks any other exception's message as "Error executing tool <name>"."""


def new_mcp() -> MCPServer:
    return MCPServer("ar-kernel-tools",
                     instructions="Privileged AutoResearcher kernel tools. Paths are container "
                                  "paths under /workspace; write candidates under /workspace/staging.")


def build_tool_app(mcp: MCPServer):
    return mcp.streamable_http_app(
        # Over a Unix socket the Host header carries no port; without this the
        # rebinding guard answers 421 (verified fact 3).
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=["localhost"], allowed_origins=[]),
        # improve_recipe can go a long time between tool calls (fact 4).
        session_idle_timeout=None,
    )


class ToolKit:
    def __init__(self, registry, recorder) -> None:
        self.registry = registry
        self.recorder = recorder

    async def call(self, ctx: Context, name: str, args: dict, fn: Callable[[Any], Any]) -> Any:
        request = getattr(ctx.request_context, "request", None)
        header = request.headers.get("authorization") if request is not None else None
        caller = self.registry.lookup(bearer(header))
        if caller is None:
            raise ToolError("unknown or revoked token")
        base = dict(node=caller.node, phase=caller.phase, attempt=caller.attempt, component="tools")
        span = self.recorder.event("tool.call", payload={"tool": name, "args": args}, tool=name, **base)
        started = time.monotonic()
        try:
            result = await asyncio.to_thread(fn, caller)
        except ToolError as exc:
            self.recorder.event("tool.error", parent_span_id=span, tool=name,
                                duration_s=time.monotonic() - started,
                                payload={"tool": name, "error": str(exc)}, **base)
            raise
        except Exception as exc:                          # noqa: BLE001 -- contain kernel bugs
            self.recorder.event("tool.error", parent_span_id=span, tool=name,
                                duration_s=time.monotonic() - started,
                                payload={"tool": name, "error": f"{type(exc).__name__}: {exc}",
                                         "traceback": traceback.format_exc()}, **base)
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        self.recorder.event("tool.result", parent_span_id=span, tool=name,
                            duration_s=time.monotonic() - started,
                            payload={"tool": name, "result": result}, **base)
        return result
```

- [ ] **Step 4: Implement the service lifecycle**

```python
# kernel/ar_kernel/services.py
"""Start/stop the gateway and tool server on a run's Unix socket directory."""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from pathlib import Path

import uvicorn


def socket_dir_for(run_dir: Path) -> Path:
    """A short, private, per-run directory. AF_UNIX paths are capped at 107 bytes
    and a socket under runs/<run_id>/ is ~125 (verified fact 12)."""
    digest = hashlib.sha1(str(Path(run_dir).resolve()).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"ar-{digest}"


class RunServices:
    def __init__(self, socket_dir: Path) -> None:
        self.socket_dir = Path(socket_dir)
        self._servers: list[uvicorn.Server] = []
        self._threads: list[threading.Thread] = []

    def start(self, gateway_app, tools_app, ready_timeout_s: float = 20.0) -> None:
        self.socket_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.socket_dir, 0o700)
        for name, app in (("gateway.sock", gateway_app), ("tools.sock", tools_app)):
            path = self.socket_dir / name
            if path.exists():
                path.unlink()
            server = uvicorn.Server(uvicorn.Config(app, uds=str(path), log_level="warning",
                                                   timeout_keep_alive=900))
            thread = threading.Thread(target=server.run, name=f"ar-{name}", daemon=True)
            thread.start()
            self._servers.append(server)
            self._threads.append(thread)
        deadline = time.monotonic() + ready_timeout_s
        while not all(s.started for s in self._servers):
            if time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(f"services did not start within {ready_timeout_s}s")
            time.sleep(0.05)
        for name in ("gateway.sock", "tools.sock"):
            os.chmod(self.socket_dir / name, 0o600)

    def stop(self) -> None:
        for server in self._servers:
            server.should_exit = True
        for thread in self._threads:
            thread.join(timeout=10)
        self._servers.clear()
        self._threads.clear()
        for name in ("gateway.sock", "tools.sock"):
            (self.socket_dir / name).unlink(missing_ok=True)
```

- [ ] **Step 5: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_tool_server.py -p no:cacheprovider`
Expected: PASS (7 tests).

- [ ] **Step 6: Commit**

```bash
git add kernel/ar_kernel/tools/server.py kernel/ar_kernel/services.py tests/test_tool_server.py
git commit -m "feat(tools): MCP tool server core over UDS with caller resolution and error containment"
```

---
### Task 7: Data tools — probe, ingest, query, commit, recipe check

Thin wrappers over the Plan 1 kernel. They map container paths, run the kernel
operation as the caller, and shape results for an agent. No data logic lives
here.

**Files:**
- Create: `kernel/ar_kernel/tools/data_tools.py`
- Modify: `kernel/ar_kernel/archive/db.py` (busy timeout)
- Test: `tests/test_data_tools.py`

**Interfaces:**
- Consumes: `ToolKit`, `ToolError`, `new_mcp` (Task 6); `to_host`, `to_container`, `PathError` (Task 4); Plan 1's `Ingestor`, `Candidate`, `CommitStore`, `CommitError`, `ClipStore`, `NodeStore`, `Gate`, `probe_video`, `open_db`, `BlobStore`.
- Produces:
  - `data_tools.DataTools(cfg, run_dir, recorder, gpus, gpu_lock: threading.Lock)` with plain methods, each taking a `Caller`: `.probe(caller, path) -> dict`, `.ingest(caller, candidates: list[dict]) -> list[dict]`, `.query(caller, filter: dict) -> dict`, `.commit(caller, parent, datasets, message) -> dict`, `.recipe_check(caller, recipe, data_commit) -> dict`.
  - `data_tools.register_data_tools(mcp, kit, tools: DataTools) -> None` registers MCP tools `video_probe`, `data_ingest`, `data_query`, `data_commit`, `recipe_check`.
- Threading: every method opens its **own** connection (`open_db(run_dir)`) in the worker thread and closes it. SQLite connections are bound to their creating thread, and tool calls run in worker threads.
- `recipe_check` holds `gpu_lock` while the gate runs. The gate's describe step launches a GPU job, and verification-log finding 1 showed that GPU contention turns into a spurious gate failure. It materializes into a throwaway directory next to the caller's workspace and deletes it afterwards (spec 5.6, "temporary view").

- [ ] **Step 1: Raise the SQLite busy timeout**

In `kernel/ar_kernel/archive/db.py`, `open_db`:

```python
    # Tool calls and the loop write from different threads through separate
    # connections; wait for a lock instead of failing with "database is locked".
    conn = sqlite3.connect(run_dir / "archive.db", isolation_level=None, timeout=30.0)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_data_tools.py
import threading

import pytest

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.data_tools import DataTools
from ar_kernel.tools.server import ToolError
from conftest import make_mp4, write_caption, write_poses

CFG = KernelConfig.load()
PROV = {"kind": "derived", "from": [], "transform": "unit test"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Build once: ingest runs WorldModel's checker (~5 s per clip)."""
    run = tmp_path_factory.mktemp("run")
    rec = Recorder(run)
    nodes = NodeStore(open_db(run))
    nodes.create("root", None, 0)
    nodes.create("n1", "root", 1)
    tools = DataTools(CFG, run, rec, [0, 1, 2, 3], threading.Lock())
    reg = TokenRegistry(rec)
    ws = run / "nodes" / "n1" / "attempts" / "improve_recipe-1" / "workspace"
    st = run / "staging" / "n1" / "improve_recipe-1"
    ws.mkdir(parents=True), st.mkdir(parents=True)
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=st)
    cands = []
    for i in range(4):
        seconds = 4.0 + 0.5 * i                      # distinct content -> distinct clips
        d = st / f"c{i}"
        make_mp4(d / "v.mp4", seconds=seconds)
        write_caption(d / "c.json")
        write_poses(d / "p.npz", n_frames=int(seconds * 30))
        cands.append({"video": f"/workspace/staging/c{i}/v.mp4",
                      "caption": f"/workspace/staging/c{i}/c.json",
                      "pose": f"/workspace/staging/c{i}/p.npz",
                      "camera_motion": "moving", "provenance": PROV})
    results = tools.ingest(caller, cands)
    return tools, caller, results, run


def test_ingest_accepts_staged_candidates_by_container_path(env):
    _, _, results, _ = env
    assert all(r["accepted"] for r in results), results
    assert len({r["clip_id"] for r in results}) == 4
    assert all("video_caption_camera" in r["formats"] for r in results)


def test_ingest_refuses_paths_outside_the_workspace(env):
    tools, caller, _, _ = env
    with pytest.raises(ToolError, match="outside /workspace"):
        tools.ingest(caller, [{"video": "/etc/passwd", "caption": "/workspace/staging/x.json",
                               "camera_motion": "moving", "provenance": PROV}])


def test_ingest_rejects_a_malformed_candidate_as_a_tool_error(env):
    tools, caller, _, _ = env
    with pytest.raises(ToolError, match="provenance"):
        tools.ingest(caller, [{"video": "/workspace/staging/x.mp4",
                               "caption": "/workspace/staging/x.json", "camera_motion": "moving"}])


def test_probe_reports_display_geometry(env):
    tools, caller, _, run = env
    make_mp4(caller.staging_host / "probe_me.mp4", seconds=3.0)
    info = tools.probe(caller, "/workspace/staging/probe_me.mp4")
    assert info["width"] == 736 and info["rotation"] == 0 and abs(info["display_aspect"] - 736 / 414) < 1e-6


def test_query_returns_the_archive_wide_pool_with_provenance(env):
    tools, caller, results, _ = env
    out = tools.query(caller, {"format": "video_caption_camera"})
    ids = {c["clip_id"] for c in out["clips"]}
    assert {r["clip_id"] for r in results} <= ids
    clip = next(c for c in out["clips"] if c["clip_id"] == results[0]["clip_id"])
    assert clip["provenance"] == PROV and clip["ingested_by"] == "n1" and clip["used_by_scores"] == []


def test_commit_returns_id_and_per_dataset_stats(env):
    tools, caller, results, _ = env
    out = tools.commit(caller, None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                              "weight": 1.0, "clips": [r["clip_id"] for r in results]}},
                       "four clips")
    assert len(out["commit_id"]) == 64
    assert out["datasets"]["cam"] == {"format": "video_caption_camera", "prompt_mode": None,
                                      "weight": 1.0, "clips": 4}


def test_commit_validation_errors_become_tool_errors(env):
    tools, caller, results, _ = env
    with pytest.raises(ToolError, match="more than once"):
        tools.commit(caller, None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                            "weight": 1.0, "clips": [results[0]["clip_id"]] * 2}}, "dup")


def test_recipe_check_reports_gate_failures_without_leaving_a_view(env):
    tools, caller, results, _ = env
    commit = tools.commit(caller, None, {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                                 "weight": 1.0, "clips": [r["clip_id"] for r in results]}},
                          "for check")["commit_id"]
    out = tools.recipe_check(caller, {"spatial_memory.enabled": False}, commit)
    assert out["ok"] is False and any("not tunable" in f for f in out["failures"])
    leftovers = list((caller.workspace_host.parent / "recipe_check").glob("*"))
    assert leftovers == []


def test_register_names_are_openai_safe():
    import re
    from ar_kernel.tools.data_tools import TOOL_NAMES
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", n) for n in TOOL_NAMES)
```

- [ ] **Step 3: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_data_tools.py -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'ar_kernel.tools.data_tools'`.

- [ ] **Step 4: Implement**

```python
# kernel/ar_kernel/tools/data_tools.py
"""Data tools (spec 10): video_probe, data_ingest, data_query, data_commit, recipe_check."""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context

from ..archive.blobs import BlobStore
from ..archive.clips import ClipStore
from ..archive.commits import CommitError, CommitStore
from ..archive.db import open_db
from ..archive.nodes import NodeStore
from ..data.ingest import Candidate, Ingestor
from ..data.probe import probe_video
from ..train.gate import Gate
from .context import PathError, to_host
from .server import ToolError

TOOL_NAMES = ("video_probe", "data_ingest", "data_query", "data_commit", "recipe_check")
QUERY_LIMIT = 500


class DataTools:
    def __init__(self, cfg, run_dir: Path, recorder, gpus: list[int], gpu_lock: threading.Lock) -> None:
        self.cfg, self.run_dir, self.recorder = cfg, Path(run_dir), recorder
        self.gpus, self.gpu_lock = list(gpus), gpu_lock

    def _host(self, caller, path: str) -> Path:
        try:
            return to_host(caller, path)
        except PathError as exc:
            raise ToolError(str(exc)) from exc

    def probe(self, caller, path: str) -> dict:
        info = probe_video(self._host(caller, path))
        return {"frames": info.frames, "fps": info.fps, "width": info.width, "height": info.height,
                "duration": info.duration, "rotation": info.rotation, "sar": info.sar,
                "display_aspect": info.display_aspect}

    def ingest(self, caller, candidates: list[dict]) -> list[dict]:
        built = []
        for i, c in enumerate(candidates):
            if not c.get("provenance"):
                raise ToolError(f"candidate {i}: provenance is required (spec 5.4)")
            for key in ("video", "caption", "camera_motion"):
                if not c.get(key):
                    raise ToolError(f"candidate {i}: {key} is required")
            built.append(Candidate(
                video=self._host(caller, c["video"]), caption=self._host(caller, c["caption"]),
                pose=self._host(caller, c["pose"]) if c.get("pose") else None,
                camera_motion=c["camera_motion"], provenance=c["provenance"],
                license=c.get("license"), derived_from=list(c.get("derived_from") or [])))
        conn = open_db(self.run_dir)
        try:
            results = Ingestor(self.cfg, self.run_dir, conn, self.recorder).ingest(built, node_id=caller.node)
        finally:
            conn.close()
        return [{"accepted": r.accepted, "clip_id": r.clip_id, "formats": r.formats,
                 "warnings": r.warnings, "reasons": r.reasons} for r in results]

    def query(self, caller, filter: dict) -> dict:
        conn = open_db(self.run_dir)
        try:
            clips = ClipStore(conn).all()
            usage = scores_by_clip(conn)
        finally:
            conn.close()
        fmt, motion = filter.get("format"), filter.get("camera_motion")
        wanted = set(filter.get("clip_ids") or [])
        out = []
        for clip in clips:
            if fmt and not any(f == fmt or f.startswith(fmt + ":") for f in clip["formats"]):
                continue
            if motion and clip["camera_motion"] != motion:
                continue
            if wanted and clip["clip_id"] not in wanted:
                continue
            if filter.get("ingested_by") and clip.get("ingested_by") != filter["ingested_by"]:
                continue
            out.append({"clip_id": clip["clip_id"], "formats": clip["formats"],
                        "camera_motion": clip["camera_motion"], "metadata": clip["metadata"],
                        "warnings": clip["warnings"], "provenance": clip["provenance"],
                        "license": clip.get("license"), "derived_from": clip.get("derived_from"),
                        "ingested_by": clip.get("ingested_by"),
                        "used_by_scores": usage.get(clip["clip_id"], [])})
        limit = int(filter.get("limit") or QUERY_LIMIT)
        return {"total": len(out), "returned": min(limit, len(out)), "clips": out[:limit]}

    def commit(self, caller, parent: str | None, datasets: dict, message: str) -> dict:
        conn = open_db(self.run_dir)
        try:
            store = CommitStore(conn, BlobStore(self.run_dir, conn), ClipStore(conn))
            try:
                commit_id = store.commit(parent, datasets, message, node_id=caller.node,
                                         attempt=caller.attempt)
            except CommitError as exc:
                raise ToolError(str(exc)) from exc
            manifest = store.manifest(commit_id)
        finally:
            conn.close()
        return {"commit_id": commit_id, "datasets": {
            name: {"format": d["format"], "prompt_mode": d["prompt_mode"], "weight": d["weight"],
                   "clips": len(d["clips"])} for name, d in manifest["datasets"].items()}}

    def recipe_check(self, caller, recipe: dict, data_commit: str) -> dict:
        scratch = caller.workspace_host.parent / "recipe_check" / uuid.uuid4().hex
        conn = open_db(self.run_dir)
        try:
            store = CommitStore(conn, BlobStore(self.run_dir, conn), ClipStore(conn))
            try:
                store.manifest(data_commit)
            except Exception as exc:
                raise ToolError(f"unknown data commit {data_commit!r}") from exc
            parent_commit = _parent_commit(conn, caller.node)
            with self.gpu_lock:          # the describe step is a GPU job (verification finding 1)
                result = Gate(self.cfg, store, self.recorder).check(
                    recipe, data_commit, parent_commit, caller.node, scratch, self.run_dir, self.gpus)
        finally:
            conn.close()
            shutil.rmtree(scratch, ignore_errors=True)
        return {"ok": result.ok, "failures": result.failures}


def _parent_commit(conn, node_id: str) -> str | None:
    nodes = NodeStore(conn)
    try:
        parent_id = nodes.get(node_id)["parent_id"]
        return nodes.get(parent_id)["data_commit"] if parent_id else None
    except KeyError:
        return None


def scores_by_clip(conn) -> dict[str, list[float]]:
    """Scores of scored nodes whose data commit contains each clip."""
    out: dict[str, list[float]] = {}
    rows = conn.execute("SELECT n.score, c.manifest FROM nodes n JOIN data_commits c "
                        "ON n.data_commit = c.commit_id WHERE n.status = 'scored'").fetchall()
    for row in rows:
        clip_ids = {cid for d in json.loads(row["manifest"])["datasets"].values() for cid in d["clips"]}
        for cid in clip_ids:
            out.setdefault(cid, []).append(float(row["score"]))
    return out


def register_data_tools(mcp, kit, tools: DataTools) -> None:
    @mcp.tool(name="video_probe", description="Frame count, fps, coded size, rotation, pixel "
              "aspect and display aspect of a video under /workspace.")
    async def video_probe(path: str, ctx: Context) -> dict:
        return await kit.call(ctx, "video_probe", {"path": path}, lambda c: tools.probe(c, path))

    @mcp.tool(name="data_ingest", description="Ingest staged candidates (spec 5.5). Each: video, "
              "caption, optional pose (container paths under /workspace/staging), camera_motion "
              "'moving'|'static', provenance, optional license and derived_from. Returns accepted + "
              "clip_id + eligible formats, or rejected + reasons, per candidate.")
    async def data_ingest(candidates: list[dict[str, Any]], ctx: Context) -> list[dict]:
        return await kit.call(ctx, "data_ingest", {"candidates": candidates},
                              lambda c: tools.ingest(c, candidates))

    @mcp.tool(name="data_query", description="Search the archive-wide clip pool. Filter keys: "
              "format, camera_motion, clip_ids, ingested_by, limit. Each clip includes provenance, "
              "metadata, eligible formats and the scores of nodes that trained on it.")
    async def data_query(filter: dict[str, Any], ctx: Context) -> dict:
        return await kit.call(ctx, "data_query", {"filter": filter}, lambda c: tools.query(c, filter))

    @mcp.tool(name="data_commit", description="Create an immutable data commit (spec 5.6). "
              "datasets: {name: {format, prompt_mode, weight, clips: [clip_id]}}.")
    async def data_commit(parent: str | None, datasets: dict[str, Any], message: str,
                          ctx: Context) -> dict:
        return await kit.call(ctx, "data_commit",
                              {"parent": parent, "datasets": datasets, "message": message},
                              lambda c: tools.commit(c, parent, datasets, message))

    @mcp.tool(name="recipe_check", description="Run every recipe-gate check (spec 8) on a recipe "
              "and data commit without consuming an attempt. Returns ok and the failures.")
    async def recipe_check(recipe: dict[str, Any], data_commit: str, ctx: Context) -> dict:
        return await kit.call(ctx, "recipe_check", {"recipe": recipe, "data_commit": data_commit},
                              lambda c: tools.recipe_check(c, recipe, data_commit))
```

`ClipStore.all()` rows expose `ingested_by`, `license` and `derived_from` under exactly
those names (verified against `kernel/ar_kernel/archive/clips.py`).

- [ ] **Step 5: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_data_tools.py -p no:cacheprovider`
Expected: PASS (9 tests; the module fixture ingests 4 clips once).

- [ ] **Step 6: Run the full suite (the busy-timeout change touches every DB user)**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add kernel/ar_kernel/tools/data_tools.py kernel/ar_kernel/archive/db.py tests/test_data_tools.py
git commit -m "feat(tools): data tools over the kernel -- probe, ingest, query, commit, recipe check"
```

---

### Task 8: Hugging Face tools

`hf_search` and `hf_download` (spec 10). Downloads pin the revision to a commit SHA, refuse anything over the byte cap **before** transferring, land only under the caller's staging directory, and return what an ingest provenance record needs (`{"kind": "hf_dataset", "repo", "revision", "files"}`, spec 5.4).

**Files:**
- Create: `kernel/ar_kernel/tools/hf_tools.py`
- Test: `tests/test_hf_tools.py`

**Interfaces:**
- Consumes: `ToolKit`, `ToolError` (Task 6), `to_container` (Task 4).
- Produces:
  - `hf_tools.HfTools(cfg, private_dir, api=None, snapshot=None)` (`private_dir` added in the final review: downloads land there, then move into staging without following links). `api` defaults to `huggingface_hub.HfApi()` and `snapshot` to `huggingface_hub.snapshot_download`; both are injectable for tests.
  - `.search(caller, query, kind="dataset", limit=20) -> list[dict]`.
  - `.download(caller, repo, revision, patterns, max_bytes=None) -> dict`.
  - `hf_tools.register_hf_tools(mcp, kit, tools)` registers `hf_search` and `hf_download`.
- The kernel process has network access and `HF_TOKEN` from the shell; the container has neither.
- *As built:* `hf_search` returns ids, license, tags, download counts and last-modified time but **no sizes** (sizes need a per-repo `dataset_info(files_metadata=True)` call; `hf_download` makes it and checks the byte cap before transferring). This is a spec §10 amendment. `hf_download` also refuses repo-reported file names that are absolute or contain `..` (review fix).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hf_tools.py
"""No network: HfApi and snapshot_download are replaced by fakes."""
from types import SimpleNamespace

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.hf_tools import HfTools
from ar_kernel.tools.server import ToolError

CFG = KernelConfig.load()
SHA = "a" * 40


class FakeApi:
    def list_datasets(self, search, limit, full=True):
        return [SimpleNamespace(id="org/walks", tags=["license:cc-by-4.0", "task:video"],
                                downloads=12, last_modified=None, card_data={"license": "cc-by-4.0"})]

    def dataset_info(self, repo_id, revision=None, files_metadata=False):
        return SimpleNamespace(id=repo_id, sha=SHA, card_data={"license": "cc-by-4.0"},
                               siblings=[SimpleNamespace(rfilename="videos/a.mp4", size=1000),
                                         SimpleNamespace(rfilename="videos/b.mp4", size=3000),
                                         SimpleNamespace(rfilename="README.md", size=10)])


def fake_snapshot(repo_id, repo_type, revision, allow_patterns, local_dir, **_):
    from pathlib import Path
    for name in ("videos/a.mp4", "videos/b.mp4"):
        p = Path(local_dir) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return local_dir


@pytest.fixture
def env(tmp_path):
    reg = TokenRegistry(Recorder(tmp_path))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1,
                       workspace_host=tmp_path / "ws", staging_host=tmp_path / "staging")
    (tmp_path / "staging").mkdir()
    return HfTools(CFG, api=FakeApi(), snapshot=fake_snapshot), caller


def test_search_reports_license(env):
    tools, caller = env
    hits = tools.search(caller, "walking", "dataset", 5)
    assert hits[0]["id"] == "org/walks" and hits[0]["license"] == "cc-by-4.0"


def test_download_pins_revision_and_lands_in_staging(env):
    tools, caller = env
    out = tools.download(caller, "org/walks", "main", ["videos/*.mp4"])
    assert out["revision"] == SHA and out["bytes"] == 4000
    assert out["files"] == [f"/workspace/staging/hf/org__walks/{SHA}/videos/a.mp4",
                            f"/workspace/staging/hf/org__walks/{SHA}/videos/b.mp4"]
    assert out["provenance"] == {"kind": "hf_dataset", "repo": "org/walks", "revision": SHA,
                                 "files": ["videos/a.mp4", "videos/b.mp4"]}
    assert out["license"] == "cc-by-4.0"


def test_download_over_the_cap_is_refused_before_transfer(env):
    tools, caller = env
    with pytest.raises(ToolError, match="4000 bytes"):
        tools.download(caller, "org/walks", "main", ["videos/*.mp4"], max_bytes=3500)
    assert not (caller.staging_host / "hf").exists()


def test_patterns_matching_nothing_is_an_error(env):
    tools, caller = env
    with pytest.raises(ToolError, match="no files match"):
        tools.download(caller, "org/walks", "main", ["*.parquet"])


def test_repo_ids_cannot_escape_staging(env):
    tools, caller = env
    with pytest.raises(ToolError):
        tools.download(caller, "../../etc", "main", ["*"])
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_hf_tools.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# kernel/ar_kernel/tools/hf_tools.py
"""hf_search / hf_download (spec 10). The kernel downloads; the container has no network."""
from __future__ import annotations

import fnmatch
import re
from typing import Any

from mcp.server.mcpserver import Context

from .context import to_container
from .server import ToolError

REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _license(obj) -> str | None:
    card = getattr(obj, "card_data", None) or getattr(obj, "cardData", None) or {}
    if isinstance(card, dict) and card.get("license"):
        lic = card["license"]
        return ",".join(lic) if isinstance(lic, list) else str(lic)
    for tag in getattr(obj, "tags", None) or []:
        if str(tag).startswith("license:"):
            return str(tag).split(":", 1)[1]
    return None


class HfTools:
    def __init__(self, cfg, api=None, snapshot=None) -> None:
        if api is None or snapshot is None:
            import huggingface_hub
            api = api or huggingface_hub.HfApi()
            snapshot = snapshot or huggingface_hub.snapshot_download
        self.api, self.snapshot = api, snapshot
        self.cap = int(cfg.get("tools.hf_download_max_bytes"))

    def search(self, caller, query: str, kind: str = "dataset", limit: int = 20) -> list[dict]:
        if kind != "dataset":
            raise ToolError("only kind='dataset' is supported: models are not training data")
        return [{"id": d.id, "license": _license(d), "tags": list(d.tags or [])[:40],
                 "downloads": getattr(d, "downloads", None),
                 "last_modified": str(getattr(d, "last_modified", None))}
                for d in self.api.list_datasets(search=query, limit=min(int(limit), 100), full=True)]

    def download(self, caller, repo: str, revision: str, patterns: list[str],
                 max_bytes: int | None = None) -> dict:
        if not REPO_RE.match(repo or ""):
            raise ToolError(f"invalid dataset repo id {repo!r}")
        info = self.api.dataset_info(repo, revision=revision, files_metadata=True)
        files = sorted(s.rfilename for s in info.siblings
                       if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns))
        if not files:
            raise ToolError(f"no files match {patterns} in {repo}@{revision}")
        sizes = {s.rfilename: int(s.size or 0) for s in info.siblings}
        total = sum(sizes[f] for f in files)
        cap = min(self.cap, int(max_bytes)) if max_bytes else self.cap
        if total > cap:
            raise ToolError(f"{len(files)} files total {total} bytes, over the {cap}-byte cap; "
                            f"narrow the patterns")
        dest = caller.staging_host / "hf" / repo.replace("/", "__") / info.sha
        self.snapshot(repo_id=repo, repo_type="dataset", revision=info.sha,
                      allow_patterns=files, local_dir=str(dest))
        return {"repo": repo, "revision": info.sha, "bytes": total, "license": _license(info),
                "files": [to_container(caller, dest / f) for f in files],
                "provenance": {"kind": "hf_dataset", "repo": repo, "revision": info.sha,
                               "files": files}}


def register_hf_tools(mcp, kit, tools: HfTools) -> None:
    @mcp.tool(name="hf_search", description="Search Hugging Face datasets. Returns ids, license and tags.")
    async def hf_search(query: str, ctx: Context, kind: str = "dataset", limit: int = 20) -> list[dict]:
        return await kit.call(ctx, "hf_search", {"query": query, "kind": kind, "limit": limit},
                              lambda c: tools.search(c, query, kind, limit))

    @mcp.tool(name="hf_download", description="Download files matching glob patterns from a dataset "
              "repo into /workspace/staging/hf/. The revision is pinned to a commit SHA; the result "
              "includes a ready-made provenance record for data_ingest.")
    async def hf_download(repo: str, revision: str, patterns: list[str], ctx: Context,
                          max_bytes: int | None = None) -> dict[str, Any]:
        return await kit.call(ctx, "hf_download",
                              {"repo": repo, "revision": revision, "patterns": patterns,
                               "max_bytes": max_bytes},
                              lambda c: tools.download(c, repo, revision, patterns, max_bytes))
```

- [ ] **Step 4: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_hf_tools.py -p no:cacheprovider`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/tools/hf_tools.py tests/test_hf_tools.py
git commit -m "feat(tools): hf_search and a pinned, capped hf_download into staging"
```

---

### Task 9: GPU job API

The asynchronous job mechanism behind every GPU tool (spec 10): a queue that runs one job at a time on the GPU list and holds the same `gpu_lock` as `recipe_check`, plus `job_status`, `job_wait` (capped) and `job_cancel` (process-group kill). Plan 3 registers real backends (`rollout_*`, `annotate_camera`). This task ships the mechanism with a test-only fake backend, and a cancellable-subprocess helper that the Plan 3 backends must use.

**Files:**
- Create: `kernel/ar_kernel/tools/jobs.py`
- Test: `tests/test_jobs.py`

**Interfaces:**
- Consumes: `ToolKit`, `ToolError` (Task 6); `subproc._kill_group` (Plan 1 review fix).
- Produces:
  - `jobs.JobBackend` (Protocol): `name: str`, `tool: str` (the MCP tool name), `run(job: Job, cancel: threading.Event, report: Callable[[dict], None]) -> dict`.
  - `jobs.Job` (dataclass): `id, backend, args, node, token, state, progress, result, error, created, started, finished`.
  - `jobs.JobQueue(recorder, gpu_lock: threading.Lock, wait_cap_s: float)`: `.register(backend)`, `.submit(caller, backend_name, args) -> str`, `.status(caller, job_id) -> dict`, `.wait(caller, job_id, timeout_s) -> dict`, `.cancel(caller, job_id) -> dict`, `.cancel_for_token(token) -> int`, `.shutdown()`. States: `queued`, `running`, `done`, `failed`, `cancelled`.
  - `jobs.run_cancellable(env, args, *, cwd, cancel, extra_env=None, log_path, poll_s=1.0) -> int`. It starts the command in its own session and kills the **group** when `cancel` is set. It returns the exit code, or -15 when cancelled. *As built:* it is a thin wrapper over the Plan 1 launcher rather than a second launcher: `subproc.run_in_env` gained `cancel: threading.Event | None = None` and `poll_s`; with `cancel` set it polls, kills the process group on cancel, records a distinct `subproc.cancelled` event and sets `.cancelled` on its result. `run_cancellable` also takes `recorder`, `node` and `phase` (passed through for telemetry) and returns -15 whenever `.cancelled` is true, even if SIGKILL was needed.
  - *As built:* `JobQueue.shutdown()` raises `RuntimeError` if the worker thread outlives its join deadline (a job may still hold a GPU); `job_wait` also caps a non-finite `timeout_s`.
  - `jobs.register_job_tools(mcp, kit, queue)` registers `job_status`, `job_wait`, `job_cancel`.
- A caller can see and cancel only jobs its own token submitted. `cancel_for_token` runs when a phase's container exits (Task 15), so an agent's orphaned jobs never keep the GPUs.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_jobs.py
import os
import threading
import time

import pytest

from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue, run_cancellable
from ar_kernel.tools.server import ToolError


class SleepyBackend:
    name, tool = "sleepy", "rollout_fake"

    def run(self, job, cancel, report):
        for i in range(int(job.args.get("steps", 3))):
            if cancel.is_set():
                return {"cancelled_at": i}
            report({"step": i + 1})
            time.sleep(job.args.get("dt", 0.05))
        return {"clip": "/workspace/staging/fake.mp4"}


class BrokenBackend:
    name, tool = "broken", "rollout_broken"

    def run(self, job, cancel, report):
        raise RuntimeError("CUDA out of memory")


@pytest.fixture
def env(tmp_path):
    rec = Recorder(tmp_path)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=1.0)
    q.register(SleepyBackend())
    q.register(BrokenBackend())
    reg = TokenRegistry(rec)
    a = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    b = reg.issue(node="n2", phase="improve_recipe", attempt=1, workspace_host=tmp_path, staging_host=tmp_path)
    yield q, a, b, rec
    q.shutdown()


def test_submit_returns_immediately_and_wait_collects_the_result(env):
    q, a, _, _ = env
    started = time.monotonic()
    job_id = q.submit(a, "sleepy", {"steps": 3})
    assert time.monotonic() - started < 0.5
    out = q.wait(a, job_id, 30)
    assert out["state"] == "done" and out["result"] == {"clip": "/workspace/staging/fake.mp4"}
    assert out["progress"] == {"step": 3}


def test_wait_is_capped_and_returns_running(env):
    """Spec 10: job.wait never blocks past tools.job_wait_max_s (1 s here)."""
    q, a, _, _ = env
    job_id = q.submit(a, "sleepy", {"steps": 200, "dt": 0.05})
    started = time.monotonic()
    out = q.wait(a, job_id, 10_000)
    assert out["state"] == "running" and time.monotonic() - started < 2.5


def test_backend_failure_is_a_failed_job_not_a_crash(env):
    q, a, _, _ = env
    out = q.wait(a, q.submit(a, "broken", {}), 30)
    assert out["state"] == "failed" and "out of memory" in out["error"]


def test_jobs_run_one_at_a_time(env):
    q, a, _, _ = env
    first = q.submit(a, "sleepy", {"steps": 20, "dt": 0.05})
    second = q.submit(a, "sleepy", {"steps": 1})
    time.sleep(0.2)
    assert q.status(a, first)["state"] == "running"
    assert q.status(a, second)["state"] == "queued"


def test_cancel_stops_a_running_job(env):
    q, a, _, _ = env
    job_id = q.submit(a, "sleepy", {"steps": 500, "dt": 0.05})
    time.sleep(0.2)
    q.cancel(a, job_id)
    assert q.wait(a, job_id, 30)["state"] == "cancelled"


def test_callers_cannot_see_each_others_jobs(env):
    q, a, b, _ = env
    job_id = q.submit(a, "sleepy", {"steps": 1})
    with pytest.raises(ToolError, match="no job"):
        q.status(b, job_id)
    with pytest.raises(ToolError, match="no job"):
        q.cancel(b, job_id)


def test_cancel_for_token_clears_an_ended_phase(env):
    q, a, _, _ = env
    running = q.submit(a, "sleepy", {"steps": 500, "dt": 0.05})
    queued = q.submit(a, "sleepy", {"steps": 500})
    time.sleep(0.2)
    assert q.cancel_for_token(a.token) == 2
    assert q.wait(a, running, 30)["state"] == "cancelled"
    assert q.status(a, queued)["state"] == "cancelled"


def test_unknown_backend_is_a_tool_error(env):
    q, a, _, _ = env
    with pytest.raises(ToolError, match="not enabled"):
        q.submit(a, "wan22", {})


def test_run_cancellable_kills_the_whole_group(tmp_path):
    """Plan 3 backends launch torch jobs through this; a cancel must not orphan ranks."""
    pidfile = tmp_path / "child.pid"
    cancel = threading.Event()
    threading.Timer(6.0, cancel.set).start()
    code = run_cancellable("autoresearcher", ["bash", "-c", f"sleep 600 & echo $! > {pidfile}; wait"],
                           cwd=tmp_path, cancel=cancel, log_path=tmp_path / "job.log")
    assert code == -15
    child = int(pidfile.read_text())
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            os.kill(child, 0)
            with open(f"/proc/{child}/stat") as fh:
                if fh.read().split()[2] == "Z":
                    break
        except (ProcessLookupError, FileNotFoundError):
            break
        time.sleep(0.2)
    else:
        pytest.fail("grandchild survived the cancel")
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_jobs.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# kernel/ar_kernel/tools/jobs.py
"""Asynchronous GPU jobs (spec 10). One job at a time, under the run's GPU lock."""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from mcp.server.mcpserver import Context

from ..subproc import _kill_group
from .server import ToolError


class JobBackend(Protocol):
    name: str
    tool: str

    def run(self, job: "Job", cancel: threading.Event, report: Callable[[dict], None]) -> dict: ...


@dataclass
class Job:
    id: str
    backend: str
    args: dict
    node: str
    token: str
    state: str = "queued"
    progress: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None


_TERMINAL = {"done", "failed", "cancelled"}


class JobQueue:
    def __init__(self, recorder, gpu_lock: threading.Lock, wait_cap_s: float) -> None:
        self.recorder, self.gpu_lock, self.wait_cap_s = recorder, gpu_lock, float(wait_cap_s)
        self._backends: dict[str, JobBackend] = {}
        self._jobs: dict[str, Job] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._cond = threading.Condition()
        self._todo: "queue.Queue[str | None]" = queue.Queue()
        self._worker = threading.Thread(target=self._loop, name="ar-gpu-jobs", daemon=True)
        self._worker.start()

    def register(self, backend: JobBackend) -> None:
        self._backends[backend.name] = backend

    @property
    def backends(self) -> dict[str, JobBackend]:
        return dict(self._backends)

    def _own(self, caller, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None or job.token != caller.token:
            raise ToolError(f"no job {job_id!r} for this caller")
        return job

    def _view(self, job: Job) -> dict:
        view = asdict(job)
        view.pop("token")
        return view

    def submit(self, caller, backend_name: str, args: dict) -> str:
        if backend_name not in self._backends:
            raise ToolError(f"generator {backend_name!r} is not enabled on this run")
        job = Job(id=uuid.uuid4().hex, backend=backend_name, args=dict(args),
                  node=caller.node, token=caller.token)
        with self._cond:
            self._jobs[job.id] = job
            self._cancel[job.id] = threading.Event()
        self.recorder.event("job.submitted", node=caller.node, phase=caller.phase,
                            attempt=caller.attempt, component="tools", job_id=job.id,
                            payload={"backend": backend_name, "args": args})
        self._todo.put(job.id)
        return job.id

    def status(self, caller, job_id: str) -> dict:
        with self._cond:
            return self._view(self._own(caller, job_id))

    def wait(self, caller, job_id: str, timeout_s: float) -> dict:
        deadline = time.monotonic() + min(float(timeout_s), self.wait_cap_s)
        with self._cond:
            job = self._own(caller, job_id)
            while job.state not in _TERMINAL:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            return self._view(job)

    def cancel(self, caller, job_id: str) -> dict:
        with self._cond:
            job = self._own(caller, job_id)
            self._cancel_locked(job)
            return self._view(job)

    def cancel_for_token(self, token: str) -> int:
        with self._cond:
            live = [j for j in self._jobs.values() if j.token == token and j.state not in _TERMINAL]
            for job in live:
                self._cancel_locked(job)
            return len(live)

    def _cancel_locked(self, job: Job) -> None:
        self._cancel[job.id].set()
        if job.state == "queued":
            job.state, job.finished = "cancelled", time.time()
            self._cond.notify_all()

    def shutdown(self) -> None:
        with self._cond:
            for job in self._jobs.values():
                if job.state not in _TERMINAL:
                    self._cancel_locked(job)
        self._todo.put(None)
        self._worker.join(timeout=30)

    def _loop(self) -> None:
        while True:
            job_id = self._todo.get()
            if job_id is None:
                return
            with self._cond:
                job = self._jobs[job_id]
                if job.state != "queued":
                    continue
                job.state, job.started = "running", time.time()
                self._cond.notify_all()
            cancel = self._cancel[job_id]

            def report(progress: dict, _job=job) -> None:
                with self._cond:
                    _job.progress = dict(progress)
                    self._cond.notify_all()

            with self.gpu_lock:
                try:
                    result = self._backends[job.backend].run(job, cancel, report)
                    final, error = ("cancelled" if cancel.is_set() else "done"), None
                except Exception as exc:                  # noqa: BLE001 -- a job failure, not a crash
                    result, final = None, "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    tb = traceback.format_exc()
            with self._cond:
                job.state, job.result, job.error, job.finished = final, result, error, time.time()
                self._cond.notify_all()
            self.recorder.event("job.finished", node=job.node, component="tools", job_id=job.id,
                                state=final, gpu_seconds=job.finished - job.started,
                                payload={"result": result, "error": error,
                                         "traceback": tb if final == "failed" else None})


def run_cancellable(env: str, args: list[str], *, cwd: Path, cancel: threading.Event,
                    extra_env: dict | None = None, log_path: Path, poll_s: float = 1.0) -> int:
    """Run a GPU job's command, killing its whole process group on cancel.

    Backends must use this, not subprocess directly: killing only the launcher
    orphans torch ranks that keep holding GPU memory (verification finding 6).
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as sink:
        proc = subprocess.Popen(["conda", "run", "--no-capture-output", "-n", env, *args],
                                cwd=str(cwd), env={**os.environ, **(extra_env or {})},
                                stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            if cancel.wait(poll_s):
                _kill_group(proc)
                proc.wait()
                return -15
        return proc.returncode


def register_job_tools(mcp, kit, q: JobQueue) -> None:
    @mcp.tool(name="job_status", description="State, progress and (when finished) the result of a GPU job.")
    async def job_status(job_id: str, ctx: Context) -> dict[str, Any]:
        return await kit.call(ctx, "job_status", {"job_id": job_id}, lambda c: q.status(c, job_id))

    @mcp.tool(name="job_wait", description="Wait for a GPU job, at most 300 s per call; returns "
              "state 'running' if it has not finished. Call again to keep waiting.")
    async def job_wait(job_id: str, ctx: Context, timeout_s: float = 300) -> dict[str, Any]:
        return await kit.call(ctx, "job_wait", {"job_id": job_id, "timeout_s": timeout_s},
                              lambda c: q.wait(c, job_id, timeout_s))

    @mcp.tool(name="job_cancel", description="Cancel a queued or running GPU job.")
    async def job_cancel(job_id: str, ctx: Context) -> dict[str, Any]:
        return await kit.call(ctx, "job_cancel", {"job_id": job_id}, lambda c: q.cancel(c, job_id))
```

- [ ] **Step 4: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_jobs.py -p no:cacheprovider`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/tools/jobs.py tests/test_jobs.py
git commit -m "feat(tools): asynchronous GPU job queue with capped wait and process-group cancel"
```

---
### Task 10: The agent image

A CPU-only image holding everything agent code may use at runtime. The container has no network, so nothing can be installed later. It is built in two layers: a base (system tools plus the pinned framework set) and a thin per-requirements layer tagged by the hash of the node's `agent/requirements.txt`, which children usually share (spec 9.4.1).

**Files:**
- Create: `docker/agent.Dockerfile`
- Create: `kernel/ar_kernel/sandbox/__init__.py` (empty), `kernel/ar_kernel/sandbox/image.py`
- Test: `tests/test_sandbox_image.py`

**Interfaces:**
- Consumes: `KernelConfig` (`sandbox.image`).
- Produces:
  - `sandbox.image.base_tag(cfg) -> str`, e.g. `ar-agent-base:<sha12 of the Dockerfile>`.
  - `sandbox.image.image_tag(cfg, requirements: str) -> str`, e.g. `ar-agent:<sha12 of base tag + requirements>`.
  - `sandbox.image.ensure_image(cfg, requirements: str, *, recorder=None, node="run") -> str` builds if missing and returns the tag. It raises `ImageBuildError` with the build log tail.
  - `sandbox.image.image_exists(tag) -> bool`.

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# docker/agent.Dockerfile -- base image for agent containers (CPU only, no network at runtime)
FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
      "langgraph==1.2.11" "langchain-core==1.6.3" "langchain-openai==1.6.2" \
      "mcp==2.2.0" "httpx==0.28.1" "httpx2==2.13.0" \
      "pydantic>=2.12,<3" "numpy>=1.26" "opencv-python-headless>=4.9" "Pillow>=10"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_sandbox_image.py
import subprocess

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.sandbox.image import base_tag, ensure_image, image_tag

CFG = KernelConfig.load()


def test_tags_are_deterministic_and_requirements_sensitive():
    assert image_tag(CFG, "requests==2.0\n") == image_tag(CFG, "requests==2.0\n")
    assert image_tag(CFG, "requests==2.0\n") != image_tag(CFG, "requests==2.1\n")
    assert base_tag(CFG).startswith("ar-agent-base:")


def test_requirement_order_and_blank_lines_do_not_change_the_tag():
    assert image_tag(CFG, "b==1\na==1\n") == image_tag(CFG, "\na==1\n\nb==1")


@pytest.mark.docker
def test_built_image_has_the_runtime_stack_and_no_network_needed():
    tag = ensure_image(CFG, "")
    out = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", tag, "python", "-c",
         "import mcp, httpx, httpx2, langgraph, langchain_core, langchain_openai, cv2, numpy, PIL; "
         "from langgraph.types import Send; print('ok')"],
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    ff = subprocess.run(["docker", "run", "--rm", "--network", "none", tag, "ffprobe", "-version"],
                        capture_output=True, text=True, timeout=120)
    assert ff.returncode == 0
```

- [ ] **Step 3: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_sandbox_image.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Implement**

```python
# kernel/ar_kernel/sandbox/image.py
"""Agent images: a pinned base plus a per-requirements layer (spec 9.4.1)."""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path


class ImageBuildError(RuntimeError):
    """The agent image could not be built; carries the build log tail."""


def _dockerfile(cfg) -> Path:
    return cfg.repo_root / "docker" / "agent.Dockerfile"


def _norm_requirements(requirements: str) -> str:
    return "\n".join(sorted(line.strip() for line in requirements.splitlines() if line.strip())) + "\n"


def base_tag(cfg) -> str:
    digest = hashlib.sha256(_dockerfile(cfg).read_bytes()).hexdigest()[:12]
    return f"{cfg.get('sandbox.image')}-base:{digest}"


def image_tag(cfg, requirements: str) -> str:
    digest = hashlib.sha256((base_tag(cfg) + _norm_requirements(requirements)).encode()).hexdigest()[:12]
    return f"{cfg.get('sandbox.image')}:{digest}"


def image_exists(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def _build(tag: str, dockerfile_text: str, context: Path, recorder, node: str) -> None:
    proc = subprocess.run(["docker", "build", "-t", tag, "-f", "-", str(context)],
                          input=dockerfile_text, capture_output=True, text=True, timeout=3600)
    if recorder is not None:
        recorder.event("sandbox.image_build", node=node, component="sandbox", tag=tag,
                       returncode=proc.returncode,
                       payload={"dockerfile": dockerfile_text, "stdout": proc.stdout, "stderr": proc.stderr})
    if proc.returncode != 0:
        raise ImageBuildError(f"docker build {tag} failed:\n{(proc.stdout + proc.stderr)[-4000:]}")


def ensure_image(cfg, requirements: str, *, recorder=None, node: str = "run") -> str:
    base = base_tag(cfg)
    if not image_exists(base):
        _build(base, _dockerfile(cfg).read_text(), cfg.repo_root / "docker", recorder, node)
    tag = image_tag(cfg, requirements)
    if image_exists(tag):
        return tag
    reqs = _norm_requirements(requirements)
    with tempfile.TemporaryDirectory() as ctx:
        (Path(ctx) / "requirements.txt").write_text(reqs)
        install = ("RUN pip install --no-cache-dir -r /tmp/requirements.txt\n"
                   if reqs.strip() else "")
        _build(tag, f"FROM {base}\nCOPY requirements.txt /tmp/requirements.txt\n{install}",
               Path(ctx), recorder, node)
    return tag
```

- [ ] **Step 5: Run the tests (unit, then docker)**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_sandbox_image.py -p no:cacheprovider`
Expected: PASS (2 unit tests; the docker test is deselected).

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_sandbox_image.py -m docker -p no:cacheprovider`
Expected: PASS. The first build takes a few minutes.

- [ ] **Step 6: Commit**

```bash
git add docker/agent.Dockerfile kernel/ar_kernel/sandbox tests/test_sandbox_image.py
git commit -m "feat(sandbox): pinned agent base image plus per-requirements layer"
```

---

### Task 11: The sandbox runner

One container per agent call (spec 9.5): the mount table, no network, the host user, CPU and memory caps, a read-only root filesystem, the timeout, stats sampling, logs, and guaranteed removal. It is the only function in the kernel that runs agent code.

**Files:**
- Create: `kernel/ar_kernel/sandbox/runner.py`
- Test: `tests/test_sandbox_runner.py`

**Interfaces:**
- Consumes: `ensure_image` (Task 10), `Recorder`.
- Produces:
  - `sandbox.runner.Mounts` (dataclass): `agent, workspace, staging, context, store, contract, sockets: Path` and `agent_readonly: bool = False`.
  - `sandbox.runner.RunResult` (dataclass): `exit_code: int | None, timed_out: bool, stdout: str, stderr: str, duration_s: float, stats: list[dict], container: str`.
  - `sandbox.runner.container_name(run_id, node, phase, attempt) -> str` returns `ar-<run_id>-<node>-<phase>-<attempt>-<rand>`, sanitized to Docker's name rules (spec 9.5: run prefix `ar-<run_id>-`).
  - `sandbox.runner.run_container(*, image, name, mounts, command, env, cpus, memory_gb, timeout_s, recorder, node, phase, attempt, stats_every_s=30.0) -> RunResult`.
  - `sandbox.runner.snapshot(root: Path, hash_files: bool) -> dict[str, str]` and `sandbox.runner.diff(before, after) -> dict` (`added`, `removed`, `changed`), used for the §13.3 filesystem diff.
- Guarantees: the container is removed on every exit path; a timeout kills the container (`docker kill` stops every process in it, so no orphans); the result records `timed_out`.
- *As built:* the argv recorded in telemetry has the `AR_TOKEN` value replaced by `<redacted>` (defence in depth; the real argv keeps it), and a `docker run -d` that fails after creating the container still removes it (review fix).

Container flags: `--network none --user <uid>:<gid> --read-only --tmpfs /tmp:rw,size=4g --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 --cpus <n> --memory <g>g`. Mounts: `/agent` (rw, or ro for `improve_recipe`), `/workspace` rw, `/workspace/staging` rw (nested), `/context` ro, `/store` ro, `/ar_contract` ro, `/run/ar` rw (the sockets). The runner always sets `HOME=/workspace/.home`, `PYTHONPATH=/ar_contract`, `AR_SOCKET_DIR=/run/ar`, `AR_AGENT_DIR=/agent`, `AR_CONTEXT_DIR=/context`, `AR_WORKSPACE=/workspace`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sandbox_runner.py
import os

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.sandbox.image import ensure_image
from ar_kernel.sandbox.runner import Mounts, container_name, diff, run_container, snapshot
from ar_kernel.telemetry.recorder import Recorder

CFG = KernelConfig.load()


def test_container_name_is_prefixed_and_docker_safe():
    name = container_name("20260921_1200", "n/7 x", "improve_recipe", 2)
    assert name.startswith("ar-20260921_1200-n-7-x-improve_recipe-2-")
    assert all(c.isalnum() or c in "_.-" for c in name)


def test_snapshot_diff_reports_changes(tmp_path):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    before = snapshot(tmp_path, hash_files=True)
    (tmp_path / "a.txt").write_text("changed")
    (tmp_path / "b.txt").unlink()
    (tmp_path / "c.txt").write_text("new")
    assert diff(before, snapshot(tmp_path, hash_files=True)) == {
        "added": ["c.txt"], "removed": ["b.txt"], "changed": ["a.txt"]}


@pytest.fixture
def mounts(tmp_path):
    dirs = {k: tmp_path / k for k in ("agent", "workspace", "context", "store", "sockets")}
    for d in dirs.values():
        d.mkdir()
    staging = dirs["workspace"] / "staging"
    staging.mkdir()
    (dirs["store"] / "blob.bin").write_bytes(b"x")
    return Mounts(agent=dirs["agent"], workspace=dirs["workspace"], staging=staging,
                  context=dirs["context"], store=dirs["store"],
                  contract=CFG.repo_root / "contract", sockets=dirs["sockets"])


def _run(tmp_path, mounts, script, timeout_s=120):
    return run_container(image=ensure_image(CFG, ""), name=container_name("t", "n1", "test", 1),
                         mounts=mounts, command=["python", "-c", script], env={},
                         cpus=2, memory_gb=2, timeout_s=timeout_s, recorder=Recorder(tmp_path / "run"),
                         node="n1", phase="test", attempt=1, stats_every_s=1)


@pytest.mark.docker
def test_isolation_holds_from_inside(tmp_path, mounts):
    """Spec 16.3 item 4: no internet, no host services, no kernel/WorldModel/WBench/.env,
    no writes to /store or /context; files written are owned by the host user."""
    script = r"""
import os, socket, urllib.request, json
out = {}
try: urllib.request.urlopen("https://pypi.org", timeout=4); out["internet"] = "open"
except Exception: out["internet"] = "blocked"
s = socket.socket(); s.settimeout(2)
try: out["host_ssh"] = "open" if s.connect_ex(("172.17.0.1", 22)) == 0 else "closed"
except OSError: out["host_ssh"] = "unreachable"
out["visible"] = [p for p in ("/mnt/biometrics", "/kernel", "/WorldModel", "/WBench") if os.path.exists(p)]
for target in ("/store/new.bin", "/context/new.json", "/etc/new"):
    try: open(target, "w").write("x"); out[target] = "writable"
    except OSError: out[target] = "denied"
open("/workspace/staging/made.txt", "w").write("x")
print(json.dumps(out))
"""
    res = _run(tmp_path, mounts, script)
    assert res.exit_code == 0, res.stderr
    import json
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["internet"] == "blocked"
    assert out["host_ssh"] in ("closed", "unreachable")
    assert out["visible"] == []
    assert out["/store/new.bin"] == out["/context/new.json"] == out["/etc/new"] == "denied"
    assert os.stat(mounts.staging / "made.txt").st_uid == os.getuid()


@pytest.mark.docker
def test_timeout_kills_and_removes_the_container(tmp_path, mounts):
    import subprocess
    res = _run(tmp_path, mounts, "import time; print('started', flush=True); time.sleep(600)", timeout_s=8)
    assert res.timed_out and "started" in res.stdout
    listed = subprocess.run(["docker", "ps", "-a", "--filter", f"name={res.container}", "-q"],
                            capture_output=True, text=True).stdout.strip()
    assert listed == ""


@pytest.mark.docker
def test_exit_code_and_streams_are_captured(tmp_path, mounts):
    res = _run(tmp_path, mounts, "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)")
    assert res.exit_code == 3 and "out" in res.stdout and "err" in res.stderr and not res.timed_out
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_sandbox_runner.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# kernel/ar_kernel/sandbox/runner.py
"""Run one agent call in one container (spec 9.5)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Mounts:
    agent: Path
    workspace: Path
    staging: Path
    context: Path
    store: Path
    contract: Path
    sockets: Path
    agent_readonly: bool = False


@dataclass
class RunResult:
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    duration_s: float
    stats: list[dict] = field(default_factory=list)
    container: str = ""


def container_name(run_id: str, node: str, phase: str, attempt: int) -> str:
    raw = f"ar-{run_id}-{node}-{phase}-{attempt}-{secrets.token_hex(3)}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)


def snapshot(root: Path, hash_files: bool) -> dict[str, str]:
    """path -> fingerprint. Hash small code trees; use size+mtime for workspaces
    that may hold gigabytes of video."""
    out = {}
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            rel = str(p.relative_to(root))
            if hash_files:
                out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
            else:
                st = p.stat()
                out[rel] = f"{st.st_size}:{st.st_mtime_ns}"
    return out


def diff(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {"added": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after)),
            "changed": sorted(k for k in set(before) & set(after) if before[k] != after[k])}


def _docker_args(image, name, mounts: Mounts, command, env, cpus, memory_gb) -> list[str]:
    base_env = {"HOME": "/workspace/.home", "PYTHONPATH": "/ar_contract", "AR_SOCKET_DIR": "/run/ar",
                "AR_AGENT_DIR": "/agent", "AR_CONTEXT_DIR": "/context", "AR_WORKSPACE": "/workspace"}
    args = ["docker", "run", "-d", "--name", name,
            "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
            "--read-only", "--tmpfs", "/tmp:rw,size=4g",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "4096",
            "--cpus", str(cpus), "--memory", f"{memory_gb}g",
            "-v", f"{mounts.agent}:/agent:{'ro' if mounts.agent_readonly else 'rw'}",
            "-v", f"{mounts.workspace}:/workspace:rw",
            "-v", f"{mounts.staging}:/workspace/staging:rw",
            "-v", f"{mounts.context}:/context:ro",
            "-v", f"{mounts.store}:/store:ro",
            "-v", f"{mounts.contract}:/ar_contract:ro",
            "-v", f"{mounts.sockets}:/run/ar:ro"]   # final review: connect works, deleting a socket does not
    for key, value in {**base_env, **env}.items():
        args += ["-e", f"{key}={value}"]
    return args + [image, *command]


def run_container(*, image: str, name: str, mounts: Mounts, command: list[str], env: dict,
                  cpus: float, memory_gb: float, timeout_s: float, recorder, node: str, phase: str,
                  attempt: int, stats_every_s: float = 30.0) -> RunResult:
    (Path(mounts.workspace) / ".home").mkdir(parents=True, exist_ok=True)
    args = _docker_args(image, name, mounts, command, env, cpus, memory_gb)
    base = dict(node=node, phase=phase, attempt=attempt, component="sandbox")
    recorder.event("sandbox.start", container=name, payload={"args": args}, **base)
    started = time.monotonic()
    stats: list[dict] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(stats_every_s):
            r = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}", name],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                stats.append({"t": time.monotonic() - started, **json.loads(r.stdout)})

    exit_code, timed_out = None, False
    launched = subprocess.run(args, capture_output=True, text=True)
    if launched.returncode != 0:
        recorder.event("sandbox.error", payload={"stderr": launched.stderr}, container=name, **base)
        return RunResult(None, False, "", launched.stderr, time.monotonic() - started, [], name)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        try:
            waited = subprocess.run(["docker", "wait", name], capture_output=True, text=True,
                                    timeout=timeout_s)
            exit_code = int(waited.stdout.strip() or -1)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["docker", "kill", name], capture_output=True)
        logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        stdout, stderr = logs.stdout, logs.stderr
    finally:
        stop.set()
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    result = RunResult(exit_code, timed_out, stdout, stderr, time.monotonic() - started, stats, name)
    recorder.event("sandbox.end", container=name, exit_code=exit_code, timed_out=timed_out,
                   duration_s=result.duration_s,
                   payload={"stdout": stdout, "stderr": stderr, "stats": stats}, **base)
    return result
```

- [ ] **Step 4: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_sandbox_runner.py -p no:cacheprovider`
Expected: PASS (3 unit tests).

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_sandbox_runner.py -m docker -p no:cacheprovider`
Expected: PASS (3 docker tests, including the isolation test).

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/sandbox/runner.py tests/test_sandbox_runner.py
git commit -m "feat(sandbox): isolated container runner -- no network, host uid, caps, kill, stats, diff"
```

---

### Task 12: The agent code repository

`runs/<run>/agents.git` (spec 5.2): the root commit is `seed_agent/`; a child's code lives on `node/<id>`; every attempt, failed ones included, is committed under `refs/attempts/<node>/<phase>-<k>` (for example `refs/attempts/n1/edit_self-2`). This uses git plumbing with a private index, so the repository never needs a checked-out working tree.

**Files:**
- Create: `kernel/ar_kernel/vcs/__init__.py` (empty), `kernel/ar_kernel/vcs/agents_repo.py`
- Test: `tests/test_agents_repo.py`

**Interfaces:**
- Produces `vcs.agents_repo.AgentsRepo(path: Path)`:
  - `.init(seed_dir: Path) -> str`: create a bare repo and commit the seed on `node/root`. Returns the commit.
  - `.checkout(commit: str, dest: Path) -> None`: materialize a tree into `dest`, which must be empty or missing.
  - `.commit_tree(src: Path, parent: str | None, message: str) -> str`: commit `src` (honoring its `.gitignore`).
  - `.set_ref(ref: str, commit: str) -> None`; `.attempt_ref(node, phase, attempt) -> str` returns `refs/attempts/<node>/<phase>-<k>`; `.branch_ref(node) -> str` returns `refs/heads/node/<node>`.
  - `.resolve(ref: str) -> str | None`.
  - `.diff(a: str, b: str) -> str`; `.diff_stats(a, b) -> list[dict]` (`path`, `added`, `removed`).
  - `.read_file(commit: str, path: str) -> str | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agents_repo.py
import pytest

from ar_kernel.vcs.agents_repo import AgentsRepo


@pytest.fixture
def seed(tmp_path):
    s = tmp_path / "seed"
    (s / "agent").mkdir(parents=True)
    (s / "agent" / "entry.py").write_text("def edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n")
    (s / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    return s


def test_init_commits_the_seed_on_the_root_branch(tmp_path, seed):
    repo = AgentsRepo(tmp_path / "agents.git")
    root = repo.init(seed)
    assert repo.resolve(repo.branch_ref("root")) == root
    assert "def edit_self" in repo.read_file(root, "agent/entry.py")


def test_checkout_then_commit_attempt_records_parent_and_ref(tmp_path, seed):
    repo = AgentsRepo(tmp_path / "agents.git")
    root = repo.init(seed)
    work = tmp_path / "work"
    repo.checkout(root, work)
    (work / "agent" / "entry.py").write_text("# edited\ndef edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n")
    (work / "agent" / "__pycache__").mkdir()
    (work / "agent" / "__pycache__" / "x.pyc").write_bytes(b"junk")
    child = repo.commit_tree(work, root, "edit_self attempt 1")
    repo.set_ref(repo.attempt_ref("n1", "edit_self", 1), child)
    assert repo.resolve("refs/attempts/n1/edit_self-1") == child
    assert repo.read_file(child, "agent/__pycache__/x.pyc") is None       # .gitignore honored
    stats = repo.diff_stats(root, child)
    assert stats == [{"path": "agent/entry.py", "added": 1, "removed": 0}]


def test_failed_attempts_keep_their_refs(tmp_path, seed):
    repo = AgentsRepo(tmp_path / "agents.git")
    root = repo.init(seed)
    for k in (1, 2):
        work = tmp_path / f"w{k}"
        repo.checkout(root, work)
        (work / "agent" / f"try{k}.py").write_text("x")
        repo.set_ref(repo.attempt_ref("n1", "edit_self", k), repo.commit_tree(work, root, f"attempt {k}"))
    assert repo.resolve("refs/attempts/n1/edit_self-1") and repo.resolve("refs/attempts/n1/edit_self-2")


def test_checkout_refuses_a_non_empty_destination(tmp_path, seed):
    repo = AgentsRepo(tmp_path / "agents.git")
    root = repo.init(seed)
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "f").write_text("x")
    with pytest.raises(FileExistsError):
        repo.checkout(root, tmp_path / "busy")


def test_resolve_unknown_ref_is_none(tmp_path, seed):
    repo = AgentsRepo(tmp_path / "agents.git")
    repo.init(seed)
    assert repo.resolve("refs/heads/node/nope") is None
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_agents_repo.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# kernel/ar_kernel/vcs/agents_repo.py
"""agents.git: one bare repo per run holding every agent version and attempt (spec 5.2)."""
from __future__ import annotations

import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

_IDENT = ["-c", "user.name=AutoResearcher kernel", "-c", "user.email=kernel@autoresearcher.local"]


class AgentsRepo:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _git(self, *args: str, work_tree: Path | None = None, index: Path | None = None,
             input: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
        env = {**os.environ}
        if index is not None:
            env["GIT_INDEX_FILE"] = str(index)
        cmd = ["git", *_IDENT, "--git-dir", str(self.path)]
        if work_tree is not None:
            cmd += ["--work-tree", str(work_tree)]
        return subprocess.run(cmd + list(args), capture_output=True, env=env, input=input, check=check)

    def init(self, seed_dir: Path) -> str:
        subprocess.run(["git", "init", "--bare", "-q", str(self.path)], check=True)
        root = self.commit_tree(seed_dir, None, "seed agent")
        self.set_ref(self.branch_ref("root"), root)
        return root

    def commit_tree(self, src: Path, parent: str | None, message: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            self._git("add", "-A", ".", work_tree=src, index=index)
            tree = self._git("write-tree", work_tree=src, index=index).stdout.decode().strip()
        args = ["commit-tree", tree, "-m", message] + (["-p", parent] if parent else [])
        return self._git(*args).stdout.decode().strip()

    def set_ref(self, ref: str, commit: str) -> None:
        self._git("update-ref", ref, commit)

    @staticmethod
    def attempt_ref(node: str, phase: str, attempt: int) -> str:
        return f"refs/attempts/{node}/{phase}-{attempt}"

    @staticmethod
    def branch_ref(node: str) -> str:
        return f"refs/heads/node/{node}"

    def resolve(self, ref: str) -> str | None:
        r = self._git("rev-parse", "--verify", "--quiet", ref + "^{commit}", check=False)
        return r.stdout.decode().strip() or None if r.returncode == 0 else None

    def checkout(self, commit: str, dest: Path) -> None:
        dest = Path(dest)
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f"{dest} is not empty")
        dest.mkdir(parents=True, exist_ok=True)
        archive = self._git("archive", "--format=tar", commit).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(dest, filter="data")

    def read_file(self, commit: str, path: str) -> str | None:
        r = self._git("show", f"{commit}:{path}", check=False)
        return r.stdout.decode() if r.returncode == 0 else None

    def diff(self, a: str, b: str) -> str:
        return self._git("diff", a, b).stdout.decode()

    def diff_stats(self, a: str, b: str) -> list[dict]:
        out = []
        for line in self._git("diff", "--numstat", a, b).stdout.decode().splitlines():
            added, removed, path = line.split("\t", 2)
            out.append({"path": path, "added": int(added) if added != "-" else 0,
                        "removed": int(removed) if removed != "-" else 0})
        return out
```

- [ ] **Step 4: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_agents_repo.py -p no:cacheprovider`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/vcs tests/test_agents_repo.py
git commit -m "feat(vcs): agents.git with node branches and per-attempt refs"
```

---

### Task 13: Context bundles

Build the `EditContext` / `RecipeContext` a container receives at `/context/context.json` (spec 9.3), from the archive. This is also where file conventions are fixed: Plan 4 writes per-node artifacts to these paths and this code reads them.

Per-node artifact conventions (Plan 4 writes, this task reads; each is optional):
`runs/<run>/nodes/<n>/recipe.yaml` (the agent's recipe),
`runs/<run>/nodes/<n>/rationale.md`,
`runs/<run>/nodes/<n>/edit.json` (the passing `edit_self` result: `summary` and `component`), and
`runs/<run>/nodes/<n>/eval/aggregates.json` (the `aggregates()` output).

**Files:**
- Create: `kernel/ar_kernel/context_bundle.py`
- Test: `tests/test_context_bundle.py`

**Interfaces:**
- Consumes: `NodeStore`, `CommitStore`, `ClipStore`, `BlobStore`, `open_db`; `AgentsRepo.diff_stats` (Task 12); `TUNABLE_KEYS`, `RECIPE_RULES` (Plan 1 review); `ar_contract.models` (Task 3); `data_tools.scores_by_clip` (Task 7).
- Produces:
  - `context_bundle.lineage(conn, run_dir, repo, node_id) -> list[dict]`: root first, ending at `node_id`. Each entry has `node_id, status, score, metrics, data (per-dataset format/weight/clip count), recipe, rationale, edit (summary and component, or None), aggregates, code_diff_stats`.
  - `context_bundle.archive_summary(conn) -> dict`: `nodes` (id, parent, status, score, subtree_value, depth), `n_scored`, `best`.
  - `context_bundle.clip_pool_summary(conn, cap=2000) -> list[dict]`.
  - `context_bundle.build_edit_context(*, conn, run_dir, repo, parent_id, attempt, max_attempts, retry, nodes_remaining, dry_run=False) -> EditContext`.
  - `context_bundle.build_recipe_context(*, cfg, conn, run_dir, repo, node_id, parent_id, attempt, max_attempts, retry, nodes_remaining, n_gpus, tools, dry_run=False) -> RecipeContext`.
  - `context_bundle.write_bundle(ctx, dest: Path) -> Path` writes `context.json`, plus `retry.json` when `ctx.retry` is set (spec 7.2).
  - `context_bundle.FORMAT_RULES: str`, the §6.2 rules as agent-facing text.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_context_bundle.py
import json

import pytest
import yaml

from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig
from ar_kernel.context_bundle import (FORMAT_RULES, archive_summary, build_edit_context,
                                      build_recipe_context, lineage, write_bundle)
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()


@pytest.fixture
def world(tmp_path):
    seed = tmp_path / "seed" / "agent"
    seed.mkdir(parents=True)
    (seed / "entry.py").write_text("x = 1\n")
    repo = AgentsRepo(tmp_path / "agents.git")
    root_commit = repo.init(tmp_path / "seed")
    conn = open_db(tmp_path)
    nodes = NodeStore(conn)
    nodes.create("root", None, 0)
    nodes.set_fields("root", agent_commit=root_commit)
    nodes.record_score("root", 0.78, ["aesthetic_quality"], {"aesthetic_quality": 0.78})
    nodes.create("n1", "root", 1)
    (tmp_path / "nodes" / "root" / "eval").mkdir(parents=True)
    (tmp_path / "nodes" / "root" / "eval" / "aggregates.json").write_text(
        json.dumps({"category": {"Nature": 0.8}}))
    (tmp_path / "nodes" / "root" / "rationale.md").write_text("released checkpoint")
    (tmp_path / "nodes" / "root" / "edit.json").write_text(json.dumps({"summary": "s", "component": "tools"}))
    return conn, repo, tmp_path


def test_lineage_is_root_first_and_carries_artifacts(world):
    conn, repo, run = world
    lin = lineage(conn, run, repo, "root")
    assert [e["node_id"] for e in lin] == ["root"]
    assert lin[0]["score"] == 0.78 and lin[0]["aggregates"] == {"category": {"Nature": 0.8}}
    assert lin[0]["rationale"] == "released checkpoint"
    assert lin[0]["edit"] == {"summary": "s", "component": "tools"}


def test_archive_summary_names_the_best_node(world):
    conn, _, _ = world
    s = archive_summary(conn)
    assert s["best"] == {"node_id": "root", "score": 0.78} and s["n_scored"] == 1


def test_edit_context_validates_and_round_trips(world, tmp_path):
    conn, repo, run = world
    ctx = build_edit_context(conn=conn, run_dir=run, repo=repo, parent_id="root", attempt=1,
                             max_attempts=3, retry=None, nodes_remaining=9)
    path = write_bundle(ctx, tmp_path / "ctx")
    loaded = json.loads(path.read_text())
    assert loaded["nodes_remaining"] == 9 and loaded["lineage"][0]["node_id"] == "root"
    assert not (tmp_path / "ctx" / "retry.json").exists()


def test_recipe_context_carries_rules_allowlists_and_retry(world, tmp_path):
    conn, repo, run = world
    retry = {"attempt": 1, "failures": ["dataset cam has 2 clips, fewer than 4 GPUs"]}
    ctx = build_recipe_context(cfg=CFG, conn=conn, run_dir=run, repo=repo, node_id="n1",
                               parent_id="root", attempt=2, max_attempts=3, retry=retry,
                               nodes_remaining=9, n_gpus=4, tools=["data_query", "data_commit"])
    assert "optimizer.max_steps" in ctx.tunable_rules
    assert ctx.resolution_allowlist == [[416, 736], [352, 608]]
    assert ctx.base_recipe["optimizer"]["batch_size"] == 1
    assert "57 frames" in ctx.format_rules and ctx.format_rules == FORMAT_RULES
    write_bundle(ctx, tmp_path / "ctx2")
    assert json.loads((tmp_path / "ctx2" / "retry.json").read_text()) == retry


def test_parent_recipe_is_read_from_the_node_artifact(world):
    conn, repo, run = world
    (run / "nodes" / "root" / "recipe.yaml").write_text(yaml.safe_dump({"optimizer.lr": 2e-5}))
    ctx = build_recipe_context(cfg=CFG, conn=conn, run_dir=run, repo=repo, node_id="n1",
                               parent_id="root", attempt=1, max_attempts=3, retry=None,
                               nodes_remaining=1, n_gpus=4, tools=[])
    assert ctx.parent_recipe == {"optimizer.lr": 2e-5}
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_context_bundle.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# kernel/ar_kernel/context_bundle.py
"""What an agent sees (spec 9.3): built from the archive, written to /context."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from ar_contract.models import EditContext, RecipeContext

from .archive.clips import ClipStore
from .archive.nodes import NodeStore
from .config import run_config_path
from .tools.data_tools import scores_by_clip
from .train.recipe import RECIPE_RULES, TUNABLE_KEYS

FORMAT_RULES = """\
Standard data formats (the only ones accepted), from WorldModel docs/TRAINING.md:
- Layout per dataset root: videos/<id>.mp4, captions/<id>.json, poses/<id>.npz (camera formats only).
- video_caption_camera: video + caption + per-frame camera poses.
- video_timed_prompts_camera: as above + caption "segments": [{"time_range_s": [start, end), "prompt"}].
  per_chunk mode: every boundary inside the clip on a round boundary 25/24 + k*32/24 s (within half a frame).
  segment mode: segments shorter than 2.375 s are never trained.
- video_caption_static: video + caption; truly fixed camera; no poses (identity is used).
- Training window: 57 frames at 24 fps (25 history + 32 target). Rollout rounds are 32 frames.
- Video: .mp4, fps >= 24 (higher is subsampled), duration >= 2.375 s, DISPLAY aspect within 2% of 16:9,
  no display rotation (re-encode with rotation applied). Frames are resized, never cropped.
- Caption: non-empty "caption" string.
- Poses: cam_c2w [N,4,4] with N = the mp4's frame count, camera-to-world, OpenCV convention, finite,
  bottom row [0,0,0,1], orthonormal rotation with det +1. Optional intrinsics [3,3] or [N,3,3] in pixels.
- Each enabled dataset needs at least as many clips as training GPUs.
- Clips are immutable: a re-captioned, cropped or trimmed clip is a new clip with derived_from set.
"""


def _node_file(run_dir: Path, node_id: str, rel: str) -> Path:
    return Path(run_dir) / "nodes" / node_id / rel


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _data_stats(conn, commit_id: str | None) -> dict:
    if not commit_id:
        return {}
    row = conn.execute("SELECT manifest FROM data_commits WHERE commit_id=?", (commit_id,)).fetchone()
    if row is None:
        return {}
    return {name: {"format": d["format"], "prompt_mode": d["prompt_mode"], "weight": d["weight"],
                   "clips": len(d["clips"])} for name, d in json.loads(row["manifest"])["datasets"].items()}


def lineage(conn, run_dir: Path, repo, node_id: str) -> list[dict]:
    nodes = NodeStore(conn)
    chain = []
    current = node_id
    while current:
        chain.append(nodes.get(current))
        current = chain[-1]["parent_id"]
    chain.reverse()
    out = []
    for i, node in enumerate(chain):
        recipe_path = _node_file(run_dir, node["node_id"], "recipe.yaml")
        rationale = _node_file(run_dir, node["node_id"], "rationale.md")
        parent_commit = chain[i - 1]["agent_commit"] if i else None
        out.append({
            "node_id": node["node_id"], "status": node["status"], "score": node["score"],
            "metrics": node["metrics"], "data": _data_stats(conn, node["data_commit"]),
            "recipe": yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else None,
            "rationale": rationale.read_text() if rationale.exists() else None,
            "edit": _read_json(_node_file(run_dir, node["node_id"], "edit.json")),
            "aggregates": _read_json(_node_file(run_dir, node["node_id"], "eval/aggregates.json")),
            "code_diff_stats": (repo.diff_stats(parent_commit, node["agent_commit"])
                                if parent_commit and node["agent_commit"] else []),
        })
    return out


def archive_summary(conn) -> dict:
    nodes = NodeStore(conn).all()
    scored = [n for n in nodes if n["status"] == "scored" and n["score"] is not None]
    best = max(scored, key=lambda n: n["score"], default=None)
    return {"nodes": [{"node_id": n["node_id"], "parent_id": n["parent_id"], "status": n["status"],
                       "score": n["score"], "subtree_value": n["subtree_value"], "depth": n["depth"]}
                      for n in nodes],
            "n_scored": len(scored),
            "best": {"node_id": best["node_id"], "score": best["score"]} if best else None}


def clip_pool_summary(conn, cap: int = 2000) -> list[dict]:
    usage = scores_by_clip(conn)
    clips = ClipStore(conn).all()[-cap:]
    return [{"clip_id": c["clip_id"], "formats": c["formats"], "camera_motion": c["camera_motion"],
             "metadata": c["metadata"], "provenance": c["provenance"], "license": c.get("license"),
             "ingested_by": c.get("ingested_by"), "used_by_scores": usage.get(c["clip_id"], [])}
            for c in clips]


def build_edit_context(*, conn, run_dir: Path, repo, parent_id: str, attempt: int, max_attempts: int,
                       retry: dict | None, nodes_remaining: int, dry_run: bool = False) -> EditContext:
    return EditContext(lineage=lineage(conn, run_dir, repo, parent_id), archive=archive_summary(conn),
                       nodes_remaining=nodes_remaining, attempt=attempt, max_attempts=max_attempts,
                       retry=retry, dry_run=dry_run)


def build_recipe_context(*, cfg, conn, run_dir: Path, repo, node_id: str, parent_id: str, attempt: int,
                         max_attempts: int, retry: dict | None, nodes_remaining: int, n_gpus: int,
                         tools: list[str], dry_run: bool = False) -> RecipeContext:
    parent = NodeStore(conn).get(parent_id)
    recipe_path = _node_file(run_dir, parent_id, "recipe.yaml")
    base = yaml.safe_load(run_config_path(cfg, run_dir, "base_recipe.yaml").read_text())
    return RecipeContext(
        lineage=lineage(conn, run_dir, repo, parent_id), archive=archive_summary(conn),
        nodes_remaining=nodes_remaining, attempt=attempt, max_attempts=max_attempts, retry=retry,
        dry_run=dry_run, clip_pool=clip_pool_summary(conn),
        parent_data_commit=parent["data_commit"],
        parent_recipe=yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else {},
        base_recipe=base,
        tunable_rules={k: {"type": RECIPE_RULES[k][0], "min": RECIPE_RULES[k][1], "max": RECIPE_RULES[k][2]}
                       for k in sorted(TUNABLE_KEYS)},
        resolution_allowlist=[list(p) for p in cfg.get("train.resolution_allowlist")],
        lora_allowlist=[list(p) for p in cfg.get("train.lora_allowlist")],
        format_rules=FORMAT_RULES, n_gpus=n_gpus, tools=list(tools))


def write_bundle(ctx, dest: Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "context.json"
    path.write_text(ctx.model_dump_json(indent=1))
    if ctx.retry:
        (dest / "retry.json").write_text(json.dumps(ctx.retry, indent=1))
    return path
```

`context_bundle` imports `ar_contract`, which Task 1 made an installed package, so the kernel and the container validate against the same schema objects.

- [ ] **Step 4: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_context_bundle.py -p no:cacheprovider`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/context_bundle.py tests/test_context_bundle.py
git commit -m "feat(kernel): context bundles for edit_self and improve_recipe from the archive"
```

---
### Task 14: Contract verification

Runs after every self-edit, in fresh containers from the child commit (spec 9.4): build, static, import, smoke run of both entry points with `dry_run=True` against a **mock** gateway and **mock** tools, then a structured report that becomes the retry's `/context/retry.json`. The mock tool server reuses the real `register_*` functions with canned tool objects, so its tool names and schemas cannot drift from the real ones.

**Files:**
- Create: `kernel/ar_kernel/contract/__init__.py` (empty), `kernel/ar_kernel/contract/verify.py`
- Test: `tests/test_contract_verify.py`, fixture agents under `tests/fixtures/agents/`

**Interfaces:**
- Consumes: `ensure_image`, `ImageBuildError` (Task 10); `run_container`, `Mounts`, `container_name` (Task 11); `AgentsRepo.checkout` (Task 12); `RunServices`, `socket_dir_for` (Task 6); `create_gateway_app`, `MockBook` (Task 5); `CallStore`, `TokenRegistry` (Task 4); `new_mcp`, `build_tool_app`, `ToolKit`, `ToolError` (Task 6); `register_data_tools`, `register_hf_tools`, `register_job_tools`, `JobQueue` (Tasks 7–9); `EditContext`, `RecipeContext` (Task 3).
- Produces:
  - `contract.verify.ContractStep(name, ok, detail="")` and `contract.verify.ContractReport(ok, steps, image=None)` with `.to_retry() -> dict`.
  - `contract.verify.static_check(entry_source: str) -> ContractStep`.
  - `contract.verify.ContractHarness(cfg, run_dir, recorder)`: `.start()`, `.stop()`, `.socket_dir`, `.registry`. It holds the mock gateway (upstream `None`, script `smoke`, model `mock-model` allowed) and the mock tools on their own socket directory.
  - `contract.verify.verify_contract(*, cfg, run_dir, run_id, repo, commit, harness, recorder, node, attempt, import_timeout_s=None, smoke_timeout_s=None, runner=run_container) -> ContractReport`.
- Timeouts: import `timeouts.contract_import_s` (60 s); smoke run hard cap = 4 x `timeouts.contract_smoke_s` (spec 14.5). Tests pass small overrides.
- *As built (review fix):* agent-controlled inputs never escape `verify_contract` as kernel exceptions: a non-UTF-8 `entry.py` or an `ast.parse` `RecursionError` fails the static step, and a malformed or non-object `result.json` fails the smoke step (with the result's traceback tail in the detail). Sandbox events use the harness's recorder, so the container token is redacted from them.

- [ ] **Step 1: Create the fixture agents**

Each is a directory holding `agent/__init__.py` (empty), `agent/entry.py` and `agent/requirements.txt` (empty unless stated).

`tests/fixtures/agents/good/agent/entry.py`:
```python
from ar_contract.models import EditResult, RecipeResult

def edit_self(ctx):
    return EditResult(summary="nothing to change")

def improve_recipe(ctx):
    return RecipeResult(data_commit=ctx.parent_data_commit or "dry-run", recipe={}, rationale="dry run")
```

`tests/fixtures/agents/no_improve/agent/entry.py`:
```python
def edit_self(ctx):
    return {"summary": "only one entry point"}
```

`tests/fixtures/agents/two_params/agent/entry.py`:
```python
def edit_self(ctx, extra):
    return {"summary": "x"}

def improve_recipe(ctx):
    return {"data_commit": "c", "recipe": {}, "rationale": "r"}
```

`tests/fixtures/agents/import_error/agent/entry.py`:
```python
import module_that_does_not_exist  # noqa: F401

def edit_self(ctx):
    return {"summary": "x"}

def improve_recipe(ctx):
    return {"data_commit": "c", "recipe": {}, "rationale": "r"}
```

`tests/fixtures/agents/hangs/agent/entry.py`:
```python
import time

def edit_self(ctx):
    time.sleep(3600)

def improve_recipe(ctx):
    time.sleep(3600)
```

`tests/fixtures/agents/invalid_result/agent/entry.py`:
```python
def edit_self(ctx):
    return {"summary": ""}

def improve_recipe(ctx):
    return {"data_commit": "", "recipe": {}, "rationale": ""}
```

`tests/fixtures/agents/bad_requirements/agent/entry.py`: the same as `good`, with `agent/requirements.txt` containing `definitely-not-a-real-package-ar==0.0.0`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_contract_verify.py
import json
from pathlib import Path

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.contract.verify import ContractHarness, static_check, verify_contract
from ar_kernel.sandbox.runner import RunResult
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()
FIXTURES = Path(__file__).parent / "fixtures" / "agents"


@pytest.mark.parametrize("src,ok,fragment", [
    ("def edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n", True, ""),
    ("async def edit_self(ctx): ...\nasync def improve_recipe(ctx): ...\n", True, ""),
    ("def edit_self(ctx): ...\n", False, "improve_recipe"),
    ("def edit_self(ctx, extra): ...\ndef improve_recipe(ctx): ...\n", False, "exactly one"),
    ("def edit_self(*args): ...\ndef improve_recipe(ctx): ...\n", False, "exactly one"),
    ("def outer():\n    def edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n", False, "edit_self"),
    ("def edit_self(ctx):\n  return (\n", False, "syntax"),
])
def test_static_check(src, ok, fragment):
    step = static_check(src)
    assert step.ok is ok and fragment in step.detail


def test_each_container_token_is_revoked_and_its_jobs_cancelled(tmp_path, monkeypatch):
    """Like the phase runner: a smoke run must not leave GPU jobs behind on the harness queue.
    No Docker: the image build and the container are faked."""
    run = tmp_path / "run"
    h = ContractHarness(CFG, run, Recorder(run))
    cancelled, tokens = [], []
    monkeypatch.setattr(h.queue, "cancel_for_token", lambda t: cancelled.append(t) or 0)
    monkeypatch.setattr("ar_kernel.contract.verify.ensure_image", lambda cfg, reqs, **k: "img:test")

    def runner(*, mounts, env, **kw):
        tokens.append(env["AR_TOKEN"])
        (mounts.workspace / "result.json").write_text(json.dumps({"ok": True, "result": {}}))
        return RunResult(0, False, "", "", 0.1, [], "c")

    repo = AgentsRepo(tmp_path / "agents.git")
    try:
        report = verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo,
                                 commit=repo.init(FIXTURES / "good"), harness=h, recorder=Recorder(run),
                                 node="n1", attempt=1, runner=runner)
    finally:
        h.queue.shutdown()
    assert report.ok, report.steps
    assert len(tokens) == 3 and cancelled == tokens          # import + two smoke runs
    assert all(h.registry.lookup(t) is None for t in tokens)


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    run = tmp_path_factory.mktemp("run")
    h = ContractHarness(CFG, run, Recorder(run))
    h.start()
    yield h, run
    h.stop()


def _verify(harness, tmp_path, fixture, **timeouts):
    h, run = harness
    repo = AgentsRepo(tmp_path / "agents.git")
    commit = repo.init(FIXTURES / fixture)
    return verify_contract(cfg=CFG, run_dir=run, run_id="t", repo=repo, commit=commit, harness=h,
                           recorder=Recorder(run), node="n1", attempt=1, **timeouts)


def _failed_step(report):
    return next((s for s in report.steps if not s.ok), None)


@pytest.mark.docker
def test_good_agent_passes_every_step(harness, tmp_path):
    report = _verify(harness, tmp_path, "good")
    assert report.ok, report.steps
    assert [s.name for s in report.steps] == ["build", "static", "import", "smoke:edit_self",
                                              "smoke:improve_recipe"]


@pytest.mark.docker
@pytest.mark.parametrize("fixture,step", [
    ("bad_requirements", "build"), ("no_improve", "static"), ("two_params", "static"),
    ("import_error", "import"), ("invalid_result", "smoke:edit_self"),
])
def test_broken_agents_fail_at_the_right_step(harness, tmp_path, fixture, step):
    report = _verify(harness, tmp_path, fixture)
    assert not report.ok and _failed_step(report).name == step
    assert report.to_retry()["failed_step"] == step


@pytest.mark.docker
def test_hanging_smoke_run_times_out(harness, tmp_path):
    report = _verify(harness, tmp_path, "hangs", smoke_timeout_s=15)
    failed = _failed_step(report)
    assert failed.name == "smoke:edit_self" and "timed out" in failed.detail
```

- [ ] **Step 3: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_contract_verify.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Implement**

```python
# kernel/ar_kernel/contract/verify.py
"""Contract verification of a child's code (spec 9.4)."""
from __future__ import annotations

import ast
import json
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ar_contract.models import EditContext, RecipeContext

from ..gateway.app import create_gateway_app
from ..gateway.mock import MockBook
from ..gateway.store import CallStore
from ..sandbox.image import ImageBuildError, ensure_image
from ..sandbox.runner import Mounts, container_name, run_container
from ..services import RunServices, socket_dir_for
from ..tools.context import TokenRegistry
from ..tools.data_tools import register_data_tools
from ..tools.hf_tools import register_hf_tools
from ..tools.jobs import JobQueue, register_job_tools
from ..tools.server import ToolError, ToolKit, build_tool_app, new_mcp

MOCK_MODEL = "mock-model"
ENTRY_POINTS = ("edit_self", "improve_recipe")


@dataclass
class ContractStep:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ContractReport:
    ok: bool
    steps: list[ContractStep] = field(default_factory=list)
    image: str | None = None

    def to_retry(self) -> dict:
        failed = next((s for s in self.steps if not s.ok), None)
        return {"kind": "contract", "ok": self.ok, "failed_step": failed.name if failed else None,
                "detail": failed.detail if failed else "", "steps": [asdict(s) for s in self.steps]}


def static_check(entry_source: str) -> ContractStep:
    try:
        tree = ast.parse(entry_source)
    except SyntaxError as exc:
        return ContractStep("static", False, f"syntax error in agent/entry.py: {exc}")
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in ENTRY_POINTS:
        fn = top.get(name)
        if fn is None:
            return ContractStep("static", False, f"agent/entry.py defines no top-level {name}")
        a = fn.args
        positional = a.posonlyargs + a.args
        if len(positional) != 1 or a.vararg or a.kwarg or a.kwonlyargs:
            return ContractStep("static", False,
                                f"{name} must take exactly one positional parameter (ctx)")
    return ContractStep("static", True)


class _MockData:
    """Canned data tools for smoke runs; same method names as DataTools."""
    def probe(self, c, path):
        return {"frames": 96, "fps": 24.0, "width": 736, "height": 414, "duration": 4.0,
                "rotation": 0, "sar": 1.0, "display_aspect": 736 / 414}
    def ingest(self, c, candidates):
        return [{"accepted": False, "clip_id": None, "formats": [], "warnings": [],
                 "reasons": ["mock tool server: smoke run"]} for _ in candidates]
    def query(self, c, filter):
        return {"total": 0, "returned": 0, "clips": []}
    def commit(self, c, parent, datasets, message):
        return {"commit_id": "0" * 64, "datasets": {}}
    def recipe_check(self, c, recipe, data_commit):
        return {"ok": True, "failures": []}


class _MockHf:
    def search(self, c, query, kind="dataset", limit=20):
        return []
    def download(self, c, repo, revision, patterns, max_bytes=None):
        raise ToolError("mock tool server: downloads are disabled during a smoke run")


class ContractHarness:
    def __init__(self, cfg, run_dir: Path, recorder) -> None:
        self.cfg, self.recorder = cfg, recorder
        self.registry = TokenRegistry(recorder)
        self.services = RunServices(socket_dir_for(Path(run_dir) / "contract-harness"))
        self.queue = JobQueue(recorder, threading.Lock(), wait_cap_s=5.0)

    @property
    def socket_dir(self) -> Path:
        return self.services.socket_dir

    def start(self) -> None:
        kit, mcp = ToolKit(self.registry, self.recorder), new_mcp()
        register_data_tools(mcp, kit, _MockData())
        register_hf_tools(mcp, kit, _MockHf())
        register_job_tools(mcp, kit, self.queue)
        gateway = create_gateway_app(registry=self.registry, store=CallStore(self.recorder),
                                     allowed_models={MOCK_MODEL}, upstream=None,
                                     mocks=MockBook.default())
        self.services.start(gateway, build_tool_app(mcp))

    def stop(self) -> None:
        self.services.stop()
        self.queue.shutdown()


def _smoke_context(kind: str) -> dict:
    common = {"nodes_remaining": 1, "attempt": 1, "max_attempts": 1, "dry_run": True}
    model = EditContext if kind == "edit_self" else RecipeContext
    return model(**common).model_dump()


def verify_contract(*, cfg, run_dir: Path, run_id: str, repo, commit: str, harness: ContractHarness,
                    recorder, node: str, attempt: int, import_timeout_s: float | None = None,
                    smoke_timeout_s: float | None = None, runner=run_container) -> ContractReport:
    import_timeout_s = import_timeout_s or float(cfg.get("timeouts.contract_import_s"))
    smoke_timeout_s = smoke_timeout_s or 4 * float(cfg.get("timeouts.contract_smoke_s"))
    report = ContractReport(ok=False)
    work = Path(run_dir) / "nodes" / node / "contract" / f"attempt-{attempt}"
    shutil.rmtree(work, ignore_errors=True)
    code = work / "agent"
    repo.checkout(commit, code)

    def finish(ok: bool) -> ContractReport:
        report.ok = ok
        recorder.event("contract.report", node=node, phase="contract", attempt=attempt,
                       component="contract", ok=ok, payload=report.to_retry())
        return report

    reqs_path = code / "agent" / "requirements.txt"
    try:
        report.image = ensure_image(cfg, reqs_path.read_text() if reqs_path.exists() else "",
                                    recorder=recorder, node=node)
        report.steps.append(ContractStep("build", True))
    except ImageBuildError as exc:
        report.steps.append(ContractStep("build", False, str(exc)))
        return finish(False)

    entry = code / "agent" / "entry.py"
    step = static_check(entry.read_text() if entry.exists() else "")
    report.steps.append(step)
    if not step.ok:
        return finish(False)

    def run(label: str, command: list[str], timeout_s: float, context: dict | None) -> tuple:
        run_dir_l = work / label.replace(":", "_")
        ws, ctx_dir = run_dir_l / "workspace", run_dir_l / "context"
        staging = Path(run_dir) / "staging" / node / f"contract-{attempt}-{label.replace(':', '_')}"
        for d in (ws, ctx_dir, staging):
            d.mkdir(parents=True, exist_ok=True)
        if context is not None:
            (ctx_dir / "context.json").write_text(json.dumps(context))
        caller = harness.registry.issue(node=node, phase="contract", attempt=attempt,
                                        workspace_host=ws, staging_host=staging, mock_script="smoke")
        try:
            result = runner(image=report.image, name=container_name(run_id, node, "contract", attempt),
                            mounts=Mounts(agent=code, workspace=ws, staging=staging, context=ctx_dir,
                                          store=Path(run_dir) / "store", contract=cfg.repo_root / "contract",
                                          sockets=harness.socket_dir, agent_readonly=True),
                            command=command, env={"AR_TOKEN": caller.token, "AR_DEFAULT_MODEL": MOCK_MODEL,
                                                  "AR_CONTEXT_WINDOW": str(cfg.get("agents.context_window_tokens")),
                                                  "AR_COMPACT_AT": str(cfg.get("agents.compact_at"))},
                            cpus=4, memory_gb=8, timeout_s=timeout_s, recorder=recorder,
                            node=node, phase="contract", attempt=attempt)
        finally:
            harness.registry.revoke(caller.token)
            harness.queue.cancel_for_token(caller.token)   # as in the phase runner (Task 15)
        return result, ws

    (Path(run_dir) / "store").mkdir(exist_ok=True)
    probe = ("import sys; sys.path.insert(0, '/agent'); import agent.entry as e; "
             "assert callable(e.edit_self) and callable(e.improve_recipe)")
    result, _ = run("import", ["python", "-c", probe], import_timeout_s, None)
    if result.timed_out or result.exit_code != 0:
        why = f"timed out after {import_timeout_s:.0f}s" if result.timed_out else result.stderr[-3000:]
        report.steps.append(ContractStep("import", False, why))
        return finish(False)
    report.steps.append(ContractStep("import", True))

    for kind in ENTRY_POINTS:
        result, ws = run(f"smoke:{kind}", ["python", "-m", "ar_contract.run", kind], smoke_timeout_s,
                         _smoke_context(kind))
        name = f"smoke:{kind}"
        if result.timed_out:
            report.steps.append(ContractStep(name, False, f"timed out after {smoke_timeout_s:.0f}s"))
            return finish(False)
        out = ws / "result.json"
        body = json.loads(out.read_text()) if out.exists() else {"ok": False, "error": "no result.json"}
        if result.exit_code != 0 or not body.get("ok"):
            report.steps.append(ContractStep(name, False, f"{body.get('error')}\n{result.stderr[-2000:]}"))
            return finish(False)
        report.steps.append(ContractStep(name, True))
    return finish(True)
```

- [ ] **Step 5: Run the tests (unit, then docker)**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_contract_verify.py -p no:cacheprovider`
Expected: PASS (13 as built: the 7 static-check cases, the token and job cleanup test, and 5 added in review: token redaction in telemetry and agent-controlled inputs, namely a non-UTF-8 `entry.py`, a recursion-depth `ast.parse` failure and malformed `result.json`).

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_contract_verify.py -m docker -p no:cacheprovider`
Expected: PASS (7 docker cases: good, 5 broken, hang).

- [ ] **Step 6: Commit**

```bash
git add kernel/ar_kernel/contract tests/test_contract_verify.py tests/fixtures/agents
git commit -m "feat(contract): build/static/import/smoke verification against a mock gateway and tools"
```

---

### Task 15: The agent phase runner (Plan 4's entry points)

`run_edit_self` and `run_improve_recipe`: build the attempt's directories and context, run the container against the **real** gateway and tools, then always revoke the token, cancel the caller's GPU jobs, collect the result, and record everything. `edit_self` commits the edited code to an attempt ref whether or not the attempt succeeded (spec 5.2). Plan 4 only sequences these calls and decides retries.

Attempt layout (Plan 4 relies on it):
```
runs/<run>/nodes/<n>/attempts/<phase>-<k>/{agent/, workspace/, context/}
runs/<run>/staging/<n>/<phase>-<k>/            (mounted at /workspace/staging)
```

**Files:**
- Create: `kernel/ar_kernel/agent_phase.py`
- Test: `tests/test_agent_phase.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `agent_phase.PhaseEnv` (dataclass): `cfg, run_dir, run_id, recorder, registry, queue, repo, socket_dir, gpus, default_model, runner=run_container`.
  - `agent_phase.PhaseOutcome` (dataclass): `ok: bool, result: dict | None, error: str | None, exit_code: int | None, timed_out: bool, commit: str | None, attempt_dir: Path, duration_s: float`.
  - `agent_phase.attempt_dirs(run_dir, node, phase, attempt) -> dict[str, Path]` with keys `attempt, agent, workspace, context, staging`.
  - `agent_phase.run_edit_self(env, *, conn, node, parent_id, base_commit, attempt, max_attempts, retry, nodes_remaining, previous_workspace=None, mock_script=None, dry_run=False) -> PhaseOutcome`. `base_commit` is the parent's code on attempt 1 and the failed attempt's commit on a retry (spec 7.2). *As built:* `previous_workspace` was added so a contract retry keeps `/workspace/edit_plan.json` (Task 17) and stays with the same component.
  - `agent_phase.run_improve_recipe(env, *, conn, node, parent_id, agent_commit, attempt, max_attempts, retry, nodes_remaining, previous_workspace=None, mock_script=None, dry_run=False) -> PhaseOutcome`. `previous_workspace` is copied in on a retry (spec 7.2).
- `improve_recipe` mounts `/agent` read-only: only `edit_self` changes code (a spec §9.5 amendment). A `RecipeResult` naming a data commit that does not exist is `ok=False`.
- *As built (differences from the code below, all from the pre-flight rulings and review):*
  - `result.json` is agent-controlled, so the kernel never trusts it: `_read_result` returns a failed body for a symlink, directory, non-UTF-8, non-JSON or non-object file; `_validate_result` requires `ok` to be literally `true` and re-validates `result` with the phase's pydantic model (`EditResult` / `RecipeResult`); anything else is a failed attempt with a reason, never a kernel exception. `error` is coerced to a string.
  - `_run` removes a stale `workspace/result.json` (unlinking a symlink without following it) before the container starts, so a retry's copied workspace cannot supply the previous attempt's result.
  - On a retry, the previous attempt's staging directory (`staging/<n>/<prev phase>-<k>`) is **moved**, not copied, into the new attempt's staging (downloads can be 20 GiB); re-running the same attempt number first clears its old staging.
  - An `ImageBuildError` from an agent-authored `requirements.txt` becomes a failed `PhaseOutcome` (and `edit_self` still commits the attempt), not an exception for Plan 4.
  - The `try` opens right after the token is issued, and a nested `try/finally` makes `queue.cancel_for_token` run even if `registry.revoke` raises.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_phase.py
import json
import threading

import pytest

from ar_kernel.agent_phase import PhaseEnv, attempt_dirs, run_edit_self, run_improve_recipe
from ar_kernel.archive.db import open_db
from ar_kernel.archive.nodes import NodeStore
from ar_kernel.config import KernelConfig
from ar_kernel.sandbox.runner import RunResult
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.jobs import JobQueue
from ar_kernel.vcs.agents_repo import AgentsRepo

CFG = KernelConfig.load()


class FakeRunner:
    """Stands in for the container: edits /agent and writes /workspace/result.json."""
    def __init__(self, result, edit=None, exit_code=0, timed_out=False, staged=None):
        self.result, self.edit, self.exit_code, self.timed_out = result, edit, exit_code, timed_out
        self.staged = staged
        self.calls = []

    def __call__(self, *, mounts, env, **kw):
        self.calls.append({"mounts": mounts, "env": env, **kw})
        if self.edit:
            (mounts.agent / "agent" / "entry.py").write_text(self.edit)
        if self.staged:                         # what hf_download would leave in /workspace/staging
            (mounts.staging / self.staged).parent.mkdir(parents=True, exist_ok=True)
            (mounts.staging / self.staged).write_text("x")
        if self.result is not None:
            (mounts.workspace / "result.json").write_text(json.dumps(self.result))
        return RunResult(self.exit_code, self.timed_out, "", "", 1.0, [], "c")


@pytest.fixture
def env(tmp_path, monkeypatch):
    seed = tmp_path / "seed" / "agent"
    seed.mkdir(parents=True)
    (seed / "entry.py").write_text("def edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n")
    repo = AgentsRepo(tmp_path / "run" / "agents.git")
    root = repo.init(tmp_path / "seed")
    rec = Recorder(tmp_path / "run")
    conn = open_db(tmp_path / "run")
    nodes = NodeStore(conn)
    nodes.create("root", None, 0)
    nodes.set_fields("root", agent_commit=root)
    nodes.create("n1", "root", 1)
    queue = JobQueue(rec, threading.Lock(), wait_cap_s=1)
    monkeypatch.setattr("ar_kernel.agent_phase.ensure_image", lambda cfg, reqs, **k: "img:test")

    def make(runner):
        return PhaseEnv(cfg=CFG, run_dir=tmp_path / "run", run_id="t", recorder=rec,
                        registry=TokenRegistry(rec), queue=queue, repo=repo,
                        socket_dir=tmp_path / "sock", gpus=[0, 1, 2, 3], default_model="gpt-x",
                        runner=runner)
    yield make, conn, root, rec, queue
    queue.shutdown()


def test_attempt_layout(tmp_path):
    d = attempt_dirs(tmp_path, "n1", "edit_self", 2)
    assert d["agent"] == tmp_path / "nodes" / "n1" / "attempts" / "edit_self-2" / "agent"
    assert d["staging"] == tmp_path / "staging" / "n1" / "edit_self-2"


def test_edit_self_commits_the_edited_code_to_an_attempt_ref(env):
    make, conn, root, rec, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "tightened prompts"}},
                        edit="# v2\ndef edit_self(ctx): ...\ndef improve_recipe(ctx): ...\n")
    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                        max_attempts=3, retry=None, nodes_remaining=5)
    assert out.ok and out.result == {"summary": "tightened prompts"}
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit
    assert "# v2" in penv.repo.read_file(out.commit, "agent/entry.py")
    assert runner.calls[0]["mounts"].agent_readonly is False


def test_failed_edit_attempt_is_still_committed(env):
    make, conn, root, _, _ = env
    runner = FakeRunner({"ok": False, "error": "RuntimeError: boom"}, edit="broken(\n", exit_code=1)
    penv = make(runner)
    out = run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                        max_attempts=3, retry=None, nodes_remaining=5)
    assert not out.ok and "boom" in out.error
    assert penv.repo.resolve("refs/attempts/n1/edit_self-1") == out.commit


def test_token_is_revoked_and_jobs_cancelled_after_the_phase(env, monkeypatch):
    make, conn, root, _, queue = env
    cancelled = []
    monkeypatch.setattr(queue, "cancel_for_token", lambda t: cancelled.append(t) or 0)
    runner = FakeRunner({"ok": True, "result": {"summary": "x"}})
    penv = make(runner)
    run_edit_self(penv, conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                  max_attempts=3, retry=None, nodes_remaining=5)
    token = runner.calls[0]["env"]["AR_TOKEN"]
    assert penv.registry.lookup(token) is None and cancelled == [token]


def test_staged_files_appear_in_the_recorded_diff(env):
    """/workspace/staging is a separate host directory (a nested mount), so the
    workspace snapshot alone would miss every staged download."""
    make, conn, root, rec, _ = env
    runner = FakeRunner({"ok": True, "result": {"summary": "x"}}, staged="hf/clip.mp4")
    run_edit_self(make(runner), conn=conn, node="n1", parent_id="root", base_commit=root, attempt=1,
                  max_attempts=3, retry=None, nodes_remaining=5)
    end = [e for e in rec.read_events("n1") if e["type"] == "phase.end"][0]
    assert rec.load_payload(end["payload"])["diffs"]["staging"]["added"] == ["hf/clip.mp4"]


def test_timeout_is_a_failed_attempt(env):
    make, conn, root, _, _ = env
    out = run_edit_self(make(FakeRunner(None, timed_out=True, exit_code=None)), conn=conn, node="n1",
                        parent_id="root", base_commit=root, attempt=1, max_attempts=3, retry=None,
                        nodes_remaining=5)
    assert not out.ok and out.timed_out and "timed out" in out.error


def test_improve_recipe_mounts_code_read_only_and_rejects_unknown_commits(env):
    make, conn, root, _, _ = env
    runner = FakeRunner({"ok": True, "result": {"data_commit": "f" * 64, "recipe": {}, "rationale": "r"}})
    out = run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root,
                             attempt=1, max_attempts=3, retry=None, nodes_remaining=5)
    assert runner.calls[0]["mounts"].agent_readonly is True
    assert not out.ok and "data commit" in out.error


def test_retry_continues_from_the_previous_workspace(env):
    make, conn, root, _, _ = env
    first = make(FakeRunner({"ok": False, "error": "x"}, exit_code=1))
    out1 = run_improve_recipe(first, conn=conn, node="n1", parent_id="root", agent_commit=root,
                              attempt=1, max_attempts=3, retry=None, nodes_remaining=5)
    (out1.attempt_dir / "workspace" / "notes.md").write_text("half-built dataset")
    runner = FakeRunner({"ok": False, "error": "y"}, exit_code=1)
    run_improve_recipe(make(runner), conn=conn, node="n1", parent_id="root", agent_commit=root,
                       attempt=2, max_attempts=3, retry={"failures": ["x"]}, nodes_remaining=5,
                       previous_workspace=out1.attempt_dir / "workspace")
    assert (runner.calls[0]["mounts"].workspace / "notes.md").read_text() == "half-built dataset"
    assert json.loads((runner.calls[0]["mounts"].context / "retry.json").read_text()) == {"failures": ["x"]}
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_agent_phase.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# kernel/ar_kernel/agent_phase.py
"""Run one agent phase attempt end to end (spec 7.2 steps 3 and 5)."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .archive.blobs import BlobStore
from .archive.clips import ClipStore
from .archive.commits import CommitStore
from .context_bundle import build_edit_context, build_recipe_context, write_bundle
from .sandbox.image import ensure_image
from .sandbox.runner import Mounts, container_name, diff, run_container, snapshot


@dataclass
class PhaseEnv:
    cfg: object
    run_dir: Path
    run_id: str
    recorder: object
    registry: object
    queue: object
    repo: object
    socket_dir: Path
    gpus: list[int]
    default_model: str
    runner: object = run_container


@dataclass
class PhaseOutcome:
    ok: bool
    result: dict | None
    error: str | None
    exit_code: int | None
    timed_out: bool
    commit: str | None
    attempt_dir: Path
    duration_s: float


def attempt_dirs(run_dir: Path, node: str, phase: str, attempt: int) -> dict[str, Path]:
    base = Path(run_dir) / "nodes" / node / "attempts" / f"{phase}-{attempt}"
    return {"attempt": base, "agent": base / "agent", "workspace": base / "workspace",
            "context": base / "context", "staging": Path(run_dir) / "staging" / node / f"{phase}-{attempt}"}


def _run(env: PhaseEnv, *, phase: str, node: str, attempt: int, code_commit: str, ctx,
         mock_script: str | None, agent_readonly: bool, previous_workspace: Path | None) -> tuple:
    dirs = attempt_dirs(env.run_dir, node, phase, attempt)
    if dirs["attempt"].exists():
        shutil.rmtree(dirs["attempt"])
    env.repo.checkout(code_commit, dirs["agent"])
    if previous_workspace is not None and Path(previous_workspace).exists():
        shutil.copytree(previous_workspace, dirs["workspace"], symlinks=True)
    for key in ("workspace", "context", "staging"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    (Path(env.run_dir) / "store").mkdir(exist_ok=True)
    write_bundle(ctx, dirs["context"])
    reqs = dirs["agent"] / "agent" / "requirements.txt"
    image = ensure_image(env.cfg, reqs.read_text() if reqs.exists() else "", recorder=env.recorder, node=node)
    caller = env.registry.issue(node=node, phase=phase, attempt=attempt, workspace_host=dirs["workspace"],
                                staging_host=dirs["staging"], mock_script=mock_script)
    soft = float(env.cfg.get(f"timeouts.{phase}_s"))
    # /workspace/staging is mounted from its own host directory, so it is snapshotted separately.
    before = {"agent": snapshot(dirs["agent"], True), "workspace": snapshot(dirs["workspace"], False),
              "staging": snapshot(dirs["staging"], False)}
    env.recorder.event("phase.start", node=node, phase=phase, attempt=attempt, component="kernel",
                       payload={"code_commit": code_commit, "image": image})
    started = time.monotonic()
    try:
        result = env.runner(
            image=image, name=container_name(env.run_id, node, phase, attempt),
            mounts=Mounts(agent=dirs["agent"], workspace=dirs["workspace"], staging=dirs["staging"],
                          context=dirs["context"], store=Path(env.run_dir) / "store",
                          contract=env.cfg.repo_root / "contract", sockets=env.socket_dir,
                          agent_readonly=agent_readonly),
            command=["python", "-m", "ar_contract.run", phase],
            env={"AR_TOKEN": caller.token, "AR_DEFAULT_MODEL": env.default_model, "AR_NODE": node,
                 "AR_PHASE": phase, "AR_ATTEMPT": str(attempt),
                 "AR_CONTEXT_WINDOW": str(env.cfg.get("agents.context_window_tokens")),
                 "AR_COMPACT_AT": str(env.cfg.get("agents.compact_at"))},
            cpus=env.cfg.get("sandbox.cpus"), memory_gb=env.cfg.get("sandbox.memory_gb"),
            timeout_s=4 * soft,                 # hard cap (spec 14.5); liveness is Plan 4
            recorder=env.recorder, node=node, phase=phase, attempt=attempt)
    finally:
        env.registry.revoke(caller.token)
        env.queue.cancel_for_token(caller.token)     # an ended phase must not keep the GPUs
    out_file = dirs["workspace"] / "result.json"
    body = json.loads(out_file.read_text()) if out_file.exists() else None
    diffs = {"agent": diff(before["agent"], snapshot(dirs["agent"], True)),
             "workspace": diff(before["workspace"], snapshot(dirs["workspace"], False)),
             "staging": diff(before["staging"], snapshot(dirs["staging"], False))}
    return dirs, result, body, diffs, time.monotonic() - started


def _outcome(env, phase, node, attempt, dirs, result, body, diffs, duration, commit, extra_error=None):
    if result.timed_out:
        ok, error = False, "the agent call timed out and was killed"
    elif body is None:
        ok, error = False, f"no result.json (exit code {result.exit_code}): {result.stderr[-2000:]}"
    elif not body.get("ok"):
        ok, error = False, body.get("error")
    else:
        ok, error = extra_error is None, extra_error
    outcome = PhaseOutcome(ok, body.get("result") if body and body.get("ok") else None, error,
                           result.exit_code, result.timed_out, commit, dirs["attempt"], duration)
    env.recorder.event("phase.end", node=node, phase=phase, attempt=attempt, component="kernel",
                       ok=ok, timed_out=result.timed_out, exit_code=result.exit_code,
                       payload={"result": outcome.result, "error": error, "diffs": diffs, "commit": commit})
    return outcome


def run_edit_self(env: PhaseEnv, *, conn, node: str, parent_id: str, base_commit: str, attempt: int,
                  max_attempts: int, retry: dict | None, nodes_remaining: int,
                  mock_script: str | None = None, dry_run: bool = False) -> PhaseOutcome:
    ctx = build_edit_context(conn=conn, run_dir=env.run_dir, repo=env.repo, parent_id=parent_id,
                             attempt=attempt, max_attempts=max_attempts, retry=retry,
                             nodes_remaining=nodes_remaining, dry_run=dry_run)
    dirs, result, body, diffs, duration = _run(env, phase="edit_self", node=node, attempt=attempt,
                                               code_commit=base_commit, ctx=ctx, mock_script=mock_script,
                                               agent_readonly=False, previous_workspace=None)
    commit = env.repo.commit_tree(dirs["agent"], base_commit, f"{node} edit_self attempt {attempt}")
    env.repo.set_ref(env.repo.attempt_ref(node, "edit_self", attempt), commit)
    return _outcome(env, "edit_self", node, attempt, dirs, result, body, diffs, duration, commit)


def run_improve_recipe(env: PhaseEnv, *, conn, node: str, parent_id: str, agent_commit: str, attempt: int,
                       max_attempts: int, retry: dict | None, nodes_remaining: int,
                       previous_workspace: Path | None = None, mock_script: str | None = None,
                       dry_run: bool = False) -> PhaseOutcome:
    tools = ["video_probe", "data_ingest", "data_query", "data_commit", "recipe_check",
             "hf_search", "hf_download", "job_status", "job_wait", "job_cancel",
             *sorted(b.tool for b in env.queue.backends.values())]
    ctx = build_recipe_context(cfg=env.cfg, conn=conn, run_dir=env.run_dir, repo=env.repo, node_id=node,
                               parent_id=parent_id, attempt=attempt, max_attempts=max_attempts,
                               retry=retry, nodes_remaining=nodes_remaining, n_gpus=len(env.gpus),
                               tools=tools, dry_run=dry_run)
    dirs, result, body, diffs, duration = _run(env, phase="improve_recipe", node=node, attempt=attempt,
                                               code_commit=agent_commit, ctx=ctx, mock_script=mock_script,
                                               agent_readonly=True, previous_workspace=previous_workspace)
    extra = None
    if body and body.get("ok"):
        commit_id = body["result"]["data_commit"]
        try:
            CommitStore(conn, BlobStore(env.run_dir, conn), ClipStore(conn)).get(commit_id)
        except KeyError:                      # CommitStore.get raises KeyError for unknown ids
            extra = f"result names data commit {commit_id!r}, which does not exist"
    return _outcome(env, "improve_recipe", node, attempt, dirs, result, body, diffs, duration, None, extra)
```

- [ ] **Step 4: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_agent_phase.py -p no:cacheprovider`
Expected: PASS (23 tests as built, including the review additions for untrusted `result.json`, image build failures and staging on retry).

- [ ] **Step 5: Add a docker end-to-end test: real container, real gateway (mock script), real tools**

Create the fixture agent `tests/fixtures/agents/tool_user/agent/entry.py`:

```python
"""Uses the real MCP tools from inside the container: queries the pool and commits it."""
import asyncio
import json

from ar_contract.client import mcp_session
from ar_contract.models import EditResult, RecipeResult


def edit_self(ctx):
    return EditResult(summary="unused")


async def improve_recipe(ctx):
    async with mcp_session() as tools:
        pool = json.loads((await tools.call_tool("data_query",
                                                  {"filter": {"format": "video_caption_camera"}})).content[0].text)
        ids = [c["clip_id"] for c in pool["clips"]][:4]
        made = json.loads((await tools.call_tool("data_commit", {
            "parent": None, "message": "from inside the sandbox",
            "datasets": {"cam": {"format": "video_caption_camera", "prompt_mode": None,
                                 "weight": 1.0, "clips": ids}}})).content[0].text)
    return RecipeResult(data_commit=made["commit_id"], recipe={}, rationale=f"{len(ids)} clips")
```

(with an empty `agent/__init__.py` and `agent/requirements.txt`). Append to `tests/test_agent_phase.py`:

```python
@pytest.mark.docker
def test_container_commits_data_through_the_real_tool_server(tmp_path):
    """No LLM: the fixture agent calls data_query and data_commit over the socket;
    the kernel creates a real commit and the phase validates it."""
    from pathlib import Path
    from ar_kernel.gateway.app import create_gateway_app
    from ar_kernel.gateway.mock import MockBook
    from ar_kernel.gateway.store import CallStore
    from ar_kernel.services import RunServices, socket_dir_for
    from ar_kernel.tools.data_tools import DataTools, register_data_tools
    from ar_kernel.tools.jobs import register_job_tools
    from ar_kernel.tools.server import ToolKit, build_tool_app, new_mcp
    from ar_kernel.tools.context import TokenRegistry
    from conftest import make_mp4, write_caption, write_poses
    from ar_kernel.data.ingest import Candidate, Ingestor

    run = tmp_path / "run"
    rec, conn = Recorder(run), open_db(run)
    for i in range(4):                                   # seed the pool with 4 distinct clips
        d = run / "staging" / "seed" / f"c{i}"
        secs = 4.0 + 0.5 * i
        make_mp4(d / "v.mp4", seconds=secs), write_caption(d / "c.json"), write_poses(d / "p.npz", int(secs * 30))
        Ingestor(CFG, run, conn, rec).ingest([Candidate(video=d / "v.mp4", caption=d / "c.json",
                                                       pose=d / "p.npz", camera_motion="moving",
                                                       provenance={"kind": "derived", "from": [], "transform": "t"})],
                                             node_id="root")
    repo = AgentsRepo(run / "agents.git")
    commit = repo.init(Path(__file__).parent / "fixtures" / "agents" / "tool_user")
    nodes = NodeStore(conn)
    nodes.create("root", None, 0), nodes.set_fields("root", agent_commit=commit), nodes.create("n1", "root", 1)
    reg, queue = TokenRegistry(rec), JobQueue(rec, threading.Lock(), wait_cap_s=5)
    kit, mcp = ToolKit(reg, rec), new_mcp()
    register_data_tools(mcp, kit, DataTools(CFG, run, rec, [0, 1, 2, 3], threading.Lock()))
    register_job_tools(mcp, kit, queue)
    services = RunServices(socket_dir_for(run))
    services.start(create_gateway_app(registry=reg, store=CallStore(rec), allowed_models={"mock-model"},
                                      upstream=None, mocks=MockBook.default()), build_tool_app(mcp))
    try:
        penv = PhaseEnv(cfg=CFG, run_dir=run, run_id="e2e", recorder=rec, registry=reg, queue=queue,
                        repo=repo, socket_dir=services.socket_dir, gpus=[0, 1, 2, 3],
                        default_model="mock-model")
        out = run_improve_recipe(penv, conn=conn, node="n1", parent_id="root", agent_commit=commit,
                                 attempt=1, max_attempts=3, retry=None, nodes_remaining=1)
    finally:
        services.stop(), queue.shutdown()
    assert out.ok, out.error
    assert out.result["rationale"] == "4 clips"
    kinds = [e["type"] for e in rec.read_events("n1")]
    assert "tool.call" in kinds and "phase.end" in kinds
```

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_agent_phase.py -m docker -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add kernel/ar_kernel/agent_phase.py tests/test_agent_phase.py tests/fixtures/agents/tool_user
git commit -m "feat(kernel): agent phase runner -- attempt dirs, real services, attempt commits"
```

---
### Task 16: The seed harness (single-agent inner loop)

The inner loop every seed role runs on. It is agent code (`seed_agent/agent/harness.py`), so `edit_self` can change it like any other component. It is an explicit LangGraph ReAct graph that reproduces `langchain.agents.create_agent` exactly (fact 14), built in three commits: first the **exact** reproduction, proven equal to `create_agent` by a comparison test and mutation controls; then the two deliberate additions, each test-first:

1. **Tool errors are reported, not raised.** `create_agent`'s default tool node re-raises any tool exception and ends the run (fact 15). The harness returns it to the model as an error `ToolMessage` using `ToolNode`'s `handle_tool_errors=True` text: `Error: <repr>\n Please fix your mistakes.`
2. **Auto-compact, Claude Code style.** Before every model call, inside the model node and on the merged state, the harness estimates the context size: the last reply's reported `usage.total_tokens` plus about 4 characters per token for the messages after it, with each image block counted as a fixed 1,500 tokens rather than by its base64 length (no tokenizer offline, fact 17). Below `compact_at × context_window` nothing happens and messages append linearly. At or above it, a dedicated summarizer call (same system prompt, tools and history; `tool_choice="none"`; the instruction in `prompts/compact.md`) condenses the history, and the history is replaced by one user message: a continuation preamble plus the summary. The real model call then runs on that message.

*As built (review ruling):* the first draft routed to a separate `compact` node through conditional edges from `START` and from `tools`. Under parallel tool calls that routing function runs once per `Send` branch, before the branches' results are merged: a reviewer reproduced a summarizer running beside a model call on the stale history, and a due compaction being skipped when only the combined tool outputs crossed the threshold. The check therefore lives in the model node, which sees the merged state, and the graph is exactly `create_agent`'s (START → model, model → tools per call, tools → model). Two tests pin the parallel cases.

The seed layout follows the simplicity ruling (File structure): the first draft's `harness/react.py` and `harness/compact.py` are one module, `harness.py`, imported as `agent.harness`.

The comparison test stays in the suite for good. `create_agent` is the reference whenever no tool raises and the context stays below the threshold.

**Files:**
- Create: `seed_agent/.gitignore`, `seed_agent/agent/__init__.py` (empty)
- Create: `seed_agent/agent/harness.py`, `seed_agent/agent/prompts/compact.md`
- Test: `tests/test_seed_harness.py`

**Interfaces:**
- Consumes: `langgraph` 1.2.11, `langchain-core` 1.6.3; `langchain` 1.4.2 in tests only (the reference).
- Produces (used by Task 17's roles):
  - `agent.harness.build_react_agent(model: BaseChatModel, tools: list[BaseTool], system_prompt: str | None = None, *, context_window: int, compact_at: float = 0.85, compact_prompt: str | None = None)` returns a compiled graph. Call it as `await graph.ainvoke({"messages": [...]})`; the result's `"messages"` is the full history. `compact_prompt=None` reads `agent/prompts/compact.md`.
  - `async agent.harness.run_tool(tools_by_name, call) -> ToolMessage`, plus the constants `RECURSION_LIMIT = 9999`, `COMPACT_AT = 0.85` and `COMPACT_PROMPT` (a `Path`).
  - `agent.harness.estimate_tokens(messages, system_prompt=None) -> int`, `needs_compaction(messages, system_prompt, context_window, compact_at) -> bool`, `CONTINUATION` (a format string with `{summary}`), and `IMAGE_TOKENS = 1500`. (The first draft's separate `summarize` helper was folded into the model node.)

- [ ] **Step 1: Scaffolding**

```gitignore
# seed_agent/.gitignore
__pycache__/
*.pyc
```

Create the empty `seed_agent/agent/__init__.py`.

- [ ] **Step 2: Write the comparison test (exact behaviour)**

The scripted model replays fixed `AIMessage`s and records every prompt it receives and every `bind_tools` call. Each scenario runs through our graph and through `create_agent`, and the test requires identical final message lists **and** identical prompts and tool bindings. The scenarios cover: a plain answer, one tool call, parallel calls that finish out of order, an unknown tool, invalid arguments, dict and list outputs, and a multi-step run. Each runs with and without a system prompt, plus a no-tools case and a raising tool.

```python
"""The seed harness (agent.harness) against langchain.agents.create_agent.

create_agent (langchain 1.4.2, factory.py @ 4af7ab8) is the reference: both must
produce identical message lists AND show the model identical prompts and tool bindings."""
import asyncio
import sys
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

SEED = Path(__file__).resolve().parents[1] / "seed_agent"
sys.path.insert(0, str(SEED))

from agent.harness import build_react_agent  # noqa: E402


class Scripted(BaseChatModel):
    """Replays AIMessages in order; records every prompt and every bind_tools call."""
    script: list
    log: dict

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.log["bind"].append(([convert_to_openai_tool(t) for t in tools], kwargs))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log["prompts"].append([_norm(m) for m in messages])
        reply = self.script[len(self.log["prompts"]) - 1].model_copy(deep=True)
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _norm(m: BaseMessage) -> dict:
    d = m.model_dump()
    d.pop("id", None)                    # message ids are random per run
    d.pop("response_metadata", None)     # carries the model run id
    return d


def _ai(text="", calls=(), total=None):
    m = AIMessage(content=text, tool_calls=[{"name": n, "args": a, "id": i, "type": "tool_call"}
                                            for n, a, i in calls])
    if total is not None:
        m.usage_metadata = {"input_tokens": total - 10, "output_tokens": 10, "total_tokens": total}
    return m


@tool
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
async def slow_echo(text: str, delay: float) -> str:
    """Echo text after a delay."""
    await asyncio.sleep(delay)
    return text


@tool
async def as_dict(key: str) -> dict:
    """Return a dict."""
    return {"key": key, "n": [1, 2]}


@tool
async def strings(n: int) -> list:
    """Return a list of strings (content the tool node re-serializes)."""
    return ["plain", f"n={n}"]


@tool
async def boom(x: int) -> str:
    """Always fails."""
    raise RuntimeError(f"boom {x}")


TOOLS = [add, slow_echo, as_dict, strings, boom]

SCENARIOS = {
    "plain_answer": [_ai("hello")],
    "one_call": [_ai("", [("add", {"a": 1, "b": 2}, "c1")]), _ai("3")],
    "parallel_calls_finish_out_of_order": [
        _ai("", [("slow_echo", {"text": "a", "delay": 0.3}, "c1"),
                 ("slow_echo", {"text": "b", "delay": 0.0}, "c2"),
                 ("add", {"a": 5, "b": 6}, "c3")]), _ai("done")],
    "unknown_tool": [_ai("", [("nope", {"q": 1}, "c1")]), _ai("sorry")],
    "bad_args": [_ai("", [("add", {"a": "x"}, "c1")]), _ai("fixed")],
    "dict_output": [_ai("", [("as_dict", {"key": "k"}, "c1")]), _ai("ok")],
    "list_output": [_ai("", [("strings", {"n": 3}, "c1")]), _ai("ok")],
    "multi_step": [_ai("", [("add", {"a": 1, "b": 1}, "c1")]),
                   _ai("thinking", [("add", {"a": 2, "b": 2}, "c2"), ("nope", {}, "c3")]),
                   _ai("", [("as_dict", {"key": "z"}, "c4")]), _ai("final")],
}


async def _run(builder, script, tools, system, history):
    log = {"bind": [], "prompts": []}
    agent = builder(Scripted(script=script, log=log), tools, system)
    out = await agent.ainvoke({"messages": history})
    return [_norm(m) for m in out["messages"]], log


def _ours(model, tools, system):
    return build_react_agent(model, tools, system)


def _theirs(model, tools, system):
    return create_agent(model, tools, system_prompt=system)


@pytest.mark.parametrize("system", [None, "You are careful."])
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_matches_create_agent(name, system):
    history = [HumanMessage("earlier"), AIMessage("earlier reply"), HumanMessage("go")]
    ours = asyncio.run(_run(_ours, SCENARIOS[name], TOOLS, system, history))
    theirs = asyncio.run(_run(_theirs, SCENARIOS[name], TOOLS, system, history))
    assert ours[0] == theirs[0]          # the final message list
    assert ours[1] == theirs[1]          # every prompt the model saw, and every tool binding


def test_no_tools_matches_create_agent():
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")])]    # a tool call with no tools ends the loop
    assert (asyncio.run(_run(_ours, script, [], "s", [HumanMessage("x")]))
            == asyncio.run(_run(_theirs, script, [], "s", [HumanMessage("x")])))


def test_a_raising_tool_propagates_in_both():
    script = [_ai("", [("boom", {"x": 1}, "c1")]), _ai("unreachable")]
    for builder in (_ours, _theirs):
        with pytest.raises(RuntimeError, match="boom 1"):
            asyncio.run(_run(builder, script, TOOLS, None, [HumanMessage("x")]))
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_harness.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError: No module named 'agent.harness'`).

- [ ] **Step 4: Implement the exact reproduction**

```python
# seed_agent/agent/harness.py
"""Single-agent inner loop: an explicit ReAct graph.

Reproduces langchain.agents.create_agent (langchain 1.4.2, factory.py @ 4af7ab8)
with no middleware, no response_format and async tools:
  START -> model; model -> END if the last AI message has no tool calls,
  else one Send("tools", [call]) per tool call (parallel); tools -> model.
Tool execution follows langgraph.prebuilt.ToolNode's default error handling.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from pydantic import ValidationError

RECURSION_LIMIT = 9_999
INVALID_TOOL = "Error: {name} is not a valid tool, try one of [{names}]."
INVALID_ARGS = ("Error invoking tool '{name}' with kwargs {args} with error:\n"
                " {error}\n Please fix the error and try again.")
TOOL_BLOCK_TYPES = {"text", "image_url", "image", "json", "search_result", "custom_tool_call_output",
                    "document", "file"}


class ReactState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _content(output: Any) -> str | list:
    if isinstance(output, str) or (isinstance(output, list) and all(
            isinstance(x, dict) and x.get("type") in TOOL_BLOCK_TYPES for x in output)):
        return output
    try:
        return json.dumps(output, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(output)


async def run_tool(tools: dict[str, BaseTool], call: ToolCall) -> ToolMessage:
    tool = tools.get(call["name"])
    if tool is None:
        return ToolMessage(INVALID_TOOL.format(name=call["name"], names=", ".join(tools)),
                           name=call["name"], tool_call_id=call["id"], status="error")
    try:
        message = await tool.ainvoke({**call, "type": "tool_call"})
    except ValidationError as exc:
        error = "\n".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', 'Unknown error')}"
                          for e in exc.errors())
        return ToolMessage(INVALID_ARGS.format(name=call["name"], args=call["args"], error=error),
                           name=call["name"], tool_call_id=call["id"], status="error")
    message.content = _content(message.content)
    return message


def build_react_agent(model: BaseChatModel, tools: list[BaseTool], system_prompt: str | None = None):
    by_name = {t.name: t for t in tools}
    system = [SystemMessage(content=system_prompt)] if system_prompt is not None else []

    async def call_model(state: ReactState) -> dict:
        bound = model.bind_tools(tools, tool_choice=None) if tools else model.bind()
        return {"messages": [await bound.ainvoke([*system, *state["messages"]])]}

    async def call_tool(calls: list[ToolCall]) -> dict:
        return {"messages": [await run_tool(by_name, call) for call in calls]}

    def after_model(state: ReactState):
        last = state["messages"][-1]
        if not tools or not isinstance(last, AIMessage) or not last.tool_calls:
            return END
        return [Send("tools", [call]) for call in last.tool_calls]

    graph = StateGraph(ReactState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    if tools:
        graph.add_node("tools", call_tool)
        graph.add_conditional_edges("model", after_model, ["tools", END])
        graph.add_edge("tools", "model")
    else:
        graph.add_edge("model", END)
    return graph.compile().with_config({"recursion_limit": RECURSION_LIMIT})
```

Two details are easy to get wrong, and the test catches both:
- `create_agent` re-binds the tools on every model call, so `bind_tools` goes inside the model node.
- The tool node re-serializes a list of plain strings as JSON (`_content`); `BaseTool` alone would pass it through.

- [ ] **Step 5: Run the test**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_harness.py -p no:cacheprovider`
Expected: PASS (18 tests).

- [ ] **Step 6: Mutation controls (the comparison test must be able to fail)**

A comparison that passes against a broken copy proves nothing (verification-log finding 9, "Run the negative control"). Apply each mutation to `harness.py`, run the test, record the failure count, and restore the file:

```bash
H=seed_agent/agent/harness.py; ORIG=$(mktemp); cp $H $ORIG
for expr in 's/try one of \[/try one of  [/' \
            's/return \[Send("tools", \[call\]) for call in last.tool_calls\]/return [Send("tools", list(reversed(last.tool_calls)))]/' \
            's/\[\*system, \*state\["messages"\]\]/[*state["messages"], *system]/' \
            's/message.content = _content(message.content)/pass/' \
            's/with error:\\n"/with error: "/'; do
  cp $ORIG $H; sed -i "$expr" $H
  cmp -s $H $ORIG && echo "MUTATION DID NOT APPLY: $expr" && continue
  conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_harness.py -p no:cacheprovider 2>&1 | grep -E '^[0-9]+ (passed|failed)|[0-9]+ failed'
done
cp $ORIG $H && rm $ORIG
```

Expected, in order: 4, 4, 9, 2 and 2 failed (the invalid-tool text; sequential calls in reverse; the system prompt after the history; no content normalisation; the invalid-argument text). The restored file passes all 18 again. Keep these counts for the Task 18 log.

*As built:* on the final `harness.py` and the 26-test file (after Steps 8–11), the model call reads a local `messages`, so the third mutation is spelled `'s/\[\*system, \*messages\]/[*messages, *system]/'`; the counts are then 4, 4, **10**, 2 and 2. The extra failure is `test_compaction_replaces_history_and_continues`, which checks the position of the continuation message in the post-compaction prompt.

- [ ] **Step 7: Commit**

```bash
git add seed_agent/.gitignore seed_agent/agent/__init__.py seed_agent/agent/harness.py tests/test_seed_harness.py
git commit -m "feat(seed): explicit ReAct harness reproducing create_agent, with a comparison test"
```

- [ ] **Step 8: Tool errors are reported (test first)**

In `tests/test_seed_harness.py`, replace `test_a_raising_tool_propagates_in_both` with:

```python
def test_tool_exception_is_reported_where_create_agent_raises():
    script = [_ai("", [("boom", {"x": 1}, "c1")]), _ai("recovered")]
    with pytest.raises(RuntimeError, match="boom 1"):
        asyncio.run(_run(_theirs, script, TOOLS, None, [HumanMessage("x")]))
    messages, _ = asyncio.run(_run(_ours, script, TOOLS, None, [HumanMessage("x")]))
    err = messages[2]
    assert err["type"] == "tool" and err["status"] == "error" and err["tool_call_id"] == "c1"
    assert err["content"] == "Error: RuntimeError('boom 1')\n Please fix your mistakes."
    assert messages[-1]["content"] == "recovered"
```

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_harness.py -p no:cacheprovider`
Expected: FAIL (1 test: `RuntimeError: boom 1` escapes our graph).

In `harness.py`, add the template after `INVALID_ARGS`:

```python
TOOL_ERROR = "Error: {error}\n Please fix your mistakes."   # ToolNode's handle_tool_errors=True text
```

and a second handler in `run_tool`, after the `ValidationError` handler:

```python
    except Exception as exc:  # noqa: BLE001 -- difference 1: report, do not crash
        return ToolMessage(TOOL_ERROR.format(error=repr(exc)), name=call["name"],
                           tool_call_id=call["id"], status="error")
```

Also update the module docstring to name the difference.

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_harness.py -p no:cacheprovider`
Expected: PASS (18 tests).

```bash
git add seed_agent/agent/harness.py tests/test_seed_harness.py
git commit -m "feat(seed): harness reports tool errors to the model instead of ending the run"
```

- [ ] **Step 9: Auto-compact (test first)**

In `tests/test_seed_harness.py`:
- Change the import line to `from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage`.
- Add `from agent.harness import CONTINUATION, estimate_tokens, needs_compaction  # noqa: E402` above the `build_react_agent` import.
- Add `BIG = 10 ** 9      # a context window compaction never reaches` below the imports.
- Make `_ours` build with `build_react_agent(model, tools, system, context_window=BIG)`.
- Append these tests:

```python
def test_estimate_uses_last_reported_usage_plus_tail():
    msgs = [HumanMessage("x" * 400), _ai("hi", total=1000), ToolMessage("y" * 80, tool_call_id="c")]
    assert estimate_tokens(msgs) == 1000 + 20
    assert estimate_tokens([HumanMessage("x" * 400)], system_prompt="s" * 40) == 110


def test_images_count_as_a_fixed_estimate_not_their_base64_length():
    from agent.harness import IMAGE_TOKENS
    image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "A" * 400_000}}
    msgs = [HumanMessage(content=[{"type": "text", "text": "x" * 40}, image])]
    assert IMAGE_TOKENS <= estimate_tokens(msgs) < IMAGE_TOKENS + 100      # not 400_000 / 4


def test_a_single_message_is_never_compacted():
    assert not needs_compaction([HumanMessage("x" * 10 ** 6)], None, 1000, 0.85)


def test_compaction_replaces_history_and_continues():
    # call 1 asks for a tool and reports 900 tokens >= 0.85 * 1000, so the harness compacts
    # before call 2 (the summarizer); call 3 continues from the summary alone.
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")], total=900), _ai("SUMMARY TEXT"),
              _ai("done", total=50)]
    log = {"bind": [], "prompts": []}
    agent = build_react_agent(Scripted(script=script, log=log), TOOLS, "sys", context_window=1000,
                              compact_prompt="COMPACT NOW")
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("task")]}))
    summarizer = log["prompts"][1]
    assert [m["type"] for m in summarizer] == ["system", "human", "ai", "tool", "human"]
    assert summarizer[0]["content"] == "sys" and summarizer[-1]["content"] == "COMPACT NOW"
    assert log["bind"][1][1] == {"tool_choice": "none"}        # the summarizer only writes text
    continuation = CONTINUATION.format(summary="SUMMARY TEXT")
    assert [m.type for m in out["messages"]] == ["human", "ai"]
    assert out["messages"][0].content == continuation and log["prompts"][2][1]["content"] == continuation
    assert out["messages"][-1].content == "done"


def test_below_the_threshold_messages_append_linearly():
    script = [_ai("", [("add", {"a": 1, "b": 2}, "c1")], total=800), _ai("done", total=820)]
    log = {"bind": [], "prompts": []}
    agent = build_react_agent(Scripted(script=script, log=log), TOOLS, "sys", context_window=1000)
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("task")]}))
    assert [m.type for m in out["messages"]] == ["human", "ai", "tool", "ai"] and len(log["prompts"]) == 2


def test_compaction_triggers_once_when_parallel_outputs_only_cross_together():
    # Neither slow_echo output alone reaches the threshold (700 + 350/4 = 787 < 850); combined
    # they do (700 + 700/4 = 875 >= 850). The compaction check must see both tool results
    # merged in one call_model invocation, not one call per Send branch, or this due
    # compaction would be skipped.
    script = [_ai("", [("slow_echo", {"text": "A" * 350, "delay": 0.0}, "c1"),
                       ("slow_echo", {"text": "B" * 350, "delay": 0.0}, "c2")], total=700),
              _ai("SUMMARY TEXT"), _ai("done")]
    log = {"bind": [], "prompts": []}
    agent = build_react_agent(Scripted(script=script, log=log), TOOLS, "sys", context_window=1000,
                              compact_prompt="COMPACT NOW")
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("task")]}))
    assert len(log["prompts"]) == 3                                       # no call on the old history
    assert sum(kwargs == {"tool_choice": "none"} for _, kwargs in log["bind"]) == 1  # one summarizer call
    continuation = CONTINUATION.format(summary="SUMMARY TEXT")
    assert [m.type for m in out["messages"]] == ["human", "ai"]
    assert out["messages"][0].content == continuation and out["messages"][-1].content == "done"


def test_compaction_triggers_once_even_when_one_parallel_output_alone_crosses():
    # slow_echo's first output alone already crosses (700 + 700/4 = 875 >= 850); the second does
    # not (700 + 10/4 = 702 < 850). Per-Send routing would send the first branch to "compact"
    # and the second to "model" in the same step, producing a stray reply on the stale merged
    # history alongside the summary; call_model must instead see them merged and compact once.
    script = [_ai("", [("slow_echo", {"text": "C" * 700, "delay": 0.0}, "c1"),
                       ("slow_echo", {"text": "D" * 10, "delay": 0.0}, "c2")], total=700),
              _ai("SUMMARY TEXT"), _ai("done")]
    log = {"bind": [], "prompts": []}
    agent = build_react_agent(Scripted(script=script, log=log), TOOLS, "sys", context_window=1000,
                              compact_prompt="COMPACT NOW")
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("task")]}))
    assert len(log["prompts"]) == 3                                       # no model call besides the summarizer
    assert sum(kwargs == {"tool_choice": "none"} for _, kwargs in log["bind"]) == 1
    continuation = CONTINUATION.format(summary="SUMMARY TEXT")
    assert [m.type for m in out["messages"]] == ["human", "ai"]
    assert out["messages"][0].content == continuation and out["messages"][-1].content == "done"


def test_the_default_compaction_prompt_is_the_seed_prompt_file():
    from agent.harness import COMPACT_PROMPT
    assert COMPACT_PROMPT.is_file() and "Next step" in COMPACT_PROMPT.read_text()
```

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_harness.py -p no:cacheprovider`
Expected: FAIL (`ImportError: cannot import name 'CONTINUATION' from 'agent.harness'`).

- [ ] **Step 10: Implement auto-compact**

The final `harness.py` (the compaction check runs inside `call_model`, on the merged state):

```python
# seed_agent/agent/harness.py
"""Single-agent inner loop: an explicit ReAct graph, plus Claude-Code-style auto-compaction.

Reproduces langchain.agents.create_agent (langchain 1.4.2, factory.py @ 4af7ab8)
with no middleware, no response_format and async tools:
  START -> model; model -> END if the last AI message has no tool calls,
  else one Send("tools", [call]) per tool call (parallel); tools -> model.
Tool execution follows langgraph.prebuilt.ToolNode's default messages.

Two deliberate differences from create_agent's default:
  1. A tool that raises is reported to the model as an error ToolMessage
     (create_agent re-raises and ends the run).
  2. Auto-compact: call_model estimates the context size before every model call
     (the last reply's reported usage plus ~4 chars/token for what came after, with
     each image counted as a fixed IMAGE_TOKENS), on the merged state -- there is no
     per-Send routing decision, so parallel tool results are never seen in isolation.
     Below compact_at * context_window nothing happens and messages append linearly.
     At or above it, a dedicated summarizer call (same system prompt, tools and
     history; tool_choice="none") condenses the history, and the history is replaced
     by one user message -- a continuation preamble plus the summary -- before the
     real model call runs on it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (AIMessage, AnyMessage, HumanMessage, RemoveMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.types import Send
from pydantic import ValidationError

RECURSION_LIMIT = 9_999
COMPACT_AT = 0.85
COMPACT_PROMPT = Path(__file__).resolve().parent / "prompts" / "compact.md"
INVALID_TOOL = "Error: {name} is not a valid tool, try one of [{names}]."
INVALID_ARGS = ("Error invoking tool '{name}' with kwargs {args} with error:\n"
                " {error}\n Please fix the error and try again.")
TOOL_ERROR = "Error: {error}\n Please fix your mistakes."   # ToolNode's handle_tool_errors=True text
TOOL_BLOCK_TYPES = {"text", "image_url", "image", "json", "search_result", "custom_tool_call_output",
                    "document", "file"}

CHARS_PER_TOKEN = 4          # no tokenizer offline (tiktoken downloads its encodings)
# An image costs a bounded number of vision tokens however long its base64 is, so it is
# counted as a fixed, generous estimate instead of by characters.
IMAGE_TOKENS = 1_500
IMAGE_BLOCK_TYPES = {"image_url", "image", "input_image"}
CONTINUATION = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation.\n\n"
    "Summary:\n{summary}\n\n"
    "Continue the work from where it left off without asking any further questions. "
    "Resume directly: do not acknowledge the summary or recap what was happening."
)


class ReactState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _content(output: Any) -> str | list:
    if isinstance(output, str) or (isinstance(output, list) and all(
            isinstance(x, dict) and x.get("type") in TOOL_BLOCK_TYPES for x in output)):
        return output
    try:
        return json.dumps(output, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(output)


async def run_tool(tools: dict[str, BaseTool], call: ToolCall) -> ToolMessage:
    tool = tools.get(call["name"])
    if tool is None:
        return ToolMessage(INVALID_TOOL.format(name=call["name"], names=", ".join(tools)),
                           name=call["name"], tool_call_id=call["id"], status="error")
    try:
        message = await tool.ainvoke({**call, "type": "tool_call"})
    except ValidationError as exc:
        error = "\n".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', 'Unknown error')}"
                          for e in exc.errors())
        return ToolMessage(INVALID_ARGS.format(name=call["name"], args=call["args"], error=error),
                           name=call["name"], tool_call_id=call["id"], status="error")
    except Exception as exc:  # noqa: BLE001 -- difference 1: report, do not crash
        return ToolMessage(TOOL_ERROR.format(error=repr(exc)), name=call["name"],
                           tool_call_id=call["id"], status="error")
    message.content = _content(message.content)
    return message


def _chars(message: AnyMessage) -> int:
    if isinstance(message.content, str):
        size = len(message.content)
    else:
        size = sum(IMAGE_TOKENS * CHARS_PER_TOKEN
                   if isinstance(block, dict) and block.get("type") in IMAGE_BLOCK_TYPES
                   else len(block if isinstance(block, str) else json.dumps(block, default=str))
                   for block in message.content)
    calls = getattr(message, "tool_calls", None) or []
    return size + (len(json.dumps(calls, default=str)) if calls else 0)


def estimate_tokens(messages: list[AnyMessage], system_prompt: str | None = None) -> int:
    """The last reply's reported usage (input + output) plus an estimate for what came after.
    Without a reported usage (first call, or right after compaction), estimate everything."""
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if isinstance(message, AIMessage) and message.usage_metadata:
            tail = sum(_chars(m) for m in messages[i + 1:])
            return message.usage_metadata["total_tokens"] + tail // CHARS_PER_TOKEN
    head = len(system_prompt or "")
    return (head + sum(_chars(m) for m in messages)) // CHARS_PER_TOKEN


def needs_compaction(messages: list[AnyMessage], system_prompt: str | None, context_window: int,
                     compact_at: float) -> bool:
    # One message cannot be condensed further; let the model call fail loudly instead of looping.
    return len(messages) > 1 and estimate_tokens(messages, system_prompt) >= compact_at * context_window


def build_react_agent(model: BaseChatModel, tools: list[BaseTool], system_prompt: str | None = None, *,
                      context_window: int, compact_at: float = COMPACT_AT,
                      compact_prompt: str | None = None):
    by_name = {t.name: t for t in tools}
    system = [SystemMessage(content=system_prompt)] if system_prompt is not None else []

    async def call_model(state: ReactState) -> dict:
        messages, reset = state["messages"], []
        if needs_compaction(messages, system_prompt, context_window, compact_at):
            # A dedicated summarizer call sees the same system prompt, tools and history as the
            # agent, plus the compaction instruction; tool_choice="none" keeps it to writing text.
            instruction = compact_prompt if compact_prompt is not None else COMPACT_PROMPT.read_text()
            summarizer = model.bind_tools(tools, tool_choice="none") if tools else model.bind()
            reply = await summarizer.ainvoke([*system, *messages, HumanMessage(content=instruction)])
            messages = [HumanMessage(content=CONTINUATION.format(summary=reply.text.strip()))]
            reset = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]
        bound = model.bind_tools(tools, tool_choice=None) if tools else model.bind()
        return {"messages": [*reset, await bound.ainvoke([*system, *messages])]}

    async def call_tool(calls: list[ToolCall]) -> dict:
        return {"messages": [await run_tool(by_name, call) for call in calls]}

    def after_model(state: ReactState):
        last = state["messages"][-1]
        if not last.tool_calls:
            return END
        return [Send("tools", [call]) for call in last.tool_calls]

    graph = StateGraph(ReactState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    if tools:
        graph.add_node("tools", call_tool)
        graph.add_conditional_edges("model", after_model, ["tools", END])
        graph.add_edge("tools", "model")
    else:
        graph.add_edge("model", END)
    return graph.compile().with_config({"recursion_limit": RECURSION_LIMIT})
```

`seed_agent/agent/prompts/compact.md`:
```markdown
Your task is to write a detailed summary of the conversation so far. The summary will
replace the conversation: the work continues from it alone, so it must keep everything
needed to carry on without losing context. Do not call any tools.

Go through the conversation in order: the task you were given, each step you took, what
each tool returned, and what you decided. Pay particular attention to exact values
(clip ids, commit ids, dataset names, file paths, numbers, error messages) and to any
instruction that changed during the work.

Write the summary with these sections:

1. Task and constraints: the task you were given, in full, with every requirement and
   constraint. Quote the original task message verbatim where it matters.
2. Key facts learned: tool results, dataset and clip details, ids, paths and numbers you
   will need again.
3. Files and artifacts: every file you created, changed or relied on, with its path and why
   it matters; include short snippets where the exact content matters.
4. Errors and fixes: every error, rejected candidate or failed check, and how you resolved
   it (or that you did not).
5. Progress: what is done and what worked or did not.
6. Pending work: everything that remains to be done for the task.
7. Current work: exactly what you were doing immediately before this summary.
8. Next step: the single next action, directly in line with the task and the most recent
   work. If there is no next step, say so.

Output only the summary.
```

- [ ] **Step 11: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_harness.py -p no:cacheprovider`
Expected: PASS (26 tests; the comparison scenarios still match `create_agent`).

- [ ] **Step 12: Commit**

```bash
git add seed_agent/agent/harness.py seed_agent/agent/prompts/compact.md tests/test_seed_harness.py
git commit -m "feat(seed): Claude-Code-style auto-compaction in the harness"
```

---

### Task 17: The seed agent (orchestration, roles, tools, prompts, knowledge)

The root node's code (spec 9.1): the agent every later version descends from. The orchestration is plain async Python. Each LLM step is a **role**: a system prompt, optionally extended with reference documents from `knowledge/`, plus a tool list, run on the Task 16 harness. A role returns its result by calling a `submit_<x>` tool whose arguments are validated against a pydantic schema. Invalid arguments come back to the model as a tool error it can fix, so the harness needs no structured-output mode. A role that stops without submitting gets one reminder, then fails the attempt.

- `improve_recipe`: planner → (data builder → recipe writer → `recipe_check`, at most `CHECK_ROUNDS` = 3 times).
- `edit_self`: an edit planner reads the code (read-only tools) and submits an `EditPlan` naming **exactly one component**: prompts, tools, harness, orchestration or knowledge (spec 9.1.1). A coder implements it, then a self-test runs; if it fails, the coder fixes it (at most `SELFTEST_ROUNDS` = 2 rounds). The chosen component is returned in `EditResult.component` and in the summary. It is **not enforced**: nothing rejects a diff that touches other components. The plan is saved to `/workspace/edit_plan.json`, so a contract retry (which inherits the workspace, Task 15) stays with the same component.

The seed's job is to be a **correct, working starting point**, not a strong researcher. The loop improves it. Keep it simple, deterministic where it can be (the recipe check and the self-test are code, not LLM calls), and honest about limits in its prompts.

**Files:**
- Create: `seed_agent/agent/requirements.txt`, `seed_agent/agent/entry.py` (entry points + settings constants)
- Create: `seed_agent/agent/tools.py` (file/bash tools, media tools, MCP adapter, submit tools)
- Create: `seed_agent/agent/orchestration.py` (roles, the `improve_recipe` workflow, the `edit_self` workflow)
- Create: `seed_agent/agent/prompts/{planner,data_builder,recipe_writer,edit_planner,coder}.md`
- Create: `seed_agent/agent/knowledge/data_building.md`, `seed_agent/agent/memory/README.md`
- Test: `tests/test_seed_agent.py`

**Interfaces:**
- Consumes (inside the container): `ar_contract.client.chat_model`, `ar_contract.client.mcp_session`, `ar_contract.models.*` including `EDIT_COMPONENTS` (Task 3); `build_react_agent` (Task 16); the kernel tools by name (Tasks 7–9); the environment variables `AR_DEFAULT_MODEL`, `AR_CONTEXT_WINDOW`, `AR_COMPACT_AT`, `AR_AGENT_DIR` and `AR_WORKSPACE`, which the kernel sets (Tasks 14–15).
- Produces:
  - `agent.entry.edit_self(ctx) -> EditResult` and `agent.entry.improve_recipe(ctx) -> RecipeResult`, both `async`.
  - Pure helpers for tests: `agent.tools.snap_segments(segments, duration)`, `agent.tools.resolve_inside(root, path)`, `agent.tools.replace_once(text, old, new)`, `agent.tools.read_utf8(path)` (strict; `edit_file` refuses non-UTF-8 files rather than corrupt them), `agent.tools.make_file_tools(root, *, writable=True)`, `agent.tools.result_text(result) -> str`, `agent.tools.submit_tool(name, description, schema) -> (StructuredTool, SimpleNamespace)` (the box's `.value` holds the submission), `agent.orchestration.selftest(root) -> list[str]`, and `agent.orchestration.COMPONENTS` / `EditPlan`.

*As built (seed simplicity ruling, see File structure).* The code in Steps 3–5 is shown per first-draft file; it was committed as three modules with the same behaviour: `settings.py` → the constants at the top of `entry.py` (`orchestration.py` imports them; `entry.py` imports `orchestration` lazily inside each entry point, which avoids a circular import); `tools/{files,media,kernel,submit}.py` → `tools.py`; `orchestration/{roles,task,meta}.py` → `orchestration.py`. Once-used helpers were inlined and relative imports adjusted. Other differences:
- The MCP adapter has one helper, `result_text(result)`: `is_error` raises `ToolException` with the joined text; otherwise it prefers `structured_content` (as JSON) over per-item text blocks, so a tool returning a list (for example an empty `hf_search`) is one JSON value. `run_task` parses `recipe_check` with `json.loads(result_text(...))` in place of `result_json`.
- `EditPlan.component` uses `EditComponent` from `ar_contract.models` (Task 3), and `COMPONENTS` names the new paths: `agent/prompts/*.md`, `agent/tools.py`, `agent/harness.py`, `agent/orchestration.py` (+ `agent/entry.py` settings), `agent/knowledge/*.md`.
- `selftest` also rejects keyword-only parameters (matching Task 14's static check) and runs `python -c "import agent.entry, agent.orchestration"`: `entry.py` imports `orchestration` lazily, so importing `agent.entry` alone would not exercise the files `edit_self` edits. `prompts/coder.md` gives the same command; `prompts/edit_planner.md` says the components are listed with their files in the context.
- `knowledge/README.md` was dropped rather than folded into `data_building.md` (folding it would add meta-text to the data builder's system prompt).
- `tests/test_seed_agent.py` imports from `agent.tools` and `agent.orchestration`, and adds `test_kernel_results_prefer_structured_content` and a keyword-only case in the self-test test.

- [ ] **Step 1: Write the failing tests**

The end-to-end tests run each entry point exactly as a container does (`python -m ar_contract.run` in a subprocess). They use the kernel's real gateway in mock mode, with scripts registered on its `MockBook`, and the real tool server with Task 14's canned tool objects. The scripts deliberately include:
- an invalid `submit_plan` (missing `actions`) and an invalid `submit_edit_plan` (unknown component), which must come back as argument errors;
- a call to a tool that does not exist;
- a kernel tool that fails server-side (`hf_download`, disabled in the mock).

All of these must reach the model as tool errors, and the run must still succeed.

```python
# tests/test_seed_agent.py
"""The seed agent: pure helpers in-process, then both entry points run the way a
container runs them (python -m ar_contract.run in a subprocess) against the kernel's
real gateway (scripted mock mode) and the Task 14 mock tool server, over Unix sockets."""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import typing
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEED = REPO / "seed_agent"
sys.path.insert(0, str(SEED))


def test_snap_segments_puts_boundaries_on_rounds():
    from agent.tools import snap_segments
    segs = [{"time_range_s": [0.0, 3.1], "prompt": "walk"}, {"time_range_s": [3.1, 8.0], "prompt": "turn"}]
    out = snap_segments(segs, duration=8.0)
    boundary = out[0]["time_range_s"][1]
    k = (boundary - 25 / 24) / (32 / 24)
    assert abs(k - round(k)) < 1e-9                         # on 25/24 + k*32/24
    assert out[0]["time_range_s"][0] == 0.0 and out[-1]["time_range_s"][1] == 8.0
    assert out[1]["time_range_s"][0] == boundary            # contiguous


def test_snap_segments_drops_segments_that_collapse():
    from agent.tools import snap_segments
    segs = [{"time_range_s": [0.0, 1.0], "prompt": "a"}, {"time_range_s": [1.0, 1.1], "prompt": "b"},
            {"time_range_s": [1.1, 6.0], "prompt": "c"}]
    assert [s["prompt"] for s in snap_segments(segs, duration=6.0)] == ["a", "c"]


@pytest.mark.parametrize("bad", ["../../etc/passwd", "/etc/passwd", "sub/../../x"])
def test_file_tools_stay_inside_their_root(tmp_path, bad):
    from agent.tools import resolve_inside
    with pytest.raises(ValueError):
        resolve_inside(str(tmp_path), bad)


def test_file_tools_accept_paths_inside(tmp_path):
    from agent.tools import resolve_inside
    assert resolve_inside(str(tmp_path), "a/b.txt") == tmp_path / "a" / "b.txt"


def test_edit_file_needs_exactly_one_match():
    from agent.tools import replace_once
    assert replace_once("a b a", "b", "c") == "a c a"
    with pytest.raises(ValueError, match="found 2"):
        replace_once("a b a", "a", "c")
    with pytest.raises(ValueError, match="found 0"):
        replace_once("a", "z", "c")


def test_edit_file_refuses_non_utf8_files_and_leaves_them_untouched(tmp_path):
    from agent.tools import make_file_tools
    raw = b"caption: caf\xe9\n"                              # Latin-1, not UTF-8
    (tmp_path / "c.txt").write_bytes(raw)
    tools = {t.name: t for t in make_file_tools(str(tmp_path))}
    assert "caf" in tools["read_file"].invoke({"path": "c.txt"})     # reading replaces bad bytes
    with pytest.raises(ValueError, match="not UTF-8"):
        tools["edit_file"].invoke({"path": "c.txt", "old": "caption", "new": "title"})
    assert (tmp_path / "c.txt").read_bytes() == raw


def test_read_only_file_tools_cannot_write(tmp_path):
    from agent.tools import make_file_tools
    assert {t.name for t in make_file_tools(str(tmp_path), writable=False)} == {"read_file", "list_dir"}


def test_submit_tool_validates_then_captures():
    from pydantic import ValidationError
    from agent.orchestration import EditPlan
    from agent.tools import submit_tool
    tool, box = submit_tool("submit_edit_plan", "d", EditPlan)
    plan = {"component": "everything", "change": "x", "files": [], "rationale": "r", "expected_effect": "e"}
    with pytest.raises(ValidationError):
        asyncio.run(tool.ainvoke(plan))
    assert box.value is None
    asyncio.run(tool.ainvoke({**plan, "component": "tools"}))
    assert box.value.component == "tools"


def test_edit_components_match_the_contract():
    from ar_contract.models import EDIT_COMPONENTS
    from agent.orchestration import COMPONENTS, EditPlan
    assert tuple(COMPONENTS) == EDIT_COMPONENTS
    assert set(typing.get_args(EditPlan.model_fields["component"].annotation)) == set(EDIT_COMPONENTS)
    for component, where in COMPONENTS.items():
        assert (SEED / where.split(" ")[0].split("*")[0]).exists(), f"{component} -> {where}"


def test_selftest_passes_on_the_seed_and_catches_a_broken_entry(tmp_path):
    from agent.orchestration import selftest
    copy = tmp_path / "copy"
    shutil.copytree(SEED, copy)
    assert selftest(str(copy)) == []
    (copy / "agent" / "entry.py").write_text("def edit_self(ctx, extra):\n    pass\n")
    errors = selftest(str(copy))
    assert any("edit_self" in e for e in errors) and any("improve_recipe" in e for e in errors)


# ---- both entry points against the kernel services -------------------------------------------

C0 = "0" * 64                                        # the Task 14 mock data_commit id


def _call(name, args):
    from ar_kernel.gateway.mock import function_call
    return function_call(name, args, f"call_{uuid.uuid4().hex[:12]}")


def _scripts():
    from ar_kernel.gateway.mock import message
    recipe = [
        [_call("submit_plan", {"hypotheses": ["more walking clips"]})],            # invalid: no actions
        [_call("submit_plan", {"hypotheses": ["more walking clips"], "actions": ["reuse the pool"]})],
        [message("planned")],
        [_call("data_query", {"filter": {"format": "video_caption_camera"}}), _call("no_such_tool", {}),
         _call("hf_download", {"repo": "x/y", "revision": "main", "patterns": ["*.mp4"]})],
        [_call("submit_data_commit", {"data_commit": C0, "notes": "pool clips"})],
        [message("built")],
        [_call("submit_recipe", {"recipe": {"optimizer.max_steps": 200.4, "optimizer.lr": 1e-5},
                                 "rationale": "fits the data"})],
        [message("recipe written")],
    ]
    edit = [
        [_call("read_file", {"path": "agent/prompts/planner.md"})],
        [_call("submit_edit_plan", {"component": "everything", "change": "x", "files": [],
                                    "rationale": "r", "expected_effect": "e"})],   # invalid component
        [_call("submit_edit_plan", {"component": "prompts", "change": "ask for posed clips first",
                                    "files": ["agent/prompts/planner.md"], "rationale": "static-only clips",
                                    "expected_effect": "more moving clips"})],
        [message("planned")],
        [_call("edit_file", {"path": "agent/prompts/planner.md", "old": "Finish by calling submit_plan.",
                             "new": "Prefer clips with poses. Finish by calling submit_plan."})],
        [message("Changed the planner prompt to prefer clips with poses.")],
    ]
    return {"recipe": recipe, "edit": edit}


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    from ar_kernel.contract.verify import _MockData, _MockHf
    from ar_kernel.gateway.app import create_gateway_app
    from ar_kernel.gateway.mock import MockBook
    from ar_kernel.gateway.store import CallStore
    from ar_kernel.services import RunServices, socket_dir_for
    from ar_kernel.telemetry.recorder import Recorder
    from ar_kernel.tools.context import TokenRegistry
    from ar_kernel.tools.data_tools import register_data_tools
    from ar_kernel.tools.hf_tools import register_hf_tools
    from ar_kernel.tools.jobs import JobQueue, register_job_tools
    from ar_kernel.tools.server import ToolKit, build_tool_app, new_mcp
    run = tmp_path_factory.mktemp("run")
    rec = Recorder(run)
    registry, queue = TokenRegistry(rec), JobQueue(rec, threading.Lock(), wait_cap_s=5.0)
    kit, mcp = ToolKit(registry, rec), new_mcp()
    register_data_tools(mcp, kit, _MockData())
    register_hf_tools(mcp, kit, _MockHf())
    register_job_tools(mcp, kit, queue)
    book = MockBook.default()
    for name, script in _scripts().items():
        book.add(name, script)
    services = RunServices(socket_dir_for(run))
    services.start(create_gateway_app(registry=registry, store=CallStore(rec), allowed_models={"mock-model"},
                                      upstream=None, mocks=book), build_tool_app(mcp))
    yield rec, registry, services
    services.stop()
    queue.shutdown()


def _run(kernel, tmp_path, kind, script, node, ctx):
    rec, registry, services = kernel
    agent = tmp_path / "agent_copy"
    shutil.copytree(SEED, agent)
    ws, ctx_dir = tmp_path / "ws", tmp_path / "ctx"
    ws.mkdir(), ctx_dir.mkdir()
    (ctx_dir / "context.json").write_text(json.dumps(ctx))
    caller = registry.issue(node=node, phase=kind, attempt=1, workspace_host=ws, staging_host=tmp_path,
                            mock_script=script)
    env = {**os.environ, "AR_SOCKET_DIR": str(services.socket_dir), "AR_TOKEN": caller.token,
           "AR_DEFAULT_MODEL": "mock-model", "AR_WORKSPACE": str(ws), "AR_CONTEXT_DIR": str(ctx_dir),
           "AR_AGENT_DIR": str(agent)}
    proc = subprocess.run([sys.executable, "-m", "ar_contract.run", kind], env=env, cwd=REPO,
                          capture_output=True, text=True, timeout=180)
    registry.revoke(caller.token)
    return proc, json.loads((ws / "result.json").read_text()), agent


def _tool_outputs(rec, node) -> list[str]:
    """Every function_call_output the agent sent back to the model, from gateway telemetry."""
    out = []
    for event in rec.read_events(node):
        if event["type"] == "llm.request":
            body = rec.load_payload(event["payload"])["body"]
            out += [i.get("output", "") for i in body.get("input", [])
                    if isinstance(i, dict) and i.get("type") == "function_call_output"]
    return out


def test_chat_model_works_sync_and_async_over_the_socket(kernel, tmp_path, monkeypatch):
    """The container has no network: both the sync and the async client must use the socket."""
    _, registry, services = kernel
    caller = registry.issue(node="n-chat", phase="edit_self", attempt=1, workspace_host=tmp_path,
                            staging_host=tmp_path, mock_script="smoke")
    monkeypatch.setenv("AR_SOCKET_DIR", str(services.socket_dir))
    monkeypatch.setenv("AR_TOKEN", caller.token)
    monkeypatch.setenv("AR_DEFAULT_MODEL", "mock-model")
    from langchain_core.messages import HumanMessage
    from ar_contract.client import chat_model
    model = chat_model()
    assert model.invoke([HumanMessage("ping")]).text == "ok"
    assert asyncio.run(model.ainvoke([HumanMessage("ping")])).text == "ok"
    registry.revoke(caller.token)


BASE = {"nodes_remaining": 3, "attempt": 1, "max_attempts": 3}


@pytest.mark.parametrize("kind", ["edit_self", "improve_recipe"])
def test_dry_run_against_the_kernel_services(kernel, tmp_path, kind):
    proc, body, _ = _run(kernel, tmp_path, kind, "smoke", f"dry-{kind}", {**BASE, "dry_run": True})
    assert proc.returncode == 0 and body["ok"], (body, proc.stderr[-2000:])


def test_improve_recipe_full_flow(kernel, tmp_path):
    rec = kernel[0]
    ctx = {**BASE, "tunable_rules": {"optimizer.max_steps": {"type": "int"}, "optimizer.lr": {"type": "float"}}}
    proc, body, _ = _run(kernel, tmp_path, "improve_recipe", "recipe", "n-recipe", ctx)
    assert proc.returncode == 0 and body["ok"], (body, proc.stderr[-2000:])
    assert body["result"]["data_commit"] == C0
    assert body["result"]["recipe"] == {"optimizer.max_steps": 200, "optimizer.lr": 1e-5}   # int coerced
    outputs = _tool_outputs(rec, "n-recipe")
    assert any("Error invoking tool 'submit_plan'" in o and "actions" in o for o in outputs)   # bad args
    assert any("no_such_tool is not a valid tool" in o for o in outputs)                       # unknown tool
    assert any(o.startswith("Error: ToolException(") and "downloads are disabled" in o
               for o in outputs)                                                 # kernel tool error reported
    kinds = [e["type"] for e in rec.read_events("n-recipe")]
    assert "tool.call" in kinds and "tool.error" in kinds                        # kernel-side records


def test_edit_self_plans_exactly_one_component(kernel, tmp_path):
    rec = kernel[0]
    proc, body, agent = _run(kernel, tmp_path, "edit_self", "edit", "n-edit", BASE)
    assert proc.returncode == 0 and body["ok"], (body, proc.stderr[-2000:])
    assert body["result"]["component"] == "prompts"
    assert body["result"]["summary"].startswith("[prompts] ask for posed clips first")
    assert "Prefer clips with poses." in (agent / "agent" / "prompts" / "planner.md").read_text()
    assert json.loads((tmp_path / "ws" / "edit_plan.json").read_text())["component"] == "prompts"
    assert any("Error invoking tool 'submit_edit_plan'" in o and "component" in o
               for o in _tool_outputs(rec, "n-edit"))


@pytest.mark.docker
def test_seed_agent_passes_contract_verification(tmp_path):
    from ar_kernel.config import KernelConfig
    from ar_kernel.contract.verify import ContractHarness, verify_contract
    from ar_kernel.telemetry.recorder import Recorder
    from ar_kernel.vcs.agents_repo import AgentsRepo
    harness = ContractHarness(KernelConfig.load(), tmp_path / "run", Recorder(tmp_path / "run"))
    harness.start()
    try:
        repo = AgentsRepo(tmp_path / "agents.git")
        commit = repo.init(SEED)
        report = verify_contract(cfg=KernelConfig.load(), run_dir=tmp_path / "run", run_id="seed", repo=repo,
                                 commit=commit, harness=harness, recorder=Recorder(tmp_path / "run"),
                                 node="root", attempt=1)
    finally:
        harness.stop()
    assert report.ok, report.to_retry()
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_agent.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError: No module named 'agent.tools'`).

- [ ] **Step 3: Settings, entry, requirements, memory, knowledge**

```text
# seed_agent/agent/requirements.txt
# Extra runtime packages for this agent version. The base image already provides
# langgraph, langchain-core, langchain-openai, mcp, httpx, httpx2, pydantic, numpy,
# opencv-python-headless, Pillow and ffmpeg. The container has no network, so
# anything listed here is baked into the image at build time.
```

```python
# seed_agent/agent/settings.py
import os

MODEL = os.environ.get("AR_DEFAULT_MODEL", "mock-model")
CONTEXT_WINDOW = int(os.environ.get("AR_CONTEXT_WINDOW", "128000"))   # the model's window, in tokens
COMPACT_AT = float(os.environ.get("AR_COMPACT_AT", "0.85"))           # auto-compact threshold
AGENT_ROOT = os.environ.get("AR_AGENT_DIR", "/agent")
WORKSPACE = os.environ.get("AR_WORKSPACE", "/workspace")
CHECK_ROUNDS = 3             # build -> recipe -> recipe_check loops inside one attempt
SELFTEST_ROUNDS = 2          # implement -> self-test loops inside one attempt
BRIEF_CHARS = 60_000         # cap on the JSON context handed to a role
```

```python
# seed_agent/agent/entry.py
"""Fixed entry points (spec 1.1.4). The kernel imports exactly these two names."""
from ar_contract.models import EditContext, EditResult, RecipeContext, RecipeResult


async def edit_self(ctx: EditContext) -> EditResult:
    from .orchestration.meta import run_meta
    return await run_meta(ctx)


async def improve_recipe(ctx: RecipeContext) -> RecipeResult:
    from .orchestration.task import run_task
    return await run_task(ctx)
```

`seed_agent/agent/memory/README.md`:
```markdown
# Agent memory

Notes written here during `edit_self` are committed with the code and inherited by
every descendant. Use them to record what was tried, what the scores said, and
which ideas to try next. `improve_recipe` sees this directory read-only.
```

`seed_agent/agent/knowledge/README.md`:
```markdown
# Agent knowledge

Reference material that roles read: formats, conversion recipes, dataset notes. The
orchestration appends the documents a role needs to that role's system prompt
(`orchestration/roles.py`, `system_prompt(name, knowledge=...)`). Unlike `memory/`,
which records what was tried, this directory holds what is known to be true.
```

`seed_agent/agent/knowledge/data_building.md`:
```markdown
## Converting clips to the standard formats

- Probe first: `ffprobe -v error -select_streams v:0 -show_entries
  stream=width,height,avg_frame_rate,nb_frames,sample_aspect_ratio:stream_side_data=rotation
  -of json in.mp4`. Display aspect = width * SAR / height, with width and height swapped for
  a 90 or 270 degree rotation.
- Center-crop to 16:9 without stretching: `ffmpeg -i in.mp4 -vf
  "crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',setsar=1" -c:v libx264 -crf 18 -an out.mp4`.
  Re-encoding also applies any display rotation.
- Frame rate: keep the source rate if it is >= 24 fps; never raise it by duplicating frames.
- Trimming changes the frame count: slice the pose array to the same frames (N poses for
  N frames).

## Camera poses

- `poses/<id>.npz` holds `cam_c2w` with shape [N, 4, 4]: camera-to-world, OpenCV convention
  (x right, y down, z forward).
- From world-to-camera matrices, invert them. From OpenGL convention (y up, z backward),
  flip the y and z axes of the camera frame: `c2w_cv = c2w_gl @ diag(1, -1, -1, 1)`.

## Timed prompts

- Segment boundaries must fall on round boundaries: 25/24 s, then every 32/24 s. Use
  snap_timed_prompts after writing the segments.
```

- [ ] **Step 4: Tools**

```python
# seed_agent/agent/tools/files.py
"""File and shell tools bound to one root (/agent for edit_self, /workspace for improve_recipe)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import tool

MAX_READ = 200_000
MAX_OUTPUT = 8_000


def resolve_inside(root: str, path: str) -> Path:
    base = Path(root).resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"{path} is outside {root}")
    return target


def read_utf8(target: Path) -> str:
    """Strict UTF-8 read for editing: replacing undecodable bytes and writing the text back
    would silently corrupt the file."""
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{target.name} is not UTF-8 text (byte {exc.start}); "
                         f"change it with run_command instead") from None


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"old text must appear exactly once, found {count} matches")
    return text.replace(old, new, 1)


def make_file_tools(root: str, *, writable: bool = True) -> list:
    @tool
    def read_file(path: str) -> str:
        """Read a text file (path relative to the tool root)."""
        return resolve_inside(root, path).read_text(encoding="utf-8", errors="replace")[:MAX_READ]

    @tool
    def list_dir(path: str = ".") -> list[str]:
        """List a directory (path relative to the tool root); directories end with '/'."""
        target = resolve_inside(root, path)
        return sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file (path relative to the tool root)."""
        target = resolve_inside(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace one exact occurrence of `old` with `new` in a text file. `old` must appear exactly once."""
        target = resolve_inside(root, path)
        target.write_text(replace_once(read_utf8(target), old, new), encoding="utf-8")
        return f"edited {path}"

    @tool
    def run_command(command: str, timeout_s: int = 600) -> str:
        """Run a shell command in the tool root (ffmpeg, ffprobe, python, ...). Returns the exit code and the output tail."""
        try:
            r = subprocess.run(["bash", "-lc", command], cwd=root, capture_output=True, text=True,
                               timeout=min(int(timeout_s), 3600))
        except subprocess.TimeoutExpired:
            return f"timed out after {timeout_s}s"
        return f"exit {r.returncode}\n{(r.stdout + r.stderr)[-MAX_OUTPUT:]}"

    if not writable:
        return [read_file, list_dir]
    return [read_file, list_dir, write_file, edit_file, run_command]
```

```python
# seed_agent/agent/tools/media.py
"""Media helpers: timed-prompt snapping and frame captioning."""
from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

FIRST_ROUND_END = 25 / 24          # the 25 history frames
ROUND = 32 / 24                    # one rollout round


def round_boundaries(duration: float) -> list[float]:
    out, t = [], FIRST_ROUND_END
    while t < duration:
        out.append(t)
        t += ROUND
    return out


def snap_segments(segments: list[dict], duration: float) -> list[dict]:
    """Snap internal boundaries to 25/24 + k*32/24 s (per_chunk rule), keep the ends,
    make segments contiguous, and drop any that collapse to zero length."""
    bounds = round_boundaries(duration)
    ordered = sorted(segments, key=lambda s: s["time_range_s"][0])
    cuts = [0.0]
    for seg in ordered[:-1]:
        end = seg["time_range_s"][1]
        cuts.append(min(bounds, key=lambda b: abs(b - end)) if bounds else end)
    cuts.append(float(duration))
    out = []
    for seg, start, end in zip(ordered, cuts, cuts[1:]):
        if end - start > 1e-9:
            out.append({"time_range_s": [start, end], "prompt": seg["prompt"]})
    return out


@tool
def snap_timed_prompts(segments_json: str, duration: float) -> str:
    """Snap timed-prompt segment boundaries to rollout-round boundaries. Input and output: a JSON list of
    {"time_range_s": [start, end], "prompt": str}."""
    return json.dumps(snap_segments(json.loads(segments_json), duration))


@tool
async def caption_clip(video_path: str, hint: str = "") -> str:
    """Caption a video: samples 4 frames with ffmpeg and asks a vision model (through the kernel
    gateway) for one factual caption of the scene and the camera motion."""
    from ar_contract.client import chat_model
    duration = float(json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path],
        capture_output=True, text=True, check=True).stdout)["format"]["duration"])
    content = [{"type": "text", "text": "Write one factual caption (1-3 sentences) describing the "
                                        "scene and how the camera moves. " + hint}]
    with tempfile.TemporaryDirectory() as tmp:
        for i, frac in enumerate((0.1, 0.35, 0.6, 0.85)):
            frame = Path(tmp) / f"{i}.jpg"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{duration * frac:.3f}",
                            "-i", video_path, "-frames:v", "1", "-vf", "scale=512:-2", str(frame)], check=True)
            b64 = base64.b64encode(frame.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    reply = await chat_model().ainvoke([HumanMessage(content=content)])
    return reply.text.strip()
```

The kernel-tool adapter. MCP 2.x results use snake_case attributes: `is_error`, `structured_content`, `input_schema` (fact 16). Each MCP tool becomes a `StructuredTool` whose `args_schema` is the tool's JSON schema, all on one MCP session. A failed kernel tool raises `ToolException`, which the harness reports to the model.

```python
# seed_agent/agent/tools/kernel.py
"""Kernel tools (the MCP tool server) as LangChain tools, over one MCP session."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool, ToolException


def result_text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


def result_json(result) -> Any:
    """A successful kernel tool result as Python data; a failed one raises ToolException."""
    if result.is_error:
        raise ToolException(result_text(result))
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result_text(result))


async def kernel_tools(session) -> list[StructuredTool]:
    listed = await session.list_tools()
    tools = []
    for spec in listed.tools:
        async def call(_name: str = spec.name, **kwargs: Any) -> str:
            result = await session.call_tool(_name, kwargs)
            if result.is_error:
                raise ToolException(result_text(result))   # the harness reports it to the model
            return result_text(result)
        tools.append(StructuredTool(name=spec.name, description=spec.description or "",
                                    args_schema=spec.input_schema, coroutine=call))
    return tools
```

```python
# seed_agent/agent/tools/submit.py
"""Result tools: a role finishes by calling submit_<x> with arguments validated against a schema.
Invalid arguments come back to the model as a tool error (the harness), so it can correct them."""
from __future__ import annotations

from pydantic import BaseModel
from langchain_core.tools import StructuredTool


class Submission:
    def __init__(self) -> None:
        self.value: BaseModel | None = None


def submit_tool(name: str, description: str, schema: type[BaseModel]) -> tuple[StructuredTool, Submission]:
    box = Submission()

    async def submit(**kwargs) -> str:
        box.value = schema.model_validate(kwargs)
        return "submitted"

    return StructuredTool(name=name, description=description, args_schema=schema, coroutine=submit), box
```

- [ ] **Step 5: Orchestration**

```python
# seed_agent/agent/orchestration/roles.py
"""A role = a system prompt (+ reference knowledge) and a tool list, run on the harness."""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AnyMessage, HumanMessage

from ..harness.react import build_react_agent
from ..settings import COMPACT_AT, CONTEXT_WINDOW, MODEL
from ..tools.submit import Submission

AGENT_PKG = Path(__file__).resolve().parents[1]
REMIND = ("You stopped without calling {tool}. Finish the task, then call {tool} with the result. "
          "The work is only recorded through {tool}.")


def system_prompt(name: str, knowledge: tuple[str, ...] = ()) -> str:
    text = (AGENT_PKG / "prompts" / f"{name}.md").read_text()
    for doc in knowledge:
        text += f"\n\n# Reference: {doc}\n\n" + (AGENT_PKG / "knowledge" / doc).read_text()
    return text


def _model():
    from ar_contract.client import chat_model
    return chat_model(MODEL)


async def run_role(system: str, tools: list, task: str, submission: Submission | None = None,
                   submit_name: str = "", model=None) -> list[AnyMessage]:
    """Run one role to completion. A role that must submit gets one reminder if it stops early."""
    agent = build_react_agent(model or _model(), tools, system, context_window=CONTEXT_WINDOW,
                              compact_at=COMPACT_AT)
    state = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    if submission is not None and submission.value is None:
        state = await agent.ainvoke({"messages": [*state["messages"],
                                                  HumanMessage(content=REMIND.format(tool=submit_name))]})
    if submission is not None and submission.value is None:
        raise RuntimeError(f"the role finished without calling {submit_name}")
    return state["messages"]
```

```python
# seed_agent/agent/orchestration/task.py
"""improve_recipe: plan -> (build data -> write recipe -> recipe_check) x up to CHECK_ROUNDS."""
from __future__ import annotations

import json

from ar_contract.client import mcp_session
from ar_contract.models import RecipeContext, RecipeResult
from pydantic import BaseModel, Field

from ..settings import BRIEF_CHARS, CHECK_ROUNDS, WORKSPACE
from ..tools.files import make_file_tools
from ..tools.kernel import kernel_tools, result_json
from ..tools.media import caption_clip, snap_timed_prompts
from ..tools.submit import submit_tool
from .roles import run_role, system_prompt


class DataPlan(BaseModel):
    hypotheses: list[str] = Field(min_length=1, description="1-3 testable data hypotheses")
    actions: list[str] = Field(min_length=1, description="concrete steps that test them")


class BuildOutcome(BaseModel):
    data_commit: str = Field(min_length=1, description="the data_commit id to train on")
    notes: str = Field(description="what the commit contains and why")


class RecipeDraft(BaseModel):
    recipe: dict[str, float] = Field(description="tunable key -> value")
    rationale: str = Field(min_length=1)


def brief(ctx: RecipeContext) -> str:
    return json.dumps({"lineage": ctx.lineage, "archive": ctx.archive, "n_gpus": ctx.n_gpus,
                       "parent_data_commit": ctx.parent_data_commit, "parent_recipe": ctx.parent_recipe,
                       "clip_pool_size": len(ctx.clip_pool), "tools": ctx.tools,
                       "retry": ctx.retry, "format_rules": ctx.format_rules}, default=str)[:BRIEF_CHARS]


def coerce(recipe: dict, rules: dict) -> dict:
    return {key: int(round(value)) if rules.get(key, {}).get("type") == "int" else float(value)
            for key, value in recipe.items()}


async def run_task(ctx: RecipeContext) -> RecipeResult:
    async with mcp_session() as session:
        ktools = await kernel_tools(session)
        if ctx.dry_run:                  # contract smoke run: prove the wiring, do no work
            messages = await run_role("Reply with the single word ok.", [], "ping")
            return RecipeResult(data_commit=ctx.parent_data_commit or "dry-run", recipe={},
                                rationale=f"dry run: {len(ktools)} kernel tools, model said "
                                          f"{messages[-1].text!r}")
        context = brief(ctx)
        plan_tool, plan = submit_tool("submit_plan", "Submit the data plan for this node.", DataPlan)
        await run_role(system_prompt("planner"), [plan_tool], context, plan, "submit_plan")
        failures: list = []
        for _ in range(CHECK_ROUNDS):
            build_tool, built = submit_tool("submit_data_commit", "Submit the data commit to train on.",
                                            BuildOutcome)
            task = f"PLAN:\n{plan.value.model_dump_json(indent=2)}\n\nCONTEXT:\n{context}"
            if failures:
                task += ("\n\nTHE LAST RECIPE CHECK FAILED. Fix the data if the failures are about data:\n"
                         + json.dumps(failures))
            await run_role(system_prompt("data_builder", knowledge=("data_building.md",)),
                           [*ktools, *make_file_tools(WORKSPACE), caption_clip, snap_timed_prompts,
                            build_tool], task, built, "submit_data_commit")
            recipe_tool, draft = submit_tool("submit_recipe", "Submit the training recipe.", RecipeDraft)
            await run_role(system_prompt("recipe_writer"), [recipe_tool], json.dumps(
                {"rules": ctx.tunable_rules, "resolution_allowlist": ctx.resolution_allowlist,
                 "lora_allowlist": ctx.lora_allowlist, "n_gpus": ctx.n_gpus, "data_notes": built.value.notes,
                 "parent_recipe": ctx.parent_recipe, "previous_failures": failures}), draft, "submit_recipe")
            recipe = coerce(draft.value.recipe, ctx.tunable_rules)
            try:
                check = result_json(await session.call_tool(
                    "recipe_check", {"recipe": recipe, "data_commit": built.value.data_commit}))
                failures = [] if check.get("ok") else check.get("failures", [])
            except Exception as exc:  # noqa: BLE001 -- a failed check is a failure to fix
                failures = [str(exc)]
            if not failures:
                break
    rationale = draft.value.rationale
    if failures:
        rationale += f"\n\nUnresolved recipe_check failures: {failures}"
    return RecipeResult(data_commit=built.value.data_commit, recipe=recipe,
                        rationale=f"{rationale}\n\nPlan: {plan.value.model_dump_json()}\nData: {built.value.notes}")
```

```python
# seed_agent/agent/orchestration/meta.py
"""edit_self: plan ONE edit to ONE component -> implement -> self-test -> (fix | finish)."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from ar_contract.models import EditContext, EditResult
from pydantic import BaseModel, Field

from ..settings import AGENT_ROOT, BRIEF_CHARS, SELFTEST_ROUNDS, WORKSPACE
from ..tools.files import make_file_tools
from ..tools.submit import submit_tool
from .roles import run_role, system_prompt

# Where each component lives in this agent (spec 9.1.1). The plan names one of them.
COMPONENTS = {
    "prompts": "agent/prompts/*.md -- the system prompt of each role",
    "tools": "agent/tools/ -- agent-local tools and the kernel-tool adapter (not the kernel tools themselves)",
    "harness": "agent/harness/ -- the single-agent inner loop: ReAct graph, tool execution, auto-compaction",
    "orchestration": "agent/orchestration/ -- which roles run, in what order, with which tools and context",
    "knowledge": "agent/knowledge/*.md -- reference material that roles read",
}


class EditPlan(BaseModel):
    component: Literal["prompts", "tools", "harness", "orchestration", "knowledge"] = Field(
        description="the ONE component this edit changes")
    change: str = Field(min_length=1, description="the concrete change")
    files: list[str] = Field(description="files you expect to change, relative to /agent")
    rationale: str = Field(min_length=1, description="evidence from the lineage for this change")
    expected_effect: str = Field(min_length=1, description="what should improve, and how you will know")


def selftest(root: str) -> list[str]:
    """The contract's static and import checks, run locally so the agent can fix itself."""
    errors = []
    try:
        tree = ast.parse((Path(root) / "agent" / "entry.py").read_text())
        top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in ("edit_self", "improve_recipe"):
            fn = top.get(name)
            if fn is None or len(fn.args.posonlyargs + fn.args.args) != 1 or fn.args.vararg or fn.args.kwarg:
                errors.append(f"agent/entry.py needs top-level {name}(ctx) with exactly one parameter")
    except (OSError, SyntaxError) as exc:
        errors.append(f"agent/entry.py: {exc}")
    r = subprocess.run([sys.executable, "-c", "import agent.entry"], cwd=root, capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        errors.append(r.stderr[-3000:])
    return errors


def brief(ctx: EditContext, previous_plan: dict | None) -> str:
    return json.dumps({"components": COMPONENTS, "lineage": ctx.lineage, "archive": ctx.archive,
                       "nodes_remaining": ctx.nodes_remaining, "retry": ctx.retry,
                       "previous_attempt_plan": previous_plan}, default=str)[:BRIEF_CHARS]


async def run_meta(ctx: EditContext) -> EditResult:
    if ctx.dry_run:
        messages = await run_role("Reply with the single word ok.", [], "ping")
        return EditResult(summary=f"dry run: model said {messages[-1].text!r}")
    plan_file = Path(WORKSPACE) / "edit_plan.json"          # carried to a retry with the workspace
    previous = json.loads(plan_file.read_text()) if ctx.retry and plan_file.exists() else None
    plan_tool, plan = submit_tool("submit_edit_plan",
                                  "Submit the edit plan: exactly one component and one focused change.",
                                  EditPlan)
    await run_role(system_prompt("edit_planner"), [*make_file_tools(AGENT_ROOT, writable=False), plan_tool],
                   brief(ctx, previous), plan, "submit_edit_plan")
    p = plan.value
    plan_file.write_text(p.model_dump_json(indent=2))
    errors: list[str] = []
    summary = ""
    for _ in range(SELFTEST_ROUNDS):
        task = f"EDIT PLAN (component: {p.component}):\n{p.model_dump_json(indent=2)}"
        if errors:
            task += "\n\nTHE SELF-TEST FAILED; fix these first:\n" + "\n".join(errors)
        messages = await run_role(system_prompt("coder"), make_file_tools(AGENT_ROOT), task)
        summary = messages[-1].text
        errors = selftest(AGENT_ROOT)
        if not errors:
            break
    note = f"\n\n(self-test still failing: {errors})" if errors else ""
    return EditResult(summary=f"[{p.component}] {p.change}\n\n{summary or 'no summary'}{note}",
                      component=p.component)
```

- [ ] **Step 6: Prompts**

`seed_agent/agent/prompts/planner.md`:
```markdown
You plan one round of training-data work for AlayaWorld, a video world model that is
fine-tuned from the same released checkpoint at every node and scored on a 40-case
WBench proxy. Only the training data (and data-coupled training settings) may change.

You receive the lineage (each ancestor's score, per-metric and per-stratum results, the
data it trained on, its recipe and rationale) and a summary of the archive-wide clip pool.

Produce 1-3 concrete, testable data hypotheses (for example "more forward-walking indoor
clips with accurate poses should raise navigation_trajectory") and the actions to test
them: which clips from the pool to reuse or drop, what to fetch from Hugging Face, and how
to convert it into a standard format. Prefer small, attributable changes over broad ones:
one node is one experiment. Finish by calling submit_plan.
```

`seed_agent/agent/prompts/data_builder.md`:
```markdown
You build the training data for this node, using the kernel tools and local tools.

Kernel tools: data_query (the archive-wide clip pool), hf_search / hf_download (downloads land
under /workspace/staging/hf/), video_probe, data_ingest (candidates must be under
/workspace/staging/), data_commit, and job_status / job_wait / job_cancel for GPU jobs if any
generator tool is listed. Local tools: read_file, list_dir, write_file, edit_file, run_command
(ffmpeg, ffprobe, python), caption_clip, snap_timed_prompts. Your working directory is /workspace.

Rules that the kernel enforces (read the format rules in the context):
- Only the three standard formats. Video .mp4, >= 24 fps, >= 2.375 s, DISPLAY aspect 16:9
  within 2%, no display rotation. Crop or pad to 16:9 with ffmpeg; never stretch content.
- Moving-camera clips need poses/<id>.npz with cam_c2w [N,4,4], N = the mp4 frame count. If a
  dataset ships poses, convert them to camera-to-world OpenCV convention. Without poses, a clip
  can only be ingested as camera_motion "static", and only if the camera truly does not move.
- Every candidate needs provenance. hf_download returns a ready provenance record; for a clip
  you derive (cropped, trimmed, re-captioned), use {"kind": "derived", "from": [clip_ids],
  "transform": "<what you did>"} and set derived_from.
- Each dataset in a commit needs at least as many clips as training GPUs.

A tool that fails returns an error message; read it and adjust instead of repeating the call.
Work in small batches: fetch a little, convert, ingest, check the rejection reasons, adjust.
When you have a data commit that tests the plan, call submit_data_commit with its id and
short notes on what it contains and why.
```

`seed_agent/agent/prompts/recipe_writer.md`:
```markdown
You write the node's training recipe: values for tunable keys only (listed in the context
with their types and bounds). Everything else comes from the base recipe. Hyperparameter-only
changes are not allowed: the recipe exists to fit the data you built.

Checks the kernel runs, so get them right the first time:
- sample.height/sample.width must be an allowed resolution pair; lora.rank/lora.alpha an
  allowed pair.
- steps_per_epoch = floor(floor(epoch_windows / n_gpus) / optimizer.grad_accum_steps) must
  be >= 1, and optimizer.epochs * steps_per_epoch >= optimizer.max_steps.
- Training time grows with max_steps; a run over the wall-time limit fails.

If you are given failures from a previous check, fix exactly those. Call submit_recipe with
the recipe (tunable key -> value) and a short rationale tying the recipe to the data.
```

`seed_agent/agent/prompts/edit_planner.md`:
```markdown
You plan one improvement to this agent's own code. The agent builds training data for
AlayaWorld; each node's score says how well the data it built worked. You see the lineage
(scores, per-metric results, code diffs, edit components, data and recipes of every
ancestor) and the archive summary. You can read the code under /agent with read_file and
list_dir.

An agent version has five components, listed in the context under "components":
prompts, tools, harness, orchestration and knowledge. Choose exactly ONE component and one
focused change to it. One edit is one experiment: a small change to one component can be
attributed to the next score; a change spread over several cannot.

Find the most likely reason recent nodes did not improve (for example: rejected candidates
wasted the attempt, poses were missing so clips became static-only, the recipe failed the
step budget, the plan changed too many things at once, context was lost in long runs) and
choose the component where a fix belongs. Look at which components ancestors already
changed and what followed. If "retry" is set, a previous attempt failed verification: fix
that failure, staying with the previous attempt's component unless that is impossible.

Hard constraints for any change: agent/entry.py keeps top-level edit_self(ctx) and
improve_recipe(ctx), each with exactly one parameter; packages the base image lacks must be
listed in agent/requirements.txt.

Finish by calling submit_edit_plan.
```

`seed_agent/agent/prompts/coder.md`:
```markdown
You implement the edit plan you are given in the agent code under the current directory
(/agent), using read_file, list_dir, write_file, edit_file and run_command. Change only what
the plan needs, in the plan's component. Make the smallest correct change.

After editing, run `python -c "import agent.entry"` to check that it imports, and fix any
error before finishing. Record in agent/memory/ what you changed, why, and what result would
confirm or refute it. Finish with a one-paragraph summary of what you changed and why.
```

- [ ] **Step 7: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_agent.py tests/test_seed_harness.py -p no:cacheprovider`
Expected: PASS (18 seed-agent tests: 13 unit and 5 against the kernel services; plus the 26 harness tests).

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_agent.py -m docker -p no:cacheprovider`
Expected: PASS (the seed agent passes contract verification in real containers).

- [ ] **Step 8: Commit**

```bash
git add seed_agent tests/test_seed_agent.py
git commit -m "feat(seed): seed agent -- plain-Python orchestration of harness roles, one-component edits"
```

---
### Task 18: Real-component verification, log, merge

**Files:**
- Modify: `docs/superpowers/plans/verification-log.md` (append a Plan 2 section)
- Modify (only if implementation diverged): `docs/superpowers/specs/2026-09-17-autoresearcher-design.md`

- [ ] **Step 1: Default suite**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -p no:cacheprovider`
Expected: PASS, with no GPU, network or Docker needed.

- [ ] **Step 2: Docker suite, alone (no GPU jobs running)**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -m docker -p no:cacheprovider`
Expected: PASS. This covers the image, the isolation checks (§16.3 item 4), timeout and kill, contract verification on 7 fixtures, the end-to-end tool test, and the seed agent contract. Record the counts and durations.

Also confirm that the harness comparison (`tests/test_seed_harness.py`, part of Step 1) still passes, and record the Task 16 mutation counts.

- [ ] **Step 3: Negative control for the isolation test**

Temporarily change the runner's `--network none` to `--network bridge` and re-run `tests/test_sandbox_runner.py::test_isolation_holds_from_inside -m docker`. It must **fail** (the internet becomes reachable). Restore `--network none`, re-run, and confirm it passes. Record both runs. An isolation test that also passes with networking on proves nothing (verification-log finding 9, "Run the negative control").

- [ ] **Step 4: Live LLM run (only if `OPENAI_API_KEY` and `OPENAI_MODEL` are set)**

In a scratch run: ingest the 18 `WorldModel/data/examples` clips (staged copies, as in Plan 1 Task 14), start real services with `Upstream(OPENAI_BASE_URL, OPENAI_API_KEY, ...)`, and call `run_improve_recipe` on the seed agent for one node. Record the outcome whether or not it succeeds: `ok`, the error, the tool calls made, and the tokens and requests seen by the gateway. Success is not required; the point is to see a real model drive the real tools end to end. **If no key is set, record "live LLM run deferred: no OPENAI_API_KEY" instead. Do not mark this step passed.**

- [ ] **Step 5: Re-check the spec amendments**

Confirm the amendments made with this plan (§9.5 network, §10 tool names, §1.1 item 9 and §9.1–9.3 frameworks and edit components) still describe what was built. If implementation diverged anywhere else, amend the spec now and list it in the log.

- [ ] **Step 6: Write the verification log section**

Append `## Plan 2 — agent runtime` to `docs/superpowers/plans/verification-log.md`. Include the verified facts 1–18 this plan relied on, the suite and docker results, the harness comparison and its mutation counts, the isolation negative control, the live-LLM outcome (or its deferral), and every defect found during implementation with its fix.

- [ ] **Step 7: Merge and push (standing rule: `main` always current)**

```bash
git checkout main && git merge --ff-only feat/agent-runtime && git push origin main
```

Confirm that the commit `main` points to is the one that passed Steps 1–3, and delete the merged local branch.

### Task 19 (addendum): OpenAI-compatible Chat Completions, gateway-enforced reasoning effort

Added 2026-09-24, after `.env` was configured with a provider that has no Responses API. Agent LLMs may be any OpenAI-compatible API (the purpose of `OPENAI_BASE_URL`); Chat Completions is the wire format because it is the most widely supported endpoint. The `/v1/responses` gateway route stays for clients that use it.

**Files:** `contract/ar_contract/client.py`, `kernel/ar_kernel/gateway/{app,mock,store}.py`, tests (`test_contract_package`, `test_gateway_app`, `test_gateway_store`, `test_seed_agent`), this plan, the spec (§1.1 item 9, §2.1).

- `chat_model()` returns `ReasoningChatOpenAI` (a small `ChatOpenAI` subclass in `client.py`) with `use_responses_api=False`, both clients on the socket, `max_retries=0`. It copies a reply's `message.reasoning_content` into `additional_kwargs["reasoning_content"]` and sends it back on that assistant message (some providers require it when tools are sent; langchain-openai 1.6.2 drops it both ways). It never invents the field: a provider that does not return it sees an unchanged request. The client sets no reasoning effort.
- `Upstream(effort=None)`: when set, the gateway overrides the forwarded body's effort (`reasoning_effort` for Chat Completions, `reasoning: {..., "effort"}` for Responses) **before** `store.begin`, so telemetry records exactly what was forwarded. Mock calls are not modified. `Upstream.from_env(environ, **kwargs)` reads `OPENAI_BASE_URL` (default `https://api.openai.com/v1`), `OPENAI_API_KEY` and `OPENAI_EFFORT` (empty → `None`). The base URL is used exactly as given; the endpoint path is appended.
- `MockBook.chat_response(model, output)` renders the same script items (`message`, `function_call`) as a `chat.completion`; the gateway's mock path uses it on `/v1/chat/completions` and `MockBook.response` on `/v1/responses`.
- `CallStore` compares Chat Completions assistant messages by what a client sends back (content or `None`, tool-call ids, names and parsed arguments, `reasoning_content`), because upstream replies carry fields the client drops (`refusal`, `annotations`, tool-call `index`) and may space the arguments JSON differently.
- Context window: `agents.context_window_tokens` stays at 128000 even for models with larger windows; it doubles as a cost cap (earlier compaction, cheaper calls).
- Verification: unit tests for the round trip (stub transport), effort override on both endpoints, `from_env`, the chat mock renderer and chat prefix linking of a real-style reply; the seed-agent flows run on the Chat Completions path. One capped live check (two turns, `max_tokens` ≤ 300) against the configured `.env` endpoint; see the Task 19 report.

---

## Interface contract for Plan 4 (the loop)

Plan 4's tasks will be written after Plan 3. Plan 4 sequences these Plan 2 pieces and owns everything listed under "Plan 4 must".

**Plan 4 consumes:**

| Piece | From | Use |
|---|---|---|
| `RunServices`, `socket_dir_for` | Task 6 | One instance per run: real gateway plus tool server. Start at run start, stop on exit and on force stop. |
| `create_gateway_app(... upstream=Upstream.from_env(os.environ, timeout_s=..., retries=...), allowed_models=gateway.model_allowlist ∪ {OPENAI_MODEL})` | Tasks 5, 19 | Real LLM route. With no key: `upstream=None`, so every agent call is served by mocks. |
| `DataTools`, `HfTools`, `JobQueue` + `register_*` | Tasks 7–9 | The real tool server. Plan 3 registers generator backends on the same `JobQueue`. `HfTools(cfg, private_dir)` takes a kernel-only download dir under the run dir (e.g. `run_dir / "hf_tmp"`) that is never mounted into a container (final review). |
| `gpu_lock` shared by `DataTools`, `JobQueue` and training/eval | Tasks 7, 9 | GPU phases never overlap (spec 7). Plan 4 holds it for precache, train, merge, render and eval. |
| `AgentsRepo` | Task 12 | `init(seed_agent)` at bootstrap; `set_ref(branch_ref(node), commit)` for the passing attempt. |
| `ContractHarness`, `verify_contract` | Task 14 | Step 4 of the cycle. `report.to_retry()` becomes the next `edit_self` attempt's `retry`. |
| `run_edit_self`, `run_improve_recipe`, `attempt_dirs` | Task 15 | Steps 3 and 5. Retries pass `base_commit=<failed attempt commit>` and `previous_workspace=<failed attempt workspace>`. `run_edit_self` returns `commit=None` when the base commit cannot be checked out or the attempt tree cannot be committed (final review); the retry must then keep the previous `base_commit`. |
| Context conventions | Task 13 | Plan 4 **writes** `nodes/<n>/recipe.yaml`, `nodes/<n>/rationale.md`, `nodes/<n>/edit.json` (the passing `edit_self` result, including `component`) and `nodes/<n>/eval/aggregates.json`. |

**Plan 4 must:**
1. **Liveness (spec 14.5).** Plan 2 enforces only the hard cap (4 x soft). Plan 4 wraps the runner with the soft timeout plus a probe window, using gateway and tool events for the caller's token, container CPU from `RunResult.stats`, and workspace changes. **GPU utilization is not a liveness signal** (verification-log finding 2).
2. **Per-attempt training directories.** Gate views, `train_config.yaml`, `train/` and `dataset_cache` go under `nodes/<n>/attempts/<phase>-<k>/`, so a failed retry can never supply the checkpoint (Plan 1 review, deferred).
3. **Termination.** Force stop and resume discard must `docker kill` containers by the prefix `ar-<run_id>-`, call `JobQueue.shutdown()` and `RunServices.stop()`, kill kernel process groups, then confirm GPU memory is released and sweep `_megasam_tmp` before the next phase (verification-log findings 6–7).
4. **Selection versus noise.** The proxy aggregate has a noise floor of about 4.4e-4 from a single re-run. Measure it properly (repeat evaluation of one checkpoint) before long unattended runs, and use it in parent selection (verification-log finding 5).
5. **Upstream outage.** Watch the `llm.response` error rate. Upstream unavailable for more than `gateway.upstream_outage_pause_min` pauses the loop without charging the node (spec 14.2).
6. **Edit components.** Store the passing attempt's `EditResult.component` with the node (the nodes table and `edit.json`) and show it in the dashboard lineage. It is recorded, never enforced (spec 9.1.1).
7. **Deferred Plan 1 items:** `score_node` try/finally and rank/alpha from the node's resolved config; schema columns `recipe_path` and `attempt_counts_json`; `resolve_gpus` visibility (14.6); run-relative paths if resume-after-move matters.
8. **Config to constructor parameters.** Plan 2's services read no config themselves. Plan 4 builds them from `KernelConfig` and passes `gateway.upstream_timeout_s` and `gateway.upstream_retries` to `Upstream.from_env(os.environ, timeout_s=, retries=)`, `gateway.model_allowlist` (∪ `{OPENAI_MODEL}`) to `create_gateway_app(allowed_models=)`, and `tools.job_wait_max_s` to `JobQueue(wait_cap_s=)`.
9. **Shutdown errors.** `JobQueue.shutdown()` raises `RuntimeError` when the worker thread outlives its join deadline (a job may still hold a GPU). Plan 4's shutdown and force-stop paths must catch it, still run `RunServices.stop()` and the container and process-group kills, and then confirm the GPUs are free before continuing.
10. **Live LLM run.** Task 19 passed a capped live check through the gateway (783 tokens). The full Task 18 Step 4 run (a live `improve_recipe`) is still owed; unless it is recorded in the verification log before merge, Plan 4 must run it, as Task 18 Step 4 describes, before the first unattended run.

## Self-review

- **Spec coverage.** §9.1 → Tasks 16–17 (§9.1.2 harness → Task 16; §9.1.1 edit components → Tasks 3 and 17); §9.2 → Task 3; §9.3 → Tasks 3 and 13; §9.4 → Task 14; §9.5 → Tasks 10–11 (network amended); §10 tools → Tasks 7–9 (generators → Plan 3); §13.1–13.3 gateway and tool capture → Tasks 1, 4–6; §5.2 → Task 12; §5.5 aspect → Task 2; §16.1 gateway / contract / job API → Tasks 4, 5, 9, 14; §16.3 item 4 → Tasks 11 and 18. Left to Plan 4, by the contract above: §7.2 sequencing, §12, §13.4, §14.3–14.6 and liveness.
- **Placeholders.** None. The one open outcome, the live LLM run, has an explicit deferral rule.
- **Tested before writing.** The Task 16 harness (all three stages, their test counts and the mutation counts) and the Task 17 seed agent (both entry points end to end over Unix sockets against a scripted gateway and tool server, including argument errors, unknown tools and a failing kernel tool) were run as prototypes on this machine before being written into the plan.
- **External review (2026-09-23).** Seven reported problems were checked against this plan. Fixed: non-JSON upstream bodies crashed the gateway (Task 5; reproduced, and the old line fails the new tests); `chat_model()` had no sync client on the socket (Task 3, fact 18; reproduced); staged files were missing from the workspace diff (Task 15). Fixed although the stated impact did not hold: the contract harness now cancels the caller's jobs (Task 14; its queue has no backends and a private lock, so nothing could be stranded yet); images are estimated at a fixed size (Task 16; the seed never puts images in a role's history, but an edited agent could); `edit_file` refuses non-UTF-8 files with a clear message (Task 17; the reported crash could not happen, because the harness reports tool errors, and switching `edit_file` to `errors="replace"` as proposed would have silently corrupted such files). Not changed: `Upstream` does not append `/v1` to the base URL, because the OpenAI SDK convention and spec §2 put the version in `OPENAI_BASE_URL`, and rewriting it would break providers with other version paths; it now defaults to `https://api.openai.com/v1` when unset (Task 5).
- **Type consistency.** `Caller` fields are fixed in Task 4 and used unchanged in Tasks 5–9, 14 and 15. `PhaseOutcome` and `PhaseEnv` are defined once, in Task 15. Tool names are identical in `register_*`, the Task 14 mock objects, `run_improve_recipe`'s tool list and the seed prompts.
