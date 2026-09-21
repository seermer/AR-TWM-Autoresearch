# AutoResearcher Agent Runtime Implementation Plan (Plan 2 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an agent version in a sandbox, where it talks to an LLM only through a recording gateway and to the kernel only through privileged MCP tools, produces data commits and a recipe, and is verified against the fixed `edit_self` / `improve_recipe` contract. Every step is recorded.

**Architecture:** The kernel starts two HTTP services on Unix domain sockets in a per-run socket directory: a recording OpenAI-compatible **gateway** and an MCP **tool server**. Each agent call runs in a fresh Docker container with **no network** (`--network none`), running as the host user, with the socket directory mounted. The fixed `ar_contract` package (mounted read-only) gives agent code preconfigured clients and runs the entry point. Agent code lives in a per-run git repo (`agents.git`). The seed agent is built on LangGraph plus the OpenAI Agents SDK.

**Tech Stack:** Python 3.12 (`autoresearcher` env); FastAPI and uvicorn (UDS); `mcp==2.2.0` (`MCPServer`, `httpx2` client transport); `openai-agents==0.22.3`; `langgraph==1.2.11` and `langgraph-checkpoint-sqlite==3.1.1`; `huggingface_hub`; `zstandard`; the Docker 27.3.1 CLI (no Python Docker library); the git CLI.

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

## Verified facts this plan relies on (pre-plan spikes, 2026-09-21)

Each of these was checked on this machine before the plan was written. Several differ from what the library documentation or older versions suggest, so do not "fix" them back.

1. **MCP 2.x** renamed `FastMCP` to `mcp.server.mcpserver.MCPServer`. The server app is `MCPServer.streamable_http_app(...)`.
2. **The MCP 2.x client uses `httpx2`, not `httpx`.** The Agents SDK's `MCPServerStreamableHttp(params={"httpx_client_factory": f})` requires `f(headers, timeout, auth)` to return an `httpx2.AsyncClient`. It raises `UserError: MCP Python SDK v2 requires httpx_client_factory to return an httpx2.AsyncClient` otherwise.
3. **Host header over UDS.** The MCP server's DNS-rebinding guard rejects `Host: localhost` (no port) with **421**. Pass `transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=["localhost"], allowed_origins=[])`, and address both services as `http://localhost/...` over the socket.
4. **Session idle timeout.** `streamable_http_app` defaults to `session_idle_timeout=1800`, which silently kills the session of an agent that makes no tool call for 30 min. Use `session_idle_timeout=None`.
5. **Client timeout.** `MCPServerStreamableHttp` defaults to `client_session_timeout_seconds=5`, but `data.ingest` runs WorldModel's checker per clip and takes longer. The contract client sets 900 s.
6. **Mock Responses API.** A scripted `/v1/responses` server works with the Agents SDK when each response carries `id, object="response", created_at, status="completed", model, output, parallel_tool_calls, tool_choice, tools, usage{input_tokens, output_tokens, total_tokens, input_tokens_details{cached_tokens}, output_tokens_details{reasoning_tokens}}`. A tool call is an output item `{"type":"function_call","id","call_id","name","arguments","status":"completed"}`. The next request's `input` then contains a `function_call_output` item.
7. **Sandbox networking.** A Docker `--internal` network blocks the internet and DNS, **but the host stays reachable at the bridge IP**: SSH on 22 and another host service were open from inside. `--network none` plus the UDS socket directory gives complete isolation. Verified: the agent loop completes, host ports are unreachable, the internet is blocked, and files are owned by the host uid.
8. **ffmpeg 6.1.1** writes rotation with `ffmpeg -display_rotation 90 -i in.mp4 -c copy out.mp4`, and ffprobe reports it as `side_data_list:[{"rotation":90}]`. A 552x414 clip with `sample_aspect_ratio` `4:3` displays at 16:9.
9. **Packages.** None of the agent-layer packages were installed; `pip check` is clean after installing the pinned set (Task 1). `openai-agents` requires `mcp<3,>=1.19`.
10. **Caller identity in a tool.** A tool that declares `ctx: Context` reads the caller's header as `ctx.request_context.request.headers["authorization"]`. Verified end to end over streamable HTTP.
11. **Tool names.** OpenAI function names must match `^[a-zA-Z0-9_-]+$`, and the Agents SDK forwards MCP tool names as function names. The spec's dotted names (`data.ingest`) would be rejected upstream, so tools are registered as `data_ingest` and so on.
12. **Socket path length.** `AF_UNIX` paths are capped at 107 bytes. A socket under `runs/<run_id>/sock/` is 125 bytes, and `bind` fails with "AF_UNIX path too long". Sockets live in a short per-run directory under the system temp dir.
13. **Async graphs.** One MCP connection has to live inside one event loop, so the seed graphs are async. `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver` (with `aiosqlite` 0.22.1) checkpoints an async `StateGraph` correctly; the sync `SqliteSaver` refuses async calls.

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

Both amendments are applied to the spec in the same commit as this plan.

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
  __init__.py  models.py  client.py  run.py  tracing.py
seed_agent/agent/              CREATE  initial agent code (the root node's commit)
  entry.py  requirements.txt  graphs/meta.py  graphs/task.py  agents/*.py
  prompts/*.md  tools/*.py  memory/README.md
docker/agent.Dockerfile        CREATE
configs/kernel.yaml            MODIFY  gateway, tools, sandbox, timeouts, generators blocks
tests/                         CREATE  one test file per component (named per task)
```

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
  - Config keys: `gateway.*`, `sandbox.*`, `timeouts.*`, `generators.*`, `tools.hf_download_max_bytes`.

- [ ] **Step 1: Pin the dependencies and make `ar_contract` importable**

In `pyproject.toml`, replace the `dependencies` line and the package-find block:

```toml
dependencies = [
    "pyyaml>=6.0", "numpy>=1.26", "pillow>=10.0", "imagehash>=4.3",
    "fastapi>=0.141", "uvicorn>=0.53", "httpx>=0.28", "pydantic>=2.12",
    "mcp==2.2.0", "openai>=3.0,<4", "openai-agents==0.22.3",
    "langgraph==1.2.11", "langgraph-checkpoint-sqlite==3.1.1",
    "huggingface_hub>=1.32", "zstandard>=0.25",
]
```

```toml
[tool.setuptools.packages.find]
where = ["kernel", "contract"]
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

Run: `conda run --no-capture-output -n autoresearcher python -m pip install -e . && conda run --no-capture-output -n autoresearcher python -m pip check`
Expected: `No broken requirements found.`

- [ ] **Step 2: Add the config blocks**

Append to `configs/kernel.yaml`. Keep the existing `tools:` block's `job_wait_max_s` and add the new key under it:

```yaml
gateway:
  model_allowlist: []            # OPENAI_MODEL from the environment is always allowed too
  upstream_timeout_s: 600
  upstream_retries: 5            # 429/5xx, exponential backoff 1, 2, 4, ... s
  upstream_outage_pause_min: 15
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

Kernel-owned and mounted read-only at `/ar_contract` in every container (spec §9.2). It holds the schemas the kernel validates results against, clients preconfigured for the sockets (facts 2–5), and the runner. This is the only way agent code gets a correctly configured LLM client and MCP connection, so the socket details live here once.

**Files:**
- Create: `contract/ar_contract/__init__.py`, `models.py`, `client.py`, `tracing.py`, `run.py`
- Test: `tests/test_contract_package.py`

**Interfaces:**
- Produces (imported by agent code, the kernel's context builder and contract verification):
  - `ar_contract.models`: `EditContext`, `EditResult`, `RecipeContext`, `RecipeResult` (pydantic v2), and `CONTEXT_MODELS = {"edit_self": EditContext, "improve_recipe": RecipeContext}`, `RESULT_MODELS = {"edit_self": EditResult, "improve_recipe": RecipeResult}`.
  - `ar_contract.client`: `SOCKET_DIR` (env `AR_SOCKET_DIR`, default `/run/ar`), `token() -> str` (env `AR_TOKEN`), `openai_client() -> openai.AsyncOpenAI`, `configure_agents_sdk() -> None`, `mcp_tools() -> agents.mcp.MCPServerStreamableHttp`.
  - `ar_contract.run.main(argv) -> int`, run as `python -m ar_contract.run <edit_self|improve_recipe>`. It reads `$AR_CONTEXT_DIR/context.json` (default `/context`), imports `agent.entry` from `$AR_AGENT_DIR` (default `/agent`), calls the entry point (sync or async), validates the result, and writes `$AR_WORKSPACE/result.json` (default `/workspace`) as `{"ok": true, "result": {...}}` or `{"ok": false, "error": "...", "traceback": "..."}`. Exit code 0 means ok.
  - `ar_contract.tracing.JsonlTraceProcessor(path)`: appends exported Agents SDK traces/spans to `$AR_WORKSPACE/trace.jsonl`.

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
        monkeypatch.setenv("AR_SKIP_SDK_SETUP", "1")   # no sockets in unit tests
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
    assert json.loads(out.read_text()) == {"ok": True, "result": {"summary": "attempt 1"}}


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


def test_clients_speak_over_the_socket_directory(monkeypatch, tmp_path):
    """Facts 2-5: gateway via httpx over UDS, MCP via an httpx2 UDS client, localhost host,
    long timeouts. Built without connecting."""
    monkeypatch.setenv("AR_SOCKET_DIR", str(tmp_path))
    monkeypatch.setenv("AR_TOKEN", "tok-abc")
    from importlib import reload
    import ar_contract.client as client
    reload(client)
    oc = client.openai_client()
    assert str(oc.base_url).startswith("http://localhost/v1")
    assert oc.api_key == "tok-abc"
    server = client.mcp_tools()
    assert server.params["url"] == "http://localhost/mcp"
    assert server.params["headers"]["Authorization"] == "Bearer tok-abc"
    httpx2_client = server.params["httpx_client_factory"]()
    assert type(httpx2_client).__module__.startswith("httpx2")
    assert server.client_session_timeout_seconds >= 900
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

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
- the OpenAI client uses httpx; the MCP 2.x client REQUIRES an httpx2 client;
- the host is 'localhost' (the MCP server's rebinding guard rejects anything else);
- MCP calls such as data.ingest run the dataset checker and take far longer than
  the SDK's 5 s default session timeout.
"""
from __future__ import annotations

import os

import httpx
import httpx2

SOCKET_DIR = os.environ.get("AR_SOCKET_DIR", "/run/ar")
MCP_TIMEOUT_S = 900.0


def token() -> str:
    return os.environ.get("AR_TOKEN", "")


def openai_client():
    from openai import AsyncOpenAI
    transport = httpx.AsyncHTTPTransport(uds=os.path.join(SOCKET_DIR, "gateway.sock"))
    return AsyncOpenAI(base_url="http://localhost/v1", api_key=token(),
                       http_client=httpx.AsyncClient(transport=transport, timeout=900.0),
                       max_retries=0)        # the gateway owns upstream retries


def _httpx2_factory(headers=None, timeout=None, auth=None):
    transport = httpx2.AsyncHTTPTransport(uds=os.path.join(SOCKET_DIR, "tools.sock"))
    return httpx2.AsyncClient(transport=transport, headers=headers,
                              timeout=timeout if timeout is not None else MCP_TIMEOUT_S, auth=auth)


def mcp_tools():
    from agents.mcp import MCPServerStreamableHttp
    return MCPServerStreamableHttp(
        params={"url": "http://localhost/mcp",
                "headers": {"Authorization": f"Bearer {token()}"},
                "timeout": MCP_TIMEOUT_S,
                "httpx_client_factory": _httpx2_factory},
        name="ar-kernel-tools",
        client_session_timeout_seconds=MCP_TIMEOUT_S,
        cache_tools_list=True,
    )


def configure_agents_sdk() -> None:
    """Route every Agents SDK call through the gateway and keep traces local."""
    from agents import set_default_openai_api, set_default_openai_client, set_trace_processors
    from .tracing import JsonlTraceProcessor
    set_default_openai_client(openai_client(), use_for_tracing=False)
    set_default_openai_api("responses")
    # Replaces the default processor, which would upload traces to OpenAI (spec 13.2).
    workspace = os.environ.get("AR_WORKSPACE", "/workspace")
    set_trace_processors([JsonlTraceProcessor(os.path.join(workspace, "trace.jsonl"))])
```

- [ ] **Step 5: Implement tracing and the runner**

```python
# contract/ar_contract/tracing.py
"""Supplementary Agents SDK spans (agent names, handoffs, guardrails), written locally.
The kernel's gateway already records every LLM call; this adds SDK structure."""
from __future__ import annotations

import json
import threading

from agents.tracing import TracingProcessor


class JsonlTraceProcessor(TracingProcessor):
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()

    def _write(self, kind: str, item) -> None:
        exported = item.export() if hasattr(item, "export") else None
        if exported is None:
            return
        with self._lock, open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, **exported}, default=str) + "\n")

    def on_trace_start(self, trace) -> None:
        self._write("trace_start", trace)

    def on_trace_end(self, trace) -> None:
        self._write("trace_end", trace)

    def on_span_start(self, span) -> None:
        pass                                   # spans are complete only at the end

    def on_span_end(self, span) -> None:
        self._write("span", span)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass
```

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
        if not os.environ.get("AR_SKIP_SDK_SETUP"):
            from .client import configure_agents_sdk
            configure_agents_sdk()
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
Expected: PASS (8 tests).

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
Expected: PASS (13 tests counting the parametrized ones).

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
    """How the Agents SDK actually continues a run: it resends the prior input and
    output items, plus the tool output (verified fact 6)."""
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
(how the Agents SDK continues a run). Prefix matching only looks at calls from the
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
  - `gateway.app.Upstream(base_url, api_key, *, timeout_s, retries, transport=None, sleep=asyncio.sleep)`: `await .post(path, body) -> (status, body, attempts)`.
  - `gateway.app.create_gateway_app(*, registry, store, allowed_models: set[str], upstream: Upstream | None, mocks: MockBook) -> FastAPI`. It serves `POST /v1/responses` and `POST /v1/chat/completions`.

Behavior, per request: token → Caller, else **401** · `stream: true` → **400** (the gateway records complete responses; agent code uses `Runner.run`, not streaming) · model not allowed → **403** · `store.begin` (a `TelemetryError` → **500**, nothing forwarded) · mock if `caller.mock_script` or `upstream is None`, else upstream · `store.end` (a `TelemetryError` → **500**) · return the upstream status and body.

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
"""Scripted Responses API outputs. The envelope is the minimum the Agents SDK
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


class Upstream:
    def __init__(self, base_url: str, api_key: str, *, timeout_s: float, retries: int,
                 transport: httpx.AsyncBaseTransport | None = None, sleep=asyncio.sleep) -> None:
        self._base = base_url.rstrip("/")
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
                    payload = r.json() if r.content else {}
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
                                                      "use non-streaming calls (Runner.run)"}},
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
        if caller.mock_script or upstream is None:
            status, payload, attempts = 200, MockBook.response(
                body.get("model", ""), mocks.next(caller.mock_script or "smoke", caller.token)), 1
        else:
            status, payload, attempts = await upstream.post(upstream_path, body)
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
Expected: PASS (11 tests).

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
  - `tools.server.ToolError(Exception)`: the message the agent sees.
  - `tools.server.ToolKit(registry, recorder)`: `await .call(ctx, name: str, args: dict, fn: Callable[[Caller], Any]) -> Any`. Resolves the caller, records, runs `fn(caller)` in a worker thread, and re-raises every failure as `ToolError`.
  - `tools.server.build_tool_app(mcp: MCPServer) -> Starlette`, with the transport settings from facts 3–4.
  - `tools.server.new_mcp() -> MCPServer`.
  - `services.socket_dir_for(run_dir: Path) -> Path`, a short per-run directory under the system temp dir (fact 12: Unix socket paths are capped at 107 bytes).
  - `services.RunServices`: `.start(gateway_app, tools_app) -> None`, `.stop() -> None`, `.socket_dir: Path`. It binds `gateway.sock` and `tools.sock` with mode 0600, in a directory with mode 0700.

Tool names use underscores: OpenAI function names must match `^[a-zA-Z0-9_-]+$`, and the SDK forwards MCP tool names as function names. So spec §10's `data.ingest` is registered as `data_ingest`, and so on.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_server.py
"""Exercise the tool server the way a container does: over a Unix socket, with
the ar_contract client configuration."""
import asyncio
import json
import os

import httpx2
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
    from agents.mcp import MCPServerStreamableHttp
    def factory(headers=None, timeout=None, auth=None):
        return httpx2.AsyncClient(transport=httpx2.AsyncHTTPTransport(uds=str(sock_dir / "tools.sock")),
                                  headers=headers, timeout=timeout, auth=auth)
    async with MCPServerStreamableHttp(
            params={"url": "http://localhost/mcp", "headers": {"Authorization": f"Bearer {token}"},
                    "httpx_client_factory": factory}, client_session_timeout_seconds=30) as s:
        return await s.call_tool(tool, args)


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
    assert not result.isError
    assert json.loads(result.content[0].text) == {"node": "n7", "word": "hi"}
    kinds = [e["type"] for e in rec.read_events("n7")]
    assert kinds == ["tool.call", "tool.result"]
    assert all(e["component"] == "tools" for e in rec.read_events("n7"))


def test_tool_error_reaches_the_agent_as_an_error_result(live):
    rec, _, services, caller = live
    result = asyncio.run(_call(services.socket_dir, caller.token, "always_fails", {}))
    assert result.isError and "too large" in result.content[0].text
    assert [e["type"] for e in rec.read_events("n7")][-1] == "tool.error"


def test_unexpected_kernel_exception_is_contained(live):
    """A bug in a tool must come back as a tool error, never take the server down."""
    rec, _, services, caller = live
    result = asyncio.run(_call(services.socket_dir, caller.token, "kernel_bug", {}))
    assert result.isError and "ZeroDivisionError" in result.content[0].text
    err = [e for e in rec.read_events("n7") if e["type"] == "tool.error"][0]
    assert "Traceback" in rec.load_payload(err["payload"])["traceback"]
    # the server still answers afterwards
    assert not asyncio.run(_call(services.socket_dir, caller.token, "echo_caller", {"word": "x"})).isError


def test_unknown_token_is_refused(live):
    _, _, services, _ = live
    result = asyncio.run(_call(services.socket_dir, "ar-forged", "echo_caller", {"word": "hi"}))
    assert result.isError and "token" in result.content[0].text


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
from mcp.server.transport_security import TransportSecuritySettings

from .context import bearer


class ToolError(Exception):
    """A failure the agent should see and may act on."""


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
Expected: PASS (6 tests).

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
  - `hf_tools.HfTools(cfg, api=None, snapshot=None)`. `api` defaults to `huggingface_hub.HfApi()` and `snapshot` to `huggingface_hub.snapshot_download`; both are injectable for tests.
  - `.search(caller, query, kind="dataset", limit=20) -> list[dict]`.
  - `.download(caller, repo, revision, patterns, max_bytes=None) -> dict`.
  - `hf_tools.register_hf_tools(mcp, kit, tools)` registers `hf_search` and `hf_download`.
- The kernel process has network access and `HF_TOKEN` from the shell; the container has neither.

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
Expected: PASS (5 tests).

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
  - `jobs.run_cancellable(env, args, *, cwd, cancel, extra_env=None, log_path, poll_s=1.0) -> int`. It starts the command in its own session and kills the **group** when `cancel` is set. It returns the exit code, or -15 when cancelled.
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
Expected: PASS (9 tests).

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
      "openai-agents==0.22.3" "mcp==2.2.0" "httpx==0.28.1" \
      "langgraph==1.2.11" "langgraph-checkpoint-sqlite==3.1.1" \
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
         "import agents, mcp, httpx, httpx2, langgraph, cv2, numpy, PIL; "
         "from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver; print('ok')"],
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
            "-v", f"{mounts.sockets}:/run/ar:rw"]
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
Expected: PASS (2 unit tests).

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_sandbox_runner.py -m docker -p no:cacheprovider`
Expected: PASS (3 docker tests, including the isolation test).

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/sandbox/runner.py tests/test_sandbox_runner.py
git commit -m "feat(sandbox): isolated container runner -- no network, host uid, caps, kill, stats, diff"
```

---

### Task 12: The agent code repository

`runs/<run>/agents.git` (spec 5.2): the root commit is `seed_agent/`; a child's code lives on `node/<id>`; every attempt, failed ones included, is committed under `refs/attempts/<node>/<k>`. This uses git plumbing with a private index, so the repository never needs a checked-out working tree.

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
`runs/<run>/nodes/<n>/rationale.md`, and
`runs/<run>/nodes/<n>/eval/aggregates.json` (the `aggregates()` output).

**Files:**
- Create: `kernel/ar_kernel/context_bundle.py`
- Test: `tests/test_context_bundle.py`

**Interfaces:**
- Consumes: `NodeStore`, `CommitStore`, `ClipStore`, `BlobStore`, `open_db`; `AgentsRepo.diff_stats` (Task 12); `TUNABLE_KEYS`, `RECIPE_RULES` (Plan 1 review); `ar_contract.models` (Task 3); `data_tools.scores_by_clip` (Task 7).
- Produces:
  - `context_bundle.lineage(conn, run_dir, repo, node_id) -> list[dict]`: root first, ending at `node_id`. Each entry has `node_id, status, score, metrics, data (per-dataset format/weight/clip count), recipe, rationale, aggregates, code_diff_stats`.
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
    return conn, repo, tmp_path


def test_lineage_is_root_first_and_carries_artifacts(world):
    conn, repo, run = world
    lin = lineage(conn, run, repo, "root")
    assert [e["node_id"] for e in lin] == ["root"]
    assert lin[0]["score"] == 0.78 and lin[0]["aggregates"] == {"category": {"Nature": 0.8}}
    assert lin[0]["rationale"] == "released checkpoint"


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
from pathlib import Path

import pytest

from ar_kernel.config import KernelConfig
from ar_kernel.contract.verify import ContractHarness, static_check, verify_contract
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
                            command=command, env={"AR_TOKEN": caller.token, "AR_DEFAULT_MODEL": MOCK_MODEL},
                            cpus=4, memory_gb=8, timeout_s=timeout_s, recorder=recorder,
                            node=node, phase="contract", attempt=attempt)
        finally:
            harness.registry.revoke(caller.token)
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
Expected: PASS (7 static-check cases).

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
  - `agent_phase.run_edit_self(env, *, conn, node, parent_id, base_commit, attempt, max_attempts, retry, nodes_remaining, mock_script=None, dry_run=False) -> PhaseOutcome`. `base_commit` is the parent's code on attempt 1 and the failed attempt's commit on a retry (spec 7.2).
  - `agent_phase.run_improve_recipe(env, *, conn, node, parent_id, agent_commit, attempt, max_attempts, retry, nodes_remaining, previous_workspace=None, mock_script=None, dry_run=False) -> PhaseOutcome`. `previous_workspace` is copied in on a retry (spec 7.2).
- `improve_recipe` mounts `/agent` read-only: only `edit_self` changes code. A `RecipeResult` naming a data commit that does not exist is `ok=False`.

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
    def __init__(self, result, edit=None, exit_code=0, timed_out=False):
        self.result, self.edit, self.exit_code, self.timed_out = result, edit, exit_code, timed_out
        self.calls = []

    def __call__(self, *, mounts, env, **kw):
        self.calls.append({"mounts": mounts, "env": env, **kw})
        if self.edit:
            (mounts.agent / "agent" / "entry.py").write_text(self.edit)
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
    before = {"agent": snapshot(dirs["agent"], True), "workspace": snapshot(dirs["workspace"], False)}
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
                 "AR_PHASE": phase, "AR_ATTEMPT": str(attempt)},
            cpus=env.cfg.get("sandbox.cpus"), memory_gb=env.cfg.get("sandbox.memory_gb"),
            timeout_s=4 * soft,                 # hard cap (spec 14.5); liveness is Plan 4
            recorder=env.recorder, node=node, phase=phase, attempt=attempt)
    finally:
        env.registry.revoke(caller.token)
        env.queue.cancel_for_token(caller.token)     # an ended phase must not keep the GPUs
    out_file = dirs["workspace"] / "result.json"
    body = json.loads(out_file.read_text()) if out_file.exists() else None
    diffs = {"agent": diff(before["agent"], snapshot(dirs["agent"], True)),
             "workspace": diff(before["workspace"], snapshot(dirs["workspace"], False))}
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
Expected: PASS (7 tests).

- [ ] **Step 5: Add a docker end-to-end test: real container, real gateway (mock script), real tools**

Create the fixture agent `tests/fixtures/agents/tool_user/agent/entry.py`:

```python
"""Uses the real MCP tools from inside the container: queries the pool and commits it."""
import asyncio
import json

from ar_contract.client import mcp_tools
from ar_contract.models import EditResult, RecipeResult


def edit_self(ctx):
    return EditResult(summary="unused")


async def improve_recipe(ctx):
    async with mcp_tools() as tools:
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
### Task 16: The seed agent

The root node's code (spec 9.1): the agent every later version descends from. `improve_recipe` runs an async LangGraph **task** graph (plan → build data → write recipe → `recipe_check` → revise, ≤ 3 rounds). `edit_self` runs a **meta** graph (analyze lineage → implement edits with file tools → self-test → finalize, ≤ 2 rounds). Each LLM step is an OpenAI Agents SDK `Agent`. Everything runs in one event loop with one MCP connection (verified fact 13: async graphs with `AsyncSqliteSaver`).

The seed's job is to be a **correct, working starting point**, not a strong researcher. The loop improves it. Keep it simple, deterministic where it can be (the recipe check and the self-test are code, not LLM calls), and honest about limits in its prompts.

**Files:**
- Create: `seed_agent/.gitignore`, `seed_agent/agent/__init__.py` (empty), `seed_agent/agent/requirements.txt`, `seed_agent/agent/settings.py`, `seed_agent/agent/entry.py`
- Create: `seed_agent/agent/tools/__init__.py` (empty), `seed_agent/agent/tools/files.py`, `seed_agent/agent/tools/media.py`
- Create: `seed_agent/agent/agents/__init__.py` (empty), `seed_agent/agent/agents/definitions.py`
- Create: `seed_agent/agent/graphs/__init__.py` (empty), `seed_agent/agent/graphs/task.py`, `seed_agent/agent/graphs/meta.py`
- Create: `seed_agent/agent/prompts/{planner,data_builder,recipe_writer,analyst,coder}.md`
- Create: `seed_agent/agent/memory/README.md`
- Test: `tests/test_seed_agent.py`

**Interfaces:**
- Consumes (inside the container): `ar_contract.client.mcp_tools`, `ar_contract.client.openai_client`, `ar_contract.models.*` (Task 3); the kernel tools by name (Tasks 7–9).
- Produces: `agent.entry.edit_self(ctx) -> EditResult` and `agent.entry.improve_recipe(ctx) -> RecipeResult`, both `async`. It also exposes pure helpers for tests: `agent.tools.media.snap_segments(segments, duration)` and `agent.tools.files.resolve_inside(root, path)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_seed_agent.py
import json
import os
import sys
from pathlib import Path

import pytest

SEED = Path(__file__).resolve().parents[1] / "seed_agent"
sys.path.insert(0, str(SEED))


def test_snap_segments_puts_boundaries_on_rounds():
    from agent.tools.media import snap_segments
    segs = [{"time_range_s": [0.0, 3.1], "prompt": "walk"}, {"time_range_s": [3.1, 8.0], "prompt": "turn"}]
    out = snap_segments(segs, duration=8.0)
    boundary = out[0]["time_range_s"][1]
    k = (boundary - 25 / 24) / (32 / 24)
    assert abs(k - round(k)) < 1e-9                         # on 25/24 + k*32/24
    assert out[0]["time_range_s"][0] == 0.0 and out[-1]["time_range_s"][1] == 8.0
    assert out[1]["time_range_s"][0] == boundary            # contiguous


def test_snap_segments_drops_segments_that_collapse():
    from agent.tools.media import snap_segments
    segs = [{"time_range_s": [0.0, 1.0], "prompt": "a"}, {"time_range_s": [1.0, 1.1], "prompt": "b"},
            {"time_range_s": [1.1, 6.0], "prompt": "c"}]
    out = snap_segments(segs, duration=6.0)
    assert [s["prompt"] for s in out] == ["a", "c"]


@pytest.mark.parametrize("bad", ["../../etc/passwd", "/etc/passwd", "sub/../../x"])
def test_file_tools_stay_inside_their_root(tmp_path, bad):
    from agent.tools.files import resolve_inside
    with pytest.raises(ValueError):
        resolve_inside(str(tmp_path), bad)


def test_file_tools_accept_paths_inside(tmp_path):
    from agent.tools.files import resolve_inside
    assert resolve_inside(str(tmp_path), "a/b.txt") == tmp_path / "a" / "b.txt"


def test_graphs_compile_without_a_model():
    from agent.graphs.meta import build_meta_graph
    from agent.graphs.task import build_task_graph
    assert build_task_graph(None, None, None).compile() is not None
    assert build_meta_graph(None, None).compile() is not None


@pytest.fixture
def harness(tmp_path):
    from ar_kernel.config import KernelConfig
    from ar_kernel.contract.verify import ContractHarness
    from ar_kernel.telemetry.recorder import Recorder
    h = ContractHarness(KernelConfig.load(), tmp_path / "run", Recorder(tmp_path / "run"))
    h.start()
    yield h
    h.stop()


@pytest.mark.parametrize("kind", ["edit_self", "improve_recipe"])
def test_dry_run_of_both_entry_points_against_mock_services(kind, harness, tmp_path, monkeypatch):
    """The contract smoke run, in-process: real SDK, real MCP client, mock gateway and tools."""
    from ar_contract.models import EditContext, RecipeContext
    from ar_contract.run import main
    ws, ctx_dir = tmp_path / "ws", tmp_path / "ctx"
    ws.mkdir(), ctx_dir.mkdir()
    ctx = (EditContext if kind == "edit_self" else RecipeContext)(
        nodes_remaining=1, attempt=1, max_attempts=1, dry_run=True)
    (ctx_dir / "context.json").write_text(ctx.model_dump_json())
    caller = harness.registry.issue(node="n1", phase="contract", attempt=1, workspace_host=ws,
                                    staging_host=tmp_path, mock_script="smoke")
    for key, value in {"AR_SOCKET_DIR": str(harness.socket_dir), "AR_TOKEN": caller.token,
                       "AR_DEFAULT_MODEL": "mock-model", "AR_WORKSPACE": str(ws),
                       "AR_CONTEXT_DIR": str(ctx_dir), "AR_AGENT_DIR": str(SEED)}.items():
        monkeypatch.setenv(key, value)
    for mod in [m for m in sys.modules if m == "ar_contract.client"]:
        del sys.modules[mod]                                  # re-read AR_SOCKET_DIR
    assert main([kind]) == 0, (ws / "result.json").read_text()
    assert json.loads((ws / "result.json").read_text())["ok"] is True


@pytest.mark.docker
def test_seed_agent_passes_contract_verification(tmp_path, harness):
    from ar_kernel.config import KernelConfig
    from ar_kernel.contract.verify import verify_contract
    from ar_kernel.telemetry.recorder import Recorder
    from ar_kernel.vcs.agents_repo import AgentsRepo
    repo = AgentsRepo(tmp_path / "agents.git")
    commit = repo.init(SEED)
    report = verify_contract(cfg=KernelConfig.load(), run_dir=tmp_path / "run", run_id="seed", repo=repo,
                             commit=commit, harness=harness, recorder=Recorder(tmp_path / "run"),
                             node="root", attempt=1)
    assert report.ok, report.to_retry()
```

- [ ] **Step 2: Run to confirm they fail**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_agent.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError: No module named 'agent'`).

- [ ] **Step 3: Scaffolding, settings, entry**

```gitignore
# seed_agent/.gitignore
__pycache__/
*.pyc
```

```text
# seed_agent/agent/requirements.txt
# Extra runtime packages for this agent version. The base image already provides
# openai-agents, mcp, httpx, langgraph (+ sqlite checkpointer), pydantic, numpy,
# opencv-python-headless, Pillow and ffmpeg. The container has no network, so
# anything listed here is baked into the image at build time.
```

```python
# seed_agent/agent/settings.py
import os

MODEL = os.environ.get("AR_DEFAULT_MODEL", "mock-model")
AGENT_ROOT = os.environ.get("AR_AGENT_DIR", "/agent")
WORKSPACE = os.environ.get("AR_WORKSPACE", "/workspace")
TASK_MAX_TURNS = 80          # tool-using data builder
META_MAX_TURNS = 60          # coding agent
CHECK_ROUNDS = 3             # recipe_check -> revise loops inside one attempt
SELFTEST_ROUNDS = 2          # implement -> self-test loops inside one attempt
```

```python
# seed_agent/agent/entry.py
"""Fixed entry points (spec 1.1.4). The kernel imports exactly these two names."""
from ar_contract.models import EditContext, EditResult, RecipeContext, RecipeResult


async def edit_self(ctx: EditContext) -> EditResult:
    from .graphs.meta import run_meta
    return await run_meta(ctx)


async def improve_recipe(ctx: RecipeContext) -> RecipeResult:
    from .graphs.task import run_task
    return await run_task(ctx)
```

```markdown
<!-- seed_agent/agent/memory/README.md -->
# Agent memory

Notes written here during `edit_self` are committed with the code and inherited by
every descendant. Use them to record what was tried, what the scores said, and
which ideas to try next. `improve_recipe` sees this directory read-only.
```

- [ ] **Step 4: Tools**

```python
# seed_agent/agent/tools/files.py
"""File and shell tools bound to one root (/agent for edit_self, /workspace for improve_recipe)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from agents import function_tool

MAX_READ = 200_000


def resolve_inside(root: str, path: str) -> Path:
    base = Path(root).resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"{path} is outside {root}")
    return target


def make_file_tools(root: str) -> list:
    @function_tool
    def read_file(path: str) -> str:
        """Read a text file (relative to the tool root)."""
        return resolve_inside(root, path).read_text(errors="replace")[:MAX_READ]

    @function_tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file (relative to the tool root)."""
        target = resolve_inside(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    @function_tool
    def list_dir(path: str = ".") -> list[str]:
        """List a directory (relative to the tool root)."""
        target = resolve_inside(root, path)
        return sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())

    @function_tool
    def run_command(command: str, timeout_s: int = 600) -> str:
        """Run a shell command in the tool root (ffmpeg, ffprobe, python, ...). Returns exit code and output tail."""
        try:
            r = subprocess.run(["bash", "-lc", command], cwd=root, capture_output=True, text=True,
                               timeout=min(int(timeout_s), 3600))
        except subprocess.TimeoutExpired:
            return f"timed out after {timeout_s}s"
        return f"exit {r.returncode}\n{(r.stdout + r.stderr)[-8000:]}"

    return [read_file, write_file, list_dir, run_command]
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

from agents import function_tool

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


@function_tool
def snap_timed_prompts(segments_json: str, duration: float) -> str:
    """Snap timed-prompt segment boundaries to rollout-round boundaries. Input and output: JSON list of
    {"time_range_s": [start, end], "prompt": str}."""
    return json.dumps(snap_segments(json.loads(segments_json), duration))


@function_tool
async def caption_clip(video_path: str, hint: str = "") -> str:
    """Caption a video: samples 4 frames with ffmpeg and asks a vision model (through the kernel
    gateway) for one factual caption of the scene and camera motion."""
    from ar_contract.client import openai_client
    from ..settings import MODEL
    duration = float(json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path],
        capture_output=True, text=True, check=True).stdout)["format"]["duration"])
    images = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, frac in enumerate((0.1, 0.35, 0.6, 0.85)):
            frame = Path(tmp) / f"{i}.jpg"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{duration * frac:.3f}",
                            "-i", video_path, "-frames:v", "1", "-vf", "scale=512:-2", str(frame)], check=True)
            images.append(base64.b64encode(frame.read_bytes()).decode())
    content = [{"type": "input_text", "text": "Write one factual caption (1-3 sentences) describing the "
                                               "scene and how the camera moves. " + hint}]
    content += [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"} for b64 in images]
    reply = await openai_client().responses.create(model=MODEL, input=[{"role": "user", "content": content}])
    return reply.output_text.strip()
```

- [ ] **Step 5: Agent definitions and prompts**

```python
# seed_agent/agent/agents/definitions.py
"""OpenAI Agents SDK agents used by the seed graphs."""
from __future__ import annotations

from pathlib import Path

from agents import Agent
from pydantic import BaseModel

from ..settings import AGENT_ROOT, MODEL, WORKSPACE
from ..tools.files import make_file_tools
from ..tools.media import caption_clip, snap_timed_prompts

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def _prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text()


class DataPlan(BaseModel):
    hypotheses: list[str]
    actions: list[str]


class BuildOutcome(BaseModel):
    data_commit: str
    notes: str


class RecipeEntry(BaseModel):
    key: str
    value: float


class RecipeDraft(BaseModel):
    # A list of pairs, not a dict: strict structured outputs reject free-form object keys.
    entries: list[RecipeEntry]
    rationale: str


class EditPlan(BaseModel):
    changes: list[str]
    rationale: str


def task_agents(tools) -> dict[str, Agent]:
    return {
        "planner": Agent(name="planner", model=MODEL, instructions=_prompt("planner"), output_type=DataPlan),
        "builder": Agent(name="data_builder", model=MODEL, instructions=_prompt("data_builder"),
                         mcp_servers=[tools] if tools else [],
                         tools=[*make_file_tools(WORKSPACE), caption_clip, snap_timed_prompts],
                         output_type=BuildOutcome),
        "recipe": Agent(name="recipe_writer", model=MODEL, instructions=_prompt("recipe_writer"),
                        output_type=RecipeDraft),
    }


def meta_agents() -> dict[str, Agent]:
    return {
        "analyst": Agent(name="analyst", model=MODEL, instructions=_prompt("analyst"), output_type=EditPlan),
        "coder": Agent(name="coder", model=MODEL, instructions=_prompt("coder"),
                       tools=make_file_tools(AGENT_ROOT)),
    }
```

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
one node is one experiment.
```

`seed_agent/agent/prompts/data_builder.md`:
```markdown
You build the training data for this node, using the kernel tools and local tools.

Kernel tools: data_query (the archive-wide clip pool), hf_search / hf_download (downloads land
under /workspace/staging/hf/), video_probe, data_ingest (candidates must be under
/workspace/staging/), data_commit, and job_status / job_wait / job_cancel for GPU jobs if any
generator tool is listed. Local tools: read_file, write_file, list_dir, run_command (ffmpeg,
ffprobe, python), caption_clip, snap_timed_prompts. Your working directory is /workspace.

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

Work in small batches: fetch a little, convert, ingest, check the rejection reasons, adjust.
When you have a data commit that tests the plan, stop and return its id with short notes
on what it contains and why.
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

If you are given failures from a previous check, fix exactly those. Return the entries as
key/value pairs and a short rationale tying the recipe to the data.
```

`seed_agent/agent/prompts/analyst.md`:
```markdown
You improve the agent's own code: the prompts, graphs and tools under /agent that decide
how training data is built. You see the lineage (scores, per-metric results, code diffs,
data and recipes of every ancestor) and the archive summary.

Find the most likely reason the recent nodes did not improve (for example: rejected
candidates wasted the attempt, poses were missing so clips became static-only, the recipe
failed the step budget, the plan changed too many things at once) and propose 1-3 focused
code or prompt changes that address it. Record what you learned in agent/memory/.

Hard constraints: agent/entry.py must keep top-level edit_self(ctx) and improve_recipe(ctx),
each taking exactly one parameter. Do not add packages the base image lacks unless you also
list them in agent/requirements.txt.
```

`seed_agent/agent/prompts/coder.md`:
```markdown
You implement the planned changes to the agent code in the current directory (/agent),
using read_file, write_file, list_dir and run_command. Make the smallest correct change.
After editing, run `python -c "import agent.entry"` to check that it imports, and fix any
error before finishing. Finish with a one-paragraph summary of what you changed and why.
```

- [ ] **Step 6: The task graph**

```python
# seed_agent/agent/graphs/task.py
"""improve_recipe: plan -> build -> recipe -> check -> (revise | finish)."""
from __future__ import annotations

import json
from typing import TypedDict

from agents import Agent, Runner
from ar_contract.client import mcp_tools
from ar_contract.models import RecipeContext, RecipeResult
from langgraph.graph import END, START, StateGraph

from ..settings import CHECK_ROUNDS, MODEL, TASK_MAX_TURNS, WORKSPACE


class TaskState(TypedDict, total=False):
    plan: str
    data_commit: str
    notes: str
    recipe: dict
    rationale: str
    failures: list[str]
    rounds: int


def _summary(ctx: RecipeContext) -> str:
    return json.dumps({"lineage": ctx.lineage, "archive": ctx.archive, "n_gpus": ctx.n_gpus,
                       "parent_data_commit": ctx.parent_data_commit, "parent_recipe": ctx.parent_recipe,
                       "clip_pool_size": len(ctx.clip_pool), "tools": ctx.tools,
                       "retry": ctx.retry, "format_rules": ctx.format_rules}, default=str)[:60_000]


def _coerce(entries, rules: dict) -> dict:
    out = {}
    for e in entries:
        rule = rules.get(e.key, {})
        out[e.key] = int(round(e.value)) if rule.get("type") == "int" else float(e.value)
    return out


def build_task_graph(agents: dict | None, tools, ctx: RecipeContext | None) -> StateGraph:
    async def plan(state: TaskState) -> TaskState:
        r = await Runner.run(agents["planner"], _summary(ctx))
        return {"plan": json.dumps(r.final_output.model_dump())}

    async def build(state: TaskState) -> TaskState:
        prompt = f"PLAN:\n{state['plan']}\n\nCONTEXT:\n{_summary(ctx)}"
        if state.get("failures"):
            prompt += f"\n\nTHE LAST RECIPE CHECK FAILED; fix the data if the failures are about data:\n{state['failures']}"
        r = await Runner.run(agents["builder"], prompt, max_turns=TASK_MAX_TURNS)
        return {"data_commit": r.final_output.data_commit, "notes": r.final_output.notes}

    async def recipe(state: TaskState) -> TaskState:
        prompt = json.dumps({"rules": ctx.tunable_rules, "resolution_allowlist": ctx.resolution_allowlist,
                             "lora_allowlist": ctx.lora_allowlist, "n_gpus": ctx.n_gpus,
                             "data_notes": state.get("notes"), "parent_recipe": ctx.parent_recipe,
                             "previous_failures": state.get("failures")})
        r = await Runner.run(agents["recipe"], prompt)
        return {"recipe": _coerce(r.final_output.entries, ctx.tunable_rules),
                "rationale": r.final_output.rationale}

    async def check(state: TaskState) -> TaskState:
        res = await tools.call_tool("recipe_check", {"recipe": state["recipe"],
                                                     "data_commit": state["data_commit"]})
        body = json.loads(res.content[0].text) if not res.isError else {"ok": False,
                                                                        "failures": [res.content[0].text]}
        return {"failures": [] if body.get("ok") else body.get("failures", []),
                "rounds": state.get("rounds", 0) + 1}

    def route(state: TaskState) -> str:
        if not state.get("failures") or state.get("rounds", 0) >= CHECK_ROUNDS:
            return "finish"
        return "revise"

    g = StateGraph(TaskState)
    for name, fn in (("plan", plan), ("build", build), ("recipe", recipe), ("check", check)):
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "build")
    g.add_edge("build", "recipe")
    g.add_edge("recipe", "check")
    g.add_conditional_edges("check", route, {"revise": "build", "finish": END})
    return g


async def run_task(ctx: RecipeContext) -> RecipeResult:
    async with mcp_tools() as tools:
        if ctx.dry_run:                  # contract smoke run: prove the wiring, do no work
            build_task_graph(None, tools, ctx).compile()
            names = [t.name for t in await tools.list_tools()]
            ping = await Runner.run(Agent(name="ping", model=MODEL, instructions="Reply ok."), "ping")
            return RecipeResult(data_commit=ctx.parent_data_commit or "dry-run", recipe={},
                                rationale=f"dry run: {len(names)} tools, model said {ping.final_output!r}")
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from ..agents.definitions import task_agents
        graph = build_task_graph(task_agents(tools), tools, ctx)
        async with AsyncSqliteSaver.from_conn_string(f"{WORKSPACE}/langgraph.db") as saver:
            app = graph.compile(checkpointer=saver)
            state = await app.ainvoke({"rounds": 0}, config={"recursion_limit": 40,
                                      "configurable": {"thread_id": f"improve_recipe-{ctx.attempt}"}})
    rationale = state.get("rationale", "")
    if state.get("failures"):
        rationale += f"\n\nUnresolved recipe_check failures: {state['failures']}"
    return RecipeResult(data_commit=state["data_commit"], recipe=state["recipe"],
                        rationale=f"{rationale}\n\nPlan: {state.get('plan')}\nData: {state.get('notes')}")
```

- [ ] **Step 7: The meta graph**

```python
# seed_agent/agent/graphs/meta.py
"""edit_self: analyze -> implement -> self-test -> (implement | finalize)."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from typing import TypedDict

from agents import Agent, Runner
from ar_contract.models import EditContext, EditResult
from langgraph.graph import END, START, StateGraph

from ..settings import AGENT_ROOT, META_MAX_TURNS, MODEL, SELFTEST_ROUNDS, WORKSPACE


class MetaState(TypedDict, total=False):
    plan: str
    summary: str
    errors: list[str]
    rounds: int


def selftest(root: str) -> list[str]:
    """The contract's static + import checks, run locally so the agent can fix itself."""
    errors = []
    try:
        tree = ast.parse(open(f"{root}/agent/entry.py").read())
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


def build_meta_graph(agents: dict | None, ctx: EditContext | None) -> StateGraph:
    async def analyze(state: MetaState) -> MetaState:
        brief = json.dumps({"lineage": ctx.lineage, "archive": ctx.archive,
                            "nodes_remaining": ctx.nodes_remaining, "retry": ctx.retry}, default=str)[:60_000]
        r = await Runner.run(agents["analyst"], brief)
        return {"plan": json.dumps(r.final_output.model_dump())}

    async def implement(state: MetaState) -> MetaState:
        prompt = f"PLAN:\n{state['plan']}"
        if state.get("errors"):
            prompt += f"\n\nTHE SELF-TEST FAILED; fix these first:\n{state['errors']}"
        r = await Runner.run(agents["coder"], prompt, max_turns=META_MAX_TURNS)
        return {"summary": str(r.final_output)}

    async def test(state: MetaState) -> MetaState:
        return {"errors": selftest(AGENT_ROOT), "rounds": state.get("rounds", 0) + 1}

    def route(state: MetaState) -> str:
        return "finish" if not state.get("errors") or state.get("rounds", 0) >= SELFTEST_ROUNDS else "fix"

    g = StateGraph(MetaState)
    for name, fn in (("analyze", analyze), ("implement", implement), ("test", test)):
        g.add_node(name, fn)
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "implement")
    g.add_edge("implement", "test")
    g.add_conditional_edges("test", route, {"fix": "implement", "finish": END})
    return g


async def run_meta(ctx: EditContext) -> EditResult:
    if ctx.dry_run:
        build_meta_graph(None, ctx).compile()
        ping = await Runner.run(Agent(name="ping", model=MODEL, instructions="Reply ok."), "ping")
        return EditResult(summary=f"dry run: model said {ping.final_output!r}")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from ..agents.definitions import meta_agents
    graph = build_meta_graph(meta_agents(), ctx)
    async with AsyncSqliteSaver.from_conn_string(f"{WORKSPACE}/langgraph.db") as saver:
        state = await graph.compile(checkpointer=saver).ainvoke(
            {"rounds": 0}, config={"recursion_limit": 30,
                                   "configurable": {"thread_id": f"edit_self-{ctx.attempt}"}})
    note = f" (self-test still failing: {state['errors']})" if state.get("errors") else ""
    return EditResult(summary=(state.get("summary") or "no summary") + note)
```

- [ ] **Step 8: Run the tests**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_agent.py -p no:cacheprovider`
Expected: PASS (unit tests plus both in-process dry runs against the mock services).

Run: `conda run --no-capture-output -n autoresearcher python -m pytest tests/test_seed_agent.py -m docker -p no:cacheprovider`
Expected: PASS (the seed agent passes contract verification in real containers).

- [ ] **Step 9: Commit**

```bash
git add seed_agent tests/test_seed_agent.py
git commit -m "feat(seed): seed agent -- LangGraph task/meta graphs on the Agents SDK, passing the contract"
```

---

### Task 17: Real-component verification, log, merge

**Files:**
- Modify: `docs/superpowers/plans/verification-log.md` (append a Plan 2 section)
- Modify (only if implementation diverged): `docs/superpowers/specs/2026-09-17-autoresearcher-design.md`

- [ ] **Step 1: Default suite**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -p no:cacheprovider`
Expected: PASS, with no GPU, network or Docker needed.

- [ ] **Step 2: Docker suite, alone (no GPU jobs running)**

Run: `conda run --no-capture-output -n autoresearcher python -m pytest -m docker -p no:cacheprovider`
Expected: PASS. This covers the image, the isolation checks (§16.3 item 4), timeout and kill, contract verification on 7 fixtures, the end-to-end tool test, and the seed agent contract. Record the counts and durations.

- [ ] **Step 3: Negative control for the isolation test**

Temporarily change the runner's `--network none` to `--network bridge` and re-run `tests/test_sandbox_runner.py::test_isolation_holds_from_inside -m docker`. It must **fail** (the internet becomes reachable). Restore `--network none`, re-run, and confirm it passes. Record both runs. An isolation test that also passes with networking on proves nothing (verification-log finding 8).

- [ ] **Step 4: Live LLM run (only if `OPENAI_API_KEY` and `OPENAI_MODEL` are set)**

In a scratch run: ingest the 18 `WorldModel/data/examples` clips (staged copies, as in Plan 1 Task 14), start real services with `Upstream(OPENAI_BASE_URL, OPENAI_API_KEY, ...)`, and call `run_improve_recipe` on the seed agent for one node. Record the outcome whether or not it succeeds: `ok`, the error, the tool calls made, and the tokens and requests seen by the gateway. Success is not required; the point is to see a real model drive the real tools end to end. **If no key is set, record "live LLM run deferred: no OPENAI_API_KEY" instead. Do not mark this step passed.**

- [ ] **Step 5: Re-check the spec amendments**

Confirm the amendments made with this plan (§9.5 network, §10 tool names) still describe what was built. If implementation diverged anywhere else, amend the spec now and list it in the log.

- [ ] **Step 6: Write the verification log section**

Append `## Plan 2 — agent runtime` to `docs/superpowers/plans/verification-log.md`. Include the verified facts 1–13 this plan relied on, the suite and docker results, the isolation negative control, the live-LLM outcome (or its deferral), and every defect found during implementation with its fix.

- [ ] **Step 7: Merge and push (standing rule: `main` always current)**

```bash
git checkout main && git merge --ff-only <plan-2-branch> && git push origin main
```

Confirm that the commit `main` points to is the one that passed Steps 1–3, and delete the merged local branch.

---

## Interface contract for Plan 4 (the loop)

Plan 4's tasks will be written after Plan 3. Plan 4 sequences these Plan 2 pieces and owns everything listed under "Plan 4 must".

**Plan 4 consumes:**

| Piece | From | Use |
|---|---|---|
| `RunServices`, `socket_dir_for` | Task 6 | One instance per run: real gateway plus tool server. Start at run start, stop on exit and on force stop. |
| `create_gateway_app(... upstream=Upstream(OPENAI_BASE_URL, OPENAI_API_KEY, ...), allowed_models=gateway.model_allowlist ∪ {OPENAI_MODEL})` | Task 5 | Real LLM route. With no key: `upstream=None`, so every agent call is served by mocks. |
| `DataTools`, `HfTools`, `JobQueue` + `register_*` | Tasks 7–9 | The real tool server. Plan 3 registers generator backends on the same `JobQueue`. |
| `gpu_lock` shared by `DataTools`, `JobQueue` and training/eval | Tasks 7, 9 | GPU phases never overlap (spec 7). Plan 4 holds it for precache, train, merge, render and eval. |
| `AgentsRepo` | Task 12 | `init(seed_agent)` at bootstrap; `set_ref(branch_ref(node), commit)` for the passing attempt. |
| `ContractHarness`, `verify_contract` | Task 14 | Step 4 of the cycle. `report.to_retry()` becomes the next `edit_self` attempt's `retry`. |
| `run_edit_self`, `run_improve_recipe`, `attempt_dirs` | Task 15 | Steps 3 and 5. Retries pass `base_commit=<failed attempt commit>` and `previous_workspace=<failed attempt workspace>`. |
| Context conventions | Task 13 | Plan 4 **writes** `nodes/<n>/recipe.yaml`, `nodes/<n>/rationale.md` and `nodes/<n>/eval/aggregates.json`. |

**Plan 4 must:**
1. **Liveness (spec 14.5).** Plan 2 enforces only the hard cap (4 x soft). Plan 4 wraps the runner with the soft timeout plus a probe window, using gateway and tool events for the caller's token, container CPU from `RunResult.stats`, and workspace changes. **GPU utilization is not a liveness signal** (verification-log finding 2).
2. **Per-attempt training directories.** Gate views, `train_config.yaml`, `train/` and `dataset_cache` go under `nodes/<n>/attempts/<phase>-<k>/`, so a failed retry can never supply the checkpoint (Plan 1 review, deferred).
3. **Termination.** Force stop and resume discard must `docker kill` containers by the prefix `ar-<run_id>-`, call `JobQueue.shutdown()` and `RunServices.stop()`, kill kernel process groups, then confirm GPU memory is released and sweep `_megasam_tmp` before the next phase (verification-log findings 6–7).
4. **Selection versus noise.** The proxy aggregate has a noise floor of about 4.4e-4 from a single re-run. Measure it properly (repeat evaluation of one checkpoint) before long unattended runs, and use it in parent selection (verification-log finding 5).
5. **Upstream outage.** Watch the `llm.response` error rate. Upstream unavailable for more than `gateway.upstream_outage_pause_min` pauses the loop without charging the node (spec 14.2).
6. **Deferred Plan 1 items:** `score_node` try/finally and rank/alpha from the node's resolved config; schema columns `recipe_path` and `attempt_counts_json`; `resolve_gpus` visibility (14.6); run-relative paths if resume-after-move matters.

## Self-review

- **Spec coverage.** §9.1 → Task 16; §9.2 → Task 3; §9.3 → Tasks 3 and 13; §9.4 → Task 14; §9.5 → Tasks 10–11 (network amended); §10 tools → Tasks 7–9 (generators → Plan 3); §13.1–13.3 gateway and tool capture → Tasks 1, 4–6; §5.2 → Task 12; §5.5 aspect → Task 2; §16.1 gateway / contract / job API → Tasks 4, 5, 9, 14; §16.3 item 4 → Tasks 11 and 17. Left to Plan 4, by the contract above: §7.2 sequencing, §12, §13.4, §14.3–14.6 and liveness.
- **Placeholders.** None. The one open outcome, the live LLM run, has an explicit deferral rule.
- **Type consistency.** `Caller` fields are fixed in Task 4 and used unchanged in Tasks 5–9, 14 and 15. `PhaseOutcome` and `PhaseEnv` are defined once, in Task 15. Tool names are identical in `register_*`, the Task 14 mock objects, `run_improve_recipe`'s tool list and the seed prompts.
