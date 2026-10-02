# Seed Agent Rewrite and Benchmark Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agents never see or reach the evaluation benchmark, `edit_self` improves the agent system without seeing scores, and the seed agent is rewritten around intent-only plans, kernel-owned skills and a planner the engineer can talk to.

**Architecture:** The kernel gains one isolation module (name refusal, copied-text check, output scrub, audit) used by the HF, ingest and GPU tools and by the loop. Contexts are built per phase: `improve_recipe` gets aliased scores and data, `edit_self` gets code edits and a larger process digest. Reference knowledge moves out of the agent into kernel tool descriptions and a read-only `read_skill` tool. The seed agent's prompts, orchestration and briefing are rewritten; harness and local tools stay.

**Tech Stack:** Python 3.12, pydantic 2, MCP 2.2 (`mcp.server.mcpserver`), LangGraph/LangChain-core, pytest. Tests run with `.envs/autoresearcher/bin/python -m pytest` (never the system Python).

**Spec:** `docs/superpowers/specs/2026-10-02-seed-agent-rewrite-design.md`

## Global Constraints

- Work in `AutoResearcher/` on branch `seed-agent-rewrite`; every path below is relative to that repo root.
- Run tests with `PY=.envs/autoresearcher/bin/python`; `$PY -m pytest tests/<file> -q`. Never use the system Python.
- No agent-visible string (seed agent files, tool names, descriptions, schemas, errors, results, contexts, skills) may contain the benchmark's name or the word "case" for an evaluation item. Kernel-internal code, comments, config and telemetry keep real names.
- Nothing on disk changes names: `eval/aggregates.json`, the DB and `run.json` keep real metric names. Aliases exist only in what `context_bundle` hands to agents.
- No backward compatibility: no shims for `live-10-01` or earlier runs, no fallbacks for old context shapes.
- Raise exceptions for failed preconditions; no print-and-exit.
- Agent-layer code (`seed_agent/`): fewer and shorter files, straightforward implementations.
- Commit messages are plain sentences in the repo's style. Never add a co-author or any Claude attribution line.
- Rejection messages, verbatim: `this clip is on the kernel's exclusion list and cannot be used as training data. Use a different clip.` and `this prompt is on the kernel's exclusion list. Write a different one.`
- Constants, verbatim: copied-text run = 8 consecutive normalised words; plan text fields ≤ 1500 characters; `MAX_QUESTIONS = 5`; `isolation.blocked_names` default `[wbench]`; `isolation.audit_patterns` default `[wbench, meituan-longcat]`.

## Deviations from the spec (decided while planning; the spec is updated to match)

1. **Audit scope.** The audit scans every tool-call argument of the attempt (commands, file contents written through tools, queries, kernel tool arguments, plans). It does not scan files on disk or command output: a cloned third-party repo or a paper that merely mentions the benchmark would quarantine a node that did nothing wrong.
2. **Two aliases changed** after reading the metric code: `perspective_consistency` measures how steadily the tracked subject stays in frame, so it becomes `subject_framing_stability` (not `viewpoint_stability`); `segment_continuity` detects hard cuts, so it becomes `no_hard_cuts`.
3. **Per-role time** in the process digest is model time (sum of LLM call latencies), not wall time: a planner's conversation spans the engineer's whole run, so its wall time would say nothing.
4. **`MAX_QUESTIONS`** lives in `agent/entry.py` beside `MAX_ROUNDS` (the spec said `entry.py` is untouched).
5. **Kernel failures hidden.** A node that ended `eval_failed` or `crashed` shows a fixed kernel-side sentence as its error in contexts: the raw error can name the benchmark.
6. **Tool output scrub.** Every kernel tool result and error passes through one scrub that replaces a blocked name with `[removed]` (worker log tails name the render script).
7. **A quarantined node's clips** leave the clip pool (`data_query`, the context's `clip_pool`).
8. **A planner answering a question** cannot submit a plan (the submit is refused) and its disk changes during the answer are not undone: undoing them could also undo what the engineer's parallel tool calls wrote.

## Amendments after review (2026-10-02; these override the task text below)

A. **Scrub word.** A blocked name is replaced by `render`, not `[removed]` (`run_wbench` reads `run_render`).
B. **No runtime in the process digest.** `phases` (seconds), per-role `model_s` and `gpu_jobs.gpu_s` are not
   produced, and the briefing has no time line. Turns, compactions, tool calls and errors, GPU job counts,
   ingest results, plan rounds and failed attempts stay.
C. **No `ask_planner`.** `request_replan` is the engineer's only way back. No `MAX_QUESTIONS`, no
   `Role.answer`, no `questions` in `plans.json` or in the digest; deviations 4 and 8 and Review Focus 1 lapse.
D. **Audit patterns** are `[wbench, meituan-longcat, "2605.25874"]`.
E. **Audit scope and gateway censor.** The audit reads everything the model wrote in the attempt: reasoning,
   text and tool-call arguments. Tool results are not audited; instead the gateway replaces blocked names
   and audit patterns in tool-result messages before the request goes upstream and before it is recorded,
   so the model never reads the name and transcripts stay clean (Task 9).

## File Structure

| file | change | responsibility |
|---|---|---|
| `kernel/ar_kernel/isolation.py` | create | blocked names, copied-text check, output scrub, audit |
| `kernel/ar_kernel/skills/*.md` | create (6) | kernel-owned reference notes |
| `kernel/ar_kernel/tools/skills.py` | create | `read_skill` tool |
| `kernel/ar_kernel/eval/score.py` | modify | alias table, `agent_metrics`, `agent_aggregates`, `metric_guide` |
| `kernel/ar_kernel/context_bundle.py` | modify | per-phase node entries; no `FORMAT_RULES` |
| `kernel/ar_kernel/process_digest.py` | modify | larger digest; owns `role_of` |
| `kernel/ar_kernel/tools/{hf_tools,data_tools,gpu_jobs,rollouts,images,captioner,server}.py` | modify | refusals, scrub, argument descriptions, renamed AlayaWorld interface |
| `kernel/ar_kernel/data/ingest.py` | modify | exclusion check and neutral message |
| `kernel/ar_kernel/archive/nodes.py`, `loop.py`, `agent_phase.py` | modify | `quarantined` status and audit |
| `kernel/ar_kernel/contract/verify.py`, `run_kit.py` | modify | prompt check; register `read_skill`; scrub names |
| `contract/ar_contract/models.py` | modify | `metric_guide` in, `format_rules` out |
| `configs/kernel.yaml`, `configs/base_recipe.yaml` | modify | `isolation` block; one comment |
| `seed_agent/agent/{orchestration,briefing,tools,entry}.py`, `prompts/*.md` | rewrite | the new seed agent |
| `seed_agent/agent/knowledge/` | delete | replaced by skills and tool descriptions |
| `panel/chat.py`, `panel/views.py` | modify | "Tools offered" shows parameters |
| `tests/test_isolation.py`, `tests/test_skills.py` | create | new units |
| other `tests/test_*.py` | modify | follow the changes above |

## Review Focus

1. **A planner submits a plan while answering a question.** Expected: the submit is refused with a message, the engineer still gets a text answer, the plan and round count do not change. Test in Task 12 (`test_improve_recipe_full_flow`).
2. **A node with no evaluation output (failed before scoring) appears in a lineage or as a sibling.** Expected: its entry has `aggregates: None` and empty metrics, and the digest renders. Test in Task 6 and Task 11.
3. **`plans.json` written by an edited agent has another shape, or is not JSON.** Expected: the process digest still builds; the rounds section is omitted or zero. Test in Task 7.
4. **A caption file with segments that are not objects, or no caption text.** Expected: the copied-text check does not raise; the format checker's own verdict stands. Test in Task 3.
5. **A legitimate dataset whose id or tags contain a blocked name, or a tool result that contains one.** Expected: the dataset is absent and reads as not found; the name is replaced in the result, never passed through. Tests in Task 2 and Task 4.

---

### Task 0: Branch and spec

**Files:**
- Modify: `docs/superpowers/specs/2026-10-02-seed-agent-rewrite-design.md` (already updated with the deviations above)

- [ ] **Step 1: Create the branch and commit the spec and this plan**

```bash
git checkout -b seed-agent-rewrite
git add docs/superpowers/specs/2026-10-02-seed-agent-rewrite-design.md docs/superpowers/plans/2026-10-02-seed-agent-rewrite.md
git commit -m "Design and plan: seed agent rewrite and benchmark isolation"
```

---

### Task 1: Isolation core

**Files:**
- Create: `kernel/ar_kernel/isolation.py`
- Modify: `configs/kernel.yaml` (after the `leakage:` block)
- Modify: `configs/base_recipe.yaml:87` (comment only)
- Test: `tests/test_isolation.py`

**Interfaces:**
- Produces:
  - `blocked(cfg, text: str | None) -> bool` — `text` contains a name of `isolation.blocked_names`, case-insensitive.
  - `strings(value) -> Iterator[str]` — every string inside nested dicts and lists.
  - `copies_held_out(cfg, *texts) -> bool` — any text shares 8 consecutive normalised words with a benchmark case; non-strings are ignored.
  - `scrub(value, names: list[str])` — the same value with each name replaced by `[removed]` in every string.
  - `EXCLUDED_CLIP`, `EXCLUDED_PROMPT` — the two neutral messages.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_isolation.py
import json

from ar_kernel.config import KernelConfig
from ar_kernel.isolation import blocked, copies_held_out, scrub

CFG = KernelConfig.load()
CASE = json.loads((CFG.wbench / "data" / "cases" / "case_196.json").read_text())
SENTENCE = "Torch sconces on both walls cast flickering orange light."      # nine words of case 196's scene text


def test_blocked_names_match_any_case_and_position():
    assert blocked(CFG, "meituan-longcat/WBench") and blocked(CFG, "someone/my-wbench-mirror")
    assert not blocked(CFG, "org/walks") and not blocked(CFG, "") and not blocked(CFG, None)


def test_a_copied_run_of_eight_words_is_caught_with_a_prefix_a_suffix_or_a_split():
    assert SENTENCE in CASE["environment_prompt"]
    assert copies_held_out(CFG, f"A clip. {SENTENCE} Then more.")
    assert copies_held_out(CFG, "torch sconces, on both walls: cast flickering orange light")     # case, punctuation
    assert copies_held_out(CFG, "Torch sconces on both walls. Cast flickering orange light.")    # split in two
    assert copies_held_out(CFG, "unrelated", CASE["interactions"][0]["prompt"])


def test_short_phrases_and_paraphrases_pass():
    assert not copies_held_out(CFG, "First-person viewer.")
    assert not copies_held_out(CFG, "Wall torches throw a wavering orange glow along a stone castle corridor.")
    assert not copies_held_out(CFG, None, "", 3)


def test_scrub_replaces_names_in_nested_values():
    out = scrub({"log": ["[run_WBench] 4 items", 3], "ok": True}, ["wbench"])
    assert out == {"log": ["[run_[removed]] 4 items", 3], "ok": True}
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_isolation.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ar_kernel.isolation'`

- [ ] **Step 3: Add the config block**

In `configs/kernel.yaml`, after the `leakage:` block:

```yaml
isolation:                       # keeps the evaluation set out of agents' reach
  blocked_names: [wbench]        # case-insensitive substrings: a Hub repo naming one does not exist for agents,
                                 #   and one in a kernel tool's output is replaced
  audit_patterns: [wbench, meituan-longcat, "2605.25874"]   # a tool call naming one quarantines the node
```

In `configs/base_recipe.yaml` line 87, replace `65536 is the value the WBench 24GB work validated.` with `65536 is the value validated on 24 GB cards.`

- [ ] **Step 4: Write the module**

```python
# kernel/ar_kernel/isolation.py
"""Keeping the evaluation set out of agents' reach: names the kernel refuses, text that copies an
evaluation prompt, and the scrub of kernel tool output. Nothing here is shown to an agent."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

RUN_WORDS = 8               # this many consecutive words shared with an evaluation prompt is a copy
SCRUBBED = "[removed]"
EXCLUDED_CLIP = ("this clip is on the kernel's exclusion list and cannot be used as training data. "
                 "Use a different clip.")
EXCLUDED_PROMPT = "this prompt is on the kernel's exclusion list. Write a different one."


def blocked(cfg, text: str | None) -> bool:
    text = (text or "").lower()
    return any(name.lower() in text for name in cfg.get("isolation.blocked_names") or [])


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def _runs(text: str) -> set[tuple]:
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return {tuple(words[i:i + RUN_WORDS]) for i in range(len(words) - RUN_WORDS + 1)}


@lru_cache(maxsize=None)
def _held_out_runs(cases_dir: Path) -> frozenset:
    runs: set[tuple] = set()
    for path in sorted(cases_dir.glob("case_*.json")):
        for text in strings(json.loads(path.read_text(encoding="utf-8"))):
            runs |= _runs(text)
    return frozenset(runs)


def copies_held_out(cfg, *texts) -> bool:
    """A word run survives an added prefix, suffix or a split sentence; a paraphrase or a short
    common phrase does not match, on purpose."""
    held = _held_out_runs(cfg.wbench / "data" / "cases")
    return any(not held.isdisjoint(_runs(text)) for text in texts if isinstance(text, str))


def scrub(value, names: list[str]):
    if isinstance(value, str):
        for name in names:
            value = re.sub(re.escape(name), SCRUBBED, value, flags=re.IGNORECASE)
        return value
    if isinstance(value, dict):
        return {key: scrub(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item, names) for item in value]
    return value
```

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest tests/test_isolation.py tests/test_config.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add kernel/ar_kernel/isolation.py tests/test_isolation.py configs/kernel.yaml configs/base_recipe.yaml
git commit -m "isolation: blocked names, the copied-text check and the output scrub"
```

---

### Task 2: Hub tools refuse blocked repos

**Files:**
- Modify: `kernel/ar_kernel/tools/hf_tools.py` (`HfTools.__init__`, `search`, `_info`)
- Test: `tests/test_hf_tools.py`

**Interfaces:**
- Consumes: `blocked(cfg, text)` from Task 1.
- Produces: a blocked or missing repo raises `ToolError("dataset '<repo>' was not found on the Hub")` from `hf_list_files` and `hf_download`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_hf_tools.py`)

```python
def test_a_blocked_repo_is_absent_from_search(env):
    tools, caller = env
    tools.api.datasets = [{"id": "org/walks"}, {"id": "meituan-longcat/WBench"},
                          {"id": "org/mirror", "tags": ["wbench"]}]
    assert [h["id"] for h in tools.search(caller, "org", "dataset", 5)] == ["org/walks"]


def test_a_blocked_repo_reads_exactly_like_a_missing_one(env, monkeypatch):
    tools, caller = env

    def missing(repo_id, revision=None, files_metadata=False):
        raise hf_errors.RepositoryNotFoundError(
            "404", response=httpx.Response(404, request=httpx.Request("GET", "http://x")))
    monkeypatch.setattr(tools.api, "dataset_info", missing)
    messages = []
    for repo in ("org/gone", "meituan-longcat/WBench"):
        for call in (lambda: tools.list_files(caller, repo, "main"),
                     lambda: tools.download(caller, repo, "main", ["*"])):
            with pytest.raises(ToolError) as err:
                call()
            messages.append(str(err.value).replace(repo, "<repo>"))
    assert set(messages) == {"dataset '<repo>' was not found on the Hub"}
```

The fake's `list_datasets` builds each row with `tags=[]` then `**d`, which fails on a row that sets `tags`. Change that line of `FakeApi.list_datasets` to:

```python
            return [SimpleNamespace(**{"tags": [], "downloads": 0, "last_modified": None, "card_data": {}, **d})
                    for d in self.datasets]
```

The first test's query is `"org"`, which is in every id; the blocked rows must be dropped by name, not by the word filter.

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_hf_tools.py -q -k "blocked"`
Expected: FAIL (blocked rows returned; messages differ)

- [ ] **Step 3: Implement**

In `hf_tools.py` add `from ..isolation import blocked`, store the config (`self.cfg = cfg` in `__init__`), and change `search` and `_info`:

```python
        rows = [{"id": d.id, "license": _license(d), "tags": list(d.tags or [])[:40],
                 "gated": getattr(d, "gated", False),
                 "downloads": getattr(d, "downloads", None),
                 "last_modified": str(getattr(d, "last_modified", None))}
                for d in hits if all(w in " ".join([d.id, *(d.tags or [])]).lower() for w in words)
                and not blocked(self.cfg, " ".join([d.id, *(d.tags or [])]))]
```

```python
    def _info(self, repo: str, revision: str):
        if not REPO_RE.match(repo or ""):
            raise ToolError(f"invalid dataset repo id {repo!r}")
        missing = ToolError(f"dataset {repo!r} was not found on the Hub")
        if blocked(self.cfg, repo):             # reads exactly like a repo that does not exist
            raise missing
        try:
            return self.api.dataset_info(repo, revision=revision, files_metadata=True)
        except RepositoryNotFoundError:
            raise missing from None
        except HfHubHTTPError as exc:
            raise ToolError(str(exc)) from None
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_hf_tools.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/tools/hf_tools.py tests/test_hf_tools.py
git commit -m "hf tools: a repo naming a blocked name does not exist for agents"
```

---

### Task 3: Ingest exclusion check with one neutral message

**Files:**
- Modify: `kernel/ar_kernel/data/ingest.py:164-171` (the leakage block of `_check_and_store`)
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `blocked`, `copies_held_out`, `EXCLUDED_CLIP` from Task 1.
- Produces: a rejected candidate with `reasons == [EXCLUDED_CLIP]`; the `ingest.leakage` and `ingest.rejected` events carry `excluded: "image" | "source" | "text"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ingest.py`)

```python
import json

from ar_kernel.isolation import EXCLUDED_CLIP

COPIED = "Torch sconces on both walls cast flickering orange light."     # from an evaluation prompt


def _rejected_events(tmp_path):
    rec = Recorder(tmp_path)
    return [rec.load_payload(e["payload"]) for e in rec.read_events("n1") if e["type"] == "ingest.rejected"]


@pytest.mark.parametrize("caption,provenance,why", [
    ({"caption": f"A corridor. {COPIED}"}, PROV, "text"),
    ({"caption": "A corridor.", "segments": [{"time_range_s": [0, 4], "prompt": COPIED}]}, PROV, "text"),
    ({"caption": "A corridor."}, {"kind": "hf_dataset", "repo": "x/WBench-mirror", "revision": "a"}, "source"),
])
def test_excluded_candidates_get_one_neutral_reason(tmp_path, caption, provenance, why):
    ing = _ingestor(tmp_path)
    c = _candidate(tmp_path)
    c.caption.write_text(json.dumps(caption))
    c = Candidate(video=c.video, caption=c.caption, pose=c.pose, camera_motion="moving", provenance=provenance)
    [result] = ing.ingest([c], node_id="n1")
    assert result.accepted is False and result.reasons == [EXCLUDED_CLIP]
    assert _rejected_events(tmp_path)[-1]["excluded"] == why


def test_a_caption_with_odd_segments_does_not_break_the_text_check(tmp_path):
    ing = _ingestor(tmp_path)
    c = _candidate(tmp_path)
    c.caption.write_text(json.dumps({"caption": "A bright room.", "segments": ["not an object", None]}))
    [result] = ing.ingest([c], node_id="n1")          # the format checker's verdict stands, whatever it is
    assert result.reasons != [EXCLUDED_CLIP]
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_ingest.py -q -k "excluded or odd_segments"`
Expected: FAIL (the candidates are accepted)

- [ ] **Step 3: Implement**

In `ingest.py` add `from ..isolation import EXCLUDED_CLIP, blocked, copies_held_out` and replace the block from `verdict = self.leakage.check(video)` through the `has_segments = ...` line with:

```python
        caption_json = json.loads(caption.read_text(encoding="utf-8"))
        texts = [caption_json.get("caption"),
                 *[s.get("prompt") for s in caption_json.get("segments") or [] if isinstance(s, dict)]]
        verdict = self.leakage.check(video)
        excluded = ("image" if verdict.rejected else
                    "source" if blocked(self.cfg, json.dumps(candidate.provenance)) else
                    "text" if copies_held_out(self.cfg, *texts) else None)
        self._event(candidate, "ingest.leakage", node_id,
                    {"matches": verdict.matches, "near": verdict.near_matches, "excluded": excluded})
        if excluded:                    # the agent gets one sentence; telemetry keeps the reason
            return self._reject(candidate, node_id, [EXCLUDED_CLIP], excluded=excluded)

        has_segments = bool(caption_json.get("segments"))
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_ingest.py tests/test_leakage.py tests/test_data_tools.py -q`
Expected: PASS. If a test still expects the old `matches WBench case` text, change its expectation to `EXCLUDED_CLIP`.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/data/ingest.py tests/test_ingest.py
git commit -m "ingest: refuse blocked sources and copied evaluation text, with one neutral reason"
```

---

### Task 4: GPU tools: prompt check, scrubbed output, AlayaWorld's own interface

**Files:**
- Modify: `kernel/ar_kernel/tools/server.py` (`ToolKit`)
- Modify: `kernel/ar_kernel/tools/gpu_jobs.py` (`GpuJob.submit`, item schemas, `register_gpu_tools`)
- Modify: `kernel/ar_kernel/tools/rollouts.py` (module docstring, constants, `case_json`, `AlayaWorldBackend.description`, `_check_item`)
- Modify: `kernel/ar_kernel/run_kit.py:63`
- Test: `tests/test_tool_server.py`, `tests/test_gpu_jobs.py`, `tests/test_rollouts.py`

**Interfaces:**
- Consumes: `copies_held_out`, `strings`, `scrub`, `EXCLUDED_PROMPT` from Task 1.
- Produces:
  - `ToolKit(registry, recorder, scrub_names: list[str] = ())`.
  - `rollout_alayaworld` item: `{image, viewpoint: 'first_person'|'third_person', scene_prompt, character_prompt?, viewpoint_prompt?, subject_mask?, turns: [{action, event?, subject_action?, viewpoint_change?}]}`.
  - `rollouts.VIEWPOINTS`, `rollouts.TURN_KEYS` (item key → the renderer's interaction type), `gpu_jobs.WorldItems`, `gpu_jobs.items_schema(properties, required)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tool_server.py` (it already imports what is needed):

```python
def test_tool_results_and_errors_have_blocked_names_replaced(tmp_path):
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    kit = ToolKit(reg, rec, scrub_names=["wbench"])
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path)
    request = type("R", (), {"headers": {"authorization": f"Bearer {caller.token}"}})()
    ctx = type("C", (), {"request_context": type("RC", (), {"request": request})()})()

    def fail(_caller):
        raise ToolError("worker log: [run_wbench] failed")
    assert asyncio.run(kit.call(ctx, "t", {}, lambda c: {"log": "[run_WBench] ok"})) == {"log": "[run_[removed]] ok"}
    with pytest.raises(ToolError, match=r"\[run_\[removed\]\] failed"):
        asyncio.run(kit.call(ctx, "t", {}, fail))
    errors = [rec.load_payload(e["payload"]) for e in rec.read_events("n1") if e["type"] == "tool.error"]
    assert "run_wbench" in errors[0]["error"]                 # telemetry keeps the real text
```

Append to `tests/test_gpu_jobs.py`:

```python
def test_a_prompt_copied_from_the_evaluation_is_refused_before_the_job_is_queued(tmp_path):
    from ar_kernel.isolation import EXCLUDED_PROMPT
    from ar_kernel.tools.images import ImageBackend
    from ar_kernel.tools.server import ToolError
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path)
    backend = ImageBackend(KernelConfig.load(), tmp_path / "run", [0], reg, rec)

    class Queue:
        submitted = 0

        def submit(self, caller, name, args):
            self.submitted += 1
            return "job1"
    q = Queue()
    copied = "A hall. Torch sconces on both walls cast flickering orange light."
    with pytest.raises(ToolError, match="item 1: " + EXCLUDED_PROMPT):
        backend.submit(q, caller, {"items": [{"prompt": "a quiet beach", "seed": 1}, {"prompt": copied, "seed": 2}]})
    assert q.submitted == 0
    assert backend.submit(q, caller, {"items": [{"prompt": "a quiet beach", "seed": 1}]}) == {"job_id": "job1"}
    assert [e["type"] for e in rec.read_events("n1")].count("isolation.refused") == 1
```

(`KernelConfig`, `Recorder`, `TokenRegistry` and `pytest` are already imported in that file; add an import if one is missing.)

In `tests/test_rollouts.py`:
- In `first_person`, `third_person` and the two real-GPU item lists (lines ~43-55, ~610-622, ~716), rename the item keys: `"perspective"` → `"viewpoint"`, `"environment_prompt"` → `"scene_prompt"`, `"perspective_prompt"` → `"viewpoint_prompt"`, `"event_edit"` → `"event"`, `"perspective_switch"` → `"viewpoint_change"`. Keyword overrides follow: `first_person(viewpoint="top_down")`, `first_person(scene_prompt="")`, `first_person(scene_prompt="PRECACHE_HANG street")`, `first_person(scene_prompt="NO_VIDEO here")`.
- In the `test_submit_refuses` table the expected fragments become `"viewpoint"` and `"scene_prompt"`.
- Leave every assertion about the staged renderer file as it is (`case["settings"]["perspective"]`, `{"type": "perspective_switch", ...}`, `m["wbench_perspective"]`): the kernel still writes the renderer's schema.
- Rename `test_submit_accepts_every_wbench_action_and_fills_defaults` to `test_submit_accepts_every_action_and_fills_defaults`.

In `tests/test_gpu_jobs.py::test_listed_item_schemas_type_every_field` replace the AlayaWorld lines with:

```python
    alaya = tools["rollout_alayaworld"].input_schema["properties"]["items"]["items"]
    from ar_kernel.tools.rollouts import VIEWPOINTS
    assert alaya["properties"]["viewpoint"]["enum"] == list(VIEWPOINTS)
    assert set(alaya["required"]) == {"image", "viewpoint", "scene_prompt", "turns"}
    turn = alaya["properties"]["turns"]["items"]
    assert turn["properties"]["action"]["type"] == "string" and turn["required"] == ["action"]
    assert set(turn["properties"]) == {"action", "event", "subject_action", "viewpoint_change"}
    for field in (*alaya["properties"].values(), *turn["properties"].values()):
        assert field["description"]                    # every item field says what it is
    assert tools["rollout_alayaworld"].input_schema["properties"]["rounds_per_turn"]["description"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_tool_server.py tests/test_gpu_jobs.py tests/test_rollouts.py -q`
Expected: FAIL (`scrub_names` unknown; no refusal; `VIEWPOINTS` missing; old item keys expected)

- [ ] **Step 3: Scrub in `ToolKit`**

In `server.py` add `from ..isolation import scrub` and change `ToolKit`:

```python
class ToolKit:
    def __init__(self, registry, recorder, scrub_names: list[str] = ()) -> None:
        self.registry = registry
        self.recorder = recorder
        self.scrub_names = list(scrub_names)      # replaced in everything a tool returns to an agent
```

In `call`, the three exits become:

```python
        try:
            result = await asyncio.to_thread(fn, caller)
        except ToolError as exc:
            self.recorder.event("tool.error", parent_span_id=span, tool=name,
                                 duration_s=time.monotonic() - started,
                                 payload={"tool": name, "error": str(exc)}, **base)
            raise ToolError(scrub(str(exc), self.scrub_names)) from None
        except Exception as exc:                          # noqa: BLE001 -- contain kernel bugs
            self.recorder.event("tool.error", parent_span_id=span, tool=name,
                                 duration_s=time.monotonic() - started,
                                 payload={"tool": name, "error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc()}, **base)
            raise ToolError(scrub(f"{type(exc).__name__}: {exc}", self.scrub_names)) from exc
        self.recorder.event("tool.result", parent_span_id=span, tool=name,
                             duration_s=time.monotonic() - started,
                             payload={"tool": name, "result": result}, **base)
        return scrub(result, self.scrub_names)
```

In `run_kit.py` line 63: `kit, mcp = ToolKit(registry, recorder, cfg.get("isolation.blocked_names") or []), new_mcp()`.

- [ ] **Step 4: Prompt check in `GpuJob.submit`**

In `gpu_jobs.py` add `from ..isolation import EXCLUDED_PROMPT, copies_held_out, strings`, and in `submit`, right after `self.check_args(args)`:

```python
        for n, item in enumerate(items):
            if copies_held_out(self.cfg, *strings(item)):       # before a GPU is scheduled
                self.recorder.event("isolation.refused", node=caller.node, phase=caller.phase,
                                    attempt=caller.attempt, component="tools", tool=self.name,
                                    payload={"item": n})
                raise ToolError(f"item {n}: {EXCLUDED_PROMPT}")
```

- [ ] **Step 5: Item schemas with descriptions**

In `gpu_jobs.py` rename `_items` to `items_schema` (same body; Task 8 reuses it) and replace the schema constants:

```python
def _str(description: str) -> dict:
    return {"type": "string", "description": description}


_PROMPT = _str("what the clip shows")
_SEED = {"type": "integer", "description": "random seed; the same item and seed give the same output"}
_FRAME = _str("first frame: an image file under /workspace (any size; it is cropped and resized)")
ImageItems = Annotated[list[dict[str, Any]], items_schema(
    {"prompt": _str("what the image shows"), "seed": _SEED}, ["prompt", "seed"])]
ClipItems = Annotated[list[dict[str, Any]], items_schema(
    {"prompt": _PROMPT, "image": _FRAME, "seed": _SEED}, ["prompt", "seed"])]
_TURN = {"type": "object", "additionalProperties": False, "required": ["action"], "properties": {
    "action": _str("camera move for this turn: W, A, S, D (translate), left, right, up, down (rotate), stop, "
                   "or two joined with '+', e.g. 'W+left'"),
    "event": _str("something that happens in the scene during this turn"),
    "subject_action": _str("something the subject does during this turn"),
    "viewpoint_change": _str("a change of viewpoint during this turn: 'fp_to_tp', 'tp_to_fp', 'fp_to_scope', "
                             "or 'tp_to_tp: <the new view>'")}}
WorldItems = Annotated[list[dict[str, Any]], items_schema(
    {"image": _str("first frame: an image file under /workspace (.jpg, .png, ...; any size)"),
     "viewpoint": {"type": "string", "enum": ["first_person", "third_person"],     # rollouts.VIEWPOINTS
                   "description": "whose eyes the first frame is seen through"},
     "scene_prompt": _str("the scene: place, objects, light"),
     "character_prompt": _str("the subject, if there is one"),
     "viewpoint_prompt": _str("how the camera sees the scene at the start"),
     "subject_mask": _str("an image under /workspace, white where the subject is in the first frame"),
     "turns": {"type": "array", "items": _TURN,
               "description": "the clip, turn by turn; each turn lasts `rounds_per_turn` rounds"}},
    ["image", "viewpoint", "scene_prompt", "turns"])]
```

In `register_gpu_tools`, add `from pydantic import Field` to the imports and give each parameter a description. Replace the five tool signatures:

```python
        async def annotate_camera(
                paths: Annotated[list[str], Field(description="mp4 files under /workspace, at most 1200 frames each")],
                ctx: Context) -> dict[str, Any]:
```

```python
        async def rollout_alayaworld(
                items: WorldItems, ctx: Context,
                variant: Annotated[str | None, Field(description="sampler; default the first one listed above")] = None,
                rounds_per_turn: Annotated[int | None, Field(description="1..3, default 3; a round is 32 frames at 24 fps")] = None,
                seed: Annotated[int | None, Field(description="default 42; one seed for the whole job")] = None,
                node: Annotated[str | None, Field(description="render with this scored node's fine-tune instead of the released model")] = None,
        ) -> dict[str, Any]:
```

```python
        async def generate_images(
                items: ImageItems, ctx: Context,
                width: Annotated[int | None, Field(description="multiple of 16 in 256..1920, default 1280")] = None,
                height: Annotated[int | None, Field(description="multiple of 16 in 256..1920, default 720")] = None,
        ) -> dict[str, Any]:
```

```python
        async def rollout_wan22(
                items: ClipItems, ctx: Context,
                frames: Annotated[int | None, Field(description="4k+1 frames at 24 fps; one value for the job")] = None,
        ) -> dict[str, Any]:
```

```python
        async def rollout_ltx25(
                items: ClipItems, ctx: Context,
                variant: Annotated[str | None, Field(description="default the first one listed above")] = None,
                frames: Annotated[int | None, Field(description="8k+1 frames at 24 fps; one value for the job")] = None,
                height: Annotated[int | None, Field(description="with width, one of the listed resolutions")] = None,
                width: Annotated[int | None, Field(description="with height, one of the listed resolutions")] = None,
        ) -> dict[str, Any]:
```

The function bodies are unchanged.

- [ ] **Step 6: AlayaWorld's interface in `rollouts.py`**

Replace the module docstring's first two paragraphs with:

```python
"""rollout_alayaworld: clips rendered by AlayaWorld from the agent's own items, through WorldModel's
turn-based render path (scripts/tools/run_wbench.py, configs/wbench_full.yaml).

The agent writes items (first frame, viewpoint, scene and character text, per-turn camera move and
instruction). The kernel stages them in the render script's input schema, pre-encodes their prompts,
renders them, and publishes each clip with its round boundaries on the per_chunk grid (25 + 32k
frames) and one caption segment per round from the prompts the render used. Nothing the agent sees
names that script, its schema or the evaluation.
```

Replace the constants:

```python
# Item key of a turn's instruction -> the render script's interaction type.
TURN_KEYS = {"subject_action": "subject_action", "event": "event_edit", "viewpoint_change": "perspective_switch"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")                 # alaya/data/wbench.py _IMAGE_EXTS
VIEWPOINTS = ("first_person", "third_person")
VARIANTS = ("dmd4", "ar30")
VARIANT_TEXT = {"dmd4": "4 sampling steps, fast", "ar30": "30 sampling steps, slower"}
```

(delete `TURN_TYPES` and `PERSPECTIVES`). Replace `case_json`:

```python
def case_json(index: int, item: dict, image_rel: str, mask_rel: str | None) -> dict:
    """One item in the render script's input schema: a navigation entry per turn, plus that turn's
    instruction under the script's own type name."""
    interactions = []
    for turn, t in enumerate(item["turns"], 1):
        interactions.append({"type": "navigation", "action": t["action"], "turn": turn})
        interactions += [{"type": kind, "action": t[key], "turn": turn} for key, kind in TURN_KEYS.items() if t.get(key)]
    return {"id": str(index), "environment_prompt": item["scene_prompt"],
            "character_prompt": item.get("character_prompt", ""),
            "perspective_prompt": item.get("viewpoint_prompt", ""),
            "settings": {"perspective": item["viewpoint"], "subject": {"type": "unknown", "desc": ""},
                         "tracking_object": None, "initial_image": image_rel, "subject_mask": mask_rel},
            "interactions": interactions, "metric_list": []}
```

Replace `AlayaWorldBackend.description` (argument and item details now live in the schema):

```python
    description = (
        "Render clips with AlayaWorld itself: the released model that every node fine-tunes, or with `node` a "
        "scored node's fine-tune. A GPU job: returns {job_id} at once; collect with job_wait. Each item is a "
        "first frame, the scene and character text, and a list of turns. A turn moves the camera and may add one "
        "instruction: an event in the scene, an action of the subject, or a change of viewpoint. A turn keeps its "
        "camera move and its text for all its rounds. Samplers (`variant`): {variants}. Camera moves steer "
        "translation reliably, rotation (turns, orbits) only weakly. Each result item gives a `candidate` (mp4, "
        "caption with one segment per round, provenance) with NO pose and no camera_motion: run annotate_camera "
        "on candidate.video, then data_ingest it with that pose and camera_motion 'moving' (eligible for "
        "video_timed_prompts_camera:per_chunk). Metadata, not labels: `commanded_camera` (npz of the camera path "
        "the moves commanded, one pose per frame; not what the video shows), `actions` and `turn_segments` "
        "(frame ranges in the published clip). At most 100 items and 9 turns per item.")
```

In `_check_item` rename the checks:

```python
        if item.get("viewpoint") not in VIEWPOINTS:
            bad(f"viewpoint must be one of {VIEWPOINTS}: got {item.get('viewpoint')!r}")
        if not isinstance(item.get("scene_prompt"), str) or not item["scene_prompt"].strip():
            bad("scene_prompt must be a non-empty string")
        for key in ("character_prompt", "viewpoint_prompt"):
            if not isinstance(item.get(key, ""), str):
                bad(f"{key} must be a string")
```

and the last loop:

```python
            for key, value in turn.items():
                if key != "action" and (key not in TURN_KEYS or not isinstance(value, str)):
                    bad(f"turn {t}: {key!r} is not one of {tuple(TURN_KEYS)} with a text value")
```

Then check nothing else in the file reads the old keys:

Run: `grep -n "TURN_TYPES\|PERSPECTIVES\|item\[\"perspective\|environment_prompt\"\]\|CaseItems" kernel/ar_kernel/tools/*.py`
Expected: only the two lines of `case_json` that write `"environment_prompt"` / `"perspective_prompt"` for the renderer. Fix any other hit (`CaseItems` → `WorldItems`).

- [ ] **Step 7: Run the tests**

Run: `$PY -m pytest tests/test_tool_server.py tests/test_gpu_jobs.py tests/test_rollouts.py tests/test_images.py tests/test_annotate.py -q`
Expected: PASS. `test_gpu_jobs.py` asserts `"'dmd4'" in alaya and "'ar30'" not in alaya` on the description; that still holds (`{variants}` is filled the same way).

- [ ] **Step 8: Commit**

```bash
git add kernel/ar_kernel/tools kernel/ar_kernel/run_kit.py tests/test_tool_server.py tests/test_gpu_jobs.py tests/test_rollouts.py
git commit -m "GPU tools: refuse copied prompts, scrub output, and give AlayaWorld its own interface with described arguments"
```

---

### Task 5: Aliases and the metric guide

**Files:**
- Modify: `kernel/ar_kernel/eval/score.py` (append after `DIMENSION_METRICS`)
- Test: `tests/test_score.py`

**Interfaces:**
- Produces:
  - `AGENT_METRICS: dict[real name, (alias, dimension alias, what it measures)]`
  - `agent_metrics(metrics: dict) -> dict` — keys aliased.
  - `agent_aggregates(aggregates: dict | None) -> dict | None` — `{"metrics", "dimensions", "groups"}` with every name aliased; `None` stays `None`.
  - `metric_guide(weights: dict | None) -> dict[alias, {"dimension", "weight", "measures"}]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_score.py`)

```python
def test_every_metric_has_one_alias_and_no_alias_is_a_real_name():
    from ar_kernel.eval.score import AGENT_METRICS, DIMENSION_METRICS
    assert set(AGENT_METRICS) == set(DIMENSION_METRICS)
    aliases = [alias for alias, _, _ in AGENT_METRICS.values()]
    assert len(set(aliases)) == len(aliases) and not set(aliases) & set(DIMENSION_METRICS)
    assert {dim for _, dim, _ in AGENT_METRICS.values()} == {"quality", "consistency", "control", "description", "physics"}


def test_agent_aggregates_alias_metrics_dimensions_axes_and_groups():
    from ar_kernel.eval.score import agent_aggregates
    real = {"metrics": {"event_edit_adherence": 0.4, "aesthetic_quality": 0.7},
            "dimensions": {"interaction": 0.4, "quality": 0.7, "setting": 0.5, "physical": 0.6},
            "strata": {"interaction_type": {"event_edit": {"interaction": 0.4}, "navigation": {"quality": 0.7}},
                       "category": {"Indoor": {"physical": 0.6}},
                       "perspective": {"first_person": {"setting": 0.5}}}}
    assert agent_aggregates(real) == {
        "metrics": {"follows_event_instruction": 0.4, "frame_aesthetics": 0.7},
        "dimensions": {"control": 0.4, "quality": 0.7, "description": 0.5, "physics": 0.6},
        "groups": {"instruction_kind": {"event": {"control": 0.4}, "camera_move": {"quality": 0.7}},
                   "category": {"Indoor": {"physics": 0.6}},
                   "viewpoint": {"first_person": {"description": 0.5}}}}
    assert agent_aggregates(None) is None


def test_the_metric_guide_carries_dimension_weight_and_meaning_only():
    from ar_kernel.eval.score import metric_guide
    guide = metric_guide({"causal_fidelity": 4.5})
    assert len(guide) == 22
    assert guide["cause_and_effect"] == {"dimension": "physics", "weight": 4.5,
                                         "measures": guide["cause_and_effect"]["measures"]}
    assert guide["frame_aesthetics"]["weight"] == 1.0
    text = " ".join(g["measures"] for g in guide.values()).lower()
    assert not any(word in text for word in ("judge", "question", "yes/no", "case", "frames per second"))
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_score.py -q -k "alias or guide"`
Expected: FAIL, `ImportError: cannot import name 'AGENT_METRICS'`

- [ ] **Step 3: Implement** (in `score.py`, after the `DIMENSION_METRICS` list)

```python
# What agents are shown instead of the benchmark's own names. Real name -> (alias, dimension alias,
# what it measures). Applied in memory when a context is built; nothing on disk uses an alias.
AGENT_METRICS = {
    "aesthetic_quality": ("frame_aesthetics", "quality", "how good single frames look: composition, colour, light"),
    "imaging_quality": ("frame_clarity", "quality", "frames free of blur, noise and compression artefacts"),
    "temporal_flickering": ("flicker_free", "quality", "no flicker between neighbouring frames"),
    "dynamic_degree": ("motion_amount", "quality", "how much the video moves; a near-static video scores low"),
    "motion_smoothness": ("smooth_motion", "quality", "motion that is smooth from frame to frame"),
    "hpsv3_quality": ("human_preference", "quality", "how a human-preference model rates the frames"),
    "background_consistency": ("background_stability", "consistency", "the background keeps its look over time"),
    "segment_continuity": ("no_hard_cuts", "consistency", "the video has no abrupt cut"),
    "perspective_consistency": ("subject_framing_stability", "consistency",
                                "the followed subject stays at a steady place in the frame"),
    "subject_consistency": ("subject_stability", "consistency", "the subject keeps its look over time"),
    "geometric_consistency": ("geometry_stability", "consistency", "the scene's 3D shape agrees between frames"),
    "photometric_consistency": ("appearance_stability", "consistency",
                                "the same surface keeps its colour and light between frames"),
    "spatial_consistency": ("revisit_match", "consistency",
                            "a place looks the same when the camera comes back to it"),
    "gated_spatial_consistency": ("revisit_match_strict", "consistency",
                                  "the same, counted less when the view barely changed on the way"),
    "navigation_trajectory": ("camera_path_accuracy", "control", "the camera follows the commanded moves"),
    "event_edit_adherence": ("follows_event_instruction", "control",
                             "an instructed event happens in the scene, fully and with the right details"),
    "subject_action_adherence": ("follows_subject_action", "control",
                                 "the subject does an instructed action, fully and naturally"),
    "perspective_switch_adherence": ("follows_viewpoint_change", "control",
                                     "an instructed change of viewpoint happens and ends in a valid view"),
    "scene_adherence": ("scene_matches_description", "description", "the scene matches its text"),
    "subject_adherence": ("subject_matches_description", "description", "the subject matches its text"),
    "visual_plausibility": ("looks_plausible", "physics", "the video looks physically plausible"),
    "causal_fidelity": ("cause_and_effect", "physics", "objects and characters obey physics and cause and effect"),
}
AGENT_DIMENSIONS = {"interaction": "control", "setting": "description", "physical": "physics"}
AGENT_AXES = {"interaction_type": "instruction_kind", "perspective": "viewpoint"}
AGENT_GROUPS = {"navigation": "camera_move", "event_edit": "event", "perspective_switch": "viewpoint_change"}


def agent_metrics(metrics: dict) -> dict:
    return {AGENT_METRICS[name][0]: value for name, value in metrics.items()}


def _agent_dimensions(dimensions: dict) -> dict:
    return {AGENT_DIMENSIONS.get(name, name): value for name, value in dimensions.items()}


def agent_aggregates(aggregates: dict | None) -> dict | None:
    """`aggregates()` as an agent sees it: every metric, dimension, group axis and group renamed."""
    if not aggregates:
        return None
    return {"metrics": agent_metrics(aggregates.get("metrics") or {}),
            "dimensions": _agent_dimensions(aggregates.get("dimensions") or {}),
            "groups": {AGENT_AXES.get(axis, axis): {AGENT_GROUPS.get(name, name): _agent_dimensions(dims)
                                                    for name, dims in groups.items()}
                       for axis, groups in (aggregates.get("strata") or {}).items()}}


def metric_guide(weights: dict | None) -> dict:
    """What each metric measures and how much it weighs in the score. Nothing about how it is judged."""
    return {alias: {"dimension": dimension, "weight": float((weights or {}).get(name, 1.0)), "measures": measures}
            for name, (alias, dimension, measures) in AGENT_METRICS.items()}
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_score.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/eval/score.py tests/test_score.py
git commit -m "score: the names agents see for metrics, dimensions and groups, and the metric guide"
```

---
### Task 6: Contexts per phase

**Files:**
- Modify: `kernel/ar_kernel/archive/nodes.py:5-9` (statuses)
- Modify: `kernel/ar_kernel/context_bundle.py` (everything from `FORMAT_RULES` to `build_recipe_context`)
- Modify: `kernel/ar_kernel/tools/data_tools.py` (`pool_clips`, `DataTools.query`)
- Modify: `kernel/ar_kernel/agent_phase.py:17,61` (`UNFINISHED` → `NOT_SHOWN`)
- Modify: `contract/ar_contract/models.py` (`RecipeContext`)
- Test: `tests/test_context_bundle.py`, `tests/test_data_tools.py`

**Interfaces:**
- Consumes: `agent_metrics`, `agent_aggregates`, `metric_guide` from Task 5.
- Produces:
  - `nodes.NOT_SHOWN = ("running", "interrupted", "quarantined")`; `"quarantined"` is a valid status.
  - `lineage(conn, run_dir, repo, node_id, phase)`, `siblings(conn, run_dir, repo, parent_id, phase)`, `archive_summary(conn, phase)`; `phase` is `"edit_self"` or `"improve_recipe"`.
  - `improve_recipe` node entry keys: `node_id, status, error, score, metrics, aggregates, data, recipe, rationale`.
  - `edit_self` node entry keys: `node_id, status, error, edit, code_diff_stats, process` (`process["attempts"]` is `[{"phase", "attempt", "outcome"}]`).
  - `RecipeContext.metric_guide: dict`; `RecipeContext.format_rules` is gone.
  - `data_tools.pool_clips(conn) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_context_bundle.py`:

Change the import to `from ar_kernel.context_bundle import (archive_summary, build_edit_context, build_recipe_context, lineage, siblings, write_bundle)` and add `from ar_kernel.eval.score import agent_aggregates`. Then replace the tests named below with these bodies (the others keep working once `lineage` / `archive_summary` calls get their `phase` argument in the same way):

```python
def test_a_data_entry_carries_scores_under_agent_names_and_no_code_or_process(world):
    conn, repo, run = world
    [entry] = lineage(conn, run, repo, "root", "improve_recipe")
    assert sorted(entry) == ["aggregates", "data", "error", "metrics", "node_id", "rationale", "recipe",
                             "score", "status"]
    assert entry["score"] == 0.78 and entry["metrics"] == {"frame_aesthetics": 0.78}
    assert entry["aggregates"] == agent_aggregates(AGGREGATES) and "instruction_kind" in entry["aggregates"]["groups"]
    assert entry["rationale"] == "released checkpoint"
    assert json.loads((run / "nodes" / "root" / "eval" / "aggregates.json").read_text()) == AGGREGATES   # disk unchanged


def test_an_edit_entry_carries_code_and_process_and_no_score(world):
    conn, repo, run = world
    NodeStore(conn).add_attempt("root", "edit_self", 1, "contract_failed", {})
    [entry] = lineage(conn, run, repo, "root", "edit_self")
    assert sorted(entry) == ["code_diff_stats", "edit", "error", "node_id", "process", "status"]
    assert entry["edit"] == {"summary": "s"}
    assert entry["process"] == {"attempts": [{"phase": "edit_self", "attempt": 1, "outcome": "contract_failed"}]}
    ctx = build_edit_context(conn=conn, run_dir=run, repo=repo, parent_id="root", attempt=1, max_attempts=3,
                             retry=None, nodes_remaining=9)
    assert "0.78" not in ctx.model_dump_json()                  # no score anywhere in an edit context
    assert ctx.archive == {"nodes": [{"node_id": "root", "parent_id": None, "status": "scored", "depth": 0},
                                     {"node_id": "n1", "parent_id": "root", "status": "running", "depth": 1}]}


def test_archive_summary_names_the_best_node(world):
    conn, _, _ = world
    s = archive_summary(conn, "improve_recipe")
    assert s["best"] == {"node_id": "root", "score": 0.78} and s["n_scored"] == 1
    assert "error" not in s["nodes"][0]


def test_kernel_failures_show_a_fixed_error_and_agent_failures_their_own(world):
    conn, repo, run = world
    nodes = NodeStore(conn)
    nodes.set_status("n1", "invalid_code")
    nodes.set_fields("n1", error="contract import failed")
    assert lineage(conn, run, repo, "n1", "edit_self")[-1]["error"] == "contract import failed"
    nodes.set_status("n1", "eval_failed")
    nodes.set_fields("n1", error="RuntimeError: wbench gpu failed (rc=1)")
    for phase in ("edit_self", "improve_recipe"):
        entry = lineage(conn, run, repo, "n1", phase)[-1]
        assert entry["error"] == "the kernel's evaluation of this node failed"
        if phase == "improve_recipe":                          # failed before scoring: nothing to alias
            assert entry["aggregates"] is None and entry["metrics"] == {}


def test_a_quarantined_node_is_in_no_context_and_its_clips_leave_the_pool(world):
    from ar_kernel.archive.clips import ClipStore
    from ar_kernel.tools.data_tools import pool_clips
    conn, repo, run = world
    nodes = NodeStore(conn)
    nodes.create("n2", "root", 1), nodes.set_status("n2", "quarantined")
    clip = {"clip_id": "a" * 64, "video_digest": "v", "caption_digest": "c", "pose_digest": None,
            "camera_motion": "static", "metadata": {}, "formats": [], "warnings": [],
            "provenance": {"kind": "derived"}, "license": None, "derived_from": []}
    ClipStore(conn).add({**clip, "ingested_by": "n2"})
    ClipStore(conn).add({**clip, "clip_id": "b" * 64, "ingested_by": "n1"})
    assert [c["clip_id"] for c in pool_clips(conn)] == ["b" * 64]
    for phase in ("edit_self", "improve_recipe"):
        assert siblings(conn, run, repo, "root", phase) == []
        assert "n2" not in [n["node_id"] for n in archive_summary(conn, phase)["nodes"]]
    ctx = build_recipe_context(cfg=CFG, conn=conn, run_dir=run, repo=repo, node_id="n1", parent_id="root",
                               attempt=1, max_attempts=3, retry=None, nodes_remaining=1, n_gpus=4, tools=[])
    assert ctx.clip_pool_size == 1 and [c["clip_id"] for c in ctx.clip_pool] == ["b" * 64]
```

In `test_recipe_context_carries_rules_allowlists_and_retry` replace the `format_rules` line with:

```python
    assert not hasattr(ctx, "format_rules")
    assert ctx.metric_guide["cause_and_effect"]["weight"] == 4.5 and len(ctx.metric_guide) == 22
```

Delete `test_lineage_is_root_first_and_carries_artifacts`, `test_lineage_and_archive_summary_carry_component_and_error` and `test_lineage_entries_carry_the_process_digest` (replaced above). In `test_siblings_are_the_parents_finished_children` pass `"improve_recipe"` to `siblings(...)`.

If `ClipStore.add` needs other columns than the dict above, copy the dict that `Ingestor._check_and_store` passes to `self.clips.add` and keep the two `ingested_by` values.

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_context_bundle.py -q`
Expected: FAIL (`lineage()` takes 4 positional arguments; `quarantined` is an unknown status)

- [ ] **Step 3: Statuses**

`kernel/ar_kernel/archive/nodes.py`, replace lines 5-9:

```python
# interrupted: unfinished when the loop stopped (forced stop, kernel death, spent budget); kept
# untouched, never resumed, never a parent, not counted toward max_nodes (user decision 2026-09-27).
# quarantined: an agent phase reached for the evaluation set; terminal, and hidden from every agent.
STATUSES = {"running", "scored", "invalid_code", "invalid_recipe", "train_failed", "eval_failed",
            "crashed", "interrupted", "quarantined"}
NOT_SHOWN = ("running", "interrupted", "quarantined")       # nodes agents never see: no context entry, no /nodes mount
```

In `agent_phase.py` replace both uses of `UNFINISHED` with `NOT_SHOWN` (the import on line 17 and the filter on line 61).

- [ ] **Step 4: The clip pool**

In `data_tools.py` add below `_parent_commit`:

```python
def pool_clips(conn) -> list[dict]:
    """The clips agents may use: all of them except those a quarantined node ingested."""
    hidden = {n["node_id"] for n in NodeStore(conn).all() if n["status"] == "quarantined"}
    return [c for c in ClipStore(conn).all() if c.get("ingested_by") not in hidden]
```

and in `DataTools.query` replace `clips = ClipStore(conn).all()` with `clips = pool_clips(conn)`.

- [ ] **Step 5: The contract model**

In `contract/ar_contract/models.py`, in `RecipeContext`, delete the `format_rules: str = ""` line and add after `recipe_guide`:

```python
    metric_guide: dict[str, Any] = Field(default_factory=dict)   # metric -> dimension, weight, what it measures
```

- [ ] **Step 6: `context_bundle.py`**

Replace the imports and everything from `FORMAT_RULES` down to (not including) `write_bundle`:

```python
"""What an agent sees: built from the archive, written to /context. Each phase gets what its mission
needs. improve_recipe: how earlier nodes scored, on what data, with which recipe. edit_self: how the
agent code changed and how each run went, and never a score."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from ar_contract.models import EditContext, RecipeContext

from .archive.clips import ClipStore
from .archive.nodes import NOT_SHOWN, NodeStore
from .config import run_config_path
from .eval.score import agent_aggregates, agent_metrics, metric_guide
from .process_digest import process_digest
from .tools.data_tools import clip_record, dataset_stats, pool_clips, scores_by_clip
from .train.recipe import RECIPE_RULES, TUNABLE_KEYS
from .train.recipe_guide import recipe_guide

# A failure inside the kernel is not the agent's doing, and its raw text can name what agents never see.
KERNEL_FAILURES = {"eval_failed": "the kernel's evaluation of this node failed",
                   "crashed": "the kernel failed while running this node"}


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
    return dataset_stats(json.loads(row["manifest"]), ClipStore(conn).all())      # an existing commit, whole


def _entry(conn, run_dir: Path, repo, node: dict, parent_commit: str | None, phase: str) -> dict:
    nid = node["node_id"]
    entry = {"node_id": nid, "status": node["status"],
             "error": KERNEL_FAILURES.get(node["status"], node["error"])}
    if phase == "improve_recipe":
        recipe_path, rationale = _node_file(run_dir, nid, "recipe.yaml"), _node_file(run_dir, nid, "rationale.md")
        return {**entry, "score": node["score"], "metrics": agent_metrics(node["metrics"]),
                "aggregates": agent_aggregates(_read_json(_node_file(run_dir, nid, "eval/aggregates.json"))),
                "data": _data_stats(conn, node["data_commit"]),
                "recipe": yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else None,
                "rationale": rationale.read_text() if rationale.exists() else None}
    attempts = [{"phase": p, "attempt": a["idx"], "outcome": a["outcome"]}
                for p in ("edit_self", "improve_recipe") for a in NodeStore(conn).attempts(nid, p)]
    return {**entry, "edit": _read_json(_node_file(run_dir, nid, "edit.json")),
            "code_diff_stats": (repo.diff_stats(parent_commit, node["agent_commit"])
                                if parent_commit and node["agent_commit"] else []),
            "process": {**process_digest(run_dir, nid), "attempts": attempts}}


def lineage(conn, run_dir: Path, repo, node_id: str, phase: str) -> list[dict]:
    nodes = NodeStore(conn)
    chain = []
    current = node_id
    while current:
        chain.append(nodes.get(current))
        current = chain[-1]["parent_id"]
    chain.reverse()
    return [_entry(conn, run_dir, repo, node, chain[i - 1]["agent_commit"] if i else None, phase)
            for i, node in enumerate(chain)]


def siblings(conn, run_dir: Path, repo, parent_id: str, phase: str) -> list[dict]:
    """The finished children of the parent: what was already tried from the same starting point."""
    parent_commit = NodeStore(conn).get(parent_id)["agent_commit"]
    return [_entry(conn, run_dir, repo, node, parent_commit, phase) for node in NodeStore(conn).all()
            if node["parent_id"] == parent_id and node["status"] not in NOT_SHOWN]


def archive_summary(conn, phase: str) -> dict:
    nodes = [n for n in NodeStore(conn).all() if n["status"] != "quarantined"]
    rows = [{"node_id": n["node_id"], "parent_id": n["parent_id"], "status": n["status"], "depth": n["depth"]}
            for n in nodes]
    if phase == "edit_self":
        return {"nodes": rows}
    scored = [n for n in nodes if n["status"] == "scored" and n["score"] is not None]
    best = max(scored, key=lambda n: n["score"], default=None)
    return {"nodes": [{**row, "score": n["score"], "subtree_value": n["subtree_value"]}
                      for row, n in zip(rows, nodes)],
            "n_scored": len(scored),
            "best": {"node_id": best["node_id"], "score": best["score"]} if best else None}


def clip_pool_summary(conn, cap: int = 2000) -> list[dict]:
    usage = scores_by_clip(conn)
    return [clip_record(c, usage) for c in pool_clips(conn)[-cap:]]


def build_edit_context(*, conn, run_dir: Path, repo, parent_id: str, attempt: int, max_attempts: int,
                       retry: dict | None, nodes_remaining: int, dry_run: bool = False) -> EditContext:
    return EditContext(lineage=lineage(conn, run_dir, repo, parent_id, "edit_self"),
                       siblings=siblings(conn, run_dir, repo, parent_id, "edit_self"),
                       archive=archive_summary(conn, "edit_self"),
                       nodes_remaining=nodes_remaining, attempt=attempt, max_attempts=max_attempts,
                       retry=retry, dry_run=dry_run)


def build_recipe_context(*, cfg, conn, run_dir: Path, repo, node_id: str, parent_id: str, attempt: int,
                         max_attempts: int, retry: dict | None, nodes_remaining: int, n_gpus: int,
                         tools: list[str], dry_run: bool = False) -> RecipeContext:
    parent = NodeStore(conn).get(parent_id)
    recipe_path = _node_file(run_dir, parent_id, "recipe.yaml")
    base = yaml.safe_load(run_config_path(cfg, run_dir, "base_recipe.yaml").read_text())
    return RecipeContext(
        lineage=lineage(conn, run_dir, repo, parent_id, "improve_recipe"),
        siblings=siblings(conn, run_dir, repo, parent_id, "improve_recipe"),
        archive=archive_summary(conn, "improve_recipe"),
        nodes_remaining=nodes_remaining, attempt=attempt, max_attempts=max_attempts, retry=retry,
        dry_run=dry_run, clip_pool=clip_pool_summary(conn), clip_pool_size=len(pool_clips(conn)),
        parent_data_commit=parent["data_commit"],
        parent_recipe=yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else {},
        base_recipe=base, recipe_guide=recipe_guide(base),
        metric_guide=metric_guide(cfg.get("eval.score_weights")),
        tunable_rules={k: {"type": RECIPE_RULES[k][0], "min": RECIPE_RULES[k][1], "max": RECIPE_RULES[k][2]}
                       for k in sorted(TUNABLE_KEYS)},
        resolution_allowlist=[list(p) for p in cfg.get("train.resolution_allowlist")],
        lora_allowlist=[list(p) for p in cfg.get("train.lora_allowlist")],
        n_gpus=n_gpus, tools=list(tools))
```

Keep the text of the old `FORMAT_RULES` constant at hand: Task 8 moves it into the `data_formats` skill.

- [ ] **Step 7: Run the tests**

Run: `$PY -m pytest tests/test_context_bundle.py tests/test_data_tools.py tests/test_agent_phase.py tests/test_contract_package.py tests/test_archive_nodes.py -q`
Expected: PASS. A test elsewhere that passes `format_rules=` to `RecipeContext` or reads `ctx.format_rules` is fixed in Task 12 (`tests/test_seed_agent.py`); anywhere else, delete that argument.

- [ ] **Step 8: Commit**

```bash
git add kernel/ar_kernel contract/ar_contract/models.py tests/test_context_bundle.py tests/test_data_tools.py
git commit -m "contexts: each phase gets only what its mission needs; edit_self never sees a score"
```

---

### Task 7: A process digest the edit planner can reason from

**Files:**
- Modify: `kernel/ar_kernel/process_digest.py` (`_digest`; new `role_of`, `_rounds`)
- Modify: `kernel/ar_kernel/transcripts.py` (use `role_of`)
- Test: `tests/test_process_digest.py`

**Interfaces:**
- Produces `process_digest(run_dir, node_id) -> dict` with keys:
  - `phases: {name: seconds}` (unchanged)
  - `roles: [{"phase", "role", "conversations", "turns", "model_s", "compactions"}]` (replaces `llm`)
  - `tools: {tool: {"calls", "errors"}}`
  - `tool_errors: [{"tool", "count", "example"}]` (unchanged)
  - `local_errors: {...}` (unchanged)
  - `gpu_jobs: {"run", "failed", "gpu_s"}` (replaces `gpu_job_s`)
  - `ingest: {"accepted", "rejected", "reasons": [{"reason", "count"}]}`
  - `rounds: {phase: {"plans", "reports", "questions"}}`
  - `gates: {...}` (unchanged)
  - `role_of(body: dict) -> str` — the role label of an LLM request body.

- [ ] **Step 1: Write the failing tests**

In `tests/test_process_digest.py`, the two existing asserts on `process_digest(...)["llm"]` (lines ~51 and ~98) check conversations, turns and compactions per phase. Change each to the new shape; the first becomes:

```python
    roles = process_digest(tmp_path, "n1")["roles"]
    assert roles == [{"phase": "improve_recipe", "role": "conversation", "conversations": 3, "turns": 6,
                      "model_s": 0, "compactions": 1}]
```

and the second the same way with its own numbers (the requests of these tests offer no `submit_` tool, so the role is `"conversation"`). Then append:

```python
def test_roles_tools_jobs_ingest_and_rounds(tmp_path):
    rec = _rec(tmp_path)
    base = dict(node="n1", phase="improve_recipe", attempt=1)

    def turn(conv, tool, latency):
        body = {"messages": [{"role": "user", "content": conv}],
                "tools": [{"type": "function", "function": {"name": tool}}]}
        rec.event("llm.request", conversation_id=conv, payload={"body": body}, **base)
        rec.event("llm.response", conversation_id=conv, latency_s=latency, payload=_reply("ok"), **base)
    turn("c1", "submit_plan", 30.0), turn("c1", "submit_plan", 31.0), turn("c2", "submit_data_and_recipe", 10.0)
    for tool, failed in (("data_ingest", False), ("data_ingest", True), ("hf_search", False)):
        rec.event("tool.call", tool=tool, component="tools", payload={"tool": tool, "args": {}}, **base)
        if failed:
            rec.event("tool.error", tool=tool, component="tools", payload={"error": "bad path"}, **base)
    rec.event("job.finished", node="n1", job_id="j1", state="done", gpu_seconds=120.0, payload={})
    rec.event("job.finished", node="n1", job_id="j2", state="failed", gpu_seconds=5.0, payload={})
    rec.event("ingest.accepted", node="n1", phase="ingest", payload={"clip_id": "x"})
    for _ in range(2):
        rec.event("ingest.rejected", node="n1", phase="ingest", payload={"reasons": ["moving clips need poses/<id>.npz"]})
    ws = tmp_path / "nodes" / "n1" / "attempts" / "improve_recipe-1" / "workspace"
    ws.mkdir(parents=True)
    (ws / "plans.json").write_text(json.dumps([
        {"plan": {}, "report": "r", "questions": [{"question": "q", "answer": "a"}]}, {"plan": {}}]))
    d = process_digest(tmp_path, "n1")
    assert d["roles"] == [
        {"phase": "improve_recipe", "role": "plan", "conversations": 1, "turns": 2, "model_s": 61, "compactions": 0},
        {"phase": "improve_recipe", "role": "data_and_recipe", "conversations": 1, "turns": 1, "model_s": 10,
         "compactions": 0}]
    assert d["tools"] == {"data_ingest": {"calls": 2, "errors": 1}, "hf_search": {"calls": 1, "errors": 0}}
    assert d["gpu_jobs"] == {"run": 2, "failed": 1, "gpu_s": 125}
    assert d["ingest"] == {"accepted": 1, "rejected": 2,
                           "reasons": [{"reason": "moving clips need poses/<id>.npz", "count": 2}]}
    assert d["rounds"] == {"improve_recipe": {"plans": 2, "reports": 1, "questions": 1}}


def test_a_plans_file_of_another_shape_does_not_break_the_digest(tmp_path):
    rec = _rec(tmp_path)
    rec.event("tool.call", node="n1", phase="edit_self", tool="ask", component="tools", payload={})
    for attempt, text in ((1, "{not json"), (2, json.dumps({"rounds": "an edited agent's own shape"})),
                          (3, json.dumps(["a string", {"plan": 1, "questions": "many"}]))):
        ws = tmp_path / "nodes" / "n1" / "attempts" / f"edit_self-{attempt}" / "workspace"
        ws.mkdir(parents=True)
        (ws / "plans.json").write_text(text)
    d = process_digest(tmp_path, "n1")
    assert d["tools"] == {"ask": {"calls": 1, "errors": 0}}
    assert d["rounds"] == {"edit_self": {"plans": 1, "reports": 0, "questions": 0}}
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_process_digest.py -q`
Expected: FAIL with `KeyError: 'roles'`

- [ ] **Step 3: Implement**

In `process_digest.py`, replace the module docstring and add `role_of` and `_rounds` above `process_digest`:

```python
"""A digest of how a node's run went, for the edit planner: time per phase, model turns and time per
role, compactions, kernel tool calls and errors, failed local commands, GPU jobs, ingest results, and
the plans, reports and questions of each phase. Built from telemetry and the attempts' plans.json;
never raises."""
```

```python
MAX_REASONS = 5


def role_of(body: dict) -> str:
    """Named after the role's submit_<x> tool: the only role label the kernel can see."""
    names = [(t.get("function") or {}).get("name", "") for t in body.get("tools") or []]
    return next((n.removeprefix("submit_") for n in names if n.startswith("submit_")), "conversation")


def _rounds(run_dir: Path, node_id: str) -> dict:
    """Plans, reports back and questions per phase, from each attempt's plans.json. The file is the
    agent's own: anything that is not a list of round objects counts as nothing."""
    out: dict[str, dict] = {}
    for path in sorted((run_dir / "nodes" / node_id / "attempts").glob("*/workspace/plans.json")):
        try:
            rounds = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rounds = [r for r in rounds if isinstance(r, dict)] if isinstance(rounds, list) else []
        if not rounds:
            continue
        row = out.setdefault(path.parent.parent.name.rsplit("-", 1)[0], {"plans": 0, "reports": 0, "questions": 0})
        row["plans"] += len(rounds)
        row["reports"] += sum(1 for r in rounds if r.get("report"))
        row["questions"] += sum(len(r["questions"]) for r in rounds if isinstance(r.get("questions"), list))
    return out
```

Replace `_digest` from `started, phases, durations = ...` to the end:

```python
    started, phases, durations = {}, {}, Counter()
    requests: dict[tuple, list] = defaultdict(list)
    responses: dict[tuple, dict] = {}             # the last response of each conversation
    model_s: Counter = Counter()                  # LLM latency per conversation
    errors: dict[tuple, list] = defaultdict(list)
    tools: dict[str, dict] = {}
    gates, jobs, ingest, reasons = Counter(), Counter(), Counter(), Counter()
    for e in events:
        kind, phase = e["type"], e.get("phase")
        if kind == "phase.start":
            started[(phase, e.get("attempt"))] = e["ts_wall"]
        elif kind == "phase.end" and (phase, e.get("attempt")) in started:
            phases[phase] = phases.get(phase, 0) + round(e["ts_wall"] - started[(phase, e.get("attempt"))])
        elif kind in ("llm.request", "llm.response") and e.get("tool"):
            pass                                  # a kernel tool's own model call (ask), not a role's conversation
        elif kind == "llm.request":
            requests[(phase, e.get("conversation_id"))].append(e)
        elif kind == "llm.response":
            responses[(phase, e.get("conversation_id"))] = e
            model_s[(phase, e.get("conversation_id"))] += e.get("latency_s") or 0.0
        elif kind == "tool.call":
            tools.setdefault(e.get("tool"), {"calls": 0, "errors": 0})["calls"] += 1
        elif kind == "tool.error":
            tools.setdefault(e.get("tool"), {"calls": 0, "errors": 0})["errors"] += 1
            errors[(e.get("tool"), _shape(str(_payload(rec, e).get("error", ""))))].append(e)
        elif kind in ("gate.failed", "gate.passed"):
            gates["failed" if kind == "gate.failed" else "passed"] += 1
        elif kind == "job.finished":
            jobs["run"] += 1
            jobs["failed"] += e.get("state") == "failed"
            jobs["gpu_s"] += e.get("gpu_seconds") or 0.0
        elif kind == "ingest.accepted":
            ingest["accepted"] += 1
        elif kind == "ingest.rejected":
            ingest["rejected"] += 1
            reasons.update(_shape(str(r)) for r in _payload(rec, e).get("reasons") or [])
        elif kind.endswith(".end") and e.get("duration_s") is not None:
            if kind == "train.end":
                durations["train_s"] += e["duration_s"]
            elif kind == "render.end":
                durations["render_s"] += e["duration_s"]
            elif kind.startswith("wbench."):
                durations["eval_s"] += e["duration_s"]
    roles: dict[tuple, dict] = {}
    role_by_conversation = {}
    for key, evs in requests.items():
        role = role_by_conversation[key] = role_of(_payload(rec, evs[0]).get("body") or {})
        row = roles.setdefault((key[0], role), {"phase": key[0], "role": role, "conversations": 0, "turns": 0,
                                                "model_s": 0.0, "compactions": 0})
        row["conversations"] += 1
        row["turns"] += len(evs)
        row["model_s"] += model_s[key]
    for key in _continuations(rec, requests, responses):
        roles[(key[0], role_by_conversation[key])]["compactions"] += 1
    out = {"phases": {**phases, **{k: round(v) for k, v in durations.items()}},
           "roles": [{**row, "model_s": round(row["model_s"])} for row in roles.values()],
           "tools": tools,
           "tool_errors": [{"tool": tool, "count": len(evs), "example": example}
                           for (tool, example), evs in sorted(errors.items(), key=lambda kv: -len(kv[1]))[:MAX_ERRORS]],
           "local_errors": _local_errors(rec, requests),
           "gpu_jobs": {"run": jobs["run"], "failed": jobs["failed"], "gpu_s": round(jobs["gpu_s"])},
           "ingest": {"accepted": ingest["accepted"], "rejected": ingest["rejected"],
                      "reasons": [{"reason": r, "count": n} for r, n in reasons.most_common(MAX_REASONS)]},
           "rounds": _rounds(run_dir, node_id), "gates": dict(gates)}
    while len(json.dumps(out)) > MAX_BYTES and out["tool_errors"]:
        out["tool_errors"].pop()
    return out
```

`jobs["failed"] += e.get("state") == "failed"` adds a bool; `Counter` holds it as an int.

In `transcripts.py`, delete the local `_role` function, change the import to `from .process_digest import _payload, _text, role_of`, and use `role_of(body)` where `_role(body)` was.

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_process_digest.py tests/test_transcripts.py tests/test_context_bundle.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/process_digest.py kernel/ar_kernel/transcripts.py tests/test_process_digest.py
git commit -m "process digest: time and turns per role, tool calls, GPU jobs, ingest results and plan rounds"
```

---

### Task 8: Kernel skills, `read_skill`, and tools that describe their own arguments

**Files:**
- Create: `kernel/ar_kernel/skills/{container,data_formats,clip_quality,retries,node_files,large_files}.md`
- Create: `kernel/ar_kernel/tools/skills.py`
- Modify: `kernel/ar_kernel/tools/data_tools.py` (`register_data_tools`)
- Modify: `kernel/ar_kernel/tools/captioner.py` (`register_caption_tool`), `kernel/ar_kernel/tools/hf_tools.py` (`register_hf_tools`), `kernel/ar_kernel/tools/jobs.py` (`register_job_tools`), `kernel/ar_kernel/tools/ask.py` (`register_ask_tool`)
- Modify: `kernel/ar_kernel/run_kit.py`, `kernel/ar_kernel/contract/verify.py` (`ContractHarness.start`), `kernel/ar_kernel/agent_phase.py` (`_recipe_tools`)
- Modify: `pyproject.toml` (package data)
- Test: `tests/test_skills.py`, `tests/test_data_tools.py`

**Interfaces:**
- Consumes: `items_schema` from Task 4.
- Produces:
  - `skills.load_skills() -> dict[name, {"description", "text"}]`; `skills.register_skill_tool(mcp, kit)` registers `read_skill(name: str) -> str`.
  - `data_query(format?, camera_motion?, clip_ids?, ingested_by?, limit?, offset?)` — no `filter` wrapper.
  - Every kernel tool argument has a `description` in its JSON schema.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skills.py
import asyncio
import threading

from ar_kernel.contract.verify import _MockCaption, _MockData, _MockHf
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.ask import MockAsk, register_ask_tool
from ar_kernel.tools.captioner import register_caption_tool
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.data_tools import register_data_tools
from ar_kernel.tools.hf_tools import register_hf_tools
from ar_kernel.tools.jobs import JobQueue, register_job_tools
from ar_kernel.tools.server import ToolKit, new_mcp
from ar_kernel.tools.skills import load_skills, register_skill_tool

SKILLS = {"container", "data_formats", "clip_quality", "retries", "node_files", "large_files"}


def _tools(tmp_path):
    rec = Recorder(tmp_path)
    queue = JobQueue(rec, threading.Lock(), wait_cap_s=5.0)
    try:
        kit, mcp = ToolKit(TokenRegistry(rec), rec), new_mcp()
        register_data_tools(mcp, kit, _MockData())
        register_hf_tools(mcp, kit, _MockHf())
        register_job_tools(mcp, kit, queue)
        queue.register(_MockCaption())
        register_caption_tool(mcp, kit, queue)
        register_ask_tool(mcp, kit, MockAsk())
        register_skill_tool(mcp, kit)
        return {t.name: t for t in asyncio.run(mcp.list_tools())}
    finally:
        queue.shutdown()


def test_every_skill_has_a_name_a_use_and_a_body():
    skills = load_skills()
    assert set(skills) == SKILLS
    for name, skill in skills.items():
        assert skill["description"].startswith("Use when") and len(skill["text"]) > 200, name
        assert "wbench" not in skill["text"].lower()
    assert "57 frames" in skills["data_formats"]["text"] and "cam_c2w" in skills["data_formats"]["text"]


def test_read_skill_lists_every_skill_in_its_description(tmp_path):
    tool = _tools(tmp_path)["read_skill"]
    for name, skill in load_skills().items():
        assert f"- {name}: {skill['description']}" in tool.description


def test_every_kernel_tool_argument_is_described(tmp_path):
    tools = _tools(tmp_path)
    assert "filter" not in tools["data_query"].input_schema["properties"]
    assert set(tools["data_query"].input_schema["properties"]) == {
        "format", "camera_motion", "clip_ids", "ingested_by", "limit", "offset"}
    for name, tool in tools.items():
        for arg, schema in tool.input_schema["properties"].items():
            assert schema.get("description"), f"{name}.{arg} has no description"
    candidate = tools["data_ingest"].input_schema["properties"]["candidates"]["items"]
    assert set(candidate["required"]) == {"video", "caption", "camera_motion", "provenance"}
    assert all(p.get("description") for p in candidate["properties"].values())
```

In `tests/test_data_tools.py::test_register_names_are_openai_safe` nothing changes (still 5 data tools).

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_skills.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ar_kernel.tools.skills'`

- [ ] **Step 3: Write the six skills**

`kernel/ar_kernel/skills/container.md`:

```markdown
---
name: container
description: Use when you need to know what the container provides: installed programs, internet, where files may be written, what the kernel mounts.
---

# The container

- Internet access is on. `pip install --user`, `curl`, `wget`, `git`, `unzip` and `7z` work.
- Installed: `pandas`, `pyarrow`, `numpy`, `opencv-python-headless`, `Pillow`, `ffmpeg`, `ffprobe`.
- `/workspace` is writable and is carried to a retry of the same phase. `/workspace/staging/` is where files for the kernel's data tools go: the kernel reads candidates only from there, and GPU jobs and downloads publish their results there.
- `/agent` is the agent's code: writable in `edit_self`, read-only in `improve_recipe`. In `edit_self`, `/code` is the parent's code, which is the code that is running, read-only.
- `/context/context.json` is the full context of this attempt. `/nodes/<node>/` holds what each finished node left behind, read-only (skill `node_files`).
- GPUs are reached only through the kernel's GPU tools. Each is a queued job: it returns a job id at once, and `job_wait` collects the result.
```

`kernel/ar_kernel/skills/data_formats.md` (the first list is the text of the old `FORMAT_RULES`):

```markdown
---
name: data_formats
description: Use when preparing clips, captions, timed prompt segments or camera poses for data_ingest: the only formats the kernel accepts, and how to convert to them.
---

# Data formats

## The standard formats (the only ones accepted)

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

## Converting a video

- Probe: `ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames,sample_aspect_ratio:stream_side_data=rotation -of json in.mp4`
- Display aspect = width * SAR / height, with width and height swapped for a 90 or 270 degree rotation.
- Center-crop to 16:9 without stretching: `ffmpeg -i in.mp4 -vf "crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',setsar=1" -c:v libx264 -pix_fmt yuv420p -crf 18 -an out.mp4`. Re-encoding also applies any rotation, and `-pix_fmt yuv420p` keeps 10-bit or 4:4:4 sources decodable.
- Keep the source frame rate if it is at least 24 fps. Never raise it by duplicating frames.
- Trimming changes the frame count, so slice the pose array to the same frames.

## Camera poses

- `poses/<id>.npz` holds `cam_c2w` with shape [N, 4, 4]: camera-to-world, OpenCV convention (x right, y down, z forward).
- From world-to-camera matrices, invert them.
- From OpenGL convention (y up, z backward): `c2w_cv = c2w_gl @ diag(1, -1, -1, 1)`.
- A clip is `static` only if a measurement shows the camera does not move (skill `clip_quality`), never because of its prompt or file name.

## Timed prompt segments

- Segment boundaries fall on round boundaries: 25/24 s, then every 32/24 s.
- A clip of 25 + 32k frames (57, 89, 121, 153, ...) ends on a round boundary.
- Caption each segment from its own part of the clip.
- A text-or-image-to-video tool given a clip's last frame as its first frame continues that clip; the continuation starts on that same frame, so a 97-frame continuation adds 96 new frames (3 rounds).
```

`kernel/ar_kernel/skills/clip_quality.md`:

```markdown
---
name: clip_quality
description: Use when checking clips for defects before ingesting them: frozen, black, blurry or cut video, camera motion that disagrees with the pose, captions that disagree with the video.
---

# Checking clip quality

Install what helps with `pip install --user` (for example `scenedetect`, or a small CLIP or aesthetic model).

- Frozen video: `ffmpeg -i in.mp4 -vf freezedetect=n=-60dB:d=1 -f null -` lists frozen spans. Or compare the mean absolute difference of consecutive downscaled frames with clips you already trust.
- Black or flat frames: `ffmpeg -i in.mp4 -vf blackdetect=d=0.2 -f null -`
- Blur: the variance of the Laplacian per frame (`cv2.Laplacian(gray, cv2.CV_64F).var()`), compared across the batch. Drop the lowest tail rather than using one absolute threshold.
- Cuts inside a clip: `scenedetect`, or a histogram difference between neighbouring frames. Poses estimated across a cut are not valid, and one caption cannot describe both shots. data_ingest adds a warning to a clip whose pose jumps between two frames (a step 15 times the clip's median step, or a turn of 20 degrees), naming the frames.
- Camera motion: the total translation and rotation in `cam_c2w` is near zero for a static clip and clearly non-zero for a moving one. Dense optical flow (`cv2.calcOpticalFlowFarneback`) should agree on the direction.
- Caption against video: ask the captioning tool a narrow question about the clip (the camera motion, the setting) and compare the answer with the saved caption.
- Sample a few clips from a source before converting all of it.
```

`kernel/ar_kernel/skills/retries.md`:

```markdown
---
name: retries
description: Use when this attempt is a retry (`retry` is set in the context): what each kind of failure means and what the retry carries.
---

# Retries after a failed attempt

The context's `retry` describes the attempt that failed; its `kind` says where it failed.

- `edit_self`: the phase itself failed (no valid result); `error` says why.
- `contract`: the kernel's check of the edited code failed; `failed_step` and `detail` say where, and `steps` lists every step. The edited tree is kept: `/agent` holds the failed attempt's edits.
- `improve_recipe`: the phase itself failed (no valid submission); `error` says why.
- `gate`: the kernel's pre-training check refused the submission; `failures` lists why.
- `train`: precache or training failed; `failure`, `detail` and `log_tail` (the last 20,000 characters of the training log) say how. That includes a run that wrote a checkpoint but failed anyway (for example a NaN loss): such a checkpoint is never scored.
- Gate and training retries also carry the submitted `data_commit`, `recipe` and `rationale`. The data commit is still valid and can be submitted again with a different recipe.
- The workspace of the failed attempt is carried over, including its `plans.json`.
```

`kernel/ar_kernel/skills/node_files.md`:

```markdown
---
name: node_files
description: Use when looking into what earlier nodes did: their transcripts, workspaces, command logs, training logs and configs under /nodes.
---

# What earlier nodes left behind: /nodes

`/nodes/<node>/` holds, read-only, what each finished node of the run produced. The root node only has an evaluation, so its directory is empty.

- `recipe.yaml`: the tunable values the node trained with.
- `rationale.md`: the recipe rationale and what the node's data role recorded with it.
- `edit.json`: the summary of the node's self-edit.
- `transcripts/<phase>-<attempt>/NN-<role>.md`: every conversation of every role, in order, with reasoning, tool calls and tool results. A file is named after the role's submit tool. A compacted conversation continues in the next file.
- `attempts/edit_self-<k>/agent/`: the agent code as that attempt left it.
- `attempts/<phase>-<k>/workspace/`: the files the roles wrote, including `tool_output/` (the full output of commands) and `plans.json` (every plan of the attempt, with the reports and questions that went back to the planner).
- `attempts/improve_recipe-<k>/train/train.log`: the full training log. Each `[Train] step=` line is one optimizer step with the dataset `source`, `sigma`, `loss`, `grad` and `lr`; its `time=` covers only the last micro-batch of the step. The loss depends mostly on `sigma`, so compare losses at similar sigma.
- `attempts/improve_recipe-<k>/train_config.yaml`: the full training config that ran.
- `attempts/improve_recipe-<k>/view/<dataset>/`: the data commit as the trainer saw it: captions and poses per clip.
- `attempts/<phase>-<k>/context/context.json`: the complete context that attempt received.
- `contract/attempt-<k>/`: the kernel's checks of the edited code.

Useful ways in:
- `ls /nodes`, then `ls -R /nodes/<parent> | head -200` for the shape; read the parent first.
- `grep -l "Error" /nodes/*/transcripts/*/*.md` finds tool failures; `grep -c "tool call: run_command"` shows how much was done by hand.
- To follow the loss, extract `sigma` and `loss` from train.log with a short Python script and group by sigma.
```

`kernel/ar_kernel/skills/large_files.md`:

```markdown
---
name: large_files
description: Use when a file or command output is too large to read whole, for example transcripts, training logs, context.json, JSON or CSV annotations.
---

# Reading large files

- Size first: `wc -c -l file`, `ls -lh dir`.
- Parts: `head -n 50`, `tail -n 50`, `sed -n '200,260p' file`, or a file-reading tool's line range.
- Search: `grep -n "pattern" file`, `grep -c` to count, `grep -l "pattern" -r dir` to find files, `grep -A 5 -B 2` for context around a match.
- JSON: `python -c "import json; d = json.load(open('f.json')); print(list(d))"` to see the keys, then print one key.
- Tables (CSV, TSV, JSON lines): `pandas.read_csv` or `pandas.read_json(lines=True)`, then `df.shape`, `df.columns`, `df.head()`, `df.describe()`, `df.groupby(...)`.
- Logs with numbers: extract the fields you need with a regular expression in a short Python script and summarise them (mean, trend, groups) instead of reading the lines.
```

- [ ] **Step 4: The `read_skill` tool**

```python
# kernel/ar_kernel/tools/skills.py
"""read_skill: the kernel's reference notes for agents (kernel/ar_kernel/skills/*.md). Agents read
them through this tool and cannot change them."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from .server import ToolError

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


def load_skills() -> dict[str, dict]:
    """name -> {description, text}. A skill file starts with front matter: `name:` and `description:`."""
    skills = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        head, text = path.read_text(encoding="utf-8").removeprefix("---\n").split("\n---\n", 1)
        meta = dict(line.split(": ", 1) for line in head.splitlines())
        skills[meta["name"]] = {"description": meta["description"], "text": text.strip()}
    return skills


def register_skill_tool(mcp, kit) -> None:
    skills = load_skills()
    index = "\n".join(f"- {name}: {skill['description']}" for name, skill in skills.items())

    @mcp.tool(name="read_skill", description="Read one of the kernel's reference notes. Read a skill at the "
              "moment you are about to do what it covers, not up front. The skills:\n" + index)
    async def read_skill(name: Annotated[str, Field(description="one of: " + ", ".join(skills))],
                         ctx: Context) -> str:
        def read(_caller) -> str:
            if name not in skills:
                raise ToolError(f"no skill {name!r}; the skills are: {', '.join(skills)}")
            return skills[name]["text"]
        return await kit.call(ctx, "read_skill", {"name": name}, read)
```

Register it in three places:
- `run_kit.py`: add `from .tools.skills import register_skill_tool` and, after `register_caption_tool(mcp, kit, queue)`, the line `register_skill_tool(mcp, kit)`.
- `contract/verify.py`, `ContractHarness.start`: add `from ..tools.skills import register_skill_tool` and, after `register_ask_tool(mcp, kit, MockAsk())`, the line `register_skill_tool(mcp, kit)`.
- `agent_phase.py`, `_recipe_tools`: add `"read_skill"` after `"ask"` in the list.

In `pyproject.toml` add, after the `[tool.setuptools.packages.find]` block:

```toml
[tool.setuptools.package-data]
ar_kernel = ["skills/*.md"]
```

- [ ] **Step 5: Describe the data tools' arguments**

In `data_tools.py` add `from typing import Annotated, Any, Literal`, `from pydantic import Field` and `from .gpu_jobs import items_schema`, and replace `register_data_tools`:

```python
_FORMATS = "video_caption_camera, video_timed_prompts_camera (prompt_mode per_chunk or segment), video_caption_static"
_STAGED = "absolute container path under /workspace/staging/"
Candidates = Annotated[list[dict[str, Any]], items_schema({
    "video": {"type": "string", "description": f"the mp4: {_STAGED}"},
    "caption": {"type": "string", "description": f"the caption JSON file (not the text): {_STAGED}. It holds "
                                                 "{\"caption\": str} and, for timed prompts, \"segments\""},
    "pose": {"type": "string", "description": f"the pose npz with cam_c2w: {_STAGED}. Required for 'moving', "
                                              "forbidden for 'static'"},
    "camera_motion": {"type": "string", "enum": ["moving", "static"],
                      "description": "'static' only when a measurement shows the camera does not move"},
    "provenance": {"type": "object", "description": "where the clip came from: the record hf_download returned, "
                   "the one in a rollout's `candidate`, or for a clip you derived {\"kind\": \"derived\", "
                   "\"from\": [...], \"transform\": \"what you did\"}"},
    "license": {"type": "string", "description": "the source's license, when known"},
    "derived_from": {"type": "array", "items": {"type": "string"},
                     "description": "ids of archive clips this clip was made from"}},
    ["video", "caption", "camera_motion", "provenance"])]


def register_data_tools(mcp, kit, tools: DataTools) -> None:
    @mcp.tool(name="video_probe", description="Frame count, fps, coded size, rotation, pixel "
              "aspect and display aspect of a video under /workspace.")
    async def video_probe(path: Annotated[str, Field(description="a video file under /workspace")],
                          ctx: Context) -> dict:
        return await kit.call(ctx, "video_probe", {"path": path}, lambda c: tools.probe(c, path))

    @mcp.tool(name="data_ingest", description="Ingest staged candidates into the run's clip pool. The formats "
              "a clip must meet are in skill data_formats. Ingest MOVES each staged file into the archive: copy "
              "it first if you still need it. Returns, per candidate, accepted + clip_id + eligible formats + "
              "warnings, or rejected + reasons. Read the reasons and change the candidate; a clip the kernel "
              "excludes cannot be made acceptable.")
    async def data_ingest(candidates: Candidates, ctx: Context) -> list[dict]:
        return await kit.call(ctx, "data_ingest", {"candidates": candidates},
                              lambda c: tools.ingest(c, candidates))

    @mcp.tool(name="data_query", description="Search the clip pool of the whole run. Every argument narrows the "
              "result; with none, all clips are listed. Each clip has its provenance, metadata, eligible formats "
              "and the scores of the nodes that trained on it. `total` counts all matches; page with `offset`.")
    async def data_query(
            ctx: Context,
            format: Annotated[str | None, Field(description="keep clips eligible for this format, e.g. "
                              "'video_caption_camera' or 'video_timed_prompts_camera:per_chunk'")] = None,
            camera_motion: Annotated[Literal["moving", "static"] | None, Field(description="keep clips with this camera motion")] = None,
            clip_ids: Annotated[list[str] | None, Field(description="keep only these clip ids")] = None,
            ingested_by: Annotated[str | None, Field(description="keep clips this node ingested, e.g. the id of the node being built")] = None,
            limit: Annotated[int, Field(description="clips per page")] = QUERY_LIMIT,
            offset: Annotated[int, Field(description="skip this many matches")] = 0) -> dict:
        given = dict(format=format, camera_motion=camera_motion, clip_ids=clip_ids, ingested_by=ingested_by,
                     limit=limit, offset=offset)
        filter = {key: value for key, value in given.items() if value is not None}
        return await kit.call(ctx, "data_query", filter, lambda c: tools.query(c, filter))

    @mcp.tool(name="data_commit", description="Create an immutable data commit: the training set of this node.")
    async def data_commit(
            parent: Annotated[str | None, Field(description="the commit this one follows (the parent node's data commit), or null")],
            datasets: Annotated[dict[str, Any], Field(description="{name: {format, prompt_mode, weight, clips: "
                                f"[clip_id]}}}}. format is one of: {_FORMATS}. Set prompt_mode only for "
                                "video_timed_prompts_camera; omit it otherwise. weight is the dataset's sampling "
                                "weight. Each dataset needs at least as many clips as training GPUs")],
            message: Annotated[str, Field(description="what this commit contains")],
            ctx: Context) -> dict:
        return await kit.call(ctx, "data_commit",
                              {"parent": parent, "datasets": datasets, "message": message},
                              lambda c: tools.commit(c, parent, datasets, message))

    @mcp.tool(name="recipe_check", description="Run every pre-training check on a recipe and a data commit "
              "without using up an attempt. Returns ok and the failures. Among the checks: steps_per_epoch = "
              "floor(floor(epoch_windows / n_gpus) / optimizer.grad_accum_steps) must be at least 1, and "
              "optimizer.epochs * steps_per_epoch at least optimizer.max_steps, where epoch_windows is the "
              "largest, over the commit's datasets, of ceil(clips / (weight / total_weight)).")
    async def recipe_check(
            recipe: Annotated[dict[str, Any], Field(description="a flat {tunable key: value} map, e.g. "
                              "{\"optimizer.lr\": 1e-4}, with no wrapper key")],
            data_commit: Annotated[str, Field(description="the commit id data_commit returned")],
            ctx: Context) -> dict:
        return await kit.call(ctx, "recipe_check", {"recipe": recipe, "data_commit": data_commit},
                              lambda c: tools.recipe_check(c, recipe, data_commit))
```

- [ ] **Step 6: Describe the remaining tools' arguments**

Add `from typing import Annotated` and `from pydantic import Field` where missing; bodies stay as they are.

`captioner.py`, `register_caption_tool` (the description also takes over the old knowledge file):

```python
    @mcp.tool(name=TOOL, description="Caption video clips with the kernel's local video model, which sees the "
              "whole clip. A GPU job: returns {job_id} at once; collect the result with job_wait. Loading the "
              "model takes minutes, then seconds per clip, so send every clip in one call. The finished job's "
              "result.clips maps each path to {caption} or {error}. The tool returns text only: write "
              "{\"caption\": \"<text>\"} to a JSON file under /workspace/staging/ yourself before data_ingest.")
    async def caption_videos(
            paths: Annotated[list[str], Field(description="video files under /workspace (relative paths resolve against /workspace)")],
            prompt: Annotated[str, Field(description="the instruction sent with every clip, e.g. 'Write one factual "
                              "caption (1-3 sentences) describing the scene and how the camera moves.'")],
            ctx: Context) -> dict[str, Any]:
```

`hf_tools.py`, `register_hf_tools`:

```python
    async def hf_search(
            query: Annotated[str, Field(description="words that must all appear in the dataset id or tags")],
            ctx: Context,
            kind: Annotated[str, Field(description="only 'dataset'")] = "dataset",
            limit: Annotated[int, Field(description="results to return, at most 100")] = 20) -> list[dict]:
```

```python
    async def hf_list_files(
            repo: Annotated[str, Field(description="dataset id, 'owner/name'")],
            revision: Annotated[str, Field(description="branch, tag or commit, e.g. 'main'")],
            ctx: Context,
            pattern: Annotated[str, Field(description="fnmatch pattern; '*' also crosses folders, e.g. 'videos/*.mp4'")] = "*",
            limit: Annotated[int, Field(description="files per page, at most 1000")] = LIST_PAGE,
            offset: Annotated[int, Field(description="skip this many matching files")] = 0) -> dict[str, Any]:
```

```python
    async def hf_download(
            repo: Annotated[str, Field(description="dataset id, 'owner/name'")],
            revision: Annotated[str, Field(description="branch, tag or commit; the result pins it to a commit")],
            patterns: Annotated[list[str], Field(description="fnmatch patterns or exact paths of the files to fetch")],
            ctx: Context,
            max_bytes: Annotated[int | None, Field(description="refuse if the matching files total more than this; the kernel's own cap is 20 GiB per call")] = None) -> dict[str, Any]:
```

`jobs.py`, `register_job_tools`: each of the three functions' `job_id` parameter becomes `job_id: Annotated[str, Field(description="the job_id a GPU tool returned (its first 8 characters are enough)")]`, and `job_wait`'s `timeout_s` becomes `timeout_s: Annotated[float, Field(description="seconds to wait, at most 300; call again if the job is still running")] = 300`.

`ask.py`, `register_ask_tool`:

```python
    async def ask_tool(
            question: Annotated[str, Field(description="the whole question: the model sees nothing else of your conversation")],
            ctx: Context,
            images: Annotated[list[str] | None, Field(description="image files under /workspace to show with the question")] = None,
    ) -> dict[str, Any]:
```

- [ ] **Step 7: Run the tests**

Run: `$PY -m pytest tests/test_skills.py tests/test_data_tools.py tests/test_hf_tools.py tests/test_jobs.py tests/test_ask.py tests/test_captioner.py tests/test_gpu_jobs.py tests/test_tool_server.py tests/test_contract_verify.py tests/test_agent_phase.py -q`
Expected: PASS. A test that calls `data_query` through MCP with `{"filter": {...}}` must send the keys directly (`{"format": ...}`); `tests/test_seed_agent.py` is updated in Task 12.

- [ ] **Step 8: Commit**

```bash
git add kernel/ar_kernel pyproject.toml tests/test_skills.py tests/test_data_tools.py
git commit -m "kernel skills behind read_skill; every kernel tool describes its arguments; data_query takes typed fields"
```

---

### Task 9: Audit and quarantine

**Files:**
- Modify: `kernel/ar_kernel/isolation.py` (add `audit`)
- Modify: `kernel/ar_kernel/loop.py` (`Quarantined`, `Loop._audit`, calls in `_edit` and `_recipe`, `cycle`)
- Test: `tests/test_isolation.py`, `tests/test_loop.py`

**Interfaces:**
- Consumes: `quarantined` status and `NOT_SHOWN` from Task 6.
- Produces:
  - `isolation.audit(cfg, recorder, node: str, phase: str, attempt: int) -> list[str]` — one line per audit pattern found in a tool-call argument of that attempt; empty when clean.
  - A node whose attempt has a hit ends with status `quarantined` and error `quarantined: <hits>`; event `isolation.audit` carries the hits.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_isolation.py`:

```python
def _tool_call(rec, name, arguments, **where):
    message = {"role": "assistant", "content": None,
               "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": arguments}}]}
    rec.event("llm.response", conversation_id="c", payload={"body": {"choices": [{"message": message}]}}, **where)


def test_the_audit_reads_tool_call_arguments_of_one_attempt_only(tmp_path):
    from ar_kernel.isolation import audit
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    here = dict(node="n2", phase="improve_recipe", attempt=1)
    _tool_call(rec, "run_command", json.dumps({"command": "ls /workspace"}), **here)
    assert audit(CFG, rec, "n2", "improve_recipe", 1) == []
    _tool_call(rec, "run_command", json.dumps(
        {"command": "curl -L https://huggingface.co/datasets/meituan-longcat/WBench/resolve/main/x.json"}), **here)
    _tool_call(rec, "write_file", json.dumps({"path": "notes.md", "content": "the eval is wbench"}),
               node="n2", phase="edit_self", attempt=1)
    assert audit(CFG, rec, "n2", "improve_recipe", 1) == [
        "run_command names meituan-longcat", "run_command names wbench"]
    assert audit(CFG, rec, "n2", "edit_self", 1) == ["write_file names wbench"]
    assert audit(CFG, rec, "n2", "improve_recipe", 2) == []


def test_text_an_agent_only_read_is_not_a_hit(tmp_path):
    from ar_kernel.isolation import audit
    from ar_kernel.telemetry.recorder import Recorder
    rec = Recorder(tmp_path)
    rec.event("llm.request", node="n2", phase="improve_recipe", attempt=1, conversation_id="c", payload={"body": {
        "messages": [{"role": "tool", "content": "a paper abstract that compares models on WBench"}]}})
    assert audit(CFG, rec, "n2", "improve_recipe", 1) == []
```

Append to `tests/test_loop.py`:

```python
def test_a_phase_that_reaches_for_the_evaluation_set_quarantines_the_node(make_loop):
    run, make = make_loop
    script = Script(run, score=[0.7, 0.9])
    loop = make(script)
    real_edit = script.edit_self

    def reaching_edit(env, *, node, attempt, **kw):
        call = {"id": "c1", "type": "function",
                "function": {"name": "run_command", "arguments": '{"command": "curl hf.co/datasets/x/WBench"}'}}
        loop.ctx.recorder.event("llm.response", node=node, phase="edit_self", attempt=attempt, conversation_id="c",
                                payload={"body": {"choices": [{"message": {"role": "assistant", "tool_calls": [call]}}]}})
        return real_edit(env, node=node, attempt=attempt, **kw)
    loop.phases.edit_self = reaching_edit
    loop.run()
    n1 = NodeStore(loop.ctx.conn).get("n1")
    assert n1["status"] == "quarantined" and n1["error"] == "quarantined: run_command names wbench"
    assert [r[0] for r in script.retries] == ["edit_self"]              # one attempt, no retry, no data phase
    assert any(e["type"] == "isolation.audit" for e in loop.ctx.recorder.read_events("n1"))
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_isolation.py tests/test_loop.py -q -k "audit or only_read or quarantines"`
Expected: FAIL, `ImportError: cannot import name 'audit'`

- [ ] **Step 3: `audit`**

Append to `isolation.py` (and add `from .process_digest import _payload` to its imports):

```python
def audit(cfg, recorder, node: str, phase: str, attempt: int) -> list[str]:
    """What an agent phase did that names an audit pattern: every tool-call argument of the attempt
    (commands, files written through tools, queries, kernel tool arguments, plans). Text the agent only
    read (a tool result, a downloaded file) is not looked at: reading a mention is not reaching for it."""
    patterns = [p.lower() for p in cfg.get("isolation.audit_patterns") or []]
    hits = set()
    for e in recorder.read_events(node):
        if e.get("type") != "llm.response" or e.get("phase") != phase or e.get("attempt") != attempt or e.get("tool"):
            continue
        for choice in (_payload(recorder, e).get("body") or {}).get("choices") or []:
            for call in (choice.get("message") or {}).get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = str(function.get("arguments")).lower()
                hits |= {f"{function.get('name')} names {p}" for p in patterns if p in arguments}
    return sorted(hits)
```

- [ ] **Step 4: The loop**

In `loop.py` add `from .isolation import audit` and, below `class StopRun`:

```python
class Quarantined(Exception):
    """An agent phase reached for the evaluation set: the node ends here, with no retry."""
```

Add the method to `Loop` (under `_node_dir`):

```python
    def _audit(self, node: str, phase: str, attempt: int) -> None:
        hits = audit(self.cfg, self.ctx.recorder, node, phase, attempt)
        if hits:
            self.ctx.recorder.event("isolation.audit", node=node, phase=phase, attempt=attempt, payload={"hits": hits})
            raise Quarantined("; ".join(hits)[:2000])
```

Call it right after each agent phase returns, before anything else looks at the outcome. In `_edit`, after the `out = self.phases.edit_self(...)` statement:

```python
            self._audit(child, "edit_self", k)
```

In `_recipe`, after the `out = self.phases.improve_recipe(...)` statement:

```python
            self._audit(child, "improve_recipe", k)
```

In `cycle`, add a handler before `except Exception as exc:`:

```python
        except Quarantined as exc:
            status, error = "quarantined", f"quarantined: {exc}"
```

`_finish` already sets any non-`scored` status, raises the `node_failed` alert and records the error; selection only ever picks `scored` nodes, and Task 6 keeps the node out of contexts and `/nodes`.

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest tests/test_isolation.py tests/test_loop.py tests/test_loop_integration.py tests/test_selection.py tests/test_status.py tests/test_panel_data.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add kernel/ar_kernel/isolation.py kernel/ar_kernel/loop.py tests/test_isolation.py tests/test_loop.py
git commit -m "audit each agent phase's tool calls; a node that reaches for the evaluation set is quarantined"
```

---

### Task 10: The contract refuses node facts in role prompts

**Files:**
- Modify: `kernel/ar_kernel/contract/verify.py` (new `prompt_check`; call it after the static step)
- Test: `tests/test_contract_verify.py`

**Interfaces:**
- Consumes: `AGENT_METRICS` from Task 5.
- Produces: `prompt_check(code: Path) -> ContractStep` named `"prompts"`; `verify_contract` reports it after `"static"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_contract_verify.py`)

```python
@pytest.mark.parametrize("text,ok,fragment", [
    ("# Mission\nYou choose the data idea.\n", True, ""),
    ("# Mission\nKeep in mind that n12 scored best.\n", False, "n12"),
    ("# How you work\nRaise follows_event_instruction first.\n", False, "follows_event_instruction"),
    ("# Mission\nThe plan needs no nine-step list; turn 1 comes first.\n", True, ""),
])
def test_prompt_check_refuses_node_ids_and_metric_names(tmp_path, text, ok, fragment):
    from ar_kernel.contract.verify import prompt_check
    (tmp_path / "agent" / "prompts").mkdir(parents=True)
    (tmp_path / "agent" / "prompts" / "planner.md").write_text(text)
    step = prompt_check(tmp_path)
    assert step.name == "prompts" and step.ok is ok and fragment in step.detail


def test_an_agent_without_a_prompts_folder_passes_the_prompt_check(tmp_path):
    from ar_kernel.contract.verify import prompt_check
    assert prompt_check(tmp_path).ok


def test_a_prompt_naming_a_node_fails_verification_before_any_container(tmp_path, monkeypatch):
    def mutate(agent):
        (agent / "prompts").mkdir(exist_ok=True)
        (agent / "prompts" / "planner.md").write_text("Remember: n3 used too many static clips.\n")
    report = _verify_tree(tmp_path, monkeypatch, mutate)
    failed = next(s for s in report.steps if not s.ok)
    assert failed.name == "prompts" and "n3" in failed.detail and "briefing" in failed.detail
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_contract_verify.py -q -k "prompt"`
Expected: FAIL, `ImportError: cannot import name 'prompt_check'`

- [ ] **Step 3: Implement**

In `verify.py` add `import re` and `from ..eval.score import AGENT_METRICS`, and below `static_check`:

```python
_NODE_FACT = re.compile(r"\b(n\d+|" + "|".join(alias for alias, _, _ in AGENT_METRICS.values()) + r")\b")


def prompt_check(code: Path) -> ContractStep:
    """A role prompt holds a mission and general behaviour: it may not name a node or a metric.
    Checked here, not in the agent's own self-test, which the agent can edit."""
    for path in sorted((code / "agent" / "prompts").glob("*.md")):
        found = sorted(set(_NODE_FACT.findall(path.read_text(encoding="utf-8", errors="replace"))))
        if found:
            return ContractStep("prompts", False,
                                f"agent/prompts/{path.name} names {', '.join(found)}. A prompt holds a role's "
                                "mission and general behaviour only: put facts about nodes and metrics in what "
                                "the role is told first (agent/briefing.py), not in its prompt")
    return ContractStep("prompts", True)
```

In `verify_contract`, right after the static step is appended and checked (`if not step.ok: return finish(False)`), add:

```python
    step = prompt_check(code)
    report.steps.append(step)
    if not step.ok:
        return finish(False)
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_contract_verify.py -q`
Expected: PASS. `test_good_agent_passes_every_step` (docker-marked, not in the default run) lists the step names; if it asserts the exact list, add `"prompts"` after `"static"`.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/contract/verify.py tests/test_contract_verify.py
git commit -m "contract: a role prompt that names a node or a metric fails verification"
```

---
### Task 11: Seed agent briefing

**Files:**
- Rewrite: `seed_agent/agent/briefing.py`
- Test: `tests/test_seed_agent.py` (the digest tests)

**Interfaces:**
- Consumes: the context shapes of Task 6, the process digest of Task 7, `RecipeContext.metric_guide`.
- Produces (all return Markdown text):
  - `recipe_context(ctx, previous_plans: list | None) -> str` — the data planner's first message.
  - `engineer_context(ctx, previous_plans: list | None) -> str` — what the data engineer gets with the plan.
  - `edit_context(ctx, components: dict, previous_plans: list | None) -> str` — the edit planner's first message.
  - `coder_context(components: dict) -> str` — what the coder gets with the plan.
  - `LINEAGE_SHOWN`, `SIBLINGS_SHOWN`, `_hypothesis(rationale)` keep their meaning.

Until Task 12 lands, `agent/orchestration.py` still imports the old names; this task's tests import `agent.briefing` only.

- [ ] **Step 1: Write the failing tests**

In `tests/test_seed_agent.py` delete `_lineage`, `test_the_digest_describes_the_recent_lineage_and_keeps_the_root_as_baseline`, `test_the_digest_describes_the_parents_finished_children_before_the_lineage`, `test_an_earlier_node_is_described_by_what_the_phase_acts_on` and `test_the_engineer_gets_what_it_acts_on_and_the_planner_also_the_history`, and add:

```python
def _data_lineage(depth):
    node = {"status": "scored", "score": 0.7, "error": None, "recipe": {"optimizer.lr": 1e-4},
            "data": {"d": {"format": "video_caption_camera", "prompt_mode": None, "clips": 8, "weight": 1.0,
                           "sources": {"hf:org/walks": 8}}},
            "rationale": 'why\n\nPlan: {"hypothesis": "more turning clips"}\nData: notes',
            "metrics": {"frame_aesthetics": 0.6},
            "aggregates": {"dimensions": {"quality": 0.7}, "metrics": {"frame_aesthetics": 0.6},
                           "groups": {"category": {"Urban": {"quality": 0.7}},
                                      "viewpoint": {"first_person": {"quality": 0.7}}}}}
    return [{**node, "node_id": "root", "score": 0.6, "data": {}, "recipe": None, "rationale": None},
            *[{**node, "node_id": f"n{i}"} for i in range(1, depth)]]


def _edit_lineage(depth):
    process = {"phases": {"edit_self": 600, "improve_recipe": 7200},
               "roles": [{"phase": "improve_recipe", "role": "plan", "conversations": 1, "turns": 12,
                          "model_s": 300, "compactions": 0}],
               "tools": {"data_ingest": {"calls": 9, "errors": 4}}, "tool_errors": [],
               "local_errors": {"run_command_nonzero": 3}, "gpu_jobs": {"run": 2, "failed": 1, "gpu_s": 1200},
               "ingest": {"accepted": 5, "rejected": 4, "reasons": [{"reason": "moving clips need poses", "count": 4}]},
               "rounds": {"improve_recipe": {"plans": 1, "reports": 0, "questions": 2}}, "gates": {},
               "attempts": [{"phase": "edit_self", "attempt": 1, "outcome": "contract_failed"},
                            {"phase": "edit_self", "attempt": 2, "outcome": "passed"}]}
    node = {"status": "scored", "error": None, "edit": {"summary": "Added a pose check tool."},
            "code_diff_stats": [{"path": "agent/tools.py", "added": 30, "removed": 2}], "process": process}
    return [{"node_id": "root", "status": "scored", "error": None, "edit": None, "code_diff_stats": [],
             "process": {"attempts": []}},
            *[{**node, "node_id": f"n{i}"} for i in range(1, depth)]]


GUIDE = {"cause_and_effect": {"dimension": "physics", "weight": 4.5, "measures": "physics holds"}}


def _recipe_ctx(**over):
    from ar_contract.models import RecipeContext
    return RecipeContext(**{**dict(
        nodes_remaining=1, attempt=1, max_attempts=3, n_gpus=4, metric_guide=GUIDE,
        tunable_rules={"optimizer.lr": {"type": "float", "min": 0, "max": None}},
        recipe_guide={"optimizer.lr": {"base": 1e-4, "meaning": "peak rate"}},
        resolution_allowlist=[[416, 736]], lora_allowlist=[[64, 64]]), **over})


def test_the_data_digest_has_scores_the_metric_guide_and_the_recent_lineage():
    from agent.briefing import LINEAGE_SHOWN, recipe_context
    ctx = _recipe_ctx(attempt=2, lineage=_data_lineage(100), siblings=_data_lineage(3)[1:],
                      retry={"kind": "train", "log_tail": "oom\nline", "data_commit": "c1"})
    text = recipe_context(ctx, [{"plan": {"hypothesis": "h"}, "report": "r"}])
    assert text.index("## Retry") < text.index("## Recipe") < text.index("## Metrics") < text.index("## Siblings") \
        < text.index("## Lineage")
    assert "```\noom\nline\n```" in text and '"hypothesis": "h"' in text
    assert "| cause_and_effect | physics | 4.5 | physics holds |" in text
    assert "|  | root | n90 | n91 |" in text                    # score tables: the root, then the last 10
    assert "| viewpoint: first_person / quality | 0.7 |" in text and "Urban" not in text
    assert text.count("\n### n") == LINEAGE_SHOWN + 2 and "### n99: scored, score 0.7" in text and "### n89:" not in text
    assert "- Hypothesis: more turning clips" in text and "hf:org/walks x8" in text
    assert "Data formats" not in text and len(text) < 40_000


def test_a_node_that_failed_before_scoring_still_renders_in_the_data_digest():
    from agent.briefing import recipe_context
    failed = {"node_id": "n1", "status": "train_failed", "score": None, "error": "loss became nan",
              "metrics": {}, "aggregates": None, "data": {}, "recipe": None, "rationale": None}
    text = recipe_context(_recipe_ctx(lineage=[_data_lineage(1)[0], failed]), None)
    assert "### n1: train_failed, score -" in text and "- Error: loss became nan" in text


def test_the_edit_digest_shows_how_runs_went_and_no_score():
    from ar_contract.models import EditContext
    from agent.briefing import edit_context
    ctx = EditContext(nodes_remaining=4, attempt=1, max_attempts=3, lineage=_edit_lineage(3),
                      siblings=_edit_lineage(2)[1:],
                      archive={"nodes": [{"node_id": "root", "status": "scored"}, {"node_id": "n1", "status": "scored"},
                                         {"node_id": "n2", "status": "train_failed"}]})
    text = edit_context(ctx, {"tools": "agent/tools.py -- the agent's tools"}, None)
    assert ", score " not in text and "| score |" not in text and "0.7" not in text      # a status may say "scored"
    assert "### Scores" not in text and "### Metrics" not in text
    assert "3 nodes: scored 2, train_failed 1" in text
    assert "  - tools: agent/tools.py -- the agent's tools" in text
    for line in ("### n2: scored", "- Edit: Added a pose check tool.", "- Code changed: agent/tools.py (+30/-2)",
                 "- Time: edit_self 10 min, improve_recipe 120 min",
                 "- improve_recipe / plan: 12 model turns, 5 min of model time, 0 compactions",
                 "- Kernel tool calls (errors): data_ingest 9 (4)", "- Local tool failures: run_command_nonzero 3",
                 "- GPU jobs: 2 run, 1 failed, 20 GPU min",
                 "- Ingest: 5 accepted, 4 rejected; 4x moving clips need poses",
                 "- improve_recipe rounds: 1 plans, 0 reports back, 2 questions",
                 "- Failed attempts: edit_self 1 contract_failed"):
        assert line in text, line


def test_engineers_get_what_they_act_on_and_not_the_history():
    from agent.briefing import coder_context, engineer_context
    text = engineer_context(_recipe_ctx(lineage=_data_lineage(5), retry={"kind": "gate", "failures": ["too few clips"]}),
                            [{"plan": {"hypothesis": "h"}}])
    assert "## Recipe" in text and "## Metrics" in text and "too few clips" in text
    assert "## Lineage" not in text and "## Archive" not in text
    assert "  - harness: agent/harness.py" in coder_context({"harness": "agent/harness.py"})
```

In `test_a_rationale_line_that_only_looks_like_the_plan_is_ignored` change the plan JSON in the last assert to `'why\n\nPlan: {"hypothesis": "more turns"}\nData: x'` (the result is still `"more turns"`).

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_seed_agent.py -q -k "digest or engineers_get or rationale_line"`
Expected: FAIL (`unexpected keyword 'siblings'`-style signature errors, missing `coder_context`)

- [ ] **Step 3: Write `seed_agent/agent/briefing.py`**

```python
"""What a role is told first: the kernel's context as a Markdown digest. The exact data is in
/context/context.json and each finished node's files are under /nodes/<node>/; the digest points there
instead of repeating them. The two phases are told different things: the data roles see how earlier
nodes scored, on what data and with which data idea; the edit roles see how the agent code changed and
how each run went, and no score. Small sections come first and the lineage last, and only the most
recent LINEAGE_SHOWN ancestors are described, so the digest stays about the same size at any depth."""
from __future__ import annotations

import json
import re
from collections import Counter

LINEAGE_SHOWN = 10          # most recent ancestors described; the root stays in the score tables as the baseline
SIBLINGS_SHOWN = 10         # most recent finished children of the parent described
TEXT_CHARS = 4000           # safety cap on a free text quoted per node
TOP_NODES = 5               # best nodes listed from the whole archive
GROUP_AXES = ("instruction_kind", "viewpoint")     # groups in the group table; scene categories are in context.json
POINTER = "Exact data: /context/context.json. What each finished node left behind: /nodes/<node>/."


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "/").replace("\n", " ")


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows])


def clip(text: str, limit: int = TEXT_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " ..."


def _fields(value: dict) -> str:
    """A small dict as Markdown bullets: multi-line text in a code block, the rest inline."""
    out = []
    for key, item in value.items():
        if isinstance(item, str) and "\n" in item:
            out.append(f"- {key}:\n```\n{item.strip()}\n```")
        else:
            out.append(f"- {key}: {item if isinstance(item, str) else json.dumps(item)}")
    return "\n".join(out)


def retry_section(retry: dict | None, previous_plans: list | None) -> str:
    if not retry:
        return ""
    parts = ["## Retry", "The previous attempt of this phase failed:", _fields(retry)]
    for i, rnd in enumerate(previous_plans or [], 1):
        parts.append(f"Its plan {i}:\n```json\n{json.dumps(rnd, indent=1)}\n```")
    return "\n\n".join(parts)


def components_section(components: dict) -> str:
    return "- Components of this agent:\n" + "\n".join(f"  - {k}: {v}" for k, v in components.items())


# ---- improve_recipe: scores, data, data ideas ----

def metric_section(guide: dict) -> str:
    if not guide:
        return ""
    return ("## Metrics\n\nThe score is the weighted mean of these metrics. A dimension is the plain mean of its "
            "metrics and is not part of the score.\n\n"
            + table(["metric", "dimension", "weight", "measures"],
                    [[name, g.get("dimension"), g.get("weight"), g.get("measures")] for name, g in guide.items()]))


def best_section(archive: dict) -> str:
    nodes = archive.get("nodes") or []
    if not nodes:
        return ""
    best = sorted((n for n in nodes if n.get("score") is not None), key=lambda n: -n["score"])[:TOP_NODES]
    return "\n\n".join([
        "## Archive",
        f"{len(nodes)} nodes, {archive.get('n_scored', len(best))} scored. The best {len(best)}:",
        table(["node", "parent", "depth", "score", "status"],
              [[n["node_id"], n.get("parent_id"), n.get("depth"), n.get("score"), n.get("status")] for n in best])])


def _score_rows(nodes: list[dict], pick) -> list[list]:
    keys: list[str] = []
    for n in nodes:
        keys += [k for k in pick(n.get("aggregates") or {}) if k not in keys]
    return [[k, *[pick(n.get("aggregates") or {}).get(k) for n in nodes]] for k in keys]


def _groups(aggregates: dict) -> dict:
    """One row per group and dimension: the dimension's score over that group of evaluation items alone."""
    return {f"{axis}: {name} / {dimension}": value
            for axis, groups in (aggregates.get("groups") or {}).items() if axis in GROUP_AXES
            for name, dimensions in groups.items() for dimension, value in dimensions.items()}


def _data(n: dict) -> str:
    return "- Data: " + "; ".join(
        f"{name} ({d.get('format')}{'/' + d['prompt_mode'] if d.get('prompt_mode') else ''}): "
        f"{d.get('clips')} clips, weight {d.get('weight')}"
        + (", from " + ", ".join(f"{s} x{k}" for s, k in d["sources"].items()) if d.get("sources") else "")
        for name, d in n["data"].items())


def _hypothesis(rationale: str | None) -> str | None:
    """The data idea a node tested: run_task records its final plan in the rationale as a `Plan: {json}` line."""
    match = re.search(r"^Plan: (\{.*\})$", rationale or "", re.MULTILINE)
    try:
        return json.loads(match.group(1)).get("hypothesis") if match else None
    except ValueError:                                      # a line of the rationale that only looks like one
        return None


def data_node(n: dict) -> str:
    """An earlier node as the data planner needs it: what data idea it tested, on what data, with which recipe."""
    lines = [f"### {n['node_id']}: {n.get('status')}, score {_cell(n.get('score'))}"]
    if n.get("error"):
        lines.append(f"- Error: {clip(n['error'])}")
    if _hypothesis(n.get("rationale")):
        lines.append(f"- Hypothesis: {clip(_hypothesis(n['rationale']))}")
    if n.get("data"):
        lines.append(_data(n))
    if n.get("recipe"):
        lines.append("- Recipe: " + ", ".join(f"{k} {v}" for k, v in n["recipe"].items()))
    return "\n".join(lines)


# ---- edit_self: code edits and how each run went ----

def status_section(archive: dict) -> str:
    nodes = archive.get("nodes") or []
    if not nodes:
        return ""
    counts = Counter(n.get("status") for n in nodes)
    return f"## Archive\n\n{len(nodes)} nodes: " + ", ".join(f"{status} {n}" for status, n in counts.most_common())


def _process(p: dict) -> list[str]:
    """How a node's run went, one line per kind of fact. The tool error messages are in context.json."""
    lines = []
    if p.get("phases"):
        lines.append("- Time: " + ", ".join(f"{k.removesuffix('_s')} {round(v / 60)} min" for k, v in p["phases"].items()))
    lines += [f"- {r['phase']} / {r['role']}: {r['turns']} model turns, {round(r['model_s'] / 60)} min of model time, "
              f"{r['compactions']} compactions" for r in p.get("roles") or []]
    if p.get("tools"):
        lines.append("- Kernel tool calls (errors): "
                     + ", ".join(f"{tool} {v['calls']} ({v['errors']})" for tool, v in p["tools"].items()))
    if p.get("local_errors"):
        lines.append("- Local tool failures: " + ", ".join(f"{k} {v}" for k, v in p["local_errors"].items()))
    jobs = p.get("gpu_jobs") or {}
    if jobs.get("run"):
        lines.append(f"- GPU jobs: {jobs['run']} run, {jobs['failed']} failed, {round(jobs['gpu_s'] / 60)} GPU min")
    ingest = p.get("ingest") or {}
    if ingest.get("accepted") or ingest.get("rejected"):
        reasons = "".join(f"; {r['count']}x {clip(r['reason'], 200)}" for r in ingest.get("reasons") or [])
        lines.append(f"- Ingest: {ingest['accepted']} accepted, {ingest['rejected']} rejected{reasons}")
    lines += [f"- {phase} rounds: {r['plans']} plans, {r['reports']} reports back, {r['questions']} questions"
              for phase, r in (p.get("rounds") or {}).items()]
    failed = [a for a in p.get("attempts") or [] if a["outcome"] != "passed"]
    if failed:
        lines.append("- Failed attempts: " + ", ".join(f"{a['phase']} {a['attempt']} {a['outcome']}" for a in failed))
    if p.get("gates"):
        lines.append("- Pre-training checks: " + ", ".join(f"{k} {v}" for k, v in p["gates"].items()))
    return lines


def edit_node(n: dict) -> str:
    """An earlier node as the edit planner needs it: how the agent code changed and how the run went."""
    lines = [f"### {n['node_id']}: {n.get('status')}"]
    if n.get("error"):
        lines.append(f"- Error: {clip(n['error'])}")
    if n.get("edit"):
        lines.append(f"- Edit: {clip(n['edit'].get('summary', ''))}")
    if n.get("code_diff_stats"):
        lines.append("- Code changed: " + ", ".join(f"{d['path']} (+{d['added']}/-{d['removed']})"
                                                    for d in n["code_diff_stats"]))
    return "\n".join(lines + _process(n.get("process") or {}))


# ---- sections shared by both phases ----

def lineage_section(lineage: list[dict], describe, scores: bool) -> str:
    if not lineage:
        return ""
    shown = lineage[-LINEAGE_SHOWN:]
    older = "" if len(shown) == len(lineage) else (
        f" Only the last {len(shown)} are described here; all of them are in /context/context.json and /nodes/.")
    parts = ["## Lineage", f"From the root to the parent: {len(lineage)} nodes, oldest first.{older}"]
    if scores:
        columns = shown if shown[0] is lineage[0] else [lineage[0], *shown]       # the root: the baseline
        ids = [n["node_id"] for n in columns]
        parts += [
            "### Scores\n\n" + table(["", *ids], [["score", *[n.get("score") for n in columns]],
                                                  *_score_rows(columns, lambda a: a.get("dimensions") or {})]),
            "### Dimensions by group\n\n" + table(["", *ids], _score_rows(columns, _groups)),
            "### Metrics\n\n" + table(["", *ids], _score_rows(columns, lambda a: a.get("metrics") or {}))]
    return "\n\n".join([*parts, *[describe(n) for n in shown]])


def siblings_section(siblings: list[dict], describe, scores: bool) -> str:
    if not siblings:
        return ""
    shown = siblings[-SIBLINGS_SHOWN:]
    older = "" if len(shown) == len(siblings) else (
        f" Only the last {len(shown)} are described below; all of them are in /context/context.json and /nodes/.")
    header = ["node", "status", "score"] if scores else ["node", "status"]
    return "\n\n".join([
        "## Siblings",
        f"The parent's other children that have finished: {len(siblings)}, oldest first.{older}",
        table(header, [[n["node_id"], n.get("status"), n.get("score")][:len(header)] for n in siblings]),
        *[describe(n) for n in shown]])


def _node_and_recipe(ctx, previous_plans: list | None) -> list[str]:
    """What the data phase needs to act: this node's facts, the retry report, the recipe keys and the metrics."""
    guide = ctx.recipe_guide or {}
    keys = [[k, r.get("type"), r.get("min"), r.get("max"), (guide.get(k) or {}).get("base"),
             ctx.parent_recipe.get(k), (guide.get(k) or {}).get("meaning")]
            for k, r in (ctx.tunable_rules or {}).items()]
    pairs = lambda allowed, sep: ", ".join(f"{a}{sep}{b}" for a, b in allowed)
    return [
        POINTER,
        "## This node\n\n" + "\n".join([
            f"- Training GPUs: {ctx.n_gpus}",
            f"- Parent data commit: {ctx.parent_data_commit or 'none (the parent is the root)'}",
            f"- Clip pool: {ctx.clip_pool_size} clips in the archive",
            f"- Kernel tools: {', '.join(ctx.tools)}"]),
        retry_section(ctx.retry, previous_plans),
        "## Recipe\n\nTunable keys, with the base recipe's and the parent's values:\n\n"
        + table(["key", "type", "min", "max", "base", "parent", "meaning"], keys)
        + f"\n\nResolutions (height x width): {pairs(ctx.resolution_allowlist, 'x')}. "
          f"LoRA (rank/alpha): {pairs(ctx.lora_allowlist, '/')}. The full base recipe is `base_recipe` "
          f"in /context/context.json.",
        metric_section(ctx.metric_guide)]


def recipe_context(ctx, previous_plans: list | None) -> str:
    """The data planner's first message."""
    return "\n\n".join(s for s in [
        *_node_and_recipe(ctx, previous_plans),
        best_section(ctx.archive),
        siblings_section(ctx.siblings, data_node, scores=True),
        lineage_section(ctx.lineage, data_node, scores=True)] if s)


def engineer_context(ctx, previous_plans: list | None) -> str:
    """What the data engineer gets with the plan: the plan already carries what the planner drew from history."""
    return "\n\n".join(s for s in _node_and_recipe(ctx, previous_plans) if s)


def edit_context(ctx, components: dict, previous_plans: list | None) -> str:
    """The edit planner's first message."""
    return "\n\n".join(s for s in [
        POINTER,
        f"## This node\n\n- Nodes left in the run after this one: {ctx.nodes_remaining}\n" + components_section(components),
        retry_section(ctx.retry, previous_plans),
        status_section(ctx.archive),
        siblings_section(ctx.siblings, edit_node, scores=False),
        lineage_section(ctx.lineage, edit_node, scores=False)] if s)


def coder_context(components: dict) -> str:
    """What the coder gets with the plan."""
    return f"{POINTER}\n\n## This agent\n\n{components_section(components)}"
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_seed_agent.py -q -k "digest or engineers_get or rationale_line"`
Expected: PASS. In the first test, `text.count("\n### n") == LINEAGE_SHOWN + 2`: ten ancestors and two siblings.

- [ ] **Step 5: Commit**

```bash
git add seed_agent/agent/briefing.py tests/test_seed_agent.py
git commit -m "seed briefing: data roles get scores and the metric guide, edit roles get how runs went"
```

---

### Task 12: Seed agent roles, plans, talking back, prompts

**Files:**
- Rewrite: `seed_agent/agent/orchestration.py`
- Rewrite: `seed_agent/agent/prompts/{planner,data_engineer,edit_planner,coder}.md` (`compact.md` stays)
- Modify: `seed_agent/agent/tools.py:287-288` (`local_tools`), `seed_agent/agent/entry.py:12-13`
- Delete: `seed_agent/agent/knowledge/` (14 files)
- Test: `tests/test_seed_agent.py`

**Interfaces:**
- Consumes: `recipe_context`, `engineer_context`, `edit_context`, `coder_context` (Task 11); kernel tools `read_skill`, `ask`, `data_query` with typed fields (Task 8); `prompt_check` (Task 10).
- Produces:
  - `DataPlan{hypothesis, expected_change, data, constraints=""}`, `EditPlan{problem, evidence, mechanism, check}`; every text field at most `PLAN_FIELD_CHARS = 1500`.
  - Engineer tools `ask_planner(question)` (at most `MAX_QUESTIONS` per phase) and `request_replan(report)`.
  - `plans.json`: `[{"plan": {...}, "report"?: str, "questions"?: [{"question", "answer"}]}]`.
  - `EditResult.summary` is the coder's own summary.
  - `tools.local_tools(root: str, papers: bool) -> list`.
  - `COMPONENTS` with keys `tools, orchestration, briefing, harness, prompts, settings`.

- [ ] **Step 1: Update the tests**

In `tests/test_seed_agent.py`:

(a) Delete the knowledge tests: `test_system_prompt_is_the_role_prompt_then_the_knowledge_index`, `test_knowledge_files_start_with_their_name_and_when_to_use_them`, `test_every_role_prompt_points_to_the_knowledge_index`, `test_the_eval_knowledge_states_the_score_weights_of_the_kernel_config`, `test_the_eval_knowledge_lists_each_dimensions_metrics_as_wbench_groups_them`. In `test_prompts_and_knowledge_have_no_wrapped_commands_or_dated_notes` drop the knowledge glob and rename it `test_prompts_have_no_wrapped_commands_or_dated_notes`.

(b) Add the prompt tests:

```python
ROLES = ("planner", "data_engineer", "edit_planner", "coder")


def test_the_system_prompt_is_the_role_prompt_file_and_nothing_else(monkeypatch):
    monkeypatch.setenv("AR_TOKEN", "t")
    from agent.orchestration import system_prompt
    for role in ROLES:
        assert system_prompt(role) == (SEED / "agent" / "prompts" / f"{role}.md").read_text()
    assert not (SEED / "agent" / "knowledge").exists()


def test_every_role_prompt_has_the_same_four_parts_and_no_node_facts():
    from ar_kernel.contract.verify import prompt_check
    for role in ROLES:
        text = (SEED / "agent" / "prompts" / f"{role}.md").read_text()
        heads = [line for line in text.splitlines() if line.startswith("# ")]
        assert heads == ["# Mission", "# What you receive", "# How you work", "# Finish"], role
    assert prompt_check(SEED).ok


def test_the_held_out_rule_is_in_the_data_prompts_and_the_prompt_rule_in_the_edit_prompts():
    read = lambda role: (SEED / "agent" / "prompts" / f"{role}.md").read_text()
    for role in ("planner", "data_engineer"):
        assert "The evaluation set is held out." in read(role)
    for role in ("edit_planner", "coder"):
        assert "A prompt holds a role's mission and general behaviour" in read(role)
    assert "score" not in read("coder").lower()


def test_plan_fields_are_capped_so_a_plan_cannot_dictate_file_contents():
    from pydantic import ValidationError
    from agent.orchestration import PLAN_FIELD_CHARS, DataPlan, EditPlan
    assert PLAN_FIELD_CHARS == 1500
    ok = dict(problem="p", evidence="e", mechanism="m", check="c")
    EditPlan(**ok)
    with pytest.raises(ValidationError, match="at most 1500"):
        EditPlan(**{**ok, "mechanism": "x" * 1501})
    assert DataPlan(hypothesis="h", expected_change="c", data="d").constraints == ""
```

(c) In `test_submit_tool_validates_then_captures` (it builds `EditPlan`), use the new fields `problem`, `evidence`, `mechanism`, `check` wherever it passes the old `change`, `files`, `rationale`, `expected_effect`.

(d) In the `kernel` fixture add `from ar_kernel.tools.skills import register_skill_tool` and, after `register_ask_tool(mcp, kit, MockAsk())`, the line `register_skill_tool(mcp, kit)`.

(e) Replace `_scripts`:

```python
PLAN = {"hypothesis": "more walking clips", "expected_change": "camera path accuracy up",
        "data": "forward-walking clips with poses from the pool"}
EDIT_PLAN = {"problem": "the planner sees too many siblings", "evidence": "the first message is long",
             "mechanism": "show fewer siblings in the briefing", "check": "fewer planner compactions"}


def _scripts():
    """An accepted submit ends the role's run, so no reply follows one. Replies are consumed in order
    by whichever role calls the model next."""
    from ar_kernel.gateway.mock import message
    recipe = [
        [_call("submit_plan", {**PLAN, "data": "x" * 1501})],                       # planner: over the cap
        [_call("submit_plan", PLAN)],
        [_call("data_query", {"format": "video_caption_camera"}), _call("no_such_tool", {}),     # engineer
         _call("hf_download", {"repo": "x/y", "revision": "main", "patterns": ["*.mp4"]})],
        [_call("ask_planner", {"question": "May I reuse pool clips?"})],
        [_call("submit_plan", {**PLAN, "hypothesis": "sneaked in while answering"})],            # planner: refused
        [message("Yes, reuse them.")],
        [_call("request_replan", {"report": "downloads are disabled; the pool has clips"})],     # engineer
        [_call("submit_plan", {**PLAN, "hypothesis": "pool clips suffice"})],                    # planner
        [_call("submit_data_and_recipe", {"data_commit": C0, "notes": "pool clips", "rationale": "fits the data",
                                          "recipe": {"optimizer.max_steps": 200.4, "optimizer.lr": 1e-5}})],
    ]
    edit = [
        [_call("read_file", {"path": "agent/prompts/planner.md"}),       # planning may try things out ...
         _call("run_command", {"command": "echo junk >> agent/prompts/coder.md && touch agent/stray.py"})],
        [_call("submit_edit_plan", {**EDIT_PLAN, "problem": ""})],       # invalid: no problem
        [_call("submit_edit_plan", EDIT_PLAN)],
        [_call("edit_file", {"path": "agent/entry.py", "old": "def edit_self(ctx: EditContext)",
                             "new": "def edit_self(ctx: EditContext, extra)"})],       # coder: breaks the self-test
        [_call("submit_edit", {"summary": "too early"})],
        [_call("edit_file", {"path": "agent/entry.py", "old": "def edit_self(ctx: EditContext, extra)",
                             "new": "def edit_self(ctx: EditContext)"})],
        [_call("edit_file", {"path": "agent/briefing.py", "old": "SIBLINGS_SHOWN = 10", "new": "SIBLINGS_SHOWN = 8"})],
        [_call("submit_edit", {"summary": "The briefing now shows eight siblings."})],
    ]
    return {"recipe": recipe, "edit": edit}
```

(f) Replace the bodies of `test_improve_recipe_full_flow` and `test_edit_self_plans_then_edits` from their first `outputs = ...` line on:

```python
    outputs = _tool_outputs(rec, "n-recipe")
    assert any("Error invoking tool 'submit_plan'" in o and "at most 1500 characters" in o for o in outputs)
    assert any("no_such_tool is not a valid tool" in o for o in outputs)                       # unknown tool
    assert any(o.startswith("Error: ToolException(") and "downloads are disabled" in o
               for o in outputs)                                                 # kernel tool error reported
    assert any("You are answering a question" in o for o in outputs)             # no plan while answering
    assert "Yes, reuse them." in outputs                                         # the engineer got the answer
    kinds = [e["type"] for e in rec.read_events("n-recipe")]
    assert "tool.call" in kinds and "tool.error" in kinds                        # kernel-side records
    tasks = {str(m["content"]) for e in rec.read_events("n-recipe") if e["type"] == "llm.request"
             for m in rec.load_payload(e["payload"])["body"]["messages"] if m.get("role") == "user"}
    first = [t for t in tasks if t.startswith("<plan>")]
    assert first and all("Resolutions (height x width): 352x640." in t for t in first)
    assert any(t.startswith("The engineer asks:\n\nMay I reuse pool clips?") for t in tasks)
    assert any(t.startswith("<engineer_report>\ndownloads are disabled") for t in tasks)      # back to the planner
    assert any(t.startswith("The planner revised the plan.") and "pool clips suffice" in t for t in tasks)
    rounds = json.loads((tmp_path / "ws" / "plans.json").read_text())
    assert [r["plan"]["hypothesis"] for r in rounds] == ["more walking clips", "pool clips suffice"]
    assert rounds[0]["questions"] == [{"question": "May I reuse pool clips?", "answer": "Yes, reuse them."}]
    assert rounds[0]["report"].startswith("downloads are disabled") and "report" not in rounds[1]
    assert "pool clips suffice" in body["result"]["rationale"]                  # the final plan
    planner_tools = _tools_of(rec, "n-recipe", "# Mission\nYou choose the one data idea")
    assert {"hf_search", "data_query", "ask", "read_skill", "read_file", "run_command", "arxiv_search"} <= planner_tools
    assert not {"hf_download", "data_ingest", "data_commit", "caption_videos", "ask_planner"} & planner_tools
    engineer_tools = _tools_of(rec, "n-recipe", "# Mission\nYou build the training data")
    assert {"ask_planner", "request_replan", "read_skill", "data_ingest", "arxiv_search"} <= engineer_tools
    # The panel shows one conversation per role, each holding all of its rounds and questions.
    chats = _panel_chats(rec, "n-recipe", "improve_recipe")
    assert [role for role, _ in chats] == ["planner", "data_engineer"]
    planner_users = [i["text"] for i in chats[0][1] if i["kind"] == "user"]
    assert planner_users[0].startswith("<context>") and planner_users[-1].startswith("<engineer_report>")
    engineer_users = [i["text"] for i in chats[1][1] if i["kind"] == "user"]
    assert engineer_users[0].startswith("<plan>") and engineer_users[-1].startswith("The planner revised the plan.")
```

(keep that test's lines above `outputs = ...` as they are: they run the phase and check `data_commit` and the coerced recipe.)

```python
    assert body["result"]["summary"] == "The briefing now shows eight siblings."      # the coder's own words
    assert "SIBLINGS_SHOWN = 8" in (agent / "agent" / "briefing.py").read_text()
    # ... but what the planner changed is undone before the coder starts; its command log stays
    assert (agent / "agent" / "prompts" / "coder.md").read_text() == (SEED / "agent" / "prompts" / "coder.md").read_text()
    assert not (agent / "agent" / "stray.py").exists()
    assert list((tmp_path / "ws" / "tool_output").glob("run_command-*.log"))
    [round_] = json.loads((tmp_path / "ws" / "plans.json").read_text())
    assert round_["plan"] == EDIT_PLAN and "report" not in round_
    outputs = _tool_outputs(rec, "n-edit")
    assert any("Error invoking tool 'submit_edit_plan'" in o and "problem" in o for o in outputs)
    assert any("The self-test failed" in o and "exactly one parameter" in o for o in outputs)   # submit refused
    assert [role for role, _ in _panel_chats(rec, "n-edit", "edit_self")] == ["edit_planner", "coder"]
    planner_tools = _tools_of(rec, "n-edit", "# Mission\nYou improve this agent system")
    assert {"read_file", "list_dir", "run_command", "ask", "read_skill"} <= planner_tools
    assert not {"data_ingest", "hf_download", "caption_videos", "arxiv_search", "arxiv_read"} & planner_tools
    coder_tools = _tools_of(rec, "n-edit", "# Mission\nYou implement the edit plan")
    assert {"ask", "read_skill", "ask_planner", "request_replan"} <= coder_tools and "data_ingest" not in coder_tools
    coder_first = next(str(m["content"]) for e in rec.read_events("n-edit") if e["type"] == "llm.request"
                       for m in rec.load_payload(e["payload"])["body"]["messages"]
                       if m.get("role") == "user" and str(m["content"]).startswith("<edit_plan>"))
    assert "- Components of this agent:" in coder_first
```

(in `test_edit_self_plans_then_edits` keep the two lines above it that run the phase and assert `body["ok"]`; delete its old `summary` and `planner.md` asserts.)

(g) In `test_seed_agent_passes_contract_verification`: replace `lineage=_lineage(3), siblings=_lineage(3)[1:]` by building the two contexts separately — `EditContext(**common, lineage=_edit_lineage(3), siblings=_edit_lineage(3)[1:])` and `RecipeContext(**common, lineage=_data_lineage(3), siblings=_data_lineage(3)[1:], metric_guide=GUIDE, ...)` — and delete the `format_rules="rules text"` argument. Delete `format_rules=` anywhere else in the file.

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_seed_agent.py -q`
Expected: FAIL (old prompts, old plan schemas, `knowledge/` still there)

- [ ] **Step 3: Settings and local tools**

`seed_agent/agent/entry.py`, replace lines 12-13:

```python
MAX_ROUNDS = 4               # plans per phase: the first and up to three replans the engineer asks for
MAX_QUESTIONS = 5            # questions the engineer may ask the planner per phase
BRIEF_CHARS = 400_000        # safety cap on the context digest handed to a role; its own caps keep it under ~350,000
```

`seed_agent/agent/tools.py`, replace `local_tools`:

```python
def local_tools(root: str, papers: bool) -> list:
    """File and shell tools bound to `root`; `papers` adds arXiv search and reading."""
    return [*make_file_tools(root), *([arxiv_search, arxiv_read] if papers else [])]
```

- [ ] **Step 4: Delete the knowledge folder**

```bash
git rm -r seed_agent/agent/knowledge
```

- [ ] **Step 5: Write the four prompts**

`seed_agent/agent/prompts/planner.md`:

```markdown
# Mission
You choose the one data idea this node tests. AlayaWorld, a video world model, is fine-tuned at every node from the same released weights on the training data the node builds, and the fine-tune is then scored by a held-out evaluation. Only the training data, and the training settings that depend on it, may change. Your job is to decide what data would most improve the model, from what earlier nodes tried and how they scored. How the data gets built is the data engineer's job, not yours.

# What you receive
- `<context>`: this node's facts, the tunable recipe keys, what each metric measures and how much it weighs in the score, the best nodes, the parent's other children, and the lineage from the root to the parent with scores, data and data ideas. On a retry, what failed. The exact data is in `/context/context.json`.
- `/nodes/<node>/`: the files every finished node left behind.
- Tools: file and shell tools, paper search, kernel tools that only read, and `read_skill` for the kernel's reference notes. While you plan, what you change on disk is undone when your turn ends.
- Later, possibly: a question from the engineer, or an `<engineer_report>` asking for a new plan.

# How you work
- One node is one experiment. Choose one idea whose result will teach the next node something whether the score rises or falls.
- Ground the idea in evidence: which metrics or groups are weak, what earlier nodes already tried, and what followed. Do not repeat an idea that was tried unless you change what made it fail.
- Check that every source you name exists and can be reached before you plan on it.
- State intent, not procedure. The plan says what the training set should contain and why; the engineer decides the steps, the tools and the formats.
- The evaluation set is held out. Do not look for it or use its contents. Aiming data at what the metrics measure is the job; reproducing the evaluation's own material is not.
- When the engineer asks a question, answer it from what you know and can check. When it reports that the plan should change, keep what works and change what the report shows must change.

# Finish
Call `submit_plan`. Your last plan is the one this node is judged on.
```

`seed_agent/agent/prompts/data_engineer.md`:

```markdown
# Mission
You build the training data that tests the planner's idea, and the training recipe that fits it. The plan says what the data should be and why; you decide how to get it, and you answer for it being correct: every clip is what its caption and its camera pose say it is. You do not change the idea on your own: if it should change, the planner decides.

# What you receive
- `<plan>`: the idea, the change it should cause, the data it calls for and any constraints. A revised plan may follow; carry out the latest one.
- `<context>`: this node's facts, the tunable recipe keys with their limits, and what each metric measures. On a retry, what failed. The exact data is in `/context/context.json`.
- `/nodes/<node>/`: the files every finished node left behind.
- Tools: file and shell tools in `/workspace`, paper search, the kernel's data and GPU tools, and `read_skill` for the kernel's reference notes. Each tool's description says how to call it.

# How you work
- Prove each step on a few clips before you run it on all of them.
- Look at what you built before you commit it: measure and inspect clips rather than trusting a prompt, a file name or a tool's success message. Drop what fails.
- When a tool refuses something, read the reason and change the input. Do not repeat a call that failed.
- The recipe exists to fit the data you built. Change a key from its base value only for a reason you can state.
- Talk to the planner. If the plan is unclear or can be read in more than one way, call `ask_planner`. If what you found means the plan cannot work as written, or a different plan would test the idea better, call `request_replan` with what you did and found. Both are normal steps, not failures, and the work you did stays.
- The evaluation set is held out. Do not look for it or use its contents. Aiming data at what the metrics measure is the job; reproducing the evaluation's own material is not.

# Finish
Call `submit_data_and_recipe` when the data commit tests the plan. Say in the notes what the commit contains and where it departs from the plan.
```

`seed_agent/agent/prompts/edit_planner.md`:

```markdown
# Mission
You improve this agent system: the code and prompts under `/agent` that run the planner, the data engineer, yourself and the coder. You find where the system made its roles waste effort, lack a capability or lose information, and you decide the mechanism that removes it. You do not plan training data, judge data ideas or reason about how well a model did: whether the data a node chose was good is not your concern, only whether the system let its roles work well. Which edits survive is decided elsewhere, from how later nodes do.

# What you receive
- `<context>`: the components of this agent with their files, and how earlier nodes' runs went: each node's code edit, its changed files and its process (time per phase, model turns and time per role, compactions, kernel tool calls and errors, failed commands, GPU jobs, ingest results, plans, reports and questions, failed attempts). On a retry, what failed. The exact data is in `/context/context.json`.
- `/nodes/<node>/`: every finished node's files, including the transcript of every role. The transcripts are your main evidence.
- `/agent`: the code to change. `/code`: the parent's code, which is running you, read-only.
- Tools: file and shell tools, `ask`, and `read_skill` for the kernel's reference notes. While you plan, what you change on disk is undone when your turn ends.
- Later, possibly: a question from the coder, or an `<engineer_report>` asking for a new plan.

# How you work
- Start from behaviour, not outcomes. Read transcripts and process lines for what roles actually did: work repeated by hand, the same error met again and again, information one role had and another needed, context lost in a long run, a plan that dictated steps, a role that never asked.
- Pick the one problem whose removal would help most nodes, and show it with evidence: which transcript, which process line.
- Choose the mechanism in this order, and take a later one only when the earlier ones cannot fix the problem:
  1. a tool: a capability a role lacks or does by hand;
  2. orchestration and briefing: which roles run, what each is told first, what passes between them;
  3. the harness: the loop, compaction, how tool results are handled;
  4. prompt wording: only when a role's mission or general behaviour is itself wrong.
- A prompt holds a role's mission and general behaviour, and nothing else. A fact about a node, a run, a dataset, a result or a tool never goes into a prompt: what a role must know about the current node belongs in what it is told first, and what it must be able to do belongs in a tool.
- The kernel's tools and skills are fixed. You change the agent's own code only.
- State intent, not procedure. The plan names the problem, the evidence, the mechanism and how a later reader of the process lines would tell it worked. The coder decides the code.
- Look at what earlier edits changed and whether the behaviour they aimed at changed afterwards. Do not redo an edit that had no effect.
- Keep `edit_self(ctx)` and `improve_recipe(ctx)` in `agent/entry.py`, each with exactly one parameter.
- When the coder asks a question, answer it from what you know and can check.

# Finish
Call `submit_edit_plan`. Your last plan is the final one.
```

`seed_agent/agent/prompts/coder.md`:

```markdown
# Mission
You implement the edit plan in the agent code under `/agent` and prove that it works. The plan names the problem and the mechanism; you write the code. You do not re-decide the mechanism on your own: if it should change, the edit planner decides.

# What you receive
- `<edit_plan>`: the problem, its evidence, the mechanism and the check. A revised plan may follow; carry out the latest one.
- `<context>`: the components of this agent with their files.
- `/agent`: the code to change, and your working directory. `/code`: the parent's code, which is running you, read-only. `/nodes/<node>/`: every finished node's files.
- Tools: file and shell tools, `ask`, and `read_skill` for the kernel's reference notes.

# How you work
- Read the code you are about to change, and the code that calls it, before you change it.
- Change only what the mechanism needs. Keep the code simple: fewer and shorter files.
- Prove the change: run the code path you touched on a small input and look at what it returns. An import that succeeds proves nothing about behaviour.
- A prompt holds a role's mission and general behaviour, and nothing else. Never write a fact about a node, a run, a dataset, a result or a tool into a prompt.
- Talk to the edit planner. If the plan is unclear or can be read in more than one way, call `ask_planner`. If what you found means the mechanism cannot work as written, or another would fix the problem better, call `request_replan` with what you did and found. Both are normal steps, not failures, and your changes so far stay.

# Finish
Call `submit_edit` with a short summary of what you changed and how you checked it. It is accepted only if the self-test passes.
```

- [ ] **Step 6: Write `seed_agent/agent/orchestration.py`**

```python
"""Which roles run, in what order, with which tools and what they are told first.

A role = a system prompt and a tool list, run on the harness. It returns its result by calling a
submit tool; one that stops without submitting gets one reminder.
Both phases are a planner and an engineer. The planner submits a plan that states intent; the engineer
carries it out and has two ways back: ask_planner (the planner answers in its own conversation and the
plan stands) and request_replan (the planner revises the plan), up to MAX_QUESTIONS questions and
MAX_ROUNDS plans per phase. Each role keeps its conversation throughout. Every plan, report and
question is recorded in plans.json. A planner has the engineer's file and shell tools, but what it
changes on disk while planning is undone, and its kernel tools only read.
- improve_recipe: planner -> data engineer, who builds the data commit and writes the recipe
  (recipe_check must pass before the submission is accepted).
- edit_self: edit planner -> coder (the self-test must pass before the submission is accepted).
"""
from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ar_contract.client import chat_model, mcp_session
from ar_contract.models import EditContext, EditResult, RecipeContext, RecipeResult
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from .entry import AGENT_ROOT, BRIEF_CHARS, COMPACT_AT, CONTEXT_WINDOW, MAX_QUESTIONS, MAX_ROUNDS, MODEL, WORKSPACE
from .briefing import coder_context, edit_context, engineer_context, recipe_context
from .harness import build_react_agent
from .tools import discarded_changes, kernel_tools, local_tools, result_text, snap_timed_prompts, submit_tool

AGENT_PKG = Path(__file__).resolve().parent
RECORD = Path(WORKSPACE) / "plans.json"       # every plan, report and question; carried to a retry with the workspace
PLAN_FIELD_CHARS = 1500                       # a plan states intent: too short to dictate file contents
REMIND = ("You stopped without calling {tools}. Finish the task, then call {tools} with the result. "
          "The work is only recorded through {tools}.")
REPLAN = "Revise the plan. The engineer carries out the plan you submit next."
QUESTION = ("The engineer asks:\n\n{question}\n\nAnswer in plain text and change no files. Do not submit a plan: "
            "the plan changes only when the engineer calls request_replan.")
ANSWER_ONLY = ("You are answering a question, not planning: reply in plain text. The plan changes only when the "
               "engineer calls request_replan.")
PLANNING_KERNEL_TOOLS = {"hf_search", "hf_list_files", "data_query", "ask", "read_skill"}    # read-only
EDIT_KERNEL_TOOLS = {"ask", "read_skill"}

# Where each component lives in this agent, in the order an edit should consider them.
COMPONENTS = {
    "tools": "agent/tools.py -- the agent's own tools and the adapter for kernel tools (the kernel's tools and "
             "skills themselves are fixed)",
    "orchestration": "agent/orchestration.py -- the roles, their tools, the plan and question loop, the plan "
                     "and result schemas",
    "briefing": "agent/briefing.py -- what each role is told first, built from the kernel's context",
    "harness": "agent/harness.py -- the single-agent inner loop: ReAct graph, tool execution, auto-compaction",
    "prompts": "agent/prompts/*.md -- each role's mission and general behaviour",
    "settings": "agent/entry.py -- limits: plans and questions per phase, the context digest cap",
}


# ---- roles ----

def block(tag: str, text: str, **attrs) -> str:
    attributes = "".join(f' {key}="{value}"' for key, value in attrs.items())
    return f"<{tag}{attributes}>\n{text.strip()}\n</{tag}>"


def system_prompt(name: str) -> str:
    return (AGENT_PKG / "prompts" / f"{name}.md").read_text()


def brief(text: str) -> str:
    """The context digest (briefing.py) in a <context> block. BRIEF_CHARS is only a safety cap: the digest
    ends with the lineage, so a cut would only shorten that."""
    if len(text) > BRIEF_CHARS:
        text = text[:BRIEF_CHARS] + " ...[truncated: the full context is in /context/context.json]"
    return block("context", text)


class Role:
    """One role on the harness. Its conversation continues across calls to run() and answer()."""

    def __init__(self, name: str, tools: list, submissions: list, discard: tuple[str, ...] = ()) -> None:
        self.submissions = submissions            # the boxes of its submit tools
        self.discard = discard                    # dirs whose changes are undone after each run (planners)
        self.answering = False                    # inside answer(): its submit tools refuse
        self.messages: list = []
        model = chat_model(MODEL)
        model.bind_tools(tools)                   # a broken tool schema fails here, not at the first call
        self.agent = build_react_agent(model, tools, system_prompt(name),
                                       context_window=CONTEXT_WINDOW, compact_at=COMPACT_AT)

    def _submitted(self):
        return next((box for box in self.submissions if box.value is not None), None)

    async def _say(self, message: str) -> None:
        state = await self.agent.ainvoke({"messages": [*self.messages, HumanMessage(content=message)]})
        self.messages = state["messages"]

    async def run(self, task: str):
        """Work on `task` until a submit tool is called; returns that tool's box. One reminder if it stops early."""
        for box in self.submissions:
            box.value = None
        names = " or ".join(box.name for box in self.submissions)
        with discarded_changes(*self.discard, keep=(str(Path(WORKSPACE) / "tool_output"),)):
            for message in (task, REMIND.format(tools=names)):
                await self._say(message)
                if self._submitted() is not None:
                    return self._submitted()
        raise RuntimeError(f"the role finished without calling {names}")

    async def answer(self, question: str) -> str:
        """Reply to the engineer in this role's own conversation. Nothing on disk is undone here: the
        engineer's own tool calls may be writing at the same time."""
        self.answering = True
        try:
            await self._say(QUESTION.format(question=question))
        finally:
            self.answering = False
        return self.messages[-1].text or "(the planner gave no answer)"


async def ping() -> str:
    """Contract smoke run: after building every role, one model call proves the wiring."""
    agent = build_react_agent(chat_model(MODEL), [], "Reply with the single word ok.",
                              context_window=CONTEXT_WINDOW, compact_at=COMPACT_AT)
    return (await agent.ainvoke({"messages": [HumanMessage(content="ping")]}))["messages"][-1].text


def planner_role(name: str, tool: str, description: str, schema: type[BaseModel], tools: list,
                 discard: tuple[str, ...]) -> tuple[Role, object]:
    """A planner and the box of its plan tool, which refuses while the planner is answering a question."""
    async def planning(_) -> None:
        if role.answering:
            raise ToolException(ANSWER_ONLY)
    plan_tool, plan = submit_tool(tool, description, schema, planning)
    role = Role(name, [plan_tool, *tools], [plan], discard=discard)
    return role, plan


class Question(BaseModel):
    question: str = Field(min_length=1, description="what you need to know to carry out the plan")


class Replan(BaseModel):
    report: str = Field(min_length=1, description="what you did and found, and why the plan should change")


def save(rounds: list) -> None:
    RECORD.write_text(json.dumps(rounds, indent=1))


def talk_back(planner: Role, rounds: list) -> tuple[list, object]:
    """The engineer's two ways back to the planner, as tools, and the box of request_replan."""
    lock = asyncio.Lock()                         # parallel questions are answered one after the other

    async def ask_planner(question: str) -> str:
        async with lock:
            if sum(len(r.get("questions", [])) for r in rounds) >= MAX_QUESTIONS:
                raise ToolException(f"No questions left: all {MAX_QUESTIONS} are used. Decide yourself, or call "
                                    f"request_replan if the plan should change.")
            answer = await planner.answer(question)
            rounds[-1].setdefault("questions", []).append({"question": question, "answer": answer})
            save(rounds)
            return answer

    async def replans_left(_) -> None:
        if len(rounds) >= MAX_ROUNDS:
            raise ToolException(f"No replans left: all {MAX_ROUNDS} plans are used. Finish with the current plan.")

    ask = StructuredTool(name="ask_planner", args_schema=Question, coroutine=ask_planner,
                         description="Ask the planner a question about the plan and get its answer. The plan "
                                     "stays as it is. Use it whenever the plan is unclear or leaves a choice open.")
    replan_tool, replan = submit_tool(
        "request_replan", "Stop and send a report back to the planner, who then revises the plan. The normal step "
        "when what you found changes what should be done.", Replan, replans_left)
    return [ask, replan_tool], replan


@dataclass
class Team:
    planner: Role
    plan: object                  # the submit boxes
    engineer: Role
    done: object
    replan: object
    rounds: list                  # every plan, with the engineer's report and questions


async def plan_and_engineer(team: Team, *, task: str, show, engineer_context: str) -> None:
    """Plan, carry out, and replan when the engineer asks, until the engineer finishes. Every round is
    appended to team.rounds and saved."""
    while True:
        await team.planner.run(task)
        team.rounds.append({"plan": team.plan.value.model_dump()})
        save(team.rounds)
        work = show(team.plan.value)
        work = f"The planner revised the plan.\n\n{work}" if team.engineer.messages else f"{work}\n\n{engineer_context}"
        if await team.engineer.run(work) is team.done:
            return
        team.rounds[-1]["report"] = team.replan.value.report
        save(team.rounds)
        task = f"{block('engineer_report', team.replan.value.report)}\n\n{REPLAN}"


def check_roles() -> None:
    """Build every role of both phases (prompts, tool schemas) without calling a model."""
    recipe_team(None, None, [])
    edit_team([])


def previous_plans(ctx) -> list | None:
    """The failed attempt's plans, on a retry: read before this attempt overwrites the record."""
    return json.loads(RECORD.read_text()) if ctx.retry and RECORD.exists() else None


def plan_text(length: str, description: str, **kwargs):
    return Field(max_length=PLAN_FIELD_CHARS, description=f"{description} ({length})", **kwargs)


# ---- improve_recipe ----

class DataPlan(BaseModel):
    hypothesis: str = plan_text("one or two sentences", "the one data idea this node tests, and what in "
                                "earlier nodes suggests it", min_length=1)
    expected_change: str = plan_text("a sentence", "which metrics or groups should move, and in which direction",
                                     min_length=1)
    data: str = plan_text("a short paragraph", "what the training set should contain and where it comes from "
                          "(sources you checked exist); not the steps to build it", min_length=1)
    constraints: str = plan_text("optional", "what the engineer must keep or must not do, if anything", default="")


class DataAndRecipe(BaseModel):
    data_commit: str = Field(min_length=1, description="the data_commit id to train on")
    notes: str = Field(description="what the commit contains, and where it departs from the plan")
    recipe: dict[str, float] = Field(description="tunable key -> value")
    rationale: str = Field(min_length=1, description="why each changed key has its value")


def typed(recipe: dict, rules: dict) -> dict:
    return {key: int(round(value)) if rules.get(key, {}).get("type") == "int" else float(value)
            for key, value in recipe.items()}


def recipe_team(ctx: RecipeContext | None, session, ktools: list) -> Team:
    async def recipe_passes(result: DataAndRecipe) -> None:
        check = json.loads(result_text(await session.call_tool("recipe_check", {
            "recipe": typed(result.recipe, ctx.tunable_rules), "data_commit": result.data_commit})))
        if not check.get("ok"):
            raise ToolException(f"recipe_check failed: {check.get('failures')}")

    rounds: list[dict] = []
    planner, plan = planner_role(
        "planner", "submit_plan", "Submit the data plan for this node.", DataPlan,
        [*[t for t in ktools if t.name in PLANNING_KERNEL_TOOLS], *local_tools(WORKSPACE, papers=True)],
        discard=(WORKSPACE,))
    done_tool, done = submit_tool("submit_data_and_recipe", "Submit the data commit and the recipe to train on. "
                                  "It is accepted only if recipe_check passes.", DataAndRecipe, recipe_passes)
    back, replan = talk_back(planner, rounds)
    engineer = Role("data_engineer", [*ktools, *local_tools(WORKSPACE, papers=True), snap_timed_prompts,
                                      done_tool, *back], [done, replan])
    return Team(planner, plan, engineer, done, replan, rounds)


async def run_task(ctx: RecipeContext) -> RecipeResult:
    async with mcp_session() as session:
        ktools = await kernel_tools(session)
        team = recipe_team(ctx, session, ktools)
        previous = previous_plans(ctx)
        # Both built in a dry run too: it gets the real context.
        context, for_engineer = brief(recipe_context(ctx, previous)), brief(engineer_context(ctx, previous))
        if ctx.dry_run:
            return RecipeResult(data_commit=ctx.parent_data_commit or "dry-run", recipe={},
                                rationale=f"dry run: roles and the first message built, {len(ktools)} kernel "
                                          f"tools, model said {await ping()!r}")
        await plan_and_engineer(team, task=context, show=lambda p: block("plan", p.model_dump_json(indent=2)),
                                engineer_context=for_engineer)
    r = team.done.value
    return RecipeResult(data_commit=r.data_commit, recipe=typed(r.recipe, ctx.tunable_rules),
                        rationale=f"{r.rationale}\n\nPlan: {json.dumps(team.rounds[-1]['plan'])}\nData: {r.notes}")


# ---- edit_self ----

class EditPlan(BaseModel):
    problem: str = plan_text("one or two sentences", "where the agent system made its roles waste effort, lack a "
                             "capability or lose information", min_length=1)
    evidence: str = plan_text("a short paragraph", "where it shows: nodes, transcript files, process lines",
                              min_length=1)
    mechanism: str = plan_text("a short paragraph", "what should change in the agent system and in which "
                               "component, as intent; not the code or the text to write", min_length=1)
    check: str = plan_text("a sentence", "what a later reader of the process lines and transcripts would see if "
                           "it worked", min_length=1)


class EditSummary(BaseModel):
    summary: str = Field(min_length=1, description="one paragraph: what you changed and how you checked it")


def selftest(root: str) -> list[str]:
    """The contract's static and import checks, plus building every role, run locally so the agent can fix
    itself. agent.orchestration is imported too: agent.entry imports it only when an entry point runs."""
    errors = []
    try:
        tree = ast.parse((Path(root) / "agent" / "entry.py").read_text())
        top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in ("edit_self", "improve_recipe"):
            fn = top.get(name)
            if (fn is None or len(fn.args.posonlyargs + fn.args.args) != 1 or fn.args.vararg
                    or fn.args.kwarg or fn.args.kwonlyargs):
                errors.append(f"agent/entry.py needs top-level {name}(ctx) with exactly one parameter")
    except (OSError, SyntaxError, ValueError, RecursionError) as exc:
        errors.append(f"agent/entry.py: {exc}")
    r = subprocess.run([sys.executable, "-c", "import agent.entry, agent.orchestration as o; o.check_roles()"],
                       cwd=root, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        errors.append(r.stderr[-6000:])
    return errors


def edit_team(ktools: list) -> Team:
    async def selftest_passes(_) -> None:
        errors = selftest(AGENT_ROOT)
        if errors:
            raise ToolException("The self-test failed. Fix these first:\n" + "\n".join(errors))

    rounds: list[dict] = []
    kernel = [t for t in ktools if t.name in EDIT_KERNEL_TOOLS]
    planner, plan = planner_role(
        "edit_planner", "submit_edit_plan", "Submit the edit plan.", EditPlan,
        [*kernel, *local_tools(AGENT_ROOT, papers=False)], discard=(AGENT_ROOT, WORKSPACE))
    done_tool, done = submit_tool("submit_edit", "Submit the summary of the change you made. It is accepted "
                                  "only if the self-test passes.", EditSummary, selftest_passes)
    back, replan = talk_back(planner, rounds)
    engineer = Role("coder", [*kernel, *local_tools(AGENT_ROOT, papers=False), done_tool, *back], [done, replan])
    return Team(planner, plan, engineer, done, replan, rounds)


async def run_meta(ctx: EditContext) -> EditResult:
    async with mcp_session() as session:
        team = edit_team(await kernel_tools(session))
        task = brief(edit_context(ctx, COMPONENTS, previous_plans(ctx)))    # built in a dry run too: it gets the real context
        if ctx.dry_run:
            return EditResult(summary=f"dry run: roles and the first message built, model said {await ping()!r}")
        await plan_and_engineer(team, task=task, show=lambda p: block("edit_plan", p.model_dump_json(indent=2)),
                                engineer_context=brief(coder_context(COMPONENTS)))
    return EditResult(summary=team.done.value.summary)
```

- [ ] **Step 7: Run the seed agent tests**

Run: `$PY -m pytest tests/test_seed_agent.py tests/test_seed_harness.py -q`
Expected: PASS. If `test_every_listed_component_exists_in_the_seed` fails, the first word of a `COMPONENTS` value is not a path that exists.

- [ ] **Step 8: Run the seed agent through the real contract check (needs Docker)**

Run: `$PY -m pytest tests/test_seed_agent.py -q -m docker -k contract_verification`
Expected: PASS, including the new `prompts` step

- [ ] **Step 9: Commit**

```bash
git add -A seed_agent tests/test_seed_agent.py
git commit -m "seed agent: intent-only plans, ask_planner, score-blind edit roles, rewritten prompts; knowledge moved to the kernel"
```

---

### Task 13: Panel shows tool parameters

**Files:**
- Modify: `panel/chat.py` (new `tool_lines`; used in `chat_items`)
- Modify: `panel/views.py:158-161` (`trace_detail`)
- Test: `tests/test_panel_data.py`

**Interfaces:**
- Produces: `chat.tool_lines(tools: list[dict]) -> str` — one line per offered tool: `- name(arg: type, optional?: type): description`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_panel_data.py`)

```python
def test_tools_offered_show_each_tools_parameters():
    from panel.chat import tool_lines
    tools = [{"type": "function", "function": {"name": "data_query", "description": "Search the pool.", "parameters": {
        "type": "object", "required": ["limit"], "properties": {
            "limit": {"type": "integer"},
            "camera_motion": {"anyOf": [{"enum": ["moving", "static"], "type": "string"}, {"type": "null"}]},
            "clip_ids": {"type": "array", "items": {"type": "string"}}}}}},
        {"type": "function", "function": {"name": "bare"}}]
    assert tool_lines(tools) == ("- data_query(limit: integer, camera_motion?: moving | static, clip_ids?: [string]): "
                                 "Search the pool.\n- bare(): ")
```

- [ ] **Step 2: Run it to see it fail**

Run: `$PY -m pytest tests/test_panel_data.py -q -k tools_offered`
Expected: FAIL, `ImportError: cannot import name 'tool_lines'`

- [ ] **Step 3: Implement**

In `panel/chat.py`, below `request_tools`:

```python
def _type(schema: dict) -> str:
    if "enum" in schema:
        return " | ".join(map(str, schema["enum"]))
    if "anyOf" in schema:
        return " | ".join(_type(s) for s in schema["anyOf"] if s.get("type") != "null")
    if schema.get("type") == "array":
        return f"[{_type(schema.get('items') or {})}]"
    return str(schema.get("type", "any"))


def tool_lines(tools: list[dict]) -> str:
    """Each offered tool as the model saw it: its parameters (optional ones marked ?) and its description."""
    lines = []
    for tool in tools:
        f = tool.get("function") or {}
        schema = f.get("parameters") or {}
        required = set(schema.get("required") or [])
        params = ", ".join(f"{name}{'' if name in required else '?'}: {_type(p)}"
                           for name, p in (schema.get("properties") or {}).items())
        lines.append(f"- {f.get('name', '?')}({params}): {f.get('description', '')}")
    return "\n".join(lines)
```

In `chat_items` replace the three lines that build `lines` and append the `tools` item with:

```python
            if tools:
                items.append(item("tools", f"Tools offered ({len(tools)})", tool_lines(tools)))
```

In `panel/views.py` add `tool_lines` to the `from .chat import (...)` list and in `trace_detail` replace the `"text": "\n".join(...)` value with `"text": tool_lines(tools)`.

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_panel_data.py tests/test_panel_ui.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add panel/chat.py panel/views.py tests/test_panel_data.py
git commit -m "panel: Tools offered shows each tool's parameters"
```

---

### Task 14: Nothing an agent sees names the benchmark

**Files:**
- Test: `tests/test_isolation.py`
- Modify: whatever the test finds

**Interfaces:**
- Consumes: every earlier task.

- [ ] **Step 1: Write the test** (append to `tests/test_isolation.py`)

```python
import asyncio
import re
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"wbench|benchmark|meituan|\b(test|eval\w*|proxy) cases?\b", re.IGNORECASE)


def _agent_visible_texts(tmp_path):
    """Everything an agent can read that the kernel or the seed ships: the seed agent, the contract
    package, the skills, and every kernel tool's description and schema."""
    from ar_kernel.contract.verify import _MockData, _MockHf
    from ar_kernel.telemetry.recorder import Recorder
    from ar_kernel.tools.ask import MockAsk, register_ask_tool
    from ar_kernel.tools.captioner import register_caption_tool
    from ar_kernel.tools.context import TokenRegistry
    from ar_kernel.tools.data_tools import register_data_tools
    from ar_kernel.tools.gpu_jobs import build_gpu_backends, register_gpu_tools
    from ar_kernel.tools.hf_tools import register_hf_tools
    from ar_kernel.tools.jobs import JobQueue, register_job_tools
    from ar_kernel.tools.server import ToolKit, new_mcp
    from ar_kernel.tools.skills import register_skill_tool
    texts = {}
    for root in (REPO / "seed_agent", REPO / "contract", REPO / "kernel" / "ar_kernel" / "skills"):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".md", ".txt"):
                texts[str(path.relative_to(REPO))] = path.read_text(encoding="utf-8")
    rec = Recorder(tmp_path)
    reg = TokenRegistry(rec)
    queue = JobQueue(rec, threading.Lock(), wait_cap_s=5.0)
    try:
        for backend in build_gpu_backends(CFG, tmp_path, [0, 1, 2, 3], reg, rec):
            queue.register(backend)
        kit, mcp = ToolKit(reg, rec), new_mcp()
        register_data_tools(mcp, kit, _MockData())
        register_hf_tools(mcp, kit, _MockHf())
        register_job_tools(mcp, kit, queue)
        register_gpu_tools(mcp, kit, queue)
        register_caption_tool(mcp, kit, queue)
        register_ask_tool(mcp, kit, MockAsk())
        register_skill_tool(mcp, kit)
        for tool in asyncio.run(mcp.list_tools()):
            texts[f"tool {tool.name}"] = f"{tool.description}\n{json.dumps(tool.input_schema)}"
    finally:
        queue.shutdown()
    return texts


def test_nothing_an_agent_sees_names_the_evaluation(tmp_path):
    texts = _agent_visible_texts(tmp_path)
    assert {"tool rollout_alayaworld", "tool read_skill", "tool data_ingest"} <= set(texts)
    found = {name: sorted(set(m.group(0) for m in FORBIDDEN.finditer(text))) for name, text in texts.items()}
    assert {name: hits for name, hits in found.items() if hits} == {}


def test_agent_facing_messages_are_neutral():
    from ar_kernel.context_bundle import KERNEL_FAILURES
    from ar_kernel.eval.score import AGENT_METRICS
    from ar_kernel.isolation import EXCLUDED_CLIP, EXCLUDED_PROMPT
    words = " ".join([EXCLUDED_CLIP, EXCLUDED_PROMPT, *KERNEL_FAILURES.values(),
                      *[f"{alias} {text}" for alias, _, text in AGENT_METRICS.values()]])
    assert not FORBIDDEN.search(words) and "leak" not in words.lower()
```

- [ ] **Step 2: Run it**

Run: `$PY -m pytest tests/test_isolation.py -q -k "agent_sees or neutral"`
Expected: PASS if the earlier tasks left nothing behind. Each failure names the file or tool and the words found.

- [ ] **Step 3: Fix what it finds**

For each hit, reword the string at its source so that it describes the tool or the model in its own terms (the rule of Task 4, Step 6). Kernel-internal names stay: only files and tools listed by `_agent_visible_texts` are in scope. Re-run Step 2 until it passes.

- [ ] **Step 4: Commit**

```bash
git add -A tests/test_isolation.py kernel seed_agent contract
git commit -m "test: nothing the seed, the contract, the skills or a kernel tool shows an agent names the evaluation"
```

---

### Task 15: Full suite, a smoke run, and main

**Files:** none new.

- [ ] **Step 1: Run the whole default suite**

Run: `$PY -m pytest -q`
Expected: PASS. Fix any test that still uses an old name (`format_rules`, `UNFINISHED`, `["llm"]`, `{"filter": ...}`, `local_tools(root)` with one argument, the old AlayaWorld item keys, the old plan fields).

- [ ] **Step 2: Run the Docker-marked tests**

Run: `$PY -m pytest -q -m docker`
Expected: PASS (the seed agent and the fixture agents pass or fail the contract at the step each test names; the new `prompts` step sits between `static` and `import`).

- [ ] **Step 3: Check the agent-visible surface by hand**

Run: `grep -rni "wbench" seed_agent contract kernel/ar_kernel/skills ; echo "exit $?"`
Expected: no lines, `exit 1`.

- [ ] **Step 4: Start a short real run and look at it** (GPUs 0,1,2,3; ask the user before starting if a run is already active)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .envs/autoresearcher/bin/ar run --max-nodes 1
```

In the panel, open node n1 and check, in this order: the edit planner's first message has no score and has process lines; "Tools offered" lists `read_skill`, `ask_planner` (engineers only) and parameters for every tool; the data planner's first message has the Metrics table with alias names; `plans.json` under each attempt's workspace has `problem/evidence/mechanism/check` or `hypothesis/expected_change/data/constraints`. Report what you saw; do not claim the run is healthy without having looked.

- [ ] **Step 5: Fast-forward main and push**

```bash
git checkout main
git merge --ff-only seed-agent-rewrite
git push origin main
```

Only `AutoResearcher` changed; `WorldModel` and `WBench` need no push.
