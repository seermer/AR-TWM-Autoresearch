# Run panel — design

Date: 2026-09-27. Status: approved in conversation, awaiting written review.

## 1. Purpose

A visualization layer for watching one AutoResearcher run, live or finished. The run's own
records (event logs, stored payloads, `archive.db`, `agents.git`, node folders, clip store)
are the source of truth and are written by the kernel, faithfully and without regard to
readability. The panel only reads them and makes them easy to follow. It never writes to a
run, never controls a run, and never changes how the kernel records anything.

Success: from one link, the owner can follow everything a run did — every event, every LLM
conversation turn (including across compaction), every tool call and output, every code
edit, the training data, training, the eval generations and scores, selection and cost —
down to the raw record.

## 2. Decisions (user, 2026-09-27)

- Gradio, served through Gradio's public share link (`share=True`), which must work fully.
- Login required: Gradio's built-in auth with `PANEL_USER` / `PANEL_PASSWORD` from
  `AutoResearcher/.env`. The panel refuses to start if either is empty. One owner, no
  per-person accounts.
- One run per panel process: `--run-id` is required.
- The overview auto-refreshes (about every 15 s); every other tab loads on demand, so a
  video being watched or a prompt being read never jumps.
- No redaction in the panel: records are shown exactly as written (the kernel already
  redacts secrets when it records them).
- Compaction is stitched in the panel only (section 6); the records stay raw.

## 3. Structure

- `AutoResearcher/panel/` — a Python package:
  - a data layer: reads and normalizes the run's files; no Gradio import; unit-tested;
  - a UI layer: Gradio Blocks built on the data layer;
  - `__main__.py`: the launcher.
- Environment: its own prefix env `AutoResearcher/.envs/panel` (git-ignored like the other
  `.envs/*`), created by `scripts/make_panel_env.sh` from a pinned `panel/requirements.txt`
  (gradio, plotly, pandas, zstandard, pyyaml). The kernel's `autoresearcher` env is not
  touched, so Gradio's dependencies cannot disturb the gateway or MCP stack.
- Launch: `.envs/panel/bin/python -m panel --run-id <run> [--port N] [--no-share]`. Prints
  the share link. All paths resolve relative to the repo (portable across machines).
- The panel does not import the kernel package. It reads the file formats directly (JSONL
  events, zstd JSON payloads, SQLite, git); the formats are pinned by tests (section 9).

## 4. Read-only guarantees

- `archive.db` is opened per query as `file:...?mode=ro` (SQLite URI), with a short retry if
  the kernel holds a write lock.
- `agents.git` is read only through `git show`, `git diff`, `git ls-tree`, `git log`.
- Nothing is created, changed or deleted under `runs/`. No cache files are written there.
- Gradio serves files only from the run folder (`allowed_paths` = the run dir); nothing else
  on the machine is reachable through the link.

## 5. Tabs

1. **Overview** (auto-refresh): loop state (running/stopped, from `control/`), current node,
   phase and attempt; tokens, spend and call count; alerts with timestamps; node table (id,
   parent, status, score, change vs parent, edit component, attempt counts, duration,
   error); score across nodes (chart); GPU memory and utilization over time (from
   `gpu.jsonl`); disk free; the last 50 events.
2. **Node** (pick a node): lineage to the root; edit summary and plan (`edit.json`);
   `rationale.md`; recipe and its difference from the parent's; the data commit; per-metric
   scores next to parent and root; per-phase timings; attempts with outcome and detail.
3. **Trace**: every event, paged (100 rows), filtered by node, phase, attempt, event type,
   component and free text. Selecting a row shows the full detail: an LLM request as a chat
   (system, user, assistant, tool messages; tool definitions folded); an LLM response
   (reasoning, text, tool calls, usage, cost, latency); a kernel tool call's arguments and
   full result or error; a subprocess's command, env, cwd and full stdout/stderr; a sandbox
   run's stdout/stderr and stats. The raw event JSON and raw payload JSON are always shown.
4. **Conversations**: per node, phase and attempt, each agent conversation as one chat:
   system and user prompts, each turn's reasoning (folded), tool calls and tool outputs
   inline, tokens and cost per turn; compaction handled as in section 6; links to
   `workspace/tool_output/run_command-*.log` where the model saw only the tail.
5. **Code edits**: each node's diff against its parent's agent commit; every `edit_self`
   attempt's diff (including failed attempts) with its contract report; a file browser for
   any agent commit.
6. **Training data**: each data commit's manifest (datasets, clip counts, formats); a paged
   clip gallery — video, caption with timed segments, a top-down plot of the camera path
   from the pose, provenance, license, leakage-check result, warnings; rejected candidates
   with their reasons (from `ingest.rejected` events); the kept staging files per attempt
   (converted and rejected candidates, generated rollouts and images, annotations).
7. **Training**: the loss curve parsed from `train.log`; the resolved train config; gate
   results (pass/fail with failures); duration; GPU use during training.
8. **Eval**: a node's rendered videos (40 on the proxy set), each with its case prompt and
   per-case metric scores; a side-by-side of two nodes on the same case; aggregate and
   strata tables (`aggregates.json`).
9. **Selection**: every `selection_events` row — the chosen parent and each candidate's
   value, penalty and probability.
10. **Cost and LLM stats**: tokens, cost, latency and errors per node, phase and role.
11. **Files**: a read-only browser of the whole run folder, rendering text, JSON, video
    and images, for anything the other tabs do not cover (vLLM logs, render logs, WBench
    output, config snapshots).

Gradio limits and their handling: long chats use `gr.Chatbot` message metadata to fold
reasoning and tool calls; diffs are shown as colored unified diffs; charts use plotly; long
lists are paged; videos load only when opened.

## 6. Conversations across compaction

What the records contain (gateway linking is by resent-history prefix):

1. The summarizer request is the agent's full history plus one appended instruction
   message; the gateway links it into the same conversation as the next turn. A forced
   retry (tools disabled) is one more turn.
2. The agent then replaces its history with one user message: a continuation preamble
   containing the summary.
3. That next request no longer shares the old prefix, so the gateway starts a new
   `conversation_id` at turn 0. The records carry no explicit link.

Stitching (panel data layer only; records untouched): conversation B continues
conversation A when both belong to the same node, phase and attempt, B starts after A's
last call, and B's first user message contains A's last response text verbatim. A's last
call is then the compaction (its final user message is the compaction instruction and its
response is the summary). The rule does not depend on the wording of `compact.md`, which
the agent may edit. Conversations with no match stand alone. Chains of any length are
followed.

Display: pre-compaction turns as they were; then a marked turn "Compaction #k: ~X tokens →
~Y tokens" showing the instruction as the user message and the summary as the reply (a
forced retry shown as "Compaction #k, retry without tools"); then the continuation, whose
first message (preamble plus summary) is folded as "context after compaction". The Trace
tab keeps every raw request and response with both conversation ids.

Each chat is built from the last request of each conversation segment plus that request's
response, not by concatenating every request (each request resends the full history).

## 7. Live-run data handling

- Event files are read incrementally by byte offset; a trailing line without a newline is
  left for the next read. Only event records are kept in memory; payloads are loaded on
  demand with an in-memory LRU cache.
- Target: the overview refresh completes in under 1 s for a run of about 50 nodes.

## 8. Errors and missing data

- Data not there yet (a running node, no eval, no compaction): the tab says so.
- An exception inside a view is shown in that tab; the app and other tabs keep working.
- Startup fails with a clear message when the login is unset or the run does not exist.

## 9. Testing

- Data-layer unit tests on small fake runs whose events are written with the kernel's own
  `Recorder` (run from the kernel test env), so formats are the real ones: incremental
  reading including a half-written line; payload loading; conversation assembly and
  compaction stitching (one compaction, a chain, a forced retry, no match); diffs and
  recipe comparison; selection history; loss-curve parsing; a missing-data case per tab.
- Read-only test: load every view against a fake run, then assert that no file, mtime or
  database byte under the run folder changed.
- UI smoke test: build the Blocks app and call every tab's handler on the fake run.
- Real check: open the panel on `runs/loopcheck_20260927` (full n1 with data, training
  and eval) before using it on a new run.

## 10. Out of scope

Controlling a run (stop, resume), editing anything, multi-run views, per-person accounts,
changing what the kernel records.
