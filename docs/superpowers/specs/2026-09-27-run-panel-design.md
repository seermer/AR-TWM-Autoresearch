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
  (gradio, plotly, pandas, numpy, zstandard, pyyaml). The kernel's `autoresearcher` env is not
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

Navigation: one node selector at the top of the page is shared by every node-level tab
(Node, Conversations, Code edits, Training data, Training, Eval, Cost); the Eval tab adds a
second node selector for its side-by-side view. Tabs whose content is per attempt (Node,
Conversations, Code edits, Training) add an attempt selector listing that node's attempts,
defaulting to the last. The root is a baseline, not an edited child: it has no parent,
edit, recipe, rationale or training, so those sections say "baseline — not applicable"
and only its eval, lineage and events are shown.

1. **Overview** (auto-refresh): loop state (running/stopped, from `control/`), current node,
   phase and attempt; tokens, spend and call count; alerts with timestamps; node table (id,
   parent, status, score, change vs parent, edit component, attempt counts, duration,
   error); score across nodes (chart); GPU memory and utilization over time (from
   `gpu.jsonl`); disk free; the last 50 events.
2. **Node** (pick a node): lineage to the root; edit summary and plan (`edit.json`);
   `rationale.md`; recipe and its difference from the parent's; the data commit; per-metric
   scores next to parent and root; per-phase timings; attempts with outcome and detail.
3. **Trace**: every event, paged (100 rows), filtered by node, phase, attempt, event type,
   component and free text. GPU samples (`gpu.jsonl`) are excluded unless the "GPU samples"
   filter is turned on, since they would otherwise be most rows; they are plotted on the
   Overview instead. Selecting a row shows the full detail: an LLM request as a chat
   (system, user, assistant, tool messages; tool definitions folded); an LLM response
   (reasoning, text, tool calls, usage, cost, latency); a kernel tool call's arguments and
   full result or error; a subprocess's command, env, cwd and full stdout/stderr; a sandbox
   run's stdout/stderr and stats. The raw event JSON and raw payload JSON are always shown.
4. **Conversations**: per node, phase and attempt, the list of agent conversations, each
   labelled with its role (section 6), and each shown as one chat:
   system and user prompts, each turn's reasoning (folded), tool calls and tool outputs
   inline, tokens and cost per turn; compaction handled as in section 6. A request with no
   response is shown as "awaiting response" while the loop is alive and the attempt is
   running, else "no response recorded". Container paths in tool outputs map to host paths:
   `/workspace/...` → `nodes/<node>/attempts/<phase>-<k>/workspace/...` and `/agent/...` →
   `.../agent/...`; a `run_command` output that names a `tool_output/*.log` links to that
   host file.
5. **Code edits**: each node's diff against its parent's agent commit; every `edit_self`
   attempt's diff against its base (the kernel commits each attempt's tree to
   `refs/attempts/<node>/edit_self-<k>` whether or not it succeeded; an attempt with no
   commit — checkout or commit failed — says so with its error); the attempt's contract
   report when the contract check ran, else "no contract check: the attempt failed before
   it" with the failure; a file browser for any agent commit.
6. **Training data**: each data commit's manifest (datasets, clip counts, formats); a paged
   clip gallery — video, caption with timed segments, a top-down plot of the camera path
   from the pose, provenance, license, leakage-check result, warnings; every ingest call
   with each candidate's paths and its outcome (accepted clip id, or rejection reasons),
   taken from the `data_ingest` tool call (candidate list) and tool result (one result per
   candidate, in order). Each clip's leakage-check result comes from the `ingest.leakage`
   event whose `candidate.video` matches that candidate (every `ingest.*` event names its
   candidate since 2026-09-27; for older runs the panel pairs by order within the ingest
   call and labels the pairing "inferred"); the kept staging files per attempt (rejected candidates,
   generated rollouts and images, annotations) — often none, since accepted candidates
   move into the clip store and runs before 2026-09-27 deleted staging.
7. **Training** (per `improve_recipe` attempt): the loss curve parsed from that attempt's
   `train/train.log`; the resolved train config; gate results (pass/fail with failures);
   duration; GPU use during training. An attempt that never reached training says why.
8. **Eval**: a node's rendered videos (40 on the proxy set), each with its case prompt and
   per-case metric scores; a side-by-side of two nodes on the same case; aggregate and
   strata tables (`aggregates.json`).
9. **Selection**: every `selection_events` row — the chosen parent and each candidate's
   value, penalty and probability.
10. **Cost and LLM stats**: tokens, cost, latency and errors per node, phase and role.
11. **Files**: a read-only browser of the whole run folder, for anything the other tabs do
    not cover (vLLM logs, render logs, WBench output, config snapshots). Every file shows
    its size and mtime. Text and JSON under 2 MB are shown in full; larger text shows its
    first and last 256 KB with a note; video and images are served by path (streamed, never
    read into panel memory); `.npz` shows array names, shapes and dtypes; other binaries
    (checkpoints, `.pt`, `.db`) show metadata only.

Gradio limits and their handling: long chats use `gr.Chatbot` message metadata to fold
reasoning and tool calls; diffs are shown as colored unified diffs; charts use plotly; long
lists are paged; videos load only when opened, one at a time, so the share tunnel carries
only what is watched. A share link lasts a limited time (the installed Gradio version's
limit is printed at startup); restarting the panel prints a new link.

## 6. Conversations across compaction

What the records contain (gateway linking is by resent-history prefix):

1. The summarizer request is the agent's full history plus one appended instruction
   message; the gateway links it into the same conversation as the next turn. A forced
   retry (tools disabled) is one more turn.
2. The agent then replaces its history with one user message: a continuation preamble
   containing the summary.
3. That next request no longer shares the old prefix, so the gateway starts a new
   `conversation_id` at turn 0. The records carry no explicit link.

Roles (user decision 2026-09-27: panel inference only, no record change): the gateway
records no role, and "role" is only the seed agent's convention, which self-edits may
change. A conversation's role label is inferred from its first request's system prompt:
the name of the prompt file, in the agent code that ran (the `code_commit` on the
attempt's `phase.start` event), whose text the system prompt starts with (e.g.
`data_builder`; appended knowledge documents do not prevent the match). If no file
matches, the label is the system prompt's first line; with no system prompt it is
"(no system prompt)". Every label is shown as inferred, and the Cost tab's per-role
figures group by these labels. This reads one payload per conversation, once, and the
result is cached in memory.

Stitching (panel data layer only; records untouched): conversation B continues
conversation A when both belong to the same node, phase and attempt, B's first call
starts after A's last call, and B's first user message contains A's last response text,
compared after stripping surrounding whitespace (the harness pastes `reply.text.strip()`).
A summary is hundreds of characters of model-written text, so two unrelated conversations
in the same attempt cannot match by accident; when several A match (not expected), the
latest one before B is used. A's last
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
response (when recorded), not by concatenating every request (each request resends the
full history).

## 7. Live-run data handling

- `telemetry/events/` is rescanned on every refresh, since a node's event file appears
  when the node starts. Each file is read incrementally by byte offset; a trailing line
  without a newline is left for the next read. Only event records are kept in memory;
  payloads are loaded on demand with an in-memory LRU cache.
- Gradio serves requests on several threads (the timer, other tabs, other browser tabs),
  so the data layer's shared state (offsets, event lists, caches) is guarded by one lock.
- Loop state: the loop is alive only if `control/loop.pid` names a live process whose
  start time (`/proc/<pid>/stat`) matches the recorded one — the kernel's own
  `Control.alive_pid` rule, reimplemented in the panel. When it is not alive, the Overview
  shows "stopped" and labels `control/state.json` as the last recorded state, never as
  the current one.
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
  and eval) through the public share link with login, and confirm in a browser that an
  eval video plays and seeks, a large log opens, and every tab loads, before using it on
  a new run.

## 10. Out of scope

Controlling a run (stop, resume), editing anything, multi-run views, per-person accounts,
changing what the kernel records.
