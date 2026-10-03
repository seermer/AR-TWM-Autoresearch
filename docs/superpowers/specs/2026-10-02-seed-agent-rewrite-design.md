# Seed agent rewrite and benchmark isolation: design

Date: 2026-10-02. Status: draft for review. Evidence: run `live-10-01` (n1-n3).

## Problems this answers

1. Every self-edit changed only prompts and knowledge files; no code changed in three nodes.
2. Self-edits wrote node ids, scores and findings into role prompts.
3. `edit_self` did data research (metric analysis, dataset hunting) instead of improving the agent.
4. Engineers never reported back to planners: five phases, one plan each.
5. The benchmark was named in the seed, downloaded by n2, and its wording copied into training captions.

## Principles

- **Two independent missions.** `improve_recipe` improves the model through data. `edit_self` improves the
  agent system and is blind to scores; parent selection alone decides which agent edits survive.
- **A prompt is mission and behaviour.** Per-node facts arrive in the first message. Reference facts are
  kernel-owned skills, read-only.
- **Plans state intent.** The planner says what to test and why; the engineer decides how.
- **The benchmark has no name and no content.** Agents see what is weak (metrics, dimensions, groups),
  never how the evaluation words, assembles or judges anything.
- **The internet stays open.** Isolation is: no motive, refusal at kernel tools, detection afterwards.

## Part A: kernel, benchmark isolation

### A1. Aliases at the agent boundary
One table in `eval/score.py`, applied in memory when `context_bundle` builds a node entry. Nothing on
disk changes: `eval/aggregates.json` (read by the panel, hidden from agents), the DB and `run.json` keep
the real names.

| dimension | real | alias |
|---|---|---|
| quality | aesthetic_quality | frame_aesthetics |
| | imaging_quality | frame_clarity |
| | temporal_flickering | flicker_free |
| | dynamic_degree | motion_amount |
| | motion_smoothness | smooth_motion |
| | hpsv3_quality | human_preference |
| consistency | background_consistency | background_stability |
| | segment_continuity | no_hard_cuts |
| | perspective_consistency | subject_framing_stability |
| | subject_consistency | subject_stability |
| | geometric_consistency | geometry_stability |
| | photometric_consistency | appearance_stability |
| | spatial_consistency | revisit_match |
| | gated_spatial_consistency | revisit_match_strict |
| control (was interaction) | navigation_trajectory | camera_path_accuracy |
| | event_edit_adherence | follows_event_instruction |
| | subject_action_adherence | follows_subject_action |
| | perspective_switch_adherence | follows_viewpoint_change |
| description (was setting) | scene_adherence | scene_matches_description |
| | subject_adherence | subject_matches_description |
| physics (was physical) | visual_plausibility | looks_plausible |
| | causal_fidelity | cause_and_effect |

Group axes: `interaction_type` becomes `instruction_kind` with values `camera_move`, `event`,
`subject_action`, `viewpoint_change`; `perspective` becomes `viewpoint`. Scene categories are unchanged.

`RecipeContext` gains `metric_guide`: per metric its dimension, weight and one line on what it measures.
Nothing about judge questions, frame sampling or prompt assembly.

### A2. Kernel strings
- `rollout_alayaworld`: description and item schema describe AlayaWorld's own interface (first frame,
  viewpoint, scene and character text, turns of a camera move plus an optional event, subject action or
  viewpoint change). No benchmark name, no "exactly as the eval does", turn keys renamed to match A1.
- Every other tool description, error and context string: no benchmark name, no "case".
- Every kernel tool result and error, and the context files, pass through one scrub that replaces a blocked
  name with `render`: in the kernel's own output the name only occurs inside file names (`run_render.py`) (worker log tails name the render script). Telemetry keeps the real text.
- A node that ended `eval_failed` or `crashed` shows a fixed kernel-side sentence as its error in
  contexts: the raw error can name the benchmark.

### A3. Refusals
- **HF tools.** A repo id containing a blocked substring (config `isolation.blocked_names`, default
  `["wbench"]`, case-insensitive) is absent from `hf_search` results, and `hf_list_files` /
  `hf_download` answer exactly as for a repo that does not exist.
- **Ingest.** A candidate is rejected when a frame matches a benchmark image (today's check), when its
  provenance names a blocked repo, or when its caption or a segment prompt shares a run of eight
  consecutive words (normalised: lower case, no punctuation) with any benchmark case text. A word run
  survives an added prefix, suffix or a split sentence. Short common phrases and paraphrases pass on
  purpose: only precise copying of a case is refused.
- **GPU tools.** The same text check runs on the prompts of `rollout_*` and `generate_images` items when
  the job is submitted, so a copied prompt fails before a GPU is scheduled.
- **One message**, with no ids or scores: `rejected: this clip is on the kernel's exclusion list and
  cannot be used as training data. Use a different clip.` (for a GPU tool item: `this prompt is on the
  kernel's exclusion list. Write a different one.`)
  Telemetry keeps the real reason.

### A4. Audit
After each agent phase, before training, the kernel scans everything the model wrote in that attempt
(reasoning, text, tool-call arguments) for `isolation.audit_patterns` (benchmark name, org, paper id).
Text the model did not write (tool results, user and system messages) is not audited: the gateway drops
every sentence that holds a pattern before the model sees it and before it is recorded, so a passive
mention neither reaches the model nor ends the node. A sentence is dropped whole, not reworded: no
substitute word reads sensibly in prose, a URL or a paper id, and the sentence is about the benchmark. A hit makes the
node `quarantined`: terminal, no retry, never a parent, not mounted under `/nodes`, absent from every
context, and its clips leave the clip pool. The panel shows the hit.

### A5. Held-out rule
The data roles' prompts say the evaluation set is held out: do not look for it or use its contents.
Building data aimed at the weak metrics is the job; reproducing the evaluation's own material is not.

## Part B: kernel, skills

Agents no longer own knowledge. `seed_agent/agent/knowledge/` is deleted.

- **Tool facts move into the tool.** Each kernel tool gets a complete description and a description per
  argument (also fixes the bare schemas seen in the run). `data_query.filter` becomes typed fields.
- **The rest are skills**, kernel-owned (`kernel/ar_kernel/skills/*.md`) and read through one kernel
  tool, `read_skill(name)`, offered to every role in both phases. The tool's description is the index:
  each skill's name and when to read it.

| today's knowledge file | goes to |
|---|---|
| captioning_with_caption_videos | `caption_videos` description |
| generating_clips_with_gpu_tools | rollout / generate / annotate descriptions |
| ingest_candidates_and_container | `data_ingest` description; container facts in skill `container` |
| recipe_constraints_checked_by_kernel | `recipe_check` description |
| camera_pose_files_cam_c2w, timed_prompt_segments, video_conversion_to_standard_format | skill `data_formats` (with today's `FORMAT_RULES`) |
| clip_quality_checks | skill `clip_quality` |
| failed_attempt_retries | skill `retries` |
| earlier_nodes_logs_transcripts_and_workspaces | skill `node_files` |
| reading_large_files | skill `large_files` |
| eval_prompts_and_turns | deleted; aggregation rules become `metric_guide` |
| eval_instruction_grades | deleted |
| writing_prompts_and_knowledge_files | deleted; its rule lives in the edit roles' prompts |

## Part C: kernel, contexts

Each phase's context holds only what its mission needs; `/context/context.json` is trimmed the same way.

- **Node entry for `improve_recipe`:** status, error, score, aliased metrics and aggregates, data stats,
  recipe, rationale. No edit summary, code diff or process digest.
- **Node entry for `edit_self`:** status, error, edit summary, code diff stats, process digest. No score,
  metrics, aggregates, rationale, recipe or data stats. The archive summary lists nodes and statuses only.
- **Process digest grows** to cover what the edit planner now reasons from: per role turns (no runtimes),
  compactions, kernel tool calls and errors per tool, local command failures, GPU jobs run and failed,
  ingest accepted and rejected with reasons, plans and reports per phase, failed
  attempts with their kind.

Known residue: `/nodes/<n>/transcripts` of data roles contain score tables, and the edit planner reads
those transcripts to study behaviour. Its mission statement carries this; nothing strips it.

## Part D: seed agent

Kept as is: `harness.py`, the local file and shell tools, `entry.py`. Rewritten: all prompts,
`orchestration.py`, `briefing.py`. The edit roles lose the arXiv tools.

### D1. Roles

| phase | role | mission | not its job |
|---|---|---|---|
| improve_recipe | planner | choose the one data idea this node tests, from what earlier nodes show | how to build it |
| improve_recipe | data engineer | build the data that tests the idea, and a recipe that fits it | changing the idea silently |
| edit_self | edit planner | find where the agent system wasted effort, lacked a capability or lost information, and decide the mechanism that removes it | data strategy, scores |
| edit_self | coder | implement and prove the mechanism | re-deciding it silently |

### D2. Prompt shape
Every prompt has the same four parts and nothing else: **Mission** (what the role is for and what it is
not for), **What you receive**, **How you work** (general behaviours), **Finish**. No metric names, node
ids, numbers, dataset names or tool usage notes.

The edit planner's "How you work" orders the levers: a missing or awkward capability (tools), then what
roles are told and in what order (orchestration, briefing), then the loop (harness), and prompt wording
last, only when a role's mission or behaviour is itself wrong. Both edit prompts state that a prompt
holds general behaviour only.

### D3. Plans
- `DataPlan`: `hypothesis`, `expected_change` (which metrics or groups, which direction), `data` (what
  kinds of clips, from which verified sources), `constraints`.
- `EditPlan`: `problem`, `evidence` (node and transcript locations), `mechanism`, `check` (what a later
  edit planner would observe in the process digest if it worked).
- Each text field has a safety cap of 4,000 characters: real intent-only fields ran up to 2,800, the ones that
  dictated file contents 5,800 and more.
- `EditResult.summary` is the coder's own short summary, not the plan text.

### D4. Talking back
- `request_replan(report)` is the only way back to the planner, described as the normal step when findings
  change what should be done. There is no question tool (decided 2026-10-02: too complicated).

### D5. Prompt check in the kernel contract
The kernel's contract check (not the agent's own `selftest`, which the agent can edit) refuses an edit
whose `agent/prompts/*.md` contain a node id (`\bn\d+\b`) or a metric alias. Section headers are not
checked: a self-edit may restructure a prompt.

## Part E: panel
"Tools offered" shows each tool's parameters as well as its description.

## Out of scope
- Restricting container network access.
- Compatibility with `live-10-01` or earlier runs: a new run starts from the new seed.

## Decided
- An audit hit ends the node with no retry.
- Plan fields are length-capped.
- No seed unit-test suite for the agent: the contract's dry run already exercises the wiring, and the
  run showed no coder reluctance (planners never asked for code changes).

## Amendments from run `live-10-02` (2026-10-03)

Two nodes showed that roles moved lists and results by typing them: 323,000 of 569,000 tool-argument
characters in one phase were hand-typed paths, clip ids and captions. Decided and built:

- **Lists come from files.** Every list argument of a kernel tool (paths, items, candidates, clip ids)
  is the list or the path of a file under `/workspace` holding it (`.json` array, or one item per line).
- **Results go to files.** A finished GPU job and `data_ingest` write the full result to
  `/workspace/staging/results/`; the caller gets the path and a summary (counts, first errors).
- **One refusal names every bad item**: the first few in the message, all of them in a file under
  `/workspace/staging/results/`, so a script can drop them and call again.
- **A job takes any number of items.** The per-job item cap is gone; the job timeout stays.
- **No GPU tool stays loaded after its job.** A warm caption server was built and removed the same day:
  the agent container now sees the run's GPUs, so the kernel holds them only while a job runs.
- **The planner reviews the engineer's result** in both phases: `accept_result` or a revised plan,
  within the plans-per-phase limit. Engineers had never reported back on their own.
- **The first message says what each folder is for** (`folders` in the context, kernel-owned).
- The context carries clip-pool counts, not clip records (`data_query` lists clips).
- The gateway retries a provider error sent with HTTP 200, at the top of the body or inside the choice.
- Kernel tools still read only `/workspace`; a file under `/nodes` is copied in first.
