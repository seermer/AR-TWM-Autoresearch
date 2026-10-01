# Review of 2026-10-01: verified findings, decisions, and tentative implementation plan

Status: **Stage 1 implemented on branch `review-fixes` in all three repos (uncommitted), under verification.** Next: user review, then Stage 2.

Stage 3 is implemented too (uncommitted). As built, where it differs from section 3:
- Fix 4: the first message keeps nodes as columns; the group table has one row per group and dimension ("Dimensions by group of test cases"), not one row per group and node.
- Fix 23: the harness ends through an extra node `tools_done` that runs once on the merged parallel tool results.
- Fix 12: `eval_prompts_and_turns.md` also states that the published video drops its last 7 frames (checked on rendered sidecars: 2 turns = 185 frames, 3 turns = 281).
- Fix 24 was done in Stage 2.

Stage 2 is implemented too (uncommitted). As built, where it differs from section 3:
- `ar score-node --run-id R --node N` takes no checkpoint or rank: the node must exist in the run and its recorded checkpoint and rank are used (`run.rescore_node`). Output directory is `paths.scores_dir` (`scores/`).
- Recording the root's case counts moved from `score_node` into the loop (`ensure_root`), so `score_node` writes nothing into `config/run.json`.
- `/nodes` mounts exclude `running` and `interrupted` nodes; siblings follow the same rule. The first message describes at most the last 10 siblings.
- The knowledge file `lineage_logs_transcripts_and_workspaces.md` is renamed `earlier_nodes_logs_transcripts_and_workspaces.md`.
- The contract check gets its contexts from `agent_phase.smoke_contexts`; the loop's contract phase is the wrapper `loop._contract`.
- Fix 24 (pandas + pyarrow in the agent image) was done here with the Dockerfile comment fix, so the base image is rebuilt once.
- Stage 1 end-to-end check: by user decision only the root is scored on the new 50 cases (run `stage1-check`); `n1` was verified by rendering four cases and inspecting frames.

Stage 1 as built, where it differs from section 3:
- `eval/merge.py` is replaced by `eval/lora.py` (`concat_eval_lora`); `score_node` takes the node rank only; `ar score-node --lora-alpha` is gone.
- The concatenated adapter is written to `nodes/<n>/eval/lora/` and deleted by the eval cleanup.
- `initial_expected_n` also counts the six judged metrics from the case files, so a judge call that never succeeded fails the root too (the root's own report used to define those counts).
- WBench: `causal_fidelity` had the same swallowing (a failed per-dimension call was averaged out); it raises now. `scene_adherence` and `subject_adherence` already left the case unscored.
- `WorldModel/docs/LOWCOMPUTE.md` section 6 now says the merge is lossy and shows the concatenation.

Working rules for the implementation (from the user):
- Stages, pause for review after each. Unit tests after each stage (`.envs/autoresearcher/bin/python -m pytest tests -q --ignore=tests/manual`, about 15 min, all passed on 2026-10-01 before any change).
- Work on a branch in each of the three repos (AutoResearcher, WorldModel, WBench). No commit or push until the user says. No Claude attribution in commits.
- Docs and comments are not trusted as evidence; verify against code or data.
- Noise is assumed not to exist: no repeat runs, no noise handling, any improvement counts.
- Only `ar run` and `ar stop` may modify a run. Every other command is read-only or works on a separate copy.
- Seed-agent rules (memory): simplicity first, narrow single-topic knowledge files, facts not steering, no per-case eval exposure, no backward compatibility, prefer exceptions.
- GPUs: default 0,1,2,3, minimum 4. On 2026-10-01 GPUs 0-3 showed ~23 GB used by an unseen process; 4 and 5 free.

---

## 1. Verified findings

Evidence sources: code, CPU measurements, and telemetry of the two earlier runs in
`WM-AutoResearch/.obsolete_runs/` (`acceptance_20260928`, `live-09-29`).

### 1.1 Eval merge loses a large part of each fine-tune (measured)
- `kernel/ar_kernel/eval/merge.py` runs `WorldModel/scripts/tools/merge_lora_for_rollout.py`, which does
  `(base_w.float() + delta).to(base_w.dtype)`; `weights/alaya-world-ar/transformer.pt` is bf16.
- Weight space (40 sampled modules): live-09-29 n1 (300 steps, r64): 59% of entries unchanged, relative error of merged delta 45%. acceptance n1 (100 steps, r32): 67% unchanged, error 68%.
- Output space (live n1, 24 modules): median error 28% on inputs in the LoRA's row space, 62% on random inputs.
- `WorldModel/tests/test_merge_lora.py` uses a float32 base, so it cannot catch this.
- Runtime LoRA (`alaya/model/lora.py`): hook computes `output + x @ A.T @ B.T`, **no alpha/rank scaling at runtime** (scaling is only folded into A at init). The merge script multiplies by alpha/rank, which only agrees because the allowlist has rank == alpha.
- DMD student LoRA (`weights/alaya-world-dmd/lora.safetensors`): rank 256, 480 modules, bf16. Node LoRA: same 480 modules. Concatenating A rows and B columns (rank 256 + r) reproduces the sum with error 0.00.
- `load()` in `lora.py` requires shapes to match `cfg.lora.rank` (built in `alaya/model/loader.py:66`), and reads `dmd_resume/lora.safetensors` (`alaya/checkpoint.py:162`).
- `history_encoder.pt` is identical in alaya-world-ar and alaya-world-dmd.

### 1.2 Score behaviour in the earlier runs
- Stored node scores reproduce exactly from each report (5 decimals).
- live-09-29: root 0.6761, n1 0.6793, n2 0.6806, n3 0.6847. Paired bootstrap over cases, node vs root: +0.003 [-0.015, +0.026], +0.004 [-0.015, +0.028], +0.008 [-0.007, +0.024].
- Per-metric effects are real even when the mean is flat: `navigation_consistency` n1 -0.093 [-0.153, -0.032], n2 -0.070 [-0.131, -0.013].
- Perspective-switch turns passed: root 0/11, n1 3/11, n2 3/11, n3 2/11. All three nodes trained on the same 16 perspective-switch clips (n2 and n3 relabelled that dataset `navigation`). So a fine-tune does show through the 4-step student.
- Eval sampling: 4 steps, `cfg_scale: 1.0` in config but `_validation_cfg_scale()` maps <= 1.0 to 3.0.
- Judge: local path uses temperature 0.1 (`WBench/src/metrics/vlm/vlm_evaluator.py`); user chose to leave it.

### 1.3 Root grades (local judge, 50 cases) and what they mean
- event_edit_adherence 0.27 (12 cases), subject_action_adherence 0.25 (14), perspective_switch_adherence 0.00 (6), causal_fidelity 0.39 (8).
- Question level: event edit "event visible" 6/31 turns, "completed" 0/31; subject action 3/32 and 1/32.
- perspective_switch is strict per turn: pass only if Q1 and Q3 and Q4 (`WBench/src/metrics/interaction/vlm_interaction.py`).
- Strata today (`eval/score.py aggregates()`): per case, mean of all applicable metrics; perspective_switch stratum = 0.78 while its adherence is 0.0.

### 1.4 How eval builds prompts (from `WorldModel/alaya/data/wbench.py`, checked on a rendered sidecar)
- Turn = 3 rounds = 96 frames = 4 s. Round = 32 frames. Clip prefix = 25 frames.
- Turn prompt = character_prompt + environment_prompt + every event_edit/subject_action text so far, appended (e.g. " Climb." then " Climb. Descend."), + the current perspective clause (a state, not accumulated). `<camera>` text is stripped; navigation is driven by poses (0.16 forward units and 6 degrees per latent frame, `alaya/config/schema.py:491`).
- Interaction turns reuse the previous navigation action; cases that never navigate use `W`.
- Judge per event/subject-action turn: 5 yes/no questions (unchanged-scene?, event visible?, reached a conclusion by the end of the segment?, key details right?, unrelated anomaly?). Score = correct/5.
- Perspective switch instructions in the full set: tp_to_fp 16, fp_to_tp 16, tp_to_tp 9, fp_to_fp 8, fp_to_scope 6, scope_to_fp 6; 10 mention "cut". Judge wording accepts camera movement or a cinematic cut.

### 1.5 Benchmark composition
- 289 cases: navigation 158, subject_action 76, event_edit 65, perspective_switch 31. Causal fidelity graded on 50 cases, 48 of them navigation-only.
- Current proxy 50: navigation 30, subject_action 14, event_edit 12, perspective_switch 6, causal 8, 179 turns.
- WBench prompt cache (`WorldModel/cache/text_embed_wbench`) has 747 files = all 289 cases.

### 1.6 Bugs reproduced or read in code
- `ar score-node` on an unknown id creates a parentless scored node; it became a selection candidate with P = 0.68 and `agent_commit None`; `repo.checkout(None)` raises TypeError. `--node root` rewrites `expected_n`. (`cli.py:103-112`)
- Submit does not end a role: in 15 of 16 real conversations the model was called again after "submitted". (`seed_agent/agent/harness.py`, edge tools -> model)
- Contract smoke: `_smoke_context` is empty and the seed's dry run returns before `briefing`/`plan_and_engineer`; mock model only answers "ok". (`contract/verify.py:138`, `orchestration.py:211,279`)
- `Recorder.read_events` raises on a torn line; `process_digest` then returns `{}` and `write_transcripts` writes nothing.
- README line 112 says default context window 128000; `kernel.yaml:88` is 1,000,000 with `compact_at: 0.6`.
- `data_query` returns at most 500 clips with no offset; `clip_pool_summary` keeps the last 2000 and the briefing reports that as the pool size.
- WBench: a failed judge call is counted as a wrong answer (`_execute_binary_tasks`: `total += 1`; perspective switch: None -> not ok). None occurred in existing runs.
- Agent image has no pandas/pyarrow although `knowledge/reading_large_files.md` recommends pandas.
- `ar status` goes through `attach_run` -> `open_db`, which runs the schema script and enables WAL (writes in the run). The panel already has a non-writing opener (`panel/runfiles.py:_db_uri`).

### 1.7 Agent-facing observations
- Lineage-only view: `lineage()` walks ancestors, `/lineage/<id>` mounts ancestors, briefing lists top 5 scored nodes.
- Earlier runs: `edit_self` about 50 s (13-16 turns); `improve_recipe` 36-97 min; training 2.2-2.8 h; render ~70 min; WBench ~58 min. Commits held 16-52 clips, 100-300 steps.
- live n1 and n3 trained on AlayaWorld's own renders; n3's self-edit told the planner to prefer the kernel render path over real footage.
- Edits: 5 of 7 were prompt rules; acceptance n2 added a rule that n3 reversed.
- 121 = 25 + 3x32, so a 121-frame LTX/Wan clip ends on a round boundary; a 97-frame image-to-video continuation from its last frame (first frame dropped = 96 frames) puts a prompt change on that boundary.

### 1.8 Not verified
- True repeat noise of the score (by decision: not measured).
- Cause of the navigation_consistency drop (Fix 21 investigates).
- Eval VRAM headroom for a rank-320 LoRA (see Fix 1 risk).
- Parts of the tree not read: `tools/vllm_server.py`, `doctor.py`, generator bridges, most of the panel, most tests; WorldModel trainer and most WBench metrics.

---

## 2. Decisions (interview of 2026-10-01)

| # | Decision |
|---|---|
| 1 | Eval loads one exact concatenated LoRA (DMD 256 + node rank) via `dmd_resume` on the untouched base. No merge. |
| 2 | No noise measurement, no noise handling. |
| 3 | Judge temperature left as is. |
| 4 | Strata table: five numbers (the 5 dimensions) per group of test cases. |
| 5 | No change columns. Leave raw grades as is. |
| 6 | Score = weighted average. `event_edit_adherence`, `subject_action_adherence`, `perspective_switch_adherence`, `causal_fidelity` together = half the score; the other 18 = half. Weights in `configs/kernel.yaml`, frozen per run. |
| 7 | Proxy size configurable (`eval.proxy_size`, default 50) = first N of a predefined ordered list. Every prefix keeps the four interaction types roughly equal; navigation picks causal-fidelity cases first. |
| 8 | Edit check runs on the node's real context and builds the first message for both phases. |
| 9 | Dropped. |
| 10 | Only `ar run`/`ar stop` modify a run. `score-node` writes to a separate directory; `status` read-only; `init-run` removed. |
| 11 | Failed judge call leaves the case unscored; kernel re-runs the judge phase once, then fails the eval. |
| 12 | Two narrow knowledge files: eval prompt/turn mechanics, and what each instruction-following grade asks. |
| 13 | The score weights are stated in the eval knowledge file of #12 (plus a unit test tying them to `kernel.yaml`). Per-metric grades stay in the first message and `context.json` as today. |
| 14 | All finished nodes mounted read-only (eval hidden). First message: direct ancestors (as now) + direct siblings (same parent). Not all nodes in context. |
| 15 | Cuts stay listed as a defect; add the pose fact. No mention of viewpoint changes. |
| 16 | Both: tool description states the clips come from the model being trained; data summaries list clip sources. |
| 17 | Knowledge fact only (frame arithmetic for boundary-aligned continuation). |
| 18 | Edit cadence left as is. |
| 19 | All per-job item limits raised to 100. |
| 20 | No mid-training signal. |
| 21 | Investigate pose scale offline first; report only. |
| 22 | Perspective-switch prompt text: skip. |
| 23 | An accepted submit ends the role. |
| 24 | Add pandas + pyarrow to the agent image. |

Small fixes to apply without asking: torn-line tolerance in `read_events`; README context-window line; `data_query` offset and true pool size in the briefing; stale "no network at runtime" Dockerfile comment.

---

## 3. Tentative implementation plan

### Stage 1: eval correctness (Fixes 1, 6, 7, 11)

**Fix 1: concatenated LoRA**
- WorldModel: new `scripts/tools/concat_loras.py --loras A/lora.safetensors B/lora.safetensors --output DIR`: for each module, `A = cat(A1, A2)` rows, `B = cat(B1, B2)` columns, dtype kept; fail on differing module sets; write `DIR/lora.safetensors`. Test in `WorldModel/tests/` with a **bf16** base check that `B@A` equals the sum.
- Kernel `eval/merge.py`: replace `merge_lora` with a call to the concat script (DMD LoRA + node LoRA) into `nodes/<n>/eval/lora/`; drop the merge slot, the disk check, `wait_for_ram` if nothing else uses it.
- Kernel `eval/render.py build_render_config`: `paths.resume_checkpoint` = released base; `paths.dmd_resume` = concat dir (root: the DMD dir); `lora.rank` = `lora.alpha` = 256 + node rank; `paths.history_encoder` = node checkpoint's (unchanged).
- Clean up: `run.py score_node` cleanup of `merge_slot`, `cli.py` resume removal of `merge_slot`, `disk.merge_min_free_gb`, `eval.free_ram_before_render_gb`, `eval.ram_wait_alert_min`, README/spec mentions, tests (`test_wbench.py`, `test_merge_guard.py`, `test_run_bootstrap.py`).
- **Risk:** LoRA tensors are not FSDP-sharded, so rank 320 adds ~0.65 GB per GPU at eval. Needs one long-case render on 4 GPUs to confirm no OOM. If it does not fit, fall back to asking the user (options: lower `vigeo_cache_budget`, or a float32-accurate merge is not possible at bf16 runtime).

**Fix 6: weighted score**
- `kernel.yaml`: `eval.score_weights` (metric -> weight, default 1). Half-the-score rule: the 4 grades get weight 4.5 each (4 x 4.5 = 18 = the other 18 x 1).
- `eval/score.py score_from_report(report, metric_set, expected_n, weights)`: weighted mean. Callers: `run.py score_node`.
- Root cache (`loop.py save_root/_reuse_root`): keep storing metrics; recompute the score from metrics with the run's weights on reuse.
- Update `tests/test_score.py`, panel if it recomputes (it reads the stored score), README root-score sentence.

**Fix 7: ordered proxy list**
- `scripts/make_proxy_cases.py`: rewrite to emit the full ordered list. Prototype (seed 20261001): repeatedly pick the interaction type with the fewest cases so far; within it prefer single-type cases, and for navigation prefer causal-fidelity cases; shuffle first for ties. Result: first 50 = 13 nav / 13 event / 12 subject / 12 viewpoint, 13 causal, 158 turns; prefixes 20, 80, 120 balanced.
- `configs/proxy_cases.txt` holds the ordered 289 ids; `kernel.yaml` gains `eval.proxy_size: 50`; `run.py bootstrap_run` takes the first N. `root_key` already includes `case_ids`.
- Update `tests/test_make_proxy_cases.py`, `test_run_bootstrap.py`, spec 11.1, README.
- Consequence: cached root no longer applies; first run re-scores the root.

**Fix 11: judge failures**
- WBench `vlm_interaction.py`: in `_execute_binary_tasks` and `evaluate_perspective_switch`, a failed call (answer None) raises, so `_vlm_eval_case` stores `score: None` with the error (it already does that on exceptions and skips valid scores on a re-run). Check `evaluate_scene_adherence`, `evaluate_subject_adherence`, `evaluate_causal_fidelity` in `src/evaluate.py` for the same swallowing.
- Kernel `eval/wbench.py`: run the `vlm` phase twice (like `gpu`), inside the same judge-server session for the local judge. A still-missing score then fails through the existing `expected_n` check.

Stage 1 verification: unit tests; then (needs 4 GPUs) root + one old checkpoint through the new eval path.

### Stage 2: kernel loop, CLI, context (Fixes 8, 10, 14, 16, 19, small fixes)

**Fix 8:** `loop._edit` builds the edit and recipe contexts (`build_edit_context`, `build_recipe_context` with `dry_run=True`) and passes them to `verify_contract`, replacing `_smoke_context`. Seed dry runs build `brief(edit_context(...))` and `brief(recipe_context(...))` before `ping()` (Stage 3 touches the seed side; do both together). Update `tests/test_contract_verify.py` and fixtures.

**Fix 10:**
- `cli.py`: remove `init-run`. `score-node`: output under a separate directory (proposal: `AutoResearcher/scores/<run>-<node>-<timestamp>/`, gitignored) with its own Recorder; reads case ids, metric set, judge, expected_n, weights from the run without writing; never touches `archive.db` or `run.json`.
- `status.py`/`cli.py`: open the archive without writing, reusing the panel's `_db_uri` approach (move it to the kernel or import it); do not go through `attach_run`.
- Update `tests/test_cli_run.py`, `test_status.py`, README.

**Fix 14:**
- `sandbox/runner.py Mounts`: mount every finished node dir at `/nodes/<id>` read-only with the eval tmpfs mask (rename from `/lineage`). `agent_phase.lineage_dirs` -> all nodes with a directory, excluding the node being built and `running` ones.
- `context_bundle.py`: add `siblings` (other finished children of the same parent) with the same fields as a lineage entry; keep `archive` as one line per node.
- Seed `briefing.py`: a short section per sibling; not added as table columns. Update prompts/knowledge that name `/lineage`.

**Fix 16:** `rollouts.py AlayaWorldBackend.description`: one sentence that the clips are generated by the model being trained. `dataset_stats`/`_data_stats`: per dataset, counts by source (provenance kind + generator/repo, following `derived_from` where recorded; otherwise `derived`).

**Fix 19:** `kernel.yaml` `max_items: 100` for alayaworld, wan22, ltx25, annotate, images; ltx25 `timeout_s` 10800 -> 28800; update description strings ("at most 64 items") and `knowledge/generating_clips_with_gpu_tools.md`.

**Small fixes:** `Recorder.read_events` skips torn lines; README line 112; `data_query` `offset` + briefing uses the true clip count; Dockerfile comment.

### Stage 3: seed agent (Fixes 4, 12, 13, 15, 17, 23, 24)

**Fix 4:** `eval/score.py aggregates()`: strata become `{axis: {group: {dimension: mean}}}`, each cell = mean over that group's cases of the dimension's metrics (mean of metric means). Seed `briefing.py` renders one row per group and node. Update `tests/test_score.py`, `test_seed_agent.py`, panel if it shows strata.

**Fix 12 + 13:** new `knowledge/eval_prompts_and_turns.md` (section 1.4 facts + the score weights) and `knowledge/eval_instruction_grades.md` (what the judge asks per grade). New unit test: the weights named in the knowledge file match `kernel.yaml`.

**Fix 15:** `knowledge/clip_quality_checks.md`: cut line reworded (how to detect; poses estimated across a cut are not valid; one caption cannot describe both shots).

**Fix 17:** `knowledge/timed_prompt_segments.md`: which frame counts end on a round boundary (25 + 32k: 57, 89, 121, ...), and that image-to-video from a clip's last frame continues it.

**Fix 23:** `harness.py`: a tool flagged `return_direct` that returns successfully ends the graph (create_agent has the same notion; ours ends only on success, so a rejected submit still returns to the model). `tools.py submit_tool` sets `return_direct=True`. Update the harness docstring's list of differences and `tests/test_seed_harness.py`.

**Fix 24:** `docker/agent.Dockerfile`: add pandas and pyarrow; update the comment in `seed_agent/agent/requirements.txt`.

### Stage 4: Fix 21, offline pose-scale comparison (no code change)
- From `.obsolete_runs/*/store/blobs/pose` and the `commanded_camera` files in staging: per-frame translation magnitude of annotate_camera poses vs the eval's commanded step (0.16 per latent = 0.02 per frame at stride 8) vs the example dataset's poses. Report numbers; decide on a fix afterwards.

---

## 4. Open items, answered by the user on 2026-10-01
- GPUs 0-3 are free for the whole fix effort (pass the device list explicitly). The real LLM API in `.env` may be used as needed.
- `score-node` output lives in `AutoResearcher/scores/<run>-<node>-<timestamp>/` (gitignored).
- Update the design spec (`docs/superpowers/specs/2026-09-17-autoresearcher-design.md`) alongside each changed behaviour.

---

## 5. Stage 4 report: pose scale, measured offline on 2026-10-01 (no code change)

Question (Fix 21): are the poses agents attach to training clips on a different scale from the camera path the eval commands, and could that explain the `navigation_consistency` drop?

How poses are used (read in code, not run):
- Neither training nor eval feeds poses to the model as an action input: `control.candidates` is `[[]]` in `configs/base_recipe.yaml` and in `WorldModel/configs/wbench_full.yaml`. Poses act only through the spatial memory (`spatial_memory.context_mode: vigeo_prefix_last_frame`), which warps earlier frames to the target camera using ViGeo geometry.
- Training (`rollout_trainer.py` near line 3473): ViGeo estimates its own poses on the clip's history frames, and `estimate_translation_scale` takes the median ratio of dataset distances to ViGeo distances. So the absolute unit of a dataset's translations is aligned to ViGeo per clip.
- Eval (source `wbench_navi`): the scale is fixed at `vigeo_single_frame_scale: 1.0`, so the commanded path is taken directly in ViGeo units.
- `annotate_camera` is ViGeo too, so its poses are already in the unit the eval's command is read in, and the two can be compared directly.

Measured, per frame (median over a clip, then percentiles 10 / 50 / 90 over clips):

| Poses | Clips | Translation per frame | Rotation, degrees per frame |
|---|---|---|---|
| Eval command (0.16 and 6 degrees per 8 frames) | - | 0.0200 | 0.75 when turning |
| `commanded_camera` files of the AlayaWorld tool | 56 | 0.0200 / 0.0200 / 0.0200 | 0 (all forward) |
| live-09-29, AlayaWorld renders, ViGeo poses | 24 | 0.0194 / 0.0262 / 0.0355 | 0.02 / 0.09 / 0.28 |
| live-09-29, source not recorded (900-frame clips) | 19 | 0.0068 / 0.0115 / 0.0316 | 0.14 / 0.42 / 0.90 |
| acceptance, Wan 2.2 renders, ViGeo poses | 10 | 0.0241 / 0.0291 / 0.0587 | 0.04 / 0.08 / 0.20 |
| acceptance, HF clips (TartanAir-derived by their notes) | 41 | 0.0470 / 0.0696 / 0.1644 | 0.61 / 1.22 / 2.80 |
| acceptance, HF dataset clips | 23 | 0.0075 / 0.0081 / 0.0086 | 0.21 / 0.22 / 0.23 |
| Example `video_caption_camera` | 6 | 0.0034 / 0.0041 / 0.0132 | 0.02 / 0.02 / 0.03 |
| Example `video_timed_prompts_camera` | 6 | 0.0057 / 0.0092 / 0.0185 | 0.01 / 0.02 / 0.14 |

Per frame is at each clip's own frame rate; some HF clips are 30 fps and are subsampled to 24 for training, which raises their step by a quarter. The example datasets' poses are not ViGeo's, so their translation unit is not comparable (training rescales it).

Reading:
- No unit mismatch. ViGeo measures AlayaWorld's own forward renders at a median 0.026 per frame against the commanded 0.020, about 1.3 times, and every one of the 56 commanded paths is exactly 0.020.
- What does differ is how fast the camera moves in the training clips: from about 0.4 times the eval's step (one HF set) to about 3.5 times (the TartanAir-derived set, which also rotates 1.6 times faster than an eval turn).
- All 56 commanded paths in these runs are forward only: the agents never rendered a turning or strafing case, so none of the self-rendered training clips showed the model a rotation at the eval's rate.
- The 24 AlayaWorld clips of live-09-29 are all two renders joined into one 121-frame clip (57 frames first-person, then 64 third-person). In 21 of the 24 the largest pose step is at the join, frame 56 to 57; in the other three it is at frame 63 or 95. Its size is 0.26 to 5.3 units, a median 50 times the clip's normal per-frame step (13 to 260 times the eval's 0.020), with rotations at that frame up to 111 degrees. These clips were the training data of n1, and n2 and n3 reused them.
- So the training clips of the nodes whose `navigation_consistency` fell contained a camera jump of about one scene unit in a single frame, which the spatial memory then used to warp earlier frames. This is a plausible cause of the drop and a more direct one than scale. It is not proven: no node was trained on the same clips without the joins.

Nothing to change in code from this. The knowledge file now states that poses estimated across a cut are not valid (Fix 15). A stricter option, not done and for the user to decide: have the kernel's ingest reject or warn on a clip whose pose has a single-frame step many times its median.


---

## 6. Prompt render and truncation check (2026-10-01)

Every prompt the seed agent sends was rendered (four system prompts with the knowledge index, the first message of each role, the five retry variants, the hand-off, reminder, replan and compaction texts, all tool descriptions) on the two earlier runs, on a scaled-up archive (30 ancestors, 25 siblings, 300 nodes) and on a worst case with every node at every cap. Script: scratchpad `render_prompts.py` (not in the repo).

Caps that real data hit often, now raised (values and evidence in the spec table under "Size caps are fail-safes"): per-node free text 1,200 -> 4,000; tool-error example 160 -> 1,000 and all errors shown (was 3); digest 2,000 -> 10,000 bytes; first message 150,000 -> 400,000; tool result 100,000 -> 200,000; training log tail 4,000 -> 20,000; failure tails 2,000-3,000 -> 6,000.

Changed because of the 100-item job limit (Fix 19): job results no longer echo `args`; `data_query` returns pages of 100 instead of 500.

Other changes from the read-through: every sibling is listed in a one-line table (detail stays on the last 10); the planner prompt no longer says "50-case"; an exhausted-retries node error no longer embeds the log tail.

Left as designed: ancestors and siblings described are the last 10 each; `read_file` (100,000 chars) and `run_command` (20,000 chars) caps were never reached in 251 real calls (largest 12,037 and 8,126).
