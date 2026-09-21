# Kernel foundations — real-hardware verification log

Task 14 of `2026-09-17-autoresearcher-kernel-foundations`. Everything below was run
on the actual box against the actual repos; nothing here is inferred from unit tests.

**Why this document exists.** The unit suite was green at 79/79 while five real
defects sat in the kernel and the two upstream repos. Every one of them was
invisible to tests that used synthetic fixtures and temporary directories, and
one destroyed real data before it was caught. This log records what was proven,
how, and what the misses imply for the loop.

---

## 1. Environment

| Component | Value |
|---|---|
| GPUs | 5 x 24 GB; training on `0,1,2,3`, eval/render on `0,1,2,3,4` |
| Conda envs | `alayaworld` (train/render/data), `wbench-main`, `wbench-vp`, `autoresearcher` (kernel) |
| Base checkpoint | `weights/alaya-world-ar` (released stage2b) |
| Disk | `/mnt/biometrics`, 2.1 TB free at time of run |
| torch | 2.7.1+cu128, NCCL 2.26.2 |

Documented prerequisite, confirmed twice: WorldModel scripts driven from outside
the repo need `PYTHONPATH=<worldmodel>`, because a script's own directory — not
the caller's cwd — goes on `sys.path[0]`.

---

## 2. Commits produced by this task

| Repo | Commit | What |
|---|---|---|
| AutoResearcher | `264df51` | `BlobStore.put` survives cross-filesystem candidates |
| AutoResearcher | `bbfb38b` | ingest refuses candidates outside the run directory |
| AutoResearcher | `b9c816f` | `classify_failure` no longer fails every successful run |
| AutoResearcher | `97653bd` | shell-out failures include stderr |
| WorldModel | `96c68fd` | launcher takes a GPU count, not a GPU-5 rejection (U2) |
| WorldModel | `2d99676` | `init_process_group(device_id=)` binds rank to GPU |
| WorldModel | `dc577e9` | `run_wbench` accepts a generated config outside the repo |
| WBench | `606f0b8` | visual plausibility takes `--work_dir` (U1) |

All pushed to the project forks (`seermer/{AR,AlayaWorld,WBench}-TWM-Autoresearch`).

Suites at time of writing: AutoResearcher 83 passed, WorldModel 77 passed.

---

## 3. INCIDENT: real training data destroyed by the verification test

**This is the most important entry in this document.** It is recorded in full
because the failure was caused by the verification work itself, and because the
recovery was partial.

### What happened

The Task 14 brief had the manual ingest test pass the real example datasets —
`WorldModel/data/examples/{video_caption_camera,video_timed_prompts_camera,video_caption_static}`
— directly into `Ingestor.ingest()`. `ingest()` **moves** its input into the
content-addressed blob store; consuming the source is its designed behaviour, and
`tests/test_blobs.py` asserts exactly that.

The test had been "passing" only because `BlobStore.put` used `os.replace`, which
raises `OSError` across filesystems. That error was masking the data loss. Fixing
it (`264df51`, an approved and correct fix) unmasked the real behaviour, and the
first successful run emptied all three example datasets: 18 clips, 48 files.

`data/examples` is git-ignored, so git could not restore it, and no script
rebuilds it. The upstream sources *are* documented in `docs/TRAINING.md:187-189`
— SpatialVID-HQ (`video_caption_camera`), C3VD colonoscopy
(`video_timed_prompts_camera`), and CDnet 2012 highway/office/peopleInShade
(`video_caption_static`) — and the CDnet source frames still exist under
`data/examples/_downloads/` (9,914 files, 105 MB). So the static set could in
principle have been rebuilt from local material; the other two would have
required re-downloading from their upstream datasets, and in every case the
exact clip selection, trims and captions would have had to be reproduced by
hand. Recovery from the blob store was both faster and exact.

### Recovery

The only surviving copy was the blob store written by the pytest run, inside a
temporary directory pytest prunes on a schedule. It was preserved to
`AutoResearcher/.recovery/` (104 MB, 40 blobs, 18 clip rows) before that could
happen. With the user's explicit authorization — the subagent could not authorize
writes outside the project, and correctly did not try — 48 files were restored,
each verified by sha256 against its blob record.

Verified after restore, using WorldModel's own checker rather than our own:

```
check_dataset.py --config configs/examples/finetune_video_caption_camera.yaml
  -> 6 clips, 1.4 min, 6 usable clips, 0 errors, 0 warnings
check_dataset.py --config configs/examples/finetune_video_timed_prompts_camera.yaml
  -> 6 clips, 0.8 min, 30 round-aligned windows, 0 errors, 0 warnings
check_dataset.py --config configs/examples/finetune_video_caption_static.yaml
  -> 6 clips, 1.0 min, 6 usable clips, 0 errors, 0 warnings
```

These match the figures in `docs/TRAINING.md` exactly.

### What was NOT recovered

**The original filenames are gone.** They were not stored in the clip records or
in telemetry, and the caption JSONs carry only content descriptions, no
provenance — checked directly, not assumed. The files now carry synthetic ids
`clip_0001..0006` per dataset. The content is byte-identical and remains
scene-identifiable from the captions (e.g. the office scene is plainly the CDnet
`office` sequence), so the datasets are fully usable; what is lost is the naming,
and anything that referenced those clips by filename must be re-derived.

### What prevents recurrence

Two independent guards, both required, on the principle that the test being
correct should not be the only thing standing between a bug and real data:

1. **The test no longer hands `ingest()` a real path.** `tests/manual/test_real_pipeline.py::_ingest_examples`
   copies each clip into `<run_dir>/staging/<dataset>/<stem>/` and ingests the copies.
2. **The kernel refuses the class of call outright.** `Ingestor._outside_run_dir()`
   rejects any candidate whose video/caption/pose resolves outside the run
   directory, evaluated before `probe_video` and before any blob write.
   Covered by `tests/test_ingest.py::test_candidate_outside_run_dir_is_rejected`,
   which asserts both that the candidate is rejected **and that the source files
   still exist afterwards** — i.e. that the guard fires before any move.

`BlobStore.put`'s move semantics are deliberately unchanged. The trade accepted:
a legitimate caller passing an absolute path outside the run tree now gets a
rejection instead of silent consumption.

### Proof the guard works

Step 3 was re-run after the fix (PASSED, 119 s). All 48 files were still present
afterwards, and all three datasets still pass `check_dataset.py` with 0 errors /
0 warnings.

---

## 4. Defects found only on real hardware

All five were invisible to a green unit suite.

### 4.1 `BlobStore.put` could not cross filesystems — `264df51`

`os.replace` raises `OSError` across devices. Tests used a single tmpdir, so the
ingest path was never exercised across the real filesystem boundary. Replaced
with `shutil.move`. Fixing this is what unmasked §3.

### 4.2 Every successful training run was classified as a failure — `b9c816f`

`INFRA_SIGNATURES` contained the bare substring `"NCCL"`. A **healthy** 4-GPU run
emits `NCCL version 2.26.2+cuda12.2` and four W-level `ProcessGroupNCCL.cpp`
device-id warnings. So the happy path was misclassified **100% of the time**.

Observed on a run that in fact succeeded completely: 2 steps, loss 0.4316 -> 0.2520,
18.28 GB peak allocated per rank, `checkpoint-2` written with a 624 MB
`lora.safetensors`. The test failed only on `assert 'infra' == 'none'`.

**Loop impact had this shipped:** every node marked infra-failed and discarded;
the archive never gains a scored node. It would have presented as "the search
finds no improvements" rather than as a string-matching bug — a silent, plausible,
and entirely wrong result.

Second defect in the same tuple: `"cache miss"` matches nothing a failing run
emits; it occurs only in WorldModel source comments. The real message is
`loader.py::_text_encoder_disabled`'s `"... a prompt missed the on-disk embedding
cache"`, so an incomplete precache would have read as a clean run.

Both replaced with tokens only a failing run produces. Signature-before-returncode
ordering was kept deliberately so log content can override a lying exit code; this
is safe because `lowcompute_4x4090.sh` is `set -euo pipefail`, ends in `exec bash
train.sh`, and its only `exit 0` is the dry-run branch.

**Mitigation for the root cause (synthetic fixtures):** the verbatim log of the
real run is committed as `tests/fixtures/real_successful_train.log`, and the
regression test asserts it **still contains** the benign NCCL banner and warnings,
so the fixture cannot later be "cleaned up" back into a blind spot.

### 4.3 Ranks were not bound to their GPUs — `2d99676` (WorldModel)

Both distributed entry points called `torch.cuda.set_device()` and then left
`init_process_group` without `device_id`. torch therefore could not infer the
rank->GPU mapping and warned, on every rank, that this "can potentially cause a
hang if this rank to GPU mapping is incorrect" — a latent hang under the
multi-hour distributed timeout, not log noise. Affects training
(`alaya/utils/distributed.py`) and context-parallel rollouts used by eval
(`alaya/inference/rollout_utils.py`).

This was fixed **upstream rather than filtered in the kernel**: the kernel had
been treating a real defect's symptom as noise.

**Method note — the first verification was vacuous.** A 2-rank script that called
`init_distributed()` with `UserWarning` promoted to an error passed against the
*unpatched* code too. The device-id warning fires when the NCCL communicator is
first built, i.e. on the first collective, not at `init_process_group`. After
adding an `all_reduce`, the negative control behaved correctly:

| | unpatched | patched |
|---|---|---|
| both ranks | `FAIL: UserWarning: No device id is provided ...` | `OK ... allreduce=3.0` |

Always run the negative control. A verification that passes before the fix proves
nothing.

### 4.4 `run_wbench` could not be driven from another work tree — `dc577e9` (WorldModel)

`main()` built `CONFIG_PATH` with `out.relative_to(REPO)`, assuming the bucket
config it generates lives inside the WorldModel checkout. AutoResearcher writes
run configs under its own `runs/` tree, so Step 5 died in 1.46 s with
`ValueError: ... is not in the subpath of ...` — after printing the bucket plan,
before generating anything. Same class as U1: a WorldModel script assuming it is
driven from inside its own repo. `train.sh` already accepts an absolute
`CONFIG_PATH`, so relativisation now happens only when the config really is
inside the repo.

### 4.5 Failure messages discarded the cause — `97653bd`

Six phase helpers (render, wbench x2, merge, precache, gate) built their
`RuntimeError` from `proc.stdout` alone. Python tracebacks go to **stderr**, so
§4.4 surfaced as nothing but a progress banner, and an earlier gate failure read
as a bare `describe failed (rc=1): CUDA_VISIBLE_DEVICES=0,1,2,3`.

**Loop impact:** a failed node would have recorded a cause-free message into the
archive, leaving the meta-agent to reason about failures whose reasons were
discarded at the point of capture. Telemetry already stored both streams, so only
the surfaced message was lossy — but the surfaced message is what an agent sees.

---

## 5. Step results

### Step 3 — ingest of the three standard formats: PASSED (119 s)

All three formats ingested through the staging path; example data intact
afterwards (§3).

### Step 4 — two-step training writes a checkpoint: PASSED (531 s)

| | |
|---|---|
| steps | 2 (loss 0.4316 -> 0.2520, grad 0.0090 / 0.0845) |
| peak memory | 17.05 GB step 1, 18.28 GB step 2 alloc; 22.93 GB reserved (24 GB cards) |
| checkpoint | `checkpoint-2`, `lora.safetensors` 624 MB, `history_encoder.pt` present |
| load profile | 4 ranks serially, ~52 GB fp32 CPU build each (`ALAYA_SERIAL_MODEL_LOAD=1`) |

**Checkpoint isolation verified — evidence for a hard rule of the design.** Each
rank logged, verbatim:

```
[Resume] keep history encoder initialized from paths.history_encoder;
         resume checkpoint history_encoder.pt is ignored
[Resume] loaded weights (base=weights/alaya-world-ar); step reset to 0
```

Every node therefore starts from the released base with the optimizer step
counter reset, and the history encoder comes from configured paths rather than
being inherited from a prior node. This is what makes node scores comparable
across the tree: without it a child would inherit its parent's training state and
clade-metaproductivity would measure accumulated training rather than data quality.

**Operational note:** GPU utilization is **not** a liveness signal during model
load. A near-idle snapshot during serial rank loading was misread as "the run
died"; the run was healthy. Phase-timeout logic must key off log progress markers
(`[Setup] rank N/4`, `[Train] step=`), not device activity.

### Step 5 — proxy score reproduction

PENDING — results appended on completion.

### Step 6 — allowlist preflight and peak memory

PENDING — results appended on completion.

Scope caveat, fixed in advance: the preflight measures only the **largest**
`lora_allowlist` pair against each resolution. Smaller ranks are covered by a
fits-by-dominance argument, **not** by measurement, and this log will say which
pairs actually ran. Peak alloc/reserved is recorded for every pair including
passing ones, as headroom evidence for the loop's scheduling.

---

## 6. Findings that feed Plan 3

1. **GPU phases must be serialized with node execution.** `test_gate.py` failed
   once with `describe failed (rc=1)` purely because the unit suite was run
   concurrently with the GPU training test; it passes with idle GPUs (9/9). The
   gate's describe step is a GPU job. Phase classification must distinguish
   "GPU busy" from "recipe invalid", or contention is recorded as
   `invalid_recipe` and the agent is penalized for a scheduling artifact.
2. **Liveness ≠ GPU utilization** (§5, Step 4).
3. **A real-hardware smoke run belongs in the loop's own health checking,** not
   only in a pre-merge gate. All five defects here were invisible to a green unit
   suite; a long autonomous run that only ever exercises unit-tested paths will
   accumulate the same class of blind spot.
4. **Fixes belong upstream when the defect is upstream.** Two of the five
   (§4.3, §4.4) were initially handled, or nearly handled, by working around
   WorldModel from inside the kernel. Both were better fixed at the source, and
   §4.3's kernel-side filter was masking a genuine hang risk.
