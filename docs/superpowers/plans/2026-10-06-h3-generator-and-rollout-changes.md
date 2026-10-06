# MiniMax H3 Generator and Rollout Tool Changes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `rollout_h3` GPU tool (MiniMax H3, per-turn prompts on the training round grid), let `rollout_ltx25` condition on images at any frame, switch Wan off, make `rollout_alayaworld` AR-teacher only, add a configurable item cap to every job tool, and stop tool descriptions naming each other.

**Architecture:** Every generator is a `GpuJob` backend in `kernel/ar_kernel/tools/` that validates items at submit, stages input files into a kernel-private job dir, launches a bridge script in the generator's own conda env, and publishes each result as a `candidate`. H3 adds one backend (`tools/h3.py`) and one bridge (`bridges/h3_generate.py`); the other changes edit the existing backends and the shared base class.

**Tech Stack:** Python 3.12, pytest, MCP tool server, diffusers 0.40.0 (`MiniMaxH3ModularPipeline`), torchao int8, ffmpeg/ffprobe.

**Spec:** `docs/superpowers/specs/2026-10-06-h3-generator-and-rollout-changes-design.md`

## Global Constraints

- Work in `AutoResearcher/` on branch `h3-generator`. Run tests with `.envs/autoresearcher/bin/python -m pytest <path> -q` from `AutoResearcher/`. The default run excludes `gpu` tests.
- Never use more than 4 GPUs; GPU runs use `AR_TEST_GPUS=0,1,2,3`.
- Never use system Python: kernel code runs in `.envs/autoresearcher`, the H3 bridge in `.envs/gen-h3`.
- No fallback for old config shapes or old item fields.
- Commit messages are one plain sentence; no co-author or attribution lines.
- H3: 960x544, 24 fps, `frames` of the form 17n+5 with 124 <= frames <= 243, default 243.
- Round grid: first boundary at frame 25, then every 32 frames. A turn lasts at least 2 rounds.
- A tool description may name `job_wait`, `data_ingest` and `annotate_camera`, and no other tool or generator but its own.
- Code style: match the surrounding file (120 columns, sparse comments, `ToolError` for anything the agent can fix).

## Review Focus

1. An H3 or LTX keyframe whose image path escapes `/workspace` or is not an image: refused at submit with the item's index, like a bad top-level `image`. (Task 4)
2. Two keyframes on the same frame, or `-1` and `frames - 1` both given for LTX: refused, not silently merged. (Task 4)
3. An H3 item with more turns than the clip holds (3 turns at 124 frames): refused at submit naming the limit, not rendered with sub-2-round turns. (Task 6)
4. A turn or scene prompt that already ends in punctuation, or has surrounding whitespace: the built prompt has exactly one full stop between sentences. (Task 5)
5. A job with exactly `max_items` items is accepted; `max_items + 1` is refused and nothing is queued. (Task 3)

## File Structure

| File | Responsibility |
|---|---|
| `kernel/ar_kernel/tools/gpu_jobs.py` (modify) | base class: item cap, keyframe checking and staging, `check_item_for`; item schemas; tool registration |
| `kernel/ar_kernel/tools/rollouts.py` (modify) | AlayaWorld (AR only), Wan (unchanged logic), LTX (keyframes) |
| `kernel/ar_kernel/tools/h3.py` (create) | H3 turn layout, prompt builder, caption segments, `H3Backend` |
| `kernel/ar_kernel/bridges/h3_generate.py` (create) | H3 worker: encode, int8 transformer across GPUs, generate |
| `kernel/ar_kernel/bridges/ltx25_generate.py` (modify) | keyframes passed to the pipeline |
| `kernel/ar_kernel/tools/captioner.py`, `images.py`, `doctor.py` (modify) | item cap; description; flat config |
| `configs/kernel.yaml` (modify) | flat generator switches, `h3` block |
| `tests/test_h3.py` (create), `tests/test_rollouts.py`, `tests/test_gpu_jobs.py`, `tests/test_captioner.py`, `tests/test_doctor.py`, `tests/fixtures/fake_gen_worker.py` (modify) | tests |
| `envs/gen-h3.yml`, `envs/gen-h3.pip.txt` (create), `scripts/setup_envs.sh`, `README.md`, both specs (modify) | env and docs |

---

### Task 1: Flat generator config, Wan off

**Files:**
- Modify: `configs/kernel.yaml` (generators block), `kernel/ar_kernel/tools/gpu_jobs.py:92-94,411-412`, `kernel/ar_kernel/tools/rollouts.py` (`Ltx25Backend.worker_groups`), `kernel/ar_kernel/doctor.py:192-194`
- Test: `tests/test_gpu_jobs.py`, `tests/test_rollouts.py`, `tests/test_doctor.py`

**Interfaces:**
- Produces: `enabled_variants(block, names) -> list[str]` returns the entries of `block["variants"]` (a list) that are in `names`, in config order. A generator is on when its block has a truthy `enabled` or a non-empty `variants` list.

- [ ] **Step 1: Update the test fixtures to the new shape (they fail until the code changes)**

`tests/test_rollouts.py`:

```python
def small_cfg(env="autoresearcher", enabled=("dmd4", "ar30")):
    raw = copy.deepcopy(REAL.raw)
    a = raw["generators"]["alayaworld"]
    a["env"] = env
    a["variants"] = list(enabled)
    return KernelConfig(raw=raw, repo_root=REAL.repo_root)
```

```python
def small_wan_cfg(env="autoresearcher", enabled=True, **over):
    raw = copy.deepcopy(REAL.raw)
    w = raw["generators"]["wan22"]
    w["env"] = env
    w["enabled"] = enabled
    w.update(over)
    return KernelConfig(raw=raw, repo_root=REAL.repo_root)
```

In `small_ltx_cfg` replace the `for v in ("distilled", "dev"): b["variants"][v]["enabled"] = v in enabled` loop with `b["variants"] = list(enabled)` and fix its docstring's first line to "The real ltx25 block with the given variants enabled, pinned to".

Replace the four `variants={...}` arguments:
- line ~1190: `q = make(gpus=gpus, workers=cap, peak_rss_gib=rss)`
- lines ~1202, ~1213, ~1223: `q = make(peak_rss_gib=40)`

GPU tests: line ~590 and ~703 become `raw["generators"]["alayaworld"]["variants"] = [variant]`; line ~976 becomes `raw["generators"]["wan22"]["enabled"] = True`; line ~1328 becomes `if variant not in REAL.get("generators.ltx25.variants"):`.

`tests/test_gpu_jobs.py`, `toggled_cfg`:

```python
def toggled_cfg(annotate=True, images=True, alaya=("dmd4", "ar30"), wan=True, ltx=("distilled",)):
    import copy
    base = KernelConfig.load()
    raw = copy.deepcopy(base.raw)
    raw["annotate"]["enabled"], raw["images"]["enabled"] = annotate, images
    g = raw["generators"]
    g["alayaworld"]["variants"] = list(alaya)
    g["wan22"]["enabled"] = wan
    g["ltx25"]["variants"] = list(ltx)
    return KernelConfig(raw=raw, repo_root=base.repo_root)
```

`tests/test_doctor.py` line ~145: `"generators": {"wan22": {"env": ".envs/gen-wan22", "enabled": False}}},`

Add to `tests/test_gpu_jobs.py`:

```python
def test_the_default_config_has_wan_off_and_ltx_distilled_only(tmp_path):
    from ar_kernel.tools.gpu_jobs import build_gpu_backends, enabled_variants
    rec = Recorder(tmp_path / "run")
    cfg = KernelConfig.load()
    names = {b.name for b in build_gpu_backends(cfg, tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)}
    assert "rollout_wan22" not in names and {"rollout_alayaworld", "rollout_ltx25"} <= names
    assert enabled_variants(cfg.get("generators.ltx25"), ("distilled", "dev")) == ["distilled"]
    assert enabled_variants({"variants": ["dev", "distilled", "x"]}, ("distilled", "dev")) == ["dev", "distilled"]
    assert enabled_variants({}, ("distilled", "dev")) == []
```

- [ ] **Step 2: Run the affected tests to see them fail**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_gpu_jobs.py tests/test_rollouts.py tests/test_doctor.py -q -x`
Expected: FAIL (e.g. `AttributeError: 'list' object has no attribute 'get'`).

- [ ] **Step 3: Change the code and the config**

`kernel/ar_kernel/tools/gpu_jobs.py`:

```python
def enabled_variants(block: dict, names: tuple[str, ...]) -> list[str]:
    """The variants config `block` lists (`variants: [a, b]`) that the backend knows, in config order."""
    return [v for v in block.get("variants") or [] if v in names]
```

and in `build_gpu_backends`:

```python
    def generator_on(name: str) -> bool:
        block = cfg.get(f"generators.{name}") or {}
        return bool(block.get("enabled") or block.get("variants"))
```

Update that function's docstring: replace "(a generator needs an enabled variant)" with "(a generator needs `enabled` or a listed variant)".

`kernel/ar_kernel/tools/rollouts.py`, `Ltx25Backend.worker_groups`: replace
`rss = float(self.block["variants"][variant]["peak_rss_gib"])` with `rss = float(self.block["peak_rss_gib"])`.

`kernel/ar_kernel/doctor.py`: replace the three lines computing `variants` / `enabled` with
`enabled = bool(block.get("enabled") or block.get("variants"))`.

`configs/kernel.yaml`, generators block:
- header comment: `generators:                       # rollout_* GPU jobs; switched on only after a smoke run`
- alayaworld: `variants: [dmd4, ar30]`
- wan22: replace the `variants:` line with `enabled: false                 # smoke run passed (2 T2V + 2 I2V); switched off 2026-10-06`
- ltx25: replace the `variants:` line with two lines:

```yaml
    variants: [distilled]         # the enabled ones, first = default; dev fits but is 5.8x slower
    peak_rss_gib: 40
```

  and in the comment above it change "so it stays disabled" to "so it is left out".

- [ ] **Step 4: Run the tests**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_gpu_jobs.py tests/test_rollouts.py tests/test_doctor.py -q`
Expected: PASS. If a test still enables Wan through the old shape, `grep -n "variants" tests/*.py` and convert it by the rules above.

- [ ] **Step 5: Commit**

```bash
git add configs/kernel.yaml kernel tests
git commit -m "generator switches are flat: enabled or a variants list; Wan is off"
```

---

### Task 2: `rollout_alayaworld` renders with the AR teacher only

**Files:**
- Modify: `kernel/ar_kernel/tools/rollouts.py` (constants, `render_config`, `AlayaWorldBackend`), `kernel/ar_kernel/tools/gpu_jobs.py` (the `rollout_alayaworld` registration), `configs/kernel.yaml`
- Test: `tests/test_rollouts.py`, `tests/test_gpu_jobs.py`

**Interfaces:**
- Produces: `render_config(cfg, *, rounds_per_turn, seed, indices, work, text_cache, node_lora=None, node_rank=0, history_encoder=None) -> dict` (no `variant`). `rollout_alayaworld` takes `items`, `rounds_per_turn`, `seed`, `node`. Generator name `alayaworld-ar30` (`@<node>` suffix for a node).

- [ ] **Step 1: Rewrite the tests**

`tests/test_rollouts.py`:

- `small_cfg`: signature `small_cfg(env="autoresearcher", enabled=True)`; body sets `a["enabled"] = enabled` and `a.pop("variants", None)` in place of the `variants` line.
- `test_render_config_has_the_listed_fields`: drop `variant="dmd4", ` from the call.
- Replace `test_ar30_differs_from_dmd4_in_exactly_the_four_keys` with:

```python
def test_render_config_uses_the_ar_teacher(tmp_path):
    c = render_config(REAL, rounds_per_turn=3, seed=1, indices=[0], work=tmp_path, text_cache=tmp_path / "te")
    assert c["paths"]["dmd_resume"] is None
    v = c["validation"]
    assert (v["sampling_steps"], v["scheduler"], v["cfg_scale"]) == (30, "shift", 3.0)
```

- `test_submit_refuses` parameters: delete the row `(first_person(), {"variant": "dmd8"}, "variant"),`.
- Delete `test_submit_refuses_a_disabled_variant` and `test_submit_with_only_ar30_enabled_defaults_to_ar30`.
- `test_submit_accepts_every_action_and_fills_defaults`: last line becomes
  `assert (args["rounds_per_turn"], args["seed"]) == (3, 42) and "variant" not in args`.
- `_job`: `"args": {"seed": 42, **args}`.
- Two `"generator": "alayaworld-dmd4"` / `== "alayaworld-dmd4"` occurrences (lines ~356, ~380): `alayaworld-ar30`.
- Replace `test_ar30_job_is_named_after_its_variant` with:

```python
def test_a_job_renders_with_the_ar_teacher(env):
    q, caller, _, run_dir = env
    job, by = run_job(q, caller, [first_person(turns=[{"action": "W"}])])
    assert by[0]["candidate"]["provenance"]["generator"] == "alayaworld-ar30"
    cfg = yaml.safe_load((run_dir / "jobs" / job / "render_config.yaml").read_text())
    assert cfg["validation"]["sampling_steps"] == 30 and cfg["paths"]["dmd_resume"] is None
```

- Replace `test_render_config_with_a_node_adds_its_lora_and_history_encoder` with:

```python
def test_render_config_with_a_node_adds_its_lora_and_history_encoder(tmp_path):
    kw = dict(rounds_per_turn=3, seed=1, indices=[0], work=tmp_path, text_cache=tmp_path / "te")
    node = dict(node_lora=tmp_path / "lora", node_rank=32, history_encoder=tmp_path / "ckpt" / "history_encoder.pt")
    plain, tuned = (_flat(render_config(REAL, **kw, **extra)) for extra in ({}, node))
    assert {k: tuned[k] for k in plain if plain[k] != tuned[k]} == {
        ("paths", "dmd_resume"): str(tmp_path / "lora"),
        ("paths", "history_encoder"): str(tmp_path / "ckpt" / "history_encoder.pt"),
        ("lora", "rank"): 32, ("lora", "alpha"): 32}
```

- Replace `test_a_node_job_renders_with_the_nodes_fine_tune` (and its `parametrize` line) with:

```python
def test_a_node_job_renders_with_the_nodes_fine_tune(env):
    q, caller, _, run_dir = env
    checkpoint = _scored_node(run_dir)
    job, by = run_job(q, caller, [first_person(turns=[{"action": "W"}])], node="n2")
    assert by[0]["candidate"]["provenance"]["generator"] == "alayaworld-ar30@n2"
    cfg = yaml.safe_load((run_dir / "jobs" / job / "render_config.yaml").read_text())
    assert cfg["paths"]["dmd_resume"] == str(checkpoint) and cfg["lora"]["rank"] == 32
    assert cfg["paths"]["history_encoder"] == str(checkpoint / "history_encoder.pt")
```

- `test_build_gpu_backends_includes_alayaworld_only_with_an_enabled_variant`: rename to `..._only_when_enabled`; loop `for enabled, present in ((False, False), (True, True)):`.
- GPU tests `test_real_alayaworld_rollout` and `test_real_alayaworld_rollout_of_a_node`: delete the `@pytest.mark.parametrize("variant", ...)` line and the `variant` parameter; replace the `raw[...]["variants"] = [variant]` line with `raw["generators"]["alayaworld"]["enabled"] = True`; delete `"variant": variant, ` from every `submit` args dict; replace any remaining `variant` in an f-string or assertion with the literal `ar30`. Check with `grep -n "variant" tests/test_rollouts.py | sed -n '1,40p'`: after this step no line between the AlayaWorld section start and `small_wan_cfg` mentions `variant`.

`tests/test_gpu_jobs.py`:
- `toggled_cfg`: parameter `alaya=True`; body `g["alayaworld"]["enabled"] = alaya` and `g["alayaworld"].pop("variants", None)`.
- In the `parametrize` rows replace `"alaya": ()` with `"alaya": False` and `"alaya": ("ar30",)` with `"alaya": True`.
- `test_listed_tools_show_only_enabled_tools_and_enabled_variants`: call `toggled_cfg(images=False, wan=False, ltx=("distilled",))`; delete the `alaya, ltx = ...` line and the `'dmd4'` assertion; keep
  `ltx = tools["rollout_ltx25"].description` and `assert "'distilled'" in ltx and "'dev'" not in ltx`; add
  `assert "variant" not in tools["rollout_alayaworld"].input_schema["properties"]`.

- [ ] **Step 2: Run to see them fail**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_rollouts.py tests/test_gpu_jobs.py -q -x`
Expected: FAIL (`render_config() missing 1 required keyword-only argument: 'variant'`).

- [ ] **Step 3: Change the code**

`kernel/ar_kernel/tools/rollouts.py`:

- Delete `from ..eval.lora import concat_eval_lora`.
- Delete the `VARIANTS = ...` and `VARIANT_TEXT = ...` lines. Keep `AR30`; its comment becomes
  `# The AR teacher without the DMD LoRA, sampled as configs/infer_i2v_camera_ar.yaml does.`
- `render_config`:

```python
def render_config(cfg, *, rounds_per_turn: int, seed: int, indices: list[int], work: Path,
                  text_cache: Path, node_lora: Path | None = None, node_rank: int = 0,
                  history_encoder: Path | None = None) -> dict:
    """configs/wbench_full.yaml as eval/render.py:build_render_config adapts it, pointed at this
    job's cases and sampled with the AR teacher (AR30). The prompt cache is the run's (the eval's
    own holds only WBench's prompts). `node_lora`: a node's checkpoint (LoRA rank `node_rank`) with
    its `history_encoder`."""
```

  Body unchanged down to `mode["wbench_chunks_per_turn"] = int(rounds_per_turn)`, then:

```python
    for (section, key), value in AR30.items():
        c[section][key] = value
    if node_lora is not None:
        c["paths"].update(dmd_resume=str(node_lora), history_encoder=str(history_encoder))
        c["lora"]["rank"] = c["lora"]["alpha"] = node_rank
    return c
```

- `AlayaWorldBackend.description`: delete the sentence `"Samplers (`variant`): {variants}. "` so the text reads `"... for all its rounds. Camera moves steer translation reliably, ..."`.
- `__init__`:

```python
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_turns = int(self.block["max_turns"])
        self.description = self.description.replace("{max_turns}", str(self.max_turns))
```

- Delete the `enabled_variants` method.
- `check_args`: delete the `args.setdefault("variant", ...)` line and the `if args["variant"] not in ...` block.
- `generator_name`: `return "alayaworld-ar30" + (f"@{node}" if node else "")`.
- `produce`: replace the whole `if job.args.get("node"):` block with

```python
            if job.args.get("node"):
                node = self._node(job.args["node"])
                checkpoint = self.run_dir / node["checkpoint_path"]
                fine_tune = dict(node_lora=checkpoint, node_rank=node["lora_rank"],
                                 history_encoder=checkpoint / "history_encoder.pt")
```

  drop `variant=job.args["variant"], ` from the `render_config(...)` call, and delete the line
  `shutil.rmtree(work / "eval", ignore_errors=True)        # the concatenated LoRA`.
- The import line keeps `enabled_variants` (LTX uses it).

`kernel/ar_kernel/tools/gpu_jobs.py`, the `rollout_alayaworld` registration: delete the `variant:` parameter and `variant=variant, ` from the `submit(...)` call.

`configs/kernel.yaml`, alayaworld block: replace the comment lines and `variants:` line with

```yaml
    # Rendered with the 30-step AR teacher (the DMD LoRA off). Clips carry no pose (user decision
    # 2026-09-26: the renders follow commanded turns/orbits only weakly); agents run annotate_camera.
    enabled: true
```

- [ ] **Step 4: Run the tests**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_rollouts.py tests/test_gpu_jobs.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/kernel.yaml kernel tests
git commit -m "rollout_alayaworld renders with the AR teacher only"
```

---

### Task 3: Configurable item cap for every job tool

**Files:**
- Modify: `kernel/ar_kernel/tools/gpu_jobs.py` (`GpuJob.__init__`, `GpuJob.submit`, `job_description`), `kernel/ar_kernel/tools/captioner.py` (`CaptionBackend.__init__`, `submit`, `register_caption_tool`)
- Test: `tests/test_gpu_jobs.py`, `tests/test_captioner.py`

**Interfaces:**
- Produces: `backend.max_items: int | None` on `GpuJob` (from `self.block`) and `CaptionBackend` (from `captioner.max_items`). Over the cap, `submit` raises `ToolError("at most N items per job: got M; send the rest in another job")`.

- [ ] **Step 1: Write the failing tests**

`tests/test_gpu_jobs.py`:

```python
def _capped(tmp_path, cap):
    import copy
    base = KernelConfig.load()
    raw = copy.deepcopy(base.raw)
    raw["annotate"]["max_items"] = cap

    class CappedJob(FakeJob):
        name = tool = "rollout_capped"
        config_key = "annotate"
    rec = Recorder(tmp_path / "run2")
    return CappedJob(KernelConfig(raw=raw, repo_root=base.repo_root), tmp_path / "run2", [0], TokenRegistry(rec), rec)


def test_a_job_over_max_items_is_refused_whole(env, tmp_path):
    from ar_kernel.tools.gpu_jobs import job_description
    q, caller, rec, ws, staging, run = env
    backend = _capped(tmp_path, 2)
    items = [{"src": "a.mp4", "seed": i} for i in range(3)]
    with pytest.raises(ToolError, match="at most 2 items per job: got 3"):
        backend.submit(q, caller, {"items": items})
    assert not q._jobs                                  # nothing was queued
    assert "At most 2 items per job." in job_description(backend)
    q.register(backend)
    assert backend.submit(q, caller, {"items": items[:2]})["job_id"]      # exactly the cap is accepted


def test_no_max_items_means_no_cap(env):
    from ar_kernel.tools.gpu_jobs import job_description
    q, caller, rec, ws, staging, run = env
    backend = q.backends["rollout_fake"]
    assert backend.max_items is None and "At most" not in job_description(backend)
```

`tests/test_captioner.py`:

```python
def test_more_clips_than_max_items_are_refused(make):
    import copy
    cfg = fake_cfg()
    raw = copy.deepcopy(cfg.raw)
    raw["captioner"]["max_items"] = 1
    q, caller, _, ws, _ = make(KernelConfig(raw=raw, repo_root=cfg.repo_root))
    for name in ("a.mp4", "b.mp4"):
        (ws / name).write_bytes(b"ok")
    with pytest.raises(ToolError, match="at most 1 items per job: got 2"):
        submit(q, caller, ["a.mp4", "b.mp4"], "Caption.")
    assert submit(q, caller, ["a.mp4"], "Caption.")["job_id"]
```

- [ ] **Step 2: Run to see them fail**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_gpu_jobs.py tests/test_captioner.py -q -k "max_items"`
Expected: FAIL (`AttributeError: ... has no attribute 'max_items'` / `DID NOT RAISE`).

- [ ] **Step 3: Implement**

`gpu_jobs.py`, `GpuJob.__init__`, after the `self.license = ...` line:

```python
        self.max_items = self.block.get("max_items")
```

`GpuJob.submit`, after the `if not items: raise ToolError("items is empty")` lines:

```python
        if self.max_items and len(items) > self.max_items:
            raise ToolError(f"at most {self.max_items} items per job: got {len(items)}; send the rest in another job")
```

Update the class docstring's last sentence to "A job takes any number of items, or at most the block's `max_items`."

`job_description`:

```python
def job_description(backend) -> str:
    limit = getattr(backend, "timeout_s", None)
    stop = f" A job is stopped after {limit / 3600:g} h; items not finished by then are item errors." if limit else ""
    cap = getattr(backend, "max_items", None)
    most = f" At most {cap} items per job." if cap else ""
    return backend.description + JOB_NOTE + most + stop
```

`captioner.py`, `CaptionBackend.__init__`, add as the last line:

```python
        self.max_items = (cfg.get("captioner") or {}).get("max_items")
```

`captioner.submit`, after the `paths is empty` check:

```python
    cap = getattr(q.backends[TOOL], "max_items", None)
    if cap and len(paths) > cap:
        raise ToolError(f"at most {cap} items per job: got {len(paths)}; send the rest in another job")
```

`register_caption_tool`: before the decorator add `cap = getattr(q.backends[TOOL], "max_items", None)`, and append to the description string expression `+ (f" At most {cap} clips per job." if cap else "")`.

- [ ] **Step 4: Run the tests**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_gpu_jobs.py tests/test_captioner.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel tests
git commit -m "every job tool takes an optional max_items cap from its config block"
```

---

### Task 4: `keyframes` items: checking, staging, and LTX conditioning at any frame

**Files:**
- Modify: `kernel/ar_kernel/tools/gpu_jobs.py` (schemas, `check_keyframes`, `GpuJob.submit/_stage/_collect`, LTX registration), `kernel/ar_kernel/tools/rollouts.py` (`Ltx25Backend`), `kernel/ar_kernel/bridges/ltx25_generate.py`, `tests/fixtures/fake_gen_worker.py` (ltx mode echo)
- Test: `tests/test_gpu_jobs.py`, `tests/test_rollouts.py`

**Interfaces:**
- Produces:
  - `check_keyframes(item: dict) -> None` — validates optional `item["keyframes"]`: a list of `{"image": str, "frame": int}` with exactly those keys, `frame >= -1`, no repeated frame. Raises `ToolError`.
  - `GpuJob.takes_keyframes: bool = False` — when true, `submit` path-checks each keyframe image and `_stage` copies it to `<in>/<index>_keyframe<n><suffix>`, rewriting `item["keyframes"][n]["image"]` to the staged path and adding `hashes["keyframe<n>"]`.
  - `GpuJob.check_item_for(self, item: dict, args: dict) -> None` — per-item checks that need the job's arguments; default does nothing; called by `submit` right after `check_item`, with the same error collection.
  - `KEYFRAMES_SCHEMA: dict` — the JSON schema of the `keyframes` field, shared by LTX and H3.
- LTX rule: a keyframe frame is `-1` or in `0..frames-1`; `-1` and `frames-1` together are a repeat. (LTX conditions frame 0 by latent replacement and any other frame as a guiding keyframe, so every index in range is valid.)

- [ ] **Step 1: Write the failing tests**

`tests/test_gpu_jobs.py`:

```python
class KeyframeJob(FakeJob):
    name = tool = "rollout_keyed"
    file_keys = ()
    takes_keyframes = True

    def check_item(self, item):
        from ar_kernel.tools.gpu_jobs import check_keyframes
        check_keyframes(item)

    def produce(self, job, items, work, out, cancel, report):
        for item in items:                                   # no worker: echo what was staged
            (out / f"{item['index']}.json").write_text(json.dumps({"ok": True, "keyframes": item["keyframes"]}))
            make_mp4(out / f"{item['index']}.mp4", seconds=1, fps=24, width=64, height=36)
        return [0], {}


@pytest.fixture
def keyed(env):
    from PIL import Image
    q, caller, rec, ws, staging, run = env
    Image.new("RGB", (64, 36), (1, 2, 3)).save(ws / "a.png")
    Image.new("RGB", (64, 36), (9, 9, 9)).save(ws / "b.png")
    q.register(KeyframeJob(KernelConfig.load(), run, [0], TokenRegistry(rec), rec, gpu_memory=lambda g: {i: 100 for i in g}))
    return q, caller, q.backends["rollout_keyed"], run


@pytest.mark.parametrize("keyframes, match", [
    ("a.png", "keyframes must be a list"),
    ([{"image": "a.png"}], "each keyframe must be"),
    ([{"image": "a.png", "frame": 0, "strength": 1}], "each keyframe must be"),
    ([{"image": 3, "frame": 0}], "each keyframe must be"),
    ([{"image": "a.png", "frame": True}], "each keyframe must be"),
    ([{"image": "a.png", "frame": -2}], "frame must be -1"),
    ([{"image": "a.png", "frame": 0}, {"image": "b.png", "frame": 0}], "repeat"),
    ([{"image": "../outside.png", "frame": 0}], "keyframes"),
    ([{"image": "missing.png", "frame": 0}], "keyframes"),      # as a missing top-level file is refused today
])
def test_bad_keyframes_are_refused_at_submit(keyed, keyframes, match):
    q, caller, backend, _ = keyed
    with pytest.raises(ToolError, match=match):
        backend.submit(q, caller, {"items": [{"keyframes": keyframes, "seed": 1}]})


def test_keyframe_images_are_staged_and_hashed_by_content(keyed):
    q, caller, backend, run = keyed
    items = [{"keyframes": [{"image": "a.png", "frame": 0}, {"image": "b.png", "frame": -1}], "seed": 1},
             {"keyframes": [{"image": "a.png", "frame": 0}, {"image": "b.png", "frame": 5}], "seed": 1},
             {"seed": 1}]
    out = q.wait(caller, backend.submit(q, caller, {"items": items})["job_id"], 120)
    assert out["state"] == "done", out
    got = job_result(out)["items"]
    staged = got[0]["worker"]["keyframes"]
    assert [k["frame"] for k in staged] == [0, -1]
    assert all(f"/jobs/{out['id']}/in/0_keyframe" in k["image"] for k in staged)
    assert got[2]["worker"]["keyframes"] == []
    h = [g["candidate"]["provenance"]["inputs_hash"] for g in got]
    assert len(set(h)) == 3                              # the frame index and the images are part of the hash
```

`tests/test_rollouts.py`, LTX section:

- `test_ltx_bad_items_are_refused_at_submit` parameters: replace the last row with
  `({"prompt": "p", "seed": 1, "keyframes": [{"image": 3, "frame": 0}]}, "each keyframe"), ({"prompt": "p", "seed": 1, "image": "frame.png"}, "image")`.
- In `test_ltx_produces_and_publishes_...`: the second item becomes
  `{"prompt": prompts[1], "keyframes": [{"image": "frame.png", "frame": 0}, {"image": "frame.png", "frame": -1}], "seed": 2}`
  and the last assertion becomes

```python
    frames1 = by[1]["worker"]["keyframes"]
    assert [k["frame"] for k in frames1] == [0, -1]
    assert all("_keyframe" in k["image"] for k in frames1) and by[0]["worker"]["keyframes"] == []
```

- Add:

```python
@pytest.mark.parametrize("frame, ok", [(0, True), (48, True), (-1, True), (49, False), (1000, False)])
def test_ltx_keyframe_frames_must_be_inside_the_clip(ltx_env, frame, ok):
    make, caller, _ = ltx_env
    q = make()
    args = {"items": [{"prompt": "p", "seed": 1, "keyframes": [{"image": "frame.png", "frame": frame}]}], "frames": 49}
    if ok:
        assert q.backends["rollout_ltx25"].submit(q, caller, args)["job_id"]
    else:
        with pytest.raises(ToolError, match="outside the clip"):
            q.backends["rollout_ltx25"].submit(q, caller, args)


def test_ltx_last_frame_given_twice_is_a_repeat(ltx_env):
    make, caller, _ = ltx_env
    q = make()
    frames = [{"image": "frame.png", "frame": -1}, {"image": "frame.png", "frame": 48}]
    with pytest.raises(ToolError, match="repeat"):
        q.backends["rollout_ltx25"].submit(q, caller, {"items": [{"prompt": "p", "seed": 1, "keyframes": frames}],
                                                       "frames": 49})
```

`tests/test_gpu_jobs.py`, `test_listed_item_schemas_type_every_field`: the second loop becomes

```python
    assert tools["rollout_wan22"].input_schema["properties"]["items"]["anyOf"][0]["items"]["properties"]["image"]["type"] == "string"
    key = tools["rollout_ltx25"].input_schema["properties"]["items"]["anyOf"][0]["items"]["properties"]["keyframes"]
    assert key["items"]["properties"]["frame"]["type"] == "integer" and key["items"]["required"] == ["image", "frame"]
```

- [ ] **Step 2: Run to see them fail**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_gpu_jobs.py tests/test_rollouts.py -q -k "keyframe or ltx or schemas"`
Expected: FAIL (`ImportError: cannot import name 'check_keyframes'`).

- [ ] **Step 3: Implement the shared parts in `gpu_jobs.py`**

Schemas, after `ClipItems`:

```python
KEYFRAMES_SCHEMA = {"type": "array", "description": "images the clip must show at given frames", "items": {
    "type": "object", "additionalProperties": False, "required": ["image", "frame"], "properties": {
        "image": _str("an image file under /workspace (any size; it is cropped and resized)"),
        "frame": {"type": "integer", "description": "0-based frame index; 0 is the first frame, -1 the last"}}}}
LtxItems = Annotated[list[dict[str, Any]] | str, items_schema(
    "the clips to render, one item each",
    {"prompt": _str("what the clip shows"), "keyframes": KEYFRAMES_SCHEMA, "seed": _SEED}, ["prompt", "seed"])]
```

After `check_item_seed`:

```python
def check_keyframes(item: dict) -> None:
    """An item's optional `keyframes`: [{'image': path, 'frame': int}], frame >= -1 (-1 = the last), none repeated."""
    frames = item.get("keyframes")
    if frames is None:
        return
    if not isinstance(frames, list):
        raise ToolError("keyframes must be a list of {'image': path, 'frame': int}")
    for k in frames:
        if not (isinstance(k, dict) and set(k) == {"image", "frame"} and isinstance(k["image"], str)
                and is_int(k["frame"])):
            raise ToolError(f"each keyframe must be {{'image': path, 'frame': int}}: got {k!r}")
        if k["frame"] < -1:
            raise ToolError(f"frame must be -1 (the last frame) or a frame index from 0: got {k['frame']}")
    seen = [k["frame"] for k in frames]
    if len(set(seen)) != len(seen):
        raise ToolError(f"keyframes repeat a frame: {seen}")
```

`GpuJob`: add class attribute `takes_keyframes = False     # items may carry `keyframes` (check_keyframes)`.

`GpuJob.submit`, the per-item loop becomes:

```python
        for n, item in enumerate(items):
            try:
                self.check_item(item)
                self.check_item_for(item, args)
            except ToolError as exc:
                bad[n] = str(exc)
            if copies_held_out(self.cfg, *strings(item)):       # before a GPU is scheduled
                self.recorder.event("isolation.refused", node=caller.node, phase=caller.phase,
                                    attempt=caller.attempt, component="tools", tool=self.name,
                                    payload={"item": n})
                bad.setdefault(n, EXCLUDED_PROMPT)
            for key in self.file_keys:
                if isinstance(item.get(key), str):
                    try:
                        clip_host_path(caller, item[key])
                    except PathError as exc:
                        bad.setdefault(n, f"{key}: {exc}")
                    item = {**item, key: container_path(item[key])}
            if self.takes_keyframes and n not in bad:
                frames = []
                for k in item.get("keyframes") or []:
                    try:
                        clip_host_path(caller, k["image"])
                    except PathError as exc:
                        bad.setdefault(n, f"keyframes: {exc}")
                    frames.append({**k, "image": container_path(k["image"])})
                item = {**item, "keyframes": frames}
            items[n] = item
```

Add after `check_item`:

```python
    def check_item_for(self, item: dict, args: dict) -> None:
        """Checks of one item that need the job's arguments (after check_args filled defaults in). Raises ToolError."""
```

`_stage`, before `return staged`:

```python
        if self.takes_keyframes:
            staged["keyframes"] = []
            for n, k in enumerate(item.get("keyframes") or []):
                dst = inp / f"{index}_keyframe{n}{Path(k['image']).suffix}"
                stage_clip(caller, k["image"], dst)
                staged["keyframes"].append({**k, "image": str(dst)})
                staged["hashes"][f"keyframe{n}"] = sha256_file(dst)
```

`_collect`: the `spec = ...` line becomes

```python
        spec = {k: v for k, v in item.items() if k not in (*self.file_keys, "index", "hashes", "keyframes")}
        if self.takes_keyframes:
            spec["keyframe_frames"] = [k["frame"] for k in item["keyframes"]]
```

The `rollout_ltx25` registration: `items: LtxItems` in place of `items: ClipItems`.

- [ ] **Step 4: Implement LTX**

`rollouts.py`: import `check_keyframes` from `.gpu_jobs`. In `Ltx25Backend`:

- `file_keys = ("image",)` becomes `takes_keyframes = True`.
- Replace `check_item = staticmethod(check_clip_item)` with:

```python
    def check_item(self, item):
        if "image" in item:
            raise ToolError("image is not a field of this tool: use keyframes [{'image': path, 'frame': 0}]")
        check_clip_item(item)
        check_keyframes(item)

    def check_item_for(self, item, args):
        frames = args["frames"]
        seen = [frames - 1 if k["frame"] == -1 else k["frame"] for k in item.get("keyframes") or []]
        if any(f >= frames for f in seen):
            raise ToolError(f"a keyframe frame is outside the clip: frames are 0..{frames - 1}, or -1 for the last")
        if len(set(seen)) != len(seen):
            raise ToolError(f"keyframes repeat a frame: {seen}")
```

- Description: replace the two sentences from `"Render training clips with LTX-2.5 (text-to-video, or image-to-video when an item carries a "` through the `Item: {...}` clause with

```python
            "Render training clips with LTX-2.5 from a text prompt, optionally pinned to images at chosen "
            "frames. A GPU job: returns {job_id} at once; collect with job_wait. Params (one value per "
            f"job): `variant` (one of {self.enabled_variants()}; default the first), `frames` (8k+1, "
            f"1 < frames <= {maximum}, default {default}), `height`/`width` (one of {self.block['resolutions']}). "
            "Item: {'prompt': str, 'keyframes'?: [{'image': file under /workspace, 'frame': int}], 'seed': int}. "
            "A keyframe at frame 0 is the first frame and one at -1 the last; any frame in between also "
            "works. Images of any size are center-cropped and resized to the clip size. "
```

  Keep the rest of the description ("Each result item gives a `candidate` ...") unchanged. Read the current string first and keep its `default`/`maximum` variable names.

`bridges/ltx25_generate.py`: replace the `images = [...]` line with

```python
            images = [ImageConditioningInput(k["image"], a.frames - 1 if k["frame"] == -1 else k["frame"], 1.0)
                      for k in item.get("keyframes") or []]
```

and in the module docstring replace "(text-to-video, or image-to-video when the item carries a first frame, conditioned at frame 0 with strength 1.0)" with "(text-to-video, or pinned to the item's keyframes: each image at its frame, -1 = the last, strength 1.0)".

`tests/fixtures/fake_gen_worker.py`, `ltx()`: in the status dict replace `"image": item.get("image")` with `"keyframes": item.get("keyframes")`.

- [ ] **Step 5: Run the tests**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_gpu_jobs.py tests/test_rollouts.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add kernel tests
git commit -m "rollout_ltx25 pins images at any frame through a keyframes list"
```

---

### Task 5: H3 turn layout, prompt builder and caption

**Files:**
- Create: `kernel/ar_kernel/tools/h3.py`, `tests/test_h3.py`

**Interfaces:**
- Produces (all in `ar_kernel.tools.h3`):
  - `turn_starts(frames: int, n_turns: int) -> list[int]` — first frame of each turn; raises `ToolError` when a turn would last under 2 rounds.
  - `build_prompt(scene: str, turns: list[str], starts: list[int], frames: int, keyframes: list[int]) -> str` — `keyframes` is the item's frame list, a subset of `[0, -1]`.
  - `build_caption(scene: str, turns: list[str], starts: list[int], frames: int) -> dict` — `{"caption": str, "segments": [{"time_range_s": [start, end], "prompt": str}]}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_h3.py`:

```python
"""rollout_h3: turn layout on the round grid, the MiniMax prompt format, captions, submit checks,
the real produce with a fake worker, and one real gpu smoke."""
import pytest

from ar_kernel.tools.h3 import build_caption, build_prompt, turn_starts
from ar_kernel.tools.server import ToolError

SCENE = "A first-person view walks along a cobblestone street"
TURNS = ["the camera pushes in slowly.", " the camera pans right to face a red door ", "a white dog runs out"]


@pytest.mark.parametrize("frames, n, starts", [
    (243, 1, [0]), (243, 2, [0, 121]), (243, 3, [0, 89, 153]), (124, 1, [0]), (158, 2, [0, 89])])
def test_turns_start_on_the_round_grid(frames, n, starts):
    got = turn_starts(frames, n)
    assert got == starts
    assert all((s - 25) % 32 == 0 for s in got[1:])


@pytest.mark.parametrize("frames, n", [(243, 4), (124, 2), (124, 3)])
def test_more_turns_than_the_clip_holds_is_refused(frames, n):
    with pytest.raises(ToolError, match="at most"):
        turn_starts(frames, n)


def test_prompt_is_one_shot_with_in_shot_timestamps():
    p = build_prompt(SCENE, TURNS, [0, 89, 153], 243, [])
    assert p == (
        "integrated_multimodal_description: [Shot 1] A first-person view walks along a cobblestone street. "
        "the camera pushes in slowly. At 00:03.708, the camera pans right to face a red door. "
        "At 00:06.375, a white dog runs out.\n\n"
        "overall_soundscape: Natural ambient sound of the scene.\n\n"
        "non_diegetic_music: N/A")
    assert "[Shot 2]" not in p and ".." not in p


@pytest.mark.parametrize("keyframes, line", [
    ([0], "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."),
    ([0, -1], "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
              "0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 10.12-second mark of "
              "the target video."),
    ([-1, 0], "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
              "0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 10.12-second mark of "
              "the target video."),
    ([-1], "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the "
           "10.12-second mark of the target video."),
])
def test_alignment_line_follows_the_keyframes(keyframes, line):
    p = build_prompt(SCENE, TURNS[:1], [0], 243, keyframes)
    assert p.startswith(line + "\n\nintegrated_multimodal_description: [Shot 1] ")


def test_caption_covers_every_turn_and_segments_tile_the_clip():
    c = build_caption(SCENE, TURNS, [0, 89, 153], 243)
    assert c["caption"] == ("A first-person view walks along a cobblestone street. the camera pushes in slowly. "
                            "Then the camera pans right to face a red door. Then a white dog runs out.")
    assert [s["time_range_s"] for s in c["segments"]] == [[0.0, 89 / 24], [89 / 24, 153 / 24], [153 / 24, 243 / 24]]
    assert c["segments"][1]["prompt"] == ("A first-person view walks along a cobblestone street. "
                                          "the camera pans right to face a red door.")
```

- [ ] **Step 2: Run to see them fail**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_h3.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'ar_kernel.tools.h3'`).

- [ ] **Step 3: Implement**

`kernel/ar_kernel/tools/h3.py`:

```python
"""rollout_h3: clips rendered by MiniMax H3 (diffusers' MiniMaxH3ModularPipeline, in its own env)
from the agent's per-turn prompts.

The agent gives a scene and a list of turns. The kernel puts the turn boundaries on the training
round grid (frame 25 + 32j), writes one continuous shot in MiniMax's prompt format with an in-shot
timestamp at each boundary ("At 00:03.708, ..."), and publishes the clip with one caption segment
per turn. A cut per turn ("[Shot 2] At ...") keeps the timing but changes the scene, so it is not
used. The clip carries no pose.
"""
from __future__ import annotations

from .server import ToolError

FPS = 24
HISTORY = 25                # the first round boundary
ROUND = 32
MIN_TURN_ROUNDS = 2         # 2.67 s: segments under 2.375 s are never trained, and H3's timing has ~0.5 s of slack
SOUND = "overall_soundscape: Natural ambient sound of the scene.\n\nnon_diegetic_music: N/A"
FIRST = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
BOTH = ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
        "0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the {end}-second mark of "
        "the target video.")
LAST = ("How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the "
        "{end}-second mark of the target video.")


def turn_starts(frames: int, n_turns: int) -> list[int]:
    """The first frame of each turn: frame 0, then round boundaries, each turn the same number of rounds."""
    rounds = (frames - HISTORY) // ROUND
    per = rounds // n_turns
    if per < MIN_TURN_ROUNDS:
        raise ToolError(f"a {frames}-frame clip holds at most {rounds // MIN_TURN_ROUNDS} turns "
                        f"(a turn lasts at least {MIN_TURN_ROUNDS} rounds of {ROUND} frames): got {n_turns}")
    return [0] + [HISTORY + ROUND * per * k for k in range(1, n_turns)]


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def _stamp(frame: int) -> str:
    seconds = frame / FPS
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def build_prompt(scene: str, turns: list[str], starts: list[int], frames: int, keyframes: list[int]) -> str:
    """The item as one continuous shot in MiniMax's base prompt format (alignment line, three fields)."""
    body = f"[Shot 1] {_sentence(scene)} {_sentence(turns[0])}"
    for text, start in zip(turns[1:], starts[1:]):
        body += f" At {_stamp(start)}, {_sentence(text)}"
    end = f"{frames / FPS:.2f}"
    line = {(0,): FIRST, (-1, 0): BOTH, (-1,): LAST}.get(tuple(sorted(keyframes)), "").format(end=end)
    return (f"{line}\n\n" if line else "") + f"integrated_multimodal_description: {body}\n\n{SOUND}"


def build_caption(scene: str, turns: list[str], starts: list[int], frames: int) -> dict:
    """The whole clip in `caption`; one segment per turn, in seconds, tiling [0, frames / 24]."""
    ends = [*starts[1:], frames]
    caption = " Then ".join(_sentence(t) for t in turns)
    return {"caption": f"{_sentence(scene)} {caption}",
            "segments": [{"time_range_s": [s / FPS, e / FPS], "prompt": f"{_sentence(scene)} {_sentence(t)}"}
                         for t, s, e in zip(turns, starts, ends)]}
```

- [ ] **Step 4: Run the tests**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_h3.py -q`
Expected: PASS. (`243 / 24 = 10.125` formats as `10.12`; the tests pin that.)

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/tools/h3.py tests/test_h3.py
git commit -m "H3 turn layout on the round grid, prompt builder and caption"
```

---

### Task 6: `H3Backend`, its tool registration and config

**Files:**
- Modify: `kernel/ar_kernel/tools/h3.py`, `kernel/ar_kernel/tools/gpu_jobs.py` (schema, registration, `build_gpu_backends`), `configs/kernel.yaml`, `tests/fixtures/fake_gen_worker.py`
- Test: `tests/test_h3.py`, `tests/test_gpu_jobs.py`

**Interfaces:**
- Consumes: `turn_starts`, `build_prompt`, `build_caption` (Task 5); `GpuJob`, `check_keyframes`, `check_item_seed`, `is_int`, `KEYFRAMES_SCHEMA`, `takes_keyframes`, `check_item_for` (Task 4); `max_items` (Task 3).
- Produces: `H3Backend(GpuJob)` with `name = tool = "rollout_h3"`, `generator = "minimax-h3"`, `config_key = "generators.h3"`; module constant `H3_BRIDGE: Path`. Bridge argv: `python H3_BRIDGE --items <work>/items.json --prompts <work>/prompts.json --out <out> --rank 0 --world 1 --weights <dir> --frames N --height H --width W --gpu0-reserve-gib G`. `prompts.json` maps `str(index)` to the built prompt. Staged items carry `keyframes: [{"image": <staged path>, "frame": 0 | -1}]`.

- [ ] **Step 1: Add the fake worker's h3 mode**

`tests/fixtures/fake_gen_worker.py`: add to the module docstring

```
With --prompts set it stands in for h3_generate.py instead (h3 mode, see h3(): checked first,
since that bridge's argv also carries --weights and --frames): it writes a --width x --height,
24 fps mp4 of --frames frames with an audio track and echoes --weights, --frames, the item's
prompt from the prompts file and its keyframes.
```

add `parser.add_argument("--prompts")                    # set: h3 mode (stands in for h3_generate.py)` and `parser.add_argument("--gpu0-reserve-gib", type=float)`, the function

```python
def h3(item, status_path):
    """h3_generate.py's contract without a model."""
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=size={args.width}x{args.height}:rate={item.get('fps', 24)}", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=32000", "-frames:v", str(args.frames), "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out / f"{item['index']}.mp4")],
                   check=True)
    status_path.write_text(json.dumps({"ok": True, "seconds": 0.01, "rank": args.rank,
                                       "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                                       "weights": args.weights, "frames": args.frames,
                                       "gpu0_reserve_gib": args.gpu0_reserve_gib,
                                       "prompt": prompts[str(item["index"])], "keyframes": item.get("keyframes")}))
```

and in the item loop, before the `if args.max_frames is not None:` branch:

```python
    if args.prompts is not None:
        h3(item, status_path)
        continue
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_h3.py` (move the new imports to the top of the file):

```python
import copy
import json
import threading
from pathlib import Path

from PIL import Image

from conftest import job_result
from ar_kernel.config import KernelConfig
from ar_kernel.data.probe import probe_video
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools import h3
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.h3 import H3Backend
from ar_kernel.tools.jobs import JobQueue
from tests.conftest import make_mp4

REAL = KernelConfig.load()
FAKE = Path(__file__).parent / "fixtures" / "fake_gen_worker.py"


def h3_cfg(env="autoresearcher", **over):
    raw = copy.deepcopy(REAL.raw)
    raw["generators"]["h3"].update(env=env, enabled=True, **over)
    return KernelConfig(raw=raw, repo_root=REAL.repo_root)


def item(**over):
    return {"scene_prompt": SCENE, "turns": [{"prompt": t} for t in TURNS], "seed": 7, **over}


@pytest.fixture
def h3_env(tmp_path, monkeypatch):
    """The real H3Backend.produce/run_workers, with h3_generate.py swapped for the fake worker."""
    monkeypatch.setattr(h3, "H3_BRIDGE", FAKE)
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=120)
    ws, staging = tmp_path / "ws", tmp_path / "staging"
    ws.mkdir(); staging.mkdir()
    Image.new("RGB", (640, 360), (10, 20, 30)).save(ws / "first.png")
    Image.new("RGB", (640, 360), (30, 20, 10)).save(ws / "last.png")
    q.register(H3Backend(h3_cfg(), tmp_path / "run", [0, 1, 4, 5], reg, rec, gpu_memory=lambda g: {i: 100 for i in g}))
    caller = reg.issue(node="n1", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    yield q, caller, staging
    q.shutdown()


def submit(q, caller, items, **params):
    return q.backends["rollout_h3"].submit(q, caller, {"items": items, **params})


@pytest.mark.parametrize("bad, params, match", [
    (item(scene_prompt=" "), {}, "scene_prompt"),
    (item(turns=[]), {}, "turns"),
    (item(turns=[{"prompt": ""}]), {}, "turn 1"),
    (item(turns=[{"prompt": "walk", "action": "W"}]), {}, "turn 1"),
    (item(turns=[{"prompt": "a"}] * 4), {}, "at most 3 turns"),
    (item(), {"frames": 124}, "at most 1 turns"),
    (item(seed="x"), {}, "seed"),
    (item(keyframes=[{"image": "first.png", "frame": 5}]), {}, "frame 0 .* or -1"),
    (item(keyframes=[{"image": "first.png", "frame": 0}, {"image": "last.png", "frame": 0}]), {}, "repeat"),
    (item(image="first.png"), {}, "image"),
    (item(), {"frames": 240}, r"17n\+5"), (item(), {"frames": 107}, "124"), (item(), {"frames": 260}, "243"),
])
def test_h3_submit_refuses(h3_env, bad, params, match):
    q, caller, _ = h3_env
    with pytest.raises(ToolError, match=match):
        submit(q, caller, [bad], **params)


def test_h3_job_over_the_configured_cap_is_refused(h3_env):
    q, caller, _ = h3_env
    cap = REAL.get("generators.h3.max_items")
    assert cap == 30
    with pytest.raises(ToolError, match="at most 30 items per job"):
        submit(q, caller, [item()] * (cap + 1))


def test_h3_produces_a_silent_clip_with_one_segment_per_turn(h3_env):
    q, caller, staging = h3_env
    items = [item(), item(turns=[{"prompt": "the camera holds still"}],
                          keyframes=[{"image": "first.png", "frame": 0}, {"image": "last.png", "frame": -1}])]
    out = q.wait(caller, submit(q, caller, items)["job_id"], 120)
    assert out["state"] == "done", out
    by = {i["index"]: i for i in job_result(out)["items"]}
    job = out["id"]
    c = by[0]["candidate"]
    assert c["provenance"]["generator"] == "minimax-h3" and c["provenance"]["seed"] == 7
    assert "pose" not in c and "camera_motion" not in c and c["frames"] == 243
    info = probe_video(staging / "rollouts" / job / "0.mp4")
    assert (info.width, info.height, info.frames, round(info.fps)) == (960, 544, 243, 24)
    import subprocess
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(staging / "rollouts" / job / "0.mp4")], capture_output=True, text=True).stdout.split()
    assert streams == ["video"]
    caption = json.loads((staging / "rollouts" / job / "0.json").read_text())
    assert caption == h3.build_caption(SCENE, TURNS, [0, 89, 153], 243)
    assert c["h3_prompt"] == by[0]["worker"]["prompt"] == h3.build_prompt(SCENE, TURNS, [0, 89, 153], 243, [])
    assert [(t["frame_start"], t["frame_end_exclusive"]) for t in c["turn_segments"]] == [(0, 89), (89, 153), (153, 243)]
    w = by[1]["worker"]
    assert w["prompt"].startswith("How the reference pictures align") and [k["frame"] for k in w["keyframes"]] == [0, -1]
    assert w["gpus"] == "0,1,4,5" and w["rank"] == 0             # one worker with every GPU
    assert w["weights"] == str(REAL.repo_root / REAL.get("generators.h3.weights")) and w["gpu0_reserve_gib"] == 16


def test_h3_finish_refuses_a_render_of_the_wrong_size_or_length(tmp_path):
    from types import SimpleNamespace
    rec = Recorder(tmp_path / "run")
    b = H3Backend(h3_cfg(), tmp_path / "run", [0, 1, 2, 3], TokenRegistry(rec), rec)
    out = tmp_path / "out"
    out.mkdir()
    staged = {"index": 0, **item()}
    make_mp4(out / "0.mp4", seconds=2, fps=24, width=1024, height=576)
    with pytest.raises(ValueError, match="960x544"):
        b.finish(SimpleNamespace(args={"frames": 243}), staged, out)
    make_mp4(out / "0.mp4", seconds=2, fps=24, width=960, height=544)
    with pytest.raises(ValueError, match="48 frames, not 243"):
        b.finish(SimpleNamespace(args={"frames": 243}), staged, out)


def test_h3_finished_candidate_passes_the_real_ingestor_as_per_chunk(tmp_path):
    from types import SimpleNamespace
    from ar_kernel.archive.db import open_db
    from ar_kernel.data.ingest import Candidate, Ingestor
    from tests.conftest import write_poses
    run_dir = tmp_path / "run"
    out = run_dir / "staging" / "rollouts"
    out.mkdir(parents=True)
    make_mp4(out / "0.mp4", seconds=243 / 24, fps=24, width=960, height=544)
    rec = Recorder(run_dir)
    res = H3Backend(h3_cfg(), run_dir, [0, 1, 2, 3], TokenRegistry(rec), rec).finish(
        SimpleNamespace(args={"frames": 243}), {"index": 0, **item()}, out)
    pose = write_poses(out / "vigeo.npz", n_frames=res["frames"], width=960, height=544)
    ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
    [r] = ing.ingest([Candidate(video=Path(res["video"]), caption=Path(res["caption"]), pose=pose,
                                camera_motion="moving", provenance={"kind": "rollout", "generator": "minimax-h3",
                                "job_id": "j1", "inputs_hash": "x", "seed": 7})], node_id="n1")
    assert r.accepted, r.reasons
    assert "video_timed_prompts_camera:per_chunk" in r.formats


def test_h3_is_registered_only_when_enabled(tmp_path):
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    for enabled, present in ((True, True), (False, False)):
        names = [b.name for b in build_gpu_backends(h3_cfg(enabled=enabled), tmp_path / "run", [0, 1, 2, 3],
                                                     TokenRegistry(rec), rec)]
        assert ("rollout_h3" in names) is present
```

If `make_mp4(seconds=243 / 24)` does not give exactly 243 frames, assert on `probe_video(out / "0.mp4").frames` first and pass that as the job's `frames` only in the ingest test (the turn layout needs 243; regenerate with `ffmpeg -frames:v 243` as the fake worker does if needed).

`tests/test_gpu_jobs.py`: `toggled_cfg` gains a parameter `h3=True` and the line `g["h3"]["enabled"] = h3`; in the first `parametrize` row add `"rollout_h3"` to the expected set; in the all-off row add `"h3": False`; in the two other rows add `"h3": False`. In `test_listed_tools_show_only_...` pass `h3=False`. In `test_listed_item_schemas_type_every_field` add:

```python
    h3 = tools["rollout_h3"].input_schema["properties"]["items"]["anyOf"][0]["items"]
    assert set(h3["required"]) == {"scene_prompt", "turns", "seed"}
    assert h3["properties"]["turns"]["items"]["required"] == ["prompt"]
    assert h3["properties"]["keyframes"] == tools["rollout_ltx25"].input_schema["properties"]["items"]["anyOf"][0]["items"]["properties"]["keyframes"]
```

- [ ] **Step 3: Run to see them fail**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_h3.py tests/test_gpu_jobs.py -q -x`
Expected: FAIL (`ImportError: cannot import name 'H3Backend'`).

- [ ] **Step 4: Config**

`configs/kernel.yaml`, append to `generators:`:

```yaml
  h3:
    env: .envs/gen-h3               # a conda-prefix env inside the repo (diffusers 0.40.0)
    weights: weights/minimax-h3     # MiniMaxAI/MiniMax-H3, diffusers layout, fl2va/t2va partition
    revision: 42ed227ee7df40d41602854ae760620d6eb651fe
    enabled: false                  # true after the smoke run
    # Spike 2026-10-06 on 4 x 24 GB, 960x544, 49 steps: int8 transformer across the 4 cards,
    # 243 frames in 1160 s, peak 21.5 GiB on GPU 0 and 14.6 GiB on the others, 64 GiB host RAM.
    # bf16 fits 124 frames only; int8 on one card with streamed offload peaks at 216 GiB host RAM.
    frames: [243, 243]              # [default, max]; 17n+5, min 124
    resolution: [544, 960]          # [height, width]
    max_items: 30                   # ~10 h at the measured speed
    gpu0_reserve_gib: 16            # GPU 0 also holds both VAEs and the small layers and decodes
    timeout_s: 43200
    license: MiniMax H3 Community License
```

- [ ] **Step 5: Implement the backend**

Append to `kernel/ar_kernel/tools/h3.py` (and extend its imports):

```python
import json
import subprocess
from pathlib import Path

from ..data.probe import probe_video
from .gpu_jobs import GpuJob, check_item_seed, check_keyframes, is_int
from .rollouts import FFMPEG_TIMEOUT_S, check_published

H3_BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "h3_generate.py"
MIN_FRAMES = 124


class H3Backend(GpuJob):
    """rollout_h3: one worker with every GPU of the run (the int8 transformer is spread across them)."""
    name = tool = "rollout_h3"
    kind = "rollout"
    generator = "minimax-h3"
    config_key = "generators.h3"        # timeout_s, max_items
    takes_keyframes = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        default, maximum = self.block["frames"]
        h, w = self.block["resolution"]
        self.description = (
            "Render training clips with MiniMax H3 from a scene and a list of turns: one continuous shot in "
            "which each turn's text takes effect at that turn's start. A GPU job: returns {job_id} at once; "
            f"collect with job_wait. Slow: about 20 minutes per {maximum}-frame clip. `frames`: 17n+5, "
            f"{MIN_FRAMES} <= frames <= {maximum}, default {default} (one value for the job). Item: "
            "{'scene_prompt': str, 'turns': [{'prompt': str}], 'keyframes'?: [{'image': file under /workspace, "
            "'frame': 0 or -1}], 'seed': int}. Turns start on training round boundaries (frame 25, then every "
            f"32 frames) and each lasts at least {MIN_TURN_ROUNDS} rounds, so a {maximum}-frame clip holds at most "
            f"{((maximum - HISTORY) // ROUND) // MIN_TURN_ROUNDS} turns. Write a turn as what happens, camera "
            "included, in phrases like: the camera pushes in / pulls out, pans left / right, trucks left / right, "
            "tilts up / down, pedestals up / down, arcs around the subject, tracks the subject, holds a static "
            "shot; add 'with small / large amplitude' or 'at slow / fast speed' when it matters. A keyframe at "
            f"frame 0 is the first frame and one at -1 the last; images are center-cropped to {w}x{h}. Each "
            f"result item gives a `candidate` for data_ingest (a {w}x{h}, 24 fps, silent mp4, a caption with one "
            "segment per turn, provenance); it carries no pose/camera_motion -- add one (annotate_camera then "
            "'moving', or 'static') before ingesting. Timing is approximate (about half a second): check the "
            "frames before trusting a segment boundary. Metadata, not labels: `turn_segments` and `h3_prompt`.")

    def check_args(self, args):
        default, maximum = self.block["frames"]
        frames = args.setdefault("frames", default)
        if not (is_int(frames) and MIN_FRAMES <= frames <= maximum and (frames - 5) % 17 == 0):
            raise ToolError(f"frames must be 17n+5 with {MIN_FRAMES} <= frames <= {maximum}: got {frames!r}")

    def check_item(self, item):
        if "image" in item:
            raise ToolError("image is not a field of this tool: use keyframes [{'image': path, 'frame': 0}]")
        if not isinstance(item.get("scene_prompt"), str) or not item["scene_prompt"].strip():
            raise ToolError("scene_prompt must be a non-empty string")
        turns = item.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ToolError("turns must be a non-empty list")
        for t, turn in enumerate(turns, 1):
            if not (isinstance(turn, dict) and set(turn) == {"prompt"} and isinstance(turn["prompt"], str)
                    and turn["prompt"].strip()):
                raise ToolError(f"turn {t} must be {{'prompt': non-empty text}}: got {turn!r}")
        check_keyframes(item)
        if any(k["frame"] not in (0, -1) for k in item.get("keyframes") or []):
            raise ToolError("a keyframe's frame must be frame 0 (the first frame) or -1 (the last)")
        check_item_seed(item)

    def check_item_for(self, item, args):
        turn_starts(args["frames"], len(item["turns"]))

    def _layout(self, job, item):
        turns = [t["prompt"] for t in item["turns"]]
        return turns, turn_starts(job.args["frames"], len(turns))

    def produce(self, job, items, work, out, cancel, report):
        frames = job.args["frames"]
        prompts = {}
        for item in items:
            turns, starts = self._layout(job, item)
            prompts[str(item["index"])] = build_prompt(item["scene_prompt"], turns, starts, frames,
                                                       [k["frame"] for k in item["keyframes"]])
        (work / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
        h, w = self.block["resolution"]
        return self.run_workers(self.block["env"], lambda r, world: [
            "python", str(H3_BRIDGE), "--items", str(work / "items.json"), "--prompts", str(work / "prompts.json"),
            "--out", str(out), "--rank", str(r), "--world", str(world),
            "--weights", str(self.cfg.repo_root / self.block["weights"]), "--frames", str(frames),
            "--height", str(h), "--width", str(w), "--gpu0-reserve-gib", str(self.block["gpu0_reserve_gib"])],
            [self.gpus], job=job, work=work, out=out, total=len(items), cancel=cancel, report=report,
            extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HUB_OFFLINE": "1"})

    def finish(self, job, item, out):
        frames = job.args["frames"]
        h, w = self.block["resolution"]
        silent = out / f"{item['index']}.silent.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / f"{item['index']}.mp4"), "-an",
                        "-c:v", "copy", str(silent)], check=True, timeout=FFMPEG_TIMEOUT_S)
        info = probe_video(silent)
        if (info.width, info.height) != (w, h):
            raise ValueError(f"rendered clip is {info.width}x{info.height}, not {w}x{h}")
        check_published(info)
        if info.frames != frames:
            raise ValueError(f"rendered clip has {info.frames} frames, not {frames}")
        turns, starts = self._layout(job, item)
        caption = self.write_caption(out, item, build_caption(item["scene_prompt"], turns, starts, frames))
        segments = [{"turn_index": k, "prompt": t, "frame_start": s, "frame_end_exclusive": e}
                    for k, (t, s, e) in enumerate(zip(turns, starts, [*starts[1:], frames]))]
        return {"video": silent, "caption": caption, "frames": info.frames, "turn_segments": segments,
                "h3_prompt": build_prompt(item["scene_prompt"], turns, starts, frames,
                                          [k["frame"] for k in item.get("keyframes") or []])}
```

`rollouts.py` does not import `h3.py`, so there is no import cycle. `finish` reads `item.get("keyframes") or []` because the unit tests call it with an unstaged item.

- [ ] **Step 6: Register the tool**

`gpu_jobs.py`, after `LtxItems`:

```python
H3Items = Annotated[list[dict[str, Any]] | str, items_schema(
    "the clips to render, one item each",
    {"scene_prompt": _str("the scene: place, objects, light, viewpoint"),
     "turns": {"type": "array", "description": "the clip, turn by turn, as one continuous shot",
               "items": {"type": "object", "additionalProperties": False, "required": ["prompt"], "properties": {
                   "prompt": _str("what happens during this turn, the camera move included")}}},
     "keyframes": KEYFRAMES_SCHEMA, "seed": _SEED},
    ["scene_prompt", "turns", "seed"])]
```

In `register_gpu_tools`, after the `rollout_ltx25` block:

```python
    if "rollout_h3" in b:
        @mcp.tool(name="rollout_h3", description=job_description(b["rollout_h3"]))
        async def rollout_h3(
                items: H3Items, ctx: Context,
                frames: Annotated[int | None, Field(description="17n+5 frames at 24 fps; one value for the job")] = None,
        ) -> dict[str, Any]:
            return await submit(ctx, "rollout_h3", items=items, frames=frames)
```

In `build_gpu_backends`: add `from .h3 import H3Backend` next to the other local imports and `(generator_on("h3"), H3Backend)` to the `enabled` list.

- [ ] **Step 7: Run the tests**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_h3.py tests/test_gpu_jobs.py tests/test_rollouts.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add configs/kernel.yaml kernel tests
git commit -m "rollout_h3: per-turn MiniMax H3 clips as a GPU job"
```

---

### Task 7: The H3 bridge

**Files:**
- Create: `kernel/ar_kernel/bridges/h3_generate.py`
- Test: `tests/test_h3.py`

**Interfaces:**
- Consumes: the argv and files of Task 6.
- Produces: `block_device_map(mem_gib: list[float], reserve0_gib: float) -> dict[str, int]`, `require_memory(mem_gib: list[float]) -> None` (raises `RuntimeError`), `fit(image, size)`; `main()` follows the bridge protocol: `<out>/<index>.mp4` then `<out>/<index>.json` (`{"ok": true, "seconds": t}` or `{"ok": false, "error": "..."}`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_h3.py`:

```python
def _bridge():
    import importlib.util
    spec = importlib.util.spec_from_file_location("h3_generate", h3.H3_BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)             # torch and diffusers are imported inside main(), not here
    return mod


def test_bridge_spreads_the_blocks_by_memory_and_keeps_small_layers_on_gpu_0():
    b = _bridge()
    dmap = b.block_device_map([23.6, 23.6, 23.6, 23.6], 16)
    blocks = [dmap[f"transformer_blocks.{i}"] for i in range(50)]
    assert [blocks.count(d) for d in range(4)] == [5, 15, 15, 15]        # the split the spike measured
    assert blocks == sorted(blocks)                                       # contiguous: one hop per card
    assert all(dmap[name] == 0 for name in b.SMALL)
    two = b.block_device_map([48.0, 48.0], 16)
    assert [list(two.values()).count(d) for d in (0, 1)][1] == 30
    assert set(b.block_device_map([80.0], 16).values()) == {0}


def test_bridge_refuses_cards_that_cannot_hold_the_text_encoder():
    b = _bridge()
    b.require_memory([23.6] * 4)
    with pytest.raises(RuntimeError, match="about 70 GiB"):
        b.require_memory([23.6, 23.6])


@pytest.mark.parametrize("size", [(1280, 720), (800, 1088), (640, 360), (960, 544)])
def test_bridge_crops_any_keyframe_to_the_canvas(size):
    img = _bridge().fit(Image.new("RGB", size, (5, 6, 7)), (960, 544))
    assert img.size == (960, 544) and img.mode == "RGB"
```

- [ ] **Step 2: Run to see them fail**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_h3.py -q -k bridge`
Expected: FAIL (`FileNotFoundError` for `h3_generate.py`).

- [ ] **Step 3: Implement**

`kernel/ar_kernel/bridges/h3_generate.py`:

```python
"""rollout_h3 worker (runs in the gen-h3 conda-prefix env, diffusers 0.40.0): MiniMax H3 through
diffusers' MiniMaxH3ModularPipeline, on every GPU it is given.

python h3_generate.py --items items.json --prompts prompts.json --out DIR --rank R --world W \
    --weights <weights/minimax-h3> --frames N --height H --width W --gpu0-reserve-gib G

Bridge protocol: handles items with index % world == rank, in index order; per item writes
<out>/<index>.mp4 then <out>/<index>.json ({"ok": true, "seconds": t} or {"ok": false, "error": "..."}).

Two phases, because the 62 GiB text encoder and the transformer do not fit together:
1. the text encoder at bf16 across the cards encodes every item (prompt and keyframes), then is freed;
2. the transformer, int8 weight-only (torchao), is placed with an explicit device map: the small
   layers and both VAEs on GPU 0 (the forward pass indexes its outputs with tensors that live
   there, so device_map="balanced" fails), the 50 blocks spread by card memory.
The pipeline index names the hub repo, so every component is pointed at the local folder
(`_component_specs` is private to diffusers; the env pins the version).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

FPS = 24
N_BLOCKS = 50
SMALL = ("proj_in", "audio_proj_in", "context_embedder", "token_refiner", "time_embedder", "time_proj", "rope",
         "norm_out", "proj_out", "audio_proj_out")
NOT_QUANTIZED = [m for m in SMALL if m != "rope"]
TEXT_ENCODER_GIB = 70       # 62 GiB of bf16 weights and room to run them


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Center-crop and resize to the canvas (the pipeline would stretch a first frame instead)."""
    return ImageOps.fit(image.convert("RGB"), size, Image.LANCZOS)


def require_memory(mem_gib: list[float]) -> None:
    if sum(mem_gib) < TEXT_ENCODER_GIB:
        raise RuntimeError(f"MiniMax H3 needs about {TEXT_ENCODER_GIB} GiB of GPU memory across the job's cards "
                           f"for its text encoder: got {sum(mem_gib):.0f} GiB on {len(mem_gib)} card(s)")


def block_device_map(mem_gib: list[float], reserve0_gib: float) -> dict[str, int]:
    """Small layers on GPU 0; the blocks in order across the cards, in proportion to each card's
    memory less `reserve0_gib` on GPU 0 (the VAEs, the small layers, decoding)."""
    room = [max(0.0, m - (reserve0_gib if i == 0 else 0.0)) for i, m in enumerate(mem_gib)]
    dmap = {name: 0 for name in SMALL}
    start = 0
    for device in range(len(room)):
        upto = round(N_BLOCKS * sum(room[:device + 1]) / sum(room))
        dmap.update({f"transformer_blocks.{b}": device for b in range(start, upto)})
        start = upto
    return dmap


def local(pipe, weights: str):
    for spec in pipe._component_specs.values():
        if getattr(spec, "pretrained_model_name_or_path", None):
            spec.pretrained_model_name_or_path = weights
    return pipe


def workflow_of(item: dict) -> str:
    return "fl2va" if item.get("keyframes") else "t2va"


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--items", "--prompts", "--out", "--weights"):
        ap.add_argument(name, required=True)
    for name in ("--rank", "--world", "--frames", "--height", "--width"):
        ap.add_argument(name, type=int, required=True)
    ap.add_argument("--gpu0-reserve-gib", type=float, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    mine = [i for i in json.loads(Path(a.items).read_text(encoding="utf-8")) if i["index"] % a.world == a.rank]
    if not mine:
        return 0

    import torch
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.utils.export_utils import encode_video
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration

    mem = [torch.cuda.mem_get_info(i)[1] / 2**30 for i in range(torch.cuda.device_count())]
    require_memory(mem)
    full = ModularPipeline.from_pretrained(a.weights)
    flows = {}                              # workflow -> (its encode blocks, the rest)
    for name in sorted({workflow_of(i) for i in mine}):
        rest = full.blocks.get_workflow(name)
        encode = SequentialPipelineBlocks.from_blocks_dict(
            {k: rest.sub_blocks.pop(k) for k in ("before_encode", "text_encoder") if k in rest.sub_blocks})
        flows[name] = (encode, rest)

    def pipelines(blocks: dict, **given) -> dict:
        """One pipeline per workflow over the same loaded components."""
        pipes = {name: local(b.init_pipeline(a.weights), a.weights) for name, b in blocks.items()}
        first = next(iter(pipes.values()))
        first.update_components(**given)
        first.load_components(dtype=torch.bfloat16)
        for other in list(pipes.values())[1:]:
            other.update_components(**{n: getattr(first, n) for n in other._component_specs
                                       if getattr(first, n, None) is not None})
            other.load_components(dtype=torch.bfloat16)
        return pipes

    def failed(index: int, exc: Exception) -> None:
        (out / f"{index}.json").write_text(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}),
                                           encoding="utf-8")

    # Phase 1: conditioning for every item.
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        a.weights, subfolder="text_encoder", dtype=torch.bfloat16, device_map="balanced",
        max_memory={i: f"{m - 2:.0f}GiB" for i, m in enumerate(mem)})
    encoders = pipelines({n: e for n, (e, _) in flows.items()}, text_encoder=encoder)
    states = {}
    for item in mine:
        index = item["index"]
        try:
            call = {"prompt": prompts[str(index)]}
            if item.get("keyframes"):
                images = {k["frame"]: fit(Image.open(k["image"]), (a.width, a.height)) for k in item["keyframes"]}
                call.update(height=a.height, width=a.width)
                if 0 in images:
                    call["image"] = images[0]
                if -1 in images:
                    call["last_image"] = images[-1]
            states[index] = encoders[workflow_of(item)](**call)
        except Exception as exc:          # noqa: BLE001 -- a per-item error; move on (protocol)
            failed(index, exc)
    del encoders, encoder
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: the transformer, then every clip.
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        a.weights, subfolder="transformer", dtype=torch.bfloat16,
        device_map=block_device_map(mem, a.gpu0_reserve_gib),
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=NOT_QUANTIZED))
    transformer.requires_grad_(False)
    renderers = pipelines({n: r for n, (_, r) in flows.items()}, transformer=transformer)
    first = next(iter(renderers.values()))
    first.vae.to("cuda:0")
    first.audio_vae.to("cuda:0")

    for item in mine:
        index, t0 = item["index"], time.monotonic()
        if index not in states:
            continue
        try:
            for i in range(len(mem)):
                torch.cuda.reset_peak_memory_stats(i)
            name = workflow_of(item)
            size = {"height": a.height, "width": a.width} if name == "t2va" else {}    # fl2va: set while encoding
            with torch.inference_mode():
                result = renderers[name](state=states.pop(index), num_frames=a.frames, **size,
                                         generator=torch.Generator().manual_seed(int(item["seed"])),
                                         output=["videos", "audio", "sampling_rate"])
            encode_video(result["videos"][0], fps=FPS, output_path=str(out / f"{index}.mp4"))
            status = {"ok": True, "seconds": round(time.monotonic() - t0, 2),
                      "peak_allocated_gib": [round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
                                             for i in range(len(mem))]}
        except Exception as exc:          # noqa: BLE001 -- a per-item error; move on (protocol)
            status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result = None
        torch.cuda.empty_cache()
        (out / f"{index}.json").write_text(json.dumps(status), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note on `block_device_map`: with `[23.6]*4` and reserve 16 the cumulative shares are 7.6, 31.2, 54.8, 78.4 of 78.4, giving boundaries 5, 20, 35, 50.

- [ ] **Step 4: Run the tests**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_h3.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel/ar_kernel/bridges/h3_generate.py tests/test_h3.py
git commit -m "H3 bridge: bf16 text encoder then an int8 transformer across the job's GPUs"
```

---

### Task 8: Tool descriptions do not name each other

**Files:**
- Modify: `kernel/ar_kernel/tools/images.py` (description), `kernel/ar_kernel/tools/rollouts.py` (Wan description)
- Test: `tests/test_gpu_jobs.py`

**Interfaces:**
- Consumes: `toggled_cfg(..., h3=True)` (Task 6), every registered tool.

- [ ] **Step 1: Write the failing test**

`tests/test_gpu_jobs.py`:

```python
OWN_WORDS = {            # the names only that tool's description may use
    "generate_images": ("generate_images", "Z-Image"),
    "rollout_alayaworld": ("rollout_alayaworld", "AlayaWorld"),
    "rollout_wan22": ("rollout_wan22", "Wan"),
    "rollout_ltx25": ("rollout_ltx25", "LTX"),
    "rollout_h3": ("rollout_h3", "MiniMax", "H3"),
}


def test_no_tool_description_names_another_generator_or_optional_tool(tmp_path):
    import re
    from ar_kernel.tools.captioner import register_caption_tool
    from ar_kernel.tools.gpu_jobs import build_gpu_backends
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=60)
    try:
        for b in build_gpu_backends(toggled_cfg(), tmp_path / "run", [0, 1, 2, 3], reg, rec):
            q.register(b)
        kit, mcp = ToolKit(reg, rec), new_mcp()
        register_gpu_tools(mcp, kit, q)
        register_caption_tool(mcp, kit, q)
        tools = {t.name: t.description for t in asyncio.run(mcp.list_tools())}
    finally:
        q.shutdown()
    assert set(OWN_WORDS) <= set(tools)
    named = [(tool, word) for tool, text in tools.items() for owner, words in OWN_WORDS.items() if owner != tool
             for word in words if re.search(rf"\b{re.escape(word)}\b", text)]
    assert named == []
```

- [ ] **Step 2: Run to see it fail**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_gpu_jobs.py -q -k names_another`
Expected: FAIL, listing `('generate_images', 'AlayaWorld')`, `('generate_images', 'Wan')`, `('generate_images', 'LTX')`, `('rollout_wan22', 'generate_images')`.

- [ ] **Step 3: Fix the descriptions**

`images.py`: the description's first two sentences become

```python
    description = ("Generate images from text prompts (Z-Image-Turbo), for use as the first or last frame of "
                   "a rollout. A GPU job: returns {job_id} at "
```

(the rest of the string is unchanged). Update the module docstring's first line to
`"""generate_images: Z-Image-Turbo images for rollouts to start or end on`.

`rollouts.py`, `Wan22Backend.description`: replace
`"size; center-cropped and resized to 1280x704, e.g. a generate_images frame), 'seed': "` with
`"size; center-cropped and resized to 1280x704), 'seed': "`.

If the test still lists a pair, remove that word from the named tool's description by describing the role instead of the name; do not touch `job_wait`, `data_ingest` or `annotate_camera` mentions.

- [ ] **Step 4: Run the tests**

Run: `.envs/autoresearcher/bin/python -m pytest tests/test_gpu_jobs.py tests/test_images.py tests/test_rollouts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel tests
git commit -m "no tool description names another generator or optional tool"
```

---

### Task 9: Env requirements and docs

**Files:**
- Create: `envs/gen-h3.yml`, `envs/gen-h3.pip.txt`
- Modify: `scripts/setup_envs.sh`, `README.md`, `docs/superpowers/specs/2026-10-06-h3-generator-and-rollout-changes-design.md`, `docs/superpowers/specs/2026-09-17-autoresearcher-design.md`

- [ ] **Step 1: Export the env the spike built**

```bash
.envs/autoresearcher/bin/python - <<'EOF'
import subprocess, yaml
raw = yaml.safe_load(subprocess.run(["conda", "env", "export", "-p", ".envs/gen-h3", "--no-builds"],
                                    capture_output=True, text=True, check=True).stdout)
deps = [d for d in raw["dependencies"] if isinstance(d, str)]        # the pip section goes to the pip file
open("envs/gen-h3.yml", "w").write(yaml.safe_dump(
    {"name": "gen-h3", "channels": ["conda-forge"], "dependencies": deps}, sort_keys=False))
EOF
{ echo "--extra-index-url https://download.pytorch.org/whl/cu132"; .envs/gen-h3/bin/pip freeze; } > envs/gen-h3.pip.txt
grep -c "" envs/gen-h3.yml envs/gen-h3.pip.txt
grep -E "^(torch|diffusers|transformers|torchao|accelerate)==" envs/gen-h3.pip.txt
```

Expected: both files non-empty; the grep prints `torch==2.13.0+cu132`, `diffusers==0.40.0`, `transformers==5.18.0`, `torchao==0.18.0` and an `accelerate` pin. Compare the yml's shape with `envs/gen-ltx25.yml` (`name`, `channels: conda-forge`, `dependencies` of `pkg=version`); if a line carries a build string or an `@ file://` pip entry, remove it.

- [ ] **Step 2: Build script and README**

`scripts/setup_envs.sh`: `ALL="autoresearcher panel vllm alayaworld wbench-main wbench-vp gen-zimage gen-wan22 gen-ltx25 gen-h3"`. No `case` entry is needed: gen-h3 has no checkout to install.

`README.md`:
- "Nine conda environments" becomes "Ten conda environments".
- The `ls .envs` check line lists `alayaworld autoresearcher gen-h3 gen-ltx25 gen-wan22 gen-zimage panel vllm wbench-main wbench-vp`.
- After the `hf download Lightricks/LTX-2.5 ...` command add:

```bash
hf download MiniMaxAI/MiniMax-H3 --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include "model_index.json" --include "modular_model_index.json" --include "transformer/*" \
  --include "text_encoder/*" --include "vae/*" --include "audio_vae/*" --include "tokenizer/*" \
  --include "processor/*" --include "scheduler/*" --include "audio_scheduler/*" \
  --local-dir weights/minimax-h3        # 137 GB; one --include per pattern
```

- [ ] **Step 3: Specs**

`docs/superpowers/specs/2026-10-06-h3-generator-and-rollout-changes-design.md`:
- §3.5: `gpu0_reserve_gib: 12                 # video VAE + small layers` becomes `gpu0_reserve_gib: 16                 # video VAE, small layers and decoding; gives the 5/15/15/15 split the spike ran`.
- §4: replace "Which indices LTX accepts is settled by the smoke run; the tool refuses the rest." with "Any index in `0..frames-1` is accepted: LTX conditions frame 0 by replacing its latent and any other frame as a guiding keyframe (`combined_image_conditionings`). An index outside the clip is refused."
- §4: add a bullet "A per-item check that needs the job's arguments (`check_item_for`) is added to `GpuJob` for the frame range, and reused by H3 for the turn count."
- §8: delete "and \"like Wan's\"" (that phrase is in a code docstring, not a description).

`docs/superpowers/specs/2026-09-17-autoresearcher-design.md`:
- §2.2 weights table: add the row
  `| `AutoResearcher/weights/minimax-h3` | MiniMax H3 (diffusers layout, t2va/fl2va partition), 24 fps | Video-generation data source |`
- The `generators` config row: replace `` `alayaworld: {dmd4, ar30}`, `ltx25: {dev, distilled}`, `wan22: {ti2v-5b}`, each `enabled` per §16.3 item 5 `` with `` `alayaworld`, `wan22`, `h3`: `enabled`; `ltx25`: `variants` (a list of the enabled ones) *(amended 2026-10-06: flat switches, AlayaWorld AR-only, Wan off, H3 added; see 2026-10-06-h3-generator-and-rollout-changes-design.md)* `` and leave the rest of the cell.

- [ ] **Step 4: Verify and commit**

Run: `bash -n scripts/setup_envs.sh && .envs/autoresearcher/bin/python -m pytest -q`
Expected: the whole default suite passes.

```bash
git add envs scripts README.md docs
git commit -m "gen-h3 env requirements, H3 weights download, and spec amendments"
```

---

### Task 10: Real smoke runs, enable H3, clean up

**Files:**
- Modify: `tests/test_h3.py`, `tests/test_rollouts.py` (`test_real_ltx25_rollout`), `configs/kernel.yaml`
- Delete: `third_party/MiniMax-H3` (untracked spike checkout)

- [ ] **Step 1: Add the H3 gpu smoke test**

Append to `tests/test_h3.py`:

```python
@pytest.mark.gpu
def test_real_h3_rollout(tmp_path):
    """AR_TEST_GPUS=0,1,2,3 pytest tests/test_h3.py -m gpu -s --basetemp=.cache/pytest/gpu   (~45 min)

    One text-only single-turn item and one three-turn item with first and last keyframes (Z-Image
    frames), through the real tool path; each candidate gets a ViGeo pose and must ingest as
    video_timed_prompts_camera:per_chunk."""
    import os
    import shutil
    from ar_kernel.archive.db import open_db
    from ar_kernel.data.ingest import Candidate, Ingestor
    from ar_kernel.tools.annotate import AnnotateBackend
    from ar_kernel.tools.images import ImageBackend
    from tests.test_rollouts import _wait

    gpus = [int(g) for g in os.environ.get("AR_TEST_GPUS", "0,1,2,3").split(",")]
    cfg = h3_cfg(env=REAL.get("generators.h3.env"))
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    staging = run_dir / "staging"
    ws.mkdir(parents=True); staging.mkdir(parents=True)
    rec = Recorder(run_dir)
    reg = TokenRegistry(rec)
    q = JobQueue(rec, threading.Lock(), wait_cap_s=3600)
    for b in (ImageBackend, AnnotateBackend, H3Backend):
        q.register(b(cfg, run_dir, gpus, reg, rec))
    caller = reg.issue(node="gpu", phase="improve_recipe", attempt=1, workspace_host=ws, staging_host=staging)
    host = lambda p: staging / Path(p).relative_to("/workspace/staging")
    try:
        img = _wait(q, caller, q.backends["generate_images"].submit(q, caller, {"width": 960, "height": 544, "items": [
            {"prompt": "a quiet harbor at dawn seen from the quay, fishing boats, photorealistic", "seed": 3},
            {"prompt": "the same harbor seen from the end of the pier looking back at the town, photorealistic",
             "seed": 4}]})["job_id"])
        assert img["state"] == "done", img.get("error")
        first, last = (i["image"] for i in job_result(img)["items"])
        items = [
            {"scene_prompt": "Live-action, a first-person view on a forest trail in autumn, tall pines, low sun",
             "turns": [{"prompt": "the camera pushes in at slow speed along the trail"}], "seed": 11},
            {"scene_prompt": "Live-action, a first-person view on a quay in a quiet harbor at dawn",
             "turns": [{"prompt": "the camera pushes in at slow speed along the quay"},
                       {"prompt": "the camera pans right with large amplitude toward the fishing boats"},
                       {"prompt": "a seagull lands on the nearest boat"}],
             "keyframes": [{"image": first, "frame": 0}, {"image": last, "frame": -1}], "seed": 12}]
        out = _wait(q, caller, submit(q, caller, items)["job_id"])
        assert out["state"] == "done", out.get("error")
        by = {i["index"]: i for i in job_result(out)["items"]}
        assert all("candidate" in by[i] for i in (0, 1)), by
        cands = [by[i]["candidate"] for i in (0, 1)]
        print("h3 worker:", [by[i]["worker"] for i in (0, 1)])
        ann = _wait(q, caller, q.backends["annotate_camera"].submit(
            q, caller, {"items": [{"video": c["video"]} for c in cands]})["job_id"])
        assert ann["state"] == "done", ann.get("error")
    finally:
        q.shutdown()
    ing = Ingestor(REAL, run_dir, open_db(run_dir), Recorder(run_dir))
    for i, c in enumerate(cands):
        a = job_result(ann)["items"][i]
        assert "error" not in a, a
        stage = staging / "ingest" / str(i)
        stage.mkdir(parents=True)
        shutil.copy(host(c["video"]), stage / "v.mp4")
        shutil.copy(host(c["caption"]), stage / "c.json")
        shutil.copy(host(a["pose"]), stage / "p.npz")
        [res] = ing.ingest([Candidate(video=stage / "v.mp4", caption=stage / "c.json", pose=stage / "p.npz",
                                      camera_motion="moving", provenance=c["provenance"], license=c["license"])],
                           node_id="gpu")
        assert res.accepted, res.reasons
        assert "video_timed_prompts_camera:per_chunk" in res.formats, res.formats
        shutil.copy(host(c["video"]), Path(".cache") / f"h3_smoke_{i}.mp4")      # to look at afterwards
```

`generate_images` item paths are `/workspace/staging/...` container paths, which the tool accepts as keyframe images.

- [ ] **Step 2: Add first-and-last keyframes to the LTX gpu smoke**

In `test_real_ltx25_rollout` (`tests/test_rollouts.py`), find the list of four items submitted to `rollout_ltx25` (two text-only, two with an `"image"`). Change the two image items to

```python
{"prompt": <unchanged>, "keyframes": [{"image": <first generated image>, "frame": 0}], "seed": <unchanged>},
{"prompt": <unchanged>, "keyframes": [{"image": <first generated image>, "frame": 0},
                                      {"image": <second generated image>, "frame": -1}], "seed": <unchanged>},
```

using the two `generate_images` results the test already makes, and replace the parametrize list with `["distilled"]`.

- [ ] **Step 3: Run both smokes on GPUs 0–3**

Check the cards are free first: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4`.

```bash
AR_TEST_GPUS=0,1,2,3 .envs/autoresearcher/bin/python -m pytest tests/test_rollouts.py -m gpu -k ltx25 -s --basetemp=.cache/pytest/gpu
AR_TEST_GPUS=0,1,2,3 .envs/autoresearcher/bin/python -m pytest tests/test_h3.py -m gpu -s --basetemp=.cache/pytest/gpu
```

Expected: both PASS; the H3 run takes about 45 minutes (two clips plus loading). Then look at the two saved clips: build a contact sheet with
`ffmpeg -y -loglevel error -i .cache/h3_smoke_1.mp4 -vf "fps=1,scale=384:-1,tile=5x2" -frames:v 1 .cache/h3_smoke_1.png`
and confirm (a) the clip starts on the first keyframe and ends on the last, (b) it is one continuous shot, (c) the pan and the seagull happen near 3.7 s and 6.4 s. Record what you see in the commit message body. If a smoke fails, fix the cause before going on; do not enable H3 on a failing smoke.

- [ ] **Step 4: Enable H3 and clean up**

`configs/kernel.yaml`: `generators.h3.enabled: true` with the comment `# smoke run 2026-10-06: 1 T2V + 1 first-and-last item pass ingest as per_chunk`.

Remove the spike checkout (it is untracked under the gitignored `third_party/`; only its prompt guide was used, and the format is recorded in the spec and `h3.py`):

```bash
git -C third_party/MiniMax-H3 remote -v      # must print MiniMax-AI/MiniMax-H3: confirms the target
rm -rf third_party/MiniMax-H3
```

- [ ] **Step 5: Run the whole default suite and commit**

Run: `.envs/autoresearcher/bin/python -m pytest -q`
Expected: PASS.

```bash
git add configs/kernel.yaml tests
git commit -m "rollout_h3 is enabled after its smoke run; LTX smoke covers first and last keyframes"
```
