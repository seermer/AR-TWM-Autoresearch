# MiniMax H3 generator and rollout tool changes

Date: 2026-10-06. Status: design, awaiting review.

## 1. Goal

Give the autoresearcher better sources of timed-prompt training clips, and trim the sources that
are not wanted:

1. A new GPU tool, `rollout_h3`, that renders clips with MiniMax H3 from per-turn prompts.
2. `rollout_ltx25` conditions on images at any frame, not only frame 0.
3. `rollout_wan22` is disabled in config; its code, env and weights stay.
4. `rollout_alayaworld` loses its 4-step DMD sampler and always renders with the 30-step AR teacher.
5. Every job tool gets an optional, configurable item cap.
6. No tool description names another generator or optional tool.

Non-goals: H3's reference mode (Ref2VA), H3 audio, 2K output, clips longer than 243 frames,
resolutions other than 960x544, and any change to how the evaluation renders.

## 2. What the spike measured (2026-10-06, 4 x 24 GB)

All runs use diffusers 0.40.0 (`MiniMaxH3ModularPipeline`), 960x544, 24 fps, 49 denoising steps.

| Setup | 5 s clip (124 frames) | 10 s clip (243 frames) | Host RAM peak |
|---|---|---|---|
| bf16 transformer across 4 GPUs | 468 s, 22.3 GiB peak | out of memory | 64 GiB |
| int8 transformer on 1 GPU, streamed offload | 540 s, 16.8 GiB peak | not run | 216 GiB |
| **int8 transformer across 4 GPUs (chosen)** | not run | **1,160 s; 21.5 GiB on GPU 0, 14.6 GiB on the others** | 64 GiB |

- The text encoder (Qwen3-VL, 62 GiB) loads at bf16 across the four cards (12–17 GiB each),
  encodes a prompt in under a second, and frees all memory when released.
- The video VAE and the transformer's small input/output layers take about 12 GiB and must share
  one card with the tensors the forward pass indexes (GPU 0). Automatic placement
  (`device_map="balanced"`) fails with a device mismatch; an explicit per-block map works.
- The pipeline index names the hub repo, so components must be pointed at the local folder.
- A first frame is stretched to the canvas, not cropped.
- Timestamps inside one continuous shot work. For "pan to a red door at 4 s, a dog runs out at
  7 s": with `At 00:04.000, ... At 00:07.000, ...` the pan began at about 4.5 s and the door
  opened at about 7 s, with no cut. With the same events and no timestamps, both happened in
  the last two seconds. With MiniMax's documented `[Shot N] At ..., the camera cuts to` format the
  timing held but the scene changed between shots. One scene, one seed.

Throughput is about 3 ten-second clips per hour with one worker on four cards.

## 3. `rollout_h3`

### 3.1 Tool interface

Job parameters:

| Parameter | Meaning | Default |
|---|---|---|
| `items` | the list below, or a `.json` file holding it | required |
| `frames` | clip length at 24 fps, of the form 17n+5 | 243 (max 243, min 124) |
| `seed` | not a job parameter: each item carries its own, as in `rollout_ltx25` | |

Item:

```json
{"scene_prompt": "Live-action, a first-person view walking along a cobblestone street ...",
 "turns": [{"prompt": "the camera pushes in slowly along the street"},
           {"prompt": "the camera pans right to face a red door"},
           {"prompt": "the door opens and a white dog runs out"}],
 "overall_soundscape": "Footsteps on stone and a light wind.",
 "non_diegetic_music": "N/A",
 "keyframes": [{"image": "first.png", "frame": 0}, {"image": "last.png", "frame": -1}],
 "seed": 42}
```

- *(Amended 2026-10-08.)* `overall_soundscape` and `non_diegetic_music` (strings, required) are
  two of the guide's three core fields; the kernel builds the third from `scene_prompt` and
  `turns`. `scene_prompt` starts with the visual style, as the guide's examples do.

- `scene_prompt` (string, required) and `turns` (non-empty list of `{"prompt": string}`) use the
  same names as `rollout_alayaworld`. The number of turns is limited by the clip length (§3.2).
- `keyframes` is the same field as in `rollout_ltx25` (§4). H3 accepts only `frame` 0 (first
  frame) and -1 (last frame), each at most once: neither, either or both. Any other index is
  refused.
- `seed` (int, required) has the same meaning as in the other rollout tools.
- Turn text is free text. The description lists MiniMax's camera vocabulary (push in / pull out,
  pan, truck, tilt, pedestal, arc, tracking, static) so agents phrase moves the way H3 was trained.

### 3.2 Prompt assembly (kernel)

Turn boundaries sit on the training round grid (frame 25 + 32j, the grid `data_formats` requires
for `video_timed_prompts_camera:per_chunk`), so a clip is eligible for per_chunk as well as
segment mode. A clip of `frames` frames holds `R = (frames - 25) // 32` whole rounds. Each of the
n turns gets `R // n` rounds and the last turn also takes what is left, so turn k (from 0)
starts at frame `25 + 32 * k * (R // n)` for k >= 1. A turn must last at least 2 rounds
(2.67 s): segments under 2.375 s are never trained, and the spike showed about half a second of
timing slack. So 243 frames (R = 6) allow up to 3 turns, with boundaries at 3.708 s and 6.375 s;
124 frames (R = 3) allow one. More turns than fit is an item error. The clip is not trimmed.

The kernel builds the prompt in MiniMax's base format (`skills/h3-prompt-writing/references/base-en.txt` of
MiniMax-AI/MiniMax-H3 at d21241f):

```text
<alignment line, only with images>

integrated_multimodal_description: [Shot 1] <scene_prompt>. The whole video is one continuous shot with smooth motion and no cuts. At 00:00.000, <turn 1>. At 00:SS.mmm, <turn 2>. At 00:SS.mmm, <turn 3>.

overall_soundscape: <the item's overall_soundscape>

non_diegetic_music: <the item's non_diegetic_music>
```

*(Amended 2026-10-08.)* The no-cuts sentence is fixed text from the kernel and every turn,
the first included, carries its start time. The guide documents a timestamp only at a cut
(`[Shot 2] At 00:03.500, the camera cuts to...`); a timestamp inside one shot is our own
extension, chosen because the cut form changes the scene between shots. Its timing is loose:
within about 0.5 s in the spike, about 1.3 s late in one smoke clip and about 4 s early in another. The tool description says
so. Each text is collapsed to one line, and a turn loses a trailing `, ; :` before its full stop.

Alignment line, copied from the guide, with `S.SS` the clip duration to two decimals:

| Images | Line |
|---|---|
| first only | `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.` |
| first and last | `How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the S.SS-second mark of the target video.` |
| last only | `How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the S.SS-second mark of the target video.` |

The first-and-last line has no angle or square brackets while the other two do. That is how
MiniMax's guide writes it, in both the rule and the worked example, so it is copied as is. The
line is prose for the model: the pipeline binds each keyframe itself, by prepending a
`<Picture i>: ` label and the image's vision block to the prompt, in the order first, last.

The builder is a pure function with unit tests; the assembled prompt is stored in the result.

### 3.3 Output

Per item, a `candidate` for `data_ingest`: a silent 960x544, 24 fps mp4, a caption, provenance
(generator `minimax-h3`), and no pose. *(Amended 2026-10-08.)* The caption holds the scene only,
`{"caption": "<scene_prompt>."}`: the timing of a turn is not reliable, so the agent checks the
frames and writes the per-turn segments it ingests. What it asked for stays in the result as
metadata. (`rollout_alayaworld`'s published caption is left as is.)

Metadata, not labels: `turn_segments` (each turn's text and frame range) and `h3_prompt`.
The published clip passes the existing checks (24 fps; 960/544 is within 2% of 16:9).

### 3.4 Bridge and backend

`kernel/ar_kernel/bridges/h3_generate.py`, run in `.envs/gen-h3`, follows the bridge protocol of
the other generators (`--items`, `--out`, per-item `<index>.mp4` then `<index>.json`):

1. Load the text encoder at bf16 across all visible GPUs; encode every item (prompt and
   crop-fitted keyframes); free it.
2. Load the transformer with torchao int8 weight-only quantization and an explicit device map:
   the small layers and both VAEs on GPU 0, the 50 blocks divided across the cards in
   proportion to each card's memory, less `gpu0_reserve_gib` on GPU 0.
3. Generate each item; an item failure is an item error.

Text-only items use the pipeline's `t2va` workflow, items with an image the `fl2va` workflow;
both share the loaded components. Keyframe images are staged into the job dir by the same
`GpuJob` extension LTX uses (§4). `H3Backend(GpuJob)` in `tools/h3.py` runs one worker
that is given every GPU of the run, so it adapts to the machine. The bridge raises if the cards
cannot hold the text encoder or the transformer.

### 3.5 Config

```yaml
generators:
  h3:
    env: .envs/gen-h3
    weights: weights/minimax-h3
    revision: 42ed227ee7df40d41602854ae760620d6eb651fe
    enabled: false                       # true after the smoke run
    frames: [243, 243]                   # [default, max]; 17n+5, min 124
    resolution: [544, 960]               # [height, width]
    max_items: 30                        # ~10 h at the measured 1,160 s per clip
    gpu0_reserve_gib: 16                 # video VAE, small layers and decoding; gives the 5/15/15/15 split the spike ran
    timeout_s: 43200
```

`envs/gen-h3.yml` and `envs/gen-h3.pip.txt` are exported from the spike's env (Python 3.12,
torch 2.13.0+cu132, diffusers 0.40.0, transformers 5.18.0, torchao 0.18.0), like `gen-ltx25`.
The README's env list gains `gen-h3`.

## 4. `rollout_ltx25`: images at any frame

The item's `image` field is replaced by:

```json
"keyframes": [{"image": "a.png", "frame": 0}, {"image": "b.png", "frame": -1}]
```

- `frame` is a 0-based frame index; `-1` is the last frame. Missing or empty is text-to-video;
  one entry at frame 0 is today's image-to-video.
- The bridge passes each entry as `ImageConditioningInput(path, frame, 1.0)`.
- Any index in `0..frames-1` is accepted: LTX conditions frame 0 by replacing its latent and any
  other frame as a guiding keyframe (`combined_image_conditionings`). An index outside the clip
  is refused.
- A per-item check that needs the job's arguments (`check_item_for`) is added to `GpuJob` for the
  frame range, and reused by H3 for the turn count.
- `GpuJob`'s input staging handles the top-level fields a backend lists in `file_keys`
  (path check, copy into the job dir). It is extended once to the `image` of each `keyframes`
  entry, for both LTX and H3.
- `prompt` and `seed` are unchanged. Wan and AlayaWorld keep their `image` field.

## 5. Wan disabled

`generators.wan22.enabled: false` (§7.1). `build_gpu_backends` then leaves
`rollout_wan22` unregistered. `Wan22Backend`, the bridge, `.envs/gen-wan22` and the weights stay.
Tests that exercise Wan enable it in their own config.

## 6. `rollout_alayaworld`: AR teacher only

- The `variant` argument, `VARIANTS`, `VARIANT_TEXT` and the `{variants}` description text go.
- `render_config` always applies the AR30 settings; a node's fine-tune is its checkpoint, so the
  rollout tool no longer calls `concat_eval_lora`.
- Config: `enabled: true` (§7.1). The generator name stays `alayaworld-ar30`.
- The evaluation's DMD rendering is untouched.

## 7. Config shape

### 7.1 Generator switches

The nested `variants: {name: {enabled: bool}}` map goes. No fallback for the old shape.

```yaml
generators:
  alayaworld: {enabled: true, ...}
  wan22:      {enabled: false, ...}
  h3:         {enabled: false, ...}
  ltx25:
    variants: [distilled]     # the enabled ones; the first is the default; empty = tool off
    peak_rss_gib: 40          # was per variant, with the same value for both
```

Single-mode generators use `enabled`, as `images` and `annotate` do. LTX lists only its enabled
variants; `dev` is left out, not listed as disabled. Provenance names are set in code and do not
change.

### 7.2 Item cap

Each job tool's config block (`captioner`, `annotate`, `images`, `generators.<name>`) accepts
`max_items`. Absent means no cap, as today. A call with more items is refused whole, with the
cap in the error, and nothing is queued. When a cap is set, `job_description` appends
"At most N items per job." Only `generators.h3` sets one (30).

### 7.3 Tools must fit the run's GPUs *(added 2026-10-08)*

Each GPU tool's config block has `min_total_gpu_gib`, an estimate of the memory it needs across
the run's cards: 96 (4 x 24 GB) for `captioner` and `generators.h3`, 48 (2 x 24 GB) for the
others. `ar run` (new and resumed) and `ar doctor` stop with one error naming every enabled tool
that needs more than the run's cards have, so a tool that cannot fit is the user's error at
start and never a failed job for the agent. A block without the key is not checked.

If a worker still finds the machine too small once a run is going (the H3 worker's memory
check), it exits with code 78. The kernel then alerts the user (`tool_does_not_fit`), asks the
run to stop as `ar stop` does, and fails the job: the agent cannot fix it, so the run does not go
on. Running out of memory while loading is an ordinary job failure: on a machine that passed the
checks it means another process holds GPU memory.

## 8. Tool descriptions

- A description does not name another generator or an optional tool. `generate_images` says its
  images can start or end a rollout, without listing rollout tools; rollout tools drop
  "e.g. a generate_images frame".
- `job_wait`, `data_ingest` and `annotate_camera` may be named: they are treated as always
  registered.
- A test fails if any registered tool's description contains another optional tool's name or
  another generator's name.

## 9. Testing and rollout

- Unit tests with the existing fake-worker pattern: H3 item checks, prompt builder, the scene-only caption,
  device map; LTX keyframes; item cap for each job tool; AlayaWorld without `variant`; Wan absent
  from the default tool list; the description test.
- Real smoke runs on GPUs 0–3 before enabling H3: one text-only single-turn item and one
  three-turn item with first and last frame through the real tool path, then `data_ingest` of
  the results; one LTX item with first and last keyframes.
- *(Amended 2026-10-08.)* The H3 smoke now renders three items (text-only, first-and-last,
  last keyframe only) and ingests each with segments the test writes from `turn_segments`. The
  Z-Image, Wan and H3 workers and WorldModel's two prompt-precache scripts run under
  `torch.inference_mode`; WorldModel's inference engine keeps `torch.no_grad` because training
  shares it.
- Docs: `configs/kernel.yaml`, README, and the weights table and `generators` row of the
  2026-09-17 design spec.
- Cleanup once implemented: delete `third_party/MiniMax-H3` (only its prompt guide was used).

## 10. Risks

- H3 timing was checked on one scene; the smoke run uses a different one, and agents are told
  to check frames before trusting a segment boundary. *(Amended 2026-10-08: later smoke clips
  were off by 1.3 s and by about 4 s, and a last keyframe unrelated to the first is reached by a
  cut in the final frames even with the no-cuts sentence; the published caption therefore
  carries no segments.)*
- The bridge sets a private diffusers attribute to load components locally; it is tied to
  diffusers 0.40.0, which the env pins.
- H3 is slow (about 20 minutes per clip); the cap and the description make the cost visible.
