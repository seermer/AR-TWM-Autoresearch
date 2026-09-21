# Moving this tree to another path or machine

Run `ar doctor` after any move. It checks everything below and exits non-zero on
a real problem (`--strict` also fails on warnings).

## The contract

**The three repos must sit side by side**, because `configs/kernel.yaml` locates
the other two relatively:

```
<anywhere>/
  AutoResearcher/     <- kernel, configs, runs
  WorldModel/         <- training, rendering, data checks
  WBench/             <- benchmark and metrics
```

`paths.worldmodel: ../WorldModel`, `paths.wbench: ../WBench`, `paths.runs_dir: runs`.
Keep them relative. An absolute path here pins the tree to one machine, and
`ar doctor` reports it as a failure.

## What is path-independent already

- **Tracked source.** WorldModel has no hardcoded absolute paths; AutoResearcher
  has them only inside `tests/fixtures/real_successful_train.log`, where they are
  captured data and must stay verbatim. WBench's only hits are upstream
  leaderboard scripts (`tools/add_three_models_*.py`, `tools/swap_cosmos3_*.py`)
  that this project never invokes.
- **Caches.** The prompt-embedding cache is keyed by SHA-1 of the prompt text and
  the DA3 cache by case id — neither encodes a path, so both survive a move.
- **`.env`.** `WorldModel/.env` and `WBench/.env` are *relative* symlinks to
  `../AutoResearcher/.env`. Keep them relative; `ar doctor` fails an absolute one.
- **Generated run configs.** Written per run under `runs/`, regenerated each time.

## What needs doing on a new machine

1. **Create the conda environments** — `alayaworld`, `wbench-main`, `wbench-vp`,
   `autoresearcher`. Never use `base` or the system Python. `ar doctor` lists any
   that are missing.
2. **System tools on PATH:** `ffmpeg`, `ffprobe`, `conda`, `git`.
3. **Weights are not in git.** Re-download or copy:
   `WorldModel/weights/` (LTX-2.3 transformer, `alaya-world-ar`, VAE, Gemma),
   `WBench/weights/` (SAM2, DA3, MegaSAM, HPSv3, DINOv2 torch hub).
4. **Fill `AutoResearcher/.env`** if API-based metrics are wanted. Without
   `VLM_API_KEY` the six VLM metrics are skipped by design, and the score is the
   mean of whatever metrics the report does contain.
5. **Run `ar doctor`,** then the unit suites.

## The failure that motivated this

`WBench/weights/hub/torchhub` and `.../hub/checkpoints` were symlinks left
pointing at a previous checkout location (`/home/...`) after the tree moved to
`/mnt/biometrics`. `Path.exists()` follows symlinks and returns `False` for a
dangling one, so the setup code believed the links were missing, tried to create
them, and hit `FileExistsError` — on **every** case.

It then failed quietly three times over: `run_megasam.py`'s workers exited 0
after failing every case; its `main()` ignored worker exit codes; and WBench's
`main.py` ignored the tool's return code. The phase printed
`MegaSAM done in 0s`, the run continued, and the report came out **missing five
metrics** (`spatial_consistency`, `gated_spatial_consistency`,
`navigation_trajectory`, `navigation_accuracy`, `navigation_consistency`) — a
wrong result that looked like a clean one.

All four faults are fixed: links are repaired in place rather than recreated
blindly, and failures now propagate at each level. `ar doctor` checks for broken
symlinks anywhere in the three repos, because this is how a move breaks this tree.
