## Environments

Everything the project runs lives inside the project: code, weights, caches (`.cache/`) and
every conda environment, as a prefix env under `AutoResearcher/.envs/<name>`:
`autoresearcher` (kernel, `ar`), `alayaworld`, `wbench-main`, `wbench-vp`, `zhantaoy-vllm`,
`gen-zimage`, `gen-wan22`, `gen-ltx25`, `panel`.

    .envs/autoresearcher/bin/ar run --run-id <run> ...        # or: conda activate $PWD/.envs/autoresearcher
    .envs/autoresearcher/bin/python -m pytest tests -q

Config still says `env: alayaworld`; `subproc.conda_command` runs `.envs/alayaworld` when it
exists (else a named conda env). `ar doctor` reports any env that is not in `.envs/`.
Rebuilding on another machine: `docs/PORTABILITY.md`.


## Run panel

A read-only Gradio panel for one run: every event, LLM conversation (stitched across
compaction), tool call, code edit, training clip, training curve, eval video and score.
Design: `docs/superpowers/specs/2026-09-27-run-panel-design.md`.

    scripts/make_panel_env.sh                    # once: creates .envs/panel (Gradio 6)
    # set PANEL_USER and PANEL_PASSWORD in AutoResearcher/.env
    .envs/panel/bin/python -m panel --run-id <run>          # prints a public share link (login required)
    .envs/panel/bin/python -m panel --run-id <run> --no-share --port 7861   # local only
    .envs/panel/bin/python -m panel --run-id <run> --no-share --host 0.0.0.0   # LAN, still behind the login

It never writes to the run. Gradio's file cache goes to `.cache/panel/`.
