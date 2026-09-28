
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
