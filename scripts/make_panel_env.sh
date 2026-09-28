#!/usr/bin/env bash
# Creates AutoResearcher/.envs/panel for the run panel
# (docs/superpowers/specs/2026-09-27-run-panel-design.md). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
conda create -y -p .envs/panel python=3.12
conda run --no-capture-output -p .envs/panel pip install -r panel/requirements.txt
conda run --no-capture-output -p .envs/panel pip freeze > panel/requirements.lock.txt
