---
name: container
description: Use when you need to know what the container provides: installed programs, internet, where files may be written, what the kernel mounts.
---

# The container

- Internet access is on. `pip install --user`, `curl`, `wget`, `git`, `unzip` and `7z` work.
- Installed: `torch`, `torchvision`, `torchaudio` (they run on the CPU here), `pandas`, `pyarrow`, `numpy`, `opencv-python-headless`, `Pillow`, `ffmpeg`, `ffprobe`, `gcc`.
- `/workspace` is writable and is carried to a retry of the same phase. `/workspace/staging/` is where files for the kernel's data tools go: the kernel reads candidates only from there, and GPU jobs and downloads publish their results there.
- `/agent` is the agent's code: writable in `edit_self`, read-only in `improve_recipe`. In `edit_self`, `/code` is the parent's code, which is the code that is running, read-only.
- `/context/context.json` is the full context of this attempt. `/nodes/<node>/` holds what each finished node left behind, read-only (skill `node_files`).
- GPUs are reached only through the kernel's GPU tools. Each is a queued job: it returns a job id at once. Jobs run one at a time, in the order they were submitted, so one job with every item is faster than several small ones.
- A finished job writes its full result to `/workspace/staging/results/<tool>-<job_id>.json`; `job_wait` returns that path and a summary. `data_ingest` does the same. Read these files with a script.
- A list argument of a kernel tool (paths, items, candidates, clip ids) may be the path of a file that holds the list, so a script can write it.
- `/workspace/staging` is a separate mount from the rest of `/workspace`: a hard link across the two fails, a copy or a move works. Kernel tools refuse symbolic links and files outside `/workspace`.
- Background processes: start with `nohup ... &`, list with `ps aux` or `pgrep -f`, stop with `pkill -f`.
