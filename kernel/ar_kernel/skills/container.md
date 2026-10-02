---
name: container
description: Use when you need to know what the container provides: installed programs, internet, where files may be written, what the kernel mounts.
---

# The container

- Internet access is on. `pip install --user`, `curl`, `wget`, `git`, `unzip` and `7z` work.
- Installed: `pandas`, `pyarrow`, `numpy`, `opencv-python-headless`, `Pillow`, `ffmpeg`, `ffprobe`.
- `/workspace` is writable and is carried to a retry of the same phase. `/workspace/staging/` is where files for the kernel's data tools go: the kernel reads candidates only from there, and GPU jobs and downloads publish their results there.
- `/agent` is the agent's code: writable in `edit_self`, read-only in `improve_recipe`. In `edit_self`, `/code` is the parent's code, which is the code that is running, read-only.
- `/context/context.json` is the full context of this attempt. `/nodes/<node>/` holds what each finished node left behind, read-only (skill `node_files`).
- GPUs are reached only through the kernel's GPU tools. Each is a queued job: it returns a job id at once, and `job_wait` collects the result.
