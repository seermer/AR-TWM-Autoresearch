---
name: ingest_candidates_and_container
description: Use when building data_ingest candidates (paths, the caption file, the provenance of a clip you derived), or when you need to know what the container provides.
---

# Ingest candidates and the container

- A candidate's `video`, `caption` and `pose` are absolute container paths under `/workspace/staging/`. `caption` is the path of a caption JSON file, not the text.
- Provenance of a downloaded file: the record hf_download returns. Provenance of a rollout: the one in its `candidate`.
- Provenance of a clip you derived (cropped, trimmed, re-captioned, joined): `{"kind": "derived", "from": [...], "transform": "<what you did>"}`. When its source is a clip already in the archive, also list that clip id in `derived_from`; the data summaries then show the clip under its source's origin.
- The container has internet access, `pip install --user`, `curl`, `wget`, `git`, `unzip` and `7z`. `pandas`, `pyarrow`, `numpy`, `opencv-python-headless`, `Pillow` and `ffmpeg` are installed.
