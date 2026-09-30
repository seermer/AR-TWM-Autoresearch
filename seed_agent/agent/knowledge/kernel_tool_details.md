# Kernel tool details

- hf_download puts files under `/workspace/staging/hf/` and returns a ready provenance record. For a clip you derive (cropped, trimmed, re-captioned), use `{"kind": "derived", "from": [clip_ids], "transform": "<what you did>"}` and set `derived_from`.
- hf_search matches only when every word of the query is in the dataset id or tags. hf_list_files reports `accessible`; a repo where it is false cannot be downloaded.
- data_ingest takes candidates under `/workspace/staging/` and moves each staged file into the archive, so copy a file first if you still need it. Candidate paths are absolute container paths, and `caption` is the path of a caption JSON file, not the text.
- `prompt_mode` in data_commit applies only to `video_timed_prompts_camera`.
- job_wait waits at most 300 s per call; call it again until the job is done.
- recipe_check takes a flat `{key: value}` map of tunable keys.
- The container has internet access, `pip install --user`, `curl`, `wget`, `git`, `unzip` and `7z`.
