# Reference: released AlayaWorld on the 40-case WBench proxy

> This reference describes the original 40 cases; the proxy is now 50 (a superset, see spec §11.1).

Produced 2026-09-13 with the released v1.1 checkpoint (stage2b + stage3 DMD student),
`configs/wbench_full.yaml`, seeded per-case rendering, on 5x RTX 4090.

- `report.json` — 16 GPU metrics (no VLM key, no visual_plausibility at the time)
- `proxy_subset_ids.txt` / `proxy_subset_report.txt` — the stratified 40-case subset and its coverage

Kept as the fixture for kernel scoring tests and as the baseline the root node must
reproduce (see the implementation plan, Tasks 12 and 14). The original
`WBench/work_dirs/alayaworld/` tree was deleted afterwards so evaluation starts fresh.
