"""Run WorldModel's standard-dataset checker for one clip root, per format.

Executed inside the `alayaworld` env:
    python check_clip.py --config <recipe> --root <clip root> --camera-motion moving|static
Writes a JSON report to stdout between the markers so conda banners cannot corrupt it.
"""
from __future__ import annotations
import argparse, json, sys

MARK = "===AR_JSON==="

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--camera-motion", required=True, choices=["moving", "static"])
    args = parser.parse_args()

    from alaya.config.loader import load_config
    from alaya.data.standard import DatasetSpec
    from alaya.data.standard_check import check_dataset, layout_from_config

    cfg = load_config(args.config)
    layout = layout_from_config(cfg)
    if args.camera_motion == "static":
        wanted = [("video_caption_static", None)]
    else:
        wanted = [("video_caption_camera", None),
                  ("video_timed_prompts_camera", "segment"),
                  ("video_timed_prompts_camera", "per_chunk")]
    out: dict[str, dict] = {}
    for fmt, mode in wanted:
        spec = DatasetSpec(name="probe", root=args.root, format=fmt, weight=1.0, prompt_mode=mode)
        report = check_dataset(spec, **layout)
        key = fmt if mode is None else f"{fmt}:{mode}"
        out[key] = {"ok": not report.errors, "errors": list(report.errors),
                    "warnings": list(report.warnings), "clips": report.clips,
                    "windows": report.windows}
    print(MARK + json.dumps(out) + MARK)
    return 0

if __name__ == "__main__":
    sys.exit(main())
