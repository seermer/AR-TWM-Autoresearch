"""Run WorldModel's standard-dataset checker for one clip root.

Executed inside the `alayaworld` env:
    python check_clip.py --config <recipe> --root <clip root>
Writes a JSON report to stdout between the markers so conda banners cannot corrupt it.
"""
from __future__ import annotations
import argparse, json, sys

MARK = "===AR_JSON==="

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    from alaya.config.loader import load_config
    from alaya.data.standard import DatasetSpec
    from alaya.data.standard_check import check_dataset, layout_from_config

    spec = DatasetSpec(name="probe", root=args.root, format="video_timed_prompts_camera", weight=1.0)
    report = check_dataset(spec, **layout_from_config(load_config(args.config)))
    print(MARK + json.dumps({"ok": not report.errors, "errors": list(report.errors),
                             "warnings": list(report.warnings)}) + MARK)
    return 0

if __name__ == "__main__":
    sys.exit(main())
