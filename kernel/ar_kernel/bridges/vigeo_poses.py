"""annotate_camera worker (runs in `alayaworld`): mp4 -> per-frame camera poses via ViGeo.

python vigeo_poses.py --items items.json --out DIR --rank R --world W \
    --repo <ViGeo dir> --checkpoint <ViGeo1.1 dir> --max-frames N

Bridge protocol (Plan 3): handles items with index % world == rank; per item writes
<out>/<index>.npz (cam_c2w [N,4,4] float32, OpenCV camera-to-world, frame 0 = identity;
intrinsics [3,3] in mp4 pixels) and then <out>/<index>.json.

ViGeo facts used (third_party/ViGeo): frames are resized to a patch-14 grid of ~1369 tokens
(vigeo.py `_resolve_model_size`); `pose_pred` is [T,3,4] camera-to-world (README); the focal
is normalized so that fx = fy = f * sqrt(w^2 + h^2) / 2 at the model resolution
(vigeo/utils.py `normalized_uv`); long clips run in chunk mode, 16 frames per call, carrying
`kv_caches` (README "Inference Modes"); `total_budget` bounds that cache (layers/attention.py
`eviction`), otherwise it grows ~170 MB per frame.

Two choices measured on WorldModel's example clips (Plan 3 Task 4 report): the cache budget is
3x WorldModel's 262144 (still fits a 24 GB card, peak ~18.3 GB; the smaller budget drifted to
27% ATE on one clip), and the focal comes from the first chunk only, which ViGeo sees with full
attention: in chunk mode the per-frame focal drifts as the cache is evicted (881 -> 1140 px
over one 450-frame clip whose true fx is 793).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch

CHUNK = 16            # ViGeo README default chunk size; clips of <= CHUNK frames run offline
NUM_TOKENS = 1369     # ViGeo infer() default
CACHE_BUDGET = 786432  # KV-cache tokens kept across chunks (see the module docstring)


def decode(path: str, model_size, max_frames: int):
    """All frames as uint8 RGB [N,h,w,3] at the model size (h, w) = model_size((W, H)), plus (W, H)."""
    frames, n, wh = [], 0, None
    with av.open(path) as container:
        for frame in container.decode(video=0):
            n += 1
            if n > max_frames:
                continue                   # keep counting so the error names the real length
            rgb = frame.to_ndarray(format="rgb24")
            if wh is None:
                wh = (rgb.shape[1], rgb.shape[0])
                h, w = model_size(wh)
            frames.append(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR))
    if n > max_frames:
        raise ValueError(f"{n} frames > max_frames ({max_frames})")
    if not frames:
        raise ValueError("no frames decoded")
    return np.stack(frames), wh


def annotate(model, recover_focal, path: str, max_frames: int, device):
    def model_size(wh):
        h, w, _, _ = model._resolve_model_size(wh[1], wh[0], NUM_TOKENS)
        return h, w

    frames, (W, H) = decode(path, model_size, max_frames)
    n, h, w = frames.shape[:3]
    mode = "offline" if n <= CHUNK else "chunk"
    poses, focal, kv = [], None, None
    with torch.inference_mode():
        for s in range(0, n, CHUNK):
            x = torch.from_numpy(frames[s:s + CHUNK]).to(device).permute(0, 3, 1, 2).float() / 255.0
            o = model.infer(x, mode=mode, chunk_size=CHUNK, num_tokens=NUM_TOKENS, total_budget=CACHE_BUDGET,
                            resize_output=False, kv_caches=kv)
            kv = o["kv_caches"]
            poses.append(o["pose_pred"].float().cpu())
            if focal is None:                                  # first chunk: median per-frame focal
                p = o["points_pred"].float()                   # [t,h,w,3] = (uv/f * z, z)
                xy = (p[..., :2] / p[..., 2:].clamp(min=1e-6)).permute(0, 3, 1, 2).unsqueeze(0)
                focal = float(recover_focal(xy)[0].median())
            del o, x
    del kv
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, :3, :] = torch.cat(poses).double().numpy()
    c2w = np.linalg.inv(c2w[0]) @ c2w
    f = focal * (w * w + h * h) ** 0.5 / 2                  # model-resolution pixels
    fx, fy, cx, cy = f * W / w, f * H / h, W / 2.0, H / 2.0
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return c2w.astype(np.float32), k, [fx, fy, cx, cy]


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--items", "--out", "--repo", "--checkpoint"):
        ap.add_argument(name, required=True)
    for name in ("--rank", "--world", "--max-frames"):
        ap.add_argument(name, type=int, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    mine = [i for i in json.loads(Path(args.items).read_text(encoding="utf-8"))
            if i["index"] % args.world == args.rank]
    if not mine:
        return 0
    sys.path.insert(0, args.repo)
    from vigeo import ViGeo
    from vigeo.utils import recover_focal_from_xy
    device = torch.device("cuda")
    model = ViGeo.from_pretrained(args.checkpoint).to(device).eval()
    for item in mine:
        index, t0 = item["index"], time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            c2w, k, intr = annotate(model, recover_focal_from_xy, item["video"], args.max_frames, device)
            np.savez(out / f"{index}.npz", cam_c2w=c2w, intrinsics=k)
            status = {"ok": True, "frames": len(c2w), "intrinsics": intr,
                      "seconds": round(time.monotonic() - t0, 2),
                      "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
        except Exception as exc:          # noqa: BLE001 -- a per-item error; move on (protocol)
            status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        torch.cuda.empty_cache()
        (out / f"{index}.json").write_text(json.dumps(status), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
