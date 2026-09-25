"""Build a YOLO detection dataset for the ball from the ground-truth ball positions.

    python -m training.build_ball_dataset --mode stack3 --stride 3

Only annotated frames are used. Frames annotated as "ball not visible" become negatives
(empty label files); unannotated frames are skipped because they may contain the ball.
game_1..4 -> train split, game_5 -> val split. Test videos are never used.

--mode rgb     ordinary colour frames
--mode stack3  grayscale (t-1, t, t+1) packed into the 3 channels so YOLO can see motion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pongai.config import REF_WIDTH, ROOT
from pongai.data.dataset import FIT_VIDEOS, VAL_VIDEOS, VideoItem, load_ball
from pongai.vision.ball import stack3


def export_video(name: str, split: str, out: Path, mode: str, stride: int, box_px: float, out_w: int) -> int:
    gt = load_ball(name)
    wanted = set(sorted(gt)[::stride])
    if not wanted:
        return 0
    cap = cv2.VideoCapture(str(VideoItem(name).video_path))
    last_needed = max(wanted) + 1
    img_dir, lbl_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    prev = cur = None
    written = 0
    idx = -1
    need_decode = wanted | {f - 1 for f in wanted} | {f + 1 for f in wanted}
    while idx < last_needed:
        idx += 1
        if idx in need_decode:
            ok, frame = cap.read()
        else:
            ok, frame = cap.grab(), None
        if not ok:
            break
        # emit frame idx-1 once its "next" frame (idx) is available
        f = idx - 1
        if f in wanted and cur is not None and frame is not None:
            p = prev if prev is not None else cur
            img = stack3(p, cur, frame) if mode == "stack3" else cur
            h, w = img.shape[:2]
            scale = out_w / w
            img = cv2.resize(img, (out_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)
            stem = f"{name}_{f:06d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            xy = gt[f]
            with open(lbl_dir / f"{stem}.txt", "w") as fh:
                if xy is not None:
                    bw = box_px * (w / REF_WIDTH) / w
                    bh = box_px * (w / REF_WIDTH) / h
                    fh.write(f"0 {xy[0] / w:.6f} {xy[1] / h:.6f} {bw:.6f} {bh:.6f}\n")
            written += 1
        prev, cur = cur, frame  # frames that were only grabbed are None
    cap.release()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rgb", "stack3"], default="stack3")
    ap.add_argument("--stride", type=int, default=3, help="use every Nth annotated frame")
    ap.add_argument("--box", type=float, default=20.0, help="box side in px at 1920 wide")
    ap.add_argument("--width", type=int, default=1280, help="saved image width")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or ROOT / "datasets" / f"ball_{args.mode}")

    total = 0
    for split, names in (("train", FIT_VIDEOS), ("val", VAL_VIDEOS)):
        for name in names:
            if not VideoItem(name).video_path.exists():
                print(f"[skip] {name}: video missing")
                continue
            n = export_video(name, split, out, args.mode, args.stride, args.box, args.width)
            print(f"{name} -> {split}: {n} images")
            total += n
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    (out / "meta.json").write_text(json.dumps(vars(args), indent=2))
    print(f"wrote {total} images to {out}")


if __name__ == "__main__":
    main()
