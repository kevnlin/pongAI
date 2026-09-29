"""Collect person-free background patches from the training games for background-swap augmentation.

    python -m training.build_bg_pool                 # ~120 patches per game -> datasets/bg_pool/*.jpg

Random frames of game_1..5 (never test videos), random square windows of 400-1000 px that do not
overlap any detected person (YOLO11-seg, box grown by 10 %). Used by training/build_vlm_dataset.py
--composite, which pastes the segmented player onto a random patch.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from pongai.config import ROOT
from pongai.data.dataset import TRAIN_VIDEOS, VideoItem
from scripts.eval_vlm import person_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=120)
    ap.add_argument("--out", default=str(ROOT / "datasets" / "bg_pool"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in TRAIN_VIDEOS:
        cap = cv2.VideoCapture(str(VideoItem(name).video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved = tries = 0
        while saved < args.per_video and tries < args.per_video * 6:
            tries += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, rng.randrange(total))
            ok, img = cap.read()
            if not ok:
                continue
            h, w = img.shape[:2]
            people = [b for b, _ in person_segments([img])[0]]
            for _ in range(4):
                s = rng.randint(400, min(1000, h))
                x, y = rng.randrange(w - s), rng.randrange(h - s)
                win = np.array([x, y, x + s, y + s], float)
                if any(win[0] < b[2] + 0.1 * (b[2] - b[0]) and b[0] - 0.1 * (b[2] - b[0]) < win[2]
                       and win[1] < b[3] + 0.1 * (b[3] - b[1]) and b[1] - 0.1 * (b[3] - b[1]) < win[3]
                       for b in people):
                    continue
                cv2.imwrite(str(out / f"{name}_{saved:04d}.jpg"), img[y:y + s, x:x + s], [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
                break
        cap.release()
        print(f"{name}: {saved} patches", flush=True)


if __name__ == "__main__":
    main()
