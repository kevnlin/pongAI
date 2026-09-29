"""Build the supervised fine-tuning set for the stroke VLM (Qwen3-VL + LoRA).

    python -m training.build_vlm_dataset                    # game_1..4 -> train, game_5 -> val
    python -m training.build_vlm_dataset --videos game_2    # one video (run several in parallel)
    python -m training.build_vlm_dataset --mirror --out datasets/vlm_strokes_mirror

Each example is exactly what scripts/eval_vlm.py sends at test time: the 4-frame player strip
(same crop, skeleton, resize and JPEG encoding) and the same prompt, with the ground-truth labels
as the JSON answer. Both players are used. Train strokes are also exported with the contact frame
shifted by +-2 frames as augmentation; val strokes are exported once. Test videos are never used.

--mirror adds a left-right flipped copy of every train example. The training games have no
left-handed players; a mirrored right-hander looks like a left-hander on the other side of the
table, so the prompt side flips and left/right lean and feet labels swap (hand and technique keep).
The flipped val strokes are written as a separate "valmirror" split: a left-handed check that
does not touch the test videos.

--racket states the racket hand in the stroke prompt (every player in the training games is
right-handed per the dataset README, so "right", or "left" for mirrored copies) and adds one
"which hand holds the racket?" example per train stroke strip at the contact frame, so the same
adapter can estimate a player's handedness by voting over their strokes (scripts/eval_vlm.py).
That run showed the model reading handedness from the mirrored background instead of the body.

--player-only greys out everything but the player (pose-skeleton mask with discs around the
wrists for the racket), so the stroke has to be judged from the body; mirroring then leaves no
background cue either. Evaluate with scripts/eval_vlm.py --player-only.

--composite POOL_DIR pastes the segmented player onto a random person-free background from the
training games (training/build_bg_pool.py; random flip and colour jitter, 1 in 5 synthetic) in
every train strip, mirrored or not, so no background is tied to a label and the model has to use
the body. Val stays raw, and the model is evaluated on raw frames as it would be deployed.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
from pathlib import Path

import cv2
import numpy as np

from pongai.coach.llm import HAND_WORD, image_part, racket_prompt, stroke_prompt
from pongai.config import ROOT
from pongai.data.dataset import FIT_VIDEOS, VAL_VIDEOS, VideoItem, strokes
from pongai.strokes import TARGETS
from pongai.vision.pose import PoseEstimator
from scripts.eval_vlm import MIRROR_LABEL, OTHER_SIDE, VOCAB, stroke_strip


def random_background(pool: list[Path], rng: random.Random) -> np.ndarray:
    """A pool patch (random flip, brightness/contrast/hue jitter) or, 1 time in 5, a synthetic one."""
    if not pool or rng.random() < 0.2:
        bg = np.full((512, 512, 3), [rng.randrange(256) for _ in range(3)], np.float32)
        bg += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0, 25), bg.shape)
        return np.clip(bg, 0, 255).astype(np.uint8)
    bg = cv2.imread(str(rng.choice(pool)))
    if rng.random() < 0.5:
        bg = cv2.flip(bg, 1)
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-12, 12)) % 180
    bg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    bg = bg * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
    return np.clip(bg, 0, 255).astype(np.uint8)


def export_video(name: str, split: str, out, pose: PoseEstimator, jitters: tuple[int, ...],
                 mirrors: tuple[bool, ...] = (False,), racket: bool = False, player_only: bool = False,
                 pool: list[Path] | None = None, rng: random.Random | None = None) -> int:
    """pool: background patches to paste every strip onto (train split only; see --composite)."""
    img_dir = out / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in strokes(name):
        for m in mirrors:
            labels = {t: getattr(e, t) for t in TARGETS}
            if m:
                labels = {t: MIRROR_LABEL.get(v, v) for t, v in labels.items()}
            side = OTHER_SIDE[e.side] if m else e.side
            hand = ("L" if m else "R") if racket else None  # training games: right-handers only
            for j in jitters:
                bg = random_background(pool, rng) if pool is not None else None
                strip = stroke_strip(VideoItem(name).video_path, e.frame + j, e.side, pose, mirror=m,
                                     player_only=player_only, background=bg)
                if strip is None:
                    continue
                url = image_part(strip, 1536)["image_url"]["url"]
                rel = f"images/{split}/{name}_{e.frame:06d}_{e.side}_{j:+d}{'_m' if m else ''}.jpg"
                (out / rel).write_bytes(base64.b64decode(url.split(",", 1)[1]))
                meta = {"video": name, "frame": e.frame, "side": side, "jitter": j, "mirror": m}
                rows.append({"image": rel, "prompt": stroke_prompt(VOCAB, side, hand), "answer": json.dumps(labels),
                             **meta})
                if hand and split == "train" and j == 0:
                    rows.append({"image": rel, "prompt": racket_prompt(side),
                                 "answer": json.dumps({"racket_hand": HAND_WORD[hand]}), "task": "racket", **meta})
    with open(out / f"{split}_{name}.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=",".join(FIT_VIDEOS + VAL_VIDEOS))
    ap.add_argument("--jitter", type=int, default=2, help="train-only contact-frame shift (0 = off)")
    ap.add_argument("--mirror", action="store_true", help="add flipped train copies and a valmirror split")
    ap.add_argument("--racket", action="store_true", help="racket hand in the prompt + handedness examples")
    ap.add_argument("--player-only", action="store_true",
                    help="grey out everything but the player (no background cues; eval with the same flag)")
    ap.add_argument("--composite", default=None, metavar="POOL_DIR",
                    help="paste every train strip onto a random background from POOL_DIR (training/build_bg_pool.py); "
                         "val stays raw and evaluation uses raw frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "datasets" / "vlm_strokes"))
    args = ap.parse_args()
    out = Path(args.out)
    pose = PoseEstimator()
    for name in args.videos.split(","):
        if name not in FIT_VIDEOS + VAL_VIDEOS:
            raise SystemExit(f"{name}: only training videos (game_*) may be used")
        opts = {"racket": args.racket, "player_only": args.player_only}
        if name in VAL_VIDEOS:
            n = export_video(name, "val", out, pose, (0,), **opts)
            print(f"{name} -> val: {n} examples", flush=True)
            if args.mirror:
                n = export_video(name, "valmirror", out, pose, (0,), (True,), **opts)
                print(f"{name} -> valmirror: {n} examples", flush=True)
        else:
            jitters = (-args.jitter, 0, args.jitter) if args.jitter else (0,)
            mirrors = (False, True) if args.mirror else (False,)
            pool = sorted(Path(args.composite).glob("*.jpg")) if args.composite else None
            rng = random.Random(f"{args.seed}-{name}")
            n = export_video(name, "train", out, pose, jitters, mirrors, pool=pool, rng=rng, **opts)
            print(f"{name} -> train: {n} examples", flush=True)


if __name__ == "__main__":
    main()
