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
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from pongai.coach.llm import image_part, stroke_prompt
from pongai.config import ROOT
from pongai.data.dataset import FIT_VIDEOS, VAL_VIDEOS, VideoItem, strokes
from pongai.strokes import TARGETS
from pongai.vision.pose import PoseEstimator
from scripts.eval_vlm import VOCAB, stroke_strip

OTHER_SIDE = {"left": "right", "right": "left"}
MIRROR_LABEL = {"left_leaning": "right_leaning", "right_leaning": "left_leaning",
                "left_foot_lifted": "right_foot_lifted", "right_foot_lifted": "left_foot_lifted"}


def export_video(name: str, split: str, out, pose: PoseEstimator, jitters: tuple[int, ...],
                 mirrors: tuple[bool, ...] = (False,)) -> int:
    img_dir = out / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in strokes(name):
        for m in mirrors:
            labels = {t: getattr(e, t) for t in TARGETS}
            if m:
                labels = {t: MIRROR_LABEL.get(v, v) for t, v in labels.items()}
            side = OTHER_SIDE[e.side] if m else e.side
            for j in jitters:
                strip = stroke_strip(VideoItem(name).video_path, e.frame + j, e.side, pose, mirror=m)
                if strip is None:
                    continue
                url = image_part(strip, 1536)["image_url"]["url"]
                rel = f"images/{split}/{name}_{e.frame:06d}_{e.side}_{j:+d}{'_m' if m else ''}.jpg"
                (out / rel).write_bytes(base64.b64decode(url.split(",", 1)[1]))
                rows.append({"image": rel, "prompt": stroke_prompt(VOCAB, side), "answer": json.dumps(labels),
                             "video": name, "frame": e.frame, "side": side, "jitter": j, "mirror": m})
    with open(out / f"{split}_{name}.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=",".join(FIT_VIDEOS + VAL_VIDEOS))
    ap.add_argument("--jitter", type=int, default=2, help="train-only contact-frame shift (0 = off)")
    ap.add_argument("--mirror", action="store_true", help="add flipped train copies and a valmirror split")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "vlm_strokes"))
    args = ap.parse_args()
    out = Path(args.out)
    pose = PoseEstimator()
    for name in args.videos.split(","):
        if name not in FIT_VIDEOS + VAL_VIDEOS:
            raise SystemExit(f"{name}: only training videos (game_*) may be used")
        if name in VAL_VIDEOS:
            print(f"{name} -> val: {export_video(name, 'val', out, pose, (0,))} examples", flush=True)
            if args.mirror:
                n = export_video(name, "valmirror", out, pose, (0,), (True,))
                print(f"{name} -> valmirror: {n} examples", flush=True)
        else:
            jitters = (-args.jitter, 0, args.jitter) if args.jitter else (0,)
            mirrors = (False, True) if args.mirror else (False,)
            n = export_video(name, "train", out, pose, jitters, mirrors)
            print(f"{name} -> train: {n} examples", flush=True)


if __name__ == "__main__":
    main()
