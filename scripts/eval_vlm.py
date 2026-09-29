"""Zero-shot evaluation of a vision-language model on ground-truth test strokes.

For each sampled stroke the VLM sees a strip of 4 frames around contact (player crop with the
pose skeleton drawn) and must output hand / technique / lean / feet. Accuracy is compared to the
Extended OpenTTGames labels, so the VLM can be benchmarked against the trained classifier.

    export OPENAI_API_KEY=...            # or PONGAI_LLM_BASE_URL for a local Qwen2.5-VL server
    python -m scripts.eval_vlm --videos test --max 60 --tag gpt-4o-mini
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter

import cv2
import numpy as np

from pongai.coach.llm import classify_racket_hand, classify_stroke, model_name
from pongai.config import RESULTS_DIR
from pongai.data.dataset import VideoItem, resolve, strokes
from pongai.data.labels import FEET, HANDS, LEANS, TECHNIQUES
from pongai.evaluation import classification_report
from pongai.perception import Perception, cache_path, perceive_cached, table_for_video
from pongai.render import draw_skeleton
from pongai.strokes import TARGETS, wrist_motion
from pongai.vision.ball import load_ball_detector
from pongai.vision.pose import PoseEstimator, select_players
from pongai.vision.table import detect_table
from training.train_strokes import segments

VOCAB = {"hand": list(HANDS), "technique": list(TECHNIQUES), "lean": list(LEANS), "feet": list(FEET)}
# Dataset README: test_3 and test_7 have one left-handed player, test_5 two; every other player is
# right-handed. Which test_3 / test_7 player it is was read from the video frames.
LEFT_HANDED = {("test_3", "right"), ("test_5", "left"), ("test_5", "right"), ("test_7", "right")}
OFFSETS = (-12, -6, 0, 4)  # frames at 120 fps


def video_racket_hands(name: str, ball_det, pose_est) -> dict[str, str]:
    """Racket hand ("R"/"L") of each player, from wrist travel around the labelled contacts, summed
    over the whole video exactly as training/train_strokes.py does. Uses the whole-video perception
    cached by scripts/evaluate.py when it exists, else the per-stroke segments (cached by train_strokes)."""
    item, evs = VideoItem(name), strokes(name)
    whole = cache_path(name, ball_det, pose_est, 0, None, 1)
    if whole.exists():
        percs = [Perception.load(whole)]
    else:
        table = table_for_video(item.video_path)
        percs = [perceive_cached(name, item.video_path, ball_det, pose_est, lo, hi, 1, table=table)
                 for lo, hi in segments([e.frame for e in evs], before=60, after=40)]
    motion = {"left": {"R": 0.0, "L": 0.0}, "right": {"R": 0.0, "L": 0.0}}
    for p in percs:
        for side in motion:
            fr = [e.frame - p.start for e in evs if e.side == side and p.start <= e.frame < p.start + len(p)]
            for k, v in wrist_motion(p.poses[side], fr, p.fps).items():
                motion[side][k] += v
    return {s: ("L" if m["L"] > m["R"] else "R") for s, m in motion.items()}


def stroke_strip(video_path, frame: int, side: str, pose: PoseEstimator, height: int = 420,
                 mirror: bool = False) -> np.ndarray | None:
    """mirror=True flips each crop left-right (frame order kept); used for training augmentation."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 120
    offs = [int(round(o * fps / 120)) for o in OFFSETS]
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame + offs[0]))
    frames = {}
    for i in range(offs[0], offs[-1] + 1):
        ok, img = cap.read()
        if not ok:
            break
        if i in offs:
            frames[i] = img
    cap.release()
    if len(frames) < len(offs):
        return None
    table = detect_table(frames[0])
    people = pose([frames[o] for o in offs])
    crops, prev = [], None
    for o, ppl in zip(offs, people):
        p = select_players(ppl, frames[o].shape[1], table.net_x if table else None, prev)[side]
        if p is None:
            return None
        prev = {side: np.array([(p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2])}
        img = frames[o].copy()
        draw_skeleton(img, p.kpts, (80, 200, 120), 3)
        x1, y1, x2, y2 = p.box
        w, h = x2 - x1, y2 - y1
        x1, x2 = int(max(0, x1 - 0.5 * w)), int(min(img.shape[1], x2 + 0.5 * w))
        y1, y2 = int(max(0, y1 - 0.15 * h)), int(min(img.shape[0], y2 + 0.1 * h))
        c = img[y1:y2, x1:x2]
        if mirror:
            c = cv2.flip(c, 1)
        crops.append(cv2.resize(c, (int(c.shape[1] * height / c.shape[0]), height)))
    return np.hstack(crops)


def oracle_racket(name: str, side: str) -> str:
    return "L" if (name, side) in LEFT_HANDED else "R"


def racket_hands(source: str, strips: list, side: str, pose: PoseEstimator) -> dict[str, str]:
    """Racket hand per video for the evaluated player (empty for source "none")."""
    names = sorted({name for name, _, _ in strips})
    if source == "oracle":
        return {n: oracle_racket(n, side) for n in names}
    if source == "wrist":
        ball_det = load_ball_detector("auto")
        return {n: video_racket_hands(n, ball_det, pose)[side] for n in names}
    if source == "vote":
        votes = {n: Counter() for n in names}
        for name, _, strip in strips:
            votes[name][classify_racket_hand(strip, side)] += 1
        return {n: ("L" if v["L"] > v["R"] else "R") for n, v in votes.items()}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="test")
    ap.add_argument("--side", default="right")
    ap.add_argument("--max", type=int, default=60, help="max strokes (API cost control)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--racket", default="none", choices=["none", "wrist", "vote", "oracle"],
                    help="state the racket hand in the prompt: wrist-motion estimate, per-player vote of the "
                         "model's own per-stroke answers, or the dataset README facts (upper bound)")
    args = ap.parse_args()

    items = [(n, e) for n in resolve(args.videos) if VideoItem(n).video_path.exists() for e in strokes(n, args.side)]
    random.Random(args.seed).shuffle(items)
    items = items[:args.max]
    pose = PoseEstimator()
    strips = []
    for k, (name, e) in enumerate(items, 1):
        strip = stroke_strip(VideoItem(name).video_path, e.frame, args.side, pose)
        if strip is None:
            print(f"[{k}/{len(items)}] {name}@{e.frame}: player not found, skipped")
            continue
        strips.append((name, e, strip))

    racket = racket_hands(args.racket, strips, args.side, pose)
    if racket:
        wrong = [n for n, h in racket.items() if h != oracle_racket(n, args.side)]
        print(f"racket hand ({args.racket}): {racket}; wrong vs README for {wrong or 'none'}")
    y = {t: {"true": [], "pred": []} for t in TARGETS}
    rows = []
    for k, (name, e, strip) in enumerate(strips, 1):
        pred = classify_stroke(strip, VOCAB, args.side, racket.get(name))
        for t in TARGETS:
            y[t]["true"].append(getattr(e, t))
            y[t]["pred"].append(str(pred.get(t, "unknown")).lower().strip())
        rows.append({"video": name, "frame": e.frame, "gt": e.raw, "pred": pred, "racket": racket.get(name)})
        print(f"[{k}/{len(strips)}] {name}@{e.frame}: gt={e.hand}/{e.technique} pred={pred.get('hand')}/{pred.get('technique')}")

    summary = {t: classification_report(v["true"], v["pred"]) for t, v in y.items()}
    tag = args.tag or model_name()
    out_dir = RESULTS_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_vlm_{tag.replace('/', '_')}.json"
    out.write_text(json.dumps({"model": model_name(), "args": vars(args), "racket": racket, "summary": summary,
                               "rows": rows}, indent=2))
    hist = RESULTS_DIR / "history_vlm.csv"
    new = not hist.exists()
    with hist.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "tag", "model", "videos", "n"] + [f"{t}_acc" for t in TARGETS] + [f"{t}_macro_f1" for t in TARGETS])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), tag, model_name(), args.videos, summary["hand"].get("n", 0)]
                   + [summary[t].get("accuracy") for t in TARGETS] + [summary[t].get("macro_f1") for t in TARGETS])
    print(json.dumps(summary, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
