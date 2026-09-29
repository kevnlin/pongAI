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
from pongai.data.dataset import VideoItem, racket_truth, resolve, strokes
from pongai.data.labels import FEET, HANDS, LEANS, TECHNIQUES
from pongai.evaluation import classification_report
from pongai.perception import Perception, cache_path, perceive_cached, table_for_video
from pongai.render import draw_skeleton
from pongai.strokes import TARGETS, wrist_motion
from pongai.vision.ball import load_ball_detector
from pongai.vision.pose import SKELETON, PoseEstimator, select_players
from pongai.vision.table import detect_table
from training.train_strokes import segments

VOCAB = {"hand": list(HANDS), "technique": list(TECHNIQUES), "lean": list(LEANS), "feet": list(FEET)}
# A left-right mirrored strip shows the player on the other side of the table with left/right swapped.
OTHER_SIDE = {"left": "right", "right": "left"}
MIRROR_LABEL = {"left_leaning": "right_leaning", "right_leaning": "left_leaning",
                "left_foot_lifted": "right_foot_lifted", "right_foot_lifted": "left_foot_lifted"}
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


_SEG = {}


def person_segments(frames: list[np.ndarray]) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """(box, polygon) of every person per frame, from YOLO11-seg (COCO class 0)."""
    from ultralytics import YOLO

    if "model" not in _SEG:
        _SEG["model"] = YOLO("yolo11m-seg.pt")
    out = []
    for r in _SEG["model"].predict(frames, imgsz=960, conf=0.3, classes=[0], verbose=False):
        polys = r.masks.xy if r.masks is not None else []
        out.append(list(zip(r.boxes.xyxy.cpu().numpy(), polys)))
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def skeleton_mask(kpts: np.ndarray, shape: tuple[int, int], box_h: float, min_conf: float = 0.3) -> np.ndarray:
    """Fallback body mask from pose keypoints (thick limbs, torso, head) when segmentation misses the player."""
    mask = np.zeros(shape, np.uint8)
    ok = kpts[:, 2] >= min_conf
    pt = lambda i: tuple(int(v) for v in kpts[i, :2])  # noqa: E731
    limb = max(6, int(0.08 * box_h))
    for a, b in SKELETON:
        if ok[a] and ok[b]:
            cv2.line(mask, pt(a), pt(b), 255, limb)
    torso = [5, 6, 12, 11]
    if ok[torso].all():
        cv2.fillPoly(mask, [kpts[torso, :2].astype(np.int32)], 255)
    face = [i for i in range(5) if ok[i]]
    if face:
        cv2.circle(mask, tuple(kpts[face, :2].mean(0).astype(int)), int(0.10 * box_h), 255, -1)
    return mask


def player_masks(shape: tuple[int, int], person, segments, racket_r: float,
                 min_conf: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """(body, racket) uint8 masks: the player's segmentation outline (skeleton mask as a fallback) and
    discs of racket_r x player height around both wrists (handedness is not known)."""
    h = person.box[3] - person.box[1]
    best = max(segments, key=lambda s: _iou(s[0], person.box), default=None)
    if best is not None and _iou(best[0], person.box) > 0.5 and len(best[1]) >= 3:
        body = np.zeros(shape, np.uint8)
        cv2.fillPoly(body, [best[1].astype(np.int32)], 255)
        body = cv2.dilate(body, np.ones((5, 5), np.uint8))
    else:
        body = skeleton_mask(person.kpts, shape, h, min_conf)
    racket = np.zeros_like(body)
    for i in (9, 10):
        if person.kpts[i, 2] >= min_conf:
            cv2.circle(racket, tuple(int(v) for v in person.kpts[i, :2]), int(racket_r * h), 255, -1)
    return body, racket


def composite_frame(img: np.ndarray, person, segments, crop: tuple[int, int, int, int],
                    background: np.ndarray) -> np.ndarray:
    """Paste the player onto `background`, resized to the crop window (only that window changes).
    The segmented body is pasted sharp; a small disc per wrist keeps the racket, with its non-body
    pixels lightly blurred so leftover background text there is unreadable; edges are feathered."""
    x1, y1, x2, y2 = crop
    h = person.box[3] - person.box[1]
    body, racket = player_masks(img.shape[:2], person, segments, racket_r=0.10)
    layer = img.copy()
    around = (racket > 0) & (body == 0)
    layer[around] = cv2.GaussianBlur(img, (0, 0), sigmaX=max(1.5, 0.012 * h))[around]
    alpha = cv2.GaussianBlur(np.maximum(body, racket)[y1:y2, x1:x2].astype(np.float32) / 255, (0, 0), 2)[..., None]
    bg = cv2.resize(background, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA).astype(np.float32)
    out = img.copy()
    out[y1:y2, x1:x2] = (alpha * layer[y1:y2, x1:x2] + (1 - alpha) * bg).astype(np.uint8)
    return out


def player_only_frame(img: np.ndarray, person, segments, min_conf: float = 0.3) -> np.ndarray:
    """Grey out everything but the player. Player pixels (segmentation outline, or the skeleton mask
    as a fallback) stay sharp; a disc around each wrist is kept heavily blurred so the racket stays
    visible as a blob while background text becomes unreadable; the rest is flat grey."""
    h = person.box[3] - person.box[1]
    body, racket = player_masks(img.shape[:2], person, segments, racket_r=0.18, min_conf=min_conf)
    out = np.full_like(img, 127)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=max(3.0, 0.02 * h))
    out[racket > 0] = blurred[racket > 0]
    out[body > 0] = img[body > 0]
    return out


def stroke_strip(video_path, frame: int, side: str, pose: PoseEstimator, height: int = 420,
                 mirror: bool = False, player_only: bool = False,
                 background: np.ndarray | None = None) -> np.ndarray | None:
    """mirror=True flips each crop left-right (frame order kept); used for training augmentation.
    player_only=True greys out everything but the player (player_only_frame), so no background cue is left.
    background: paste the player onto this image in every frame (composite_frame; training augmentation)."""
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
    need_seg = player_only or background is not None
    segs = person_segments([frames[o] for o in offs]) if need_seg else [None] * len(offs)
    crops, prev = [], None
    for o, ppl, seg in zip(offs, people, segs):
        p = select_players(ppl, frames[o].shape[1], table.net_x if table else None, prev)[side]
        if p is None:
            return None
        prev = {side: np.array([(p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2])}
        img = frames[o].copy()
        x1, y1, x2, y2 = p.box
        w, h = x2 - x1, y2 - y1
        x1, x2 = int(max(0, x1 - 0.5 * w)), int(min(img.shape[1], x2 + 0.5 * w))
        y1, y2 = int(max(0, y1 - 0.15 * h)), int(min(img.shape[0], y2 + 0.1 * h))
        if background is not None:
            img = composite_frame(img, p, seg, (x1, y1, x2, y2), background)
        if player_only:
            img = player_only_frame(img, p, seg)
        draw_skeleton(img, p.kpts, (80, 200, 120), 3)
        c = img[y1:y2, x1:x2]
        if mirror:
            c = cv2.flip(c, 1)
        crops.append(cv2.resize(c, (int(c.shape[1] * height / c.shape[0]), height)))
    return np.hstack(crops)


def oracle_racket(name: str, side: str) -> str:
    return racket_truth(name)[side]


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
    ap.add_argument("--player-only", action="store_true", help="grey out everything but the player")
    ap.add_argument("--mirror-check", action="store_true",
                    help="also classify each strip mirrored and report how often the answers agree once "
                         "left/right labels are swapped (a body-only model is mirror-consistent)")
    ap.add_argument("--shift-check", action="store_true",
                    help="also classify each stroke one frame later and report agreement: the noise floor "
                         "that --mirror-check should be read against")
    args = ap.parse_args()

    items = [(n, e) for n in resolve(args.videos) if VideoItem(n).video_path.exists() for e in strokes(n, args.side)]
    random.Random(args.seed).shuffle(items)
    items = items[:args.max]
    pose = PoseEstimator()
    strips = []
    for k, (name, e) in enumerate(items, 1):
        strip = stroke_strip(VideoItem(name).video_path, e.frame, args.side, pose, player_only=args.player_only)
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
        if args.mirror_check:
            m = stroke_strip(VideoItem(name).video_path, e.frame, args.side, pose, mirror=True,
                             player_only=args.player_only)
            if m is not None:
                mp = classify_stroke(m, VOCAB, OTHER_SIDE[args.side])
                rows[-1]["pred_mirror"] = {t: MIRROR_LABEL.get(str(v).lower(), str(v).lower()) for t, v in mp.items()}
        if args.shift_check:
            s1 = stroke_strip(VideoItem(name).video_path, e.frame + 1, args.side, pose, player_only=args.player_only)
            if s1 is not None:
                rows[-1]["pred_shift"] = {t: str(v).lower() for t, v in
                                          classify_stroke(s1, VOCAB, args.side, racket.get(name)).items()}
        print(f"[{k}/{len(strips)}] {name}@{e.frame}: gt={e.hand}/{e.technique} pred={pred.get('hand')}/{pred.get('technique')}")

    summary = {t: classification_report(v["true"], v["pred"]) for t, v in y.items()}
    for key, out_key in (("pred_mirror", "mirror_consistency"), ("pred_shift", "shift_consistency")):
        checked = [r for r in rows if key in r]
        if checked:
            summary[out_key] = {
                t: round(sum(str(r["pred"].get(t, "")).lower() == r[key].get(t) for r in checked) / len(checked), 4)
                for t in TARGETS} | {"n": len(checked)}
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
