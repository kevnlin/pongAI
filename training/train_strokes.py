"""Train the stroke classifiers (hand / technique / lean / feet) on ground-truth stroke frames.

    python -m training.train_strokes                  # fit on game_1-4, report on game_5
    python -m training.train_strokes --final          # then refit on all 5 training games and save
    python -m training.train_strokes --oracle-ball    # use GT ball positions for the ball features

Perception (ball + pose) is only run on short windows around labelled strokes, and cached, so
this is fast to iterate on. Strokes from BOTH players are used (the left player is mirrored in
feature space), doubling the training data. Test videos are never touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pongai.config import WEIGHTS_DIR
from pongai.data.dataset import FIT_VIDEOS, TRAIN_VIDEOS, VAL_VIDEOS, VideoItem, gt_trajectory, racket_truth, strokes
from pongai.evaluation import classification_report
from pongai.perception import perceive_cached, table_for_video
from pongai.strokes import TARGETS, StrokeClassifier, StrokeContext, stroke_features, wrist_motion
from pongai.vision.ball import load_ball_detector
from pongai.vision.pose import PoseEstimator


def segments(frames: list[int], before: int, after: int) -> list[tuple[int, int]]:
    """Merge [f-before, f+after] windows that overlap into contiguous segments."""
    segs: list[list[int]] = []
    for f in sorted(frames):
        lo, hi = max(0, f - before), f + after
        if segs and lo <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], hi)
        else:
            segs.append([lo, hi])
    return [tuple(s) for s in segs]


def build(names, ball_det, pose_est, oracle_ball: bool, device=None, known_racket: bool = False):
    X, Y, meta = [], {t: [] for t in TARGETS}, []
    for name in names:
        item = VideoItem(name)
        if not item.video_path.exists():
            print(f"[skip] {name}: video missing")
            continue
        evs = strokes(name)
        table = table_for_video(item.video_path)
        percs = []
        for lo, hi in segments([e.frame for e in evs], before=60, after=40):
            p = perceive_cached(name, item.video_path, ball_det, pose_est, lo, hi, 1, table=table)
            if oracle_ball:
                p.ball = gt_trajectory(name, lo, len(p))
            percs.append(p)
        print(f"{name}: {len(evs)} strokes in {len(percs)} segments")

        # racket hand per side, aggregated over the whole video
        motion = {"left": {"R": 0.0, "L": 0.0}, "right": {"R": 0.0, "L": 0.0}}
        for p in percs:
            for side in motion:
                fr = [e.frame - p.start for e in evs if e.side == side and p.start <= e.frame < p.start + len(p)]
                for k, v in wrist_motion(p.poses[side], fr, p.fps).items():
                    motion[side][k] += v
        racket = {s: ("L" if m["L"] > m["R"] else "R") for s, m in motion.items()}
        if known_racket:
            racket = racket_truth(name)
        print(f"  racket hand: {racket}")

        prev = None
        for e in evs:
            p = next(p for p in percs if p.start <= e.frame < p.start + len(p))
            # timing only: using the GT technique here would leak the "serve" label into features
            first = prev is None or (e.frame - prev) > 2.5 * p.fps
            ctx = StrokeContext(e.frame - p.start, e.side, racket[e.side], first,
                                np.nan if first else (e.frame - prev) / p.fps)
            prev = e.frame
            X.append(stroke_features(p.poses[e.side], p.ball, ctx, p.fps, p.table, p.width))
            for t in TARGETS:
                Y[t].append(getattr(e, t))
            meta.append((name, e.frame, e.side))
    return (np.stack(X) if X else np.zeros((0, 0))), Y, meta


def report(clf: StrokeClassifier, X, Y, meta, side_filter: str | None = "right") -> dict:
    preds = clf.predict(X)
    out = {}
    for t in TARGETS:
        idx = [i for i, m in enumerate(meta) if side_filter is None or m[2] == side_filter]
        out[t] = classification_report([Y[t][i] for i in idx], [preds[i][t] for i in idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ball", default="auto")
    ap.add_argument("--pose", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--oracle-ball", action="store_true")
    ap.add_argument("--known-racket", action="store_true",
                    help="use each player's true racket hand (dataset README) instead of the wrist-motion estimate")
    ap.add_argument("--final", action="store_true", help="refit on all training games before saving")
    ap.add_argument("--out", default=str(WEIGHTS_DIR / "stroke" / "stroke_clf.joblib"))
    args = ap.parse_args()

    ball_det = load_ball_detector(args.ball, device=args.device)
    pose_est = PoseEstimator(args.pose, device=args.device)

    Xf, Yf, Mf = build(FIT_VIDEOS, ball_det, pose_est, args.oracle_ball, known_racket=args.known_racket)
    Xv, Yv, Mv = build(VAL_VIDEOS, ball_det, pose_est, args.oracle_ball, known_racket=args.known_racket)
    print(f"fit: {len(Xf)} strokes, val: {len(Xv)} strokes, {Xf.shape[1]} features")
    clf = StrokeClassifier().fit(Xf, Yf)
    if len(Xv):
        print("validation on", VAL_VIDEOS, "(right player):")
        print(json.dumps(report(clf, Xv, Yv, Mv), indent=2))
        print("validation (both players):")
        print(json.dumps(report(clf, Xv, Yv, Mv, None), indent=2))

    if args.final and len(Xv):
        X = np.concatenate([Xf, Xv])
        Y = {t: Yf[t] + Yv[t] for t in TARGETS}
        clf = StrokeClassifier().fit(X, Y)
        print(f"refit on all {len(X)} training strokes ({TRAIN_VIDEOS})")
    clf.meta.update({"ball": ball_det.name, "pose": pose_est.name, "oracle_ball": args.oracle_ball,
                     "known_racket": args.known_racket})
    clf.save(Path(args.out))
    print(f"saved {args.out}; evaluate on test videos with: python -m scripts.evaluate --videos test")


if __name__ == "__main__":
    main()
