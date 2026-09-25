"""End-to-end analysis of one uploaded video for one player.

    python -m pongai.pipeline path/to/video.mp4 --out out_dir [--side right] [--no-llm]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from pongai.coach.llm import coach_feedback, llm_available, model_name
from pongai.coach.rules import coaching_tips
from pongai.events import detect_events
from pongai.perception import perceive
from pongai.render import render_overlay
from pongai.report import build_report
from pongai.strokes import StrokeClassifier, contexts_for_hits, heuristic_prediction, stroke_features
from pongai.vision.ball import load_ball_detector
from pongai.vision.pose import PoseEstimator
from pongai.vision.table import Table, order_corners

_MODELS: dict = {}


def _models(ball: str, pose: str | None, device: str | None):
    """Load models once per process (the web server reuses them across jobs)."""
    key = (ball, pose, device)
    if key not in _MODELS:
        _MODELS[key] = (load_ball_detector(ball, device=device), PoseEstimator(pose, device=device),
                        StrokeClassifier.load())
    return _MODELS[key]


def analyze_video(video_path: str | Path, out_dir: str | Path, side: str = "right",
                  corners: list | None = None, ball: str = "auto", pose: str | None = None,
                  pose_stride: int | None = None, device: str | None = None, use_llm: bool = True,
                  max_frames: int | None = None,
                  progress: Callable[[str, float], None] = lambda stage, p: None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    progress("loading models", 0.0)
    ball_det, pose_est, clf = _models(ball, pose, device)

    import cv2
    fps = cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS) or 30.0
    stride = pose_stride or (2 if fps > 60 else 1)
    table = Table(order_corners(np.array(corners, dtype=np.float32)), source="manual") if corners else None

    perc = perceive(video_path, ball_det, pose_est, 0, max_frames, stride, table=table,
                    progress=lambda p: progress("tracking ball & players", p))
    progress("detecting strokes", 0.0)
    ev = detect_events(perc.ball, perc.width, perc.fps, perc.table)
    ctxs = [c for c in contexts_for_hits(ev.hits, perc.poses, perc.fps) if c.side == side]
    feats = [stroke_features(perc.poses[side], perc.ball, c, perc.fps, perc.table, perc.width) for c in ctxs]
    if clf.ready and feats:
        preds = clf.predict(np.stack(feats))
    else:
        preds = [heuristic_prediction(c, f) for c, f in zip(ctxs, feats)]

    models = {"ball": ball_det.name, "pose": pose_est.name,
              "stroke": f"trained (n={clf.meta.get('n_train')})" if clf.ready else "heuristic (untrained)"}
    report = build_report(perc, ev, side, preds, models)

    keyframes = render_overlay(video_path, out_dir / "overlay.mp4", perc, report, side,
                               progress=lambda p: progress("rendering overlay", p))
    report["tips"] = coaching_tips(report)
    report["coach"] = None
    if use_llm and llm_available():
        progress("writing coaching feedback", 0.0)
        try:
            report["coach"] = coach_feedback(report, report["tips"], keyframes)
            report["models"]["coach"] = model_name()
        except Exception as e:  # network / quota problems must not lose the CV results
            report["coach_error"] = str(e)
    report["processing_s"] = round(time.time() - t0, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    progress("done", 1.0)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="out")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--ball", default="auto")
    ap.add_argument("--pose", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    last = {"stage": None}

    def progress(stage, p):
        if stage != last["stage"]:
            print(f"\n{stage}", end="", flush=True)
            last["stage"] = stage
        print(".", end="", flush=True) if p < 1 else None

    r = analyze_video(args.video, args.out, args.side, ball=args.ball, pose=args.pose, device=args.device,
                      use_llm=not args.no_llm, max_frames=args.max_frames, progress=progress)
    print("\n" + json.dumps({k: r[k] for k in ("totals", "speed", "technique_counts", "tips")}, indent=2))
    print(f"wrote {args.out}/report.json and {args.out}/overlay.mp4")


if __name__ == "__main__":
    main()
