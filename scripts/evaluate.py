"""Evaluate the full CV stack against Extended OpenTTGames ground truth.

    python -m scripts.evaluate --videos test --tag yolo-stack3-v1
    python -m scripts.evaluate --videos test_2 --limit-frames 1200 --ball motion   # quick local smoke test

Every run is saved to results/runs/<time>_<tag>.json and appended to results/history.csv so
improvements (or regressions) are visible over time.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from pongai.config import RESULTS_DIR
from pongai.data.dataset import NO_BOUNCE_LABELS, VideoItem, load_ball, load_events, racket_truth, resolve
from pongai.evaluation import ball_counts, classification_report, match_events, prf, summarize_ball
from pongai.events import detect_events
from pongai.perception import perceive_cached
from pongai.strokes import TARGETS, StrokeClassifier, contexts_for_hits, heuristic_prediction, stroke_features
from pongai.vision.ball import load_ball_detector
from pongai.vision.pose import PoseEstimator

HISTORY_COLS = ["time", "tag", "videos", "ball_model", "pose_model", "stroke_model",
                "ball_recall@10px", "ball_median_err_px", "bounce_f1", "hit_right_f1", "hit_left_f1",
                "oracle_hand_acc", "oracle_technique_acc", "oracle_technique_macro_f1", "oracle_lean_acc",
                "oracle_feet_acc", "e2e_hand_acc", "e2e_technique_acc", "rally_coverage", "rally_winner_acc"]


def _throttled(prefix: str, step: float = 0.1):
    state = {"next": step}

    def cb(p: float):
        if p >= state["next"]:
            print(f"{prefix} {p:4.0%}", flush=True)
            state["next"] = (int(p / step) + 1) * step
    return cb


def classify(clf: StrokeClassifier, perc, contexts):
    feats = [stroke_features(perc.poses[c.side], perc.ball, c, perc.fps, perc.table, perc.width) for c in contexts]
    if clf.ready and feats:
        return clf.predict(np.stack(feats))
    return [heuristic_prediction(c, f) for c, f in zip(contexts, feats)]


def evaluate_video(name, perc, clf, side: str, tol_hit: int, tol_bounce: int,
                   racket: dict[str, str] | None = None) -> dict:
    """racket: known racket hand per side; None estimates it from wrist motion (as the app does)."""
    n = len(perc)
    start = perc.start
    gt_events = [e for e in load_events(name) if start <= e.frame < start + n]
    ev = detect_events(perc.ball, perc.width, perc.fps, perc.table)
    res = {"ball": ball_counts(perc.ball, load_ball(name), start, perc.width)}

    if name not in NO_BOUNCE_LABELS:
        pairs, fp, fn = match_events([b.frame + start for b in ev.bounces],
                                     [e.frame for e in gt_events if e.kind == "bounce"], tol_bounce)
        res["bounce"] = (len(pairs), fp, fn)

    gt_strokes = [e for e in gt_events if e.kind == "stroke"]
    for s in ("left", "right"):
        pairs, fp, fn = match_events([h.frame + start for h in ev.hits if h.side == s],
                                     [e.frame for e in gt_strokes if e.side == s], tol_hit)
        res[f"hit_{s}"] = (len(pairs), fp, fn)
        if s == side:
            res["hit_pairs"] = pairs

    y = defaultdict(lambda: {"true": [], "pred": []})
    # oracle: classify at GT contact frames (isolates the classifier from hit detection)
    gt_hits = [SimpleNamespace(frame=e.frame - start, side=e.side, serve=False) for e in gt_strokes]
    ctxs = contexts_for_hits(gt_hits, perc.poses, perc.fps, dict(racket) if racket else None)
    idx = [i for i, e in enumerate(gt_strokes) if e.side == side]
    preds = classify(clf, perc, [ctxs[i] for i in idx])
    for i, p in zip(idx, preds):
        for tgt in TARGETS:
            y[f"oracle_{tgt}"]["true"].append(getattr(gt_strokes[i], tgt))
            y[f"oracle_{tgt}"]["pred"].append(p[tgt])
    # end-to-end: classify detected hits that matched a GT stroke
    ctxs = contexts_for_hits(ev.hits, perc.poses, perc.fps, dict(racket) if racket else None)
    by_frame = {c.frame + start: c for c in ctxs}
    gt_by_frame = {e.frame: e for e in gt_strokes if e.side == side}
    pairs = res.pop("hit_pairs")
    preds = classify(clf, perc, [by_frame[p] for p, _ in pairs])
    for (_, g), p in zip(pairs, preds):
        for tgt in TARGETS:
            y[f"e2e_{tgt}"]["true"].append(getattr(gt_by_frame[g], tgt))
            y[f"e2e_{tgt}"]["pred"].append(p[tgt])
    res["cls"] = dict(y)

    # rallies: does a predicted rally cover each GT rally ending, and who won the point?
    ends = [e for e in gt_events if e.kind == "rally_end"]
    covered = correct = 0
    for e in ends:
        f = e.frame - start
        r = next((r for r in ev.rallies if r.start <= f <= r.end + 2 * perc.fps), None)
        if r is not None:
            covered += 1
            correct += r.point_winner == e.point_winner
    res["rally"] = (len(ends), covered, correct)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="test")
    ap.add_argument("--ball", default="auto", help="motion | coco | path to trained ball weights dir")
    ap.add_argument("--pose", default=None, help="YOLO pose weights (default: m on GPU, n on CPU)")
    ap.add_argument("--pose-stride", type=int, default=1)
    ap.add_argument("--stroke-model", default=None)
    ap.add_argument("--side", default="right")
    ap.add_argument("--limit-frames", type=int, default=0, help="only the first N frames (smoke tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--refresh", action="store_true", help="ignore cached perception")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--racket", default="estimate", choices=["estimate", "oracle"],
                    help="racket hand per player: wrist-motion estimate (as the app does) or the dataset README")
    args = ap.parse_args()

    ball_det = load_ball_detector(args.ball, device=args.device)
    pose_est = PoseEstimator(args.pose, device=args.device)
    clf = StrokeClassifier.load(args.stroke_model)
    videos = resolve(args.videos)

    agg = {"ball": defaultdict(lambda: 0), "bounce": np.zeros(3, int), "hit_left": np.zeros(3, int),
           "hit_right": np.zeros(3, int), "rally": np.zeros(3, int)}
    agg["ball"]["errors"] = []
    cls = defaultdict(lambda: {"true": [], "pred": []})
    per_video = {}
    for name in videos:
        item = VideoItem(name)
        if not item.video_path.exists():
            print(f"[skip] {name}: missing {item.video_path}")
            continue
        t0 = time.time()
        perc = perceive_cached(name, item.video_path, ball_det, pose_est, 0,
                               args.limit_frames or None, args.pose_stride, refresh=args.refresh,
                               progress=_throttled(f"  {name} perception"))
        r = evaluate_video(name, perc, clf, args.side, tol_hit=int(round(6 * perc.fps / 120)) or 1,
                           tol_bounce=int(round(4 * perc.fps / 120)) or 1,
                           racket=racket_truth(name) if args.racket == "oracle" else None)
        for k, v in r["ball"].items():
            agg["ball"][k] += v
        for k in ("bounce", "hit_left", "hit_right", "rally"):
            if k in r:
                agg[k] += np.array(r[k])
        for k, v in r["cls"].items():
            cls[k]["true"] += v["true"]
            cls[k]["pred"] += v["pred"]
        per_video[name] = {"ball": summarize_ball(r["ball"]), "hit_right": prf(*r["hit_right"]),
                           "hit_left": prf(*r["hit_left"]), "bounce": prf(*r["bounce"]) if "bounce" in r else None,
                           "rally": dict(zip(("n", "covered", "winner_correct"), map(int, r["rally"])))}
        print(f"\r  {name}: {len(perc)} frames in {time.time() - t0:.0f}s  "
              f"ball@10px={per_video[name]['ball']['recall@10px']:.3f}  "
              f"hitR F1={per_video[name]['hit_right']['f1']:.3f}")

    n_ends, covered, correct = agg["rally"]
    summary = {
        "ball": summarize_ball(agg["ball"]),
        "bounce": prf(*agg["bounce"]),
        "hit_right": prf(*agg["hit_right"]),
        "hit_left": prf(*agg["hit_left"]),
        "classification": {k: classification_report(v["true"], v["pred"]) for k, v in sorted(cls.items())},
        "rally": {"n_gt_endings": int(n_ends), "coverage": round(covered / max(n_ends, 1), 4),
                  "winner_accuracy": round(correct / max(covered, 1), 4)},
    }
    run = {"tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args),
           "components": {"ball": ball_det.name, "pose": pose_est.name, "stroke": clf.meta or "heuristic"},
           "summary": summary, "per_video": per_video}

    out_dir = RESULTS_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.tag}.json"
    out.write_text(json.dumps(run, indent=2, default=str))

    c = summary["classification"]
    row = {"time": run["time"], "tag": args.tag, "videos": args.videos, "ball_model": ball_det.name,
           "pose_model": pose_est.name, "stroke_model": f"n_train={clf.meta.get('n_train')}" if clf.ready else "heuristic",
           "ball_recall@10px": summary["ball"]["recall@10px"], "ball_median_err_px": summary["ball"]["median_err_px"],
           "bounce_f1": summary["bounce"]["f1"], "hit_right_f1": summary["hit_right"]["f1"],
           "hit_left_f1": summary["hit_left"]["f1"],
           "oracle_hand_acc": c.get("oracle_hand", {}).get("accuracy"),
           "oracle_technique_acc": c.get("oracle_technique", {}).get("accuracy"),
           "oracle_technique_macro_f1": c.get("oracle_technique", {}).get("macro_f1"),
           "oracle_lean_acc": c.get("oracle_lean", {}).get("accuracy"),
           "oracle_feet_acc": c.get("oracle_feet", {}).get("accuracy"),
           "e2e_hand_acc": c.get("e2e_hand", {}).get("accuracy"),
           "e2e_technique_acc": c.get("e2e_technique", {}).get("accuracy"),
           "rally_coverage": summary["rally"]["coverage"], "rally_winner_acc": summary["rally"]["winner_accuracy"]}
    hist = RESULTS_DIR / "history.csv"
    prev_rows = list(csv.DictReader(hist.open())) if hist.exists() else []
    with hist.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
        if not prev_rows:
            w.writeheader()
        w.writerow(row)

    print(json.dumps(summary, indent=2))
    same = [r for r in prev_rows if r["videos"] == args.videos]
    if same:
        last = same[-1]
        print(f"\nvs previous run on '{args.videos}' ({last['tag']}, {last['time']}):")
        for k in HISTORY_COLS[6:]:
            try:
                a, b = float(last[k]), float(row[k])
                print(f"  {k:28s} {a:8.4f} -> {b:8.4f}  ({b - a:+.4f})")
            except (TypeError, ValueError):
                pass
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
