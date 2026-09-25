"""Metrics against Extended OpenTTGames ground truth."""
from __future__ import annotations

import numpy as np

from pongai.config import REF_WIDTH


def match_events(pred: list[int], gt: list[int], tol: int) -> tuple[list[tuple[int, int]], int, int]:
    """Greedy one-to-one matching of predicted to GT frames within ±tol. Returns (pairs, fp, fn)."""
    cand = sorted((abs(p - g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt) if abs(p - g) <= tol)
    used_p, used_g, pairs = set(), set(), []
    for _, i, j in cand:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((pred[i], gt[j]))
    return sorted(pairs), len(pred) - len(pairs), len(gt) - len(pairs)


def prf(tp: int, fp: int, fn: int) -> dict:
    tp, fp, fn = int(tp), int(fp), int(fn)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def ball_counts(traj: np.ndarray, gt: dict[int, tuple | None], start: int, width: int) -> dict:
    """Compare the tracked ball to GT on annotated frames only (unannotated != no ball)."""
    k = width / REF_WIDTH
    c = {"visible": 0, "tp5": 0, "tp10": 0, "detected_on_visible": 0, "errors": [],
         "invisible": 0, "fp_on_invisible": 0}
    for f, xy in gt.items():
        i = f - start
        if not 0 <= i < len(traj):
            continue
        b = traj[i]
        has = not np.isnan(b[0])
        if xy is None:
            c["invisible"] += 1
            c["fp_on_invisible"] += has
            continue
        c["visible"] += 1
        if has:
            c["detected_on_visible"] += 1
            e = float(np.hypot(b[0] - xy[0], b[1] - xy[1])) / k
            c["errors"].append(e)
            c["tp5"] += e <= 5
            c["tp10"] += e <= 10
    return c


def summarize_ball(c: dict) -> dict:
    v = max(c["visible"], 1)
    errs = c["errors"]
    return {"recall@5px": round(c["tp5"] / v, 4), "recall@10px": round(c["tp10"] / v, 4),
            "detection_rate": round(c["detected_on_visible"] / v, 4),
            "median_err_px": round(float(np.median(errs)), 2) if errs else None,
            "fp_rate_on_invisible": round(c["fp_on_invisible"] / max(c["invisible"], 1), 4),
            "n_visible": c["visible"]}


def classification_report(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    if not y_true:
        return {"n": 0}
    labels = sorted(set(y_true))
    return {"n": len(y_true), "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4)}
