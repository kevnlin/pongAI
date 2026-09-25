"""Ball candidate detectors and a trajectory tracker.

Every detector consumes frame triplets (prev, cur, next) so motion-aware detectors and
single-frame YOLO share one interface. Detectors return, per triplet, a list of candidates
(x, y, conf) in full-resolution pixel coordinates of `cur`.

Detector specs (see `load_ball_detector`):
    "motion"        classical frame-differencing baseline, no training needed
    "coco"          pretrained YOLO11 'sports ball' class (zero-shot baseline)
    <dir or .pt>    fine-tuned YOLO11 from training/train_ball.py (reads ball.json next to it)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from pongai.config import REF_WIDTH, WEIGHTS_DIR


class Candidate(NamedTuple):
    x: float
    y: float
    conf: float


Triplet = tuple[np.ndarray, np.ndarray, np.ndarray]


def stack3(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Encode three consecutive frames as one 3-channel image (grayscale t-1, t, t+1).

    Lets a standard YOLO see motion (TTNet-style temporal stacking) without changing its
    architecture. Must be used identically at training and inference time.
    """
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (prev, cur, nxt)]
    return cv2.merge(g)


class MotionBallDetector:
    """Baseline: blobs that changed in both (t-1 -> t) and (t -> t+1), filtered by size/shape."""

    name = "motion"

    def __init__(self, diff_thresh: int = 18, scale: float = 0.5):
        self.diff_thresh = diff_thresh
        self.scale = scale

    def __call__(self, triplets: list[Triplet]) -> list[list[Candidate]]:
        return [self._one(*t) for t in triplets]

    def _one(self, prev, cur, nxt) -> list[Candidate]:
        s = self.scale
        g = [cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
             for f in (prev, cur, nxt)]
        motion = cv2.min(cv2.absdiff(g[1], g[0]), cv2.absdiff(g[2], g[1]))
        mask = (motion > self.diff_thresh).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        # ball diameter at the dataset reference (1920 px wide) is ~8-20 px incl. motion blur
        k = cur.shape[1] / REF_WIDTH * s
        amin, amax = 12 * k * k, 900 * k * k
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (amin <= area <= amax) or max(w, h) > 4.0 * max(1, min(w, h)):
                continue
            cx, cy = cents[i]
            bright = float(g[1][int(cy), int(cx)]) / 255.0
            fill = area / float(w * h)
            conf = 0.5 * fill + 0.5 * bright
            out.append(Candidate(cx / s, cy / s, conf))
        out.sort(key=lambda c: -c.conf)
        return out[:8]


class YoloBallDetector:
    def __init__(self, weights: str, mode: str = "rgb", imgsz: int = 1280, conf: float = 0.1,
                 class_id: int | None = None, device: str | None = None, name: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.mode, self.imgsz, self.conf, self.class_id, self.device = mode, imgsz, conf, class_id, device
        self.name = name or f"yolo:{Path(weights).stem}:{mode}"

    def __call__(self, triplets: list[Triplet]) -> list[list[Candidate]]:
        imgs = [stack3(*t) if self.mode == "stack3" else t[1] for t in triplets]
        classes = [self.class_id] if self.class_id is not None else None
        results = self.model.predict(imgs, imgsz=self.imgsz, conf=self.conf, classes=classes,
                                     device=self.device, verbose=False)
        out = []
        for r in results:
            b = r.boxes
            xywh = b.xywh.cpu().numpy() if len(b) else np.zeros((0, 4))
            confs = b.conf.cpu().numpy() if len(b) else np.zeros(0)
            cands = [Candidate(float(x), float(y), float(c)) for (x, y, _, _), c in zip(xywh, confs)]
            out.append(sorted(cands, key=lambda c: -c.conf)[:8])
        return out


def load_ball_detector(spec: str = "auto", device: str | None = None):
    if spec == "auto":
        spec = str(WEIGHTS_DIR / "ball") if (WEIGHTS_DIR / "ball" / "ball.json").exists() else "motion"
    if spec == "motion":
        return MotionBallDetector()
    if spec == "coco":
        return YoloBallDetector("yolo11m.pt", mode="rgb", imgsz=1280, conf=0.05, class_id=32,
                                device=device, name="coco:yolo11m")
    p = Path(spec)
    meta_path = (p if p.is_dir() else p.parent) / "ball.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    weights = p / meta.get("weights", "best.pt") if p.is_dir() else p
    return YoloBallDetector(str(weights), mode=meta.get("mode", "rgb"), imgsz=meta.get("imgsz", 1280),
                            conf=meta.get("conf", 0.1), device=device,
                            name=f"yolo:{p.name}:{meta.get('mode', 'rgb')}")


class _Tracklet:
    __slots__ = ("pts", "confs")

    def __init__(self, t: int, c: Candidate):
        self.pts = [(t, c.x, c.y)]
        self.confs = [c.conf]

    def predict(self, t: int) -> tuple[float, float]:
        lt, lx, ly = self.pts[-1]
        if len(self.pts) < 2:
            return lx, ly
        pt, px, py = self.pts[-2]
        vx, vy = (lx - px) / (lt - pt), (ly - py) / (lt - pt)
        return lx + vx * (t - lt), ly + vy * (t - lt)

    def score(self, k: float, fps_k: float) -> float:
        """Ball-likeness: long, confident, fast and smooth (limbs are slow and jittery)."""
        p = np.array(self.pts, dtype=np.float64)
        if len(p) < 3:
            return 0.0
        dt = np.diff(p[:, 0])
        v = np.diff(p[:, 1:], axis=0) / dt[:, None]
        speed = np.median(np.hypot(v[:, 0], v[:, 1])) / (k * fps_k)
        jerk = np.median(np.hypot(*np.diff(v, axis=0).T)) / (k * fps_k) if len(v) > 1 else 0.0
        speed_term = np.clip(speed / 6.0, 0.0, 1.5)       # ball: > ~6 px/frame at 1080p/120fps
        smooth_term = 1.0 / (1.0 + jerk / 4.0)
        return len(p) * float(np.mean(self.confs)) * speed_term * smooth_term


def track_ball(cands: list[list[Candidate]], frame_w: int, fps: float,
               max_gap: int | None = None, min_len: int = 5, min_score: float = 1.5) -> np.ndarray:
    """Link per-frame candidates into one ball trajectory.

    Multi-hypothesis: every candidate either extends the nearest compatible tracklet
    (constant-velocity prediction + gate) or starts a new one, so noise blobs from moving
    players cannot hijack the ball track. Tracklets are then ranked by ball-likeness and each
    frame takes the position from the best tracklet covering it. Small gaps are interpolated.
    Returns an (N, 2) float array with NaN where no ball was found.
    """
    n = len(cands)
    k = frame_w / REF_WIDTH
    fps_k = 120.0 / fps  # the ball moves further per frame at lower fps
    gate = 60.0 * k * fps_k
    max_gap = max_gap or max(2, int(round(6 / fps_k)))

    active: list[_Tracklet] = []
    done: list[_Tracklet] = []
    for t, cs in enumerate(cands):
        still = []
        for tr in active:
            (done if t - tr.pts[-1][0] > max_gap else still).append(tr)
        active = still
        # greedy assignment by distance between predictions and candidates
        pairs = []
        for ti, tr in enumerate(active):
            ex, ey = tr.predict(t)
            dt = t - tr.pts[-1][0]
            radius = gate * (1 + 0.5 * (dt - 1)) * (1.5 if len(tr.pts) < 2 else 1.0)
            for ci, c in enumerate(cs):
                d = np.hypot(c.x - ex, c.y - ey)
                if d <= radius:
                    pairs.append((d, ti, ci))
        used_t, used_c = set(), set()
        for _, ti, ci in sorted(pairs):
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            active[ti].pts.append((t, cs[ci].x, cs[ci].y))
            active[ti].confs.append(cs[ci].conf)
        active += [_Tracklet(t, c) for ci, c in enumerate(cs) if ci not in used_c]
    done += active

    ranked = sorted(((tr.score(k, fps_k), tr) for tr in done if len(tr.pts) >= min_len),
                    key=lambda s: -s[0])
    traj = np.full((n, 2), np.nan)
    for s, tr in ranked:
        if s < min_score:
            break
        for (t0, x0, y0), (t1, x1, y1) in zip(tr.pts, tr.pts[1:]):
            for t in range(t0, t1):  # includes interpolation of small gaps
                if np.isnan(traj[t, 0]):
                    a = (t - t0) / (t1 - t0)
                    traj[t] = (x0 + a * (x1 - x0), y0 + a * (y1 - y0))
        t_last = tr.pts[-1][0]
        if np.isnan(traj[t_last, 0]):
            traj[t_last] = tr.pts[-1][1:]
    return traj
