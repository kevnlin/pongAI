"""Stroke feature extraction (pose + ball around the contact frame) and classifiers.

Targets, trained against the Extended OpenTTGames labels:
    hand       forehand / backhand
    technique  serve, loop, block, push, flick, lob, chop, smash
    lean       neutral, back_heavy, front_heavy, right_leaning, left_leaning, unknown
    feet       both_feet_planted, left_foot_lifted, right_foot_lifted, both_feet_lifted, unknown

Geometry is normalised so every player is described the same way: coordinates are relative
to the hip centre at contact, scaled by torso length, and the LEFT player is mirrored so both
players face -x. Keypoint channels are reordered so "racket arm" is always the same channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pongai.config import REF_FPS, WEIGHTS_DIR
from pongai.vision.pose import L_ANK, L_ELB, L_HIP, L_KNE, L_SHO, L_WRI, LR_PAIRS, NOSE, R_ANK, R_ELB, \
    R_HIP, R_KNE, R_SHO, R_WRI

TARGETS = ("hand", "technique", "lean", "feet")
OFFSETS = (-16, -12, -8, -4, -2, 0, 2, 4, 8)  # frames at 120 fps relative to contact
KP_USE = (NOSE, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK)
FEATURE_VERSION = 1


@dataclass
class StrokeContext:
    frame: int                 # contact frame (index into the perception arrays)
    side: str                  # "left" | "right"
    racket_hand: str           # "R" | "L" (player's anatomical side)
    first_of_rally: bool
    since_prev_hit_s: float    # NaN if first


def _pick(track: np.ndarray, t: int) -> np.ndarray:
    if 0 <= t < len(track):
        return track[t]
    return np.full(track.shape[1:], np.nan)


def wrist_motion(pose: np.ndarray, hit_frames: list[int], fps: float) -> dict[str, float]:
    """Total right / left wrist travel around the given contact frames."""
    w = max(2, int(round(12 * fps / REF_FPS)))
    speed = {"R": 0.0, "L": 0.0}
    for t in hit_frames:
        seg = pose[max(0, t - w):t + w // 2]
        for hand, k in (("R", R_WRI), ("L", L_WRI)):
            d = np.diff(seg[:, k, :2], axis=0)
            speed[hand] += float(np.nansum(np.hypot(d[:, 0], d[:, 1])))
    return speed


def racket_hand(pose: np.ndarray, hit_frames: list[int], fps: float) -> str:
    """The racket wrist moves much more than the free wrist around contact."""
    m = wrist_motion(pose, hit_frames, fps)
    return "L" if m["L"] > m["R"] else "R"


def _angle(a, b, c) -> float:
    """Angle at b in degrees."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1)))) if n > 0 else np.nan


def stroke_features(pose: np.ndarray, ball: np.ndarray, ctx: StrokeContext, fps: float,
                    table=None, frame_w: int = 1920) -> np.ndarray:
    kt = fps / REF_FPS
    t = ctx.frame
    mirror = -1.0 if ctx.side == "left" else 1.0

    p0 = _pick(pose, t)[:, :2]
    hip = (p0[L_HIP] + p0[R_HIP]) / 2
    sho = (p0[L_SHO] + p0[R_SHO]) / 2
    torso = np.linalg.norm(sho - hip)
    if not np.isfinite(torso) or torso < 1:
        # fall back to a window median torso length
        win = pose[max(0, t - 20):t + 20, :, :2]
        tl = np.linalg.norm((win[:, L_SHO] + win[:, R_SHO]) / 2 - (win[:, L_HIP] + win[:, R_HIP]) / 2, axis=1)
        torso = np.nanmedian(tl) if np.any(np.isfinite(tl)) else np.nan
        if np.any(np.isfinite(win[:, L_HIP, 0])):
            hip = np.nanmedian((win[:, L_HIP] + win[:, R_HIP]) / 2, axis=0)

    swap = ctx.racket_hand == "L"
    order = np.arange(17)
    if swap:  # make channel R_* always the racket arm
        for a, b in LR_PAIRS:
            order[a], order[b] = b, a

    def norm(pt):
        q = (pt - hip) / torso
        q[..., 0] *= mirror
        return q

    feats: list[float] = []
    frames_kp = []
    for off in OFFSETS:
        kp = _pick(pose, t + int(round(off * kt)))[order][:, :2]
        nk = norm(kp)
        frames_kp.append(nk)
        feats += list(nk[list(KP_USE)].ravel())
    K = np.stack(frames_kp)  # (len(OFFSETS), 17, 2) normalised
    i0 = OFFSETS.index(0)

    # derived pose features
    rw = K[:, R_WRI]
    rw_speed = np.hypot(*np.diff(rw, axis=0).T) / np.diff(np.array(OFFSETS))
    k0 = K[i0]
    feats += [
        rw_speed[i0 - 1] if i0 > 0 else np.nan,          # racket wrist speed into contact
        np.nanmax(rw_speed) if np.any(np.isfinite(rw_speed)) else np.nan,
        k0[R_WRI, 1] - k0[R_SHO, 1],                      # wrist height vs shoulder
        k0[R_WRI, 0], K[OFFSETS.index(-8), R_WRI, 0],      # wrist x at contact / backswing
        _angle(k0[R_SHO], k0[R_ELB], k0[R_WRI]),
        _angle(k0[L_HIP], k0[L_KNE], k0[L_ANK]), _angle(k0[R_HIP], k0[R_KNE], k0[R_ANK]),
        np.degrees(np.arctan2(*(k0[L_SHO] + k0[R_SHO] - k0[L_HIP] - k0[R_HIP])[::-1])) if np.all(np.isfinite(k0)) else np.nan,
        k0[L_ANK, 0] - k0[R_ANK, 0], k0[L_ANK, 1] - k0[R_ANK, 1],   # stance width / foot height diff
        K[-1, L_ANK, 1] - K[0, L_ANK, 1], K[-1, R_ANK, 1] - K[0, R_ANK, 1],  # ankle vertical motion
        K[-1, 0, 0] - K[0, 0, 0],                         # head travel (weight transfer)
        (k0[L_HIP, 1] + k0[R_HIP, 1]) / 2 - (k0[L_ANK, 1] + k0[R_ANK, 1]) / 2,  # crouch
    ]

    # ball features (velocities in torso lengths per reference frame)
    def ball_vel(lo, hi):
        seg = ball[max(0, t + int(round(lo * kt))):max(0, t + int(round(hi * kt)))]
        seg = seg[~np.isnan(seg[:, 0])]
        if len(seg) < 2:
            return np.array([np.nan, np.nan])
        v = (seg[-1] - seg[0]) / (len(seg) - 1) / kt
        return np.array([v[0] * mirror, v[1]]) / torso

    vin, vout = ball_vel(-8, -1), ball_vel(1, 9)
    b0 = _pick(ball, t)
    bpos = norm(b0.copy()) if np.all(np.isfinite(b0)) else np.array([np.nan, np.nan])
    table_h = (b0[1] - table.surface_y(b0[0])) / torso if table is not None and np.isfinite(b0[0]) else np.nan
    feats += [*vin, *vout, float(np.hypot(*vout)), float(np.degrees(np.arctan2(-vout[1], -vout[0]))),
              float(np.hypot(*vout) / max(np.hypot(*vin), 1e-3)) if np.all(np.isfinite(vin)) else np.nan,
              *bpos, table_h]

    # context
    near_cam = float((ctx.side == "left") == (ctx.racket_hand == "R"))
    feats += [float(ctx.first_of_rally), ctx.since_prev_hit_s, near_cam, float(ctx.side == "left")]
    return np.asarray(feats, dtype=np.float32)


class StrokeClassifier:
    """One gradient-boosted tree model per target (handles NaN features natively)."""

    def __init__(self, models: dict | None = None, meta: dict | None = None):
        self.models = models or {}
        self.meta = meta or {}

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def fit(self, X: np.ndarray, labels: dict[str, list[str]], seed: int = 0):
        from sklearn.ensemble import HistGradientBoostingClassifier

        for target in TARGETS:
            y = np.asarray(labels[target])
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                   l2_regularization=1.0, class_weight="balanced",
                                                   random_state=seed)
            model.fit(X, y)
            self.models[target] = model
        self.meta = {"feature_version": FEATURE_VERSION, "n_features": X.shape[1], "n_train": len(X)}
        return self

    def predict(self, X: np.ndarray) -> list[dict]:
        if len(X) == 0:
            return []
        out = [dict() for _ in range(len(X))]
        for target, model in self.models.items():
            proba = model.predict_proba(X)
            for i, row in enumerate(proba):
                j = int(np.argmax(row))
                out[i][target] = str(model.classes_[j])
                out[i][f"{target}_conf"] = round(float(row[j]), 3)
        return out

    def save(self, path: Path):
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path | None = None) -> "StrokeClassifier":
        import joblib

        path = path or WEIGHTS_DIR / "stroke" / "stroke_clf.joblib"
        if not Path(path).exists():
            return cls()
        d = joblib.load(path)
        if d["meta"].get("feature_version") != FEATURE_VERSION:
            print(f"[strokes] {path} was trained on feature v{d['meta'].get('feature_version')}, "
                  f"need v{FEATURE_VERSION}; retrain with training/train_strokes.py")
            return cls()
        return cls(d["models"], d["meta"])


def heuristic_prediction(ctx: StrokeContext, feats: np.ndarray) -> dict:
    """Used before a trained model exists: serve from context, everything else unknown."""
    return {"hand": "unknown", "technique": "serve" if ctx.first_of_rally else "unknown",
            "lean": "unknown", "feet": "unknown"}


def contexts_for_hits(hits, pose_by_side: dict[str, np.ndarray], fps: float,
                      racket: dict[str, str] | None = None) -> list[StrokeContext]:
    """Build StrokeContext for detected (or ground-truth) hits in temporal order.

    `hits` items need .frame, .side and optionally .serve / rally boundaries via .serve.
    """
    racket = racket or {}
    for side in ("left", "right"):
        if side not in racket:
            racket[side] = racket_hand(pose_by_side[side], [h.frame for h in hits if h.side == side], fps)
    out, prev = [], None
    for h in hits:
        first = bool(getattr(h, "serve", False)) or prev is None or (h.frame - prev) > 2.5 * fps
        out.append(StrokeContext(h.frame, h.side, racket[h.side], first,
                                 np.nan if first else (h.frame - prev) / fps))
        prev = h.frame
    return out
