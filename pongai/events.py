"""Hit / bounce / net-crossing / rally detection from the tracked ball trajectory.

All thresholds are defined at the dataset reference (1920 px wide, 120 fps) and rescaled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from pongai.config import REF_FPS, REF_WIDTH


@dataclass
class Hit:
    frame: int
    side: str
    serve: bool = False
    score: float = 0.0


@dataclass
class Bounce:
    frame: int
    x: float
    y: float
    half: str  # "left" | "right" half of the table


@dataclass
class Rally:
    start: int
    end: int
    hits: list[Hit] = field(default_factory=list)
    bounces: list[Bounce] = field(default_factory=list)
    last_hitter: str | None = None
    outcome: str = "unknown"          # "winner" | "net" | "out" (heuristic)
    point_winner: str | None = None

    def to_dict(self):
        d = asdict(self)
        d["n_strokes"] = len(self.hits)
        return d


@dataclass
class Events:
    hits: list[Hit]
    bounces: list[Bounce]
    net_crossings: list[int]
    rallies: list[Rally]


def _scales(width: int, fps: float) -> tuple[float, float]:
    """(pixel scale, frames scale) relative to the reference video."""
    return width / REF_WIDTH, fps / REF_FPS


def _smooth(traj: np.ndarray, k: int = 3) -> np.ndarray:
    """Moving average over valid samples only (NaN stays NaN)."""
    out = traj.copy()
    valid = ~np.isnan(traj[:, 0])
    for c in range(2):
        v = np.where(valid, traj[:, c], 0.0)
        num = np.convolve(v, np.ones(k), "same")
        den = np.convolve(valid.astype(float), np.ones(k), "same")
        out[:, c] = np.where(valid & (den > 0), num / np.maximum(den, 1e-9), np.nan)
    return out


def _mean(a: np.ndarray, lo: int, hi: int) -> float:
    seg = a[max(lo, 0):max(hi, 0)]
    seg = seg[~np.isnan(seg)]
    return float(seg.mean()) if len(seg) >= max(1, (hi - lo) // 2) else np.nan


def _nms(frames: list[int], scores: list[float], radius: int) -> list[int]:
    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    for i in order:
        if all(abs(frames[i] - k) > radius for k in kept):
            kept.append(frames[i])
    return sorted(kept)


def detect_hits(traj: np.ndarray, net_x: float, width: int, fps: float, table=None) -> list[Hit]:
    """A hit is where the ball's horizontal direction reverses on that player's side.

    Right player: the ball moves right (vx > 0) then left (vx < 0) with x > net. Serves are
    the first strong leftward motion after the ball was absent or moving mostly vertically
    (the toss). The hit frame is the extreme x position (≈ racket contact).
    """
    kx, kt = _scales(width, fps)
    s = _smooth(traj)
    vx = np.gradient(s[:, 0]) if len(s) > 1 else np.zeros(len(s))
    w = max(2, int(round(5 * kt)))
    vmin = 3.5 * kx / kt
    # contact happens away from the net and above the floor
    if table is not None:
        min_dx = 0.15 * table.length_px
        max_y = table.corners[:, 1].max() + 0.25 * table.length_px
    else:
        min_dx, max_y = 0.05 * width, np.inf
    frames, scores, sides, serves = [], [], [], []
    for t in range(w, len(s) - w):
        x, y = s[t]
        if np.isnan(x) or y > max_y:
            continue
        before, after = _mean(vx, t - w, t), _mean(vx, t + 1, t + 1 + w)
        if np.isnan(after):
            continue
        for side, sign in (("right", 1), ("left", -1)):
            if sign * (x - net_x) < min_dx:
                continue
            moving_away = sign * after < -vmin
            if not moving_away:
                continue
            if not np.isnan(before) and sign * before > vmin:
                # rally stroke: take the extreme position in the neighbourhood as contact
                win = s[t - w:t + w + 1, 0]
                tt = t - w + int(np.nanargmax(sign * win))
                frames.append(tt); scores.append(abs(before - after)); sides.append(side); serves.append(False)
            elif np.isnan(before) or abs(before) < vmin / 2:
                # serve: first strong motion away from the server after a toss / no ball
                prev_seg = traj[max(0, t - int(0.3 * fps)):t - w, 0]
                if len(prev_seg) and np.all(np.isnan(prev_seg) | (sign * (prev_seg - net_x) > 0)):
                    frames.append(t); scores.append(abs(after) * 0.5); sides.append(side); serves.append(True)
    keep = _nms(frames, scores, radius=max(4, int(round(18 * kt))))
    info = {}
    for f, sd, sv, sc in zip(frames, sides, serves, scores):
        if f not in info or sc > info[f][2]:
            info[f] = (sd, sv, sc)
    return [Hit(f, *info[f]) for f in keep]


def _enforce_alternation(hits: list[Hit]) -> list[Hit]:
    """Within a rally the two players alternate; of consecutive same-side hits keep the stronger."""
    out: list[Hit] = []
    for h in hits:
        if out and out[-1].side == h.side and not h.serve:
            if h.score > out[-1].score:
                out[-1] = h
            continue
        out.append(h)
    return out


def detect_bounces(traj: np.ndarray, width: int, fps: float, table=None, hits: list[Hit] = ()) -> list[Bounce]:
    """A bounce is a sharp downward -> upward reversal (image y max) over the table area."""
    kx, kt = _scales(width, fps)
    s = _smooth(traj)
    vy = np.gradient(s[:, 1]) if len(s) > 1 else np.zeros(len(s))
    w = max(2, int(round(4 * kt)))
    vmin = 1.0 * kx / kt
    hit_frames = np.array([h.frame for h in hits])
    guard = max(3, int(round(6 * kt)))
    net_x = table.net_x if table is not None else width / 2
    frames, scores = [], []
    for t in range(w, len(s) - w):
        x, y = s[t]
        if np.isnan(y):
            continue
        down, up = _mean(vy, t - w, t), _mean(vy, t + 1, t + 1 + w)
        if np.isnan(down) or np.isnan(up) or down < vmin or up > -vmin:
            continue
        win = s[t - w:t + w + 1, 1]
        if np.nanargmax(win) != w:  # must be the lowest point locally
            continue
        if len(hit_frames) and np.min(np.abs(hit_frames - t)) <= guard:
            continue
        if table is not None:
            x0, x1 = table.x_range
            margin = 0.03 * (x1 - x0)
            top = table.corners[:, 1].min() - 0.06 * table.length_px
            bottom = table.corners[:, 1].max() + 0.02 * table.length_px
            if not (x0 - margin <= x <= x1 + margin and top <= y <= bottom):
                continue
        frames.append(t); scores.append(down - up)
    keep = _nms(frames, scores, radius=max(3, int(round(10 * kt))))
    out = []
    for f in keep:
        x, y = traj[f] if not np.isnan(traj[f, 0]) else s[f]
        half = table.half(x, y) if table is not None else ("right" if x > net_x else "left")
        out.append(Bounce(f, float(x), float(y), half))
    return out


def detect_net_crossings(traj: np.ndarray, net_x: float, fps: float) -> list[int]:
    x = traj[:, 0]
    out = []
    for t in range(1, len(x)):
        if not (np.isnan(x[t]) or np.isnan(x[t - 1])) and (x[t - 1] - net_x) * (x[t] - net_x) < 0:
            if not out or t - out[-1] > int(0.08 * fps):
                out.append(t)
    return out


def segment_rallies(hits: list[Hit], bounces: list[Bounce], fps: float, max_gap_s: float = 2.5) -> list[Rally]:
    """Group hits into rallies and guess how each ended from the bounces after the last hit.

    After the last stroke of side S:
      first bounce on the opponent's half  -> opponent failed to return   -> S wins ("winner")
      first bounce on S's own half         -> S's ball did not clear the net -> S loses ("net")
      no bounce before the rally times out -> S's ball missed the table   -> S loses ("out")
    """
    groups: list[list[Hit]] = []
    cur: list[Hit] = []
    for h in hits:
        if cur and (h.frame - cur[-1].frame > max_gap_s * fps or h.serve):
            groups.append(cur)
            cur = []
        cur.append(h)
    if cur:
        groups.append(cur)
    rallies = [Rally(g[0].frame, g[-1].frame, g) for g in map(_enforce_alternation, groups)]

    for i, r in enumerate(rallies):
        horizon = r.end + int(1.5 * fps)
        if i + 1 < len(rallies):
            horizon = min(horizon, rallies[i + 1].start)
        r.bounces = [b for b in bounces if r.start <= b.frame <= horizon]
        last = r.hits[-1]
        r.last_hitter = last.side
        opp = "left" if last.side == "right" else "right"
        after = [b for b in r.bounces if b.frame > last.frame]
        if after and after[0].half == opp:
            r.outcome, r.point_winner = "winner", last.side
        elif after:
            r.outcome, r.point_winner = "net", opp
        else:
            r.outcome, r.point_winner = "out", opp
        r.end = max(r.end, after[-1].frame if after else r.end)
    # a lone "hit" with no table bounce is almost always the ball being fetched / tossed
    return [r for r in rallies if len(r.hits) > 1 or r.bounces]


def detect_events(traj: np.ndarray, width: int, fps: float, table=None) -> Events:
    net_x = table.net_x if table is not None else width / 2
    raw_hits = detect_hits(traj, net_x, width, fps, table)
    bounces = detect_bounces(traj, width, fps, table, raw_hits)
    nets = detect_net_crossings(traj, net_x, fps)
    rallies = segment_rallies(raw_hits, bounces, fps)
    hits = [h for r in rallies for h in r.hits]
    return Events(hits, bounces, nets, rallies)
