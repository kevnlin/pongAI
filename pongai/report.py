"""Turn perception + events + stroke predictions into the dashboard report for one player."""
from __future__ import annotations

from collections import Counter

import numpy as np

from pongai.config import REF_FPS, TABLE_LENGTH_M


def _m_per_px(perc, side: str) -> tuple[float | None, str]:
    """Metres per pixel near the table plane: from the table if found, else from player height."""
    if perc.table is not None:
        return perc.table.m_per_px, "table"
    boxes = perc.boxes[side]
    h = boxes[:, 3] - boxes[:, 1]
    h = h[np.isfinite(h)]
    if len(h) > 10:
        return 1.55 / float(np.median(h)), "player-height"  # ready stance ≈ 1.55 m tall
    return None, "none"


def shot_speed_kmh(ball: np.ndarray, t: int, fps: float, m_per_px: float | None) -> float | None:
    """Median image-plane ball speed shortly after contact, converted to km/h.

    The camera looks at the table from the side, so the flight is mostly in the image plane;
    motion towards/away from the camera is ignored, hence this is an estimate (lower bound).
    """
    if m_per_px is None:
        return None
    kt = fps / REF_FPS
    seg = ball[t + max(1, int(round(2 * kt))):t + max(3, int(round(14 * kt)))]
    seg = seg[~np.isnan(seg[:, 0])]
    if len(seg) < 3:
        return None
    step = np.hypot(*np.diff(seg, axis=0).T)
    v = float(np.median(step)) * fps * m_per_px * 3.6
    return round(v, 1) if 3 < v < 200 else None


def build_report(perc, events, side: str, strokes: list[dict], models: dict) -> dict:
    fps = perc.fps
    opp = "left" if side == "right" else "right"
    m_per_px, scale_src = _m_per_px(perc, side)
    table = perc.table
    side_hits = [h for h in events.hits if h.side == side]

    # rally index for every hit
    rally_of = {}
    for i, r in enumerate(events.rallies):
        for h in r.hits:
            rally_of[h.frame] = i

    shots = []
    for n, (h, pred) in enumerate(zip(side_hits, strokes), start=1):
        nxt = next((x.frame for x in events.hits if x.frame > h.frame), len(perc))
        landing = next((b for b in events.bounces if h.frame < b.frame < nxt), None)
        place = None
        if landing is not None and landing.half == opp and table is not None:
            u, v = table.to_table([[landing.x, landing.y]])[0]
            if -0.2 <= u <= TABLE_LENGTH_M + 0.2:
                # report as seen from the player's own end: depth 0 = net, 1 = opponent's end line
                depth = (TABLE_LENGTH_M / 2 - u) / (TABLE_LENGTH_M / 2) if side == "right" else \
                        (u - TABLE_LENGTH_M / 2) / (TABLE_LENGTH_M / 2)
                place = {"u": round(float(u), 3), "v": round(float(v), 3), "depth": round(float(depth), 3)}
        shots.append({
            "n": n, "frame": h.frame, "time_s": round(h.frame / fps, 2), "serve_detected": h.serve,
            **pred, "speed_kmh": shot_speed_kmh(perc.ball, h.frame, fps, m_per_px),
            "landing": place, "rally": rally_of.get(h.frame),
        })

    speeds = [s["speed_kmh"] for s in shots if s["speed_kmh"]]
    by_hand = {hand: [s["speed_kmh"] for s in shots if s["speed_kmh"] and s.get("hand") == hand]
               for hand in ("forehand", "backhand")}
    avg = lambda xs: round(float(np.mean(xs)), 1) if xs else None  # noqa: E731

    rallies = []
    for i, r in enumerate(events.rallies):
        mine = [s for s in shots if s["rally"] == i]
        rallies.append({"idx": i, "start_s": round(r.start / fps, 2), "end_s": round(r.end / fps, 2),
                        "n_strokes": len(r.hits), "player_strokes": len(mine), "outcome": r.outcome,
                        "last_hitter": r.last_hitter, "won": r.point_winner == side if r.point_winner else None})

    count = lambda key: dict(Counter(s.get(key, "unknown") for s in shots).most_common())  # noqa: E731
    return {
        "player_side": side,
        "video": {"fps": fps, "frames": len(perc), "duration_s": round(len(perc) / fps, 2),
                  "width": perc.width, "height": perc.height},
        "table": table.to_dict() if table is not None else None,
        "scale_source": scale_src,
        "models": models,
        "totals": {"strokes": len(shots), "forehand": sum(s.get("hand") == "forehand" for s in shots),
                   "backhand": sum(s.get("hand") == "backhand" for s in shots),
                   "rallies": len(rallies), "points_won": sum(r["won"] is True for r in rallies),
                   "points_lost": sum(r["won"] is False for r in rallies),
                   "ball_visible_pct": round(float(np.mean(~np.isnan(perc.ball[:, 0]))) * 100, 1)},
        "speed": {"avg_kmh": avg(speeds), "peak_kmh": max(speeds) if speeds else None,
                  "forehand_avg_kmh": avg(by_hand["forehand"]), "backhand_avg_kmh": avg(by_hand["backhand"])},
        "technique_counts": count("technique"), "lean_counts": count("lean"), "feet_counts": count("feet"),
        "shots": shots,
        "rallies": rallies,
    }
