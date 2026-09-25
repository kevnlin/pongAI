"""Deterministic coaching observations from the report.

These always run (no API needed), and are also passed to the language model as grounded
evidence so its advice is tied to what the CV stack actually measured.
"""
from __future__ import annotations


def _share(counts: dict, keys, total: int) -> float:
    return sum(counts.get(k, 0) for k in keys) / total if total else 0.0


def coaching_tips(report: dict) -> list[dict]:
    t = report["totals"]
    n = t["strokes"]
    tips: list[dict] = []
    if n < 4:
        return [{"title": "Not enough strokes detected",
                 "detail": "Record a longer rally from the side with the whole table in view so the ball and your "
                           "strokes can be tracked reliably.", "evidence": f"{n} strokes detected"}]

    fh, bh = t["forehand"], t["backhand"]
    if fh + bh >= 6:
        bh_share = bh / (fh + bh)
        if bh_share < 0.25:
            tips.append({"title": "Backhand is under-used",
                         "detail": "Most balls are played on the forehand. Opponents will target your backhand "
                                   "corner; drill backhand-to-backhand rallies and forehand/backhand switching.",
                         "evidence": f"{fh} forehands vs {bh} backhands"})
        elif bh_share > 0.75:
            tips.append({"title": "Forehand is under-used",
                         "detail": "You rarely step around to use the forehand, usually the stronger attacking "
                                   "side. Practise the 'Falkenberg' footwork drill to open more forehands.",
                         "evidence": f"{fh} forehands vs {bh} backhands"})

    lean = report["lean_counts"]
    back = _share(lean, ["back_heavy"], n)
    if back > 0.2:
        tips.append({"title": "Weight falling backwards at contact",
                     "detail": "On many strokes your weight is on the back foot when you hit. Load on the back leg "
                               "during the backswing, then push forward into the ball so your body weight goes "
                               "through the shot.", "evidence": f"{back:.0%} of strokes back-heavy"})

    feet = report["feet_counts"]
    lifted = _share(feet, ["both_feet_lifted", "left_foot_lifted", "right_foot_lifted"], n)
    if lifted > 0.45:
        tips.append({"title": "Balance at contact",
                     "detail": "A foot is often off the ground at contact. Some lift is normal on powerful loops, "
                               "but for control shots keep a wide, stable base and finish balanced.",
                     "evidence": f"{lifted:.0%} of strokes with a foot lifted"})

    tech = report["technique_counts"]
    push = _share(tech, ["push"], n)
    loop = _share(tech, ["loop"], n)
    if push > 0.35 and loop < 0.25:
        tips.append({"title": "Passive rallies",
                     "detail": "You push a lot and loop rarely. Look for long or high balls to open with a loop "
                               "instead of pushing back.", "evidence": f"{push:.0%} pushes, {loop:.0%} loops"})

    sp = report["speed"]
    if sp["forehand_avg_kmh"] and sp["backhand_avg_kmh"]:
        f_, b_ = sp["forehand_avg_kmh"], sp["backhand_avg_kmh"]
        if b_ < 0.75 * f_:
            tips.append({"title": "Backhand lacks pace",
                         "detail": "Your backhand is noticeably slower than your forehand. Use more forearm and "
                                   "wrist snap, and contact the ball in front of the body.",
                         "evidence": f"forehand ≈ {f_} km/h vs backhand ≈ {b_} km/h (estimates)"})

    lost = [r for r in report["rallies"] if r["won"] is False and r["last_hitter"] == report["player_side"]]
    if len(lost) >= 3:
        nets = sum(r["outcome"] == "net" for r in lost)
        outs = sum(r["outcome"] == "out" for r in lost)
        if nets > outs:
            tips.append({"title": "Errors into the net",
                         "detail": "Most of your own errors do not clear the net. Lift the ball with more "
                                   "upward brush (topspin) and don't hit the ball too late / too low.",
                         "evidence": f"{nets} net errors vs {outs} long/wide"})
        elif outs > nets:
            tips.append({"title": "Errors long or wide",
                         "detail": "Most of your own errors miss the table. Close the racket angle slightly and "
                                   "generate more spin instead of more speed.",
                         "evidence": f"{outs} long/wide errors vs {nets} into the net"})

    if not tips:
        tips.append({"title": "Solid, balanced session",
                     "detail": "No major issues stood out in the measured stats. Keep working on consistency "
                               "and shot placement.", "evidence": f"{n} strokes analysed"})
    return tips
