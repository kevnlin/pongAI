"""Annotated output video (ball trail, player skeleton, stroke labels) + contact keyframes."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pongai.vision.pose import SKELETON

GREEN = (80, 200, 120)
YELLOW = (40, 230, 240)
WHITE = (245, 245, 245)
GREY = (170, 170, 170)


def draw_skeleton(img, kpts, color, thickness=2, min_conf=0.3, scale=1.0):
    if kpts is None or np.isnan(kpts[0, 0]):
        return
    pts = kpts[:, :2] * scale
    ok = kpts[:, 2] >= min_conf
    for a, b in SKELETON:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, thickness, cv2.LINE_AA)
    for i in range(5, 17):
        if ok[i]:
            cv2.circle(img, tuple(pts[i].astype(int)), thickness + 1, color, -1, cv2.LINE_AA)


def _label(shot: dict) -> str:
    parts = [f"#{shot['n']}", str(shot.get("hand", "")).upper(), str(shot.get("technique", "")).upper()]
    if shot.get("speed_kmh"):
        parts.append(f"{shot['speed_kmh']:.0f} km/h")
    return "  ".join(p for p in parts if p and p != "UNKNOWN")


def pick_keyframes(shots: list[dict], k: int = 6) -> list[dict]:
    """Prefer one example of each (hand, technique) combination, then fill in time order."""
    seen, out = set(), []
    for s in shots:
        key = (s.get("hand"), s.get("technique"))
        if key not in seen:
            seen.add(key)
            out.append(s)
    out += [s for s in shots if s not in out]
    return sorted(out[:k], key=lambda s: s["frame"])


def render_overlay(video_path: str | Path, out_path: str | Path, perc, report: dict, side: str,
                   max_width: int = 1280, progress=None) -> list[tuple[str, np.ndarray]]:
    import imageio_ffmpeg

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, perc.start)
    fps = perc.fps
    step = max(1, int(round(fps / 60)))  # write at most ~60 fps, same real-time speed
    scale = min(1.0, max_width / perc.width)
    ow, oh = int(perc.width * scale) // 2 * 2, int(perc.height * scale) // 2 * 2
    writer = imageio_ffmpeg.write_frames(
        str(out_path), (ow, oh), fps=fps / step, codec="libx264", pix_fmt_in="rgb24", quality=7,
        pix_fmt_out="yuv420p", output_params=["-movflags", "+faststart"], macro_block_size=2)
    writer.send(None)

    shots = report["shots"]
    shot_at = {s["frame"]: s for s in shots}
    key_frames = {s["frame"]: s for s in pick_keyframes(shots)}
    keyframes: list[tuple[str, np.ndarray]] = []
    show_for = int(0.8 * fps)
    trail = int(0.12 * fps)
    max_step = 80 * perc.width / 1920 * 120 / fps
    active: dict | None = None
    active_until = -1
    corners = np.array(report["table"]["corners"]) * scale if report.get("table") else None
    n = len(perc)

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if i in key_frames:
            box = perc.boxes[side][i]
            if np.all(np.isfinite(box)):
                img = frame.copy()
                draw_skeleton(img, perc.poses[side][i], GREEN, 3)
                x1, y1, x2, y2 = box
                pw, ph = x2 - x1, y2 - y1
                x1, x2 = int(max(0, x1 - 0.4 * pw)), int(min(perc.width, x2 + 0.4 * pw))
                y1, y2 = int(max(0, y1 - 0.15 * ph)), int(min(perc.height, y2 + 0.1 * ph))
                keyframes.append((f"Stroke {_label(key_frames[i])}", img[y1:y2, x1:x2]))
        if i in shot_at:
            active, active_until = shot_at[i], i + show_for
        if i % step:
            continue

        img = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        if corners is not None:
            cv2.polylines(img, [corners.astype(np.int32)], True, WHITE, 1, cv2.LINE_AA)
        other = "left" if side == "right" else "right"
        draw_skeleton(img, perc.poses[other][i], GREY, 1, scale=scale)
        draw_skeleton(img, perc.poses[side][i], GREEN, 2, scale=scale)
        # fading ball trail
        pts = perc.ball[max(0, i - trail):i + 1]
        for j in range(1, len(pts)):
            a, b = pts[j - 1], pts[j]
            # a big jump means the tracker switched tracks; don't draw a line across the frame
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.hypot(*(a - b)) < max_step:
                cv2.line(img, tuple((a * scale).astype(int)), tuple((b * scale).astype(int)), YELLOW,
                         max(1, int(3 * j / len(pts))), cv2.LINE_AA)
        if np.all(np.isfinite(perc.ball[i])):
            cv2.circle(img, tuple((perc.ball[i] * scale).astype(int)), 7, YELLOW, 2, cv2.LINE_AA)
        if active is not None and i <= active_until:
            text = _label(active)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            cv2.rectangle(img, (16, 16), (32 + tw, 30 + th), (36, 80, 44), -1)
            cv2.putText(img, text, (24, 22 + th), cv2.FONT_HERSHEY_DUPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
        writer.send(np.ascontiguousarray(img[:, :, ::-1]))
        if progress and i % 60 == 0:
            progress(i / n)
    writer.close()
    cap.release()
    return keyframes
