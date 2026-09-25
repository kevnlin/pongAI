"""Single pass over a video range: ball candidates + trajectory, player poses, table.

Results are cached as .npz keyed by (video, range, detector names) so evaluation and
training re-runs only redo the cheap downstream steps (tracking, events, classifiers).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from pongai.config import CACHE_DIR
from pongai.vision.ball import Candidate, track_ball
from pongai.vision.pose import interpolate_kpts, select_players
from pongai.vision.table import Table, detect_table

MAX_CANDS = 8
SIDES = ("left", "right")


@dataclass
class Perception:
    start: int
    fps: float
    width: int
    height: int
    cands: np.ndarray                      # (N, MAX_CANDS, 3) x y conf, NaN padded
    poses: dict[str, np.ndarray]           # side -> (N, 17, 3), NaN when missing
    boxes: dict[str, np.ndarray]           # side -> (N, 4)
    table: Table | None = None
    ball: np.ndarray = field(default=None)  # (N, 2) tracked trajectory

    def __post_init__(self):
        if self.ball is None:
            self.retrack()

    def __len__(self):
        return len(self.cands)

    def retrack(self):
        cl = [[Candidate(*c) for c in row if not np.isnan(c[0])] for row in self.cands]
        self.ball = track_ball(cl, self.width, self.fps)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, start=self.start, fps=self.fps, width=self.width, height=self.height, cands=self.cands,
            **{f"pose_{s}": self.poses[s] for s in SIDES}, **{f"box_{s}": self.boxes[s] for s in SIDES},
            table=self.table.corners if self.table is not None else np.zeros(0))

    @classmethod
    def load(cls, path: Path) -> "Perception":
        z = np.load(path)
        table = Table(z["table"]) if z["table"].size else None
        return cls(int(z["start"]), float(z["fps"]), int(z["width"]), int(z["height"]), z["cands"],
                   {s: z[f"pose_{s}"] for s in SIDES}, {s: z[f"box_{s}"] for s in SIDES}, table)


def find_table(cap: cv2.VideoCapture, n_frames: int, tries: int = 8) -> Table | None:
    """Try several frames spread over the video (players/ball may occlude one frame)."""
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    found = []
    for f in np.linspace(0, max(n_frames - 1, 0), tries).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok and (t := detect_table(img)) is not None:
            found.append(t.corners)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    if not found:
        return None
    return Table(np.median(np.stack(found), axis=0))


def table_for_video(video_path) -> Table | None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        return find_table(cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()


def iter_triplets(cap: cv2.VideoCapture, start: int, end: int):
    """Yield (frame_index, (prev, cur, next)) for frame_index in [start, end).

    At the ends of the video the missing neighbour is replaced by the current frame.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start - 1))

    def read():
        ok, f = cap.read()
        return f if ok else None

    prev = read() if start > 0 else None
    cur = read()
    if cur is None:
        return
    prev = cur if prev is None else prev
    for i in range(start, end):
        nxt = read()
        yield i, (prev, cur, cur if nxt is None else nxt)
        if nxt is None:
            return
        prev, cur = cur, nxt


def perceive(video_path: str | Path, ball_det, pose_est, start: int = 0, end: int | None = None,
             pose_stride: int = 1, table: Table | None = None, batch: int = 16,
             progress: Callable[[float], None] | None = None) -> Perception:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end = total if end is None or end <= 0 else min(end, total)
    start = max(0, start)
    n = end - start
    if table is None:
        table = find_table(cap, total)
    net_x = table.net_x if table else None

    cands = np.full((n, MAX_CANDS, 3), np.nan, np.float32)
    poses = {s: np.full((n, 17, 3), np.nan, np.float32) for s in SIDES}
    boxes = {s: np.full((n, 4), np.nan, np.float32) for s in SIDES}

    pending: list[tuple[int, tuple]] = []  # (index, triplet)
    prev_centres: dict[str, np.ndarray | None] = {"left": None, "right": None}

    def flush():
        nonlocal prev_centres
        if not pending:
            return
        idx = [i for i, _ in pending]
        for i, cs in zip(idx, ball_det([t for _, t in pending])):
            for j, c in enumerate(cs[:MAX_CANDS]):
                cands[i, j] = c
        pose_items = [(i, t[1]) for i, t in pending if i % pose_stride == 0]
        if pose_est is not None and pose_items:
            for (i, _), people in zip(pose_items, pose_est([f for _, f in pose_items])):
                chosen = select_players(people, w, net_x, prev_centres)
                for s, p in chosen.items():
                    if p is not None:
                        poses[s][i], boxes[s][i] = p.kpts, p.box
                        prev_centres[s] = np.array([(p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2])
        pending.clear()

    emitted = 0
    for i, trip in iter_triplets(cap, start, end):
        pending.append((i - start, trip))
        emitted += 1
        if len(pending) >= batch:
            flush()
            if progress:
                progress(emitted / n)
    flush()
    cap.release()

    if emitted < n:  # container frame count can overestimate
        cands = cands[:emitted]
        poses = {s: a[:emitted] for s, a in poses.items()}
        boxes = {s: a[:emitted] for s, a in boxes.items()}
    for s in SIDES:
        poses[s] = interpolate_kpts(poses[s], max_gap=max(8, 2 * pose_stride))
    return Perception(start, fps, w, h, cands, poses, boxes, table)


def cache_path(video_name: str, ball_det, pose_est, start: int, end: int | None, pose_stride: int) -> Path:
    key = f"{getattr(ball_det, 'name', '?')}|{getattr(pose_est, 'name', 'none')}|{start}|{end}|{pose_stride}"
    return CACHE_DIR / video_name / f"{hashlib.md5(key.encode()).hexdigest()[:12]}.npz"


def perceive_cached(video_name: str, video_path, ball_det, pose_est, start=0, end=None, pose_stride=1,
                    refresh: bool = False, **kw) -> Perception:
    path = cache_path(video_name, ball_det, pose_est, start, end, pose_stride)
    if path.exists() and not refresh:
        return Perception.load(path)
    p = perceive(video_path, ball_det, pose_est, start, end, pose_stride, **kw)
    p.save(path)
    return p
