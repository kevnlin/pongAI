"""Multi-person pose (YOLO11-pose, COCO-17 keypoints) and left/right player selection.

MediaPipe Pose is single-person, so it cannot separate the two players plus the umpire;
YOLO11-pose detects everyone in one pass and we pick the players by position and size.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

# COCO-17 keypoint indices
NOSE, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = \
    0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
# left/right keypoint pairs, used to swap sides for handedness normalisation
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


class Person(NamedTuple):
    box: np.ndarray   # (4,) x1 y1 x2 y2
    kpts: np.ndarray  # (17, 3) x y conf
    conf: float


class PoseEstimator:
    def __init__(self, weights: str | None = None, imgsz: int = 960, conf: float = 0.3, device: str | None = None):
        from ultralytics import YOLO
        import torch

        if weights is None:
            weights = "yolo11m-pose.pt" if torch.cuda.is_available() else "yolo11n-pose.pt"
        self.model = YOLO(weights)
        self.name = f"pose:{weights}"
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def __call__(self, frames: list[np.ndarray]) -> list[list[Person]]:
        results = self.model.predict(frames, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        out = []
        for r in results:
            if r.keypoints is None or len(r.boxes) == 0:
                out.append([])
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            kp = r.keypoints.data.cpu().numpy()  # (n, 17, 3)
            out.append([Person(b, k, float(c)) for b, k, c in zip(boxes, kp, confs)])
        return out


def select_players(people: list[Person], frame_w: int, net_x: float | None,
                   prev: dict[str, np.ndarray | None] | None = None) -> dict[str, Person | None]:
    """Pick the left and right player from all detected people in a frame.

    Players are the tallest people on each side of the net; the umpire sits behind the net
    and is excluded by requiring the person's centre to be clearly off the net line. When the
    previous frame's choice is known, stay close to it to avoid switching to spectators.
    """
    net_x = net_x if net_x is not None else frame_w / 2
    margin = 0.08 * frame_w
    chosen: dict[str, Person | None] = {"left": None, "right": None}
    for side in ("left", "right"):
        best, best_score = None, -np.inf
        for p in people:
            cx = (p.box[0] + p.box[2]) / 2
            if (side == "right" and cx < net_x + margin) or (side == "left" and cx > net_x - margin):
                continue
            height = p.box[3] - p.box[1]
            score = height
            if prev and prev.get(side) is not None:
                score -= 0.5 * abs(cx - prev[side][0])
            if score > best_score:
                best, best_score = p, score
        chosen[side] = best
    return chosen


def interpolate_kpts(track: np.ndarray, max_gap: int = 8) -> np.ndarray:
    """Fill short NaN gaps in a (N, 17, 3) keypoint track by linear interpolation."""
    out = track.copy()
    valid = ~np.isnan(track[:, 0, 0])
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return out
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= max_gap:
            w = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1:b] = track[a] + w[:, None, None] * (track[b] - track[a])
    return out
