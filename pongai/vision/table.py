"""Table-top localisation and image <-> table-plane mapping.

Table coordinates (metres): u runs along the table length from the LEFT end (0) to the RIGHT
end (2.74) as seen in the video; v runs across the table from the FAR edge (0) to the NEAR
edge (1.525). The net is at u = 1.37.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pongai.config import TABLE_LENGTH_M, TABLE_WIDTH_M


@dataclass
class Table:
    # image corners in order: far-left, far-right, near-right, near-left
    corners: np.ndarray  # (4, 2) float32
    source: str = "auto"  # "auto" | "manual"

    def __post_init__(self):
        self.corners = np.asarray(self.corners, dtype=np.float32).reshape(4, 2)
        dst = np.float32([[0, 0], [TABLE_LENGTH_M, 0], [TABLE_LENGTH_M, TABLE_WIDTH_M], [0, TABLE_WIDTH_M]])
        self.H = cv2.getPerspectiveTransform(self.corners, dst)

    @property
    def length_px(self) -> float:
        """Mean image length of the far and near long edges (ball flies roughly mid-table)."""
        fl, fr, nr, nl = self.corners
        return float((np.linalg.norm(fr - fl) + np.linalg.norm(nr - nl)) / 2)

    @property
    def m_per_px(self) -> float:
        return TABLE_LENGTH_M / self.length_px

    @property
    def net_x(self) -> float:
        """Image x of the table centre (the net), projected through the homography.

        Averaging the corners would be biased by perspective (the near edge looks longer).
        """
        centre = np.float32([[[TABLE_LENGTH_M / 2, TABLE_WIDTH_M / 2]]])
        return float(cv2.perspectiveTransform(centre, np.linalg.inv(self.H))[0, 0, 0])

    def half(self, x: float, y: float) -> str:
        """Which half ("left" / "right") of the table an image point on the surface lies over."""
        return "left" if self.to_table([[x, y]])[0, 0] < TABLE_LENGTH_M / 2 else "right"

    @property
    def x_range(self) -> tuple[float, float]:
        return float(self.corners[:, 0].min()), float(self.corners[:, 0].max())

    def surface_y(self, x: float) -> float:
        """Approximate image y of the table surface's near edge line at image column x."""
        _, _, nr, nl = self.corners
        t = (x - nl[0]) / max(nr[0] - nl[0], 1e-6)
        return float(nl[1] + t * (nr[1] - nl[1]))

    def to_table(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {"corners": self.corners.round(1).tolist(), "source": self.source,
                "net_x": round(self.net_x, 1), "length_px": round(self.length_px, 1)}


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 arbitrary points as far-left, far-right, near-right, near-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    by_y = pts[np.argsort(pts[:, 1])]
    far = by_y[:2][np.argsort(by_y[:2, 0])]
    near = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.stack([far[0], far[1], near[1], near[0]])


def detect_table(frame: np.ndarray) -> Table | None:
    """Find a blue/green table top by colour + shape. Returns None when not confident.

    Works for standard blue and green tables filmed roughly from the side. The table must be
    a large bright saturated region that does not touch the top of the frame (which rules out
    blue backdrops like the one in the OpenTTGames venue).
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    candidates = []
    for lo, hi in (((100, 120, 130), (130, 255, 255)),   # blue table
                   ((50, 80, 60), (90, 255, 255))):      # green table
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            # the net splits the top into two halves, so these size limits apply per half
            if y <= 2 or bw < 0.12 * w or bw < 2.0 * bh:
                continue
            area = cv2.contourArea(c)
            if area < 0.005 * w * h:
                continue
            candidates.append((area, c))
    if not candidates:
        return None
    best_area, best = max(candidates, key=lambda t: t[0])
    parts = [best] + [c for a, c in candidates
                      if c is not best and a > 0.3 * best_area and _same_row(c, best)]
    hull = cv2.convexHull(np.vstack(parts))
    if cv2.boundingRect(hull)[2] < 0.25 * w:
        return None
    quad = _hull_to_quad(hull)
    if quad is None:
        return None
    return Table(order_corners(quad), source="auto")


def _same_row(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two contours overlap strongly in their vertical extent (the two table halves)."""
    _, ya, _, ha = cv2.boundingRect(a)
    _, yb, _, hb = cv2.boundingRect(b)
    overlap = min(ya + ha, yb + hb) - max(ya, yb)
    return overlap > 0.5 * min(ha, hb)


def _hull_to_quad(hull: np.ndarray) -> np.ndarray | None:
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    # fallback: extreme points of the hull
    p = hull.reshape(-1, 2).astype(np.float32)
    s, d = p.sum(1), p[:, 0] - p[:, 1]
    quad = np.stack([p[np.argmin(s)], p[np.argmax(d)], p[np.argmax(s)], p[np.argmin(d)]])
    return quad if len(np.unique(quad, axis=0)) == 4 else None
