"""Access to the Extended OpenTTGames videos and ground-truth annotations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pongai.config import LABELS_DIR, VIDEOS_DIR
from pongai.data.labels import Event, parse_label

TRAIN_VIDEOS = [f"game_{i}" for i in range(1, 6)]
TEST_VIDEOS = [f"test_{i}" for i in range(1, 8)]
# game_5 is held out from training for model selection; test_* is never trained on.
VAL_VIDEOS = ["game_5"]
FIT_VIDEOS = [v for v in TRAIN_VIDEOS if v not in VAL_VIDEOS]
# test_2 has stroke labels but no bounce / net-crossing labels.
NO_BOUNCE_LABELS = {"test_2"}
# Videos with left-handed players (README).
LEFT_HANDED = {"test_3": "one", "test_5": "both", "test_7": "one"}
# Which players those are; the test_3 / test_7 side was read from the video frames.
LEFT_HANDED_PLAYERS = {("test_3", "right"), ("test_5", "left"), ("test_5", "right"), ("test_7", "right")}


def racket_truth(name: str) -> dict[str, str]:
    """True racket hand ("R"/"L") of each player; every player not listed in the README is right-handed."""
    return {s: ("L" if (name, s) in LEFT_HANDED_PLAYERS else "R") for s in ("left", "right")}


@dataclass(frozen=True)
class VideoItem:
    name: str

    @property
    def split(self) -> str:
        return "train" if self.name.startswith("game_") else "test"

    @property
    def video_path(self) -> Path:
        return VIDEOS_DIR / self.split / f"{self.name}.mp4"

    @property
    def events_path(self) -> Path:
        return LABELS_DIR / "game_data" / self.split / f"{self.name}.json"

    @property
    def ball_path(self) -> Path:
        # ball files for training videos are named train_i instead of game_i
        return LABELS_DIR / "ball_data" / self.split / f"{self.name.replace('game_', 'train_')}.json"


def resolve(names: str | list[str]) -> list[str]:
    """Accept 'train', 'test', 'fit', 'val', 'all' or explicit comma-separated names."""
    if isinstance(names, list):
        return names
    groups = {"train": TRAIN_VIDEOS, "test": TEST_VIDEOS, "fit": FIT_VIDEOS, "val": VAL_VIDEOS,
              "all": TRAIN_VIDEOS + TEST_VIDEOS}
    out: list[str] = []
    for n in names.split(","):
        out += groups.get(n.strip(), [n.strip()])
    return out


@lru_cache(maxsize=None)
def load_events(name: str) -> tuple[Event, ...]:
    raw = json.loads(VideoItem(name).events_path.read_text())
    events = (parse_label(int(k), v) for k, v in raw.items())
    return tuple(sorted((e for e in events if e is not None), key=lambda e: e.frame))


@lru_cache(maxsize=None)
def load_ball(name: str) -> dict[int, tuple[int, int] | None]:
    """Frame -> (x, y) pixel position, or None when annotated as not visible.

    Frames absent from the dict are UNLABELLED (the annotation only covers windows around
    rallies), so they must not be counted as negatives.
    """
    raw = json.loads(VideoItem(name).ball_path.read_text())
    return {int(k): (None if v["x"] < 0 else (v["x"], v["y"])) for k, v in raw.items()}


def gt_trajectory(name: str, start: int, n: int) -> "np.ndarray":
    """Ground-truth ball as an (n, 2) trajectory array (NaN when unknown / not visible)."""
    import numpy as np

    traj = np.full((n, 2), np.nan)
    for f, xy in load_ball(name).items():
        if xy is not None and start <= f < start + n:
            traj[f - start] = xy
    return traj


def strokes(name: str, side: str | None = None) -> list[Event]:
    return [e for e in load_events(name) if e.kind == "stroke" and (side is None or e.side == side)]
