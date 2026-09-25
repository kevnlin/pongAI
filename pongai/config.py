"""Project-wide paths and constants. Paths can be overridden with env vars on the server."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Load KEY=value lines from .env (git-ignored) without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")

LABELS_DIR = Path(os.environ.get("PONGAI_LABELS", ROOT / "external" / "table_tennis_data" / "data" / "raw"))
VIDEOS_DIR = Path(os.environ.get("PONGAI_VIDEOS", ROOT / "data" / "videos"))
CACHE_DIR = Path(os.environ.get("PONGAI_CACHE", ROOT / "cache"))
WEIGHTS_DIR = Path(os.environ.get("PONGAI_WEIGHTS", ROOT / "weights"))
RESULTS_DIR = ROOT / "results"

# ITTF regulation table
TABLE_LENGTH_M = 2.74
TABLE_WIDTH_M = 1.525

# Dataset videos are 1920x1080 @ 120 fps. Pixel/frame thresholds are expressed at this
# reference and rescaled to the actual video so user uploads at other resolutions/fps work.
REF_WIDTH = 1920
REF_FPS = 120.0
