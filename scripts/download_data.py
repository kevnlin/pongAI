"""Download the Extended OpenTTGames labels and videos (resumable).

    python -m scripts.download_data --labels                 # annotations only (~2 MB)
    python -m scripts.download_data --videos test            # 7 test videos (~5.6 GB)
    python -m scripts.download_data --videos all             # all 12 videos (~33 GB)
    python -m scripts.download_data --videos game_4,test_2   # specific videos
"""
from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile

from pongai.config import LABELS_DIR, ROOT, VIDEOS_DIR
from pongai.data.dataset import VideoItem, resolve

LABELS_ZIP = "https://github.com/moamal01/table_tennis_data/archive/refs/heads/main.zip"
VIDEO_URL = "https://lab.osai.ai/datasets/openttgames/data/{name}.mp4"


def download_labels(force: bool = False):
    dest = ROOT / "external" / "table_tennis_data"
    if LABELS_DIR.exists() and not force:
        print(f"labels already at {LABELS_DIR}")
        return
    print("downloading labels ...")
    data = urllib.request.urlopen(LABELS_ZIP).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(ROOT / "external")
    src = ROOT / "external" / "table_tennis_data-main"
    if dest.exists():
        shutil.rmtree(dest)
    src.rename(dest)
    print(f"labels -> {dest}")


def download_video(name: str):
    path = VideoItem(name).video_path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = VIDEO_URL.format(name=name)
    total = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    if path.exists() and path.stat().st_size == total:
        print(f"{name}: already complete")
        return
    part = path.with_suffix(".mp4.part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r, part.open("ab" if have else "wb") as f:
        done, last = have, -1
        while chunk := r.read(8 << 20):
            f.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total)
            if pct // 5 != last:
                last = pct // 5
                print(f"\r{name}: {done / 2**30:.2f} / {total / 2**30:.2f} GB ({pct}%)", end="", flush=True)
    part.rename(path)
    print(f"\n{name} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--videos", default=None, help="all | train | test | comma-separated names")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.labels or not LABELS_DIR.exists():
        download_labels(args.force)
    if args.videos:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        for name in resolve(args.videos):
            download_video(name)


if __name__ == "__main__":
    main()
