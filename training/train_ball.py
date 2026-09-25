"""Fine-tune YOLO11 as a ball detector on the dataset from build_ball_dataset.py.

    python -m training.train_ball --data datasets/ball_stack3 --model yolo11s.pt --epochs 60

Copies the best checkpoint to weights/ball/ with a ball.json describing the input mode, which
the pipeline and evaluate.py pick up automatically (--ball auto).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pongai.config import ROOT, WEIGHTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "ball_stack3"))
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=str(WEIGHTS_DIR / "ball"))
    args = ap.parse_args()

    from ultralytics import YOLO

    data = Path(args.data)
    meta = json.loads((data / "meta.json").read_text())
    mode = meta["mode"]
    name = args.name or f"{Path(args.model).stem}_{mode}"

    aug = dict(fliplr=0.5, flipud=0.0, degrees=0.0, scale=0.3, translate=0.1, mosaic=0.5, close_mosaic=10)
    if mode == "stack3":
        # colour jitter would break the temporal meaning of the channels
        aug.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.2)

    model = YOLO(args.model)
    model.train(data=str(data / "data.yaml"), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                device=args.device, workers=args.workers, project=str(ROOT / "runs" / "ball"), name=name,
                patience=15, exist_ok=True, **aug)

    best = Path(model.trainer.best)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out / "best.pt")
    (out / "ball.json").write_text(json.dumps(
        {"weights": "best.pt", "mode": mode, "imgsz": args.imgsz, "conf": 0.15, "base": args.model,
         "run": name}, indent=2))
    print(f"saved {out / 'best.pt'} (mode={mode}); evaluate with: python -m scripts.evaluate --ball {out}")


if __name__ == "__main__":
    main()
