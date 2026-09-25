"""Local web app: upload a match video, get the analysis dashboard.

    python -m app.server            # http://localhost:8000
    python -m app.server --host 0.0.0.0 --port 8000   # on the GPU server (then ssh -L 8000:localhost:8000)
"""
from __future__ import annotations

import argparse
import json
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pongai.coach.llm import llm_available, model_name
from pongai.config import ROOT, WEIGHTS_DIR
from pongai.pipeline import analyze_video

UPLOADS = ROOT / "uploads"
STATIC = Path(__file__).parent / "static"
ALLOWED = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

app = FastAPI(title="PongAI")
jobs: dict[str, dict] = {}
lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)  # one GPU job at a time


def _run(job_id: str, video: Path, side: str, corners: list | None, use_llm: bool):
    def progress(stage: str, p: float):
        with lock:
            jobs[job_id].update(stage=stage, progress=round(float(p), 3))

    with lock:
        jobs[job_id]["status"] = "running"
    try:
        analyze_video(video, video.parent, side=side, corners=corners, use_llm=use_llm, progress=progress)
        with lock:
            jobs[job_id].update(status="done", stage="done", progress=1.0)
    except Exception as e:
        traceback.print_exc()
        with lock:
            jobs[job_id].update(status="error", error=f"{type(e).__name__}: {e}")


@app.get("/api/status")
def status():
    import torch

    return {"llm": llm_available(), "llm_model": model_name() if llm_available() else None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "ball_model": "trained" if (WEIGHTS_DIR / "ball" / "ball.json").exists() else "motion baseline",
            "stroke_model": "trained" if (WEIGHTS_DIR / "stroke" / "stroke_clf.joblib").exists() else "untrained"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), side: str = Form("right"), corners: str = Form(""),
                  use_llm: bool = Form(True)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"unsupported file type {ext or '?'}; use one of {sorted(ALLOWED)}")
    if side not in ("left", "right"):
        raise HTTPException(400, "side must be left or right")
    pts = None
    if corners:
        try:
            pts = json.loads(corners)
            assert len(pts) == 4 and all(len(p) == 2 for p in pts)
        except Exception:
            raise HTTPException(400, "corners must be a JSON list of 4 [x, y] points")

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOADS / job_id
    job_dir.mkdir(parents=True)
    video = job_dir / f"input{ext}"
    with video.open("wb") as fh:
        shutil.copyfileobj(file.file, fh, length=4 << 20)
    with lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0.0,
                        "filename": file.filename}
    executor.submit(_run, job_id, video, side, pts, use_llm)
    return {"id": job_id}


def _job(job_id: str) -> dict:
    with lock:
        if job_id not in jobs:
            raise HTTPException(404, "unknown job")
        return dict(jobs[job_id])


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    return _job(job_id)


@app.get("/api/jobs/{job_id}/report")
def report(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(400, "bad job id")
    path = UPLOADS / job_id / "report.json"
    if not path.exists():
        raise HTTPException(404, "report not ready")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/jobs/{job_id}/overlay.mp4")
def overlay(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(400, "bad job id")
    path = UPLOADS / job_id / "overlay.mp4"
    if not path.exists():
        raise HTTPException(404, "overlay not ready")
    return FileResponse(path, media_type="video/mp4")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
