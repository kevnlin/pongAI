"""Vision-language coaching through any OpenAI-compatible chat API.

Default: OpenAI (set OPENAI_API_KEY; model gpt-4o-mini, a few cents per report).
Free alternative: serve an open VLM on the GPU server with an OpenAI-compatible endpoint and
point the client at it, e.g.

    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001
    export PONGAI_LLM_BASE_URL=http://localhost:8001/v1 PONGAI_LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct PONGAI_LLM_API_KEY=none

or Ollama (`ollama pull qwen2.5vl`, base url http://localhost:11434/v1).
"""
from __future__ import annotations

import base64
import json
import os

import cv2
import numpy as np

import pongai.config  # noqa: F401  (loads API keys from the git-ignored .env)

SYSTEM = (
    "You are an experienced table tennis coach. You receive measurements from a computer-vision "
    "system (ball tracking, pose estimation, stroke classification) plus still frames of the player "
    "at the moment of contact with the skeleton drawn on. Measurements are estimates and may contain "
    "detection errors; weigh them accordingly and do not invent numbers that are not given."
)

PROMPT = """Player analysed: the {side}-side player.

Measured session report (JSON):
{report}

Rule-based observations already shown to the player:
{tips}

The attached images are the player at ball contact for several strokes (label above each).

Write concise coaching feedback in Markdown with these sections:
### Summary  (2-3 sentences)
### Strengths  (2-3 bullets)
### Priorities  (the 3 most important improvements; for each: what you see, why it matters, one drill)
### Form notes from the frames  (grip/racket angle, elbow, stance, weight transfer; only what is visible)
Keep it under 350 words."""


def _client():
    from openai import OpenAI

    key = os.environ.get("PONGAI_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("PONGAI_LLM_BASE_URL")
    if not key and not base:
        return None
    return OpenAI(api_key=key or "none", base_url=base) if base else OpenAI(api_key=key)


def model_name() -> str:
    return os.environ.get("PONGAI_LLM_MODEL", "gpt-4o-mini")


def llm_available() -> bool:
    return bool(os.environ.get("PONGAI_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("PONGAI_LLM_BASE_URL"))


def image_part(img: np.ndarray, max_side: int = 768) -> dict:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}}


def _compact(report: dict) -> dict:
    """Drop per-frame detail the model does not need."""
    keep = {k: report[k] for k in ("totals", "speed", "technique_counts", "lean_counts", "feet_counts")}
    keep["racket_hand"] = report.get("racket_hand")
    keep["rallies"] = [{k: r[k] for k in ("n_strokes", "outcome", "won")} for r in report["rallies"]]
    keep["shots"] = [{k: s.get(k) for k in ("n", "hand", "technique", "lean", "feet", "speed_kmh")}
                     for s in report["shots"][:40]]
    return keep


def coach_feedback(report: dict, tips: list[dict], keyframes: list[tuple[str, np.ndarray]]) -> str | None:
    client = _client()
    if client is None:
        return None
    content = [{"type": "text", "text": PROMPT.format(
        side=report["player_side"], report=json.dumps(_compact(report)),
        tips="\n".join(f"- {t['title']}: {t['evidence']}" for t in tips))}]
    for caption, img in keyframes[:6]:
        content += [{"type": "text", "text": caption}, image_part(img)]
    resp = client.chat.completions.create(
        model=model_name(), max_tokens=900, temperature=0.4,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}])
    return resp.choices[0].message.content


HAND_WORD = {"R": "right", "L": "left"}


def stroke_prompt(vocab: dict[str, list[str]], side: str, racket: str | None = None) -> str:
    """Prompt for classify_stroke; also used to build the LoRA fine-tuning set (training/build_vlm_dataset.py).
    racket ("R"/"L") states the player's racket hand; None keeps the original prompt."""
    spec = "\n".join(f"- {k}: one of {v}" for k, v in vocab.items())
    holds = f" They hold the racket in their {HAND_WORD[racket]} hand." if racket else ""
    return (f"These frames (left to right, ~33 ms apart, last-but-one is ball contact) show the {side}-side "
            f"table tennis player hitting the ball, with their skeleton drawn.{holds} Classify the stroke.\n{spec}\n"
            "Lean = where the player's weight is at contact; feet = which feet are off the ground at contact. "
            'Answer with JSON only, e.g. {"hand": "...", "technique": "...", "lean": "...", "feet": "..."}')


def racket_prompt(side: str) -> str:
    return (f"These frames (left to right) show the {side}-side table tennis player hitting the ball, with their "
            "skeleton drawn. Which of the player's own hands holds the racket? "
            'Answer with JSON only: {"racket_hand": "left"} or {"racket_hand": "right"}')


def _ask_json(text: str, strip: np.ndarray, max_tokens: int) -> dict:
    client = _client()
    if client is None:
        raise RuntimeError("no LLM configured: set OPENAI_API_KEY or PONGAI_LLM_BASE_URL")
    resp = client.chat.completions.create(
        model=model_name(), max_tokens=max_tokens, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [{"type": "text", "text": text}, image_part(strip, 1536)]}])
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return {}


def classify_stroke(strip: np.ndarray, vocab: dict[str, list[str]], side: str, racket: str | None = None) -> dict:
    """Stroke labelling of a frame strip (used by scripts/eval_vlm.py)."""
    return _ask_json(stroke_prompt(vocab, side, racket), strip, 100)


def classify_racket_hand(strip: np.ndarray, side: str) -> str:
    """"R" / "L" / "?" for one stroke strip; vote over a player's strokes for a per-player answer."""
    ans = str(_ask_json(racket_prompt(side), strip, 30).get("racket_hand", "")).lower()
    return {"right": "R", "left": "L"}.get(ans, "?")
