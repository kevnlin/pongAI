# PongAI: table tennis video coach

Upload a video of a table tennis match. PongAI tracks the ball and both players' skeletons, then works out:

- **Strokes:** when the right-side player hits the ball, and what kind of stroke it was (forehand/backhand, loop/push/…, weight/lean, feet).
- **Shot stats:** estimated shot speed and where each shot landed.
- **Rallies:** who won each point.
- **Coaching:** tips from built-in rules, and optionally from a vision-language model.

Everything is scored against the **Extended OpenTTGames** ground truth
([moamal01/table_tennis_data](https://github.com/moamal01/table_tennis_data), CC BY-NC-SA 4.0).

```
video ─► perception.py ─► events.py ─► strokes.py ─► report.py ─► coach/ ─► app/
         ball (YOLO11 /    hits,        features +     speed,        rules +     FastAPI +
         motion) + pose    bounces,     classifiers    placement,    GPT / VLM   HTML dashboard
         (YOLO11-pose) +   net, rallies (hand, tech,   rallies
         table             from ball    lean, feet)
```

| Module | What it does |
|---|---|
| `pongai/data/` | Loads and cleans the labels (typos like `xright_…` and `back_heavyn` are fixed), train/val/test splits |
| `pongai/vision/ball.py` | Ball detectors (`motion` baseline, `coco` zero-shot, fine-tuned YOLO11 using `rgb` or `stack3` input) and a multi-track tracker |
| `pongai/vision/pose.py` | YOLO11-pose for all people in the frame, then picks the left and right players (the umpire is excluded) |
| `pongai/vision/table.py` | Finds the table top automatically (or from 4 clicked corners), maps image points onto the table in metres, locates the net |
| `pongai/events.py` | Detects hits (ball changes horizontal direction), bounces, net crossings, and rallies with a guessed ending |
| `pongai/strokes.py` | 263 pose and ball features per stroke; gradient-boosted classifiers |
| `pongai/pipeline.py` | End-to-end analysis of one video: `report.json` plus `overlay.mp4` |
| `scripts/evaluate.py` | **Scores everything against the ground truth**, saves each run and appends it to `results/history.csv` |
| `scripts/eval_vlm.py` | Scores a vision-language model on the same stroke labels, with no training (zero-shot) |
| `training/` | Builds the ball dataset, fine-tunes YOLO, trains the stroke classifiers |
| `app/` | Local web UI: drag-and-drop upload, optional table-corner clicks, dashboard |

## Dataset facts that matter
- 12 videos (5 train `game_*`, 7 test `test_*`), 1920×1080 at 120 fps, one fixed camera side-on to the table, about 33 GB in total.
- **Ball positions only exist in windows around rallies.** Frames without a label are *unknown*, not "no ball", so ball metrics only use labelled frames.
- A plain `net` label is the ball **crossing** the net. `left_net` / `right_net` are rally endings, where the ball hit the net or the player's own side.
- Rally endings are prefixed by the player who *caused* them. That player wins the point only for `winner` and `double_bounce`.
- `test_2` has no bounce labels. `test_3`, `test_5` and `test_7` include left-handed players.
- Splits: train on `game_1`–`game_4`, validate on `game_5`, **never train on `test_*`**.

## Quick start (laptop, CPU)
```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use .venv/bin/activate on Linux
pip install -r requirements.txt
python -m scripts.download_data --labels --videos test_2   # 215 MB test video
python -m app.server                                        # open http://localhost:8000
```
With no trained weights the app falls back to the motion ball baseline, and stroke types show as `unknown`.

## Training on the GPU server
```bash
git clone <your repo> pongAI && cd pongAI
VIDEOS=all bash scripts/setup_server.sh          # CUDA torch + deps + labels + all 12 videos
source .venv/bin/activate

# 0. baseline numbers before any training (goes into results/history.csv)
python -m scripts.evaluate --videos test --ball motion --tag baseline-motion
python -m scripts.evaluate --videos test --ball coco   --tag baseline-coco

# 1. ball detector
python -m training.build_ball_dataset --mode stack3 --stride 3
python -m training.train_ball --data datasets/ball_stack3 --model yolo11s.pt --epochs 60
python -m scripts.evaluate --videos test --tag yolo11s-stack3          # picks up weights/ball automatically
#    (try --mode rgb too, and compare the two rows in results/history.csv)

# 2. stroke classifiers (validation on game_5 is printed, then refit on all 5 games)
python -m training.train_strokes --final
python -m scripts.evaluate --videos test --tag strokes-v1

# 3. optional: zero-shot VLM on the same labels (≈60 API calls with gpt-4o-mini, a few cents)
#    put OPENAI_API_KEY=sk-... in .env (git-ignored; setup_server.sh creates it from .env.example)
python -m scripts.eval_vlm --videos test --max 60

# 4. run the app on the server and open it on your laptop
python -m app.server --host 127.0.0.1 --port 8000
# on your laptop:  ssh -L 8000:localhost:8000 user@server   then open http://localhost:8000
```
Commit `results/` after each run so the history travels with the repo. To use the trained models
on your laptop, copy `weights/` back (`scp -r user@server:pongAI/weights .`).

### Free VLM instead of GPT
Any OpenAI-compatible server works. On the GPU server:
```bash
pip install vllm && vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001
# then in .env:
#   PONGAI_LLM_BASE_URL=http://localhost:8001/v1
#   PONGAI_LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
#   PONGAI_LLM_API_KEY=none
```

### Secrets
API keys go in `.env` at the repo root (copy `.env.example`). It is loaded automatically by `pongai/config.py`
and is git-ignored, so keys are never committed. Real environment variables take precedence.

## Metrics (`scripts/evaluate.py`)
| Metric | Meaning |
|---|---|
| `ball recall@10px` | Share of labelled visible-ball frames where the tracked ball is within 10 px (at 1920 wide) |
| `bounce_f1` | Bounce detection F1, counting a match within ±4 frames (test_2 excluded) |
| `hit_right_f1` / `hit_left_f1` | Stroke-contact detection F1, counting a match within ±6 frames |
| `oracle_*` | Classifier accuracy at the **ground-truth** contact frames (isolates the classifier) |
| `e2e_*` | Classifier accuracy on **detected** hits that matched a ground-truth stroke (full system) |
| `rally_coverage` / `rally_winner_acc` | Share of labelled rally endings inside a detected rally, and how often the point winner is right |

Baseline measured locally on `test_2` (CPU, motion ball, no training): ball recall@10px 0.76 and right-hit F1 0.39.

## Recording tips for your own videos
Film from the side, at the height of the table, with the camera still (a tripod) and as much of the table in view as
possible. 60 fps or more helps a lot (the dataset is 120 fps). If the table isn't found automatically, click its
4 corners in the upload screen.

## Roadmap
- A learned hit/bounce detector (a temporal model over the ball path and pose) to replace the heuristics in `events.py`
- A temporal stroke model (e.g. a 1D-CNN or transformer over keypoint sequences) as an alternative to the tree models
- Fine-tuning a small VLM (Qwen2.5-VL + LoRA) on the stroke labels, compared on `eval_vlm.py`
- Form feedback against reference technique (comparing joint-angle profiles with a pro template for each stroke type)
