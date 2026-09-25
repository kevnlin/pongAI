#!/usr/bin/env bash
# One-time setup on the GPU server after `git clone`.
#   bash scripts/setup_server.sh            # env + labels + test videos
#   VIDEOS=all bash scripts/setup_server.sh # also the 5 training videos (~28 GB more)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
CUDA_INDEX=${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}
VIDEOS=${VIDEOS:-test}

if [ ! -d .venv ]; then
  $PY -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url "$CUDA_INDEX"
pip install -r requirements.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

[ -f .env ] || cp .env.example .env   # add your OPENAI_API_KEY here (git-ignored)
python -m scripts.download_data --labels --videos "$VIDEOS"
echo "done. next: see README 'Training on the GPU server'"
