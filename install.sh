#!/usr/bin/env bash
# ============================================================================
# VoiceClone installer — provisions the RVC (Applio) engine + Python env.
# Idempotent / re-runnable. Run it from the project root after placing app.py
# and run-vc.sh there (default root: /ai/VoiceClone).
#
#   bash install.sh
#
# Requirements on the host: NVIDIA GPU + driver (CUDA 12.x+), git, curl,
# ffmpeg, and internet access. No root / sudo needed.
# ============================================================================
set -euo pipefail

ROOT="${VOICECLONE_ROOT:-/ai/VoiceClone}"
APPLIO_DIR="$ROOT/engine/Applio"
VENV="$ROOT/.venv"
# Pinned for reproducibility (the version this deployment was built/verified on).
APPLIO_COMMIT="${APPLIO_COMMIT:-dc9fa3b5fe91d2c2be9eb391526b29ccb3b9d22b}"
PY_VERSION="3.12"

log() { printf "\n\033[1;32m== %s ==\033[0m\n" "$*"; }
die() { printf "\n\033[1;31m!! %s\033[0m\n" "$*" >&2; exit 1; }

command -v git    >/dev/null || die "git is required"
command -v curl   >/dev/null || die "curl is required"
command -v ffmpeg >/dev/null || die "ffmpeg is required (apt install ffmpeg)"

log "Workspace: $ROOT"
mkdir -p "$ROOT"/{engine,voices,data,outputs,logs}
cd "$ROOT"
[ -f "$ROOT/app.py" ]     || die "app.py not found in $ROOT — copy the project files here first"
[ -f "$ROOT/run-vc.sh" ]  || die "run-vc.sh not found in $ROOT — copy the project files here first"

# 1) uv (Python/venv manager) ------------------------------------------------
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 2) Python venv -------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  log "Creating Python $PY_VERSION venv"
  uv venv "$VENV" --python "$PY_VERSION"
fi
PY="$VENV/bin/python"
"$PY" --version

# 3) Applio engine (pinned) --------------------------------------------------
if [ ! -d "$APPLIO_DIR/.git" ]; then
  log "Cloning Applio"
  git clone https://github.com/IAHispano/Applio.git "$APPLIO_DIR"
fi
cd "$APPLIO_DIR"
log "Pinning Applio to $APPLIO_COMMIT"
git fetch origin "$APPLIO_COMMIT" 2>/dev/null || git fetch origin
git checkout "$APPLIO_COMMIT" 2>/dev/null || echo "(could not pin commit; staying on current branch)"

# 4) Python dependencies (CUDA 12.8 wheels) ----------------------------------
log "Installing dependencies (torch+cu128 — large, can take a few minutes)"
export UV_HTTP_TIMEOUT=300
uv pip install --python "$PY" -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
uv pip install --python "$PY" python-ffmpeg

# 5) Verify torch sees the GPU ----------------------------------------------
log "Verifying torch / CUDA"
"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PYEOF

# 6) Prerequisite models (~1.8 GB: ContentVec, RMVPE, HiFi-GAN bases) --------
log "Downloading prerequisite models (~1.8 GB)"
"$PY" core.py prerequisites --models True --pretraineds_hifigan True --exe True

# 7) Applio needs assets/config.json (else trained weights are never saved) ---
[ -f assets/config.json ] || cp assets/config_template.json assets/config.json

# 8) Local credentials file (untracked). Edit it to set the real password. -----
if [ ! -f "$ROOT/.env" ] && [ -f "$ROOT/.env.example" ]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  log "Created $ROOT/.env from .env.example — EDIT IT to set VOICECLONE_PASSWORD"
fi

log "Install complete."
cat <<EOF

Start the UI:
  tmux new-session -d -s voiceclone "bash $ROOT/run-vc.sh >> $ROOT/logs/app.log 2>&1"

Then browse to  http://<host>:${VOICECLONE_PORT:-7865}   (login user 'kxkm'; set the password in $ROOT/.env -> VOICECLONE_PASSWORD).
Logs:  tail -f $ROOT/logs/app.log
EOF
