#!/bin/bash
# Applio dependency install into the uv venv. Logs to install.log.
set -o pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=300
cd /ai/VoiceClone/engine/Applio || exit 99
VENV=/ai/VoiceClone/.venv
PY="$VENV/bin/python"

echo "=== install start $(date) ==="
echo "python: $($PY --version 2>&1)"

uv pip install --python "$PY" -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
rc_req=$?

uv pip install --python "$PY" python-ffmpeg
rc_ffmpeg=$?

echo "=== verify torch/cuda ==="
"$PY" - <<'PYEOF'
try:
    import torch, torchaudio
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
          "device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"))
except Exception as e:
    print("TORCH_IMPORT_ERROR", repr(e))
PYEOF

echo "INSTALL_DONE rc_req=$rc_req rc_ffmpeg=$rc_ffmpeg date=$(date)"
