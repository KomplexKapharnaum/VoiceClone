#!/bin/bash
# Launch the VoiceClone web UI (binds 0.0.0.0:7865).
# Login user/password come from .env (VOICECLONE_USER / VOICECLONE_PASSWORD);
# defaults to kxkm/kxkm if .env is absent.
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
# Applio writes deployable weights only if assets/config.json exists (normally
# created on first GUI run). We drive the CLI, so ensure it's present.
APPLIO=/ai/VoiceClone/engine/Applio
[ -f "$APPLIO/assets/config.json" ] || cp "$APPLIO/assets/config_template.json" "$APPLIO/assets/config.json"
# Load credentials / overrides from the untracked .env (never committed)
set -a; [ -f /ai/VoiceClone/.env ] && . /ai/VoiceClone/.env; set +a
cd /ai/VoiceClone
exec /ai/VoiceClone/.venv/bin/python /ai/VoiceClone/app.py
