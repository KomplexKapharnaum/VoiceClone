# VoiceClone — RVC timbre transfer (speech)

Self-hosted **voice conversion**: clone a voice from a few source files, then
convert any speech recording into that voice. Runs on a single NVIDIA GPU.

- **Engine:** [Applio](https://github.com/IAHispano/Applio) (an RVC distribution),
  driven head-less through its CLI.
- **UI:** a small custom **Gradio** app (`app.py`) — 5 tabs, login-protected.
- **Currently deployed on:** `kxkm-ai` (RTX 4090, Ubuntu 24.04) at `/ai/VoiceClone`,
  published at **https://voice.kxkm.net** via the `kxkm-prod` nginx reverse proxy.

---

## 1. Quick re-deploy (from scratch)

On a host with an NVIDIA GPU + driver, `git`, `curl`, `ffmpeg` and internet:

```bash
# 1. Put the project files in place
mkdir -p /ai/VoiceClone && cd /ai/VoiceClone
#    copy app.py, run-vc.sh, install.sh, README.md, nginx-voiceclone.conf.example here

# 2. Provision engine + Python env + models (idempotent, ~5-15 min)
bash install.sh

# 3. Launch (persists across logout in tmux)
tmux new-session -d -s voiceclone "bash /ai/VoiceClone/run-vc.sh >> /ai/VoiceClone/logs/app.log 2>&1"
```

Browse to `http://<host>:7865`. Login is user **kxkm**; set the password via
`VOICECLONE_PASSWORD` in `/ai/VoiceClone/.env` (copy `.env.example` → `.env`).
`.env` is gitignored and never committed.

`install.sh` does: install `uv` → create a Python 3.12 venv → clone Applio
(pinned commit) → install deps (torch **cu128**) → download ~1.8 GB of base
models → ensure `assets/config.json` exists. Re-running it is safe.

Override defaults with env vars: `VOICECLONE_ROOT`, `VOICECLONE_PORT`,
`APPLIO_COMMIT`.

---

## 2. Layout

```
/ai/VoiceClone/
├── app.py                       # the Gradio web UI (5 tabs)
├── run-vc.sh                    # launcher (ensures assets/config.json, then runs app.py)
├── install.sh                   # full provisioner (this repo)
├── README.md
├── nginx-voiceclone.conf.example
├── .venv/                       # uv venv (Python 3.12, torch 2.7.1+cu128)
├── engine/Applio/               # RVC engine (CLI: core.py)
│   ├── assets/config.json       # REQUIRED — created from config_template.json
│   ├── rvc/models/              # downloaded bases (ContentVec, RMVPE, HiFi-GAN)
│   └── logs/<voice>/            # trained models: <voice>_<E>e_<S>s.pth + <voice>.index
├── voices/<voice>.json          # per-voice "sweet spot" preset (from the Tune tab)
├── data/<voice>/                # staged training audio (auto-filled from uploads)
├── outputs/                     # converted files (history)
└── logs/app.log                 # UI / server log
```

---

## 3. Running it

The UI runs in a tmux session named `voiceclone`.

```bash
# start / restart
tmux kill-session -t voiceclone 2>/dev/null
tmux new-session -d -s voiceclone 'bash /ai/VoiceClone/run-vc.sh >> /ai/VoiceClone/logs/app.log 2>&1'

tail -f /ai/VoiceClone/logs/app.log     # logs
tmux attach -t voiceclone               # attach (Ctrl-b d to detach)
```

**Auto-start after reboot** (already installed on `kxkm-ai` — to set up elsewhere,
add to your crontab with `crontab -e`):

```
@reboot tmux new-session -d -s voiceclone "bash /ai/VoiceClone/run-vc.sh >> /ai/VoiceClone/logs/app.log 2>&1"
```

---

## 4. Access

- **Private / quick test (SSH tunnel):**
  `ssh -L 7865:localhost:7865 kxkm-ai` → http://localhost:7865
- **Public:** `https://voice.kxkm.net`, terminated by nginx on **`kxkm-prod`**,
  proxied to the app over Tailscale at `100.87.54.119:7865` (the LAN IP is not
  routable from prod). TLS uses the `*.kxkm.net` wildcard cert. See
  `nginx-voiceclone.conf.example` — note the **WebSocket upgrade** headers,
  `proxy_buffering off` (live log/SSE streaming) and large `client_max_body_size`.
  The proxy config lives at `/etc/nginx/sites-available/voice.conf` on `kxkm-prod`
  (Nginx-UI–managed; symlinked into `sites-enabled`).

Change login / port by editing the `launch(...)` call at the bottom of `app.py`.

---

## 5. Using the UI (workflow order)

**① Clone a voice** — name it, upload clean single-speaker audio (several minutes
total is best), train (10–30 min on the GPU). A checkpoint is saved every 50
epochs. Live progress bar + log.

**② Tune / Compare** — find the sweet spot on a short clip:
- *Epoch sweep*: render the clip with each saved checkpoint → hear which epoch is
  best (RVC overtrains, so the last isn't always best).
- *Parameter sweep*: vary one setting (index rate / protect / pitch) on a grid.
- **★ Set as default** saves the winner (checkpoint + params) to `voices/<voice>.json`;
  the Transform tab applies it automatically.

**③ Transform a file** — pick a voice (its preset auto-loads), upload audio,
convert. Result has an inline player **and a download link**. Key options have
tooltips. Use *Auto-match pitch* when source/target genders differ.

**④ Results** — every converted file, with player + download + delete.

**⑤ Manage voices** — list (name · checkpoints · size · date · ⭐preset) and delete
old attempts (removes model, training data, and preset).

---

## 6. Versions / key facts

| Item | Value |
|---|---|
| OS / GPU | Ubuntu 24.04 · RTX 4090 (24 GB) |
| Python | 3.12 (uv-managed venv) |
| Torch | 2.7.1+cu128 |
| Engine | Applio @ `dc9fa3b` |
| Embedder / pitch | ContentVec · RMVPE |
| Default training | 48 kHz, HiFi-GAN vocoder, save every 50 epochs |
| Port / login | 7865 · user `kxkm` (password via `.env` `VOICECLONE_PASSWORD`) |

---

## 7. Troubleshooting

- **502 from voice.kxkm.net** → app not listening or Tailscale down. Check
  `ss -ltn | grep 7865` on `kxkm-ai` and `nginx -t && systemctl reload nginx` on prod.
- **Training finishes but no usable model / Transform can't find it** → Applio's
  `assets/config.json` is missing (weights aren't exported without it). `run-vc.sh`
  and `install.sh` create it from `config_template.json`.
- **`extract: error: --include_mutes is required`** → the app already passes it;
  if calling the CLI by hand, add `--include_mutes 2`.
- **CUDA not available** → reinstall torch with the cu128 index (see `install.sh`).
- **Poor clone quality** → give more clean, single-speaker audio; in Tune, pick a
  lower epoch if it sounds raspy/robotic (overtraining) and tune index/protect.

---

## 8. Maintenance

```bash
# update the engine (then re-run deps)
git -C /ai/VoiceClone/engine/Applio pull
export PATH="$HOME/.local/bin:$PATH"
uv pip install --python /ai/VoiceClone/.venv -r /ai/VoiceClone/engine/Applio/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match

# disk: trained models live in engine/Applio/logs/<voice>/ (G_/D_ checkpoints are
# the largest). Converted files accumulate in outputs/ — prune from the Results tab.
```
