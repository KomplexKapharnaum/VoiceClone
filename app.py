#!/usr/bin/env python3
"""
VoiceClone — a web UI over the Applio (RVC) engine.

  ① Clone     : upload source audio -> train a reusable voice model
  ② Transform : pick a voice -> convert an audio file's timbre to it (downloadable)
  ③ Tune      : sweep epochs / parameters on a clip, compare, save the best as default
  ④ Results   : every converted file, with player + download
  ⑤ Manage    : list / delete previously created voices

Drives Applio's documented CLI (core.py) as subprocesses.
"""
import os
import re
import glob
import json
import time
import shutil
import subprocess

import gradio as gr

# ---------------------------------------------------------------- paths / config
ROOT       = "/ai/VoiceClone"
APPLIO_DIR = os.path.join(ROOT, "engine", "Applio")
VENV_PY    = os.path.join(ROOT, ".venv", "bin", "python")
CORE       = os.path.join(APPLIO_DIR, "core.py")
LOGS_DIR   = os.path.join(APPLIO_DIR, "logs")     # Applio stores trained models here
DATA_DIR   = os.path.join(ROOT, "data")           # per-voice training datasets
OUT_DIR    = os.path.join(ROOT, "outputs")        # converted files
VOICES_DIR = os.path.join(ROOT, "voices")         # per-voice presets (sweet spots)

for d in (DATA_DIR, OUT_DIR, LOGS_DIR, VOICES_DIR):
    os.makedirs(d, exist_ok=True)

N_CORES = min(os.cpu_count() or 4, 16)
MAX_VARIANTS = 12  # max audio players shown in the Tune tab

NON_MODEL_DIRS = {"mute", "mute_spin", "mute_spin-v2", "reference", "zips"}

INDEX_GRID   = [0.0, 0.25, 0.5, 0.75, 1.0]
PROTECT_GRID = [0.0, 0.17, 0.33, 0.5]
PITCH_GRID   = [-12, -7, -5, 0, 5, 7, 12]


# ---------------------------------------------------------------- helpers
def sanitize(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return name.strip("_")


def find_pth(model_dir: str):
    """Newest deployable weight (exclude full G_/D_ training checkpoints)."""
    cands = [p for p in glob.glob(os.path.join(model_dir, "*.pth"))
             if not os.path.basename(p).startswith(("G_", "D_"))]
    return max(cands, key=os.path.getmtime) if cands else None


def find_index(model_dir: str, name: str):
    pref = os.path.join(model_dir, f"{name}.index")
    if os.path.isfile(pref):
        return pref
    others = glob.glob(os.path.join(model_dir, "*.index"))
    return others[0] if others else None


def list_models():
    out = []
    if os.path.isdir(LOGS_DIR):
        for name in sorted(os.listdir(LOGS_DIR)):
            d = os.path.join(LOGS_DIR, name)
            if not os.path.isdir(d) or name in NON_MODEL_DIRS:
                continue
            if find_pth(d) and find_index(d, name):
                out.append(name)
    return out


def refresh_models():
    return gr.update(choices=list_models())


def _human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def _dir_size(d):
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def epoch_of(pth):
    m = re.search(r"_(\d+)e_\d+s\.pth$", os.path.basename(pth))
    return int(m.group(1)) if m else -1


def list_checkpoints(name):
    """Deployable epoch weights for a voice, sorted by epoch."""
    md = os.path.join(LOGS_DIR, sanitize(name))
    cks = [p for p in glob.glob(os.path.join(md, "*.pth"))
           if not os.path.basename(p).startswith(("G_", "D_"))]
    return sorted(cks, key=epoch_of)


# ---- per-voice presets (the "sweet spot" chosen in the Tune tab) -------------
def preset_path(name):
    return os.path.join(VOICES_DIR, f"{sanitize(name)}.json")


def load_preset(name):
    p = preset_path(name)
    if os.path.isfile(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_preset(name, d):
    with open(preset_path(name), "w") as f:
        json.dump(d, f, indent=2)


def resolve_checkpoint(name):
    """Preset checkpoint if it still exists, else the newest one."""
    cp = load_preset(name).get("checkpoint")
    if cp and os.path.isfile(cp):
        return cp
    return find_pth(os.path.join(LOGS_DIR, sanitize(name)))


def voice_table():
    names = list_models()
    rows = []
    for n in names:
        d = os.path.join(LOGS_DIR, n)
        star = "⭐" if load_preset(n).get("checkpoint") else ""
        rows.append([n, len(list_checkpoints(n)), _human(_dir_size(d)),
                     time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(d))), star])
    return rows, names


def refresh_manage():
    rows, names = voice_table()
    return gr.update(value=rows), gr.update(choices=names, value=[])


# ---- outputs / history ------------------------------------------------------
def list_outputs():
    files = []
    if os.path.isdir(OUT_DIR):
        for f in os.listdir(OUT_DIR):
            if f.startswith((".", "_")):       # skip temp (_tune_) / hidden
                continue
            p = os.path.join(OUT_DIR, f)
            if os.path.isfile(p):
                files.append(p)
    return sorted(files, key=os.path.getmtime, reverse=True)


def output_rows():
    return [[os.path.basename(p),
             time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p))),
             _human(os.path.getsize(p))] for p in list_outputs()]


def refresh_results():
    names = [os.path.basename(p) for p in list_outputs()]
    return (gr.update(value=output_rows()),
            gr.update(choices=names, value=(names[0] if names else None)))


def select_result(name):
    p = os.path.join(OUT_DIR, name) if name else None
    if p and os.path.isfile(p):
        return p, p
    return None, None


def delete_result(name):
    if name:
        p = os.path.join(OUT_DIR, name)
        if os.path.isfile(p):
            os.remove(p)
    tbl, dd = refresh_results()
    return tbl, dd, None, None


def stream(cmd):
    """Yield output lines of a subprocess; final line is '__RC__<code>'."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, cwd=APPLIO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        yield line.rstrip("\n")
    proc.wait()
    yield f"__RC__{proc.returncode}"


def infer_once(model, audio, checkpoint, index_rate, protect, pitch,
               proposed_pitch, f0_method, out_path):
    """One blocking conversion; returns the produced file path or None."""
    md = os.path.join(LOGS_DIR, sanitize(model))
    idx = find_index(md, model)
    if not (checkpoint and os.path.isfile(checkpoint) and idx):
        return None
    cmd = [
        VENV_PY, CORE, "infer",
        "--input_path", audio, "--output_path", out_path,
        "--pth_path", checkpoint, "--index_path", idx,
        "--f0_method", f0_method, "--pitch", str(int(pitch)),
        "--proposed_pitch", "True" if proposed_pitch else "False",
        "--index_rate", str(round(float(index_rate), 2)),
        "--protect", str(round(float(protect), 3)),
        "--volume_envelope", "1.0", "--embedder_model", "contentvec",
        "--clean_audio", "False", "--split_audio", "True", "--export_format", "WAV",
    ]
    r = subprocess.run(cmd, cwd=APPLIO_DIR, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        return None
    if os.path.isfile(out_path):
        return out_path
    cands = glob.glob(out_path.rsplit(".", 1)[0] + ".*")
    return cands[0] if cands else None


# ---------------------------------------------------------------- ① CLONE
def do_clone(name, files, sample_rate, total_epoch, batch_size, f0_method,
             progress=gr.Progress()):
    log = ""

    def emit(msg):
        nonlocal log
        log += msg + "\n"
        return log

    model = sanitize(name)
    if not model:
        yield emit("❌ Please enter a voice name (letters, digits, _ or -).")
        return
    if not files:
        yield emit("❌ Please upload at least one source audio file.")
        return

    sr = str(int(sample_rate))
    te = int(total_epoch)
    progress(0.0, desc="Staging files")

    ds = os.path.join(DATA_DIR, model)
    if os.path.isdir(ds):
        shutil.rmtree(ds)
    os.makedirs(ds, exist_ok=True)
    for i, f in enumerate(files):
        src = f if isinstance(f, str) else getattr(f, "name", None)
        if not src:
            continue
        ext = os.path.splitext(src)[1] or ".wav"
        shutil.copy(src, os.path.join(ds, f"src_{i:02d}{ext}"))
    yield emit(f"📁 Voice '{model}' — staged {len(files)} file(s) at {sr} Hz.")

    steps = [
        ("preprocess", "Preprocess (slice & clean)", (0.02, 0.10), [
            VENV_PY, CORE, "preprocess",
            "--model_name", model, "--dataset_path", ds,
            "--sample_rate", sr, "--cut_preprocess", "Automatic",
            "--cpu_cores", str(N_CORES), "--process_effects", "True",
            "--noise_reduction", "False",
        ]),
        ("extract", "Extract features (RMVPE + ContentVec)", (0.10, 0.25), [
            VENV_PY, CORE, "extract",
            "--model_name", model, "--f0_method", f0_method,
            "--embedder_model", "contentvec", "--sample_rate", sr,
            "--gpu", "0", "--cpu_cores", str(N_CORES), "--include_mutes", "2",
        ]),
        ("train", "Train model (GPU)", (0.25, 0.95), [
            VENV_PY, CORE, "train",
            "--model_name", model, "--sample_rate", sr,
            "--total_epoch", str(te),
            "--save_every_epoch", "50", "--batch_size", str(int(batch_size)),
            "--gpu", "0", "--vocoder", "HiFi-GAN", "--pretrained", "True",
            "--save_only_latest", "True", "--save_every_weights", "True",
            "--overtraining_detector", "False", "--index_algorithm", "Auto",
        ]),
        ("index", "Build retrieval index", (0.95, 1.0), [
            VENV_PY, CORE, "index",
            "--model_name", model, "--index_algorithm", "Auto",
        ]),
    ]

    for key, title, (lo, hi), cmd in steps:
        progress(lo, desc=title)
        yield emit(f"\n=== {title} ===")
        for line in stream(cmd):
            if line.startswith("__RC__"):
                rc = int(line[6:])
                if rc != 0:
                    progress(lo, desc=f"{title} — failed")
                    yield emit(f"❌ Step failed (exit {rc}). See log above.")
                    return
            else:
                if key == "train" and te > 0:
                    m = re.search(r"epoch=(\d+)", line)
                    if m:
                        ep = int(m.group(1))
                        frac = lo + (hi - lo) * min(ep / te, 1.0)
                        progress(frac, desc=f"Training — epoch {ep}/{te}")
                if line.strip():
                    yield emit(line)
        progress(hi, desc=f"{title} ✓")

    md = os.path.join(LOGS_DIR, model)
    pth, idx = find_pth(md), find_index(md, model)
    progress(1.0, desc="Done")
    if pth and idx:
        yield emit(f"\n✅ Done. Voice '{model}' is ready.\n   model: {pth}\n   index: {idx}"
                   f"\n→ Use it in the Transform tab, or tune it in the Tune tab.")
    else:
        yield emit("\n⚠️ Training finished but model/index not found — check the log.")


# ---------------------------------------------------------------- ② TRANSFORM
def on_select_voice(model):
    """Auto-apply the voice's saved sweet-spot preset to the sliders."""
    blank = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    if not model:
        return (*blank, "")
    pre = load_preset(model)
    md = os.path.join(LOGS_DIR, sanitize(model))
    if not pre:
        cp = find_pth(md)
        note = ("ℹ️ No saved preset — using the newest checkpoint"
                + (f" (`{os.path.basename(cp)}`)" if cp else "")
                + " with the settings below.")
        return (*blank, note)
    cp = pre.get("checkpoint")
    note = (f"⭐ Using saved preset — checkpoint `{os.path.basename(cp) if cp else 'newest'}`, "
            f"index {pre.get('index_rate')}, protect {pre.get('protect')}, "
            f"pitch {pre.get('pitch')}, auto-pitch {'on' if pre.get('proposed_pitch') else 'off'}.")
    return (gr.update(value=pre.get("pitch", 0)),
            gr.update(value=bool(pre.get("proposed_pitch", False))),
            gr.update(value=pre.get("index_rate", 0.7)),
            gr.update(value=pre.get("protect", 0.5)),
            gr.update(value=pre.get("f0_method", "rmvpe")),
            note)


def do_transform(model, audio_path, pitch, proposed_pitch, index_rate,
                 protect, f0_method, clean_audio, split_audio, export_format):
    log = ""

    def emit(msg):
        nonlocal log
        log += msg + "\n"
        return log, None, None

    if not model:
        yield emit("❌ Pick a voice model (click 🔄 if the list is empty).")
        return
    if not audio_path:
        yield emit("❌ Upload an audio file to transform.")
        return

    md = os.path.join(LOGS_DIR, sanitize(model))
    pth, idx = resolve_checkpoint(model), find_index(md, model)
    if not (pth and idx):
        yield emit(f"❌ Could not locate model files for '{model}'.")
        return

    stem = os.path.splitext(os.path.basename(audio_path))[0]
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(OUT_DIR, f"{model}__{stem}__{ts}.{export_format.lower()}")

    cmd = [
        VENV_PY, CORE, "infer",
        "--input_path", audio_path, "--output_path", out_path,
        "--pth_path", pth, "--index_path", idx,
        "--f0_method", f0_method, "--pitch", str(int(pitch)),
        "--proposed_pitch", "True" if proposed_pitch else "False",
        "--index_rate", str(round(float(index_rate), 2)),
        "--protect", str(round(float(protect), 3)),
        "--volume_envelope", "1.0", "--embedder_model", "contentvec",
        "--clean_audio", "True" if clean_audio else "False",
        "--clean_strength", "0.5",
        "--split_audio", "True" if split_audio else "False",
        "--export_format", export_format,
    ]

    yield emit(f"🎙️  Converting '{stem}' → voice '{model}' …  ⏳ (using `{os.path.basename(pth)}`)")
    for line in stream(cmd):
        if line.startswith("__RC__"):
            rc = int(line[6:])
            if rc != 0:
                yield emit(f"❌ Conversion failed (exit {rc}).")
                return
        elif line.strip():
            yield emit(line)

    final = out_path
    if not os.path.isfile(final):
        cands = glob.glob(os.path.join(OUT_DIR, f"{model}__{stem}__{ts}.*"))
        final = cands[0] if cands else None

    if final and os.path.isfile(final):
        log += f"\n✅ Saved: {final}\n"
        yield log, final, final
    else:
        yield emit("⚠️ Finished but output file not found — check the log.")


# ---------------------------------------------------------------- ③ TUNE
def _audio_updates(results):
    ups = []
    for i in range(MAX_VARIANTS):
        if i < len(results) and results[i].get("audio"):
            ups.append(gr.update(value=results[i]["audio"], visible=True,
                                 label=results[i]["label"]))
        else:
            ups.append(gr.update(value=None, visible=False))
    return ups


def toggle_param(mode):
    return gr.update(visible=(mode == "Parameter sweep"))


def do_tune(model, audio, mode, param, progress=gr.Progress()):
    if not model or not audio:
        return ["❌ Pick a voice and upload a short reference clip.", [],
                gr.update(choices=[], value=None)] + _audio_updates([])

    base = load_preset(model)
    b = dict(index_rate=base.get("index_rate", 0.7),
             protect=base.get("protect", 0.5),
             pitch=base.get("pitch", 0),
             proposed_pitch=bool(base.get("proposed_pitch", False)),
             f0_method=base.get("f0_method", "rmvpe"))

    variants = []
    if mode == "Epoch sweep":
        cks = list_checkpoints(model)
        if not cks:
            return ["❌ No checkpoints found for this voice.", [],
                    gr.update(choices=[], value=None)] + _audio_updates([])
        if len(cks) > MAX_VARIANTS:                # evenly subsample, keep ends
            step = (len(cks) - 1) / (MAX_VARIANTS - 1)
            cks = [cks[round(i * step)] for i in range(MAX_VARIANTS)]
            cks = list(dict.fromkeys(cks))
        for cp in cks:
            variants.append({**b, "checkpoint": cp, "label": f"epoch {epoch_of(cp)}"})
    else:
        cp = resolve_checkpoint(model)
        if not cp:
            return ["❌ No checkpoint found for this voice.", [],
                    gr.update(choices=[], value=None)] + _audio_updates([])
        grid = {"Index rate": INDEX_GRID, "Protect": PROTECT_GRID,
                "Pitch (semitones)": PITCH_GRID}.get(param, INDEX_GRID)
        for v in grid:
            spec = {**b, "checkpoint": cp}
            if param == "Index rate":
                spec["index_rate"] = v; spec["label"] = f"index {v}"
            elif param == "Protect":
                spec["protect"] = v; spec["label"] = f"protect {v}"
            else:
                spec["pitch"] = v; spec["label"] = f"pitch {v:+d}"
            variants.append(spec)

    results, n = [], len(variants)
    for i, spec in enumerate(variants):
        progress(i / max(n, 1), desc=f"Rendering {spec['label']} ({i+1}/{n})")
        out = os.path.join(OUT_DIR, f"_tune_{sanitize(model)}_{i}.wav")
        spec["audio"] = infer_once(model, audio, spec["checkpoint"],
                                   spec["index_rate"], spec["protect"], spec["pitch"],
                                   spec["proposed_pitch"], spec["f0_method"], out)
        results.append(spec)
    progress(1.0, desc="Done")

    ok = [r for r in results if r.get("audio")]
    keep = [r["label"] for r in ok]
    status = (f"✅ Rendered {len(ok)}/{n} variants. Listen below, then pick the best "
              f"and click **★ Set as default**." if ok else "❌ All renders failed — check the model.")
    return [status, results, gr.update(choices=keep, value=(keep[0] if keep else None))] \
        + _audio_updates(results)


def set_default(model, results, keep_label):
    if not model:
        return "⚠️ No voice selected."
    if not results or not keep_label:
        return "⚠️ Run a comparison and pick a variant first."
    chosen = next((r for r in results if r["label"] == keep_label), None)
    if not chosen:
        return "⚠️ Could not find that variant."
    preset = load_preset(model)
    preset.update({k: chosen[k] for k in
                   ("checkpoint", "index_rate", "protect", "pitch", "proposed_pitch", "f0_method")})
    save_preset(model, preset)
    cp = os.path.basename(chosen["checkpoint"]) if chosen.get("checkpoint") else "newest"
    return (f"⭐ Saved **{keep_label}** as default for **{model}** (checkpoint `{cp}`). "
            f"The Transform tab now uses it automatically.")


# ---------------------------------------------------------------- ⑤ MANAGE
def delete_voices(names, confirm):
    rows, current = voice_table()
    if not names:
        return ("⚠️ Select at least one voice first.",
                gr.update(value=rows), gr.update(choices=current, value=[]),
                gr.update(choices=current))
    if not confirm:
        return ("⚠️ Tick the confirmation box to delete.",
                gr.update(value=rows), gr.update(choices=current, value=names),
                gr.update(choices=current))
    deleted = []
    for nm in names:
        nm = sanitize(nm)
        if not nm:
            continue
        for p in (os.path.join(LOGS_DIR, nm), os.path.join(DATA_DIR, nm)):
            shutil.rmtree(p, ignore_errors=True)
        pp = preset_path(nm)
        if os.path.isfile(pp):
            os.remove(pp)
        deleted.append(nm)
    rows, current = voice_table()
    return (f"✅ Deleted: {', '.join(deleted)}",
            gr.update(value=rows), gr.update(choices=current, value=[]),
            gr.update(choices=current))


# ---------------------------------------------------------------- UI
CSS = """
.wrap-log textarea {font-family: monospace; font-size: 12px;}
/* spinner shown ONLY while a player's block is in a loading state */
.preparing.generating, .preparing.pending {position: relative;}
.preparing.generating::after, .preparing.pending::after {
  content: ""; position: absolute; top: 50%; left: 50%;
  width: 26px; height: 26px; margin: -13px 0 0 -13px; border-radius: 50%;
  border: 3px solid var(--border-color-primary); border-top-color: var(--color-accent);
  animation: vc-spin 0.8s linear infinite; z-index: 5; pointer-events: none;}
@keyframes vc-spin {to {transform: rotate(360deg);}}
"""

with gr.Blocks(title="VoiceClone (RVC)", css=CSS) as demo:
    gr.Markdown(
        "# 🎚️ VoiceClone — timbre transfer (RVC / Applio)\n"
        "**①** Clone · **②** Tune/Compare · **③** Transform · **④** Results · **⑤** Manage"
    )

    # ---------------------------------------------------------- ① Clone
    with gr.Tab("① Clone a voice"):
        gr.Markdown("Upload **clean, single-speaker** audio of the target voice. "
                    "Training runs on the GPU and can take 10–30 min.")
        with gr.Row():
            with gr.Column(scale=1):
                c_name = gr.Textbox(
                    label="Voice name", placeholder="e.g. alice",
                    info="Short name (letters, digits, _ or -) used to store and "
                         "later select this voice.")
                c_files = gr.File(
                    label="Source audio file(s)", file_count="multiple",
                    file_types=["audio"], type="filepath")
                gr.Markdown("<small>Tip: several minutes of clean, single-speaker "
                            "speech (no music/noise) gives the best clone.</small>")
                with gr.Accordion("Training options", open=False):
                    c_sr = gr.Radio(
                        [40000, 48000], value=48000, label="Sample rate (Hz)",
                        info="Quality the model trains at. 48 kHz = best fidelity "
                             "(recommended); 40 kHz trains a bit faster / less VRAM.")
                    c_epoch = gr.Slider(
                        50, 1000, value=300, step=10, label="Total epochs",
                        info="How long to train. Too few = doesn't capture the voice; "
                             "too many = overtraining artifacts. 200–400 suits a few "
                             "minutes of speech. (A checkpoint is saved every 50 epochs "
                             "so you can compare them in the Tune tab.)")
                    c_bs = gr.Slider(
                        1, 24, value=8, step=1, label="Batch size",
                        info="Samples per training step. Higher = faster but more "
                             "GPU memory; the 4090 handles 8–16. Lower if out-of-memory.")
                    c_f0 = gr.Dropdown(
                        ["rmvpe", "crepe", "fcpe"], value="rmvpe",
                        label="Pitch extraction",
                        info="Algorithm used to learn the voice's pitch. rmvpe is the "
                             "most accurate for speech.")
                c_btn = gr.Button("① Create cloned voice", variant="primary")
            with gr.Column(scale=2):
                c_log = gr.Textbox(label="Progress", lines=26,
                                   elem_classes="wrap-log", show_copy_button=True)

    # ---------------------------------------------------------- ② Tune
    with gr.Tab("② Tune / Compare") as tune_tab:
        gr.Markdown("Find the **sweet spot** for a voice: render a short clip several "
                    "ways, compare side-by-side, and save the winner as the voice's "
                    "default (used automatically in Transform).")
        with gr.Row():
            with gr.Column(scale=1):
                tu_model = gr.Dropdown(choices=list_models(), label="Voice model",
                                       info="Voice to tune. Click 🔄 if missing.")
                tu_refresh = gr.Button("🔄 Refresh voice list")
                tu_audio = gr.Audio(
                    label="Reference clip (5–15 s works best)", type="filepath",
                    elem_classes="preparing")
                tu_mode = gr.Radio(
                    ["Epoch sweep", "Parameter sweep"], value="Epoch sweep",
                    label="What to compare",
                    info="Epoch sweep = same settings, each saved training checkpoint "
                         "(finds the best-trained, non-overtrained epoch). "
                         "Parameter sweep = fixed checkpoint, vary one setting.")
                tu_param = gr.Dropdown(
                    ["Index rate", "Protect", "Pitch (semitones)"], value="Index rate",
                    label="Parameter to sweep", visible=False,
                    info="Which setting to vary across a small grid of values.")
                tu_run = gr.Button("▶️ Render comparison", variant="primary")
                tu_status = gr.Markdown()
                with gr.Row():
                    tu_keep = gr.Dropdown(choices=[], label="Best variant to keep")
                    tu_save = gr.Button("★ Set as default", variant="primary")
                tu_save_status = gr.Markdown()
            with gr.Column(scale=2):
                gr.Markdown("**Variants** — play back-to-back to compare:")
                tu_audios = [gr.Audio(visible=False, type="filepath",
                                      interactive=False, elem_classes="preparing")
                             for _ in range(MAX_VARIANTS)]
        tu_state = gr.State([])

    # ---------------------------------------------------------- ③ Transform
    with gr.Tab("③ Transform a file") as transform_tab:
        with gr.Row():
            with gr.Column(scale=1):
                t_model = gr.Dropdown(
                    choices=list_models(), label="Voice model",
                    info="The cloned voice to convert into. Train one in the Clone "
                         "tab; click 🔄 if it's missing.")
                t_refresh = gr.Button("🔄 Refresh voice list")
                t_note = gr.Markdown()
                t_audio = gr.Audio(label="Input audio (to convert)", type="filepath")
                with gr.Accordion("Conversion options", open=False):
                    t_proposed = gr.Checkbox(
                        value=False, label="Auto-match pitch to target",
                        info="Shifts the source pitch into the target's natural range. "
                             "Turn ON when source and target are different "
                             "genders/registers; leave OFF when they're similar.")
                    t_pitch = gr.Slider(
                        -12, 12, value=0, step=1, label="Manual pitch shift (semitones)",
                        info="Transpose by semitones (+12 = one octave up). Use only if "
                             "auto-match is off and the result sounds too high/low. "
                             "0 = keep original pitch.")
                    t_index = gr.Slider(
                        0.0, 1.0, value=0.7, step=0.05, label="Index rate (timbre strength)",
                        info="How strongly the target's timbre is applied. Higher = more "
                             "like the target but more artifacts; lower = cleaner but "
                             "less resemblance. 0.5–0.8 is typical.")
                    t_protect = gr.Slider(
                        0.0, 0.5, value=0.5, step=0.01, label="Protect consonants/breath",
                        info="Shields breaths and unvoiced consonants (s, t, f) from "
                             "over-conversion. Higher = clearer consonants, slightly "
                             "less timbre transfer. 0.5 = max protection.")
                    t_f0 = gr.Dropdown(
                        ["rmvpe", "crepe", "fcpe"], value="rmvpe",
                        label="Pitch extraction",
                        info="Tracks the input's pitch. rmvpe = most accurate for "
                             "speech; fcpe = faster; crepe = alternative.")
                    t_clean = gr.Checkbox(
                        value=False, label="Clean / denoise output",
                        info="Runs a noise reducer on the result. Helps noisy inputs; "
                             "can dull already-clean speech.")
                    t_split = gr.Checkbox(
                        value=True, label="Split long audio",
                        info="Processes long files in chunks for stability and lower "
                             "memory. Recommended ON for anything over ~30 s.")
                    t_fmt = gr.Dropdown(
                        ["WAV", "MP3", "FLAC"], value="WAV", label="Output format",
                        info="WAV = lossless/largest; MP3/FLAC = smaller files.")
                t_btn = gr.Button("③ Transform", variant="primary")
            with gr.Column(scale=2):
                t_out = gr.Audio(label="Converted audio", type="filepath",
                                 elem_classes="preparing")
                t_download = gr.File(label="⬇️ Download converted file")
                t_log = gr.Textbox(label="Progress", lines=14,
                                   elem_classes="wrap-log", show_copy_button=True)

    # ---------------------------------------------------------- ④ Results
    with gr.Tab("④ Results") as results_tab:
        gr.Markdown("Every converted file. Select one to play and download.")
        with gr.Row():
            with gr.Column(scale=1):
                r_select = gr.Dropdown(choices=[], label="Result file (newest first)")
                with gr.Row():
                    r_refresh = gr.Button("🔄 Refresh")
                    r_delete = gr.Button("🗑️ Delete", variant="stop")
                r_player = gr.Audio(label="Playback", type="filepath",
                                    elem_classes="preparing")
                r_download = gr.File(label="⬇️ Download")
            with gr.Column(scale=1):
                r_table = gr.Dataframe(
                    headers=["File", "Date", "Size"],
                    datatype=["str", "str", "str"],
                    interactive=False, wrap=True, label="All results")

    # ---------------------------------------------------------- ⑤ Manage
    with gr.Tab("⑤ Manage voices") as manage_tab:
        gr.Markdown("Review and delete previously created voices. Deleting removes the "
                    "trained model, its staged training data, and its saved preset.")
        _rows0, _names0 = voice_table()
        manage_table = gr.Dataframe(
            value=_rows0,
            headers=["Voice", "Checkpoints", "Size on disk", "Last modified", "Preset"],
            datatype=["str", "number", "str", "str", "str"],
            interactive=False, wrap=True, label="Saved voices")
        manage_select = gr.CheckboxGroup(
            choices=_names0, label="Select voice(s) to delete",
            info="Tick the voices you want to remove.")
        manage_confirm = gr.Checkbox(
            value=False, label="I understand this permanently deletes them",
            info="Deletion cannot be undone.")
        with gr.Row():
            manage_refresh = gr.Button("🔄 Refresh")
            manage_delete = gr.Button("🗑️ Delete selected", variant="stop")
        manage_status = gr.Markdown()

    # ---------------------------------------------------------- wiring
    c_btn.click(do_clone,
                inputs=[c_name, c_files, c_sr, c_epoch, c_bs, c_f0],
                outputs=[c_log]) \
        .then(refresh_models, outputs=[t_model]) \
        .then(refresh_models, outputs=[tu_model]) \
        .then(refresh_manage, outputs=[manage_table, manage_select])

    t_refresh.click(refresh_models, outputs=[t_model])
    transform_tab.select(refresh_models, outputs=[t_model])
    t_model.change(on_select_voice, inputs=[t_model],
                   outputs=[t_pitch, t_proposed, t_index, t_protect, t_f0, t_note])
    t_btn.click(do_transform,
                inputs=[t_model, t_audio, t_pitch, t_proposed, t_index,
                        t_protect, t_f0, t_clean, t_split, t_fmt],
                outputs=[t_log, t_out, t_download], show_progress="minimal") \
        .then(refresh_results, outputs=[r_table, r_select])

    tu_refresh.click(refresh_models, outputs=[tu_model])
    tune_tab.select(refresh_models, outputs=[tu_model])
    tu_mode.change(toggle_param, inputs=[tu_mode], outputs=[tu_param])
    tu_run.click(do_tune, inputs=[tu_model, tu_audio, tu_mode, tu_param],
                 outputs=[tu_status, tu_state, tu_keep] + tu_audios,
                 show_progress="minimal")
    tu_save.click(set_default, inputs=[tu_model, tu_state, tu_keep],
                  outputs=[tu_save_status])

    results_tab.select(refresh_results, outputs=[r_table, r_select])
    r_refresh.click(refresh_results, outputs=[r_table, r_select])
    r_select.change(select_result, inputs=[r_select], outputs=[r_player, r_download])
    r_delete.click(delete_result, inputs=[r_select],
                   outputs=[r_table, r_select, r_player, r_download])

    manage_tab.select(refresh_manage, outputs=[manage_table, manage_select])
    manage_refresh.click(refresh_manage, outputs=[manage_table, manage_select])
    manage_delete.click(delete_voices,
                        inputs=[manage_select, manage_confirm],
                        outputs=[manage_status, manage_table, manage_select, t_model])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7865,
        auth=(os.environ.get("VOICECLONE_USER", "kxkm"),
              os.environ.get("VOICECLONE_PASSWORD", "kxkm")),
        show_error=True,
    )
