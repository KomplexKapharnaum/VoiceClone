#!/bin/bash
# Realistic end-to-end test using edge-tts speech.
set -e
PY=/ai/VoiceClone/.venv/bin/python
cd /ai/VoiceClone/engine/Applio
M=_smoke2
DS=/ai/VoiceClone/data/$M
rm -rf logs/$M

echo "[generating speech via edge-tts]"
$PY /ai/VoiceClone/gen_speech.py
ls -la "$DS"

$PY core.py preprocess --model_name $M --dataset_path "$DS" --sample_rate 40000 \
  --cut_preprocess Automatic --cpu_cores 8 --process_effects True --noise_reduction False
echo "[preprocess ok]"

$PY core.py extract --model_name $M --f0_method rmvpe --embedder_model contentvec \
  --sample_rate 40000 --gpu 0 --cpu_cores 8 --include_mutes 2
echo "[extract ok]"

$PY core.py train --model_name $M --sample_rate 40000 --total_epoch 3 \
  --save_every_epoch 1 --batch_size 6 --gpu 0 --vocoder "HiFi-GAN" --pretrained True \
  --save_only_latest True --save_every_weights True --overtraining_detector False \
  --index_algorithm Auto
echo "[train ok]"

$PY core.py index --model_name $M --index_algorithm Auto
echo "[index ok]"

PTH=$(ls -t logs/$M/*.pth 2>/dev/null | grep -vE '/[GD]_[0-9]' | head -1)
IDX=$(ls logs/$M/*.index 2>/dev/null | head -1)
echo "PTH=$PTH"
echo "IDX=$IDX"
[ -z "$PTH" ] && { echo "NO_DEPLOYABLE_WEIGHT"; ls -la logs/$M/; exit 7; }

$PY core.py infer --input_path /ai/VoiceClone/outputs/_smoke2_input.wav \
  --output_path /ai/VoiceClone/outputs/_smoke2_out.wav \
  --pth_path "$PTH" --index_path "$IDX" --f0_method rmvpe --pitch 0 \
  --proposed_pitch True --index_rate 0.7 --protect 0.5 --volume_envelope 1.0 \
  --embedder_model contentvec --clean_audio False --split_audio True --export_format WAV
echo "[infer ok]"
ls -la /ai/VoiceClone/outputs/_smoke2_out.wav
echo "SMOKE_DONE"
