#!/bin/bash
# End-to-end pipeline smoke test with synthetic audio (tiny epochs).
set -e
PY=/ai/VoiceClone/.venv/bin/python
cd /ai/VoiceClone/engine/Applio
M=_smoke
DS=/ai/VoiceClone/data/$M
rm -rf "$DS" logs/$M; mkdir -p "$DS"

i=0
for f in 110 130 150 170 190 210; do
  ffmpeg -nostdin -hide_banner -loglevel error -f lavfi \
    -i "sine=frequency=$f:duration=4" -ar 40000 "$DS/src_$i.wav"
  i=$((i+1))
done
echo "[audio generated: $i files]"

$PY core.py preprocess --model_name $M --dataset_path "$DS" --sample_rate 40000 \
  --cut_preprocess Automatic --cpu_cores 8 --process_effects True --noise_reduction False
echo "[preprocess ok]"

$PY core.py extract --model_name $M --f0_method rmvpe --embedder_model contentvec \
  --sample_rate 40000 --gpu 0 --cpu_cores 8 --include_mutes 2
echo "[extract ok]"

$PY core.py train --model_name $M --sample_rate 40000 --total_epoch 2 \
  --save_every_epoch 2 --batch_size 4 --gpu 0 --vocoder "HiFi-GAN" --pretrained True \
  --save_only_latest True --save_every_weights True --overtraining_detector False \
  --index_algorithm Auto
echo "[train ok]"

$PY core.py index --model_name $M --index_algorithm Auto
echo "[index ok]"

PTH=$(ls -t logs/$M/*.pth | grep -vE '/[GD]_[0-9]' | head -1)
IDX=$(ls logs/$M/*.index 2>/dev/null | head -1)
echo "PTH=$PTH"
echo "IDX=$IDX"

$PY core.py infer --input_path "$DS/src_0.wav" \
  --output_path /ai/VoiceClone/outputs/_smoke_out.wav \
  --pth_path "$PTH" --index_path "$IDX" --f0_method rmvpe --pitch 0 \
  --proposed_pitch False --index_rate 0.7 --protect 0.5 --volume_envelope 1.0 \
  --embedder_model contentvec --clean_audio False --split_audio True --export_format WAV
echo "[infer ok]"
ls -la /ai/VoiceClone/outputs/_smoke_out.wav
echo "SMOKE_DONE"
