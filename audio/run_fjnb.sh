#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PY:-python}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
mkdir -p outputs/logs

run() {
    local name="$1"
    local rc
    shift
    echo "START $name GPU=$GPU LOG=$ROOT/outputs/logs/$name.log"
    if CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$@" > "outputs/logs/$name.log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    echo "$rc" > "outputs/logs/$name.rc"
    echo DONE > "outputs/logs/$name.done"
    echo "DONE $name RC=$rc"
    return "$rc"
}

run nsynth_fjnb_raw scripts/train_fjnb.py \
    --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_fkan_noff_q6_lr1em3_bs4096_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --width 256 \
    --depth 4 --degree 6 --num_bands 0 \
    --sigma 1 --lr 0.001 --batch_size 4096 \
    --max_steps 5000 --log_every 100 --device "$DEVICE" \
    --seed 0

run librispeech_fjnb_raw scripts/train_fjnb.py \
    --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_fkan_noff_q2_lr1em3_bs4096_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --width 256 \
    --depth 4 --degree 2 --num_bands 0 \
    --sigma 1 --lr 0.001 --batch_size 4096 \
    --max_steps 5000 --log_every 100 --device "$DEVICE" \
    --seed 0

run nsynth_fjnb_ff16 scripts/train_fjnb.py \
    --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_fkan_ff16_s100_q6_lr1em3_bs4096_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --width 256 \
    --depth 4 --degree 6 --num_bands 16 \
    --sigma 100 --ff_seed 0 --lr 0.001 \
    --batch_size 4096 --max_steps 5000 --log_every 100 \
    --device "$DEVICE" --seed 0

run librispeech_fjnb_ff16 scripts/train_fjnb.py \
    --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_fkan_ff16_s100_q6_lr1em3_bs4096_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --width 256 \
    --depth 4 --degree 6 --num_bands 16 \
    --sigma 100 --ff_seed 0 --lr 0.001 \
    --batch_size 4096 --max_steps 5000 --log_every 100 \
    --device "$DEVICE" --seed 0
