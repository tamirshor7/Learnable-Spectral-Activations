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

run nsynth_siren scripts/train_core.py \
    --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_siren_noff_48k_audit_v1 \
    --method siren --sample_rate 48000 --n_samples 48000 \
    --width 256 --depth 4 --num_bands 0 \
    --sigma 1 --w0 30 --lr 0.0005 \
    --lr_decay 0.2 --decay_every 5000 --max_steps 5000 \
    --batch_size 0 --log_every 250 --seed 0 \
    --device "$DEVICE" --force

run librispeech_siren scripts/train_core.py \
    --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_siren_noff_48k_audit_v1 \
    --method siren --sample_rate 48000 --n_samples 48000 \
    --width 256 --depth 4 --num_bands 0 \
    --sigma 1 --w0 30 --lr 0.0005 \
    --lr_decay 0.2 --decay_every 5000 --max_steps 5000 \
    --batch_size 0 --log_every 250 --seed 0 \
    --device "$DEVICE" --force
