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

run nsynth_sl2a_512_128 scripts/train_sl2a.py \
    --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_sl2a_deg512_rank128_lr1em3_sched_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --hidden_features 256 \
    --hidden_layers 3 --deg 512 --rank 128 \
    --lr 0.001 --scheduler --scheduler_b 0.1 \
    --batch_size 0 --max_steps 5000 --log_every 100 \
    --seed 0 --device "$DEVICE"

run nsynth_sl2a_512_64 scripts/train_sl2a.py \
    --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_sl2a_deg512_rank64_lr1em3_sched_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --hidden_features 256 \
    --hidden_layers 3 --deg 512 --rank 64 \
    --lr 0.001 --scheduler --scheduler_b 0.1 \
    --batch_size 0 --max_steps 5000 --log_every 100 \
    --seed 0 --device "$DEVICE"

run nsynth_sl2a_256_128 scripts/train_sl2a.py \
    --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_sl2a_deg256_rank128_lr1em3_sched_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --hidden_features 256 \
    --hidden_layers 3 --deg 256 --rank 128 \
    --lr 0.001 --scheduler --scheduler_b 0.1 \
    --batch_size 0 --max_steps 5000 --log_every 100 \
    --seed 0 --device "$DEVICE"

run nsynth_sl2a_256_64 scripts/train_sl2a.py \
    --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_sl2a_deg256_rank64_lr1em3_sched_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --hidden_features 256 \
    --hidden_layers 3 --deg 256 --rank 64 \
    --lr 0.001 --scheduler --scheduler_b 0.1 \
    --batch_size 0 --max_steps 5000 --log_every 100 \
    --seed 0 --device "$DEVICE"

run librispeech_sl2a_512_128 scripts/train_sl2a.py \
    --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_sl2a_deg512_rank128_lr1em3_sched_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --hidden_features 256 \
    --hidden_layers 3 --deg 512 --rank 128 \
    --lr 0.001 --scheduler --scheduler_b 0.1 \
    --batch_size 0 --max_steps 5000 --log_every 100 \
    --seed 0 --device "$DEVICE"

run librispeech_sl2a_512_64 scripts/train_sl2a.py \
    --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_sl2a_deg512_rank64_lr1em3_sched_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --hidden_features 256 \
    --hidden_layers 3 --deg 512 --rank 64 \
    --lr 0.001 --scheduler --scheduler_b 0.1 \
    --batch_size 0 --max_steps 5000 --log_every 100 \
    --seed 0 --device "$DEVICE"

run librispeech_sl2a_256_128 scripts/train_sl2a.py \
    --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_sl2a_deg256_rank128_lr1em3_sched_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --hidden_features 256 \
    --hidden_layers 3 --deg 256 --rank 128 \
    --lr 0.001 --scheduler --scheduler_b 0.1 \
    --batch_size 0 --max_steps 5000 --log_every 100 \
    --seed 0 --device "$DEVICE"

run librispeech_sl2a_256_64 scripts/train_sl2a.py \
    --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_sl2a_deg256_rank64_lr1em3_sched_48k_v1 \
    --sample_rate 48000 --n_samples 48000 --hidden_features 256 \
    --hidden_layers 3 --deg 256 --rank 64 \
    --lr 0.001 --scheduler --scheduler_b 0.1 \
    --batch_size 0 --max_steps 5000 --log_every 100 \
    --seed 0 --device "$DEVICE"
