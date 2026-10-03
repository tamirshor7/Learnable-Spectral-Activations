#!/usr/bin/env bash
set +e
set +u
set +o pipefail

PY="${PY:-python}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
mkdir -p outputs/logs

core() {
  NAME="$1"
  shift
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_core_audio.py "$@" > "outputs/logs/${NAME}.log" 2>&1
  echo "$?" > "outputs/logs/${NAME}.rc"
}

fjnb() {
  NAME="$1"
  shift
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_fjnb_audio.py "$@" > "outputs/logs/${NAME}.log" 2>&1
  echo "$?" > "outputs/logs/${NAME}.rc"
}

sl2a() {
  NAME="$1"
  shift
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_sl2a_audio.py "$@" > "outputs/logs/${NAME}.log" 2>&1
  echo "$?" > "outputs/logs/${NAME}.rc"
}

core nsynth_lsa --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_act16_ff16_sigma100_K16_lr2em3_am3p0_48k_full_v1 --method lsa --sample_rate 48000 --n_samples 48000 --width 256 --depth 4 --num_bands 16 --sigma 100 --K 16 --lr 0.002 --act_lr_mult 3.0 --lr_decay 0.2 --decay_every 5000 --max_steps 5000 --batch_size 0 --log_every 250 --seed 0 --device "$DEVICE" --force
core librispeech_lsa --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_act16_ff16_sigma100_K16_lr2em3_am3p0_48k_full_v1 --method lsa --sample_rate 48000 --n_samples 48000 --width 256 --depth 4 --num_bands 16 --sigma 100 --K 16 --lr 0.002 --act_lr_mult 3.0 --lr_decay 0.2 --decay_every 5000 --max_steps 5000 --batch_size 0 --log_every 250 --seed 0 --device "$DEVICE" --force
core nsynth_finer --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_ff16_finer_48k_sigma20_v1 --method finer --sample_rate 48000 --n_samples 48000 --width 256 --depth 4 --num_bands 16 --sigma 20 --fbs 20 --lr 0.0005 --lr_decay 0.2 --decay_every 5000 --max_steps 5000 --batch_size 0 --log_every 250 --seed 0 --device "$DEVICE" --force
core librispeech_finer --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_ff16_finer_48k_sigma20_v1 --method finer --sample_rate 48000 --n_samples 48000 --width 256 --depth 4 --num_bands 16 --sigma 20 --fbs 20 --lr 0.0005 --lr_decay 0.2 --decay_every 5000 --max_steps 5000 --batch_size 0 --log_every 250 --seed 0 --device "$DEVICE" --force
core nsynth_siren --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_siren_noff_48k_audit_v1 --method siren --sample_rate 48000 --n_samples 48000 --width 256 --depth 4 --num_bands 0 --sigma 1 --w0 30 --lr 0.0005 --lr_decay 0.2 --decay_every 5000 --max_steps 5000 --batch_size 0 --log_every 250 --seed 0 --device "$DEVICE" --force
core librispeech_siren --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_siren_noff_48k_audit_v1 --method siren --sample_rate 48000 --n_samples 48000 --width 256 --depth 4 --num_bands 0 --sigma 1 --w0 30 --lr 0.0005 --lr_decay 0.2 --decay_every 5000 --max_steps 5000 --batch_size 0 --log_every 250 --seed 0 --device "$DEVICE" --force
fjnb nsynth_fjnb_raw --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_fkan_noff_q6_lr1em3_bs4096_48k_v1 --sample_rate 48000 --n_samples 48000 --width 256 --depth 4 --degree 6 --num_bands 0 --sigma 1 --lr 0.001 --batch_size 4096 --max_steps 5000 --log_every 100 --device "$DEVICE" --seed 0
fjnb librispeech_fjnb_raw --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_fkan_noff_q2_lr1em3_bs4096_48k_v1 --sample_rate 48000 --n_samples 48000 --width 256 --depth 4 --degree 2 --num_bands 0 --sigma 1 --lr 0.001 --batch_size 4096 --max_steps 5000 --log_every 100 --device "$DEVICE" --seed 0
fjnb nsynth_fjnb_ff16 --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_fkan_ff16_s100_q6_lr1em3_bs4096_48k_v1 --sample_rate 48000 --n_samples 48000 --width 256 --depth 4 --degree 6 --num_bands 16 --sigma 100 --ff_seed 0 --lr 0.001 --batch_size 4096 --max_steps 5000 --log_every 100 --device "$DEVICE" --seed 0
fjnb librispeech_fjnb_ff16 --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_fkan_ff16_s100_q6_lr1em3_bs4096_48k_v1 --sample_rate 48000 --n_samples 48000 --width 256 --depth 4 --degree 6 --num_bands 16 --sigma 100 --ff_seed 0 --lr 0.001 --batch_size 4096 --max_steps 5000 --log_every 100 --device "$DEVICE" --seed 0
sl2a nsynth_sl2a_512_128 --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_sl2a_deg512_rank128_lr1em3_sched_48k_v1 --sample_rate 48000 --n_samples 48000 --hidden_features 256 --hidden_layers 3 --deg 512 --rank 128 --lr 0.001 --scheduler --scheduler_b 0.1 --batch_size 0 --max_steps 5000 --log_every 100 --seed 0 --device "$DEVICE"
sl2a nsynth_sl2a_512_64 --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_sl2a_deg512_rank64_lr1em3_sched_48k_v1 --sample_rate 48000 --n_samples 48000 --hidden_features 256 --hidden_layers 3 --deg 512 --rank 64 --lr 0.001 --scheduler --scheduler_b 0.1 --batch_size 0 --max_steps 5000 --log_every 100 --seed 0 --device "$DEVICE"
sl2a nsynth_sl2a_256_128 --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_sl2a_deg256_rank128_lr1em3_sched_48k_v1 --sample_rate 48000 --n_samples 48000 --hidden_features 256 --hidden_layers 3 --deg 256 --rank 128 --lr 0.001 --scheduler --scheduler_b 0.1 --batch_size 0 --max_steps 5000 --log_every 100 --seed 0 --device "$DEVICE"
sl2a nsynth_sl2a_256_64 --dataset nsynth --manifest manifests/nsynth_pilot10_manifest.csv --group nsynth_sl2a_deg256_rank64_lr1em3_sched_48k_v1 --sample_rate 48000 --n_samples 48000 --hidden_features 256 --hidden_layers 3 --deg 256 --rank 64 --lr 0.001 --scheduler --scheduler_b 0.1 --batch_size 0 --max_steps 5000 --log_every 100 --seed 0 --device "$DEVICE"
sl2a librispeech_sl2a_512_128 --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_sl2a_deg512_rank128_lr1em3_sched_48k_v1 --sample_rate 48000 --n_samples 48000 --hidden_features 256 --hidden_layers 3 --deg 512 --rank 128 --lr 0.001 --scheduler --scheduler_b 0.1 --batch_size 0 --max_steps 5000 --log_every 100 --seed 0 --device "$DEVICE"
sl2a librispeech_sl2a_512_64 --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_sl2a_deg512_rank64_lr1em3_sched_48k_v1 --sample_rate 48000 --n_samples 48000 --hidden_features 256 --hidden_layers 3 --deg 512 --rank 64 --lr 0.001 --scheduler --scheduler_b 0.1 --batch_size 0 --max_steps 5000 --log_every 100 --seed 0 --device "$DEVICE"
sl2a librispeech_sl2a_256_128 --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_sl2a_deg256_rank128_lr1em3_sched_48k_v1 --sample_rate 48000 --n_samples 48000 --hidden_features 256 --hidden_layers 3 --deg 256 --rank 128 --lr 0.001 --scheduler --scheduler_b 0.1 --batch_size 0 --max_steps 5000 --log_every 100 --seed 0 --device "$DEVICE"
sl2a librispeech_sl2a_256_64 --dataset librispeech --manifest manifests/librispeech_pilot40_manifest.csv --group librispeech_sl2a_deg256_rank64_lr1em3_sched_48k_v1 --sample_rate 48000 --n_samples 48000 --hidden_features 256 --hidden_layers 3 --deg 256 --rank 64 --lr 0.001 --scheduler --scheduler_b 0.1 --batch_size 0 --max_steps 5000 --log_every 100 --seed 0 --device "$DEVICE"
