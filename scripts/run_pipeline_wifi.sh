#!/usr/bin/env bash
set -euo pipefail

DEVICE="cuda"
SOURCE="volunteer_0"
SEED="42"
PYTHON_BIN="${PYTHON:-python}"

usage() {
    echo "Usage: $0 [--device cuda|cpu] [--source volunteer_0] [--seed 42]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SPLIT_INFO="split_info_wifi"

"$PYTHON_BIN" "model code/preprocess.py" \
    --signal_type wifi \
    --src "WiFi data" \
    --dst "WiFi data_pad" \
    --volunteers "$SOURCE"

"$PYTHON_BIN" "model code/extract_offset.py" \
    --signal_type wifi \
    --data_dir "WiFi data_pad/base" \
    --output_dir offsets_WiFi \
    --volunteers "$SOURCE" \
    --sg_window 51 --sg_order 3 --sg_axis 1 \
    --seed "$SEED"

"$PYTHON_BIN" "model code/extract_offset.py" \
    --signal_type wifi \
    --data_dir "WiFi data_pad/unseen" \
    --output_dir offsets_unseen_wifi \
    --volunteers "$SOURCE" \
    --fewshot_only --num_shots 3 \
    --split_info_dir "$SPLIT_INFO" \
    --sg_window 51 --sg_order 3 --sg_axis 1 \
    --seed "$SEED" --force_split

"$PYTHON_BIN" "model code/siamese/train_siamese.py" \
    --signal_type wifi \
    --offset_dir offsets_WiFi \
    --output_dir checkpoints_siamese_wifi \
    --source "$SOURCE" \
    --epochs 100 --batch_size 128 --lr 1e-3 \
    --device "$DEVICE" --seed "$SEED"

"$PYTHON_BIN" "model code/siamese/filter_offsets.py" \
    --signal_type wifi \
    --siamese_checkpoint checkpoints_siamese_wifi/siamese_best.pt \
    --base_offset_dir offsets_WiFi \
    --unseen_offset_dir offsets_unseen_wifi \
    --output_dir offsets_filtered_wifi \
    --split_info_dir "$SPLIT_INFO" \
    --source "$SOURCE" \
    --threshold 0.65 --upper_threshold 0.95 \
    --per_class --index_only --save_report \
    --batch_size 256 --device "$DEVICE"

"$PYTHON_BIN" "model code/diffusion/train.py" \
    --signal_type wifi \
    --data_dir "WiFi data_pad/base" \
    --offset_dir offsets_WiFi \
    --volunteers "$SOURCE" \
    --use_offset_gate --use_cross_attn \
    --offset_mode both --offset_gate_channels 64 \
    --save_best --max_samples_per_epoch 400 \
    --batch_size 40 \
    --output_dir checkpoints_wifi \
    --device "$DEVICE" --seed "$SEED"

"$PYTHON_BIN" "model code/diffusion/sample.py" \
    --signal_type wifi \
    --checkpoint checkpoints_wifi/model_final.pt \
    --data_dir "WiFi data_pad/unseen" \
    --base_offset_dir offsets_filtered_wifi \
    --split_info_dir "$SPLIT_INFO" \
    --source "$SOURCE" \
    --per_class_offsets \
    --num_shots 3 --samples_per_shot 20 \
    --sample_batch_size 20 \
    --sdedit_t 500 --cfg_scale 3.0 --offset_scale 1.0 \
    --progress \
    --output_dir generated_WiFi \
    --device "$DEVICE" --seed "$SEED"

"$PYTHON_BIN" "cnn_classifier/cnn_eval_unseen.py" \
    --signal_type wifi \
    --real_dir "WiFi data_pad/unseen" \
    --gen_dir generated_WiFi \
    --split_info_dir "$SPLIT_INFO" \
    --source "$SOURCE" \
    --num_shots 3 --gen_samples 60 \
    --batch_size 64 \
    --early_stop_patience 50 \
    --device "$DEVICE" \
    --trials 10
