param(
    [string]$Device = "cuda",
    [string]$Source = "volunteer_0",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$Python = "python"
$SplitInfo = "split_info_wifi"

& $Python "model code/preprocess.py" `
    --signal_type wifi `
    --src "WiFi data" `
    --dst "WiFi data_pad" `
    --volunteers $Source

& $Python "model code/extract_offset.py" `
    --signal_type wifi `
    --data_dir "WiFi data_pad/base" `
    --output_dir offsets_WiFi `
    --volunteers $Source `
    --sg_window 51 --sg_order 3 --sg_axis 1 `
    --seed $Seed

& $Python "model code/extract_offset.py" `
    --signal_type wifi `
    --data_dir "WiFi data_pad/unseen" `
    --output_dir offsets_unseen_wifi `
    --volunteers $Source `
    --fewshot_only --num_shots 3 `
    --split_info_dir $SplitInfo `
    --sg_window 51 --sg_order 3 --sg_axis 1 `
    --seed $Seed --force_split

& $Python "model code/siamese/train_siamese.py" `
    --signal_type wifi `
    --offset_dir offsets_WiFi `
    --output_dir checkpoints_siamese_wifi `
    --source $Source `
    --epochs 100 --batch_size 128 --lr 1e-3 `
    --device $Device --seed $Seed

& $Python "model code/siamese/filter_offsets.py" `
    --signal_type wifi `
    --siamese_checkpoint checkpoints_siamese_wifi/siamese_best.pt `
    --base_offset_dir offsets_WiFi `
    --unseen_offset_dir offsets_unseen_wifi `
    --output_dir offsets_filtered_wifi `
    --split_info_dir $SplitInfo `
    --source $Source `
    --threshold 0.65 --upper_threshold 0.95 `
    --per_class --index_only --save_report `
    --batch_size 256 --device $Device

& $Python "model code/diffusion/train.py" `
    --signal_type wifi `
    --data_dir "WiFi data_pad/base" `
    --offset_dir offsets_WiFi `
    --volunteers $Source `
    --use_offset_gate --use_cross_attn `
    --offset_mode both --offset_gate_channels 64 `
    --save_best --max_samples_per_epoch 400 `
    --batch_size 40 `
    --output_dir checkpoints_wifi `
    --device $Device --seed $Seed

& $Python "model code/diffusion/sample.py" `
    --signal_type wifi `
    --checkpoint checkpoints_wifi/model_final.pt `
    --data_dir "WiFi data_pad/unseen" `
    --base_offset_dir offsets_filtered_wifi `
    --split_info_dir $SplitInfo `
    --source $Source `
    --per_class_offsets `
    --num_shots 3 --samples_per_shot 20 `
    --sample_batch_size 20 `
    --sdedit_t 500 --cfg_scale 3.0 --offset_scale 1.0 `
    --progress `
    --output_dir generated_WiFi `
    --device $Device --seed $Seed

& $Python "cnn_classifier/cnn_eval_unseen.py" `
    --signal_type wifi `
    --real_dir "WiFi data_pad/unseen" `
    --gen_dir generated_WiFi `
    --split_info_dir $SplitInfo `
    --source $Source `
    --num_shots 3 --gen_samples 60 `
    --batch_size 64 `
    --early_stop_patience 50 `
    --device $Device `
    --trials 10
