param(
    [string]$Device = "cuda",
    [string]$Source = "volunteer_1",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$Python = "python"
$SplitInfo = "split_info_mmwave"

& $Python "model code/preprocess.py" `
    --signal_type mmwave `
    --src "mmWave_data" `
    --dst "mmWave_data_proc" `
    --splits base unseen `
    --volunteers $Source `
    --shared_stats

& $Python "model code/extract_offset.py" `
    --signal_type mmwave `
    --data_dir "mmWave_data_proc/base" `
    --output_dir offsets_mmWave `
    --volunteers $Source `
    --sg_window 5 --sg_order 3 --sg_axis 0 `
    --seed $Seed

& $Python "model code/extract_offset.py" `
    --signal_type mmwave `
    --data_dir "mmWave_data_proc/unseen" `
    --output_dir offsets_unseen_mmwave `
    --volunteers $Source `
    --fewshot_only --num_shots 3 `
    --split_info_dir $SplitInfo `
    --sg_window 5 --sg_order 3 --sg_axis 0 `
    --seed $Seed --force_split

& $Python "model code/siamese/train_siamese.py" `
    --signal_type mmwave `
    --offset_dir offsets_mmWave `
    --output_dir checkpoints_siamese_mmwave `
    --source $Source `
    --epochs 100 --batch_size 256 --lr 1e-3 `
    --device $Device --seed $Seed

& $Python "model code/siamese/filter_offsets.py" `
    --signal_type mmwave `
    --siamese_checkpoint checkpoints_siamese_mmwave/siamese_best.pt `
    --base_offset_dir offsets_mmWave `
    --unseen_offset_dir offsets_unseen_mmwave `
    --output_dir offsets_filtered_mmwave `
    --split_info_dir $SplitInfo `
    --source $Source `
    --threshold 0.65 --upper_threshold 0.95 `
    --per_class --index_only --save_report `
    --batch_size 512 --device $Device

& $Python "model code/diffusion/train.py" `
    --signal_type mmwave `
    --data_dir "mmWave_data_proc/base" `
    --offset_dir offsets_mmWave `
    --volunteers $Source `
    --use_offset_gate --use_cross_attn `
    --offset_mode both --offset_gate_channels 64 `
    --save_best --max_samples_per_epoch 150 `
    --batch_size 192 `
    --output_dir checkpoints_mmWave `
    --device $Device --seed $Seed

& $Python "model code/diffusion/sample.py" `
    --signal_type mmwave `
    --checkpoint checkpoints_mmWave/model_final.pt `
    --data_dir "mmWave_data_proc/unseen" `
    --base_offset_dir offsets_filtered_mmwave `
    --split_info_dir $SplitInfo `
    --source $Source `
    --per_class_offsets `
    --num_shots 3 --samples_per_shot 20 `
    --sample_batch_size 20 `
    --sdedit_t 500 --cfg_scale 3.0 --offset_scale 1.0 `
    --progress `
    --output_dir generated_mmWave `
    --device $Device --seed $Seed

& $Python "cnn_classifier/cnn_eval_unseen.py" `
    --signal_type mmwave `
    --real_dir "mmWave_data_proc/unseen" `
    --gen_dir generated_mmWave `
    --split_info_dir $SplitInfo `
    --source $Source `
    --num_shots 3 --gen_samples 60 `
    --batch_size 128 `
    --early_stop_patience 50 `
    --device $Device `
    --trials 10
