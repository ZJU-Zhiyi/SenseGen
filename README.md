# SenseGen

SenseGen is a virtual data generation tool for wireless HCI applications, with examples for WiFi CSI spectrograms, mmWave range-Doppler heatmaps, and ultrasound spectrograms. The project combines behavioral offset extraction, Siamese offset filtering, an offset-conditioned diffusion model, and few-shot CNN evaluation.

The system targets few-shot sensing scenarios where only a small number of samples are available for customized unseen classes.

The deployment of SenseGen consists of two stages: generative model construction and virtual sample generation.

During generative model construction, users first define a set of common task categories, such as gestures or activities, referred to as base classes, and collect sufficient real samples for each class. Notably, data from the base classes only needs to be collected once and can be reused to generate samples for different unseen tasks. The collected samples undergo necessary preprocessing, such as denoising and STFT. Within each base class, real samples are then paired to extract two complementary behavioral offsets, namely the static offset and dynamic offset. The paired samples and their corresponding offsets are subsequently used to train a generative model.

During virtual sample generation, users can define customized task categories based on application requirements, referred to as unseen classes. Only a few real samples, for example three samples, need to be collected for each unseen class, and these samples undergo the same preprocessing as in the generative model construction stage. A Siamese network is then employed to measure the similarity between the behavioral offsets of unseen-class samples and those extracted from the base classes. Based on these similarities, SenseGen selects base-class offsets that provide both transferability and diversity. Guided by the selected offsets, SenseGen generates a large number of high-fidelity and diverse virtual samples for the unseen classes. These virtual samples can then be combined with the limited real samples and fed into downstream sensing models, such as gesture classifiers, to improve sensing performance while substantially reducing data collection costs.

This repository provides a single default diffusion implementation shared by WiFi, mmWave, and ultrasound.

## System Overview

SenseGen treats each sensing sample as a two-dimensional representation. Depending on the modality, this representation may be a CSI spectrogram, a range-Doppler heatmap, or an ultrasound spectrogram. All modules consume the same directory convention and `.npy` sample format, which keeps the implementation consistent across modalities.

At the implementation level, the pipeline passes data through a small set of reusable artifacts: preprocessed samples, base-class offset pools, few-shot unseen split files, Siamese-filtered offset indexes, diffusion checkpoints, generated samples, and CNN evaluation outputs. Each artifact is stored on disk with explicit paths, so stages can be inspected or rerun independently.

## Module Overview

### Preprocessing

`model code/preprocess.py` converts raw modality-specific files into normalized `.npy` arrays with consistent shapes. The preprocessing stage is intentionally thin: it prepares data layout, shape, and numeric format so the later offset and diffusion code can operate through a shared interface.

For WiFi and ultrasound, samples are represented as spectrogram-like arrays. For mmWave, samples are represented as range-Doppler heatmaps. The script exposes a unified `--signal_type` argument while preserving modality-specific shape handling.

### Offset Extraction

`model code/extract_offset.py` builds static and dynamic offset pools from within-class sample pairs. These offsets are the main bridge between observed base-class motion patterns and unseen-class generation. The same script also creates and reuses few-shot split files, so the support samples used for unseen-query offsets can stay aligned with later generation and evaluation.

The offset directory is organized by offset type, source, and class. This makes it possible to filter offsets per unseen class without copying large arrays when JSON indexes are used.

### Siamese Offset Filtering

The Siamese module contains:

| File | Role |
|------|------|
| `model code/siamese/siamese_network.py` | Shared embedding network for offset similarity |
| `model code/siamese/train_siamese.py` | Training script for offset-pair similarity learning |
| `model code/siamese/filter_offsets.py` | Selection script for class-specific reusable offsets |

The filtering stage compares base-class offsets with few-shot unseen-query offsets. With `--index_only`, it writes compact JSON indexes that point to selected base offsets instead of duplicating `.npy` files. This keeps full experiments manageable when offset pools are large.

### Diffusion Generation

`model code/diffusion/` contains the default generator used by the project. The model is an offset-conditioned diffusion pipeline with residual offset gating and optional bottleneck cross-attention. The training script learns from base-class samples and base offsets; the sampling script transfers filtered offsets to unseen-class support samples.

Key files:

| File | Role |
|------|------|
| `config.py` | Shared defaults and modality-specific settings |
| `dataset.py` | Dataset loading for samples and offsets |
| `diffusion.py` | Diffusion process and sampling utilities |
| `unet.py` | Conditional U-Net backbone |
| `train.py` | Training entry point |
| `sample.py` | Unseen-class generation entry point |

The sampling code uses DDPM sampling. SDEdit-style generation is available through `--sdedit_t`.

### CNN Evaluation

`cnn_classifier/` provides the downstream few-shot evaluation protocol. `cnn_eval_unseen.py` trains and evaluates a small CNN on real few-shot unseen samples, then optionally augments the support set with generated samples. This stage is not part of the generator itself; it is included to provide a consistent way to check whether generated samples are useful for classifier training.

For multi-trial evaluation, omitting `--seed` lets each trial use an automatically incremented seed. Passing `--seed` fixes the seed across all trials.

## Repository Layout

```text
SenseGen/
|-- README.md
|-- requirements.txt
|-- demo/
|   |-- WiFi/
|   |-- mmWave/
|   +-- ultrasound/
|-- scripts/
|   |-- run_pipeline_wifi.sh
|   |-- run_pipeline_mmwave.sh
|   |-- run_pipeline_ultrasound.sh
|   |-- run_pipeline_wifi.ps1
|   |-- run_pipeline_mmwave.ps1
|   +-- run_pipeline_ultrasound.ps1
|-- model code/
|   |-- preprocess.py
|   |-- extract_offset.py
|   |-- see_npy.py
|   |-- diffusion/
|   |   |-- config.py
|   |   |-- dataset.py
|   |   |-- diffusion.py
|   |   |-- sample.py
|   |   |-- train.py
|   |   +-- unet.py
|   +-- siamese/
|       |-- filter_offsets.py
|       |-- siamese_network.py
|       +-- train_siamese.py
+-- cnn_classifier/
    |-- cnn_models.py
    |-- cnn_classify.py
    +-- cnn_eval_unseen.py
```

Note that `model code` contains a space, so quote paths that point into it.

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use CUDA, install the PyTorch build that matches your CUDA driver and hardware. The default `pip install -r requirements.txt` may install a CPU build depending on your environment.

## Packaged Demo

The `demo/` directory contains one compact generation example per modality: a trained diffusion checkpoint, a small base offset pool, and a small unseen sample set. It is intended to verify the generation path and file conventions, not to report classification accuracy or reproduce full benchmark numbers.

The packaged checkpoints are lightweight demo artifacts. The commands below use `--sdedit_t 100` for faster local verification.

Current demo contents:

| Modality | Checkpoint | Offset pairs |
|----------|------------|--------------|
| WiFi | `demo/WiFi/checkpoints_wifi/model_final.pt` | 50 |
| mmWave | `demo/mmWave/checkpoints_mmwave/model_final.pt` | 50 |
| Ultrasound | `demo/ultrasound/checkpoints_ultrasound/model_final.pt` | 50 |

Example generation commands:

```bash
python "model code/diffusion/sample.py" \
    --checkpoint "demo/WiFi/checkpoints_wifi/model_final.pt" \
    --signal_type wifi \
    --data_dir "demo/WiFi/WiFi data_pad/unseen" \
    --base_offset_dir "demo/WiFi/WiFi_offsets" \
    --source volunteer_0 --class_name class_18 \
    --num_shots 1 --samples_per_shot 2 --sample_batch_size 2 \
    --sdedit_t 100 \
    --output_dir "generated_demo/WiFi" \
    --device cuda --seed 42

python "model code/diffusion/sample.py" \
    --checkpoint "demo/mmWave/checkpoints_mmwave/model_final.pt" \
    --signal_type mmwave \
    --data_dir "demo/mmWave/mmWave_data_proc/unseen" \
    --base_offset_dir "demo/mmWave/mmWave_offsets" \
    --source volunteer_1 --class_name class_12 \
    --num_shots 1 --samples_per_shot 2 --sample_batch_size 2 \
    --sdedit_t 100 \
    --output_dir "generated_demo/mmWave" \
    --device cuda --seed 42

python "model code/diffusion/sample.py" \
    --checkpoint "demo/ultrasound/checkpoints_ultrasound/model_final.pt" \
    --signal_type ultrasound \
    --data_dir "demo/ultrasound/ultrasound data_proc/unseen" \
    --base_offset_dir "demo/ultrasound/ultrasound_offsets" \
    --source volunteer_5 --class_name F \
    --num_shots 1 --samples_per_shot 2 --sample_batch_size 2 \
    --sdedit_t 100 \
    --output_dir "generated_demo/ultrasound" \
    --device cuda --seed 42
```

## Expected Shapes

| Modality | Preprocessed shape |
|----------|--------------------|
| WiFi | `(64, 256)` |
| mmWave | `(16, 128)` |
| Ultrasound | `(256, 256)` |

## Supported Modalities

### WiFi CSI

WiFi samples are handled as CSI-derived spectrograms. The default shape is compact enough for efficient diffusion training while preserving the time-frequency structure used by the offset extractor. WiFi examples use `volunteer_0` and class names such as `class_00`.

### mmWave

mmWave samples are handled as range-Doppler heatmaps. Their smaller spatial shape reflects the representation used by the preprocessed radar data. The mmWave workflow follows the same offset, diffusion, and CNN evaluation stages as WiFi, with modality-specific defaults for source names, batch sizes, and paths.

### Ultrasound

Ultrasound samples are handled as higher-resolution spectrograms. Because the default shape is larger, ultrasound diffusion training usually needs a smaller batch size than WiFi or mmWave. Ultrasound examples use letter-style class names such as `A` and `F`.

## Data Layout

The pipeline expects samples stored as `.npy` files under `base` and `unseen` splits. Full-scale generated files and checkpoints are ignored by Git; the compact artifacts under `demo/` are the only packaged exception.

### WiFi

```text
WiFi data/
|-- base/
|   +-- volunteer_0/
|       +-- class_00/
|           +-- sample_00.npy
+-- unseen/
    +-- volunteer_0/
        +-- class_15/
            +-- sample_00.npy
```

Legacy `channel_X` directories are still readable for compatibility, but the recommended convention is `volunteer_X`.

### mmWave

```text
mmWave_data/
|-- base/
|   +-- volunteer_1/
|       +-- class_00/
|           +-- sample_00.npy
+-- unseen/
    +-- volunteer_1/
        +-- class_15/
            +-- sample_00.npy
```

### Ultrasound

```text
ultrasound data_split/
|-- base/
|   +-- volunteer_5/
|       +-- A/
|           +-- sample_00.npy
+-- unseen/
    +-- volunteer_5/
        +-- F/
            +-- sample_00.npy
```

## Full Pipeline Runners

Bash runners in `scripts/` encode the default end-to-end workflows. Windows PowerShell equivalents are also provided in the same directory. They are intended as reproducible starting points and can be edited for your dataset paths, class lists, or GPU memory budget.

```bash
bash ./scripts/run_pipeline_wifi.sh
bash ./scripts/run_pipeline_mmwave.sh
bash ./scripts/run_pipeline_ultrasound.sh
```

Each runner accepts:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--device` | `cuda` | PyTorch device, usually `cuda` or `cpu` |
| `--source` | modality-specific | Volunteer/source directory |
| `--seed` | `42` | Seed for deterministic split creation and model runs |

Example:

```bash
bash ./scripts/run_pipeline_wifi.sh --device cuda --source volunteer_0 --seed 42
```

Windows PowerShell example:

```powershell
.\scripts\run_pipeline_wifi.ps1 -Device cuda -Source volunteer_0 -Seed 42
```

The runners use compact split directories such as `split_info_wifi`, so generated samples and evaluation reuse exactly the same support/eval split.

## Manual Workflow

The same workflow can be run manually stage by stage. The examples below use WiFi paths; replace `--signal_type`, paths, and source names for mmWave or ultrasound.

### 1. Preprocess

```bash
python "model code/preprocess.py" \
    --signal_type wifi \
    --src "WiFi data" \
    --dst "WiFi data_pad" \
    --volunteers volunteer_0
```

### 2. Extract Base Offsets

```bash
python "model code/extract_offset.py" \
    --signal_type wifi \
    --data_dir "WiFi data_pad/base" \
    --output_dir offsets_WiFi \
    --volunteers volunteer_0 \
    --sg_window 51 --sg_order 3 --sg_axis 1 \
    --seed 42
```

### 3. Extract Few-Shot Unseen Query Offsets

```bash
python "model code/extract_offset.py" \
    --signal_type wifi \
    --data_dir "WiFi data_pad/unseen" \
    --output_dir offsets_unseen_wifi \
    --volunteers volunteer_0 \
    --fewshot_only --num_shots 3 \
    --split_info_dir split_info_wifi \
    --sg_window 51 --sg_order 3 --sg_axis 1 \
    --seed 42 --force_split
```

### 4. Train Siamese Offset Similarity Model

```bash
python "model code/siamese/train_siamese.py" \
    --signal_type wifi \
    --offset_dir offsets_WiFi \
    --output_dir checkpoints_siamese_wifi \
    --source volunteer_0 \
    --epochs 100 --batch_size 128 --lr 1e-3 \
    --device cuda --seed 42
```

### 5. Filter Base Offsets

```bash
python "model code/siamese/filter_offsets.py" \
    --signal_type wifi \
    --siamese_checkpoint checkpoints_siamese_wifi/siamese_best.pt \
    --base_offset_dir offsets_WiFi \
    --unseen_offset_dir offsets_unseen_wifi \
    --output_dir offsets_filtered_wifi \
    --split_info_dir split_info_wifi \
    --source volunteer_0 \
    --threshold 0.8 --upper_threshold 0.99 \
    --per_class --index_only --save_report \
    --batch_size 256 --device cuda
```

`--index_only` writes lightweight JSON indexes instead of copying all selected offset files. The diffusion sampler can read these indexes directly.

### 6. Train Diffusion Model

```bash
python "model code/diffusion/train.py" \
    --signal_type wifi \
    --data_dir "WiFi data_pad/base" \
    --offset_dir offsets_WiFi \
    --volunteers volunteer_0 \
    --use_offset_gate --use_cross_attn \
    --offset_mode both --offset_gate_channels 64 \
    --save_best --max_samples_per_epoch 400 \
    --batch_size 40 \
    --output_dir checkpoints_wifi \
    --device cuda --seed 42
```

The default diffusion implementation uses residual offset gating and bottleneck cross-attention.

### 7. Generate Unseen Samples

```bash
python "model code/diffusion/sample.py" \
    --signal_type wifi \
    --checkpoint checkpoints_wifi/model_final.pt \
    --data_dir "WiFi data_pad/unseen" \
    --base_offset_dir offsets_filtered_wifi \
    --split_info_dir split_info_wifi \
    --source volunteer_0 \
    --per_class_offsets \
    --num_shots 3 --samples_per_shot 20 \
    --sample_batch_size 20 \
    --sdedit_t 500 --cfg_scale 3.0 --offset_scale 1.0 \
    --progress \
    --output_dir generated_WiFi \
    --device cuda --seed 42
```

Full-scale generation can use a larger SDEdit timestep, such as `--sdedit_t 500`, when more variation from the anchor sample is desired.

### 8. Evaluate Unseen Classes

```bash
python "cnn_classifier/cnn_eval_unseen.py" \
    --signal_type wifi \
    --real_dir "WiFi data_pad/unseen" \
    --gen_dir generated_WiFi \
    --split_info_dir split_info_wifi \
    --source volunteer_0 \
    --num_shots 3 --gen_samples 60 \
    --batch_size 64 \
    --early_stop_patience 50 \
    --device cuda \
    --trials 10
```

If `--seed` is omitted in multi-trial CNN unseen evaluation, trials use auto-incremented seeds starting from 42. If `--seed N` is provided, every trial uses the same seed `N`.

## Modality Defaults

| Setting | WiFi | mmWave | Ultrasound |
|---------|------|--------|------------|
| Default source | `volunteer_0` | `volunteer_1` | `volunteer_5` |
| Base data | `WiFi data_pad/base` | `mmWave_data_proc/base` | `ultrasound data_proc/base` |
| Unseen data | `WiFi data_pad/unseen` | `mmWave_data_proc/unseen` | `ultrasound data_proc/unseen` |
| Base offsets | `offsets_WiFi` | `offsets_mmWave` | `offsets_ultrasound` |
| Diffusion checkpoints | `checkpoints_wifi` | `checkpoints_mmWave` | `checkpoints_ultrasound` |
| Generated output | `generated_WiFi` | `generated_mmWave` | `generated_ultrasound` |
| CNN batch size | `64` | `128` | `32` |

Suggested starting batch sizes for diffusion training:

| Modality | Batch size |
|----------|------------|
| WiFi | `40` |
| mmWave | `192` |
| Ultrasound | `8` |

Adjust these values for your GPU memory.

## Main Scripts

| Script | Purpose |
|--------|---------|
| `model code/preprocess.py` | Unified preprocessing for WiFi, mmWave, and ultrasound |
| `model code/extract_offset.py` | Static/dynamic offset extraction and reusable few-shot split creation |
| `model code/siamese/train_siamese.py` | Siamese offset similarity training |
| `model code/siamese/filter_offsets.py` | Offset filtering and JSON index creation |
| `model code/diffusion/train.py` | Default gated diffusion training |
| `model code/diffusion/sample.py` | Offset-conditioned generation |
| `cnn_classifier/cnn_models.py` | Shared CNN model definitions |
| `cnn_classifier/cnn_classify.py` | Optional base-class CNN training |
| `cnn_classifier/cnn_eval_unseen.py` | Few-shot unseen-class evaluation |

## Outputs

Typical generated directories:

```text
offsets_WiFi/
offsets_unseen_wifi/
offsets_filtered_wifi/
checkpoints_siamese_wifi/
checkpoints_wifi/
generated_WiFi/
split_info_wifi/
```

The matching mmWave and ultrasound outputs use the same naming pattern with modality-specific names. These outputs are ignored by `.gitignore`.
