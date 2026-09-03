'''
Training script for SenseGen Diffusion Model with G1 Offset-Gated ResBlock.
Supports checkpoint resume with automatic config inheritance.

Usage:
  # Train the default final diffusion model
  python train.py --signal_type wifi --use_offset_gate --volunteers volunteer_0

'''

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .config import SenseGenConfig, UNetConfig, DiffusionConfig, TrainingConfig
    from .unet import UNetModel
    from .diffusion import GaussianDiffusion, EMA
    from .dataset import SenseGenDataset
except ImportError:
    from config import SenseGenConfig, UNetConfig, DiffusionConfig, TrainingConfig
    from unet import UNetModel
    from diffusion import GaussianDiffusion, EMA
    from dataset import SenseGenDataset


MODALITY_BATCH_DEFAULTS = {
    "wifi": 40,
    "mmwave": 192,
    "ultrasound": 8,
}

MODALITY_PATH_DEFAULTS = {
    "wifi": {
        "data_dir": "WiFi data_pad/base",
        "offset_dir": "offsets_WiFi",
        "output_dir": "checkpoints_wifi",
    },
    "mmwave": {
        "data_dir": "mmWave_data_proc/base",
        "offset_dir": "offsets_mmWave",
        "output_dir": "checkpoints_mmWave",
    },
    "ultrasound": {
        "data_dir": "ultrasound data_proc/base",
        "offset_dir": "offsets_ultrasound",
        "output_dir": "checkpoints_ultrasound",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the default SenseGen diffusion model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train the default final diffusion model
  python train.py --signal_type wifi --use_offset_gate --volunteers volunteer_0

  # Resume from checkpoint
  python train.py --resume checkpoints_wifi/checkpoint_epoch_0100.pt
        ''',
    )

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Data
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: from checkpoint or modality default)')
    parser.add_argument('--offset_dir', type=str, default=None,
                        help='Offset directory (default: from checkpoint or modality default)')
    parser.add_argument('--signal_type', type=str, default=None,
                        choices=['wifi', 'mmwave', 'ultrasound'],
                        help='Signal type')
    parser.add_argument('--volunteers', type=str, nargs='+', default=None,
                        help='Volunteer/source directories (default: modality default)')
    parser.add_argument('--channels', type=str, nargs='+', default=None,
                        help='[deprecated] Legacy WiFi channel ids; mapped to volunteer_X')
    parser.add_argument('--volunteer', type=str, default=None,
                        help='Single volunteer alias; prefer --volunteers')
    parser.add_argument('--classes', type=str, nargs='+', default=None)

    # Model
    parser.add_argument('--model_channels', type=int, default=None)
    parser.add_argument('--channel_mult', type=int, nargs='+', default=None)
    parser.add_argument('--num_res_blocks', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)

    # G1: Offset-Gated ResBlock
    parser.add_argument('--use_offset_gate', action='store_true', default=None,
                        help='Compatibility flag for the default residual offset gate')
    parser.add_argument('--use_cross_attn', action='store_true', default=None,
                        help='Compatibility flag for the default bottleneck offset cross-attention')
    parser.add_argument('--offset_gate_channels', type=int, default=None,
                        help='Hidden channels in OffsetGate (default: 64)')

    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=None)
    parser.add_argument('--beta_schedule', type=str, default=None,
                        choices=['linear', 'cosine'])

    # Training
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--ema_decay', type=float, default=None)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--cfg_drop_prob', type=float, default=None)
    parser.add_argument('--offset_mode', type=str, default=None,
                        choices=['both', 'static', 'dynamic_condition', 'dynamic'],
                        help='Offset ablation mode. both=current behavior; '
                             'static=static only; dynamic_condition=dynamic in condition only; '
                             'dynamic=dynamic in condition and timestep embedding.')
    parser.add_argument('--offset_scale_min', type=float, default=None,
                        help='Lower bound for offset scaling (1.0=no scaling)')
    parser.add_argument('--offset_scale_max', type=float, default=None,
                        help='Upper bound for offset scaling (min=max gives fixed scale)')
    parser.add_argument('--max_samples_per_epoch', type=int, default=None,
                        help='Max samples randomly selected per epoch. '
                             'None = use all offsets.')

    # System
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=None)
    parser.add_argument('--save_interval', type=int, default=None)
    parser.add_argument('--save_best', action='store_true', default=None,
                        help='Save best model (lowest avg loss) as model_best.pt')

    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_wifi_volunteer(value) -> str:
    value = str(value)
    if value.startswith('volunteer_'):
        return value
    if value.startswith('channel_'):
        value = value[len('channel_'):]
    return f'volunteer_{value}'


def migrate_training_sources(tc):
    """Migrate old checkpoint path metadata without touching model weights."""
    if 'sources' in getattr(tc, '__dict__', {}) and tc.sources:
        sources = list(tc.sources)
    elif getattr(tc, 'signal_type', 'wifi') == 'wifi':
        sources = [normalize_wifi_volunteer(v) for v in (getattr(tc, 'channels', None) or ['0'])]
    else:
        sources = list(getattr(tc, 'channels', None) or [getattr(tc, 'volunteer', 'volunteer_1')])
    if getattr(tc, 'signal_type', 'wifi') == 'wifi':
        sources = [normalize_wifi_volunteer(v) for v in sources]
        tc.source_prefix = ''
    tc.sources = sources


def merge_config(base: SenseGenConfig, args) -> SenseGenConfig:
    '''Merge CLI args into config: None args keep the base value.'''
    bc = base.unet
    dc = base.diffusion
    tc = base.training
    migrate_training_sources(tc)
    if not hasattr(tc, 'offset_mode'):
        tc.offset_mode = 'both'
    if not hasattr(bc, 'use_dynamic_timestep'):
        bc.use_dynamic_timestep = tc.offset_mode in ('both', 'dynamic')

    # --- Model ---
    if args.model_channels is not None:
        bc.model_channels = args.model_channels
    if args.channel_mult is not None:
        bc.channel_mult = tuple(args.channel_mult)
    if args.num_res_blocks is not None:
        bc.num_res_blocks = args.num_res_blocks
    if args.dropout is not None:
        bc.dropout = args.dropout

    # --- G1: Offset-Gated ResBlock ---
    if args.use_offset_gate is not None:
        bc.use_offset_gate = args.use_offset_gate
    if args.use_cross_attn is not None:
        bc.use_cross_attn = args.use_cross_attn
    if args.offset_gate_channels is not None:
        bc.offset_gate_channels = args.offset_gate_channels

    # --- Diffusion ---
    if args.num_timesteps is not None:
        dc.num_timesteps = args.num_timesteps
    if args.beta_schedule is not None:
        dc.beta_schedule = args.beta_schedule

    if args.signal_type is not None:
        tc.signal_type = args.signal_type
        tc.source_prefix = ''
        if args.volunteers is None and args.volunteer is None and args.channels is None:
            tc.sources = {'wifi': ['volunteer_0'], 'mmwave': ['volunteer_1'], 'ultrasound': ['volunteer_5']}[args.signal_type]
    path_defaults = MODALITY_PATH_DEFAULTS.get(tc.signal_type, MODALITY_PATH_DEFAULTS['wifi'])
    if args.data_dir is not None:
        tc.data_dir = args.data_dir
    elif not args.resume:
        tc.data_dir = path_defaults["data_dir"]
    if args.offset_dir is not None:
        tc.offset_dir = args.offset_dir
    elif not args.resume:
        tc.offset_dir = path_defaults["offset_dir"]
    if args.volunteers is not None:
        tc.sources = args.volunteers
    elif args.channels is not None:
        tc.sources = [normalize_wifi_volunteer(v) for v in args.channels]
    if args.volunteer is not None:
        tc.volunteer = args.volunteer
        tc.sources = [args.volunteer]
    if tc.signal_type == 'wifi':
        tc.sources = [normalize_wifi_volunteer(v) for v in tc.sources]
        tc.source_prefix = ''
    tc.channels = None
    if args.classes is not None:
        tc.classes = args.classes

    # --- Training ---
    if args.batch_size is not None:
        tc.batch_size = args.batch_size
    if args.learning_rate is not None:
        tc.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        tc.weight_decay = args.weight_decay
    if args.num_epochs is not None:
        tc.num_epochs = args.num_epochs
    if args.ema_decay is not None:
        tc.ema_decay = args.ema_decay
    if args.grad_clip is not None:
        tc.grad_clip = args.grad_clip
    if args.cfg_drop_prob is not None:
        tc.cfg_drop_prob = args.cfg_drop_prob
    if args.offset_mode is not None:
        tc.offset_mode = args.offset_mode
        bc.use_dynamic_timestep = args.offset_mode in ('both', 'dynamic')
    if args.offset_scale_min is not None:
        tc.offset_scale_min = args.offset_scale_min
    if args.offset_scale_max is not None:
        tc.offset_scale_max = args.offset_scale_max
    if args.max_samples_per_epoch is not None:
        tc.max_samples_per_epoch = args.max_samples_per_epoch

    # --- System ---
    if args.output_dir is not None:
        tc.output_dir = args.output_dir
    elif not args.resume:
        tc.output_dir = path_defaults["output_dir"]
    if args.num_workers is not None:
        tc.num_workers = args.num_workers
    if args.seed is not None:
        tc.seed = args.seed
    if args.log_interval is not None:
        tc.log_interval = args.log_interval
    if args.save_interval is not None:
        tc.save_interval = args.save_interval
    if args.save_best is not None:
        tc.save_best = args.save_best

    return base


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    #  Determine config: from scratch or from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    resume_state = None

    if args.resume:
        print(f'[Resume] Loading checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        config = ckpt.get('config', SenseGenConfig())
        resume_state = ckpt
        start_epoch = ckpt.get('epoch', -1) + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f'[Resume] Epoch {start_epoch} / {config.training.num_epochs}')
        print(f'[Resume] Offset-gating: {config.unet.use_offset_gate}')
    else:
        config = SenseGenConfig()
        if args.batch_size is None:
            signal_type = args.signal_type or config.training.signal_type
            config.training.batch_size = MODALITY_BATCH_DEFAULTS.get(
                signal_type,
                config.training.batch_size,
            )

    # Merge CLI overrides
    config = merge_config(config, args)
    if 'best_loss' not in locals():
        best_loss = float('inf')

    bc = config.unet
    dc = config.diffusion
    tc = config.training

    # ------------------------------------------------------------------
    #  Setup
    # ------------------------------------------------------------------
    set_seed(tc.seed)
    os.makedirs(tc.output_dir, exist_ok=True)
    device = torch.device(tc.device if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')

    print(f'[Signal] {tc.signal_type}')
    print(f'[Data] data_dir={tc.data_dir}  offset_dir={tc.offset_dir}')
    print(f'[Sources] {tc.sources}')
    print(f'[Output] {tc.output_dir}')

    ca = getattr(bc, 'use_cross_attn', False)
    dyn_t = getattr(bc, 'use_dynamic_timestep', True)
    gate_state = 'ENABLED' if bc.use_offset_gate else 'DISABLED'
    gate_detail = f' (hidden_ch={bc.offset_gate_channels})' if bc.use_offset_gate else ''
    print(f'[Gated FiLM] {gate_state}{gate_detail}  '
          f'[Cross-Attn] {"ON" if ca else "OFF"}  '
          f'[Dynamic->Timestep] {"ON" if dyn_t else "OFF"}')
    print(f'[OffsetMode] {tc.offset_mode}')

    # ------------------------------------------------------------------
    #  Data
    # ------------------------------------------------------------------
    dataset = SenseGenDataset(
        data_dir=tc.data_dir,
        offset_dir=tc.offset_dir,
        sources=tc.sources,
        classes=tc.classes,
        signal_type=tc.signal_type,
        source_prefix=tc.source_prefix,
        offset_mode=tc.offset_mode,
        offset_scale_min=tc.offset_scale_min,
        offset_scale_max=tc.offset_scale_max,
        max_samples_per_epoch=tc.max_samples_per_epoch,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=tc.num_workers,
        pin_memory=True,
    )
    dataset_stats = {
        'mean': dataset.mean,
        'std': dataset.std,
    }

    # ------------------------------------------------------------------
    #  Model
    # ------------------------------------------------------------------
    _first_sample = next(iter(dataloader))
    _input_h = _first_sample[0].shape[2]
    unet = UNetModel(config.unet, input_size=_input_h).to(device)
    diffusion = GaussianDiffusion(unet, config.diffusion).to(device)
    ema = EMA(unet, tc.ema_decay).to(device)

    total_params = sum(p.numel() for p in unet.parameters())
    print(f'[Model] Total parameters: {total_params:,}')

    # ------------------------------------------------------------------
    #  Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=tc.learning_rate,
        weight_decay=tc.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc.num_epochs * len(dataloader)
    )

    # ------------------------------------------------------------------
    #  Load state if resuming
    # ------------------------------------------------------------------
    if resume_state is not None:
        unet.load_state_dict(resume_state['model'])
        ema.shadow.load_state_dict(resume_state['ema'])
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        print(f'[Resume] Model, EMA, optimizer, scheduler states restored.')
        print(f'[Resume] LR = {scheduler.get_last_lr()[0]:.2e}')

    # ------------------------------------------------------------------
    #  Training loop
    # ------------------------------------------------------------------
    print('\nStarting training...\n')
    steps_per_epoch = len(dataloader)
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, tc.num_epochs):
        # Resample active sample subset for this epoch
        if tc.max_samples_per_epoch is not None:
            dataset.resample_samples(seed=tc.seed + epoch)

        unet.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (sample, cond, class_name, sample_idx) in enumerate(dataloader):
            sample = sample.to(device)
            cond = cond.to(device)

            loss = diffusion.training_loss(sample, cond, cfg_drop_prob=tc.cfg_drop_prob)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f'  [WARN] NaN/Inf loss at epoch {epoch}, step {batch_idx}. '
                      f'Skipping batch.')
                continue

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            grad_nan = False
            for name, param in unet.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any()
                                               or torch.isinf(param.grad).any()):
                    grad_nan = True
                    break
            if grad_nan:
                print(f'  [WARN] NaN/Inf gradient at epoch {epoch}, step {batch_idx}. '
                      f'Skipping batch.')
                optimizer.zero_grad()
                continue

            if tc.grad_clip > 0:
                nn.utils.clip_grad_norm_(unet.parameters(), tc.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update()

            global_step += 1

            if global_step % tc.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(f'  [Epoch {epoch:4d} | Step {global_step:6d}] '
                      f'loss={loss.item():.6f}  lr={lr:.2e}')

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / steps_per_epoch
        print(f'[Epoch {epoch:4d}] avg_loss={avg_loss:.6f}  time={epoch_time:.1f}s')

        # -- Best model tracking --
        if tc.save_best:
            if best_loss == float("inf"):  # init only if not recovered from ckpt
                best_loss = avg_loss
            if avg_loss <= best_loss:
                best_loss = avg_loss
                best_path = os.path.join(tc.output_dir, 'model_best.pt')
                torch.save({
                    'model': ema.shadow.state_dict(),
                    'config': config,
                    'dataset_stats': dataset_stats,
                }, best_path)
                print(f'  [Best]  epoch={epoch}  loss={best_loss:.6f}  -> {best_path}')

        # -- Checkpoint --
        if (epoch + 1) % tc.save_interval == 0 or epoch == tc.num_epochs - 1:
            ckpt_path = os.path.join(tc.output_dir, f'checkpoint_epoch_{epoch:04d}.pt')
            torch.save({
                'epoch': epoch,
                    'best_loss': best_loss if tc.save_best else float('inf'),
                'model': unet.state_dict(),
                'ema': ema.shadow.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config,
                'dataset_stats': dataset_stats,
            }, ckpt_path)
            print(f'  [Save] {ckpt_path}')

    # -- Final model export --
    final_path = os.path.join(tc.output_dir, 'model_final.pt')
    torch.save({
        'model': ema.shadow.state_dict(),
        'config': config,
        'dataset_stats': dataset_stats,
    }, final_path)
    print(f'\n[Final model] {final_path}')
    print('Training complete.')


if __name__ == '__main__':
    main()
