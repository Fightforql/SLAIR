"""
Training Codes of LightningDiT together with VA-VAE.
It envolves advanced training methods, sampling methods, 
architecture design methods, computation methods. We achieve
state-of-the-art FID 1.35 on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import math
import yaml
import json
import numpy as np
import logging
import os
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm

#from diffusers.models import AutoencoderKL
#from vavae.ldm.models.autoencoder import AutoencoderKL
from models.lightningdit_addy import LightningDiT_models
from transport import create_transport, Sampler
from accelerate import Accelerator
from datasets.dataset import ImgPatchDataset
from accelerate.utils import set_seed
from tokenizer.vavae import VA_VAE

def do_train(train_config, accelerator):
    """
    Trains a LightningDiT.
    """
    # Setup accelerator:
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        model_string_name = train_config['model']['model_type'].replace("/", "-")
        if train_config['train']['exp_name'] is None:
            exp_name = f'{experiment_index:03d}-{model_string_name}'
        else:
            exp_name = train_config['train']['exp_name']
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        validation_sample_dir = f"{experiment_dir}/validation_samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(validation_sample_dir, exist_ok=True)
        logger = create_logger(experiment_dir, accelerator)
        logger.info(f"Experiment directory created at {experiment_dir}")
        tensorboard_dir_log = f"tensorboard_logs/{exp_name}"
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir_log)

        # add configs to tensorboard
        config_str=json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)
    if not accelerator.is_main_process:
        exp_name = train_config['train']['exp_name']
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        validation_sample_dir = f"{experiment_dir}/validation_samples"
    checkpoint_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/checkpoints"

    # get rank
    rank = accelerator.local_process_index

    # Create model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
    )

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # load pretrained model
    if 'weight_init' in train_config['train']:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location=lambda storage, loc: storage)
        # remove the prefix 'module.' from the keys
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model = load_weights_with_shape_check(model, checkpoint, rank=rank)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']}")
    requires_grad(ema, False)
    
    # model = DDP(model.to(device), device_ids=[rank])
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity; 
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
    opt = torch.optim.AdamW(model.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    
    # Setup data
    latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 1.0
    print(f"latent multiplier is {latent_multiplier}.")
    image_size = train_config['data'].get('image_size', 512)
    stats_path = train_config['data']['stats_path']
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Latent stats not found at {stats_path}")
    dataset = ImgPatchDataset(
        csv_path=train_config['data']['csv_path'],
        patch_size=train_config['data']['image_size'],
        is_train=True
    )
    latent_stats = torch.load(stats_path, map_location='cpu')
    latent_mean = latent_stats['mean'].to(device)
    latent_std = latent_stats['std'].to(device)
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    #print(train_config['train']['global_batch_size'],accelerator.num_processes,batch_size_per_gpu)
    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    all_labels = pd.read_csv(train_config['data']['csv_path']).iloc[:, 2].values
    label_counts = Counter(all_labels)
    if accelerator.is_main_process:
        logger.info(f"Label distribution: { {k: v for k, v in sorted(label_counts.items())} }")
    sample_weights = np.array([1.0 / label_counts[l] for l in all_labels], dtype=np.float32)
    sample_weights = torch.from_numpy(sample_weights)
    num_samples = len(dataset)
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset)} images")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
        logger.info(f"WeightedRandomSampler: num_samples={num_samples}")
    
    valid_dataset = None
    if 'valid_data' in train_config and 'csv_path' in train_config['valid_data']:
        valid_dataset = ImgPatchDataset(
            csv_path=train_config['valid_data']['csv_path'],
            patch_size=train_config['valid_data']['image_size'],
            is_train=False
        )
        valid_stats_path = train_config['valid_data']['stats_path']
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Latent stats not found at {stats_path}")
        valid_latent_stats = torch.load(valid_stats_path, map_location='cpu')
        valid_latent_mean = valid_latent_stats['mean'].to(device)
        valid_latent_std = valid_latent_stats['std'].to(device)
        if accelerator.is_main_process:
            logger.info(f"Validation Dataset contains {len(valid_dataset)} images")
        sampler = Sampler(transport)
        if train_config['sample']['mode'] != "ODE":
            raise NotImplementedError(f"Sampling mode {train_config['sample']['mode']} is not supported.")
        sample_fn = sampler.sample_ode(
            sampling_method=train_config['sample']['sampling_method'],
            num_steps=train_config['sample']['num_sampling_steps'],
            atol=train_config['sample']['atol'],
            rtol=train_config['sample']['rtol'],
            reverse=train_config['sample']['reverse'],
            timestep_shift=train_config['sample']['timestep_shift'],
        )

    # Prepare models for training:
    model = model.to(device)
    ema.load_state_dict(model.state_dict()) # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # Prepare model, optimizer and loader with accelerator first
    model, opt, loader = accelerator.prepare(model, opt, loader)
    
    # Determine checkpoint path to load
    checkpoint_path = None
    train_steps = 0
    # Ensure `writer` exists on non-main ranks, but do not overwrite main-rank writer.
    if not accelerator.is_main_process:
        writer = None
    
    # Priority 1: Check if ckpt is specified in train.ckpt
    if 'ckpt' in train_config['train'] and train_config['train']['ckpt'] is not None:
        checkpoint_path = train_config['train']['ckpt']
        if accelerator.is_main_process:
            logger.info(f"Checkpoint path specified in config: {checkpoint_path}")
    
    # Priority 2: Check if ckpt_path is specified at root level (for backward compatibility)
    elif 'ckpt_path' in train_config and train_config['ckpt_path'] is not None:
        checkpoint_path = train_config['ckpt_path']
        if accelerator.is_main_process:
            logger.info(f"Checkpoint path specified in config (ckpt_path): {checkpoint_path}")
    
    # Priority 3: Check if resume is enabled, then look for latest checkpoint in checkpoint_dir
    elif train_config['train'].get('resume', False):
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            # Sort by step number (filename format: 0200000.pt)
            def get_step_number(path):
                try:
                    return int(os.path.basename(path).split('.')[0])
                except:
                    return 0
            checkpoint_files.sort(key=get_step_number)
            checkpoint_path = checkpoint_files[-1]
            if accelerator.is_main_process:
                logger.info(f"Auto-resume: Found latest checkpoint: {checkpoint_path}")
    
    # Load checkpoint if path is specified
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            if accelerator.is_main_process:
                logger.info(f"Loading checkpoint from: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            
            # Load model state dict
            if 'model' in checkpoint:
                # Handle DDP wrapped model
                unwrapped_model = accelerator.unwrap_model(model)
                # Remove 'module.' prefix if present
                model_state = checkpoint['model']
                if any(k.startswith('module.') for k in model_state.keys()):
                    model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
                unwrapped_model.load_state_dict(model_state, strict=False)
                if accelerator.is_main_process:
                    logger.info("Model state dict loaded successfully")
            
            # Load EMA state dict
            if 'ema' in checkpoint:
                ema.load_state_dict(checkpoint['ema'], strict=False)
                if accelerator.is_main_process:
                    logger.info("EMA state dict loaded successfully")
            
            # Load optimizer state dict (after accelerator.prepare)
            if 'opt' in checkpoint:
                try:
                    opt.load_state_dict(checkpoint['opt'])
                    if accelerator.is_main_process:
                        logger.info("Optimizer state dict loaded successfully")
                except Exception as e:
                    if accelerator.is_main_process:
                        logger.warning(f"Failed to load optimizer state dict: {e}. Continuing with fresh optimizer state.")
            
            # Extract train_steps from checkpoint
            # Try to get from checkpoint dict first, then from filename
            if 'train_steps' in checkpoint:
                train_steps = checkpoint['train_steps']
            elif 'step' in checkpoint:
                train_steps = checkpoint['step']
            else:
                # Extract from filename (format: 0200000.pt)
                try:
                    filename = os.path.basename(checkpoint_path)
                    train_steps = int(filename.split('.')[0])
                except:
                    if accelerator.is_main_process:
                        logger.warning(f"Could not extract step number from checkpoint filename: {checkpoint_path}")
                    train_steps = 0
            
            if accelerator.is_main_process:
                logger.info(f"Resuming training from step: {train_steps}")
        else:
            if accelerator.is_main_process:
                logger.warning(f"Checkpoint path specified but file does not exist: {checkpoint_path}. Starting training from scratch.")
            train_steps = 0
    else:
        if accelerator.is_main_process:
            logger.info("No checkpoint specified. Starting training from scratch.")
        train_steps = 0
    
    vae = None
    try:
        if accelerator.is_main_process:
            logger.info(f"Loading VAE for debug visualization...")
            
        vae_config_path = "./tokenizer/configs/vavae_f16d32.yaml"
        with accelerator.main_process_first(): 
            vae = VA_VAE(vae_config_path)
            
        vae.model.to(device) 
        vae.model.eval()
        
        if accelerator.is_main_process:
            logger.info("VAE loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load VAE on rank {accelerator.local_process_index}: {e}")
        raise e

    #latent_mean, latent_std = dataset.get_latent_stats()
   # latent_mean = latent_mean.to(device)
    #latent_std = latent_std.to(device)

    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    use_checkpoint = train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")

    while True:
        for gts, inputs, label in loader: # x = gt, y = degradation
            if accelerator.mixed_precision == 'no':
                gts = gts.to(device,dtype=torch.float32)
                inputs = inputs.to(device,dtype=torch.float32)
            else:
                gts = gts.to(device)
                inputs = inputs.to(device)
                label = label.to(device)
            if len(gts.shape) == 5:
                gts = gts.squeeze(1)
            if len(inputs.shape) == 5:
                inputs = inputs.squeeze(1)
            with torch.no_grad():
                posterior_gt, _ = vae.model.encode(gts)
                x = posterior_gt.sample() 
                posterior_in, feats_in = vae.model.encode(inputs)
                y = posterior_in.sample()
                if train_config['data'].get('latent_norm', True):
                    x = (x - latent_mean) / latent_std
                    y = (y - latent_mean) / latent_std

                x = x * latent_multiplier
                y = y * latent_multiplier
            
            inputs_y = y # F.interpolate(inputs, size=y.shape[-2:], mode='bilinear', align_corners=False)
            model_kwargs = dict(y=y, inputs=inputs_y, label=label)
            timestep_shift = train_config['sample'].get('timestep_shift', 1.0)
            loss_dict = transport.training_losses(model, x, model_kwargs, timestep_shift=timestep_shift)
            base_loss = loss_dict["loss"].mean()
            cos_loss = loss_dict["cos_loss"].mean() if "cos_loss" in loss_dict else None
            y_loss = loss_dict["y_loss"].mean() if "y_loss" in loss_dict else None
            loss = base_loss
            if cos_loss is not None:
                loss = loss + cos_loss
            if y_loss is not None:
                loss = loss + y_loss
            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if accelerator.is_main_process:
                writer.add_scalar("Loss/total", loss.item(), train_steps)
                writer.add_scalar("Loss/loss", base_loss.item(), train_steps)
                writer.add_scalar("Loss/y_loss", 0.0 if y_loss is None else y_loss.item(), train_steps)
                writer.add_scalar("Loss/cos_loss", 0.0 if cos_loss is None else cos_loss.item(), train_steps)
            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss_value = running_loss / log_steps
                if accelerator.is_main_process:
                    #print(f"(step={train_steps:07d}) Train Loss: {avg_loss_value:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss_value:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    writer.add_scalar('Loss/train', avg_loss_value, train_steps)

                # Sample one validation item per 10 log intervals.
                # Run validation computation on ALL ranks (only main writes IO/logs),
                # so step timing stays identical across ranks.
                if valid_dataset is not None and train_steps % (10 * train_config['train']['log_every']) == 0:
                    val_count = train_steps // (10 * train_config['train']['log_every'])
                    val_idx = val_count % len(valid_dataset)
                    val_psnr, val_ssim = sample_validation(
                        model,
                        valid_dataset,
                        device,
                        sample_fn,
                        vae,
                        valid_latent_mean,
                        valid_latent_std,
                        latent_multiplier,
                        validation_sample_dir,
                        train_steps,
                        train_config,
                        accelerator,
                        writer,
                        idx=val_idx,
                    )
                    if accelerator.is_main_process:
                        logger.info(
                            f"(step={train_steps:07d}) Val PSNR/SSIM (deterministic sample): "
                            f"{val_psnr:.2f}/{val_ssim:.4f}"
                        )
                        writer.add_scalar('Metrics/psnr_validation_sample', val_psnr, train_steps)
                        writer.add_scalar('Metrics/ssim_validation_sample', val_ssim, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                accelerator.wait_for_everyone()
                model_state = accelerator.get_state_dict(model)
                opt_state = opt.state_dict()
                ema_state = ema.state_dict()
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model_state,
                        "ema": ema_state,
                        "opt": opt_state,
                        "config": train_config,
                        "train_steps": train_steps,  # Save train_steps for resuming
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    accelerator.save(checkpoint, checkpoint_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                #dist.barrier()
                accelerator.wait_for_everyone()

                # Optional: keep heavy validation out of the checkpoint loop
            if train_steps >= train_config['train']['max_steps']:
                break
        if train_steps >= train_config['train']['max_steps']:
            break

    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator

def load_weights_with_shape_check(model, checkpoint, rank=0):
    
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                # special case for x_embedder.proj.weight
                # the pretrained model is trained with 256x256 images
                # we can load the weights by resizing the weights
                # and keep the first 3 channels the same
                weight = torch.zeros_like(model_state_dict[name])
                weight[:, :16] = param[:, :16]
                model_state_dict[name] = weight
            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # Use lerp_ for better numerical stability and potential kernel fallback
        ema_params[name].lerp_(param.data, 1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def _compute_psnr(x, y, max_val=1.0):
    mse = F.mse_loss(x, y, reduction="mean")
    if mse.item() == 0:
        return torch.tensor(float("inf"), device=x.device)
    return 20.0 * torch.log10(torch.tensor(max_val, device=x.device) / torch.sqrt(mse))


def _compute_ssim(x, y, window_size=11, max_val=1.0):
    # Simple SSIM over NCHW tensors in [0, 1]
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean()


@torch.no_grad()
def sample_validation(
    model,
    valid_dataset,
    device,
    sample_fn,
    vae,
    latent_mean,
    latent_std,
    latent_multiplier,
    validation_sample_dir,
    train_steps,
    train_config,
    accelerator,
    writer,
    idx=None,
):
    model.eval()
    if idx is None:
        idx = np.random.randint(len(valid_dataset))
    pixel_gt, pixel_in, label = valid_dataset[idx]  # dataset returns (latent_gt, latent_in, pixel_gt, pixel_in, label)
    pixel_gt = pixel_gt.unsqueeze(0).to(device)
    pixel_in = pixel_in.unsqueeze(0).to(device)
    with torch.no_grad():
        posterior_gt, _ = vae.model.encode(pixel_gt)
        x_val = posterior_gt.sample() 
        posterior_in, feats_in = vae.model.encode(pixel_in)
        y_val = posterior_in.sample()
        if train_config['data'].get('latent_norm', True):
            x_val = (x_val - latent_mean) / latent_std
            y_val = (y_val - latent_mean) / latent_std

        x_val = x_val * latent_multiplier
        y_val = y_val * latent_multiplier
    # Run inference from degraded to clean
    cfg_scale = train_config['sample'].get('cfg_scale', 1.0)
    cfg_interval_start = train_config['sample'].get('cfg_interval_start', 0)
    using_cfg = cfg_scale > 1.0
    z = y_val
    inputs_y = y_val # F.interpolate(pixel_in, size=y_val.shape[-2:], mode='bilinear', align_corners=False)
    if using_cfg:
        z = torch.cat([z, z], 0)
        inputs_y = torch.cat([inputs_y, inputs_y], 0)
        model_kwargs = dict(
            y=inputs_y,
            cfg_scale=cfg_scale,
            cfg_interval=True,
            cfg_interval_start=cfg_interval_start,
        )
        model_fn = model.forward_with_cfg
    else:
        model_kwargs = dict(y=inputs_y)
        model_fn = model.forward

    samples = sample_fn(z, model_fn, **model_kwargs)[-1]
    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)

    val_psnr = torch.tensor(float("nan"), device=device)
    val_ssim = torch.tensor(float("nan"), device=device)
    if vae is not None:
        sample_latents = (samples / latent_multiplier) * latent_std + latent_mean
        degraded_latents = (y_val / latent_multiplier) * latent_std + latent_mean
        gt_latents = (x_val / latent_multiplier) * latent_std + latent_mean
        
        # Add diagnostic logging for debugging
        if accelerator.is_main_process and train_steps % (10 * train_config['train']['log_every']) == 0:
            logger = logging.getLogger(__name__)
            logger.info(f"Sample stats - min: {samples.min().item():.4f}, max: {samples.max().item():.4f}, mean: {samples.mean().item():.4f}, std: {samples.std().item():.4f}")
            logger.info(f"Sample latents stats - min: {sample_latents.min().item():.4f}, max: {sample_latents.max().item():.4f}, mean: {sample_latents.mean().item():.4f}")
        
        sample_imgs = vae.decode_to_images(sample_latents)
        degraded_imgs = vae.decode_to_images(degraded_latents)
        gt_imgs = vae.decode_to_images(gt_latents)
        
        gts = (pixel_gt + 1.0) / 2.0
        inputs = (pixel_in + 1.0) / 2.0

        # Perform disk IO and TensorBoard logging only on the main process.
        if accelerator.is_main_process:
            os.makedirs(validation_sample_dir, exist_ok=True)
            Image.fromarray(sample_imgs[0]).save(
                f"{validation_sample_dir}/step_{train_steps:07d}_clean.png"
            )
            Image.fromarray(degraded_imgs[0]).save(
                f"{validation_sample_dir}/step_{train_steps:07d}_vae_degraded.png"
            )
            Image.fromarray(gt_imgs[0]).save(
                f"{validation_sample_dir}/step_{train_steps:07d}_vae_gt.png"
            )
        
        gt_orig_np = (gts.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        in_orig_np = (inputs.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if accelerator.is_main_process:
            Image.fromarray(gt_orig_np).save(f"{validation_sample_dir}/step_{train_steps:07d}_gt.png")
            Image.fromarray(in_orig_np).save(f"{validation_sample_dir}/step_{train_steps:07d}_input.png")
        
        # Get ground truth dimensions for resizing
        h_gt, w_gt = gt_orig_np.shape[:2]
        
        # Log images to TensorBoard (HWC uint8 -> CHW float)
        
        sample_tensor = torch.from_numpy(sample_imgs[0]).permute(2, 0, 1).float() / 255.0
        #degraded_tensor = torch.from_numpy(degraded_imgs[0]).permute(2, 0, 1).float() / 255.0
        #gt_tensor = torch.from_numpy(gt_imgs[0]).permute(2, 0, 1).float() / 255.0
        
        # TensorBoard: resize clean to central-crop size so all three same size
        sample_pil = Image.fromarray(sample_imgs[0]).resize((w_gt, h_gt), Image.LANCZOS)
        sample_at_gt_size = np.array(sample_pil)
        sample_tensor = torch.from_numpy(sample_at_gt_size).permute(2, 0, 1).float() / 255.0
        gt_tensor = torch.from_numpy(gt_orig_np).permute(2, 0, 1).float() / 255.0
        degraded_tensor = torch.from_numpy(in_orig_np).permute(2, 0, 1).float() / 255.0
        view_tensor = torch.cat([sample_tensor, degraded_tensor, gt_tensor], dim=2)
        if writer is not None and accelerator.is_main_process:
            writer.add_image("Validation/validation_sample", view_tensor, train_steps)

        pred_img = sample_tensor.unsqueeze(0).clamp(0.0, 1.0)
        #gt_img = gt_tensor.unsqueeze(0).clamp(0.0, 1.0)
        gt_img = gt_tensor.unsqueeze(0).clamp(0.0, 1.0)
        val_psnr = _compute_psnr(pred_img, gt_img)
        val_ssim = _compute_ssim(pred_img, gt_img)

    model.train()
    return val_psnr.item(), val_ssim.item()

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_logger(logging_dir, accelerator):
    """
    Create a logger that writes to a log file and stdout.
    """
    logger = logging.getLogger(__name__)

    if accelerator.is_main_process:  # real logger
        os.makedirs(logging_dir, exist_ok=True)

        # IMPORTANT:
        # In multi-process environments, other libs may configure logging before us.
        # `logging.basicConfig(...)` then becomes a no-op, which causes `logger.info(...)`
        # not to be emitted (your symptoms: console/log.txt missing logs, validation never printing).
        # So we explicitly rebuild root handlers.
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)

        formatter = logging.Formatter(
            fmt='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            filename=f"{logging_dir}/log.txt",
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        root_logger.addHandler(stream_handler)
        root_logger.addHandler(file_handler)
    else:  # dummy logger (does nothing)
        logger.addHandler(logging.NullHandler())

    return logger

if __name__ == "__main__":
    # read config
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    do_train(train_config, accelerator)
    
