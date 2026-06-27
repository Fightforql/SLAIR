import argparse
import logging
import os
from collections import Counter
from datetime import datetime

import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataset import ImgPatchDataset
from models.lightningdit_addy import LightningDiT_models
from tokenizer.vavae import VA_VAE
from transport import Sampler, create_transport


class FusionLoss(torch.nn.Module):
    def __init__(self, mse_weight=1.0, perceptual_weight=0.1, device="cuda"):
        super().__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = lpips.LPIPS(net="vgg").to(device).eval()
        self.perceptual_loss.requires_grad_(False)

    def forward(self, inputs, rec):
        mse_loss = F.mse_loss(rec.float(), inputs.float(), reduction="mean")
        p_loss = self.perceptual_loss(inputs, rec).mean()
        loss = self.perceptual_weight * p_loss + self.mse_weight * mse_loss
        return loss, mse_loss, p_loss


def denorm(tensor):
    return (tensor.detach().cpu() + 1.0) / 2.0


def save_image_local(canvas, log_dir, global_step, prefix="train"):
    save_path = os.path.join(log_dir, "visual_results", prefix)
    os.makedirs(save_path, exist_ok=True)
    file_name = f"step_{global_step:07d}.png"
    vutils.save_image(canvas, os.path.join(save_path, file_name))


def build_weighted_sampler(csv_path, num_samples):
    all_labels = pd.read_csv(csv_path).iloc[:, 2].values
    label_counts = Counter(all_labels)
    sample_weights = np.array([1.0 / label_counts[label] for label in all_labels], dtype=np.float32)
    sample_weights = torch.from_numpy(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )
    return sampler, label_counts


def build_dataloader(csv_path, patch_size, batch_size, num_workers, is_train, balance_labels):
    dataset = ImgPatchDataset(csv_path=csv_path, patch_size=patch_size, is_train=is_train)
    sampler = None
    label_counts = None
    if is_train and balance_labels:
        sampler, label_counts = build_weighted_sampler(csv_path, len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None and is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    return dataset, loader, label_counts


def refine_latents(pixel_in, label, autoencoder, dit_model, sample_fn, latent_mean, latent_std, latent_multiplier, cfg_scale, cfg_interval_start):
    posterior, feats = autoencoder.encode(pixel_in)
    z_raw = posterior.sample()
    if latent_mean is not None and latent_std is not None:
        z = (z_raw - latent_mean) / latent_std * latent_multiplier
    else:
        z = z_raw

    if cfg_scale > 1.0:
        z_in = torch.cat([z, z], 0)
        label_in = torch.cat([label, label], 0)
        model_kwargs = dict(
            y=z_in,
            inputs=z_in,
            cfg_scale=cfg_scale,
            cfg_interval=True,
            cfg_interval_start=cfg_interval_start,
            label=label_in,
        )
        model_fn = dit_model.forward_with_cfg
        samples_list = sample_fn(z_in, model_fn, **model_kwargs)
        z_refined = samples_list[-1]
        z_refined, _ = z_refined.chunk(2, dim=0)
    else:
        model_kwargs = dict(y=z, label=label, inputs=z)
        model_fn = dit_model.forward
        samples_list = sample_fn(z, model_fn, **model_kwargs)
        z_refined = samples_list[-1]

    if latent_mean is not None and latent_std is not None:
        z_refined = (z_refined * latent_std) / latent_multiplier + latent_mean
    return z_refined, feats


@torch.no_grad()
def run_validation(autoencoder, dit_model, sample_fn, val_loader, loss_module, device, accelerator, writer, global_step, logger, log_dir, latent_stats, cfg_scale, cfg_interval_start):
    autoencoder.eval()
    latent_mean, latent_std, latent_multiplier = latent_stats
    val_mses, val_lpips = [], []

    num_val_batches = len(val_loader)
    vis_indices = set(np.random.choice(num_val_batches, size=min(num_val_batches, 4), replace=False).tolist()) if num_val_batches > 0 else set()

    val_pbar = tqdm(
        enumerate(val_loader),
        total=num_val_batches,
        desc=f"Validating @ Step {global_step}",
        leave=False,
        disable=not accelerator.is_main_process,
    )

    for i, (pixel_gt, pixel_in, label) in val_pbar:
        if i > 20:
            break

        pixel_gt = pixel_gt.to(device)
        pixel_in = pixel_in.to(device)
        label = label.to(device)

        z_refined, feats = refine_latents(
            pixel_in,
            label,
            autoencoder,
            dit_model,
            sample_fn,
            latent_mean,
            latent_std,
            latent_multiplier,
            cfg_scale,
            cfg_interval_start,
        )

        rec_fine = autoencoder.decode(z_refined, feats=feats)
        rec_orig = autoencoder.decode(z_refined, feats=None)

        _, mse_loss, p_loss = loss_module(pixel_gt, rec_fine)
        val_mses.append(mse_loss.item())
        val_lpips.append(p_loss.item())

        if i in vis_indices and accelerator.is_main_process:
            diff = torch.clamp(torch.abs(rec_fine - rec_orig)[0].cpu() * 5.0, 0, 1)
            canvas = torch.cat(
                [
                    denorm(pixel_in[0]),
                    denorm(pixel_gt[0]),
                    denorm(rec_orig[0]),
                    denorm(rec_fine[0]),
                    diff,
                ],
                dim=2,
            )
            writer.add_image(f"Visual_Val/Batch_{i}", canvas, global_step)
            save_image_local(canvas, log_dir, global_step, prefix=f"val_batch_{i}")

    if accelerator.is_main_process and len(val_mses) > 0:
        logger.info(f"[Validation] Step {global_step} | MSE: {np.mean(val_mses):.5f} | LPIPS: {np.mean(val_lpips):.5f}")
        writer.add_scalar("Val/MSE", np.mean(val_mses), global_step)
        writer.add_scalar("Val/LPIPS", np.mean(val_lpips), global_step)

    autoencoder.train()


def train_fusion_layer(config_path, dit_ckpt_path=None, custom_save_dir=None):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    if custom_save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_save_dir = f"/openbayes/input/input0/logs/decoder/fusion_train_{timestamp}"

    log_dir = custom_save_dir
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    tb_dir = os.path.join(log_dir, "tensorboard")
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(f"{log_dir}/train_log.txt"),
                logging.StreamHandler(),
            ],
        )
        logger = logging.getLogger(__name__)
        writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"save path: {log_dir}")
    else:
        logger = None
        writer = None

    with open(config_path, "r", encoding="utf-8") as file:
        train_config = yaml.safe_load(file)

    train_csv_path = train_config["data"].get("csv_path")
    if not train_csv_path:
        raise ValueError("data.csv_path is required for online decoder training.")
    stats_path = train_config["data"].get("stats_path")
    if train_config["data"].get("latent_norm", True) and not stats_path:
        raise ValueError("data.stats_path is required when latent_norm=true.")

    resolved_dit_ckpt = (
        dit_ckpt_path
        or train_config.get("ckpt_path")
        or train_config.get("train", {}).get("ckpt")
    )
    if not resolved_dit_ckpt:
        raise ValueError("DiT checkpoint path is required. Pass --dit_ckpt_path or set ckpt_path/train.ckpt in config.")

    vae_wrapper = VA_VAE(f'tokenizer/configs/{train_config["vae"]["model_name"]}.yaml')
    autoencoder = vae_wrapper.model
    autoencoder.requires_grad_(False)
    for name, param in autoencoder.decoder.named_parameters():
        if "fusion_layers" in name:
            param.requires_grad = True
            if "encode_enc_3.conv_out" in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.constant_(param, 1e-6)

    latent_size = train_config["data"]["image_size"] // train_config["vae"]["downsample_ratio"]
    dit_model = LightningDiT_models[train_config["model"]["model_type"]](
        input_size=latent_size,
        num_classes=train_config["data"]["num_classes"],
        use_qknorm=train_config["model"]["use_qknorm"],
        use_swiglu=train_config["model"].get("use_swiglu", False),
        use_rope=train_config["model"].get("use_rope", False),
        use_rmsnorm=train_config["model"].get("use_rmsnorm", False),
        wo_shift=train_config["model"].get("wo_shift", False),
        in_channels=train_config["model"].get("in_chans", 4),
        use_checkpoint=train_config["model"].get("use_checkpoint", False),
    )

    dit_ckpt = torch.load(resolved_dit_ckpt, map_location="cpu")
    dit_model.load_state_dict(dit_ckpt.get("ema", dit_ckpt.get("model", dit_ckpt)))
    dit_model.requires_grad_(False)
    dit_model.eval()

    transport = create_transport(
        train_config["transport"]["path_type"],
        train_config["transport"]["prediction"],
        train_config["transport"]["loss_weight"],
        train_config["transport"]["train_eps"],
        train_config["transport"]["sample_eps"],
        use_cosine_loss=train_config["transport"].get("use_cosine_loss", False),
        use_lognorm=train_config["transport"].get("use_lognorm", False),
    )
    sampler = Sampler(transport)
    if train_config["sample"]["mode"] != "ODE":
        raise NotImplementedError(f"Sampling mode {train_config['sample']['mode']} is not supported.")
    sample_fn = sampler.sample_ode(
        sampling_method=train_config["sample"]["sampling_method"],
        num_steps=train_config["sample"]["num_sampling_steps"],
        atol=train_config["sample"]["atol"],
        rtol=train_config["sample"]["rtol"],
        reverse=train_config["sample"]["reverse"],
        timestep_shift=train_config["sample"].get("timestep_shift", 0),
    )

    if train_config["data"].get("latent_norm", True):
        latent_stats = torch.load(stats_path, map_location="cpu")
        latent_mean = latent_stats["mean"].to(device)
        latent_std = latent_stats["std"].to(device)
    else:
        latent_mean = None
        latent_std = None
    latent_multiplier = train_config["data"].get("latent_multiplier", 1.0)
    latent_stats_tuple = (latent_mean, latent_std, latent_multiplier)

    batch_size_per_gpu = int(np.round(train_config["train"]["global_batch_size"] / accelerator.num_processes))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes

    train_dataset, train_loader, train_label_counts = build_dataloader(
        csv_path=train_csv_path,
        patch_size=train_config["data"]["image_size"],
        batch_size=batch_size_per_gpu,
        num_workers=train_config["data"]["num_workers"],
        is_train=True,
        balance_labels=True,
    )

    val_loader = None
    if "valid_data" in train_config and train_config["valid_data"].get("csv_path"):
        _, val_loader, _ = build_dataloader(
            csv_path=train_config["valid_data"]["csv_path"],
            patch_size=train_config["valid_data"]["image_size"],
            batch_size=min(4, batch_size_per_gpu),
            num_workers=min(4, train_config["valid_data"].get("num_workers", train_config["data"]["num_workers"])),
            is_train=False,
            balance_labels=False,
        )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, autoencoder.decoder.parameters()),
        lr=5e-5,
        weight_decay=1e-4,
    )
    loss_module = FusionLoss(mse_weight=0.01, perceptual_weight=0.001, device=device)

    if val_loader is not None:
        autoencoder, dit_model, optimizer, train_loader, val_loader = accelerator.prepare(
            autoencoder, dit_model, optimizer, train_loader, val_loader
        )
    else:
        autoencoder, dit_model, optimizer, train_loader = accelerator.prepare(
            autoencoder, dit_model, optimizer, train_loader
        )

    autoencoder.train()
    dit_model.eval()

    cfg_scale = train_config["sample"].get("cfg_scale", 1.0)
    cfg_interval_start = train_config["sample"].get("cfg_interval_start", 0.0)
    log_every = train_config["train"].get("log_every", 100)
    ckpt_every = train_config["train"].get("ckpt_every", 500)
    val_every = train_config["train"].get("val_every", 3000)
    max_steps = train_config["train"].get("max_steps", 100000)

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset)} images")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
        logger.info(f"Label distribution: {dict(sorted(train_label_counts.items()))}")
        logger.info(f"WeightedRandomSampler: num_samples={len(train_dataset)}")

    global_step = 0
    epoch = 0
    while global_step < max_steps:
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, dynamic_ncols=True, desc=f"Epoch {epoch}")
        for pixel_gt, pixel_in, label in pbar:
            pixel_gt = pixel_gt.to(device)
            pixel_in = pixel_in.to(device)
            label = label.to(device)

            with torch.no_grad():
                z_refined, feats = refine_latents(
                    pixel_in,
                    label,
                    autoencoder,
                    dit_model,
                    sample_fn,
                    latent_mean,
                    latent_std,
                    latent_multiplier,
                    cfg_scale,
                    cfg_interval_start,
                )

            rec = autoencoder.decode(z_refined, feats=feats)
            total_loss, mse_loss, p_loss = loss_module(pixel_gt, rec)

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()

            if accelerator.is_main_process:
                writer.add_scalar("Loss/total_loss", total_loss.item(), global_step)
                writer.add_scalar("Loss/mse_loss", mse_loss.item(), global_step)
                writer.add_scalar("Loss/p_loss", p_loss.item(), global_step)

                if global_step % log_every == 0:
                    pbar.set_postfix(loss=f"{total_loss.item():.4f}", mse=f"{mse_loss.item():.4f}", lpips=f"{p_loss.item():.4f}")

                if global_step % 1000 == 0:
                    with torch.no_grad():
                        rec_orig = autoencoder.decode(z_refined, feats=None)
                        diff = torch.clamp(torch.abs(rec - rec_orig)[0].cpu() * 5.0, 0.0, 1.0)
                    canvas = torch.cat(
                        [
                            denorm(pixel_in[0]),
                            denorm(pixel_gt[0]),
                            denorm(rec_orig[0]),
                            denorm(rec[0]),
                            diff,
                        ],
                        dim=2,
                    )
                    writer.add_image("Visual/Detailed_Comparison", canvas, global_step)
                    save_image_local(canvas, log_dir, global_step)

                if global_step > 0 and global_step % ckpt_every == 0:
                    fusion_state = {
                        key: value.cpu()
                        for key, value in accelerator.unwrap_model(autoencoder).state_dict().items()
                        if "fusion_layers" in key
                    }
                    torch.save(fusion_state, os.path.join(ckpt_dir, f"step_{global_step}.pt"))

                if global_step > 0 and global_step % val_every == 0 and val_loader is not None:
                    run_validation(
                        autoencoder,
                        dit_model,
                        sample_fn,
                        val_loader,
                        loss_module,
                        device,
                        accelerator,
                        writer,
                        global_step,
                        logger,
                        log_dir,
                        latent_stats_tuple,
                        cfg_scale,
                        cfg_interval_start,
                    )

            global_step += 1
            if global_step >= max_steps:
                break

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/train.yaml")
    parser.add_argument("--dit_ckpt_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    train_fusion_layer(
        config_path=args.config_path,
        dit_ckpt_path=args.dit_ckpt_path,
        custom_save_dir=args.save_dir,
    )
