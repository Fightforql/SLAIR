import argparse
import logging
import os
import gc
from datetime import datetime

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataset import ImgPatchDataset
from models.lightningdit_addy import LightningDiT_models
from tokenizer.vavae import VA_VAE
from transport import Sampler, create_transport


class WaveletLoss(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        kernel_low = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        kernel_h = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        kernel_v = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        kernel_d = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        self.kernel = torch.stack([kernel_low, kernel_h, kernel_v, kernel_d], dim=0).unsqueeze(1).to(device)
        self.kernel.requires_grad_(False)

    def get_wavelet_coeffs(self, x):
        b, c, h, w = x.shape
        x = x.view(b * c, 1, h, w)
        coeffs = F.conv2d(x, self.kernel, stride=2)
        coeffs = coeffs.view(b, c, 4, h // 2, w // 2)
        return coeffs # [B, C, 4, H/2, W/2]

    def forward(self, pred, target):
        pred_coeffs = self.get_wavelet_coeffs(pred)
        target_coeffs = self.get_wavelet_coeffs(target)

        hf_loss = F.l1_loss(pred_coeffs[:, :, 1:], target_coeffs[:, :, 1:])
        return hf_loss

class FusionLoss(torch.nn.Module):
    def __init__(self, mse_weight=1.0, perceptual_weight=0.1, high_freq_weight=0.5, device='cuda',
                 adaptive_weight=True, momentum=0.99):
        super().__init__()
        self.base_mse_weight = mse_weight
        self.base_perceptual_weight = perceptual_weight
        self.base_high_freq_weight = high_freq_weight


        # Track loss statistics for adaptive weights after a 1,000-step warmup.
        self.adaptive_weight = adaptive_weight
        self.momentum = momentum


        self.register_buffer('mse_mean', torch.tensor(0.0))
        self.register_buffer('mse_std', torch.tensor(1.0))
        self.register_buffer('p_loss_mean', torch.tensor(0.1))
        self.register_buffer('p_loss_std', torch.tensor(0.05))
        self.register_buffer('hf_loss_mean', torch.tensor(0.01))
        self.register_buffer('hf_loss_std', torch.tensor(0.01))
        self.register_buffer('step_count', torch.tensor(0))


        self.current_mse_weight = mse_weight
        self.current_p_weight = perceptual_weight
        self.current_hf_weight = high_freq_weight

        self.perceptual_loss = lpips.LPIPS(net='vgg').to(device).eval()
        self.perceptual_loss.requires_grad_(False)
        self.wavelet_loss = WaveletLoss(device=device)

    def forward(self, inputs, rec):


        if inputs.dtype != torch.float32:
            inputs = inputs.float()
        if rec.dtype != torch.float32:
            rec = rec.float()

        mse_loss = F.mse_loss(rec, inputs, reduction='mean')
        hf_loss = self.wavelet_loss(rec, inputs)
        p_loss = self.perceptual_loss(inputs, rec).mean()

        if self.adaptive_weight and self.training:

            warmup_steps = 1000

            with torch.no_grad():
                self.step_count += 1


                mse_val = mse_loss.item()
                p_val = p_loss.item()
                hf_val = hf_loss.item()

                if self.step_count <= warmup_steps:


                    if self.step_count.item() == 1:
                        alpha = 0.5
                    else:
                        alpha = 1.0 / self.step_count.item()
                    self.mse_mean = (1 - alpha) * self.mse_mean + alpha * mse_val
                    self.p_loss_mean = (1 - alpha) * self.p_loss_mean + alpha * p_val
                    self.hf_loss_mean = (1 - alpha) * self.hf_loss_mean + alpha * hf_val
                else:

                    self.mse_mean = self.momentum * self.mse_mean + (1 - self.momentum) * mse_val
                    self.p_loss_mean = self.momentum * self.p_loss_mean + (1 - self.momentum) * p_val
                    self.hf_loss_mean = self.momentum * self.hf_loss_mean + (1 - self.momentum) * hf_val


                mse_diff = abs(mse_val - self.mse_mean.item())
                p_diff = abs(p_val - self.p_loss_mean.item())
                hf_diff = abs(hf_val - self.hf_loss_mean.item())

                if self.step_count <= warmup_steps:

                    if self.step_count.item() == 1:
                        alpha = 0.5
                    else:
                        alpha = 1.0 / self.step_count.item()
                    self.mse_std = (1 - alpha) * self.mse_std + alpha * mse_diff
                    self.p_loss_std = (1 - alpha) * self.p_loss_std + alpha * p_diff
                    self.hf_loss_std = (1 - alpha) * self.hf_loss_std + alpha * hf_diff
                else:
                    self.mse_std = self.momentum * self.mse_std + (1 - self.momentum) * mse_diff
                    self.p_loss_std = self.momentum * self.p_loss_std + (1 - self.momentum) * p_diff
                    self.hf_loss_std = self.momentum * self.hf_loss_std + (1 - self.momentum) * hf_diff


            if self.step_count <= warmup_steps:
                mse_weight = self.base_mse_weight
                p_weight = self.base_perceptual_weight
                hf_weight = self.base_high_freq_weight


                self.current_mse_weight = mse_weight
                self.current_p_weight = p_weight
                self.current_hf_weight = hf_weight
            else:


                mse_z = (mse_val - self.mse_mean.item()) / (self.mse_std.item() + 1e-6)
                p_z = (p_val - self.p_loss_mean.item()) / (self.p_loss_std.item() + 1e-6)
                hf_z = (hf_val - self.hf_loss_mean.item()) / (self.hf_loss_std.item() + 1e-6)


                # Reduce weights for unusually large losses and gently raise them for small losses.
                def adjust_weight(base_weight, z_score, scale=0.15):


                    if z_score > 0:

                        adjustment = 1.0 / (1.0 + scale * z_score)
                    else:

                        adjustment = 1.0 + scale * abs(z_score) * 0.5
                    return base_weight * adjustment

                mse_weight = adjust_weight(self.base_mse_weight, mse_z, scale=0.15)
                p_weight = adjust_weight(self.base_perceptual_weight, p_z, scale=0.15)
                hf_weight = adjust_weight(self.base_high_freq_weight, hf_z, scale=0.15)


                mse_weight = max(self.base_mse_weight * 0.3, min(mse_weight, self.base_mse_weight * 2.0))
                p_weight = max(self.base_perceptual_weight * 0.3, min(p_weight, self.base_perceptual_weight * 2.0))
                hf_weight = max(self.base_high_freq_weight * 0.3, min(hf_weight, self.base_high_freq_weight * 2.0))


                self.current_mse_weight = mse_weight
                self.current_p_weight = p_weight
                self.current_hf_weight = hf_weight
        else:

            mse_weight = self.base_mse_weight
            p_weight = self.base_perceptual_weight
            hf_weight = self.base_high_freq_weight


            self.current_mse_weight = mse_weight
            self.current_p_weight = p_weight
            self.current_hf_weight = hf_weight

        loss = (p_weight * p_loss +
                mse_weight * mse_loss +
                hf_weight * hf_loss)

        return loss, mse_loss, p_loss, hf_loss


def denorm(tensor):
    return (tensor.detach().cpu() + 1.0) / 2.0


def save_image_local(canvas, log_dir, global_step, prefix="train"):
    save_path = os.path.join(log_dir, "visual_results", prefix)
    os.makedirs(save_path, exist_ok=True)
    file_name = f"step_{global_step:07d}.png"
    vutils.save_image(canvas, os.path.join(save_path, file_name))


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y, crop_x = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class OnlineDecoderDataset(ImgPatchDataset):
    """Match the cached decoder training images without storing latents."""

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        with Image.open(row.iloc[0]) as image:
            pixel_in = center_crop_arr(image.convert("RGB"), 512)
        with Image.open(row.iloc[1]) as image:
            pixel_gt = center_crop_arr(image.convert("RGB"), 512)
        pixel_in = Image.fromarray(self.histogram_match_np(np.array(pixel_in), np.array(pixel_gt)))
        return self.pixel_transform(pixel_gt), self.pixel_transform(pixel_in), torch.tensor(int(row.iloc[2]), dtype=torch.long)


def build_decoder_dataloaders(csv_path):
    dataset = OnlineDecoderDataset(csv_path=csv_path, patch_size=512, is_train=False)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train_size = int(0.98 * len(dataset))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    rain_prefixes = ("/openbayes/input/input0/Rain1400", "/openbayes/input/input0/Outdoor-Rain")
    moire_prefix = "/openbayes/input/input0/AIO/train/9"
    hard_samples = [
        any(prefix in str(dataset.data_info.iloc[i, 0]) for prefix in rain_prefixes)
        or moire_prefix in str(dataset.data_info.iloc[i, 0])
        for i in train_idx
    ]
    num_hard = sum(hard_samples)
    sampler = WeightedRandomSampler(
        [2.0 if hard else 1.0 for hard in hard_samples],
        num_samples=len(train_idx) + num_hard,
        replacement=True,
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=2, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=2, shuffle=True)
    return dataset, train_loader, val_loader, num_hard


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

        _, mse_loss, p_loss, hf_loss = loss_module(pixel_gt, rec_fine)
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
            if "conv_out" in name:
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

    train_dataset, train_loader, val_loader, num_hard = build_decoder_dataloaders(train_csv_path)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, autoencoder.decoder.parameters()),
        lr=1e-4,
    )
    # Keep the reference weights, including its disabled high-frequency contribution.
    loss_module = FusionLoss(
        mse_weight=1.0, perceptual_weight=0.5, high_freq_weight=0,
        device=device, adaptive_weight=True, momentum=0.99,
    ).to(device)

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
    log_every = 100
    ckpt_every = 2000
    val_every = 2000

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset)} images; hard training samples: {num_hard}")
        logger.info("Batch size: 2 per process; training for 10 epochs")
        logger.info("Loss weights: MSE=1.0, LPIPS=0.5, high_freq=0; adaptive=True, momentum=0.99")

    global_step = 0
    for epoch in range(10):
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
            total_loss, mse_loss, p_loss, hf_loss = loss_module(pixel_gt, rec)

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()

            if accelerator.is_main_process:
                writer.add_scalar("Loss/total_loss", total_loss.item(), global_step)
                writer.add_scalar("Loss/mse_loss", mse_loss.item(), global_step)
                writer.add_scalar("Loss/p_loss", p_loss.item(), global_step)
                writer.add_scalar("Loss/high_freq", hf_loss.item(), global_step)
                writer.add_scalar("Weights/mse", loss_module.current_mse_weight, global_step)
                writer.add_scalar("Weights/perceptual", loss_module.current_p_weight, global_step)
                writer.add_scalar("Weights/high_freq", loss_module.current_hf_weight, global_step)

                if global_step % log_every == 0:
                    pbar.set_postfix(loss=f"{total_loss.item():.4f}", mse=f"{mse_loss.item():.4f}", lpips=f"{p_loss.item():.4f}", hf=f"{hf_loss.item():.4f}")

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
            del rec, total_loss, mse_loss, p_loss, hf_loss, feats, z_refined
            if global_step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()


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
