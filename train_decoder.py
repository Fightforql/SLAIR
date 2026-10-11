import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import lpips
import glob
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
import logging
from tokenizer.vavae import VA_VAE
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.transforms as transforms
import gc
import pickle
import cv2
def histogram_match_np(src_img_np, ref_img_np):
        """以 ref_img_np 为参考，对 src_img_np 进行直方图匹配"""
        img = src_img_np.copy()
        imgRef = ref_img_np.copy()
        _, _, channel = img.shape
        imgOut = np.zeros_like(img)
        for i in range(channel):
            histImg, _ = np.histogram(img[:,:,i].flatten(), 256, [0, 256])
            histRef, _ = np.histogram(imgRef[:,:,i].flatten(), 256, [0, 256])
            cdfImg = np.cumsum(histImg)
            cdfRef = np.cumsum(histRef)
            total_pixels = img[:,:,i].size
            cdfImg_norm = cdfImg / total_pixels
            cdfRef_norm = cdfRef / total_pixels
            lut = np.zeros(256, dtype=np.uint8)
            for r in range(256):
                idx = np.argmin(np.abs(cdfImg_norm[r] - cdfRef_norm))
                lut[r] = idx
            imgOut[:,:,i] = cv2.LUT(img[:,:,i], lut)
        return imgOut
    
class WaveletLoss(nn.Module):
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
 
class FusionLoss(nn.Module):
    def __init__(self, mse_weight=1.0, perceptual_weight=0.1, high_freq_weight=0.5, device='cuda', 
                 adaptive_weight=True, momentum=0.99):
        super().__init__()
        self.base_mse_weight = mse_weight
        self.base_perceptual_weight = perceptual_weight
        self.base_high_freq_weight = high_freq_weight
        
        # 自适应权重相关
        self.adaptive_weight = adaptive_weight
        self.momentum = momentum
        
        # 使用滑动平均跟踪各 loss 分量的统计信息
        self.register_buffer('mse_mean', torch.tensor(0.0))
        self.register_buffer('mse_std', torch.tensor(1.0))
        self.register_buffer('p_loss_mean', torch.tensor(0.1))
        self.register_buffer('p_loss_std', torch.tensor(0.05))
        self.register_buffer('hf_loss_mean', torch.tensor(0.01))
        self.register_buffer('hf_loss_std', torch.tensor(0.01))
        self.register_buffer('step_count', torch.tensor(0))
        
        # 存储当前使用的权重（用于日志）
        self.current_mse_weight = mse_weight
        self.current_p_weight = perceptual_weight
        self.current_hf_weight = high_freq_weight
        
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(device).eval()
        self.perceptual_loss.requires_grad_(False)
        self.wavelet_loss = WaveletLoss(device=device)
 
    def forward(self, inputs, rec):
        # 计算各 loss 分量
        # 确保输入是 float 类型（如果已经是 float，这个操作很轻量）
        if inputs.dtype != torch.float32:
            inputs = inputs.float()
        if rec.dtype != torch.float32:
            rec = rec.float()
        
        mse_loss = F.mse_loss(rec, inputs, reduction='mean')
        hf_loss = self.wavelet_loss(rec, inputs)
        p_loss = self.perceptual_loss(inputs, rec).mean()
        
        if self.adaptive_weight and self.training:
            # Warmup: 前 1000 步使用基础权重，让统计信息稳定
            warmup_steps = 1000
            
            with torch.no_grad():
                self.step_count += 1
                
                # 更新均值（指数移动平均）
                mse_val = mse_loss.item()
                p_val = p_loss.item()
                hf_val = hf_loss.item()
                
                if self.step_count <= warmup_steps:
                    # Warmup 阶段：使用移动平均初始化
                    # 第一步使用较小的 alpha 避免突变，后续逐渐增加
                    if self.step_count.item() == 1:
                        alpha = 0.5  # 第一步使用 50% 的当前值
                    else:
                        alpha = 1.0 / self.step_count.item()
                    self.mse_mean = (1 - alpha) * self.mse_mean + alpha * mse_val
                    self.p_loss_mean = (1 - alpha) * self.p_loss_mean + alpha * p_val
                    self.hf_loss_mean = (1 - alpha) * self.hf_loss_mean + alpha * hf_val
                else:
                    # 正常阶段：使用 momentum
                    self.mse_mean = self.momentum * self.mse_mean + (1 - self.momentum) * mse_val
                    self.p_loss_mean = self.momentum * self.p_loss_mean + (1 - self.momentum) * p_val
                    self.hf_loss_mean = self.momentum * self.hf_loss_mean + (1 - self.momentum) * hf_val
                
                # 更新标准差（使用绝对偏差的滑动平均）
                mse_diff = abs(mse_val - self.mse_mean.item())
                p_diff = abs(p_val - self.p_loss_mean.item())
                hf_diff = abs(hf_val - self.hf_loss_mean.item())
                
                if self.step_count <= warmup_steps:
                    # 使用相同的 alpha 计算策略
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
            
            # 在 warmup 阶段使用基础权重
            if self.step_count <= warmup_steps:
                mse_weight = self.base_mse_weight
                p_weight = self.base_perceptual_weight
                hf_weight = self.base_high_freq_weight
                
                # 存储当前权重用于日志
                self.current_mse_weight = mse_weight
                self.current_p_weight = p_weight
                self.current_hf_weight = hf_weight
            else:
                # 根据当前 loss 相对于历史均值的偏差动态调整权重
                # 使用 Z-score 归一化：如果 loss 偏离均值超过 1 个标准差，调整权重
                mse_z = (mse_val - self.mse_mean.item()) / (self.mse_std.item() + 1e-6)
                p_z = (p_val - self.p_loss_mean.item()) / (self.p_loss_std.item() + 1e-6)
                hf_z = (hf_val - self.hf_loss_mean.item()) / (self.hf_loss_std.item() + 1e-6)
                
                # 权重调整策略：对称调整，loss 异常大时降低权重，异常小时增加权重
                # 使用平滑的调整函数，避免突变
                def adjust_weight(base_weight, z_score, scale=0.15):
                    # scale 控制调整幅度，越小调整越平滑
                    # 对称调整：Z > 0 时降低权重，Z < 0 时增加权重
                    # 使用 tanh 风格的调整，确保平滑过渡
                    if z_score > 0:
                        # loss 异常大，降低权重
                        adjustment = 1.0 / (1.0 + scale * z_score)
                    else:
                        # loss 异常小，增加权重（但幅度较小，避免过度）
                        adjustment = 1.0 + scale * abs(z_score) * 0.5
                    return base_weight * adjustment
                
                mse_weight = adjust_weight(self.base_mse_weight, mse_z, scale=0.15)
                p_weight = adjust_weight(self.base_perceptual_weight, p_z, scale=0.15)
                hf_weight = adjust_weight(self.base_high_freq_weight, hf_z, scale=0.15)
                
                # 确保权重在合理范围内（0.3x - 2.0x 基础权重）
                # 放宽上限，允许在 loss 异常小时适当增加权重
                mse_weight = max(self.base_mse_weight * 0.3, min(mse_weight, self.base_mse_weight * 2.0))
                p_weight = max(self.base_perceptual_weight * 0.3, min(p_weight, self.base_perceptual_weight * 2.0))
                hf_weight = max(self.base_high_freq_weight * 0.3, min(hf_weight, self.base_high_freq_weight * 2.0))
                
                # 存储当前权重用于日志
                self.current_mse_weight = mse_weight
                self.current_p_weight = p_weight
                self.current_hf_weight = hf_weight
        else:
            # 不使用自适应权重，使用基础权重
            mse_weight = self.base_mse_weight
            p_weight = self.base_perceptual_weight
            hf_weight = self.base_high_freq_weight
            
            # 存储当前权重用于日志
            self.current_mse_weight = mse_weight
            self.current_p_weight = p_weight
            self.current_hf_weight = hf_weight
        
        loss = (p_weight * p_loss + 
                mse_weight * mse_loss + 
                hf_weight * hf_loss)
        
        return loss, mse_loss, p_loss, hf_loss
 
 
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y, crop_x = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
 
def denorm(t):
    return (t.detach().cpu() + 1.0) / 2.0
 
def save_image_local(canvas, log_dir, global_step, prefix="train"):
    save_path = os.path.join(log_dir, "visual_results", prefix)
    os.makedirs(save_path, exist_ok=True)
    vutils.save_image(canvas, os.path.join(save_path, f"step_{global_step:07d}.png"))
 
 
class LatentDataset(Dataset):
    def __init__(self, pt_dir, pixel_transform=None, logger=None):
        self.pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
        self.pixel_transform = pixel_transform or transforms.Compose([
            transforms.Lambda(lambda pil: center_crop_arr(pil, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 缓存文件路径
        cache_file = os.path.join(pt_dir, ".path_info_cache.pkl")
        
        rain_prefixes = ["/openbayes/input/input0/Rain1400", "/openbayes/input/input0/Outdoor-Rain"]
        moire_prefix = "/openbayes/input/input0/AIO/train/9"
        
        # 检查缓存是否存在
        if os.path.exists(cache_file):
            if logger: logger.info(f"Loading cached path info from {cache_file}...")
            try:
                with open(cache_file, 'rb') as f:
                    self.path_info = pickle.load(f)
                if logger: logger.info(f"Loaded {len(self.path_info)} entries from cache")
                # 只验证文件数量是否匹配
                if len(self.path_info) != len(self.pt_files):
                    if logger: logger.warning(f"Cache size mismatch ({len(self.path_info)} vs {len(self.pt_files)}), rebuilding...")
                    self.path_info = None
            except Exception as e:
                if logger: logger.warning(f"Error loading cache: {e}, rebuilding...")
                self.path_info = None
        else:
            self.path_info = None
        
        # 如果缓存无效或不存在，重新构建索引
        if self.path_info is None:
            if logger: logger.info(f"Scanning {len(self.pt_files)} files to pre-index categories...")
            
            # 单进程加载
            self.path_info = []
            for f in tqdm(self.pt_files, desc="Indexing", disable=(logger is None)):
                try:
                    data = torch.load(f, map_location='cpu')
                    in_path = data.get('in_path', "")
                    is_hard = any(p in in_path for p in rain_prefixes) or (moire_prefix in in_path)
                    self.path_info.append((f, is_hard))
                    del data  # 立即释放内存
                except Exception as e:
                    if logger: logger.warning(f"Error loading {f}: {e}")
                    self.path_info.append((f, False))
            
            # 保存缓存
            if logger: logger.info(f"Saving path info cache to {cache_file}...")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.path_info, f)
                if logger: logger.info("Cache saved successfully")
            except Exception as e:
                if logger: logger.warning(f"Error saving cache: {e}")
 
    def __len__(self): return len(self.pt_files)
 
    def __getitem__(self, idx):
        pt_path, _ = self.path_info[idx]
        data = torch.load(pt_path, map_location='cpu')
        z_refined = data['z_refined'].float()
        
        img_gt_pil = center_crop_arr(Image.open(data['gt_path']).convert('RGB'), 512)
        img_in_pil = center_crop_arr(Image.open(data['in_path']).convert('RGB'), 512)
        img_gt_np = np.array(img_gt_pil)
        img_in_np = np.array(img_in_pil)
        img_in_matched_np = histogram_match_np(img_in_np, img_gt_np)
        img_in_matched_pil = Image.fromarray(img_in_matched_np)
        final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        pixel_gt = final_transform(img_gt_pil)
        pixel_in = final_transform(img_in_matched_pil)
        #pixel_gt = self.pixel_transform(Image.open(data['gt_path']).convert('RGB'))
        #pixel_in = self.pixel_transform(Image.open(data['in_path']).convert('RGB'))
        return z_refined, pixel_gt, pixel_in
 
@torch.no_grad()
def run_validation(vae, val_loader, loss_module, device, accelerator, writer, global_step, logger, log_dir):
    vae.model.eval()
    val_mses, val_lpips = [], []
    model_internal = vae.module.model if hasattr(vae, 'module') else vae.model
    for i, (vz, v_gt, v_in) in enumerate(val_loader):
        if i > 20: break
        vz, v_gt, v_in = vz.to(device), v_gt.to(device), v_in.to(device)
        _, v_feats = model_internal.encode(v_in)
        v_rec_fine = model_internal.decode(vz, feats=v_feats)
        v_rec_orig = model_internal.decode(vz, feats=None)
        _, vmse, vlpips, _ = loss_module(v_gt, v_rec_fine) # 增加解包
        val_mses.append(vmse.item())
        val_lpips.append(vlpips.item())
        if i == 0 and accelerator.is_main_process:
            v_diff = torch.clamp((v_rec_fine - v_rec_orig)[0].cpu() * 5.0, 0, 1)
            canvas = torch.cat([denorm(v_in[0]), denorm(v_gt[0]), denorm(v_rec_orig[0]), denorm(v_rec_fine[0]), v_diff], dim=2)
            writer.add_image(f"Visual_Val/Step_{global_step}", canvas, global_step, dataformats='CHW')
            save_image_local(canvas, log_dir, global_step, prefix="val")
    if accelerator.is_main_process:
        avg_mse, avg_lpips = np.mean(val_mses), np.mean(val_lpips)
        logger.info(f"[Val] Step {global_step} | MSE: {avg_mse:.5f} | LPIPS: {avg_lpips:.5f}")
        writer.add_scalar("Val/MSE", avg_mse, global_step)
        writer.add_scalar("Val/LPIPS", avg_lpips, global_step)
    vae.model.train()
 
 
def train_fusion_layer(config_path, precomputed_dir):
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    
    log_dir = f"/openbayes/input/input0/logs/decoder/fusion_train_{datetime.now().strftime('%m%d_%H%M')}"
    ckpt_dir, tb_dir = os.path.join(log_dir, "checkpoints"), os.path.join(log_dir, "tensorboard")
    
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(tb_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(f"{log_dir}/log.txt"), logging.StreamHandler()])
        logger = logging.getLogger(__name__)
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        logger = None
 
    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)
 
    vae = VA_VAE(f'tokenizer/configs/{train_config["vae"]["model_name"]}.yaml')
    vae.model.requires_grad_(False)
    for name, param in vae.model.decoder.named_parameters():
        if 'fusion_layers' in name:
            param.requires_grad = True
            torch.nn.init.constant_(param, 1e-6) if 'conv_out' not in name else torch.nn.init.zeros_(param)
 
    full_dataset = LatentDataset(pt_dir=precomputed_dir, logger=logger)
    train_size = int(0.98 * len(full_dataset))
    
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:train_size], indices[train_size:]
 
    hard_indices = [i for i in train_idx if full_dataset.path_info[i][1]]
    num_hard = len(hard_indices)
    num_total = len(train_idx)
    
    if logger: logger.info(f"Train set: {num_total} samples. Hard samples (Rain/Moire): {num_hard}.")
    
    extra_repeats = 1
    total_samples_per_epoch = num_total + (num_hard * extra_repeats)
 
    weights = []
    for i in train_idx:
        if full_dataset.path_info[i][1]:
            weights.append(float(extra_repeats + 1))
        else:
            weights.append(1.0)
            
    sampler = WeightedRandomSampler(weights, num_samples=total_samples_per_epoch, replacement=True)
    
    if logger: logger.info(f"Oversampling configured: Each epoch will run {total_samples_per_epoch} steps.")
 
    loader = DataLoader(Subset(full_dataset, train_idx), batch_size=2, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=2, shuffle=True)
 
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, vae.model.decoder.parameters()), lr=1e-4)
    loss_module = FusionLoss(
        mse_weight=1.0, 
        perceptual_weight=0.5, 
        high_freq_weight=0,  # 提高高频损失权重，更关注雨痕去除
        device=device,
        adaptive_weight=True,  # 启用自适应权重
        momentum=0.99  # 滑动平均的 momentum
    )
 
    # 计算可学习参数总量
    trainable_params = sum(p.numel() for p in vae.model.decoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in vae.model.decoder.parameters())
    if logger:
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        logger.info(f"Total decoder parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
        logger.info(f"Trainable ratio: {trainable_params / total_params * 100:.2f}%")
 
    vae, optimizer, loader, val_loader = accelerator.prepare(vae, optimizer, loader, val_loader)
 
    global_step = 0
    for epoch in range(10): 
        vae.model.train()
        
        pbar = tqdm(
            loader, 
            disable=not accelerator.is_main_process, 
            dynamic_ncols=True, 
            desc=f"Epoch {epoch}", 
            total=total_samples_per_epoch // 2 
        )
        
        for z_refined, pixel_gt, pixel_in in pbar:
            z_refined, pixel_gt, pixel_in = z_refined.to(device), pixel_gt.to(device), pixel_in.to(device)
 
            with torch.no_grad():
                model_internal = vae.module.model if hasattr(vae, 'module') else vae.model
                _, feats = model_internal.encode(pixel_in)
 
            rec = model_internal.decode(z_refined, feats=feats)
            total_loss, mse, lpips_val, hf_loss = loss_module(pixel_gt, rec)
 
            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()
 
            if accelerator.is_main_process:
                if global_step % 100 == 0:
                    # 显示当前使用的权重（如果使用自适应权重）
                    if hasattr(loss_module, 'adaptive_weight') and loss_module.adaptive_weight:
                        pbar.set_postfix({
                            'step': global_step, 
                            'loss': f"{total_loss.item():.4f}", 
                            'hf': f"{hf_loss.item():.4f}", 
                            'p': f"{lpips_val.item():.4f}",
                            'w_mse': f"{loss_module.current_mse_weight:.3f}",
                            'w_p': f"{loss_module.current_p_weight:.3f}",
                            'w_hf': f"{loss_module.current_hf_weight:.3f}"
                        })
                        writer.add_scalar("Loss/total", total_loss.item(), global_step)
                        writer.add_scalar("Loss/mse", mse.item(), global_step)
                        writer.add_scalar("Loss/high_freq", hf_loss.item(), global_step)
                        writer.add_scalar("Weights/mse", loss_module.current_mse_weight, global_step)
                        writer.add_scalar("Weights/perceptual", loss_module.current_p_weight, global_step)
                        writer.add_scalar("Weights/high_freq", loss_module.current_hf_weight, global_step)
                    else:
                        pbar.set_postfix({'step': global_step, 'loss': f"{total_loss.item():.4f}", 'hf': f"{hf_loss.item():.4f}", 'p': f"{lpips_val.item():.4f}"})
                        writer.add_scalar("Loss/total", total_loss.item(), global_step)
                        writer.add_scalar("Loss/mse", mse.item(), global_step)
                        writer.add_scalar("Loss/high_freq", hf_loss.item(), global_step)
                    
                if global_step % 1000 == 0:
                    with torch.no_grad():
                        rec_orig = model_internal.decode(z_refined, feats=None)
                        diff = torch.clamp((rec - rec_orig) * 5.0, 0.0, 1.0)
                        canvas = torch.cat([denorm(pixel_in[0]), denorm(pixel_gt[0]), denorm(rec_orig[0]), denorm(rec[0]), diff[0].cpu()], dim=2)
                        save_image_local(canvas, log_dir, global_step, prefix="train")
 
                if global_step > 0 and global_step % 2000 == 0:
                    run_validation(vae, val_loader, loss_module, device, accelerator, writer, global_step, logger, log_dir)
                    fusion_state = {k: v for k, v in vae.model.state_dict().items() if 'fusion_layers' in k}
                    torch.save(fusion_state, os.path.join(ckpt_dir, f"step_{global_step}.pt"))
 
            global_step += 1
            del rec, total_loss, mse, lpips_val, hf_loss, feats
            if global_step % 100 == 0: gc.collect(); torch.cuda.empty_cache()
 
if __name__ == "__main__":
    train_fusion_layer(
        config_path='./configs/lightningdit_xl_vavae_f16d32.yaml',
        precomputed_dir='/openbayes/input/input0/precomputed_latents1'
    )