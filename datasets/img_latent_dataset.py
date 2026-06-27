import os
import numpy as np
import json
from glob import glob
from tqdm import tqdm
from PIL import Image
 
import torch
from torch.utils.data import Dataset
from safetensors import safe_open
from torchvision import transforms
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.LANCZOS
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.LANCZOS
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]) 
 
def resize_and_crop(image_pil, max_size=720, base=16):
    """
    Resize image so long_edge <= max_size, keeping aspect ratio.
    Then snap both dimensions DOWN to the nearest multiple of base.
    This ensures the pixel tensor can be evenly divided by base (VAE downsample ratio).
    """
    w, h = image_pil.size
    long_edge = max(w, h)

    # Step 1: scale down if long_edge exceeds max_size
    if long_edge > max_size:
        scale = max_size / long_edge
        w = int(w * scale)
        h = int(h * scale)
        image_pil = image_pil.resize((w, h), Image.LANCZOS)

    # Step 2: snap to multiple of base (round down)
    new_w = (w // base) * base
    new_h = (h // base) * base
    # Ensure at least one base-unit in each dimension
    new_w = max(new_w, base)
    new_h = max(new_h, base)
    if new_w != w or new_h != h:
        image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)

    return image_pil
 
def histogram_match_np(src_img_np, ref_img_np):
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
        # Use numpy indexing instead of cv2.LUT (equivalent operation)
        imgOut[:,:,i] = lut[img[:,:,i]]
    return imgOut
 
 
class ImgLatentDataset(Dataset):
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0, is_train=True):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        self.is_train = is_train
 
        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats()
 
        self.pixel_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
 
    @staticmethod
    def _detect_format(safe_file):
        """Detect safetensors key format.

        Returns:
            'batched'    – tensors are stacked: latents_gt (N,C,H,W)
            'per_sample' – tensors are individual: gt_00000 (C,H,W)
        """
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
        if 'latents_gt' in keys:
            return 'batched'
        return 'per_sample'

    def get_img_to_safefile_map(self):
        img_to_file = {}
        if not self.files:
            print(f"Error: No .safetensors found in {self.data_dir}")
            return img_to_file

        for safe_file in self.files:
            try:
                json_path = safe_file.replace(".safetensors", "_paths.json")
                with open(json_path, 'r', encoding='utf-8') as jf:
                    paths_data = json.load(jf)

                fmt = self._detect_format(safe_file)

                if fmt == 'batched':
                    # Keys: latents_gt (N,C,H,W), latents_input (N,C,H,W), label (N,)
                    num_imgs_in_shard = len(paths_data['paths_gt'])
                else:
                    # Keys: gt_00000 (C,H,W), in_00000 (C,H,W), label (N,)
                    num_imgs_in_shard = len(paths_data['paths_gt'])

                cur_len = len(img_to_file)
                for i in range(num_imgs_in_shard):
                    img_to_file[cur_len + i] = {
                        'safe_file': safe_file,
                        'sample_idx': i,
                        'fmt': fmt,
                        'gt_path': paths_data['paths_gt'][i],
                        'input_path': paths_data['paths_input'][i]
                    }
            except Exception as e:
                print(f"Failed to parse file {safe_file}: {e}")
        return img_to_file
 
    def get_latent_stats(self):
        latent_stats_cache_file = os.path.join(self.data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            latent_stats = self.compute_latent_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            latent_stats = torch.load(latent_stats_cache_file, map_location='cpu')
        return latent_stats['mean'], latent_stats['std']
    
    def compute_latent_stats(self):
        num_samples = min(10000, len(self.img_to_file_map))
        random_indices = np.random.choice(len(self.img_to_file_map), num_samples, replace=False)
        sum_x, sum_x2, total_pixels = None, None, 0
        
        for idx in tqdm(random_indices, desc="Computing Stats"):
            info = self.img_to_file_map[idx]
            s_idx = info['sample_idx']
            with safe_open(info['safe_file'], framework="pt", device="cpu") as f:
                if info['fmt'] == 'batched':
                    feat = f.get_tensor("latents_gt")[s_idx].float().unsqueeze(0)
                else:
                    feat = f.get_tensor(f"gt_{s_idx:05d}").float().unsqueeze(0)
            
            total_pixels += feat.shape[2] * feat.shape[3]
            s = feat.sum(dim=[2, 3], keepdim=True)
            s2 = (feat**2).sum(dim=[2, 3], keepdim=True)
            
            if sum_x is None:
                sum_x, sum_x2 = s, s2
            else:
                sum_x += s; sum_x2 += s2
        
        mean = sum_x / total_pixels
        std = torch.sqrt(((sum_x2 / total_pixels) - (mean ** 2)).clamp(min=1e-6))
        return {'mean': mean, 'std': std}
 
    def __len__(self):
        return len(self.img_to_file_map)
 
    def __getitem__(self, idx):
        try:
            return self._getitem_impl(idx)
        except Exception as e:
            # Re-raise as a plain RuntimeError so multiprocessing workers can
            # pickle and propagate it (safetensors_rust.SafetensorError is not
            # picklable across fork'd workers).
            raise RuntimeError(f"[ImgLatentDataset] failed to load idx={idx}: {e}") from None

    def _getitem_impl(self, idx):
        info = self.img_to_file_map[idx]
        s_idx = info['sample_idx']
        
        do_flip = self.is_train and (np.random.uniform(0, 1) > 0.5)
        fmt = info['fmt']

        with safe_open(info['safe_file'], framework="pt", device="cpu") as f:
            if fmt == 'batched':
                # Keys: latents_gt (N,C,H,W) / latents_gt_flip (N,C,H,W)
                key_gt = "latents_gt_flip" if do_flip else "latents_gt"
                key_in = "latents_input_flip" if do_flip else "latents_input"
                # Clone tensors to ensure they have resizable storage for DataLoader collation
                latent_gt = f.get_tensor(key_gt)[s_idx].clone()
                latent_in = f.get_tensor(key_in)[s_idx].clone()
                label = f.get_tensor('label')[s_idx].item()
            else:
                # Keys: gt_00000 (C,H,W) / gt_f_00000 (C,H,W)
                key_gt = f"gt_f_{s_idx:05d}" if do_flip else f"gt_{s_idx:05d}"
                key_in = f"in_f_{s_idx:05d}" if do_flip else f"in_{s_idx:05d}"
                # Clone tensors to ensure they have resizable storage for DataLoader collation
                latent_gt = f.get_tensor(key_gt).clone()
                latent_in = f.get_tensor(key_in).clone()
                label = int(f.get_tensor('label')[s_idx].item())
 
        if self.latent_norm:
            mean, std = self._latent_mean.squeeze(0), self._latent_std.squeeze(0)
            latent_gt = (latent_gt - mean) / std
            latent_in = (latent_in - mean) / std
        
        latent_gt = latent_gt * self.latent_multiplier
        latent_in = latent_in * self.latent_multiplier
 
        raw_in = Image.open(info['input_path']).convert("RGB")
        raw_gt = Image.open(info['gt_path']).convert("RGB")
       
        raw_in = resize_and_crop(raw_in, max_size=720, base=16)
        raw_gt = resize_and_crop(raw_gt, max_size=720, base=16)
 
        if label in [3, 10]:
            in_np = np.array(raw_in)[:, :, ::-1] # RGB -> BGR for cv2
            gt_np = np.array(raw_gt)[:, :, ::-1]
            matched_in_np = histogram_match_np(in_np, gt_np)
            raw_in = Image.fromarray(matched_in_np[:, :, ::-1], 'RGB')
 
        if do_flip:
            raw_in = raw_in.transpose(Image.FLIP_LEFT_RIGHT)
            raw_gt = raw_gt.transpose(Image.FLIP_LEFT_RIGHT)
 
        pixel_in = self.pixel_transform(raw_in)
        pixel_gt = self.pixel_transform(raw_gt)
 
        return latent_gt, latent_in, pixel_gt, pixel_in, torch.tensor(label, dtype=torch.long)


class ImgLatentDataset1(Dataset):
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0, is_train=True):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier

        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats()
        self.is_train = is_train
        self.transform_orig = self.make_deterministic_transform(p_hflip=0.0)
        self.transform_flip = self.make_deterministic_transform(p_hflip=1.0)

    def get_img_to_safefile_map(self):
        img_to_file = {}
        for safe_file in self.files:
            try:
                json_path = safe_file.replace(".safetensors", "_paths.json")
                with open(json_path, 'r', encoding='utf-8') as jf:
                    paths_data = json.load(jf)
                
                with safe_open(safe_file, framework="pt", device="cpu") as f:
                    keys = f.keys()
                    if 'latents_gt' not in keys:
                        continue
                    
                    target_tensor = f.get_slice('latents_gt')
                    num_imgs = target_tensor.get_shape()[0]
                    
                    cur_len = len(img_to_file)
                    for i in range(num_imgs):
                        img_to_file[cur_len+i] = {
                            'safe_file': safe_file,
                            'idx_in_file': i,
                            'gt_path': paths_data['paths_gt'][i],
                            'input_path': paths_data['paths_input'][i]
                        }
            except Exception as e:
                print(f"Error reading {safe_file} or its json: {e}")
        return img_to_file

    def get_latent_stats(self):
        latent_stats_cache_file = os.path.join(self.data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            print("Calculating latent stats from GT...")
            latent_stats = self.compute_latent_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']
    
    def compute_latent_stats(self):
        num_samples = min(10000, len(self.img_to_file_map))
        random_indices = np.random.choice(len(self.img_to_file_map), num_samples, replace=False)
        latents = []
        
        for idx in tqdm(random_indices, desc="Computing Stats"):
            img_info = self.img_to_file_map[idx]
            safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                features = f.get_slice('latents_gt')
                feature = features[img_idx:img_idx+1]
                latents.append(feature)
        
        latents = torch.cat(latents, dim=0)
        mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        std = latents.std(dim=[0, 2, 3], keepdim=True)
        latent_stats = {'mean': mean, 'std': std}
        return latent_stats

    def __len__(self):
        return len(self.img_to_file_map.keys())
    
    def make_deterministic_transform(self, p_hflip):
        return transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 512)),
            transforms.RandomHorizontalFlip(p=p_hflip), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        gt_path, input_path = img_info['gt_path'], img_info['input_path']
        do_flip = self.is_train and (np.random.uniform(0, 1) > 0.5)
        
        if do_flip:
            gt_key, in_key = 'latents_gt_flip', 'latents_input_flip'
            current_transform = self.transform_flip
        else:
            gt_key, in_key = 'latents_gt', 'latents_input'
            current_transform = self.transform_orig
        
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            gt_slice = f.get_slice(gt_key)
            latent_gt = gt_slice[img_idx:img_idx+1]
            in_slice = f.get_slice(in_key)
            latent_in = in_slice[img_idx:img_idx+1]
            label_slice = f.get_slice('label')
            label = label_slice[img_idx:img_idx+1]

        # Clone tensors to ensure they have resizable storage for DataLoader collation
        latent_gt = latent_gt.squeeze(0).clone()
        latent_in = latent_in.squeeze(0).clone()
        label = label.item()

        if self.latent_norm:
            latent_gt = (latent_gt - self._latent_mean) / self._latent_std
            latent_in = (latent_in - self._latent_mean) / self._latent_std
        
        latent_gt = latent_gt * self.latent_multiplier
        latent_in = latent_in * self.latent_multiplier
        
        pixel_gt = current_transform(Image.open(gt_path).convert("RGB"))
        pixel_in = current_transform(Image.open(input_path).convert("RGB"))
        
        return latent_gt, latent_in, pixel_gt, pixel_in, torch.tensor(label, dtype=torch.long)