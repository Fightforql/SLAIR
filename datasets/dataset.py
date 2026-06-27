import os
import pandas as pd
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize


class ImgPatchDataset(Dataset):
    def __init__(self, csv_path, patch_size=256, is_train=True):
        self.data_info = pd.read_csv(csv_path)
        self.patch_size = patch_size
        self.is_train = is_train
        self.pixel_transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @staticmethod
    def histogram_match_np(src_img_np, ref_img_np):
        """Match source brightness/color statistics to a reference image."""
        img = src_img_np.copy()
        img_ref = ref_img_np.copy()
        _, _, channel = img.shape
        img_out = np.zeros_like(img)

        for i in range(channel):
            hist_img, _ = np.histogram(img[:, :, i].flatten(), 256, [0, 256])
            hist_ref, _ = np.histogram(img_ref[:, :, i].flatten(), 256, [0, 256])

            cdf_img = np.cumsum(hist_img)
            cdf_ref = np.cumsum(hist_ref)

            total_pixels = img[:, :, i].size
            cdf_img_norm = cdf_img / total_pixels
            cdf_ref_norm = cdf_ref / total_pixels
            lut = np.zeros(256, dtype=np.uint8)
            for r in range(256):
                idx = np.argmin(np.abs(cdf_img_norm[r] - cdf_ref_norm))
                lut[r] = idx
            img_out[:, :, i] = lut[img[:, :, i]]

        return img_out

    def _sync_random_crop(self, pil_in, pil_gt, target_size):
        """Resize both images consistently, then crop aligned patches."""
        while min(*pil_in.size) >= 2 * target_size:
            new_size = tuple(x // 2 for x in pil_in.size)
            pil_in = pil_in.resize(new_size, resample=Image.BOX)
            pil_gt = pil_gt.resize(new_size, resample=Image.BOX)

        scale = target_size / min(*pil_in.size)
        scaled_size = tuple(round(x * scale) for x in pil_in.size)

        pil_in = pil_in.resize(scaled_size, resample=Image.BICUBIC)
        pil_gt = pil_gt.resize(scaled_size, resample=Image.BICUBIC)

        w, h = pil_in.size
        diff_x = w - target_size
        diff_y = h - target_size

        if self.is_train:
            left = random.randint(0, diff_x) if diff_x > 0 else 0
            top = random.randint(0, diff_y) if diff_y > 0 else 0
        else:
            left = diff_x // 2
            top = diff_y // 2

        rect = (left, top, left + target_size, top + target_size)
        patch_in = pil_in.crop(rect)
        patch_gt = pil_gt.crop(rect)

        return patch_in, patch_gt

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        line = self.data_info.iloc[idx]
        in_path = line.iloc[0]
        gt_path = line.iloc[1]
        label = int(line.iloc[2])

        try:
            raw_in = Image.open(in_path).convert("RGB")
            raw_gt = Image.open(gt_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {in_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.data_info) - 1))

        patch_in, patch_gt = self._sync_random_crop(raw_in, raw_gt, self.patch_size)

        if self.is_train:
            if random.random() > 0.5:
                patch_in = patch_in.transpose(Image.FLIP_LEFT_RIGHT)
                patch_gt = patch_gt.transpose(Image.FLIP_LEFT_RIGHT)

        if label in [3, 10]:
            np_in = np.array(patch_in)
            np_gt = np.array(patch_gt)
            matched_in = self.histogram_match_np(np_in, np_gt)
            patch_in = Image.fromarray(matched_in)

        pixel_in = self.pixel_transform(patch_in)
        pixel_gt = self.pixel_transform(patch_gt)

        return pixel_gt, pixel_in, torch.tensor(label, dtype=torch.long)
