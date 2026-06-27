import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import os
import csv
import json
from PIL import Image
from safetensors.torch import save_file
from tokenizer.vavae import VA_VAE

try:
    from datasets.img_latent_dataset import ImgLatentDataset1
except ImportError:
    ImgLatentDataset1 = None

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
from accelerate.utils import set_seed
import cv2
import numpy as np


def histogram_match_np(src_img_np, ref_img_np):
    """Match the source image histogram to the reference image."""
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
        img_out[:, :, i] = cv2.LUT(img[:, :, i], lut)
    return img_out


class CsvPairedDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_pairs = []
        self.transform = transform
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                input_path = os.path.abspath(row[0].strip())
                gt_path = os.path.abspath(row[1].strip())
                try:
                    label = int(row[2].strip())
                except ValueError:
                    label = -1
                self.data_pairs.append((input_path, gt_path, label))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        input_path, gt_path, label = self.data_pairs[idx]
        try:
            in_img = Image.open(input_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
        except Exception as e:
            print(f"Error reading images: {input_path}. Error: {e}")
            in_img = Image.new("RGB", (512, 512))
            gt_img = Image.new("RGB", (512, 512))

        if label in [3, 10]:
            in_np = np.array(in_img)[:, :, ::-1]
            gt_np = np.array(gt_img)[:, :, ::-1]
            matched_in_np = histogram_match_np(in_np, gt_np)
            in_img = Image.fromarray(matched_in_np[:, :, ::-1], "RGB")

        if self.transform:
            in_img = self.transform(in_img)
            gt_img = self.transform(gt_img)

        return gt_img, in_img, torch.tensor(label, dtype=torch.long), gt_path, input_path


def save_shard(cache, rank, shard_idx, output_dir):
    base_name = f"latents_rank{rank:02d}_shard{shard_idx:03d}"

    save_dict = {k: torch.cat(cache[k], dim=0) for k in cache if "latents" in k or k == "label"}
    num_samples = save_dict["latents_gt"].shape[0]

    st_path = os.path.join(output_dir, f"{base_name}.safetensors")
    save_file(
        save_dict,
        st_path,
        metadata={
            "total_size": str(num_samples),
            "paths_json": f"{base_name}_paths.json",
            "dtype": str(save_dict["latents_gt"].dtype),
        },
    )

    paths_dict = {
        "paths_gt": cache["paths_gt"][:num_samples],
        "paths_input": cache["paths_input"][:num_samples],
    }

    json_path = os.path.join(output_dir, f"{base_name}_paths.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(paths_dict, f, indent=2, ensure_ascii=False)

    print(f"[Rank {rank}] Saved shard {shard_idx} with {num_samples} samples.")


def main(args):
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device
    set_seed(args.seed + rank)

    output_dir = os.path.join(args.output_path, f"{args.data_split}_{args.image_size}")
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"World Size: {world_size}, Output: {output_dir}")

    tokenizer = VA_VAE(args.config)
    tokenizer.model.to(device)
    tokenizer.model.eval()

    transforms_list = [
        tokenizer.img_transform(p_hflip=0.0, img_size=args.image_size),
        tokenizer.img_transform(p_hflip=1.0, img_size=args.image_size),
    ]

    datasets = [CsvPairedDataset(args.csv_path, transform=t) for t in transforms_list]
    samplers = [
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
        for ds in datasets
    ]
    loaders = [
        DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sp,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for ds, sp in zip(datasets, samplers)
    ]

    cache = {
        "latents_gt": [],
        "latents_input": [],
        "latents_gt_flip": [],
        "latents_input_flip": [],
        "label": [],
        "paths_gt": [],
        "paths_input": [],
    }
    saved_files = 0

    if accelerator.is_main_process:
        print(f"Total pairs in dataset: {len(datasets[0])}")
        print("Starting extraction...")

    for batch_data in zip(*loaders):
        data_orig, data_flip = batch_data

        x_gt, x_in, labels = data_orig[0].to(device), data_orig[1].to(device), data_orig[2].to(device)
        paths_gt, paths_in = data_orig[3], data_orig[4]
        x_gt_flip, x_in_flip = data_flip[0].to(device), data_flip[1].to(device)

        with torch.no_grad():
            z_gt = tokenizer.encode_images(x_gt)
            z_in = tokenizer.encode_images(x_in)
            z_gt_flip = tokenizer.encode_images(x_gt_flip)
            z_in_flip = tokenizer.encode_images(x_in_flip)

        cache["latents_gt"].append(z_gt.cpu())
        cache["latents_input"].append(z_in.cpu())
        cache["latents_gt_flip"].append(z_gt_flip.cpu())
        cache["latents_input_flip"].append(z_in_flip.cpu())
        cache["label"].append(labels.cpu())
        cache["paths_gt"].extend(paths_gt)
        cache["paths_input"].extend(paths_in)

        current_rows = sum(t.shape[0] for t in cache["latents_gt"])
        if current_rows >= 10000:
            save_shard(cache, rank, saved_files, output_dir)
            for k in cache:
                cache[k] = []
            saved_files += 1

    if len(cache["latents_gt"]) > 0:
        save_shard(cache, rank, saved_files, output_dir)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if ImgLatentDataset1 is not None:
            print("All processes finished. Computing global statistics (Mean/Std)...")
            try:
                _ = ImgLatentDataset1(output_dir, latent_norm=True)
                print(f"Statistics successfully saved in {output_dir}")
            except Exception as e:
                print(f"Warning: Failed to compute stats: {e}")
        else:
            print("Warning: ImgLatentDataset not found, skipping stats calculation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/openbayes/input/input0/5-task_train.csv")
    parser.add_argument("--output_path", type=str, default="/openbayes/input/input0/latents_with_origin_data")
    parser.add_argument("--data_split", type=str, default="5-task_train")
    parser.add_argument("--config", type=str, default="./tokenizer/configs/vavae_f16d32.yaml")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
