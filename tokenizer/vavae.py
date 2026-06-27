"""
Vision Foundation Model Aligned VAE wrapper used by SLAIR.
"""

import os

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

from vavae.ldm.models.autoencoder import AutoencoderKL


class VA_VAE:
    """Vision Foundation Model Aligned VAE implementation."""

    def __init__(self, config, img_size=256, horizon_flip=0.5, fp16=True):
        self.config = OmegaConf.load(config)
        self.vae_params = self.config.model.params

        if "ddconfig" not in self.vae_params or "lossconfig" not in self.vae_params:
            raise ValueError("Config must contain 'ddconfig' and 'lossconfig' under model.params.")

        self.embed_dim = self.config.model.params.embed_dim
        self.ckpt_path = self.config.ckpt_path
        print(f"load vae from {self.ckpt_path}")
        self.img_size = img_size
        self.horizon_flip = horizon_flip
        self.load()

    def load(self):
        """Load and initialize the VAE model."""
        print(f"Loading AutoencoderKL from checkpoint: {self.ckpt_path}")
        init_kwargs = dict(self.vae_params)
        if "ckpt_path" in init_kwargs:
            del init_kwargs["ckpt_path"]

        try:
            self.model = AutoencoderKL.load_from_checkpoint(
                checkpoint_path=self.ckpt_path,
                map_location="cpu",
                strict=False,
                **init_kwargs,
            ).cuda().eval()
            print("Model loaded successfully using load_from_checkpoint.")
        except Exception as e:
            print(f"Error loading via load_from_checkpoint: {e}")
            print("Falling back to direct model instantiation.")
            self.model = AutoencoderKL(
                **init_kwargs,
                ckpt_path=self.ckpt_path,
            ).cuda().eval()

        return self

    def load_fusion_layer(self, fusion_ckpt_path):
        if not os.path.exists(fusion_ckpt_path):
            print(f"Warning: Fusion checkpoint not found at {fusion_ckpt_path}")
            return self

        print(f"Loading Fusion Layer from: {fusion_ckpt_path}")
        fusion_state = torch.load(fusion_ckpt_path, map_location="cpu")

        current_model_dict = self.model.state_dict()
        new_state_dict = {}

        for k, v in fusion_state.items():
            if k in current_model_dict:
                new_state_dict[k] = v
            elif f"decoder.{k}" in current_model_dict:
                new_state_dict[f"decoder.{k}"] = v
            elif k.startswith("decoder.") and k[8:] in current_model_dict:
                new_state_dict[k[8:]] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Fusion weights applied. Matched: {len(new_state_dict)} keys.")

        if any("gate_generator" in k for k in new_state_dict.keys()):
            print("Verified: Gated Fusion Layer keys found.")
        return self

    def img_transform(self, p_hflip=0, img_size=None):
        """Create the image preprocessing pipeline."""
        img_size = img_size if img_size is not None else self.img_size
        img_transforms = [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
        return transforms.Compose(img_transforms)

    def encode_images(self, images, return_feats=False):
        """Encode images into latent tensors."""
        with torch.no_grad():
            posterior, feats = self.model.encode(images.cuda())
            latent = posterior.sample()
            if return_feats:
                return latent, feats
            return latent

    def decode_to_images(self, z, feats=None):
        """Decode latent tensors into uint8 RGB images."""
        with torch.no_grad():
            images = self.model.decode(z.cuda(), feats=feats)
            images = (
                torch.clamp(127.5 * images + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )
        return images


def center_crop_arr(pil_image, image_size):
    """Center crop after progressive resizing, following ADM preprocessing."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


if __name__ == "__main__":
    vae = VA_VAE("tokenizer/configs/vavae_f16d32.yaml")
    vae.load()
