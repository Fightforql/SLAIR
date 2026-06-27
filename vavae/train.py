import argparse
import datetime
import os
import sys

import torch
import torchvision
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vavae.ldm.util import instantiate_from_config


class GradientMonitor(Callback):
    """Log gradient statistics for selected model parameters."""

    def __init__(
        self,
        log_frequency=100,
        param_names=None,
        log_norm=True,
        log_max=True,
        log_mean=True,
    ):
        """
        Args:
            log_frequency: Number of batches between gradient logging events.
            param_names: Optional parameter names to monitor. If None, monitor all parameters.
            log_norm: Whether to log the L2 norm of each gradient.
            log_max: Whether to log the maximum gradient value.
            log_mean: Whether to log the mean gradient value.
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.param_names = param_names
        self.log_norm = log_norm
        self.log_max = log_max
        self.log_mean = log_mean

    def get_param_grad(self, pl_module, param_name):
        """Return the gradient tensor for a named parameter, or None if unavailable."""
        for name, param in pl_module.named_parameters():
            if name == param_name:
                return param.grad
        return None

    def on_after_backward(self, trainer, pl_module):
        """Run after backpropagation, when gradients are available."""
        if trainer.global_step % self.log_frequency != 0:
            return

        if self.param_names is not None:
            for param_name in self.param_names:
                grad = self.get_param_grad(pl_module, param_name)
                if grad is not None:
                    self._log_gradient(trainer, pl_module, param_name, grad)
                else:
                    print(f"Warning: Parameter '{param_name}' not found or has no gradient")
        else:
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    self._log_gradient(trainer, pl_module, name, param.grad)

    def _log_gradient(self, trainer, pl_module, param_name, grad):
        """Log gradient statistics to the Lightning logger."""
        log_name = param_name.replace(".", "/")

        if self.log_norm:
            grad_norm = grad.norm().item()
            pl_module.log(f"grad/{log_name}/norm", grad_norm, on_step=True, on_epoch=False, logger=True)

        if self.log_max:
            grad_max = grad.max().item()
            pl_module.log(f"grad/{log_name}/max", grad_max, on_step=True, on_epoch=False, logger=True)

        if self.log_mean:
            grad_mean = grad.mean().item()
            pl_module.log(f"grad/{log_name}/mean", grad_mean, on_step=True, on_epoch=False, logger=True)

        if trainer.global_step % (self.log_frequency * 10) == 0:
            print(f"Step {trainer.global_step}, Param: {param_name}")
            if self.log_norm:
                print(f"  Gradient Norm: {grad.norm().item():.6f}")
            if self.log_max:
                print(f"  Gradient Max: {grad.max().item():.6f}")
            if self.log_mean:
                print(f"  Gradient Mean: {grad.mean().item():.6f}")


class ImageLogger(Callback):
    def __init__(self, batch_frequency=5000, max_images=4, clamp=True, config=None):
        super().__init__()
        self.batch_frequency = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.config = config
        self.original_model = None

    @torch.no_grad()
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if self.original_model is None and self.config is not None:
            print(f"--- [ImageLogger] Loading Original VAE from {self.config.model.params.ckpt_path} for comparison ---")
            self.original_model = instantiate_from_config(self.config.model)
            sd = torch.load(self.config.model.params.ckpt_path, map_location="cpu")["state_dict"]
            self.original_model.load_state_dict(sd, strict=False)
            self.original_model.eval()
            self.original_model.to(pl_module.device)

        if self.original_model.device != pl_module.device:
            self.original_model.to(pl_module.device)

        root = pl_module.logger.save_dir if pl_module.logger else "logs"
        base_dir = os.path.join(root, "visualizations", f"step_{pl_module.global_step:06}")
        os.makedirs(base_dir, exist_ok=True)

        is_train = pl_module.training
        pl_module.eval()

        images, _ = pl_module.get_input(batch)
        images = images.to(pl_module.device)
        rec_fine_tuned, _, _ = pl_module(images, sample_posterior=False)

        post_orig, feats_orig = self.original_model.encode(images)
        rec_original = self.original_model.decode(post_orig.mode(), feats=feats_orig)
        diff = torch.abs(rec_fine_tuned - rec_original)
        diff_enhanced = torch.clamp(diff * 5.0, 0.0, 1.0)

        def postprocess(x):
            n = min(x.shape[0], self.max_images)
            x = x[:n].detach().cpu()
            if self.clamp:
                x = torch.clamp(x, -1.0, 1.0)
            return (x + 1.0) / 2.0

        img_input = postprocess(images)
        img_orig = postprocess(rec_original)
        img_fine = postprocess(rec_fine_tuned)
        img_diff = diff_enhanced[: min(diff_enhanced.shape[0], self.max_images)].detach().cpu()

        grid_comp = torch.cat([img_orig, img_fine], dim=0)
        torchvision.utils.save_image(
            grid_comp,
            os.path.join(base_dir, f"{split}_comparison_orig_vs_fine.png"),
            nrow=self.max_images,
        )

        torchvision.utils.save_image(
            img_input,
            os.path.join(base_dir, f"{split}_input_view.png"),
            nrow=self.max_images,
        )

        torchvision.utils.save_image(
            img_diff,
            os.path.join(base_dir, f"{split}_diff_map_x5.png"),
            nrow=self.max_images,
        )

        if is_train:
            pl_module.train()
        self.original_model.to("cpu")
        torch.cuda.empty_cache()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.batch_frequency > 0 and trainer.global_step % self.batch_frequency == 0:
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self.log_img(pl_module, batch, batch_idx, split="val")


class WrappedDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.params.batch_size

    def setup(self, stage=None):
        self.train_dataset = instantiate_from_config(self.cfg.params.train)
        self.val_dataset = instantiate_from_config(self.cfg.params.validation)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./configs/train.yaml", help="Path to config yaml")
    parser.add_argument("-s", "--seed", type=int, default=42, help="seed for seed_everything")
    parser.add_argument("-l", "--logdir", type=str, default="/openbayes/input/input0/logs/vae", help="directory for logging")
    opt = parser.parse_args()

    pl.seed_everything(opt.seed)

    config = OmegaConf.load(opt.config)

    print(f"Loading model from config: {config.model.target}")
    model = instantiate_from_config(config.model)

    model.learning_rate = config.model.base_learning_rate
    print(f"Setting learning rate to {model.learning_rate:.2e}")

    print("Loading data...")
    datamodule = WrappedDataModule(config.data)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if opt.logdir != "logs":
        logdir = os.path.join(opt.logdir, now)
    else:
        logdir = os.path.join("logs", now)

    os.makedirs(logdir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logdir, "checkpoints"),
        filename="epoch={epoch:02d}-psnr={val/psnr:.2f}",
        monitor="val/psnr",
        mode="max",
        save_top_k=20,
        every_n_train_steps=500,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    image_logger = ImageLogger(batch_frequency=1000, max_images=4, config=config)
    trainer_config = config.lightning.trainer
    trainer = Trainer(
        devices=trainer_config.devices,
        max_epochs=trainer_config.max_epochs,
        accelerator=trainer_config.accelerator,
        precision=trainer_config.precision,
        strategy=trainer_config.get("strategy", "auto"),
        callbacks=[checkpoint_callback, lr_monitor, image_logger],
        default_root_dir=logdir,
        log_every_n_steps=10,
        val_check_interval=5000,
    )

    print("Starting training...")
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
