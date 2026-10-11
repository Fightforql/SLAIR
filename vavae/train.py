import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback # 引入 Callback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vavae.ldm.util import instantiate_from_config
from vavae.ldm.data.cls_dataset import PairedTrainDataset, PairedTestDataset


class ImageLogger(Callback):
    def __init__(self, batch_frequency=5000, max_images=4, clamp=True):
        super().__init__()
        self.batch_frequency = batch_frequency
        self.max_images = max_images
        self.clamp = clamp

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        root = pl_module.logger.save_dir if pl_module.logger else "logs"
        filename = "{}_step-{:06}_batch-{:06}.png".format(split, pl_module.global_step, batch_idx)
        path = os.path.join(root, "images", filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)

        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, split=split)

        # 数据后处理
        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)

        if is_train:
            pl_module.train()

        def norm_grid(x):
            return (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        grids = []
        if "inputs_lq" in images: grids.append(norm_grid(images["inputs_lq"]))
        if "lq_reconstructions" in images: grids.append(norm_grid(images["lq_reconstructions"]))
        if "inputs_hq" in images: grids.append(norm_grid(images["inputs_hq"]))
        if "hq_reconstructions" in images: grids.append(norm_grid(images["hq_reconstructions"]))

        # 保存拼接图
        if len(grids) > 0:
            grid = torch.cat(grids, dim=0) 
            torchvision.utils.save_image(grid, path, nrow=self.max_images)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.batch_frequency > 0 and batch_idx % self.batch_frequency == 0:
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./configs/train.yaml", help="Path to config yaml")
    parser.add_argument("-s", "--seed", type=int, default=42, help="seed for seed_everything")
    parser.add_argument("-l", "--logdir", type=str, default="/openbayes/input/input0/logs/vae10", help="directory for logging")
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
        monitor="val/psnr_lq",
        mode="max",
        save_top_k=2,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    image_logger = ImageLogger(batch_frequency=1000, max_images=4)

    trainer_config = config.lightning.trainer
    trainer = Trainer(
        devices=trainer_config.devices,
        max_epochs=trainer_config.max_epochs,
        accelerator=trainer_config.accelerator,
        precision=trainer_config.precision,
        strategy=trainer_config.get("strategy", "auto"),
        callbacks=[checkpoint_callback, lr_monitor, image_logger], 
        default_root_dir=logdir,
        log_every_n_steps=10
    )

    print("Starting training...")
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()