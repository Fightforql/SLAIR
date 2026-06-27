import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from vavae.ldm.modules.diffusionmodules.model import Encoder, Decoder
from vavae.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from vavae.ldm.util import instantiate_from_config

from peft import LoraConfig, get_peft_model
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision.models as models
from vavae.ldm.modules.losses.loss import GANLoss

class VAE_Latent_Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        """
        input_dim: Number of VAE latent channels, equal to embed_dim.
        """
        super().__init__()
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),            
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier_head(x)


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_vf=None,
                 reverse_proj=False,
                 proj_fix=False,
                 num_classes=11,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.classifier = VAE_Latent_Classifier(
            input_dim=embed_dim, 
            num_classes=num_classes
        )

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        self.use_vf = use_vf
        self.reverse_proj = reverse_proj
        self.automatic_optimization = False
        self.proj_fix = proj_fix
        self.gan_loss = GANLoss()
       
        self.requires_grad_(False)
        self.loss.requires_grad_(False)
        self.classifier.requires_grad_(False)
        self.gan_loss.requires_grad_(True)
        print("Restoring selective gradients: Unfreezing fusion layers and setting residual init.")
      
        for name, param in self.decoder.named_parameters():
            if 'fusion' in name:
                param.requires_grad = True
                if 'encode_enc_3.conv_out' in name: # need to modify
                    torch.nn.init.zeros_(param)
                else:
                    torch.nn.init.constant_(param, 1e-6)
        # Metrics
        data_range = 2.0 
        self.val_psnr = PeakSignalNoiseRatio(data_range=data_range)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Skipping key {k} from checkpoint (ignored).")
                    del sd[k]
        missing,unexpected=self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print(f"missing keys:{missing}, Unexpected keys:{unexpected}")

    def encode(self, x):
        h, feats = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior,feats
    
    def decode(self, z, feats=None):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, enc_features=feats)
        return dec
    
    def forward(self, input, sample_posterior=True):
        posterior, feats = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, feats=feats)
        return dec, posterior, z

    def get_input(self, batch):
        x = batch["image"]
        label = batch.get("label", None) 
        if len(x.shape) == 3: x = x[..., None]
        return x, label

    def training_step(self, batch, batch_idx):
        inputs, labels = self.get_input(batch)
        ae_opt = self.optimizers()
        
        reconstructions, posterior, z = self(inputs, sample_posterior=True)
        aeloss, log_dict_ae = self.loss(
            inputs, reconstructions, posterior, labels, self.global_step, split="train",
        )
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        ae_opt.step()
        
    

    def validation_step(self, batch, batch_idx):
        inputs, labels = self.get_input(batch)
        reconstructions, posterior, z = self(inputs, sample_posterior=False)
        self.log("val/psnr", self.val_psnr(reconstructions, inputs), prog_bar=True)

    def configure_optimizers(self):
        trainable_params = list(filter(lambda p: p.requires_grad, self.decoder.parameters()))
        print(f"Detected {len(trainable_params)} trainable parameter tensors in Decoder (Fusion Layers).")
        trainable_params += list(filter(lambda p: p.requires_grad, self.gan_loss.parameters()))

        opt_ae = torch.optim.Adam(trainable_params, lr=1e-6, betas=(0.5, 0.9))
        #opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=1e-6, betas=(0.5, 0.9))
        
        return [opt_ae], []

    def get_last_layer(self):
        try:
            return self.decoder.base_model.model.conv_out.weight
        except AttributeError:
            return self.decoder.conv_out.weight
        
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        images, labels = self.get_input(batch)
        images =images.to(self.device)
        
        if not only_inputs:
            rec, post, _= self(images)
            if images.shape[1] > 3:
                images = self.to_rgb(images)
                rec = self.to_rgb(rec)
            log["reconstructions"] = rec
            log["samples"] = self.decode(torch.randn_like(post.sample()))
        
        log["inputs"] = images
        return log
    
    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
    @torch.no_grad()
    def log_images_comparison(self, batch, original_model, split="train"):
        log = dict()
        images, _ = self.get_input(batch)
        images = images.to(self.device)
        rec_fine_tuned, _, _ = self(images)
        
        original_model.to(self.device)
        original_model.eval()
        post_orig, feats_orig = original_model.encode(images)
        rec_original = original_model.decode(post_orig.mode(), feats=feats_orig)

        diff = torch.abs(rec_fine_tuned - rec_original)
        diff_enhanced = torch.clamp(diff * 5.0, 0.0, 1.0) 

        log["inputs"] = images
        log["original_vavae"] = rec_original
        log["fine_tuned_vavae"] = rec_fine_tuned
        log["difference_x5"] = diff_enhanced
        
        return log
    
class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
